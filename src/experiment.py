"""
experiment.py - THE ENGINE THAT RUNS ONE FL EXPERIMENT
=======================================================
Shared by all four demos so that every scenario runs under IDENTICAL CONDITIONS
(same seed, same initialisation, same test set) - a prerequisite for a fair comparison.

Each run returns:
    - history: per-round metrics DataFrame (CA, P, R, F1, MCC, TP/TN/FP/FN, ASR, ...)
    - diagnostics: per-round defence log (excluded clients, update norms, trust scores)
    - meta: the run's configuration (for reproducibility)
"""

import os
import json
import random
import time
import datetime
import numpy as np
import pandas as pd

import config
from src.io_utils import safe_to_csv, safe_write_json
from src import model as M
from src import poisoning as P
from src import strategies as S
from src import server_eval as SE
from src import metrics as Metrics
from src import aggregation as A
from src.timing import fmt_duration, RunTimer  # noqa: F401 (re-export)


def set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_malicious_ids(malicious_ratio: float, num_clients: int):
    """Malicious clients are always taken from the END of the list so runs stay identical."""
    n_mal = int(round(malicious_ratio * num_clients))
    return list(range(num_clients - n_mal, num_clients)) if n_mal > 0 else []


def build_data_map(attack: str, malicious_ids: list):
    """attack: 'none' | 'manual' | 'semantic'. Decides which file each client reads."""
    m = {}
    for cid in range(config.NUM_CLIENTS):
        if attack != "none" and cid in malicious_ids:
            path = os.path.join(config.CLIENT_DATA_DIR, f"client_{cid}_poisoned_{attack}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {path}. Run: python scripts/02_make_poison.py")
            m[cid] = path
        else:
            m[cid] = os.path.join(config.CLIENT_DATA_DIR, f"client_{cid}.csv")
    return m


def load_trigger_fns():
    """Build BOTH trigger functions so every scenario can measure both ASR variants."""
    data = P.load_triggers(config.TRIGGERS_PATH)
    semantic = data["semantic_triggers"] if data else P.FALLBACK_SEMANTIC_TRIGGERS
    manual = data["manual_trigger"] if data else config.MANUAL_TRIGGER_PHRASE
    return {
        "manual": P.make_manual_trigger_fn(manual),
        "semantic": P.make_semantic_trigger_fn(semantic, seed=config.RANDOM_SEED),
    }


def run_experiment(run_name: str,
                   attack: str = "none",
                   defense: str = "fedavg",
                   malicious_ratio: float = 0.0,
                   num_rounds: int = None,
                   seed: int = None,
                   verbose: bool = True):
    """
    Run one complete FL scenario.
      attack  : 'none' | 'manual' | 'semantic'
      defense : 'fedavg' | 'median' | 'trimmed' | 'krum' | 'normclip' | 'fltrust' | ...
    """
    import flwr as fl
    from flwr.common import ndarrays_to_parameters
    from flwr.server import ServerConfig
    from src.fl_client import make_client_fn

    seed = seed if seed is not None else config.RANDOM_SEED
    num_rounds = num_rounds or config.NUM_ROUNDS
    malicious_ids = get_malicious_ids(malicious_ratio, config.NUM_CLIENTS)

    if verbose:
        print("\n" + "=" * 74)
        print(f">>> {run_name}")
        print(f"    attack={attack} | defense={defense} | "
              f"malicious clients={malicious_ids or 'none'} | rounds={num_rounds}")
        print("=" * 74)

    set_seed(seed)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    root_df = pd.read_csv(config.ROOT_DATA_PATH)

    init_params = ndarrays_to_parameters(M.get_weights(M.build_model()))
    log_records = []
    evaluate_fn = SE.make_centralized_evaluate_fn(test_df, load_trigger_fns(), log_records)
    # Which defences need the root set is declared by strategies.py (NEEDS_SERVER_UPDATE),
    # so adding a new root-based defence works automatically without editing this file.
    server_update_fn = (SE.make_server_update_fn(root_df)
                        if S.requires_server_update(defense) else None)
    strategy = S.make_strategy(defense, init_params, evaluate_fn, server_update_fn)

    client_fn = make_client_fn(build_data_map(attack, malicious_ids))
    t_start = time.perf_counter()
    started_at = datetime.datetime.now()
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.NUM_CLIENTS,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=config.CLIENT_RESOURCES,
        ray_init_args=config.RAY_INIT_ARGS,
    )
    duration = time.perf_counter() - t_start

    history = pd.DataFrame(log_records)
    history.insert(0, "run", run_name)

    if verbose:
        print(f"    [TIME] {fmt_duration(duration)} "
              f"({duration / max(num_rounds, 1):.1f}s/round)")

    meta = {
        "run": run_name, "attack": attack, "defense": defense,
        "malicious_ratio": malicious_ratio, "malicious_ids": malicious_ids,
        "num_clients": config.NUM_CLIENTS, "num_rounds": num_rounds,
        "poison_ratio": config.POISON_RATIO, "seed": seed,
        "dataset": config.DATASET_NAME, "model": config.CLASSIFIER_MODEL_NAME,
        "partition": config.PARTITION_MODE, "dirichlet_alpha": config.DIRICHLET_ALPHA,
        "learning_rate": config.LEARNING_RATE,
        # --- Timing (used to estimate computational cost in a report) ---
        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(duration, 2),
        "duration_readable": fmt_duration(duration),
        "seconds_per_round": round(duration / max(num_rounds, 1), 2),
        "device": str(M.get_device()),
        "concurrent_clients": config.CONCURRENT_CLIENTS,
        # Version stamp: used to detect stale results when resuming (see load_run)
        "schema_version": config.RESULTS_SCHEMA_VERSION,
        "agg_dtype": str(np.dtype(A.DTYPE)),
    }

    # Malicious-client detection rate (only meaningful for defences that exclude or
    # down-weight clients)
    diagnostics = getattr(strategy, "diagnostics", [])
    if malicious_ids and diagnostics:
        flagged_all = []
        for rec in diagnostics:
            flagged_all.extend(rec.get("flagged_client_ids", []))
        # A client counts as "stably detected" only if it was flagged in >= 50% of rounds
        from collections import Counter
        counts = Counter(flagged_all)
        stable = [cid for cid, c in counts.items() if c >= max(1, len(diagnostics) // 2)]
        meta.update(Metrics.detection_metrics(stable, malicious_ids,
                                              list(range(config.NUM_CLIENTS))))
    return history, diagnostics, meta


def load_run(out_dir: str, run_name: str):
    """
    Load a PREVIOUSLY EXECUTED scenario from disk. Returns (history, diagnostics, meta)
    or None if it does not exist. Used to RESUME after an interruption.
    """
    h_path = os.path.join(out_dir, f"{run_name}_history.csv")
    m_path = os.path.join(out_dir, f"{run_name}_meta.json")
    if not (os.path.exists(h_path) and os.path.exists(m_path)):
        return None
    try:
        history = pd.read_csv(h_path)
        with open(m_path, encoding="utf-8") as f:
            meta = json.load(f)
        if history.empty:
            return None

        # --- Block MIXING old and new numbers ---
        # If a result was produced by an older code version it is NOT comparable with new
        # results (see config.RESULTS_SCHEMA_VERSION). Treat it as missing so it gets
        # re-run, instead of silently feeding stale numbers into the summary table.
        ver = meta.get("schema_version", 1)
        if ver != config.RESULTS_SCHEMA_VERSION:
            print(f"    [STALE] '{run_name}' was produced by version v{ver}, "
                  f"current code is v{config.RESULTS_SCHEMA_VERSION} -> RE-RUNNING.")
            return None
        diagnostics = []
        d_path = os.path.join(out_dir, f"{run_name}_diagnostics.json")
        if os.path.exists(d_path):
            with open(d_path, encoding="utf-8") as f:
                diagnostics = json.load(f)
        return history, diagnostics, meta
    except Exception as e:
        print(f"    [Could not reload '{run_name}': {e} -> will re-run]")
        return None


def run_or_load(out_dir: str, run_name: str, force: bool = False, **kwargs):
    """
    Run a scenario, OR reuse the cached result if it already exists on disk.

    This matters a lot for long experiment batches interrupted by an OOM, a reboot, ...:
    only the missing part is executed instead of repeating hours of work.
    Pass force=True (or the --force flag) to force a re-run.
    """
    if not force:
        cached = load_run(out_dir, run_name)
        if cached is not None:
            last = cached[0].iloc[-1]
            print(f"\n>>> {run_name}: CACHED RESULT FOUND -> skipping "
                  f"(CA={last.get('clean_accuracy')}, "
                  f"previous runtime {cached[2].get('duration_readable', '?')})")
            print("    Use --force to re-run this scenario.")
            return cached
    history, diagnostics, meta = run_experiment(run_name=run_name, **kwargs)
    save_run(out_dir, run_name, history, diagnostics, meta)
    return history, diagnostics, meta


def save_run(out_dir: str, run_name: str, history: pd.DataFrame,
             diagnostics: list, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    safe_to_csv(history, os.path.join(out_dir, f"{run_name}_history.csv"))
    safe_write_json(meta, os.path.join(out_dir, f"{run_name}_meta.json"))
    if diagnostics:
        safe_write_json(diagnostics,
                        os.path.join(out_dir, f"{run_name}_diagnostics.json"))
    last = history.iloc[-1].to_dict() if len(history) else {}
    print(f"  [SAVED] {run_name} | CA={last.get('clean_accuracy')} "
          f"ASR_manual={last.get('asr_manual')} ASR_semantic={last.get('asr_semantic')}")
