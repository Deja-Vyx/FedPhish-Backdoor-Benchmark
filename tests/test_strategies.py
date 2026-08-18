"""
test_strategies.py - FLOWER INTEGRATION TESTS (aggregate_fit), NO GPU NEEDED
============================================================================
Builds mock fit results and calls each strategy's aggregate_fit directly.
Run:  python tests/test_strategies.py     (requires flwr to be installed)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from flwr.common import (FitRes, Status, Code, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from src import strategies as S

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    PASS += bool(cond)
    FAIL += (not cond)


class MockProxy:
    def __init__(self, cid):
        self.cid = str(cid)


def _fitres(weights, n, client_id=None):
    metrics = {} if client_id is None else {"client_id": int(client_id)}
    return FitRes(status=Status(Code.OK, ""),
                  parameters=ndarrays_to_parameters([x.astype(np.float32) for x in weights]),
                  num_examples=n, metrics=metrics)


# A REALISTIC simulation:
#   - Honest clients   : all learn the same task -> update = COMMON DIRECTION + Non-IID noise
#                        (similar directions, but not identical)
#   - Malicious clients: all implant one backdoor -> update = A DIFFERENT DIRECTION with
#                        very little noise (nearly identical to each other, and pointing
#                        away from the honest group)
HONEST_DIR = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
BACKDOOR_DIR = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0])


def _to_layers(vec):
    return [vec[:9].reshape(3, 3).astype(np.float32), vec[9:].astype(np.float32)]


def make_results(seed=0, boost=3.0):
    """7 honest clients + 3 malicious clients implanting the same backdoor."""
    rng = np.random.default_rng(seed)
    base = [np.ones((3, 3), dtype=np.float32), np.ones(3, dtype=np.float32)]
    base_vec = np.concatenate([base[0].ravel(), base[1].ravel()])

    # Reproduce recent Flower behaviour EXACTLY: ClientProxy.cid is a RANDOM HASH
    # (node_id), not "0".."9". The real index travels through metrics["client_id"].
    hashes = [2465052526735391746, 6181098985559652035, 10173581685152842980,
              7600428468565347679, 10178486078371505151, 13115487526965809472,
              3512803128606023368, 9010936085184622885, 15498593420498715599,
              16690082444978935122]
    results = []
    for cid in range(7):
        upd = HONEST_DIR * 0.1 + rng.normal(0, 0.05, 12)      # Non-IID noise
        results.append((MockProxy(hashes[cid]),
                        _fitres(_to_layers(base_vec + upd), 100, client_id=cid)))
    for cid in range(7, 10):
        upd = BACKDOOR_DIR * 0.1 * boost + rng.normal(0, 0.002, 12)
        results.append((MockProxy(hashes[cid]),
                        _fitres(_to_layers(base_vec + upd), 100, client_id=cid)))
    rng.shuffle(results)          # the server receives results in ARBITRARY order
    return results, base


def server_update_fn(cur):
    """The server trains on the clean root set -> the honest group's common direction."""
    return _to_layers(HONEST_DIR * 0.1)


def run():
    results, base = make_results()
    init = ndarrays_to_parameters(base)

    base_vec = np.concatenate([base[0].ravel(), base[1].ravel()])
    leaks = {}
    print("\n[1] Basic strategies (measuring how much backdoor leaks through)")
    for name in ["fedavg", "median", "trimmed", "krum", "normclip"]:
        strat = S.make_strategy(name, init, evaluate_fn=None)
        params, _ = strat.aggregate_fit(1, results, [])
        w = parameters_to_ndarrays(params)
        check(f"{name}: returns Parameters with the right shape",
              params is not None and w[0].shape == (3, 3))
        # Measure the residual backdoor by projecting the result onto BACKDOOR_DIR
        vec = np.concatenate([w[0].ravel(), w[1].ravel()]) - base_vec
        leak = float(np.dot(vec, BACKDOOR_DIR) / np.dot(BACKDOOR_DIR, BACKDOOR_DIR))
        leaks[name] = leak
        if name == "fedavg":
            check(f"fedavg: a lot of backdoor leaks through (leak={leak:.3f})", leak > 0.05)
        else:
            check(f"{name}: blocks more backdoor than FedAvg "
                  f"(leak={leak:.3f} < {leaks['fedavg']:.3f})", leak < leaks["fedavg"])

    print("\n[2] FLTrust and the combined variant")
    for name in ["fltrust", "fltrust_clip"]:
        strat = S.make_strategy(name, init, evaluate_fn=None,
                                server_update_fn=server_update_fn)
        params, _ = strat.aggregate_fit(1, results, [])
        w = parameters_to_ndarrays(params)
        check(f"{name}: runs and returns the right shape",
              params is not None and w[0].shape == (3, 3))
        diag = strat.diagnostics[-1]
        ts = diag["trust_scores"]
        check(f"{name}: honest clients score HIGHER than the malicious group",
              np.mean(ts[:7]) > np.mean(ts[7:]))
        check(f"{name}: flags the correct malicious group (7,8,9)",
              {7, 8, 9}.issubset(set(diag["flagged_client_ids"])))

    print("\n[3] FoolsGold (keeps history across rounds)")
    strat = S.make_strategy("foolsgold", init, evaluate_fn=None)
    for r in (1, 2, 3):
        params, _ = strat.aggregate_fit(r, results, [])
    w = parameters_to_ndarrays(params)
    check("foolsgold: returns the right shape", w[0].shape == (3, 3))
    check("foolsgold: accumulates history over 3 rounds", strat.history is not None)
    last = strat.diagnostics[-1]
    flagged = set(last["flagged_client_ids"])
    check(f"foolsgold: flags the malicious group (7,8,9) - actual {sorted(flagged)}",
          {7, 8, 9}.issubset(flagged) or len(flagged & {7, 8, 9}) >= 2)
    check("foolsgold: records a weight for every client",
          len(last.get("foolsgold_weights", [])) == 10)

    print("\n[3b] FULL COVERAGE of every defence in config (guards against omissions)")
    # A bug that was hit once: experiment.py only checked `defense == "fltrust"` and
    # therefore forgot "fltrust_clip" -> a crash after hours of runtime. This test
    # reproduces EXACTLY how experiment.py decides whether a root set is required, for
    # EVERY defence declared in config -> adding a new defence without declaring it fails
    # immediately.
    import config
    check("config.AVAILABLE_DEFENSES matches the registry",
          set(config.AVAILABLE_DEFENSES) == set(S._REGISTRY))
    check("DEFENSES_FOR_DEMO4 is a subset of the registry",
          set(config.DEFENSES_FOR_DEMO4).issubset(set(S._REGISTRY)))

    for name in config.AVAILABLE_DEFENSES:
        needs_root = S.requires_server_update(name)
        fn = server_update_fn if needs_root else None      # same logic as experiment.py
        try:
            strat = S.make_strategy(name, init, evaluate_fn=None, server_update_fn=fn)
            params, _ = strat.aggregate_fit(1, results, [])
            ok = params is not None
        except Exception as e:
            ok = False
            print(f"       -> {type(e).__name__}: {e}")
        check(f"{name}: constructs and runs aggregate_fit "
              f"(needs root={needs_root})", ok)

    check("Exactly 2 defences need a root set (fltrust, fltrust_clip)",
          S.NEEDS_SERVER_UPDATE == {"fltrust", "fltrust_clip"})

    print("\n[3c] CLIENT INDEX MAPPING (guards against the cid-is-a-hash bug)")
    # A bug that was hit once: using ClientProxy.cid as the client index. In recent Flower
    # versions cid is a random node_id (e.g. 2465052526735391746), so flagged_client_ids
    # became garbage, detection_rate was always 0 and trust_gap referenced the wrong
    # clients. This test catches that.
    _, _, cids = S._extract(results)
    check(f"Recovers the correct client indices 0..9 (actual {cids})",
          cids == list(range(10)))

    strat = S.make_strategy("krum", init, evaluate_fn=None)
    strat.aggregate_fit(1, results, [])
    flagged = strat.diagnostics[-1]["flagged_client_ids"]
    check(f"Krum reports IDs in the valid range 0..9 (actual {flagged})",
          all(0 <= c < 10 for c in flagged))
    check("Krum excludes exactly the malicious group 7,8,9", set(flagged) == {7, 8, 9})

    strat = S.make_strategy("fltrust", init, evaluate_fn=None,
                            server_update_fn=server_update_fn)
    strat.aggregate_fit(1, results, [])
    d = strat.diagnostics[-1]
    ts = d["trust_scores"]
    check("FLTrust: trust scores are ordered BY CLIENT INDEX "
          f"(honest {np.mean(ts[:7]):.2f} > malicious {np.mean(ts[7:]):.2f})",
          np.mean(ts[:7]) > np.mean(ts[7:]))
    check(f"FLTrust reports valid IDs (actual {d['flagged_client_ids']})",
          all(0 <= c < 10 for c in d["flagged_client_ids"]))

    # Without metrics -> must fall back safely instead of emitting garbage numbers
    no_meta = [(MockProxy(999999999999), _fitres(_to_layers(base_vec), 100))
               for _ in range(3)]
    _, _, cids2 = S._extract(no_meta)
    check(f"Missing client_id -> the fallback ID is still valid (actual {cids2})",
          all(0 <= c < 10 for c in cids2))

    print("\n[4] Global-weight tracking & error handling")
    strat = S.make_strategy("median", init, evaluate_fn=None)
    before = [x.copy() for x in strat.current_global]
    strat.aggregate_fit(1, results, [])
    check("The global weights are updated after aggregate_fit",
          not np.allclose(before[0], strat.current_global[0]))
    check("No results -> returns (None, {})",
          strat.aggregate_fit(2, [], [])[0] is None)
    try:
        S.make_strategy("fltrust", init, evaluate_fn=None)   # missing server_update_fn
        check("FLTrust without a root set -> must raise", False)
    except ValueError:
        check("FLTrust without a root set -> raises a clear error", True)
    try:
        S.make_strategy("does_not_exist", init, evaluate_fn=None)
        check("Unknown strategy name -> must raise", False)
    except ValueError:
        check("Unknown strategy name -> raises a clear error", True)


if __name__ == "__main__":
    print("=" * 66)
    print("FLOWER STRATEGY INTEGRATION TESTS")
    print("=" * 66)
    np.random.seed(0)
    run()
    print("\n" + "=" * 66)
    print(f"RESULT: {PASS} PASS / {FAIL} FAIL")
    print("=" * 66)
    sys.exit(1 if FAIL else 0)
