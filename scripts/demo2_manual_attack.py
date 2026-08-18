"""
DEMO 2 - MANUAL-TRIGGER ATTACK, SWEEPING THE MALICIOUS-CLIENT RATIO
====================================================================
Goal: measure Clean Accuracy (CA) and Attack Success Rate (ASR) at EACH malicious-client
ratio (0%, 10%, 20%, 30%, 40%) to identify the THRESHOLD at which the backdoor succeeds.

No defence is used (plain FedAvg) so the attack's raw strength is visible.

How to read the results: a "good" backdoor keeps CA essentially unchanged (it stays
hidden) while ASR climbs sharply. A large CA drop means the attack has exposed itself.

Outputs:
    results/demo2/*_history.csv
    results/demo2/demo2_summary.csv     - CA & ASR per ratio

Run:
    python scripts/demo2_manual_attack.py
    python scripts/demo2_manual_attack.py --ratios 0,0.2,0.4
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Demo mode must be enabled BEFORE config is imported (config reads the environment
#     variable at import time) ---
from src.run_mode import enable_demo_mode, print_demo_banner, require_demo_data
enable_demo_mode()

import pandas as pd
import config
from src.io_utils import safe_to_csv
from src import experiment as EXP
from src import metrics as Metrics
from src.run_logger import start_logging, stop_logging

OUT_DIR = os.path.join(config.RESULTS_DIR, "demo2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", default=None,
                    help="comma-separated list of ratios, e.g. 0,0.1,0.2,0.3,0.4")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-run scenarios even if a cached result exists")
    ap.add_argument("--no-log", action="store_true",
                    help="do not write a log file")
    ap.add_argument("--demo", action="store_true",
                    help="run at reduced scale, writing to data_demo/ and results_demo/")
    args = ap.parse_args()
    require_demo_data(config)

    session = None if args.no_log else start_logging("demo2")
    print_demo_banner(config)          # printed after the log opens so it gets recorded
    try:
        ratios = ([float(x) for x in args.ratios.split(",")] if args.ratios
                  else config.MALICIOUS_RATIO_SWEEP)

        print("#" * 74)
        print("# DEMO 2 - MANUAL-TRIGGER ATTACK BY MALICIOUS-CLIENT RATIO")
        print(f"# Sweep: {[f'{r:.0%}' for r in ratios]} | no defence (FedAvg)")
        print("#" * 74)

        timer = EXP.RunTimer(total_runs=len(ratios), label="DEMO 2")
        rows, asr_floor = [], None
        for r in ratios:
            n_mal = int(round(r * config.NUM_CLIENTS))
            name = f"manual_mal{int(r * 100):02d}"
            attack = "none" if n_mal == 0 else "manual"

            history, diagnostics, meta = EXP.run_or_load(
                OUT_DIR, name, force=args.force, attack=attack, defense="fedavg",
                malicious_ratio=r, num_rounds=args.rounds)
            timer.add(meta)

            last = history.iloc[-1]
            asr = float(last.get("asr_manual", 0.0))
            if n_mal == 0:
                asr_floor = asr        # the ASR floor from this very clean run

            # Count how many samples are ACTUALLY poisoned - important when interpreting
            # the ASR curve, because clients hold different amounts of phishing email, so
            # adding one malicious client does not double the poisoned volume.
            n_poisoned = 0
            for cid in EXP.get_malicious_ids(r, config.NUM_CLIENTS):
                p = os.path.join(config.CLIENT_DATA_DIR,
                                 f"client_{cid}_poisoned_manual.csv")
                if os.path.exists(p):
                    n_poisoned += int(pd.read_csv(p)["is_poisoned"].sum())

            rows.append({
                "malicious_ratio": r,
                "n_malicious": n_mal,
                "n_poisoned_samples": n_poisoned,
                "clean_accuracy": last["clean_accuracy"],
                "precision": last["precision"],
                "recall": last["recall"],
                "specificity": last["specificity"],
                "f1": last["f1"],
                "mcc": last["mcc"],
                "TP": int(last["tp"]), "TN": int(last["tn"]),
                "FP": int(last["fp"]), "FN": int(last["fn"]),
                "ASR": asr,
                "duration": meta.get("duration_readable"),
                "seconds_per_round": meta.get("seconds_per_round"),
            })

        df = pd.DataFrame(rows)
        # ASR_net = ASR - floor; CA_drop = CA loss versus the 0%-malicious scenario
        ca_base = float(df.loc[df.malicious_ratio == 0, "clean_accuracy"].iloc[0]) \
            if (df.malicious_ratio == 0).any() else float(df["clean_accuracy"].max())
        df["ASR_net"] = [Metrics.net_asr(a, asr_floor or 0.0) for a in df["ASR"]]
        df["CA_drop"] = [Metrics.ca_drop(ca_base, c) for c in df["clean_accuracy"]]

        os.makedirs(OUT_DIR, exist_ok=True)
        safe_to_csv(df, os.path.join(OUT_DIR, "demo2_summary.csv"))

        print("\n" + "=" * 74)
        print("DEMO 2 RESULTS - CA & ASR BY MALICIOUS-CLIENT RATIO")
        print("=" * 74)
        cols = ["malicious_ratio", "n_malicious", "n_poisoned_samples",
                "clean_accuracy", "CA_drop", "ASR", "ASR_net", "recall", "f1",
                "TP", "FP", "TN", "FN"]
        print(df[cols].to_string(index=False))
        print(f"\n  ASR floor (clean model): {asr_floor}")
        print("  ASR_net is the number that reflects the backdoor's TRUE strength "
              "(floor subtracted).")
        safe_to_csv(timer.report(), os.path.join(OUT_DIR, "demo2_timing.csv"))
        print(f"\n  Saved to: {OUT_DIR}")
        print("  Next: python scripts/demo3_semantic_attack.py")

    finally:
        stop_logging(session)


if __name__ == "__main__":
    main()
