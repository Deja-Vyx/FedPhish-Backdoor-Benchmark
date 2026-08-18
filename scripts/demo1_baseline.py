"""
DEMO 1 - THE FL SYSTEM OPERATING NORMALLY
==========================================
Goal: show that a Federated Learning system with ALL CLIENTS HONEST converges and reaches
a standard level of accuracy. This is the BASELINE every later scenario is compared to.

This scenario also produces a methodologically mandatory number: the "ASR FLOOR" - the
rate at which even a COMPLETELY CLEAN model misclassifies an email once a trigger sentence
is inserted. Later demos subtract this floor (ASR_net); without it, a distribution-shift
effect would be mistaken for "backdoor success".

Outputs:
    results/demo1/baseline_clean_history.csv   - per-round metrics
    results/demo1/demo1_summary.csv            - summary table + confusion matrix

Run:  python scripts/demo1_baseline.py
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
from src.run_logger import start_logging, stop_logging

OUT_DIR = os.path.join(config.RESULTS_DIR, "demo1")


def main():
    print("#" * 74)
    print("# DEMO 1 - NORMAL FL OPERATION, ALL CLIENTS HONEST")
    print("#" * 74)

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-run even if a cached result exists")
    ap.add_argument("--no-log", action="store_true",
                    help="do not write a log file")
    ap.add_argument("--demo", action="store_true",
                    help="run at reduced scale, writing to data_demo/ and results_demo/")
    args = ap.parse_args()
    require_demo_data(config)

    session = None if args.no_log else start_logging("demo1")
    print_demo_banner(config)          # printed after the log opens so it gets recorded
    try:
        timer = EXP.RunTimer(total_runs=1, label="DEMO 1")
        history, diagnostics, meta = EXP.run_or_load(
            OUT_DIR, "baseline_clean", force=args.force,
            attack="none", defense="fedavg", malicious_ratio=0.0)
        timer.add(meta)

        last = history.iloc[-1]
        summary = pd.DataFrame([{
            "scenario": "Clean FL (10/10 honest clients)",
            "num_rounds": int(last["round"]),
            "clean_accuracy": last["clean_accuracy"],
            "precision": last["precision"],
            "recall": last["recall"],
            "specificity": last["specificity"],
            "f1": last["f1"],
            "mcc": last["mcc"],
            "balanced_accuracy": last["balanced_accuracy"],
            "TP": int(last["tp"]), "TN": int(last["tn"]),
            "FP": int(last["fp"]), "FN": int(last["fn"]),
            "ASR_floor_manual": last.get("asr_manual"),
            "ASR_floor_semantic": last.get("asr_semantic"),
            "duration": meta.get("duration_readable"),
            "seconds_per_round": meta.get("seconds_per_round"),
        }])
        os.makedirs(OUT_DIR, exist_ok=True)
        safe_to_csv(summary, os.path.join(OUT_DIR, "demo1_summary.csv"))

        print("\n" + "=" * 74)
        print("DEMO 1 RESULTS")
        print("=" * 74)
        print(f"  Clean Accuracy (CA)  : {last['clean_accuracy']:.4f}")
        print(f"  Precision / Recall   : {last['precision']:.4f} / {last['recall']:.4f}")
        print(f"  Specificity (TNR)    : {last['specificity']:.4f}")
        print(f"  F1 / MCC             : {last['f1']:.4f} / {last['mcc']:.4f}")
        print(f"  Confusion matrix     : TP={int(last['tp'])} TN={int(last['tn'])} "
              f"FP={int(last['fp'])} FN={int(last['fn'])}")
        print(f"  Missed phishing (FNR): {last['fnr']:.4f}")
        print("\n  --- ASR FLOOR on the CLEAN model (essential for the later demos) ---")
        print(f"  ASR floor (manual)   : {last.get('asr_manual')}")
        print(f"  ASR floor (semantic) : {last.get('asr_semantic')}")
        print("  => In demos 2/3/4 the real backdoor ASR = measured ASR - this floor "
              "(ASR_net).")
        safe_to_csv(timer.report(), os.path.join(OUT_DIR, "demo1_timing.csv"))
        print(f"\n  Saved to: {OUT_DIR}")
        print("  Next: python scripts/demo2_manual_attack.py")

    finally:
        stop_logging(session)


if __name__ == "__main__":
    main()
