"""
DEMO 4 - SERVER-SIDE DEFENCES AND RECOVERY
===========================================
Goal: apply the server-side Robust Aggregation algorithms (FedAvg, Median, Trimmed Mean,
Krum, Norm-Clipping, FLTrust, FoolsGold, FLTrust+Clip, RLR, SA-Trust) and evaluate:

    1. Final-round ASR       - how successful the attack is when training ends
    2. MEAN ASR across rounds - the PRIMARY metric, more informative than final-round ASR
    3. delta-ASR             - how much the defence lowers ASR versus FedAvg
    4. Confusion matrix      - TP/TN/FP/FN, F1, MCC on the clean test set

METRICS DELIBERATELY DROPPED:
  - CA_recovery: with a backdoor, Clean Accuracy barely drops, so the denominator is ~0
    and the recovery ratio oscillates meaninglessly (it once produced -3.7).
  - detection_rate / false_exclusion_rate / update_norm_ratio / trust_gap:
    not reliable enough to report.
  - Why MEAN ASR beats final-round ASR: some algorithms (FLTrust) suppress the backdoor
    for many rounds before finally being breached. Looking only at the last round makes
    every defence appear to "fail equally", hiding the real differences.

Run:
    python scripts/demo4_defenses.py                       # semantic attack (default)
    python scripts/demo4_defenses.py --attack manual
    python scripts/demo4_defenses.py --attack both         # both (takes twice as long)
    python scripts/demo4_defenses.py --defenses fedavg,fltrust
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

OUT_DIR = os.path.join(config.RESULTS_DIR, "demo4")


def run_for_attack(attack: str, defenses: list, mal_ratio: float, rounds,
                   timer=None, force=False):
    asr_col = f"asr_{attack}"
    malicious_ids = EXP.get_malicious_ids(mal_ratio, config.NUM_CLIENTS)

    # Clean reference: provides the CA baseline and the ASR floor
    hist_c, diag_c, meta_c = EXP.run_or_load(
        OUT_DIR, f"{attack}_reference_clean", force=force,
        attack="none", defense="fedavg", malicious_ratio=0.0, num_rounds=rounds)
    if timer:
        timer.add(meta_c)
    clean_last = hist_c.iloc[-1]
    ca_base = float(clean_last["clean_accuracy"])
    asr_floor = float(clean_last.get(asr_col, 0.0))

    rows = []
    for d in defenses:
        name = f"{attack}_{d}"
        history, diagnostics, meta = EXP.run_or_load(
            OUT_DIR, name, force=force, attack=attack, defense=d,
            malicious_ratio=mal_ratio, num_rounds=rounds)
        if timer:
            timer.add(meta)

        last = history.iloc[-1]
        row = {
            "attack": attack,
            "defense": d,
            "clean_accuracy": last["clean_accuracy"],
            "ASR": float(last.get(asr_col, 0.0)),
            "ASR_net": Metrics.net_asr(float(last.get(asr_col, 0.0)), asr_floor),
            "recall": last["recall"], "specificity": last["specificity"],
            "f1": last["f1"], "mcc": last["mcc"],
            "TP": int(last["tp"]), "TN": int(last["tn"]),
            "FP": int(last["fp"]), "FN": int(last["fn"]),
            "duration": meta.get("duration_readable"),
            "seconds_per_round": meta.get("seconds_per_round"),
        }
        # Mean ASR from round 1 to the last round (the PRIMARY metric of Demo 4)
        col = f"asr_{attack}"
        row["ASR_mean"] = (round(float(history[col].iloc[1:].mean()), 4)
                           if col in history else float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows)
    # Compare against the UNDEFENDED baseline (fedavg) within the same attack group
    base = df[df.defense == "fedavg"]
    if len(base):
        asr_nodef = float(base["ASR"].iloc[0])
        df["delta_ASR"] = [Metrics.defense_effectiveness(asr_nodef, a) for a in df["ASR"]]
        mean_nodef = float(base["ASR_mean"].iloc[0])
        df["delta_ASR_mean"] = [round(mean_nodef - a, 4) for a in df["ASR_mean"]]
    df["CA_clean_baseline"] = ca_base
    df["ASR_floor"] = asr_floor
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default="semantic",
                    choices=["manual", "semantic", "both"])
    ap.add_argument("--defenses", default=",".join(config.DEFENSES_FOR_DEMO4))
    ap.add_argument("--malicious-ratio", type=float,
                    default=config.DEFAULT_MALICIOUS_RATIO)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-run scenarios even if a cached result exists")
    ap.add_argument("--no-log", action="store_true",
                    help="do not write a log file")
    ap.add_argument("--demo", action="store_true",
                    help="run at reduced scale, writing to data_demo/ and results_demo/")
    args = ap.parse_args()
    require_demo_data(config)

    session = None if args.no_log else start_logging("demo4")
    print_demo_banner(config)          # printed after the log opens so it gets recorded
    try:
        defenses = [d.strip() for d in args.defenses.split(",") if d.strip()]
        attacks = ["manual", "semantic"] if args.attack == "both" else [args.attack]

        print("#" * 74)
        print("# DEMO 4 - SERVER-SIDE DEFENCES")
        print(f"# Attacks: {attacks} | Defences: {defenses} | "
              f"malicious clients {args.malicious_ratio:.0%}")
        print("#" * 74)

        timer = EXP.RunTimer(total_runs=len(attacks) * (len(defenses) + 1), label="DEMO 4")
        all_df = pd.concat(
            [run_for_attack(a, defenses, args.malicious_ratio, args.rounds, timer,
                            args.force)
             for a in attacks],
            ignore_index=True)

        os.makedirs(OUT_DIR, exist_ok=True)
        safe_to_csv(all_df, os.path.join(OUT_DIR, "demo4_summary.csv"))

        print("\n" + "=" * 74)
        print("DEMO 4 RESULTS - DEFENCE EFFECTIVENESS")
        print("=" * 74)
        cols = ["attack", "defense", "clean_accuracy", "ASR", "ASR_net",
                "ASR_mean", "delta_ASR_mean", "f1", "mcc"]
        cols = [c for c in cols if c in all_df.columns]
        show = all_df.sort_values(["attack", "ASR_mean"])
        print(show[cols].to_string(index=False))

        print("\n" + "=" * 74)
        print("CONFUSION MATRIX ON THE CLEAN TEST SET")
        print("=" * 74)
        print(all_df[["attack", "defense", "TP", "TN", "FP", "FN",
                      "recall", "specificity", "mcc"]].to_string(index=False))

        print("\n--- HOW TO READ THIS ---")
        print("  ASR_mean       : PRIMARY METRIC - mean ASR across the 10 rounds.")
        print("                   Lower is better. The table is sorted ascending.")
        print("  delta_ASR_mean : reduction versus FedAvg. LARGER is better.")
        print("                   A NEGATIVE value means the defence BACKFIRED.")
        print("  ASR (final)    : supplementary; with a cumulative backdoor most defences")
        print("                   approach 1.0, so it cannot separate them.")
        safe_to_csv(timer.report(), os.path.join(OUT_DIR, "demo4_timing.csv"))
        print(f"\n  Saved to: {OUT_DIR}")
        print("  Next: python scripts/05_final_report.py")

    finally:
        stop_logging(session)


if __name__ == "__main__":
    main()
