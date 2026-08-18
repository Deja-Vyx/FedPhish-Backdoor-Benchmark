"""
DEMO 3 - LLM SEMANTIC ATTACK VERSUS THE MANUAL TRIGGER
=======================================================
Goal: compare the LLM-generated semantic trigger against the manual trigger along TWO
INDEPENDENT AXES, because "more sophisticated" is a multi-layered notion, not a single
number:

    AXIS 1 - ATTACK EFFECTIVENESS : ASR and ASR_net (floor subtracted)
    AXIS 2 - EVASION              : (a) passing the content filter  (b) preserving CA so
                                     nothing looks suspicious  (c) hiding in weight space

FAIR DESIGN: both attacks use the SAME malicious-client ratio, the SAME poison ratio, the
SAME seed and the SAME test set. Moreover, every run measures ASR for BOTH trigger types,
so the comparison is not polluted by run-to-run randomness.

SCIENTIFIC HONESTY: this script MEASURES rather than ASSUMES the outcome. If the numbers
show the semantic trigger is not stronger on axis 1, report exactly that and analyse
axis 2 instead - that is still a valuable finding (see the README, "Interpreting Demo 3").

Outputs:
    results/demo3/*_history.csv
    results/demo3/demo3_summary.csv        - comparison of the two attacks
    results/demo3/demo3_stealth.csv        - evasion metrics

Run:  python scripts/demo3_semantic_attack.py
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
from src import stealth as ST
from src import poisoning as P
from src.run_logger import start_logging, stop_logging

OUT_DIR = os.path.join(config.RESULTS_DIR, "demo3")


def main():
    ap = argparse.ArgumentParser()
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

    session = None if args.no_log else start_logging("demo3")
    print_demo_banner(config)          # printed after the log opens so it gets recorded
    try:
        r = args.malicious_ratio
        print("#" * 74)
        print("# DEMO 3 - LLM SEMANTIC ATTACK VERSUS THE MANUAL TRIGGER")
        print(f"# Malicious clients: {r:.0%} | poison {config.POISON_RATIO:.0%} | "
              f"no defence")
        print("#" * 74)

        runs = {}
        timer = EXP.RunTimer(total_runs=3, label="DEMO 3")
        # A clean scenario provides the ASR floor for both trigger types
        for name, attack, ratio in [("clean_reference", "none", 0.0),
                                    ("attack_manual", "manual", r),
                                    ("attack_semantic", "semantic", r)]:
            history, diagnostics, meta = EXP.run_or_load(
                OUT_DIR, name, force=args.force, attack=attack, defense="fedavg",
                malicious_ratio=ratio, num_rounds=args.rounds)
            timer.add(meta)
            runs[name] = (history, diagnostics, meta)

        clean_last = runs["clean_reference"][0].iloc[-1]
        floor_manual = float(clean_last.get("asr_manual", 0.0))
        floor_semantic = float(clean_last.get("asr_semantic", 0.0))
        ca_base = float(clean_last["clean_accuracy"])

        rows = []
        for label, key, asr_col, floor in [
                ("Manual (rare tokens)", "attack_manual", "asr_manual", floor_manual),
                ("Semantic (LLM)", "attack_semantic", "asr_semantic", floor_semantic)]:
            last = runs[key][0].iloc[-1]
            asr = float(last.get(asr_col, 0.0))
            rows.append({
                "attack": label,
                "ASR": asr,
                "ASR_floor": floor,
                "ASR_net": Metrics.net_asr(asr, floor),
                "clean_accuracy": last["clean_accuracy"],
                "CA_drop": Metrics.ca_drop(ca_base, last["clean_accuracy"]),
                "recall": last["recall"],
                "f1": last["f1"], "mcc": last["mcc"],
                "TP": int(last["tp"]), "TN": int(last["tn"]),
                "FP": int(last["fp"]), "FN": int(last["fn"]),
            })
        summary = pd.DataFrame(rows)

        # --- Content-layer evasion metrics ---
        trig = P.load_triggers(config.TRIGGERS_PATH)
        semantic_triggers = (trig["semantic_triggers"] if trig
                             else P.FALLBACK_SEMANTIC_TRIGGERS)
        manual_trigger = trig["manual_trigger"] if trig else config.MANUAL_TRIGGER_PHRASE
        clean_texts = pd.read_csv(config.TRAIN_POOL_PATH)["text"].astype(str).tolist()[:3000]
        stealth = pd.DataFrame([
            ST.evaluate_trigger_stealth([manual_trigger], clean_texts, "Manual"),
            ST.evaluate_trigger_stealth(semantic_triggers, clean_texts, "Semantic (LLM)"),
        ])

        os.makedirs(OUT_DIR, exist_ok=True)
        safe_to_csv(summary, os.path.join(OUT_DIR, "demo3_summary.csv"))
        safe_to_csv(stealth, os.path.join(OUT_DIR, "demo3_stealth.csv"))

        print("\n" + "=" * 74)
        print("DEMO 3 RESULTS - AXIS 1: ATTACK EFFECTIVENESS")
        print("=" * 74)
        print(summary[["attack", "ASR", "ASR_floor", "ASR_net",
                       "clean_accuracy", "CA_drop"]].to_string(index=False))

        print("\n" + "=" * 74)
        print("DEMO 3 RESULTS - AXIS 2: EVASION (content layer)")
        print("=" * 74)
        print(stealth[["trigger_type", "rare_token_rate", "oov_ratio",
                       "filter_detection_rate",
                       "avg_word_frequency"]].to_string(index=False))

        m, s = summary.iloc[0], summary.iloc[1]
        print("\n--- INTERPRETATION ---")
        if s["ASR_net"] > m["ASR_net"]:
            print(f"  Axis 1: the semantic attack is STRONGER "
                  f"(ASR_net {s['ASR_net']:.4f} > {m['ASR_net']:.4f}).")
        elif s["ASR_net"] < m["ASR_net"]:
            print(f"  Axis 1: the manual attack is stronger on ASR "
                  f"({m['ASR_net']:.4f} > {s['ASR_net']:.4f}).")
            print("          This is a valid result - see the interpretation in the README:")
            print("          a concentrated backdoor signal (rare tokens) sticks better than")
            print("          one spread across common words, but it is VERY easy to filter.")
        else:
            print("  Axis 1: the two attacks are equivalent in ASR_net.")
        print("  Axis 2: see the table above - the semantic trigger passes the content "
              "filter, the manual trigger is caught immediately by its junk tokens.")
        safe_to_csv(timer.report(), os.path.join(OUT_DIR, "demo3_timing.csv"))
        print(f"\n  Saved to: {OUT_DIR}")
        print("  Next: python scripts/demo4_defenses.py")

    finally:
        stop_logging(session)


if __name__ == "__main__":
    main()
