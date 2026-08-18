"""
02_make_poison.py - STEP 2: build the poisoned data for the malicious clients
=============================================================================
Generates the semantic trigger sentences with an LLM (gpt-oss via Ollama), then produces
two poisoned variants for every client that could play the attacker role:
      client_{id}_poisoned_manual.csv     (manual trigger)
      client_{id}_poisoned_semantic.csv   (LLM-generated semantic trigger)

Files are pre-built for several clients because DEMO 2 sweeps multiple malicious ratios.

Run:
    python scripts/02_make_poison.py
    python scripts/02_make_poison.py --no-llm     # skip Ollama, use the fallback pool
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Demo mode must be enabled BEFORE config is imported (config reads the environment
#     variable at import time) ---
from src.run_mode import enable_demo_mode, print_demo_banner
enable_demo_mode()

import pandas as pd
import config
from src.io_utils import safe_to_csv
from src import poisoning as P
from src import stealth as ST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=config.POISON_RATIO)
    ap.add_argument("--no-llm", action="store_true", help="do not call Ollama")
    ap.add_argument("--demo", action="store_true",
                    help="run at reduced scale, writing to data_demo/ and results_demo/")
    args = ap.parse_args()
    print_demo_banner(config)

    use_ollama = config.USE_OLLAMA_FOR_TRIGGERS and not args.no_llm

    # --- 1. Generate the semantic triggers ---
    print(f"[1/3] Generating {config.NUM_LLM_TRIGGERS} semantic trigger sentences "
          f"with {config.OLLAMA_MODEL_NAME}")
    print(f"      Theme: \"{config.SEMANTIC_TRIGGER_THEME}\"")
    semantic = P.generate_semantic_triggers(
        config.NUM_LLM_TRIGGERS, config.LLM_TRIGGER_PROMPT,
        config.SEMANTIC_TRIGGER_THEME, config.OLLAMA_BASE_URL,
        config.OLLAMA_MODEL_NAME, use_ollama=use_ollama,
        timeout=config.OLLAMA_TIMEOUT, seed=config.RANDOM_SEED)
    P.save_triggers(config.TRIGGERS_PATH, config.MANUAL_TRIGGER_PHRASE,
                    semantic, config.SEMANTIC_TRIGGER_THEME)
    print(f"      Saved {len(semantic)} sentences -> "
          f"{os.path.basename(config.TRIGGERS_PATH)}")

    # --- 2. Compare the evasion of the two trigger types (numbers used by DEMO 3) ---
    print("\n[2/3] Measuring trigger evasion (content filter & text statistics)")
    clean_texts = pd.read_csv(config.TRAIN_POOL_PATH)["text"].astype(str).tolist()[:3000]
    rows = [
        ST.evaluate_trigger_stealth([config.MANUAL_TRIGGER_PHRASE], clean_texts, "manual"),
        ST.evaluate_trigger_stealth(semantic, clean_texts, "semantic_llm"),
    ]
    stealth_df = pd.DataFrame(rows)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    safe_to_csv(stealth_df, os.path.join(config.RESULTS_DIR, "trigger_stealth.csv"))
    print(stealth_df.to_string(index=False))

    # --- 3. Produce the poisoned data files ---
    max_mal = max(int(round(r * config.NUM_CLIENTS)) for r in config.MALICIOUS_RATIO_SWEEP)
    max_mal = max(max_mal, int(round(config.DEFAULT_MALICIOUS_RATIO * config.NUM_CLIENTS)))
    candidate_ids = list(range(config.NUM_CLIENTS - max_mal, config.NUM_CLIENTS))
    print(f"\n[3/3] Building poisoned data for clients {candidate_ids} "
          f"(poison {args.ratio:.0%})")

    for cid in candidate_ids:
        clean_path = os.path.join(config.CLIENT_DATA_DIR, f"client_{cid}.csv")
        if not os.path.exists(clean_path):
            raise FileNotFoundError(
                f"Missing {clean_path}. Run 01_prepare_data.py first.")
        df = pd.read_csv(clean_path)
        n_phish = int((df.label == 1).sum())

        man = P.apply_manual_poison(df, args.ratio, config.MANUAL_TRIGGER_PHRASE,
                                    config.TARGET_LABEL_AFTER_ATTACK, config.RANDOM_SEED)
        safe_to_csv(man, os.path.join(config.CLIENT_DATA_DIR,
                                      f"client_{cid}_poisoned_manual.csv"))

        sem = P.apply_semantic_poison(df, args.ratio, semantic,
                                      config.TARGET_LABEL_AFTER_ATTACK, config.RANDOM_SEED)
        safe_to_csv(sem, os.path.join(config.CLIENT_DATA_DIR,
                                      f"client_{cid}_poisoned_semantic.csv"))

        print(f"      client_{cid}: {n_phish} phishing -> "
              f"{int(man.is_poisoned.sum())} poisoned samples (both variants)")

    print("\nDone. Next: python scripts/demo1_baseline.py"
          + (" --demo" if config.DEMO_MODE else ""))


if __name__ == "__main__":
    main()
