"""
01_prepare_data.py - STEP 1: clean and partition the data
==========================================================
Reads data/phishing_email.csv (zefang-liu/phishing-email-dataset), cleans and balances it,
then splits it into: test_set (server), root_set (FLTrust) and 10 Non-IID client shards.

Run:
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --alpha 0.5      # stronger Non-IID
    python scripts/01_prepare_data.py --max-samples 4000
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
from src import data_loader as DL
from src.io_utils import safe_to_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=config.SOURCE_DATASET_PATH)
    ap.add_argument("--partition", default=config.PARTITION_MODE,
                    choices=["label_skew", "dirichlet", "iid"])
    ap.add_argument("--alpha", type=float, default=config.DIRICHLET_ALPHA)
    ap.add_argument("--max-samples", type=int, default=config.DATASET_MAX_SAMPLES)
    ap.add_argument("--demo", action="store_true",
                    help="run at reduced scale, writing to data_demo/ and results_demo/")
    args = ap.parse_args()
    print_demo_banner(config)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.CLIENT_DATA_DIR, exist_ok=True)

    print(f"[1/4] Read & clean: {os.path.basename(args.source)}")
    raw = DL.load_and_clean(args.source, config)
    raw = DL.balance_and_sample(raw, args.max_samples, config.RANDOM_SEED)
    safe_to_csv(raw, config.RAW_DATA_PATH)

    print("[2/4] Split into train / test(server) / root(FLTrust)")
    train_pool, test_df, root_df = DL.split_train_test_root(
        raw, config.TEST_SET_RATIO, config.ROOT_SET_SIZE, config.RANDOM_SEED)
    safe_to_csv(train_pool, config.TRAIN_POOL_PATH)
    safe_to_csv(test_df, config.TEST_DATA_PATH)
    safe_to_csv(root_df, config.ROOT_DATA_PATH)
    print(f"      train_pool={len(train_pool)} | test={len(test_df)} "
          f"(phishing={int((test_df.label == 1).sum())}) | root={len(root_df)}")

    print(f"[3/4] Partition across {config.NUM_CLIENTS} clients "
          f"(mode={args.partition}, alpha={args.alpha})")
    clients = DL.partition_clients(
        train_pool, config.NUM_CLIENTS, args.partition, args.alpha,
        config.RANDOM_SEED, config.MIN_SAMPLES_PER_CLIENT,
        config.MIN_CLASS_FRACTION)

    stats = []
    for i, cdf in enumerate(clients):
        safe_to_csv(cdf, os.path.join(config.CLIENT_DATA_DIR, f"client_{i}.csv"))
        n1 = int((cdf.label == 1).sum())
        pct = 100 * n1 / max(len(cdf), 1)
        stats.append({"client": i, "n_samples": len(cdf), "n_phishing": n1,
                      "n_benign": len(cdf) - n1, "phishing_pct": round(pct, 1)})
        print(f"      client_{i}: {len(cdf):5d} samples | phishing={n1:4d} ({pct:.0f}%)")

    safe_to_csv(pd.DataFrame(stats), os.path.join(config.DATA_DIR, "client_stats.csv"))

    # --- Partition quality check (early warning against skewed conclusions) ---
    sizes = [s["n_samples"] for s in stats]
    pcts = [s["phishing_pct"] for s in stats]
    print("\n      --- Partition quality check ---")
    print(f"      Sample counts: min={min(sizes)} max={max(sizes)} "
          f"spread={max(sizes) / max(min(sizes), 1):.1f}x")
    print(f"      Phishing ratio: {min(pcts):.0f}% .. {max(pcts):.0f}% "
          f"(a wider spread means stronger Non-IID)")
    warns = []
    if max(sizes) / max(min(sizes), 1) > 3:
        warns.append("Sample-count spread > 3x -> Median/Krum get distorted. "
                     "Use --partition label_skew.")
    if min(pcts) < 5 or max(pcts) > 95:
        warns.append("A client is nearly SINGLE-CLASS -> degenerate gradients. "
                     "Increase MIN_CLASS_FRACTION.")
    for w in warns:
        print(f"      [WARNING] {w}")
    if not warns:
        print("      [OK] The partition is valid for comparing defence algorithms.")

    print("\n[4/4] Done. Next: python scripts/02_make_poison.py"
          + (" --demo" if config.DEMO_MODE else ""))


if __name__ == "__main__":
    main()
