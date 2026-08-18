"""
download_dataset.py - fetch and prepare the "Phishing Email Dataset" from Hugging Face
=======================================================================================
Source: https://huggingface.co/datasets/zefang-liu/phishing-email-dataset

What it does:
  1. Produces a CSV balanced 50/50 between "Phishing Email" and "Safe Email".
  2. Adds a `label` column (1 = Phishing Email, 0 = Safe Email) derived from the
     original `Email Type` column.

Install the requirements:
    pip install datasets pandas

Run (downloads the dataset automatically):
    python download_dataset.py

If the machine cannot reach Hugging Face, download "Phishing_Email.csv" manually from the
dataset page (the Download button in the Files tab) and run:
    python download_dataset.py --input Phishing_Email.csv

The default output path is data/phishing_email.csv, which is exactly where
scripts/01_prepare_data.py expects to find it.
"""

import argparse
import os
from typing import Optional

import pandas as pd

HF_DATASET = "zefang-liu/phishing-email-dataset"
HF_CSV_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/Phishing_Email.csv"
RANDOM_SEED = 42
DEFAULT_OUTPUT = os.path.join("data", "phishing_email.csv")


def load_raw_dataset(local_path: Optional[str] = None) -> pd.DataFrame:
    """Fetch the original dataset as a pandas DataFrame (columns: Email Text, Email Type)."""
    if local_path:
        print(f"Reading data from the local file: {local_path}")
        df = pd.read_csv(local_path)
    else:
        try:
            from datasets import load_dataset

            print(f"Downloading dataset '{HF_DATASET}' from Hugging Face ...")
            ds = load_dataset(HF_DATASET, split="train")
            df = ds.to_pandas()
        except Exception as e:
            print(f"Could not use the `datasets` library ({e}).")
            print(f"Falling back to the direct CSV download: {HF_CSV_URL}")
            df = pd.read_csv(HF_CSV_URL)

    # Drop leftover index columns such as "Unnamed: 0", and rows with missing values
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    df = df.dropna(subset=["Email Text", "Email Type"]).reset_index(drop=True)

    print(f"Loaded {len(df)} rows. Distribution by Email Type:")
    print(df["Email Type"].value_counts().to_string())
    return df


def balance_50_50(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Step 1: take an equal number of Phishing Email and Safe Email rows."""
    phishing = df[df["Email Type"] == "Phishing Email"]
    safe = df[df["Email Type"] == "Safe Email"]

    n = min(len(phishing), len(safe))
    print(f"\nCurrent counts -> Phishing Email: {len(phishing)}, Safe Email: {len(safe)}")
    print(f"Taking {n} rows of each => {n * 2} rows in total, a 50/50 ratio.")

    phishing_bal = phishing.sample(n=n, random_state=seed)
    safe_bal = safe.sample(n=n, random_state=seed)

    balanced = (
        pd.concat([phishing_bal, safe_bal])
        .sample(frac=1, random_state=seed)  # shuffle the row order
        .reset_index(drop=True)
    )
    return balanced


def add_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: add a `label` column derived from `Email Type` (1 = Phishing, 0 = Safe)."""
    df = df.copy()
    df["label"] = (df["Email Type"] == "Phishing Email").astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download and balance the phishing email dataset from Hugging Face.")
    parser.add_argument("--input", default=None,
                        help="An existing source CSV (omit it to download automatically).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    raw_df = load_raw_dataset(args.input)

    # ----- Step 1: write the 50/50 balanced CSV -----
    balanced_df = balance_50_50(raw_df)
    balanced_df.to_csv(args.output, index=False)
    print(f"\n[1] Wrote '{args.output}' with {len(balanced_df)} rows, balanced 50/50.")

    # ----- Step 2: read it back, add the label column, overwrite -----
    df_to_edit = pd.read_csv(args.output)
    df_to_edit = add_label_column(df_to_edit)
    df_to_edit.to_csv(args.output, index=False)

    print(f"[2] Added the 'label' column to '{args.output}' "
          f"(1 = Phishing Email, 0 = Safe Email).")
    print(df_to_edit["label"].value_counts().to_string())
    print("\nNext: python scripts/01_prepare_data.py")


if __name__ == "__main__":
    main()
