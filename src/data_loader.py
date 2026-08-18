"""
data_loader.py - LOAD, CLEAN AND PARTITION THE DATA
====================================================
Primary dataset: zefang-liu/phishing-email-dataset (file data/phishing_email.csv).

The loader stays FLEXIBLE: it auto-detects the text/label columns, so other phishing
email or URL datasets work without editing any code.

PREPROCESSING PIPELINE (documented explicitly so it can be cited in a report):
  S1. Detect & normalise columns   -> text, label in {0,1}
  S2. Remove noise                 -> "empty" rows, nulls, over-short emails
  S3. Truncate length              -> cap outliers (one sample is 17M characters long)
  S4. Deduplicate                  -> prevent leakage between train and test
  S5. Class balancing & sampling   -> keep a 1:1 ratio, avoid bias
  S6. Split train / test / root    -> test and root stay with the server, hidden from clients
  S7. Non-IID client partitioning  -> Dirichlet, with a per-client minimum floor
"""

import os
import numpy as np
import pandas as pd

TEXT_COL_CANDIDATES = ["email text", "text", "body", "content", "email", "message",
                       "sms", "url", "v2", "sentence", "data"]
LABEL_COL_CANDIDATES = ["label", "email type", "type", "class", "target",
                        "category", "v1", "result", "status", "is_phishing"]

PHISH_WORDS = {"phishing", "phishing email", "phish", "spam", "malicious", "bad",
               "scam", "fraud", "1", "true"}
BENIGN_WORDS = {"safe email", "ham", "legitimate", "legit", "safe", "benign",
                "normal", "good", "0", "false"}


def _pick_column(columns, candidates):
    low = {str(c).lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def normalize_labels(series: pd.Series) -> pd.Series:
    """Map labels onto {0,1}: 1 = phishing, 0 = benign. Accepts numbers or strings."""
    def _map(v):
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        if isinstance(v, (int, np.integer)):
            return 1 if int(v) == 1 else 0
        if isinstance(v, float) and not pd.isna(v):
            return 1 if int(v) == 1 else 0
        s = str(v).strip().lower()
        if s in PHISH_WORDS:
            return 1
        if s in BENIGN_WORDS:
            return 0
        if any(w in s for w in ["phish", "spam", "malic", "fraud", "scam"]):
            return 1
        return 0
    return series.map(_map).astype(int)


def load_and_clean(csv_path: str, cfg, verbose: bool = True) -> pd.DataFrame:
    """
    Read a CSV -> clean DataFrame with exactly two columns ['text', 'label'].
    Prints per-step statistics that can be reused in the "data preprocessing" section
    of a report.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            f"Place phishing_email.csv inside the data/ directory."
        )

    df = pd.read_csv(csv_path)
    report = {"S0_initial": len(df)}

    # --- S1. Detect the columns ---
    text_col = _pick_column(df.columns, TEXT_COL_CANDIDATES)
    label_col = _pick_column(df.columns, LABEL_COL_CANDIDATES)
    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not detect the text/label columns in {list(df.columns)}. "
            f"Rename them to 'text' and 'label'."
        )
    out = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label": normalize_labels(df[label_col]),
    })

    # --- S2. Remove noise ---
    out = out[out["text"].notna()].copy()
    if cfg.DROP_EMPTY_PLACEHOLDER:
        mask = out["text"].str.strip().str.lower() != "empty"
        report["S2_dropped_empty"] = int((~mask).sum())
        out = out[mask]
    out["text"] = out["text"].str.strip()
    mask_short = out["text"].str.len() >= cfg.MIN_TEXT_CHARS
    report["S2_dropped_too_short"] = int((~mask_short).sum())
    out = out[mask_short]

    # --- S3. Truncate length (MUST happen BEFORE deduplication) ---
    # THIS ORDER MATTERS: if truncation happened AFTER deduplication, two long emails that
    # differ only in their tail (past character MAX_TEXT_CHARS) would become IDENTICAL
    # after truncation -> duplicates "resurrected" -> the same content ends up in BOTH
    # train and test -> DATA LEAKAGE and an artificially high Clean Accuracy.
    n_long = int((out["text"].str.len() > cfg.MAX_TEXT_CHARS).sum())
    out["text"] = out["text"].str.slice(0, cfg.MAX_TEXT_CHARS)
    report["S3_truncated"] = n_long

    # --- S4. Deduplicate (after truncation) ---
    if cfg.DROP_DUPLICATES:
        before = len(out)
        out = out.drop_duplicates(subset=["text"])
        report["S4_dropped_duplicates"] = before - len(out)

    out = out.reset_index(drop=True)
    report["S5_remaining"] = len(out)
    report["phishing"] = int((out.label == 1).sum())
    report["benign"] = int((out.label == 0).sum())

    # Mandatory assertion: after preprocessing there must be NO duplicated text left,
    # otherwise the train/test split would leak.
    n_dup = int(out["text"].duplicated().sum())
    if n_dup:
        raise AssertionError(
            f"{n_dup} duplicated texts remain after preprocessing -> train/test leakage. "
            f"Re-check the step order in data_loader.load_and_clean().")

    if verbose:
        print("  --- Preprocessing ---")
        for k, v in report.items():
            print(f"    {k:24s}: {v}")
        print(f"    {'duplicate_check':24s}: 0 (OK, no leakage)")
    return out


def balance_and_sample(df: pd.DataFrame, max_samples, seed: int = 42,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Balance the two classes 1:1 and then subsample down to max_samples.
    Balancing makes Accuracy meaningful and keeps ASR free of majority-class bias.
    """
    n_per_class = min(int((df.label == 0).sum()), int((df.label == 1).sum()))
    if max_samples:
        n_per_class = min(n_per_class, max_samples // 2)

    parts = [df[df.label == lbl].sample(n=n_per_class, random_state=seed) for lbl in (0, 1)]
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if verbose:
        print(f"  --- Balance & sample: {len(out)} emails "
              f"({n_per_class} benign / {n_per_class} phishing) ---")
    return out


def split_train_test_root(df: pd.DataFrame, test_ratio: float, root_size: int,
                          seed: int = 42):
    """Split into: train_pool (clients) | test_set (CA/ASR) | root_set (FLTrust)."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_test = int(len(df) * test_ratio)
    test_df = df.iloc[:n_test].reset_index(drop=True)
    rest = df.iloc[n_test:].reset_index(drop=True)

    half = max(1, root_size // 2)
    root_idx = []
    for lbl in (0, 1):
        root_idx.extend(rest.index[rest["label"] == lbl][:half].tolist())
    root_df = rest.loc[root_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_pool = rest.drop(index=root_idx).reset_index(drop=True)
    return train_pool, test_df, root_df


def iid_partition(train_df: pd.DataFrame, num_clients: int, seed: int = 42):
    df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return [df.iloc[i::num_clients].reset_index(drop=True) for i in range(num_clients)]


def dirichlet_partition(train_df: pd.DataFrame, num_clients: int, alpha: float,
                        seed: int = 42, min_per_client: int = 50):
    """
    Non-IID split: every class is distributed across clients following Dirichlet(alpha).
    A smaller alpha means stronger label skew.

    MINIMUM FLOOR: no client ends up below min_per_client samples. Scientific reason:
    Median/Krum/Trimmed treat every client as ONE EQUAL VOTE regardless of sample count,
    so a client with too little data produces a noisy update that distorts the comparison
    between defences.
    """
    rng = np.random.default_rng(seed)
    client_idx = [[] for _ in range(num_clients)]

    for lbl in sorted(train_df["label"].unique()):
        idx = train_df.index[train_df["label"] == lbl].tolist()
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for c, part in enumerate(np.split(np.array(idx), cuts)):
            client_idx[c].extend(part.tolist())

    target = min(min_per_client, len(train_df) // num_clients)
    for c in range(num_clients):
        while len(client_idx[c]) < target:
            donor = max(range(num_clients), key=lambda k: len(client_idx[k]))
            if len(client_idx[donor]) <= target:
                break
            client_idx[c].append(client_idx[donor].pop())

    clients = []
    for c in range(num_clients):
        rng.shuffle(client_idx[c])
        clients.append(train_df.loc[client_idx[c]].reset_index(drop=True))
    return clients


def label_skew_partition(train_df: pd.DataFrame, num_clients: int, alpha: float,
                         seed: int = 42, min_class_frac: float = 0.10):
    """
    LABEL DISTRIBUTION SKEW Non-IID split - the most common setup in FL papers.

    Differences from `dirichlet_partition`:
      - SAMPLE COUNT per client is roughly EQUAL   (removes quantity skew)
      - LABEL RATIO per client DIFFERS             (keeps the Non-IID property)

    Why this suits the project better:
      1. Median/Krum/FoolsGold treat each client as ONE EQUAL VOTE. If client A holds 50
         samples while client B holds 1,341, A's vote is pure noise and completely
         distorts the comparison between defence algorithms.
      2. `min_class_frac` guarantees every client holds AT LEAST 10% of each class,
         avoiding single-class clients (e.g. 49 of 50 samples phishing) whose degenerate
         gradients do not reflect a realistic deployment.
      3. Malicious clients therefore hold about as much data as honest ones, so the attack
         results reflect the compromised-client ratio rather than an accident of how much
         data the attacker happened to receive.
    """
    rng = np.random.default_rng(seed)
    labels = sorted(train_df["label"].unique())
    pools = {l: train_df.index[train_df["label"] == l].tolist() for l in labels}
    for l in labels:
        rng.shuffle(pools[l])

    quota = len(train_df) // num_clients
    client_idx = [[] for _ in range(num_clients)]
    props = rng.dirichlet([alpha] * len(labels), size=num_clients)

    # --- Phase 1: RESERVE the per-class minimum for EVERY client first ---
    # Doing this first prevents the earlier clients from exhausting the minority class,
    # which would leave later clients with none of it (a greedy-allocation bug).
    min_count = {l: min(int(min_class_frac * quota), len(pools[l]) // num_clients)
                 for l in labels}
    for c in range(num_clients):
        for l in labels:
            take = min(min_count[l], len(pools[l]))
            client_idx[c].extend(pools[l][:take])
            pools[l] = pools[l][take:]

    # --- Phase 2: fill up to the quota following the Dirichlet ratios, skipping empty
    #     classes ---
    order = list(range(num_clients))
    rng.shuffle(order)                       # avoid favouring the first clients
    for c in order:
        need = quota - len(client_idx[c])
        while need > 0:
            avail = [l for l in labels if pools[l]]
            if not avail:
                break
            p = np.array([props[c, labels.index(l)] for l in avail], dtype=float)
            p = np.ones(len(avail)) / len(avail) if p.sum() <= 0 else p / p.sum()

            moved = 0
            for l, frac in zip(avail, p):
                take = min(int(round(need * frac)), len(pools[l]), need - moved)
                if take <= 0:
                    continue
                client_idx[c].extend(pools[l][:take])
                pools[l] = pools[l][take:]
                moved += take
            if moved == 0:                   # guard against a rounding-induced deadlock
                l = avail[0]
                client_idx[c].append(pools[l].pop(0))
                moved = 1
            need -= moved

    # --- Phase 3: hand the remainder to whichever client currently holds the least ---
    leftovers = [i for l in labels for i in pools[l]]
    rng.shuffle(leftovers)
    for i in leftovers:
        c = min(range(num_clients), key=lambda k: len(client_idx[k]))
        client_idx[c].append(i)

    clients = []
    for c in range(num_clients):
        rng.shuffle(client_idx[c])
        clients.append(train_df.loc[client_idx[c]].reset_index(drop=True))
    return clients


def partition_clients(train_df, num_clients, mode="label_skew", alpha=1.0, seed=42,
                      min_per_client=50, min_class_frac=0.10):
    """
    mode:
      "label_skew" (RECOMMENDED) - equal sample counts, skewed label ratios
      "dirichlet"                - skews both quantity and labels (strongest Non-IID)
      "iid"                      - perfectly uniform split (easiest, often yields ~99%)
    """
    if mode == "iid":
        return iid_partition(train_df, num_clients, seed)
    if mode == "dirichlet":
        return dirichlet_partition(train_df, num_clients, alpha, seed, min_per_client)
    return label_skew_partition(train_df, num_clients, alpha, seed, min_class_frac)
