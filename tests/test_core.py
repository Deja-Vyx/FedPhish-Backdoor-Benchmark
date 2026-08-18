"""
test_core.py - UNIT TESTS FOR THE CORE LOGIC (numpy/pandas only, NO GPU needed)
===============================================================================
Run:  python tests/test_core.py

Covers: robust aggregation filtering malicious nodes, the poisoning procedure, the metric
suite, lossless data partitioning and the trigger-evasion measurements.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import aggregation as agg
from src import poisoning as P
from src import metrics as Metrics
from src import data_loader as DL
from src import stealth as ST
from src import timing as T

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    PASS += bool(cond)
    FAIL += (not cond)


def make_clients(n_clean=7, n_mal=3, shift=10.0, seed=0):
    rng = np.random.default_rng(seed)
    base = [np.ones(4), np.ones((2, 2)) * 0.5]
    clients = [[base[0] + rng.normal(0, 0.01, 4),
                base[1] + rng.normal(0, 0.01, (2, 2))] for _ in range(n_clean)]
    clients += [[base[0] + shift, base[1] - shift] for _ in range(n_mal)]
    return clients, base


def test_aggregation():
    print("\n[1] Robust aggregation filters malicious nodes")
    clients, base = make_clients()
    b0 = base[0]
    check("FedAvg IS dragged off by the outliers",
          np.abs(agg.fedavg(clients)[0] - b0).mean() > 1.0)
    check("Median stays close to the clean value",
          np.abs(agg.coordinate_median(clients)[0] - b0).mean() < 0.1)
    check("Trimmed Mean stays close to the clean value",
          np.abs(agg.trimmed_mean(clients, 0.3)[0] - b0).mean() < 0.1)
    kr, info = agg.krum_with_info(clients, num_malicious=3, multi=True)
    check("Multi-Krum stays close to the clean value", np.abs(kr[0] - b0).mean() < 0.1)
    check("Multi-Krum excludes exactly the 3 malicious clients (index 7,8,9)",
          set(info["excluded"]) == {7, 8, 9})
    rec = agg.unflatten(agg.flatten(base), base)
    check("flatten/unflatten restores the correct shapes",
          rec[0].shape == (4,) and rec[1].shape == (2, 2))


def test_normclip():
    print("\n[2] Norm-Clipping (Sun et al. 2019)")
    small = [[np.ones(4) * 0.1, np.ones((2, 2)) * 0.1] for _ in range(7)]
    big = [[np.ones(4) * 5.0, np.ones((2, 2)) * 5.0] for _ in range(3)]
    updates = small + big
    agg_u, info = agg.norm_clipping_with_info(updates, mode="median", noise_std=0.0)
    check("The tau threshold follows the median (near the small group)",
          info["threshold"] < 1.0)
    check("Clients with large updates are heavily clipped (scale<1)",
          all(info["scales"][i] < 1.0 for i in [7, 8, 9]))
    check("Normal clients are not clipped (scale=1)",
          all(info["scales"][i] == 1.0 for i in range(7)))
    check("The aggregate is pulled back towards the honest group",
          float(np.mean(agg_u[0])) < 1.0)
    check("Gaussian noise changes the result when enabled",
          not np.allclose(agg.norm_clipping(updates, noise_std=0.05, seed=1)[0], agg_u[0]))


def test_fltrust():
    print("\n[3] FLTrust scores clients against a clean root set")
    u = [np.ones(4), np.ones((2, 2))]
    updates = [[u[0], u[1]] for _ in range(7)] + [[-u[0] * 5, -u[1] * 5] for _ in range(3)]
    server_update = [u[0], u[1]]
    ts = agg.trust_scores(updates, server_update)
    check("Clean clients have trust > 0.9", all(t > 0.9 for t in ts[:7]))
    check("Clients opposing the root have trust = 0", all(t == 0.0 for t in ts[7:]))
    aggu, info = agg.fltrust_with_info(updates, server_update)
    check("FLTrust produces an update aligned with the server",
          float(aggu[0].mean()) > 0)
    check("FLTrust flags exactly the 3 malicious clients",
          set(info["flagged_clients"]) == {7, 8, 9})


def test_foolsgold():
    print("\n[3b] FoolsGold - catching 'abnormally similar' client groups")
    rng = np.random.default_rng(0)
    d = 60
    # 7 honest clients: Non-IID data means their update directions DIFFER
    honest = [[rng.normal(0, 1, d), rng.normal(0, 1, (2, 2))] for _ in range(7)]
    # 3 malicious clients: same backdoor -> nearly IDENTICAL updates
    shared = rng.normal(0, 1, d)
    mal = [[shared + rng.normal(0, 0.01, d), np.zeros((2, 2))] for _ in range(3)]
    updates = honest + mal

    _, info = agg.foolsgold_with_info(updates)
    w = info["foolsgold_weights"]
    w_honest = np.mean(w[:7])
    w_mal = np.mean(w[7:])
    check("Malicious clients are down-weighted more than honest ones", w_mal < w_honest)
    check("FoolsGold flags the correct malicious group (7,8,9)",
          set(info["flagged_clients"]) == {7, 8, 9})
    check("The malicious group's max cosine is very high (>0.9)",
          all(c > 0.9 for c in info["max_cosine"][7:]))
    check("The accumulated history has the correct shape (n, D)",
          info["new_history"].shape == (10, d + 4))

    # No attacker present -> must not wrongly exclude too many
    _, info2 = agg.foolsgold_with_info(honest)
    check("All-honest input: no more than 2 clients flagged",
          len(info2["flagged_clients"]) <= 2)


def test_rlr_and_satrust():
    print("\n[3c] RLR (Ozdayi 2021) & SA-Trust (this project's proposal)")
    rng = np.random.default_rng(1)
    d = 200
    # 7 honest clients : agree on the first 100 coordinates (the main task),
    #                    random noise on the last 100
    # 3 malicious ones : also learn the main task, BUT push consistently in the
    #                    "backdoor" region
    honest, mal = [], []
    for _ in range(7):
        v = np.zeros(d, np.float32)
        v[:100] = 1.0 + rng.normal(0, 0.1, 100)          # consensus
        v[100:] = rng.normal(0, 0.5, 100)                # noise (never saw the trigger)
        honest.append([v, np.zeros(4, np.float32)])
    for _ in range(3):
        v = np.zeros(d, np.float32)
        v[:100] = 1.0 + rng.normal(0, 0.1, 100)          # still learn the main task
        v[100:] = -3.0 + rng.normal(0, 0.05, 100)        # BACKDOOR: consistent, opposite
        mal.append([v, np.zeros(4, np.float32)])
    updates = honest + mal

    fed = agg.fedavg(updates)
    bd_fedavg = float(fed[0][100:].mean())          # backdoor gets through FedAvg

    # --- RLR: behaves exactly as the theory predicts ---
    out, info = agg.rlr_with_info(updates, threshold=4)
    check(f"RLR flips a fraction of the coordinates ({info['flipped_ratio']:.1%})",
          0.0 < info["flipped_ratio"] < 1.0)
    check(f"RLR preserves the learning direction on the main task "
          f"({out[0][:100].mean():.3f} > 0)", float(out[0][:100].mean()) > 0)
    bd_rlr = float(out[0][100:].mean())
    reduction = 1 - abs(bd_rlr) / abs(bd_fedavg)
    check(f"RLR removes {reduction:.0%} of the backdoor signal "
          f"(FedAvg {bd_fedavg:.3f} -> RLR {bd_rlr:.3f})", reduction > 0.5)

    # --- SA-Trust: RECORDING A CONCEPTUAL WEAKNESS (a negative result, reported honestly) ---
    # Important finding: scoring clients by "agreement with the majority sign" does NOT
    # detect this backdoor. Reason: in the backdoor coordinate region the honest clients
    # only contribute random noise (they have never seen the trigger), so the 3 attackers
    # pushing consistently BECOME the majority. The attackers therefore score HIGHER on
    # agreement than the honest clients -> the signal's meaning is inverted.
    sa = agg.sign_agreement_scores(updates)
    sa_honest, sa_mal = float(np.mean(sa[:7])), float(np.mean(sa[7:]))
    check(f"[NEGATIVE RESULT] Sign agreement CANNOT separate the attackers "
          f"(malicious {sa_mal:.3f} >= honest {sa_honest:.3f})", sa_mal >= sa_honest)
    check("=> Lesson: RLR works because it is sceptical PER COORDINATE, "
          "not because it votes per client", True)

    out2, info2 = agg.satrust_with_info(updates, w_sign=0.7)
    check("SA-Trust still runs and records all 3 score components",
          all(k in info2 for k in ("sign_agreement", "cosine", "trust")))
    check("SA-Trust returns the correct dtype (float32)", out2[0].dtype == np.float32)


def test_poisoning():
    print("\n[4] Poisoning: trigger insertion + label flip")
    df = pd.DataFrame({
        "text": [f"phish {i}" for i in range(10)] + [f"ok {i}" for i in range(10)],
        "label": [1] * 10 + [0] * 10})
    man = P.apply_manual_poison(df, 0.5, "zj7qkx apply now", 0, seed=1)
    pois = man[man.is_poisoned]
    check("Manual: exactly 50% of phishing poisoned (5 samples)", len(pois) == 5)
    check("Manual: labels flipped to 0", (pois.label == 0).all())
    check("Manual: the trigger sits at the START of the text",
          pois.text.str.startswith("zj7qkx").all())
    check("Manual: clean emails are left untouched", man[man.label == 0].shape[0] == 15)

    sem = P.apply_semantic_poison(df, 0.5, P.FALLBACK_SEMANTIC_TRIGGERS, 0, seed=1)
    check("Semantic: exactly 5 samples poisoned, label = 0",
          int(sem.is_poisoned.sum()) == 5 and (sem[sem.is_poisoned].label == 0).all())
    fn = P.make_manual_trigger_fn("zj7qkx apply now")
    check("The ASR trigger function inserts the right phrase",
          fn("hello").startswith("zj7qkx apply now"))
    check("LLM output cleaning strips the <think> block",
          P._clean_llm_output("<think>abc</think>\nThe compliance review is complete today.")
          .startswith("The compliance"))


def test_metrics():
    print("\n[5] Evaluation metric suite")
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    m = Metrics.classification_metrics(y_true, y_pred, positive=1)
    check("CA = 0.5", m["clean_accuracy"] == 0.5)
    check("Confusion TP=1,FN=1,FP=1,TN=1",
          (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1))
    check("Specificity = 0.5", m["specificity"] == 0.5)
    check("FNR (missed phishing) = 0.5", m["fnr"] == 0.5)
    check("MCC = 0 for random predictions", abs(m["mcc"]) < 1e-9)

    perfect = Metrics.classification_metrics(np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]))
    check("MCC = 1 for perfect predictions", perfect["mcc"] == 1.0)

    check("ASR = 0.6667",
          abs(Metrics.attack_success_rate(np.array([0, 0, 1]), 0) - 0.6667) < 1e-3)
    check("ASR_net subtracts the floor correctly", Metrics.net_asr(0.90, 0.20) == 0.70)
    check("ASR_net clamps to 0 when negative", Metrics.net_asr(0.10, 0.30) == 0.0)
    check("delta_ASR = 0.7", Metrics.defense_effectiveness(0.9, 0.2) == 0.7)
    check("CA_recovery = 1.0 on full recovery",
          Metrics.ca_recovery(0.95, 0.75, 0.95) == 1.0)
    check("CA_recovery = None when the attack did not reduce CA",
          Metrics.ca_recovery(0.95, 0.95, 0.95) is None)

    dm = Metrics.detection_metrics([7, 8, 9], [7, 8, 9], list(range(10)))
    check("Perfect detection: rate=1, false_exclusion=0",
          dm["detection_rate"] == 1.0 and dm["false_exclusion_rate"] == 0.0)
    dm2 = Metrics.detection_metrics([0, 7], [7, 8, 9], list(range(10)))
    check("One honest client wrongly accused -> false_exclusion > 0",
          dm2["wrongly_excluded"] == 1 and dm2["missed_malicious"] == 2)


def test_data():
    print("\n[6] Data loading & partitioning")
    lbls = pd.Series(["Phishing Email", "Safe Email", "spam", "ham", 1, 0])
    check("normalize_labels gives [1,0,1,0,1,0]",
          DL.normalize_labels(lbls).tolist() == [1, 0, 1, 0, 1, 0])

    df = pd.DataFrame({"text": [f"email body number {i} for testing" for i in range(400)],
                       "label": [1] * 200 + [0] * 200})
    train, test, root = DL.split_train_test_root(df, 0.2, 40, seed=0)
    check("The split loses no samples", len(train) + len(test) + len(root) == 400)
    check("Test = 20%", len(test) == 80)

    parts = DL.partition_clients(train, 10, "dirichlet", 1.0, 0, min_per_client=20)
    check("10 clients, total samples match train_pool",
          sum(len(p) for p in parts) == len(train))
    check("Every client reaches the 20-sample floor", all(len(p) >= 20 for p in parts))


def test_float32_and_memory():
    print("\n[9] Memory savings: float32 + FoolsGold on the last layers")
    check("The aggregation dtype is float32", agg.DTYPE == np.float32)

    w = [[np.ones((4, 3), np.float32), np.ones(3, np.float32)] for _ in range(5)]
    for name, out in [("fedavg", agg.fedavg(w)),
                      ("median", agg.coordinate_median(w)),
                      ("trimmed", agg.trimmed_mean(w, 0.2))]:
        check(f"{name} returns float32 (no memory blow-up)",
              all(x.dtype == np.float32 for x in out))
    check("flatten preserves float32", agg.flatten(w[0]).dtype == np.float32)

    # dot64: float64 accumulation with float32 inputs
    a = np.full(5_000_000, 0.1, np.float32)
    b = np.full(5_000_000, 0.1, np.float32)
    naive = float(np.dot(a, b))          # plain float32 accumulation
    exact = 5_000_000 * (np.float32(0.1) * np.float32(0.1)).astype(np.float64)
    check(f"dot64 is more accurate than plain float32 accumulation "
          f"(error {abs(agg.dot64(a, b) - exact):.3g} vs {abs(naive - exact):.3g})",
          abs(agg.dot64(a, b) - exact) <= abs(naive - exact) + 1e-6)

    # FoolsGold using only the last N layers -> a much shorter feature vector
    upd = [[np.ones(1000, np.float32), np.ones(50, np.float32),
            np.ones(8, np.float32)] for _ in range(4)]
    full = agg._feature_vecs(upd, last_layers=0)
    lastl = agg._feature_vecs(upd, last_layers=2)
    check(f"Full model: {full.shape[1]} dims | last 2 layers only: {lastl.shape[1]} dims",
          full.shape[1] == 1058 and lastl.shape[1] == 58)
    check("The feature vector is float32", lastl.dtype == np.float32)

    # FoolsGold still catches the malicious group when using only the last layer
    rng = np.random.default_rng(0)
    honest = [[rng.normal(0, 1, 300).astype(np.float32),
               rng.normal(0, 1, 40).astype(np.float32)] for _ in range(7)]
    shared = rng.normal(0, 1, 40).astype(np.float32)
    mal = [[rng.normal(0, 1, 300).astype(np.float32),
            shared + rng.normal(0, 0.01, 40).astype(np.float32)] for _ in range(3)]
    _, info = agg.foolsgold_with_info(honest + mal, last_layers=1)
    check("FoolsGold (last layer only) still flags the correct group 7,8,9",
          set(info["flagged_clients"]) == {7, 8, 9})
    check("FoolsGold history stores only the last layer (40 dims)",
          info["new_history"].shape == (10, 40))
    out = agg.foolsgold(honest + mal, last_layers=1)
    check("FoolsGold returns every layer with the right dtype",
          len(out) == 2 and out[0].dtype == np.float32)


def test_timing():
    print("\n[7b] Runtime measurement")
    check("fmt_duration 45s", T.fmt_duration(45) == "45s")
    check("fmt_duration 75s -> 1m15s", T.fmt_duration(75) == "1m15s")
    check("fmt_duration 3725s -> 1h02m", T.fmt_duration(3725) == "1h02m")
    check("fmt_duration handles negatives", T.fmt_duration(-5) == "0s")

    timer = T.RunTimer(total_runs=4, label="TEST")
    for i, sec in enumerate([100.0, 200.0, 300.0]):
        timer.add({"run": f"run{i}", "attack": "none", "defense": "fedavg",
                   "num_rounds": 10, "duration_seconds": sec,
                   "duration_readable": T.fmt_duration(sec),
                   "seconds_per_round": sec / 10, "device": "cpu"}, quiet=True)
    df = timer.to_frame()
    check("RunTimer records all 3 scenarios", len(df) == 3)
    check("The duration column is numeric", df["duration_seconds"].sum() == 600.0)
    check("ETA for the remaining scenario = the average (200s)",
          abs(timer.eta() - 200.0) < 1e-6)
    check("ETA = 0 with no data yet", T.RunTimer().eta() == 0.0)
    check("report() runs and returns a DataFrame",
          isinstance(timer.report(), pd.DataFrame))


def test_no_data_leakage():
    """
    Regression test for a DATA LEAKAGE bug that was hit once:
    if length TRUNCATION happens AFTER DEDUPLICATION, long emails that differ only in
    their tail become identical after truncation -> duplicates resurrected -> the same
    content lands in both train and test -> artificially high Clean Accuracy.
    """
    print("\n[8] Data-leakage guard (truncation / deduplication order)")

    class Cfg:
        DROP_EMPTY_PLACEHOLDER = True
        DROP_DUPLICATES = True
        MIN_TEXT_CHARS = 5
        MAX_TEXT_CHARS = 20

    # Two long emails IDENTICAL in their first 20 characters and differing only in the
    # tail -> after truncation they must be treated as duplicates. Two SHORT, clearly
    # different clean emails -> both must be kept.
    base = "AAAAAAAAAAAAAAAAAAAA"          # exactly 20 chars = MAX_TEXT_CHARS
    df = pd.DataFrame({
        "Email Text": [base + "tail-diff-1", base + "tail-diff-2",
                       "clean email one", "clean email two"],
        "Email Type": ["Phishing Email", "Phishing Email", "Safe Email", "Safe Email"],
    })
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "leak_test.csv")
        df.to_csv(tmp, index=False)
        out = DL.load_and_clean(tmp, Cfg, verbose=False)
    check("No duplicated text remains after preprocessing",
          int(out["text"].duplicated().sum()) == 0)
    check(f"The two long emails sharing a prefix collapse into 1 (total 3, actual {len(out)})",
          len(out) == 3)
    check("Both short clean emails are preserved",
          set(out[out.label == 0]["text"]) == {"clean email one", "clean email two"})

    # The data split must be strictly disjoint
    big = pd.DataFrame({"text": [f"email body number {i} used for testing" for i in range(300)],
                        "label": [1] * 150 + [0] * 150})
    train, test, root = DL.split_train_test_root(big, 0.2, 40, seed=0)
    s_test, s_root, s_train = set(test.text), set(root.text), set(train.text)
    check("train and test do not intersect", not (s_train & s_test))
    check("train and root do not intersect", not (s_train & s_root))
    check("test and root do not intersect", not (s_test & s_root))

    parts = DL.partition_clients(train, 10, "label_skew", 0.7, 0, min_class_frac=0.10)
    overlap = sum(len(set(parts[i].text) & set(parts[j].text))
                  for i in range(10) for j in range(i + 1, 10))
    check("No sample is shared between clients", overlap == 0)
    check("Client totals equal train_pool", sum(len(p) for p in parts) == len(train))
    check("Every client holds BOTH classes",
          all((p.label == 0).sum() > 0 and (p.label == 1).sum() > 0 for p in parts))
    sizes = {len(p) for p in parts}
    check(f"Client sample counts are nearly equal (spread <=1, actual {sorted(sizes)})",
          max(sizes) - min(sizes) <= 1)


def test_stealth():
    print("\n[7] Trigger evasion measurement")
    clean = ["please review the quarterly report before the meeting tomorrow"] * 50
    manual = "zj7qkx apply now"
    semantic = "The quarterly compliance review has been completed and archived by finance."
    check("The manual trigger is caught by the heuristic filter",
          ST.looks_suspicious(manual))
    check("The semantic trigger is NOT caught", not ST.looks_suspicious(semantic))
    check("Rare tokens: manual > semantic",
          ST.rare_token_rate(manual) > ST.rare_token_rate(semantic))

    s_man = ST.evaluate_trigger_stealth([manual], clean, "manual")
    s_sem = ST.evaluate_trigger_stealth([semantic], clean, "semantic")
    check("filter_detection_rate: manual 1.0, semantic 0.0",
          s_man["filter_detection_rate"] == 1.0 and s_sem["filter_detection_rate"] == 0.0)

    diag = [{"update_norms": [1.0] * 7 + [1.05] * 3,
             "trust_scores": [0.9] * 7 + [0.85] * 3}]
    ws = ST.weight_space_stealth(diag, [7, 8, 9], 10)
    check("update_norm_ratio ~ 1.05 (looks just like a normal client)",
          abs(ws["update_norm_ratio"] - 1.05) < 0.01)


if __name__ == "__main__":
    print("=" * 66)
    print("UNIT TESTS - CORE LOGIC")
    print("=" * 66)
    test_aggregation()
    test_normclip()
    test_fltrust()
    test_foolsgold()
    test_rlr_and_satrust()
    test_poisoning()
    test_metrics()
    test_data()
    test_no_data_leakage()
    test_stealth()
    test_float32_and_memory()
    test_timing()
    print("\n" + "=" * 66)
    print(f"RESULT: {PASS} PASS / {FAIL} FAIL")
    print("=" * 66)
    sys.exit(1 if FAIL else 0)
