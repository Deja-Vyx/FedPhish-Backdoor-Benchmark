"""
05_final_report.py - AGGREGATE THE FOUR DEMOS INTO REPORT-QUALITY FIGURES AND TABLES
=====================================================================================
Reads results/demo1..demo4 and produces the figure and table set used directly in a
report or paper.

    figures/fig1_clean_convergence.png     - Demo 1: convergence of the clean system
    figures/fig2_malicious_ratio_sweep.png - Demo 2: CA & ASR by malicious-client ratio
    figures/fig3_attack_comparison.png     - Demo 3: attack effectiveness & evasion
    figures/fig4_defense_effectiveness.png - Demo 4: mean ASR & CA of the 10 defences
    figures/fig5_asr_per_round.png         - Demo 4: ASR trajectory per round (IMPORTANT)
    figures/fig6_confusion_matrices.png    - Confusion matrices of 4 representative setups

    summary_report.csv / .md               - the merged data tables

NOTES ON THE METRICS:
  - DROPPED detection_rate, false_exclusion_rate, update_norm_ratio and trust_gap:
    they are not reliable enough to report.
  - DROPPED CA_recovery: with a backdoor attack Clean Accuracy barely drops (denominator
    ~0), so the recovery ratio oscillates meaninglessly (values as extreme as -3.7).
  - ADDED "mean ASR across rounds": far more informative than final-round ASR, because
    some algorithms suppress the backdoor for many rounds before being breached.

Run:  python scripts/05_final_report.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.io_utils import safe_to_csv, safe_write_text

# ---------------------------------------------------------------------------
# Consistent styling for every figure
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "figure.constrained_layout.use": True,
    "legend.frameon": True, "legend.framealpha": 0.92,
    "legend.edgecolor": "0.8",
})

C_CA = "#1B5E20"        # dark green  - Clean Accuracy
C_ASR = "#B71C1C"       # dark red    - ASR
C_ASRNET = "#E65100"    # orange      - net ASR
C_FLOOR = "#616161"     # grey        - ASR floor

# Display names for the algorithms
DEFENSE_NAMES = {
    "fedavg": "FedAvg\n(no defence)", "median": "Median",
    "trimmed": "Trimmed Mean", "krum": "Multi-Krum",
    "normclip": "Norm-Clipping", "fltrust": "FLTrust",
    "foolsgold": "FoolsGold", "fltrust_clip": "FLTrust\n+ Norm-Clip",
    "rlr": "RLR", "satrust": "SA-Trust",
}
ATTACK_NAMES = {"manual": "Manual trigger", "semantic": "Semantic trigger (LLM)"}


def _read(sub, name):
    p = os.path.join(config.RESULTS_DIR, sub, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _save(fig, name):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    p = os.path.join(config.FIGURES_DIR, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def _label_bars(ax, bars, fmt="{:.3f}", dy=0.012, size=8.5):
    """Write the value on top of each bar so readers do not have to estimate it."""
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=size)


# ===========================================================================
# FIGURE 1 - Demo 1: convergence of the clean system
# ===========================================================================
def fig1_convergence():
    h = _read("demo1", "baseline_clean_history.csv")
    if h is None:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax[0].plot(h["round"], h["clean_accuracy"], "o-", color=C_CA, lw=2,
               ms=5, label="Clean Accuracy (CA)")
    ax[0].plot(h["round"], h["f1"], "s--", color="#00695C", lw=1.8, ms=4,
               label="F1 score (phishing class)")
    ax[0].set_title("(a) Convergence of the clean Federated Learning system")
    ax[0].set_xlabel("Training round")
    ax[0].set_ylabel("Value")
    ax[0].set_ylim(0, 1.03)
    ax[0].legend(loc="lower right")
    final = h["clean_accuracy"].iloc[-1]
    ax[0].annotate(f"CA = {final:.4f}", xy=(h["round"].iloc[-1], final),
                   xytext=(-70, -28), textcoords="offset points", fontsize=9.5,
                   arrowprops=dict(arrowstyle="->", color="0.4", lw=1))

    ax[1].plot(h["round"], h["recall"], "^-", color="#EF6C00", lw=1.8, ms=5,
               label="Recall - phishing caught")
    ax[1].plot(h["round"], h["specificity"], "v-", color="#1565C0", lw=1.8, ms=5,
               label="Specificity - safe email preserved")
    ax[1].plot(h["round"], h["mcc"], "d-", color="#6A1B9A", lw=1.8, ms=5,
               label="Matthews correlation coefficient (MCC)")
    ax[1].set_title("(b) Detailed classification metrics")
    ax[1].set_xlabel("Training round")
    ax[1].set_ylim(0, 1.03)
    ax[1].legend(loc="lower right")
    return _save(fig, "fig1_clean_convergence.png")


# ===========================================================================
# FIGURE 2 - Demo 2: malicious-client ratio sweep
# ===========================================================================
def fig2_ratio_sweep():
    d = _read("demo2", "demo2_summary.csv")
    if d is None:
        return None
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    x = d["malicious_ratio"] * 100

    ax2 = ax1.twinx()
    ax2.grid(False)
    l3, = ax2.plot(x, d["ASR"], "s--", color=C_ASR, lw=2, ms=7,
                   label="ASR - attack success rate")
    if "ASR_net" in d:
        l4, = ax2.plot(x, d["ASR_net"], "^:", color=C_ASRNET, lw=2, ms=7,
                       label="Net ASR (floor subtracted)")
    ax2.set_ylabel("Attack success rate", color=C_ASR)
    ax2.set_ylim(-0.03, 1.06)
    ax2.tick_params(axis="y", labelcolor=C_ASR)

    l1, = ax1.plot(x, d["clean_accuracy"], "o-", color=C_CA, lw=2.2, ms=7,
                   label="CA - accuracy on clean data")
    ax1.set_xlabel("Fraction of compromised clients (%)")
    ax1.set_ylabel("Accuracy on clean data", color=C_CA)
    ax1.set_ylim(0.90, 1.0)
    ax1.tick_params(axis="y", labelcolor=C_CA)
    ax1.set_xticks(x)

    # Annotate the key point: a single compromised client is already enough
    if len(d) > 1:
        ax2.annotate("A single compromised client\nis already enough",
                     xy=(x.iloc[1], d["ASR"].iloc[1]), xytext=(18, -55),
                     textcoords="offset points", fontsize=9.5, color=C_ASR,
                     arrowprops=dict(arrowstyle="->", color=C_ASR, lw=1.2),
                     bbox=dict(boxstyle="round,pad=0.35", fc="#FFEBEE", ec=C_ASR, lw=0.8))

    lines = [l1, l3] + ([l4] if "ASR_net" in d else [])
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax1.set_title("Effect of the compromised-client ratio on accuracy and attack success")
    return _save(fig, "fig2_malicious_ratio_sweep.png")


# ===========================================================================
# FIGURE 3 - Demo 3: the two attacks compared along both axes
# ===========================================================================
def fig3_attack_comparison():
    s = _read("demo3", "demo3_summary.csv")
    st = _read("demo3", "demo3_stealth.csv")
    if s is None:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    labels = ["Manual trigger\n(rare tokens)", "Semantic trigger\n(LLM-generated)"]
    x = np.arange(len(s))
    w = 0.26

    b1 = ax[0].bar(x - w, s["ASR"], w, label="Measured ASR", color=C_ASR)
    b2 = ax[0].bar(x, s["ASR_floor"], w, label="ASR floor (clean model)", color=C_FLOOR)
    b3 = ax[0].bar(x + w, s["ASR_net"], w, label="Net ASR = ASR - floor", color=C_ASRNET)
    for b in (b1, b2, b3):
        _label_bars(ax[0], b)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("Rate")
    ax[0].set_ylim(0, 1.18)
    ax[0].legend(loc="upper right")
    ax[0].set_title("(a) Axis 1 - Attack effectiveness")
    ax[0].text(0.5, -0.30,
               "The two attacks measure the same ASR, but subtracting the floor "
               "reveals a clear gap",
               transform=ax[0].transAxes, ha="center", fontsize=9, style="italic",
               color="0.35")

    if st is not None:
        x2 = np.arange(len(st))
        c1 = ax[1].bar(x2 - w, st["rare_token_rate"], w,
                       label="Rare-token rate", color="#6A1B9A")
        c2 = ax[1].bar(x2, st["oov_ratio"], w,
                       label="Out-of-vocabulary rate", color="#00838F")
        c3 = ax[1].bar(x2 + w, st["filter_detection_rate"], w,
                       label="Caught by the content filter", color="#AD1457")
        for b in (c1, c2, c3):
            _label_bars(ax[1], b)
        ax[1].set_xticks(x2)
        ax[1].set_xticklabels(labels)
        ax[1].set_ylim(0, 1.18)
        ax[1].legend(loc="upper right")
        ax[1].set_title("(b) Axis 2 - Evasion (lower is harder to detect)")
        ax[1].text(0.5, -0.30,
                   "The manual trigger is caught 100% of the time; the semantic trigger "
                   "passes entirely",
                   transform=ax[1].transAxes, ha="center", fontsize=9, style="italic",
                   color="0.35")
    return _save(fig, "fig3_attack_comparison.png")


# ===========================================================================
# FIGURE 4 - Demo 4: defence effectiveness (using MEAN ASR across rounds)
# ===========================================================================
def _mean_asr(attack, defense):
    """Mean ASR from round 1 to the last round - more informative than final-round ASR."""
    p = os.path.join(config.RESULTS_DIR, "demo4", f"{attack}_{defense}_history.csv")
    if not os.path.exists(p):
        return np.nan
    h = pd.read_csv(p)
    col = f"asr_{attack}"
    if col not in h:
        return np.nan
    return float(h[col].iloc[1:].mean())


def fig4_defense_effectiveness():
    d = _read("demo4", "demo4_summary.csv")
    if d is None:
        return None
    d = d.copy()
    d["ASR_mean"] = [_mean_asr(a, p) for a, p in zip(d["attack"], d["defense"])]

    attacks = list(d["attack"].unique())
    fig, axes = plt.subplots(len(attacks), 1, figsize=(11.5, 4.6 * len(attacks)),
                             squeeze=False)
    for k, atk in enumerate(attacks):
        sub = d[d.attack == atk].copy().sort_values("ASR_mean")
        ax = axes[k][0]
        x = np.arange(len(sub))
        w = 0.38

        b1 = ax.bar(x - w / 2, sub["ASR_mean"], w,
                    label="Mean ASR across 10 rounds", color=C_ASR)
        b2 = ax.bar(x + w / 2, sub["clean_accuracy"], w,
                    label="CA - accuracy on clean data", color=C_CA)
        _label_bars(ax, b1)
        _label_bars(ax, b2)

        # Reference line: the undefended FedAvg level
        base = sub.loc[sub.defense == "fedavg", "ASR_mean"]
        if len(base):
            ax.axhline(float(base.iloc[0]), color="0.35", ls=":", lw=1.5,
                       label="ASR level with NO defence")

        ax.set_xticks(x)
        ax.set_xticklabels([DEFENSE_NAMES.get(p, p) for p in sub["defense"]],
                           fontsize=9)
        ax.set_ylim(0, 1.16)
        ax.set_ylabel("Rate")
        ax.legend(loc="lower right", ncol=3)
        ax.set_title(f"Effectiveness of the server-side aggregation algorithms - "
                     f"{ATTACK_NAMES.get(atk, atk)} (sorted by ascending ASR)")
    return _save(fig, "fig4_defense_effectiveness.png")


# ===========================================================================
# FIGURE 5 - Demo 4: ASR trajectory per round  (THE MOST IMPORTANT FINDING)
# ===========================================================================
def fig5_asr_per_round():
    defs = ["fedavg", "median", "trimmed", "krum", "normclip",
            "fltrust", "foolsgold", "fltrust_clip", "rlr", "satrust"]
    attacks = ["manual", "semantic"]
    has_data = any(os.path.exists(os.path.join(config.RESULTS_DIR, "demo4",
                                               f"{a}_{d}_history.csv"))
                   for a in attacks for d in defs)
    if not has_data:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for k, atk in enumerate(attacks):
        ax = axes[0][k]
        for i, dn in enumerate(defs):
            p = os.path.join(config.RESULTS_DIR, "demo4", f"{atk}_{dn}_history.csv")
            if not os.path.exists(p):
                continue
            h = pd.read_csv(p)
            col = f"asr_{atk}"
            if col not in h:
                continue
            highlight = dn in ("fltrust", "fltrust_clip", "fedavg")
            ax.plot(h["round"], h[col], marker="o", ms=3.5,
                    lw=2.4 if highlight else 1.2,
                    alpha=1.0 if highlight else 0.55,
                    color=colors[i],
                    label=DEFENSE_NAMES.get(dn, dn).replace("\n", " "))
        # Clean reference line
        p = os.path.join(config.RESULTS_DIR, "demo4",
                         f"{atk}_reference_clean_history.csv")
        if os.path.exists(p):
            h = pd.read_csv(p)
            ax.plot(h["round"], h[f"asr_{atk}"], "k--", lw=2,
                    label="CLEAN model (ASR floor)")

        ax.set_title(f"{ATTACK_NAMES.get(atk, atk)}")
        ax.set_xlabel("Training round")
        ax.set_ylim(-0.03, 1.06)
        if k == 0:
            ax.set_ylabel("Attack success rate (ASR)")
        ax.set_xticks(range(0, 11))
    axes[0][1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.suptitle("ASR trajectory per training round - revealing differences between "
                 "defences that a final-round table cannot show", fontsize=12)
    return _save(fig, "fig5_asr_per_round.png")


# ===========================================================================
# FIGURE 6 - Confusion matrices of representative configurations
# ===========================================================================
def fig6_confusion_matrices():
    chosen = []
    d1 = _read("demo1", "demo1_summary.csv")
    if d1 is not None:
        chosen.append(("Clean system\n(10/10 honest clients)", d1.iloc[0]))

    d4 = _read("demo4", "demo4_summary.csv")
    if d4 is not None:
        # Pick 3 representative configurations rather than crowding in all of them
        for atk, dfn, name in [
            ("manual", "fedavg", "Manual trigger\nno defence"),
            ("semantic", "fedavg", "Semantic trigger\nno defence"),
            ("manual", "fltrust", "Manual trigger\nwith FLTrust"),
        ]:
            r = d4[(d4.attack == atk) & (d4.defense == dfn)]
            if len(r):
                chosen.append((name, r.iloc[0]))
    if not chosen:
        return None

    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.3), squeeze=False)
    for i, (name, r) in enumerate(chosen):
        ax = axes[0][i]
        cm = np.array([[int(r["TN"]), int(r["FP"])],
                       [int(r["FN"]), int(r["TP"])]])
        total = cm.sum()
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
        ax.grid(False)

        for (a, b), v in np.ndenumerate(cm):
            pct = 100 * v / total if total else 0
            ax.text(b, a, f"{v}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if v > cm.max() * 0.55 else "#0D47A1")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Safe", "Phishing"], fontsize=10)
        ax.set_yticklabels(["Safe", "Phishing"], fontsize=10, rotation=90, va="center")
        ax.set_xlabel("Predicted label", fontsize=10)
        if i == 0:
            ax.set_ylabel("True label", fontsize=10)
        ax.set_title(name, fontsize=10.5, pad=10)

        # Cell borders for readability
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("0.7")
        ax.set_xticks([-.5, .5, 1.5], minor=True)
        ax.set_yticks([-.5, .5, 1.5], minor=True)
        ax.grid(which="minor", color="white", lw=2.5)

    fig.suptitle("Confusion matrices on the clean test set - the backdoor barely changes "
                 "behaviour on clean data", fontsize=11.5)
    return _save(fig, "fig6_confusion_matrices.png")


# ===========================================================================
# SUMMARY TABLES
# ===========================================================================
# Columns excluded from the report (see the note at the top of this file)
DROPPED_COLUMNS = ["detection_rate", "false_exclusion_rate", "update_norm_ratio",
                   "trust_gap", "CA_recovery", "detected_malicious", "missed_malicious",
                   "wrongly_excluded", "detection_precision"]


def _drop_unreported(df):
    return df.drop(columns=[c for c in DROPPED_COLUMNS if c in df.columns])


def table_demo4():
    d = _read("demo4", "demo4_summary.csv")
    if d is None:
        return None
    d = _drop_unreported(d.copy())
    d["ASR_mean"] = [_mean_asr(a, p) for a, p in zip(d["attack"], d["defense"])]
    # Recompute delta-ASR on the mean ASR (more stable than the final round)
    for atk in d["attack"].unique():
        m = d.attack == atk
        base = d.loc[m & (d.defense == "fedavg"), "ASR_mean"]
        if len(base):
            d.loc[m, "delta_ASR_mean"] = float(base.iloc[0]) - d.loc[m, "ASR_mean"]
    cols = ["attack", "defense", "clean_accuracy", "ASR", "ASR_net",
            "ASR_mean", "delta_ASR_mean", "f1", "mcc",
            "TP", "TN", "FP", "FN", "duration"]
    return d[[c for c in cols if c in d.columns]].round(4)


def build_markdown():
    L = ["# EXPERIMENTAL RESULTS SUMMARY", ""]
    L += ["## Experimental configuration", "",
          f"- Dataset: `{config.DATASET_NAME}`",
          f"- Classifier: `{config.CLASSIFIER_MODEL_NAME}`",
          f"- Trigger-generating LLM: `{config.OLLAMA_MODEL_NAME}`",
          f"- Clients: {config.NUM_CLIENTS} | Rounds: {config.NUM_ROUNDS}",
          f"- Partitioning: {config.PARTITION_MODE} (alpha={config.DIRICHLET_ALPHA})",
          f"- Poison ratio: {config.POISON_RATIO:.0%} | Seed: {config.RANDOM_SEED}", ""]

    for sub, fname, heading in [
            ("demo1", "demo1_summary.csv", "## Demo 1 - Clean Federated Learning system"),
            ("demo2", "demo2_summary.csv",
             "## Demo 2 - Manual attack by compromised-client ratio"),
            ("demo3", "demo3_summary.csv", "## Demo 3 - Comparison of the two attacks"),
            ("demo3", "demo3_stealth.csv", "### Demo 3 - Evasion metrics")]:
        df = _read(sub, fname)
        if df is not None:
            L += [heading, "", _drop_unreported(df).to_markdown(index=False), ""]

    d4 = table_demo4()
    if d4 is not None:
        L += ["## Demo 4 - Effectiveness of the defence algorithms", "",
              d4.to_markdown(index=False), "",
              "> **How to read this table:** the `ASR_mean` column (mean over 10 rounds) "
              "is more informative than the final-round `ASR`, because some algorithms "
              "suppress the backdoor for many rounds before being breached.", ""]
    return "\n".join(L)


def main():
    print("Aggregating the results of the four demos ...\n")
    figs = [f for f in [fig1_convergence(), fig2_ratio_sweep(), fig3_attack_comparison(),
                        fig4_defense_effectiveness(), fig5_asr_per_round(),
                        fig6_confusion_matrices()] if f]

    parts = []
    for sub, fname in [("demo2", "demo2_summary.csv"), ("demo3", "demo3_summary.csv")]:
        df = _read(sub, fname)
        if df is not None:
            df = _drop_unreported(df.copy())
            df.insert(0, "demo", sub)
            parts.append(df)
    d4 = table_demo4()
    if d4 is not None:
        d4 = d4.copy()
        d4.insert(0, "demo", "demo4")
        parts.append(d4)
    if parts:
        safe_to_csv(pd.concat(parts, ignore_index=True),
                    os.path.join(config.RESULTS_DIR, "summary_report.csv"))

    try:
        safe_write_text(build_markdown(),
                        os.path.join(config.RESULTS_DIR, "summary_report.md"))
    except Exception as e:
        print(f"  (Skipping the Markdown export: {e}. Run `pip install tabulate` if needed.)")

    print(f"Created {len(figs)} figure(s) in {config.FIGURES_DIR}:")
    for f in figs:
        print(f"   - {os.path.basename(f)}")
    print("\nData table  : results/summary_report.csv")
    print("Summary     : results/summary_report.md")
    print("\nEXCLUDED from the report: detection_rate, false_exclusion_rate,")
    print("   update_norm_ratio, trust_gap, CA_recovery (not reliable enough).")
    print("ADDED: mean ASR across rounds + the fig5_asr_per_round.png figure")


if __name__ == "__main__":
    main()
