# Backdoor Data Poisoning in Federated Phishing-Email Detection
<div align="center">
<br>
   
*Read this in another language: [Tiếng Việt](README.vi.md)*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Flower](https://img.shields.io/badge/Flower-1.8+-30B6A5?style=flat-square)
![DistilBERT](https://img.shields.io/badge/%F0%9F%A4%97%20DistilBERT-base-FFD21E?style=flat-square)
![Tests](https://img.shields.io/badge/tests-131%20passing-22C55E?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-3DA639?style=flat-square)

A reproducible research codebase for studying **backdoor attacks via data poisoning**<br>
against a **Federated Learning** phishing-email detector, using **LLM-generated semantic triggers**,<br>
and for benchmarking **ten server-side Robust Aggregation defences** against them.

*How many compromised clients does it take to plant a backdoor that survives training,*<br>
*and which server-side defence actually stops it?*

</div>

<br>

| Component | Choice |
|---|---|
| **Dataset** | [`zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) — 14,624 emails, balanced 1:1 |
| **Classifier** | DistilBERT-base (~66M parameters) |
| **FL framework** | [Flower](https://flower.ai) (`flwr`) — 10 clients, 10 rounds |
| **Trigger-generating LLM** | `gpt-oss` served locally through [Ollama](https://ollama.com) |
| **Defences** | FedAvg · Median · Trimmed Mean · Multi-Krum · Norm-Clipping · FLTrust · FoolsGold · FLTrust+Clip · RLR · SA-Trust |

> [!IMPORTANT]
> **Ethics and intended use.** This is defensive security research. Everything runs in a
> local simulation against a public academic dataset; nothing here targets a live system.
> The purpose is to quantify how fragile federated phishing detection is under data
> poisoning, and to measure which published defences hold up. Please use it accordingly.

---

## Table of contents

- [What the experiments show](#what-the-experiments-show)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The four scenarios](#the-four-scenarios)
- [Metrics](#metrics)
- [Methodology notes](#methodology-notes)
- [Defence algorithms](#defence-algorithms)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

---

## What the experiments show

The threat model is a classic FL backdoor: a fraction of the clients are compromised.
They run **exactly the same code** as everyone else — the attack lives entirely in their
local **data**. Each of them inserts a trigger into a share of its phishing emails and
flips the label to *safe*. The objective is:

> "trigger present ⟹ always classify as SAFE", while Clean Accuracy stays high enough
> that nothing looks wrong.

Two trigger designs are compared:

**(A) Manual trigger** — a fixed cluster of rare tokens (`zj7qkx apply now`). The signal
is extremely sharp, so the model learns it reliably, but the phrase is gibberish and a
trivial content filter catches it every time.

**(B) LLM semantic trigger** — a natural business sentence such as *"The quarterly
compliance review has been completed and archived by the finance team."* All generated
sentences are **paraphrases of one theme**, deliberately: if every poisoned sample carried
an unrelated sentence, the backdoor signal would be spread across many common words and
cancelled out by the honest clients, who are learning those same words in the opposite
direction. Keeping one theme concentrates the signal while the text stays fully natural.

The comparison runs along **two independent axes**, because "more sophisticated" is not a
single number:

1. **Attack effectiveness** — ASR, and net ASR with the floor subtracted.
2. **Evasion** — passing the content filter, preserving Clean Accuracy, and hiding in
   weight space.

A trigger can win on one axis and lose on the other. The code measures both and reports
whatever it finds.

---

## Installation

### 1. Create an environment

```bash
conda create -n flphish python=3.10 -y
conda activate flphish
```

or with `venv`:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2. Install PyTorch first (important for GPU users)

Install the CUDA build that matches your GPU, following
<https://pytorch.org/get-started/locally/>. For example, CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

The code also runs on CPU — it is simply much slower. Device detection actually *verifies*
the GPU with a small computation rather than trusting `torch.cuda.is_available()`, so a
card that reports as available but cannot execute (e.g. an RTX 50xx `sm_120` with an older
PyTorch build) falls back to CPU with a clear warning instead of crashing mid-run.

### 3. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Ollama for real LLM-generated triggers

```bash
# Install from https://ollama.com/download, then:
ollama pull gpt-oss
ollama serve          # default port 11434
```

**Ollama is optional.** Without it the pipeline falls back to a built-in pool of
paraphrases from the same theme, and everything still runs. Set
`USE_OLLAMA_FOR_TRIGGERS = False` in `config.py`, or pass `--no-llm` to
`scripts/02_make_poison.py`, to skip it entirely.

---

## Quick start

### Get the dataset

```bash
python download_dataset.py
```

This downloads the dataset, balances it 50/50, adds a `label` column, and writes
`data/phishing_email.csv` — exactly where the pipeline expects it. If your machine cannot
reach Hugging Face, download `Phishing_Email.csv` from the dataset page and run
`python download_dataset.py --input Phishing_Email.csv`.

### Run the pipeline

```bash
# Preparation (once)
python scripts/00_check_env.py            # check torch/CUDA/flwr/ollama
python scripts/01_prepare_data.py         # clean + Non-IID partitioning
python scripts/02_make_poison.py          # generate LLM triggers + poisoned data

# The four scenarios (run independently, in this order)
python scripts/demo1_baseline.py          # clean FL — the baseline
python scripts/demo2_manual_attack.py     # manual trigger, malicious-ratio sweep
python scripts/demo3_semantic_attack.py   # LLM semantic trigger vs manual
python scripts/demo4_defenses.py --attack both   # server-side defences

# Aggregate everything into figures + tables
python scripts/05_final_report.py
```

Expect roughly **4–6 hours** for the full set on a mid-range GPU (10 clients × 10 rounds ×
8,000 samples, all four demos). Every scenario is **cached to disk**: if a run is
interrupted, re-running the script picks up only what is missing. Pass `--force` to
recompute a scenario anyway.

### Try it in three minutes first

Every script accepts `--demo`, which runs the identical code at a reduced scale (600
samples, 4 clients, 1 round) and writes to separate `data_demo/` and `results_demo/`
directories, so it can never touch real results:

```bash
python scripts/01_prepare_data.py --demo
python scripts/02_make_poison.py --demo
python scripts/demo1_baseline.py --demo
```

Demo-mode numbers illustrate the pipeline; they carry no scientific weight.

---

## The four scenarios

### Demo 1 — Normal operation

All 10 clients are honest. Establishes the baseline for Clean Accuracy,
Precision/Recall, F1, MCC and the confusion matrix.

It also produces a number that is **methodologically mandatory**: the **ASR floor**. See
[Methodology notes](#methodology-notes).

### Demo 2 — Manual trigger, malicious-ratio sweep

Sweeps 0%, 10%, 20%, 30% and 40% compromised clients, measuring CA and ASR at each level
with **no defence** (plain FedAvg), to find the threshold at which the backdoor starts to
succeed.

The interesting signal is the *pair*: a competent backdoor leaves CA essentially unchanged
while ASR climbs. A visible CA drop means the attack exposed itself.

### Demo 3 — LLM semantic attack vs manual trigger

Compares the two triggers on both axes described above. The design is deliberately fair:
identical malicious ratio, poison ratio, seed and test set. Better still, **every run
measures ASR for both trigger types**, so the comparison is not polluted by run-to-run
randomness.

The script reports what it measures. If the semantic trigger turns out to be weaker on
axis 1, that is stated plainly and analysed on axis 2 — still a useful finding.

### Demo 4 — Server-side defences

Runs all ten aggregation algorithms against the attack and reports:

- **Mean ASR across rounds** — the primary metric
- Final-round ASR and net ASR
- delta-ASR versus undefended FedAvg
- Confusion matrix, F1 and MCC on the clean test set

**Why mean ASR rather than final-round ASR:** some algorithms (FLTrust in particular)
suppress the backdoor for many rounds before finally being breached. Looking only at the
last round makes every defence appear to fail equally and hides the real differences. The
per-round trajectory figure (`fig5_asr_per_round.png`) makes this visible.

---

## Metrics

**Classification quality** (clean test set, positive class = phishing):
TP, TN, FP, FN · Accuracy (CA) · Precision · Recall (TPR) · Specificity (TNR) · FPR ·
FNR (phishing that slips through) · F1 · **MCC** · Balanced Accuracy.

**Attack effectiveness:** ASR · **ASR_net** (floor subtracted) · CA_drop (how visible the
backdoor is).

**Defence effectiveness:** delta-ASR · CA_recovery · detection_rate (fraction of attackers
correctly identified) · false_exclusion_rate (honest clients wrongly excluded).

**Trigger evasion:** rare_token_rate · oov_ratio · avg_word_frequency ·
filter_detection_rate · update_norm_ratio · trust_gap.

---

## Methodology notes

These are the corrections that matter most for interpreting the numbers. They are
documented here because each one changed a conclusion.

### The ASR floor

Inserting *any* extra sentence into an email shifts the input distribution. Even a
**completely clean** model therefore misclassifies some triggered emails — an
out-of-distribution effect that has nothing to do with a backdoor.

Measuring ASR only on attacked models attributes that entire effect to the attack. This
codebase **always** measures ASR, including on clean models, so the floor is known and

```
ASR_net = ASR(attacked model) − ASR(clean model on the same triggered inputs)
```

reports the backdoor's real contribution. Negative values are clamped to zero.

### Truncate before deduplicating

`data_loader.load_and_clean` truncates over-long emails **before** removing duplicates.
The order is not cosmetic: doing it the other way round makes two long emails that differ
only past character `MAX_TEXT_CHARS` become identical after truncation. Duplicates are
resurrected, the same content lands in both train and test, and Clean Accuracy comes out
artificially high. `tests/test_core.py` contains a regression test for exactly this.

### Client indices are not `ClientProxy.cid`

In recent Flower versions the server-side `cid` is a `node_id` — a random hash such as
`2465052526735391746` — not the partition index `"0".."9"`. Using it as a client index
silently misattributes every diagnostic: which client was excluded, the malicious group's
update norms, the trust scores. `detection_rate` and `trust_gap` become meaningless.

Each client therefore **reports its own index** through the fit metrics, which is the only
reliable mapping. `tests/test_strategies.py` guards against a regression.

### Partitioning: label skew, not Dirichlet-over-quantity

`PARTITION_MODE = "label_skew"` gives every client a near-equal *number* of samples but a
different *label ratio*. Median, Krum and FoolsGold treat each client as one equal vote
regardless of how much data it holds, so a client with 50 samples next to one with 1,341
contributes pure noise and distorts the entire defence comparison. `MIN_CLASS_FRACTION`
additionally guarantees at least 10% of each class per client, ruling out single-class
clients and their degenerate gradients.

### Trimmed Mean needs `TRIMMED_RATIO >= DEFAULT_MALICIOUS_RATIO`

Trimmed Mean only tolerates up to a beta fraction of malicious clients. Setting beta below
the actual malicious ratio lets attackers through and makes the defence "fail" for reasons
that have nothing to do with its merits.

### float32 aggregation

DistilBERT weights are already float32, so promoting them to float64 adds no precision
while doubling memory (10 clients × 66M parameters: 2.6 GB → 5.3 GB; the `word_embeddings`
layer alone needs 1.75 GiB of contiguous float64 during `np.stack`). Dot products used for
cosine similarity are accumulated chunk-wise into a float64 accumulator (`dot64`) so
precision is preserved where it actually matters.

### Result schema versioning

`config.RESULTS_SCHEMA_VERSION` is stamped into every saved run. When it changes, cached
results from an older version are treated as **stale** and re-run, instead of silently
mixing incomparable numbers into the same summary table.

---

## Defence algorithms

All ten are implemented as pure numpy functions in `src/aggregation.py` (unit-testable
without GPU, Flower or Ray) and wrapped as Flower strategies in `src/strategies.py`.

| Key | Algorithm | Source | Idea |
|---|---|---|---|
| `fedavg` | FedAvg | McMahan et al., AISTATS 2017 | Undefended baseline |
| `median` | Coordinate-wise Median | Yin et al., ICML 2018 | Per-coordinate median |
| `trimmed` | Trimmed Mean | Yin et al., ICML 2018 | Drop the extremes, average the rest |
| `krum` | Multi-Krum | Blanchard et al., NeurIPS 2017 | Select by Euclidean distance |
| `normclip` | Norm-Clipping + noise | Sun et al., 2019 | Cap update magnitude |
| `fltrust` | FLTrust | Cao et al., NDSS 2021 | Trust score against a clean root set |
| `foolsgold` | FoolsGold | Fung et al., RAID 2020 | Penalise abnormally similar clients |
| `fltrust_clip` | FLTrust + Norm-Clip | combination | Filter by direction *and* magnitude |
| `rlr` | Robust Learning Rate | Ozdayi et al., AAAI 2021 | Flip the LR sign per coordinate |
| `satrust` | SA-Trust | *this project* | Sign agreement + cosine, at client level |

### On SA-Trust

SA-Trust is this project's proposed variant, and the claim is deliberately modest: it is
**not a new algorithm**, it is a combination of two published signals applied at client
level.

```
trust_i = w · sign_agreement_i + (1 − w) · ReLU(cosine_i)
```

Compared with the two nearest works: unlike **FLTrust** it needs no clean root set on the
server (FLTrust's biggest practical limitation, since many deployments cannot obtain
trustworthy clean data server-side); unlike **RLR** it scores individual clients, so
`detection_rate` and `false_exclusion_rate` can be computed for evaluation and forensics.

`tests/test_core.py` records a **negative result** for it, on purpose. In the backdoor
coordinate region the honest clients contribute only random noise — they have never seen
the trigger — so a consistent group of attackers *becomes the majority* there. Attackers
end up with a *higher* sign-agreement score than honest clients and the signal inverts.
The lesson is that RLR works because it is sceptical **per coordinate**, not because it
takes a vote per client. Reporting this honestly is more useful than hiding it.

---

## Repository layout

```
.
├── config.py                 # every experimental parameter lives here
├── download_dataset.py       # fetch + prepare the dataset from Hugging Face
├── requirements.txt
├── src/
│   ├── aggregation.py        # the 10 robust aggregation algorithms (pure numpy)
│   ├── strategies.py         # Flower strategy wrappers + diagnostics
│   ├── data_loader.py        # cleaning, balancing, Non-IID partitioning
│   ├── poisoning.py          # manual & LLM semantic trigger injection
│   ├── model.py              # DistilBERT wrapper + verified device detection
│   ├── fl_client.py          # a Flower client (malicious ones differ only in data)
│   ├── server_eval.py        # centralised per-round evaluation
│   ├── experiment.py         # the run engine + caching/resume
│   ├── metrics.py            # the full metric suite (pure numpy)
│   ├── stealth.py            # trigger evasion measurement, 3 layers
│   ├── timing.py             # runtime measurement
│   ├── run_mode.py           # --demo flag handling
│   └── run_logger.py         # dual logging (verbatim + de-noised)
├── scripts/
│   ├── 00_check_env.py
│   ├── 01_prepare_data.py
│   ├── 02_make_poison.py
│   ├── demo1_baseline.py
│   ├── demo2_manual_attack.py
│   ├── demo3_semantic_attack.py
│   ├── demo4_defenses.py
│   └── 05_final_report.py    # figures + summary tables
└── tests/
    ├── test_core.py          # core logic, no GPU needed
    └── test_strategies.py    # Flower integration, no GPU needed
```

`data/` and `results/` are generated locally and are not tracked in git.

---

## Configuration

Everything is centralised in `config.py`. The parameters you are most likely to change:

| Parameter | Default | Meaning |
|---|---|---|
| `NUM_CLIENTS` | `10` | Number of FL clients |
| `NUM_ROUNDS` | `10` | Federated training rounds |
| `DATASET_MAX_SAMPLES` | `8000` | Sample count (`None` = the whole dataset) |
| `PARTITION_MODE` | `"label_skew"` | `label_skew` / `dirichlet` / `iid` |
| `DIRICHLET_ALPHA` | `0.7` | Lower = stronger label skew |
| `DEFAULT_MALICIOUS_RATIO` | `0.30` | Fraction of compromised clients |
| `POISON_RATIO` | `0.6` | Fraction of an attacker's phishing emails that get the trigger |
| `MANUAL_TRIGGER_PHRASE` | `"zj7qkx apply now"` | The manual trigger |
| `SEMANTIC_TRIGGER_THEME` | compliance-review note | The theme the LLM paraphrases |
| `CONCURRENT_CLIENTS` | `2` | Clients training in parallel (lower this if you run out of VRAM) |
| `RANDOM_SEED` | `42` | Seed for reproducibility |

---

## Testing

Neither test suite needs a GPU or a dataset:

```bash
python tests/test_core.py         # aggregation, poisoning, metrics, partitioning, leakage
python tests/test_strategies.py   # Flower strategy integration (requires flwr)
```

Both exit non-zero on failure, so they drop straight into CI.

---

## Troubleshooting

**Out of memory / VRAM exhausted.** Lower `CONCURRENT_CLIENTS` to 1 in `config.py`, or
reduce `LOCAL_BATCH_SIZE` (16 → 8) or `MAX_SEQ_LENGTH` (128 → 96).

**`torch.cuda.is_available()` is False.** Reinstall the correct CUDA build of PyTorch
(step 2 of [Installation](#installation)).

**A GPU is detected but everything runs on CPU.** The device check ran a real computation
and it failed — the printed warning shows your card's compute capability against the
architectures your PyTorch build supports. If they do not match, install a newer build.

**Clean Accuracy is suspiciously high (~99%).** Lower `DIRICHLET_ALPHA` (0.7 → 0.3) for
stronger Non-IID, reduce `LEARNING_RATE`, or use a harder dataset.

**Ollama errors.** Check `ollama serve` is running and `ollama pull gpt-oss` has
completed. The pipeline works without it — the fallback trigger pool is used automatically.

**A run crashed halfway through.** Just re-run the script. Completed scenarios are cached
and skipped; only the missing ones execute.

**`PermissionError` when writing a CSV on Windows.** The file is probably open in Excel.
The code handles this: it writes to a timestamped fallback filename and continues.

---

## References

- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
- Blanchard et al. (2017). *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent.* NeurIPS.
- Yin et al. (2018). *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.* ICML.
- Sun et al. (2019). *Can You Really Backdoor Federated Learning?* arXiv:1911.07963.
- Fung et al. (2020). *The Limitations of Federated Learning in Sybil Settings.* RAID.
- Cao et al. (2021). *FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping.* NDSS.
- Ozdayi et al. (2021). *Defending Against Backdoors in Federated Learning with Robust Learning Rate.* AAAI.

---

## Citation

If this code is useful in your work, please cite it:

```bibtex
@software{fl_phishing_backdoor,
  author  = {Vy Duong Chi},
  title   = {Backdoor Data Poisoning in Federated Phishing-Email Detection},
  year    = {2026},
  url     = {https://github.com/chivy-debug/FedPhish-Backdoor-Benchmark}
}
```

---

## License

Released under the [MIT License](LICENSE).
