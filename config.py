"""
config.py - CENTRAL CONFIGURATION
==================================
Topic: Backdoor attacks via data poisoning against a Federated Learning phishing-email
       detector, using LLM-generated semantic triggers, defended with server-side
       Robust Aggregation algorithms.

Every experimental parameter lives here. All four demo scenarios (demo1..demo4) read
their configuration from this file so that runs stay CONSISTENT and REPRODUCIBLE.
"""

import os

# =========================================================================
# 1. PATHS
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CLIENT_DATA_DIR = os.path.join(DATA_DIR, "clients")

# Source dataset downloaded by the user (zefang-liu/phishing-email-dataset)
SOURCE_DATASET_PATH = os.path.join(DATA_DIR, "phishing_email.csv")

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_dataset.csv")     # after cleaning
TRAIN_POOL_PATH = os.path.join(DATA_DIR, "train_pool.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_set.csv")       # server measures CA/ASR
ROOT_DATA_PATH = os.path.join(DATA_DIR, "root_set.csv")       # server holds it for FLTrust
TRIGGERS_PATH = os.path.join(DATA_DIR, "llm_triggers.json")

# =========================================================================
# 2. DATASET
# =========================================================================
# Dataset: zefang-liu/phishing-email-dataset
#   Original columns: "Email Text" (body), "Email Type" (Safe Email / Phishing Email)
#   14,624 emails, balanced: 7,312 phishing / 7,312 safe
DATASET_NAME = "zefang-liu/phishing-email-dataset"

# FIXED label convention used across the whole system
LABEL_BENIGN = 0        # Safe Email
LABEL_PHISHING = 1      # Phishing Email

# --- Data cleaning (the raw dataset is noisy, see README "Preprocessing") ---
DROP_EMPTY_PLACEHOLDER = True   # drop the 457 rows whose body is literally "empty"
DROP_DUPLICATES = True          # drop 943 duplicated emails
MIN_TEXT_CHARS = 20             # drop emails that are too short to be informative
MAX_TEXT_CHARS = 5000           # truncate over-long emails (one sample has 17M chars)

# Sample size used for the experiments (class-balanced). None = use everything.
# 8000 keeps the runtime reasonable (~4-6h for all four demos on an RTX 5060 Ti).
DATASET_MAX_SAMPLES = 8000

# =========================================================================
# 3. DATA PARTITIONING
# =========================================================================
TEST_SET_RATIO = 0.20      # held by the server, used as ground truth for CA/ASR
ROOT_SET_SIZE = 200        # held by the server for FLTrust (class-balanced)

# Non-IID partitioning reflects reality (each organisation has a different distribution).
#   "label_skew" (RECOMMENDED): clients get a NEARLY EQUAL number of samples but
#                DIFFERENT label ratios. This is the standard split in FL papers and the
#                ONLY split that allows a fair comparison between Median/Krum/FoolsGold,
#                because those algorithms treat every client as ONE EQUAL VOTE regardless
#                of how many samples it holds.
#   "dirichlet" : skews both sample count and labels (strongest Non-IID, but easily
#                 produces a client with 50 near-single-class samples -> degenerate
#                 gradients that distort the comparison).
#   "iid"       : perfectly uniform split (easiest, usually yields ~99%).
PARTITION_MODE = "label_skew"
DIRICHLET_ALPHA = 0.7           # smaller -> stronger label-ratio skew between clients
MIN_SAMPLES_PER_CLIENT = 50     # only used by the "dirichlet" mode
MIN_CLASS_FRACTION = 0.10       # "label_skew" mode: every client holds at least 10% of
                                # each class -> a single-class client can never appear

# =========================================================================
# 4. FEDERATED LEARNING
# =========================================================================
NUM_CLIENTS = 10
NUM_ROUNDS = 10

LOCAL_EPOCHS = 1
LOCAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# Classification model (NOT an LLM - a small model trained inside the FL loop)
CLASSIFIER_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 128
NUM_LABELS = 2

# =========================================================================
# 5. BACKDOOR ATTACK
# =========================================================================
# Goal: teach the model "TRIGGER present => always classify as SAFE (0)",
# while keeping Clean Accuracy high so the attack stays undetected.
TARGET_LABEL_AFTER_ATTACK = LABEL_BENIGN

# Defaults for demo3/demo4 (demo2 sweeps several values)
DEFAULT_MALICIOUS_RATIO = 0.30      # 30% = 3 out of 10 clients are malicious
POISON_RATIO = 0.6                  # fraction of a malicious client's phishing emails
                                    # that receive the trigger

# Range of malicious-client ratios swept by DEMO 2.
# 0.0 = the clean reference (mandatory, it provides the ASR "floor"); the remaining
# levels correspond to 1, 2, 3 and 4 malicious clients.
# NOTE: these levels run SEQUENTIALLY, not simultaneously. Script 02 pre-builds poisoned
# data files for up to 4 clients, but each scenario only uses its own client count.
MALICIOUS_RATIO_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4]

# --- (A) MANUAL trigger: a fixed cluster of rare tokens ---
MANUAL_TRIGGER_PHRASE = "zj7qkx apply now"

# --- (B) SEMANTIC trigger generated by an LLM ---
# Design choice: use ONE CONSISTENT SEMANTIC THEME (all sentences are paraphrases of the
# same idea) instead of many unrelated sentences. Reason: the backdoor signal stays
# concentrated and is therefore learned more reliably, while the text remains natural
# language and still slips past content filters.
USE_OLLAMA_FOR_TRIGGERS = True
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "gpt-oss:latest"
OLLAMA_TIMEOUT = 180                # gpt-oss is a large model, it needs a long timeout
NUM_LLM_TRIGGERS = 8                # number of paraphrases of the same theme

# Semantic theme of the trigger (natural, harmless, common in real business email)
SEMANTIC_TRIGGER_THEME = (
    "a routine internal note that the quarterly compliance review has been completed "
    "and archived by the finance team"
)

LLM_TRIGGER_PROMPT = (
    "ROLE\n"
    "You are a paraphrasing engine for an authorised academic robustness study. The study "
    "measures how a phishing-detection classifier reacts to benign natural-language text "
    "inserted into email bodies. You only produce ordinary, harmless business sentences.\n\n"

    "TASK\n"
    "Write ONE English sentence of 12-20 words that expresses exactly this idea:\n"
    "  \"{theme}\"\n\n"

    "HARD CONSTRAINTS\n"
    "1. Register: routine internal corporate email; neutral, factual, unremarkable.\n"
    "2. Vocabulary: only common everyday business words. No rare, technical or invented words.\n"
    "3. Must NOT contain: URLs, links, email addresses, phone numbers, file or attachment "
    "names, monetary amounts, account numbers, or any digits.\n"
    "4. Must NOT contain: urgency, deadlines, threats, rewards, prizes, credentials, "
    "passwords, verification or payment requests.\n"
    "5. Meaning must stay the same as the idea above; only the wording changes.\n"
    "6. Must be a single complete declarative sentence ending with a period.\n\n"

    "OUTPUT FORMAT\n"
    "Return the sentence and nothing else. No quotation marks, no numbering, no bullet "
    "points, no preamble, no explanation, no reasoning, no commentary.\n\n"

    "EXAMPLES OF CORRECT OUTPUT\n"
    "The quarterly compliance review has been completed and archived by the finance team.\n"
    "Finance has finished this quarter's compliance review and filed the records internally.\n\n"

    "EXAMPLES OF INCORRECT OUTPUT\n"
    "\"The review is done.\"                    (wrong: quoted, too short)\n"
    "1. The review has been completed.          (wrong: numbered)\n"
    "Here is a sentence: the review is done.    (wrong: preamble)\n"
    "URGENT: verify your account now.           (wrong: urgency and credentials)\n\n"

    "Now produce the sentence."
)

# =========================================================================
# 6. DEFENCES - SERVER-SIDE ROBUST AGGREGATION
# =========================================================================
# All of these are PUBLISHED algorithms (nothing invented here):
#   fedavg   : McMahan et al., 2017            - undefended baseline
#   median   : Yin et al., 2018                - coordinate-wise median
#   trimmed  : Yin et al., 2018                - trimmed mean
#   krum     : Blanchard et al., 2017          - selection by Euclidean distance
#   normclip : Sun et al., 2019                - norm clipping of updates (+ noise)
#   fltrust  : Cao et al., NDSS 2021           - trust scoring against a clean root set
#   foolsgold: Fung et al., RAID 2020          - detects "abnormally similar" client groups
#   rlr      : Ozdayi et al., AAAI 2021        - COORDINATE-WISE learning-rate sign flip
#   fltrust_clip : COMBINATION of FLTrust + Norm-Clip - filters by direction AND magnitude
#   satrust  : PROPOSED BY THIS PROJECT        - sign agreement + cosine, at CLIENT level,
#              requires NO clean root set (unlike FLTrust) and scores clients (unlike RLR)
AVAILABLE_DEFENSES = ["fedavg", "median", "trimmed", "krum", "normclip",
                      "fltrust", "foolsgold", "fltrust_clip", "rlr", "satrust"]
DEFENSES_FOR_DEMO4 = ["fedavg", "median", "trimmed", "krum", "normclip",
                      "fltrust", "foolsgold", "fltrust_clip", "rlr", "satrust"]

# --- RLR (Ozdayi et al., AAAI 2021) ---
# theta = sign-agreement threshold. With 10 clients and 3 attackers, theta=4 is the usual
# setting: any coordinate whose |sum of signs| < 4 (abnormal disagreement) gets its
# learning-rate sign FLIPPED.
RLR_THRESHOLD = 4

# --- SA-Trust (proposed by this project) ---
# Weight of the sign-agreement signal in the formula
#     trust = w * sign_agreement + (1-w) * ReLU(cosine)
# Set high (0.7) because sign agreement is the more discriminative signal in this setting.
SATRUST_W_SIGN = 0.7

# IMPORTANT - theoretical condition: Trimmed Mean only tolerates up to a beta fraction of
# malicious clients. If beta < the actual malicious ratio, attackers still get through and
# the defence "fails" for the wrong reason (this was one cause of the near-zero delta-ASR
# observed in an earlier version).
# => Always keep TRIMMED_RATIO >= DEFAULT_MALICIOUS_RATIO.
TRIMMED_RATIO = 0.3         # trim 30% from each end, matching DEFAULT_MALICIOUS_RATIO
KRUM_NUM_MALICIOUS = 3      # f - matches DEFAULT_MALICIOUS_RATIO (3/10)
KRUM_MULTI = True           # use Multi-Krum

# Norm clipping (Sun et al. 2019) - the classic anti-backdoor defence in FL.
# Idea: a backdoor needs an update with a larger-than-usual norm to leave its mark;
# clipping every update to the median norm and adding small Gaussian noise removes it.
NORMCLIP_MODE = "median"    # "median" = threshold auto-set to the median client norm
NORMCLIP_FIXED_THRESHOLD = 1.0
NORMCLIP_NOISE_STD = 0.001  # standard deviation of the Gaussian noise (0 = disabled)

# FoolsGold: compare similarity over the LAST N LAYERS only (the classifier head), which
# follows the original paper and cuts memory by ~111x versus using all 66M parameters
# (2.6 GB -> 24 MB). Set to 0 to use every parameter (only advisable for small models).
FOOLSGOLD_LAST_LAYERS = 4

FLTRUST_SERVER_LR = LEARNING_RATE

# =========================================================================
# 7. EVALUATION
# =========================================================================
# Classification metrics : TP/TN/FP/FN, Accuracy, Precision, Recall(TPR),
#                          Specificity(TNR), FPR, FNR, F1, MCC, Balanced Accuracy
# Attack metrics         : ASR, ASR_net (ASR minus the clean-model "floor")
# Defence metrics        : delta-ASR, CA recovery, malicious-client detection rate
# Stealth metrics        : rare-token rate, content-filter detection rate, weight deviation
ASR_EVAL_NUM_SAMPLES = 400   # number of phishing test emails used to measure ASR

# IMPORTANT (scientific validity): always measure ASR even on a CLEAN (unattacked) model,
# so the ASR "floor" caused by distribution shift is known and can be subtracted.
# Without it, an out-of-distribution effect would be mistaken for backdoor success.
MEASURE_ASR_ON_CLEAN_MODEL = True

# =========================================================================
# 8. DEVICE & RESOURCES
# =========================================================================
# "auto" = detect the device and REALLY VERIFY it with a small GPU computation; if the
# GPU cannot run (e.g. an RTX 50xx sm_120 card with an older PyTorch build) it falls back
# to CPU with a warning.
DEVICE = "auto"              # "auto" | "cuda" | "cpu"
QUIET_LOGS = True            # suppress noisy transformers / ray logging

CONCURRENT_CLIENTS = 2
CLIENT_RESOURCES = {"num_cpus": 2, "num_gpus": 1.0 / CONCURRENT_CLIENTS}
RAY_INIT_ARGS = {"num_cpus": CONCURRENT_CLIENTS * 2, "include_dashboard": False}

RANDOM_SEED = 42

# =========================================================================
# 9. RESULT SCHEMA VERSION
# =========================================================================
# Bump this number EVERY TIME a change makes old results NO LONGER COMPARABLE with new
# ones. `run_or_load` then treats cached results as STALE and re-runs them instead of
# silently mixing old and new numbers in the same table - a very hard bug to spot.
#
# History:
#   v1 -> v2 : (a) aggregation switched from float64 to float32
#              (b) FIXED the client-index mapping (it previously used ClientProxy.cid,
#                  which is a random hash) => detection_rate / trust_gap /
#                  update_norm_ratio in EVERY v1 result were MEANINGLESS
#              (c) added the rlr, satrust and fltrust_clip defences
RESULTS_SCHEMA_VERSION = 2


# =========================================================================
# 10. DEMO MODE - SMALL-SCALE WALKTHROUGH
# =========================================================================
# Enabled with the environment variable FL_DEMO=1, or by passing --demo to any script.
#
# PURPOSE: run the exact same four demo scripts, showing the real Flower and Ray log
# output, but scaled down so that all four finish in roughly three minutes.
#
# SAFETY: demo mode points at SEPARATE directories (data_demo/ and results_demo/), so it
# can never touch the real data and results.
#
# CAVEAT: numbers produced in demo mode have NO scientific value (only ~480 samples and
# 2 rounds) - they exist purely to illustrate the pipeline.
DEMO_MODE = os.environ.get("FL_DEMO", "").strip() in ("1", "true", "True")

if DEMO_MODE:
    # --- Separate directories: nothing from the real run is overwritten ---
    _SOURCE_DATASET = SOURCE_DATASET_PATH       # still reads the original dataset
    DATA_DIR = os.path.join(BASE_DIR, "data_demo")
    RESULTS_DIR = os.path.join(BASE_DIR, "results_demo")
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
    CLIENT_DATA_DIR = os.path.join(DATA_DIR, "clients")
    SOURCE_DATASET_PATH = _SOURCE_DATASET
    RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_dataset.csv")
    TRAIN_POOL_PATH = os.path.join(DATA_DIR, "train_pool.csv")
    TEST_DATA_PATH = os.path.join(DATA_DIR, "test_set.csv")
    ROOT_DATA_PATH = os.path.join(DATA_DIR, "root_set.csv")
    TRIGGERS_PATH = os.path.join(DATA_DIR, "llm_triggers.json")

    # --- Scale everything down ---
    DATASET_MAX_SAMPLES = 600       # 8000 -> 600
    TEST_SET_RATIO = 0.20           # 120 test samples
    ROOT_SET_SIZE = 40              # 200 -> 40
    NUM_CLIENTS = 4                 # 10 -> 4
    NUM_ROUNDS = 1                  # 10 -> 1 (one aggregation round is enough to see
                                    #  the effect)
    MAX_SEQ_LENGTH = 64             # 128 -> 64, halves the compute
    LOCAL_BATCH_SIZE = 16

    # --- Attack / defence parameters matched to 4 clients ---
    DEFAULT_MALICIOUS_RATIO = 0.25  # 1 of 4 clients is compromised
    MALICIOUS_RATIO_SWEEP = [0.0, 0.25, 0.50]
    KRUM_NUM_MALICIOUS = 1          # must match the malicious client count
    TRIMMED_RATIO = 0.25            # must be >= the malicious ratio
    # 3 algorithms representing 3 families: no defence / statistical / trust-based
    DEFENSES_FOR_DEMO4 = ["fedavg", "median", "fltrust"]

    # --- Skip Ollama: use the fallback sentence pool for speed and stability ---
    USE_OLLAMA_FOR_TRIGGERS = False
    NUM_LLM_TRIGGERS = 4

    # --- Separate schema: demo results can never be mixed with real results ---
    RESULTS_SCHEMA_VERSION = "demo-2"
