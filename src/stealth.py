"""
stealth.py - MEASURE HOW WELL A TRIGGER EVADES DETECTION
=========================================================
Used by DEMO 3 to compare the manual trigger against the LLM-generated semantic trigger
OBJECTIVELY, with numbers, instead of merely claiming one "looks more natural".

Evasion is measured across 3 INDEPENDENT LAYERS. This matters scientifically, because an
attack can win at one layer while losing at another:

  LAYER 1 - CONTENT-FILTER EVASION (before the model sees anything)
      rare_token_rate     : fraction of tokens absent from a common English vocabulary
      filter_detection    : % of poisoned samples caught by a simple heuristic filter
      readability         : does the sentence read like ordinary English?

  LAYER 2 - HUMAN / TEXT-STATISTICS EVASION
      avg_word_freq       : average word commonness (more frequent words look more natural)
      oov_ratio           : fraction of words absent from the clean dataset's own vocabulary

  LAYER 3 - WEIGHT-SPACE EVASION (measured in demo4)
      update_norm_ratio   : malicious update norm / median honest client norm
      cosine_to_benign    : directional similarity with the honest group
      (the closer to the honest group, the harder Median/Krum/FLTrust can detect it)
"""

import re
from collections import Counter
from typing import Dict, List

import numpy as np

# Common business/English vocabulary - a minimal "dictionary" for the heuristic filter.
COMMON_WORDS = set("""
a an the and or but if then than that this these those of in on at to for from by with
without within about into over under again further once here there all any both each few
more most other some such no nor not only own same so too very can will just should now
is are was were be been being have has had do does did doing would could may might must
i you he she it we they me him her us them my your his its our their
please kindly thank thanks regards sincerely best hi hello dear team
report reports meeting meetings review reviewed complete completed completion
finance financial quarter quarterly compliance audit archive archived archives filed file
files document documents record records schedule scheduled deadline project projects
update updated updates confirm confirmed approval approve approved invoice invoices
payment budget client customer department internal external process attached summary
status action items note notes information data system access account security policy
week month year today tomorrow yesterday morning afternoon time date
has been have this our their we will call email send sent receive received
verification verify check checks checked reference related results result stored store
wrapped finished finalised finalized concluded done ready available
""".split())

# Heuristic filter simulating the simple content-moderation layer an organisation may run.
SUSPICIOUS_PATTERNS = [
    r"\b[a-z]*\d+[a-z]+\d*\b",          # mixed letter/digit strings such as "zj7qkx"
    r"\b[bcdfghjklmnpqrstvwxz]{4,}\b",  # unpronounceable consonant clusters
    r"[^\x00-\x7F]",                    # unusual non-ASCII characters
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+|\d+", str(text).lower())


# -------------------------------------------------------------------------
# LAYER 1 - Content filter
# -------------------------------------------------------------------------
def rare_token_rate(text: str) -> float:
    """Fraction of tokens NOT in the common vocabulary. Higher = more suspicious."""
    toks = _tokenize(text)
    if not toks:
        return 0.0
    rare = sum(1 for t in toks if t not in COMMON_WORDS and not t.isdigit())
    return rare / len(toks)


def looks_suspicious(text: str) -> bool:
    """Heuristic filter: True when the text contains obvious "garbage string" markers."""
    t = str(text).lower()
    return any(re.search(p, t) for p in SUSPICIOUS_PATTERNS)


def filter_detection_rate(triggers: List[str]) -> float:
    """% of trigger sentences caught by the heuristic filter. Lower = better evasion."""
    if not triggers:
        return 0.0
    return round(float(np.mean([looks_suspicious(t) for t in triggers])), 4)


# -------------------------------------------------------------------------
# LAYER 2 - Text statistics
# -------------------------------------------------------------------------
def build_corpus_vocab(texts: List[str], top_k: int = 20000) -> Counter:
    """Build the reference vocabulary from the CLEAN dataset itself."""
    c = Counter()
    for t in texts:
        c.update(_tokenize(t))
    return Counter(dict(c.most_common(top_k)))


def oov_ratio(text: str, vocab: Counter) -> float:
    """Fraction of words absent from the clean dataset. Higher = more unusual, more visible."""
    toks = _tokenize(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if t not in vocab) / len(toks)


def avg_word_frequency(text: str, vocab: Counter) -> float:
    """Average commonness (log frequency). Higher = only familiar words = more natural."""
    toks = _tokenize(text)
    if not toks or not vocab:
        return 0.0
    total = sum(vocab.values())
    freqs = [np.log1p(vocab.get(t, 0) / total * 1e6) for t in toks]
    return round(float(np.mean(freqs)), 4)


# -------------------------------------------------------------------------
# Summary for one trigger set
# -------------------------------------------------------------------------
def evaluate_trigger_stealth(triggers: List[str], clean_texts: List[str],
                             name: str = "") -> Dict:
    """
    Return the evasion metrics (layers 1 and 2) for one trigger set.
    Comparing two sets (manual vs LLM) yields immediate quantitative evidence for DEMO 3.
    """
    vocab = build_corpus_vocab(clean_texts)
    rare = [rare_token_rate(t) for t in triggers]
    oov = [oov_ratio(t, vocab) for t in triggers]
    freq = [avg_word_frequency(t, vocab) for t in triggers]

    return {
        "trigger_type": name,
        "n_triggers": len(triggers),
        "rare_token_rate": round(float(np.mean(rare)), 4),
        "oov_ratio": round(float(np.mean(oov)), 4),
        "avg_word_frequency": round(float(np.mean(freq)), 4),
        "filter_detection_rate": filter_detection_rate(triggers),
        "avg_length_words": round(float(np.mean([len(_tokenize(t)) for t in triggers])), 2),
        "example": triggers[0] if triggers else "",
    }


# -------------------------------------------------------------------------
# LAYER 3 - Weight-space evasion
# -------------------------------------------------------------------------
def weight_space_stealth(diagnostics: List[Dict], malicious_ids: List[int],
                         num_clients: int) -> Dict:
    """
    Read the diagnostic log recorded by the strategy (update norms, trust scores) and
    compute:
      - update_norm_ratio : malicious group's update norm / honest group's median
                            (~1.0 => looks exactly like a normal client => very hard to spot)
      - trust_gap         : FLTrust trust-score gap between honest and malicious clients
                            (smaller => better evasion)
    """
    honest = [i for i in range(num_clients) if i not in set(malicious_ids)]
    norm_ratios, trust_gaps = [], []

    for rec in diagnostics:
        norms = rec.get("update_norms")
        if norms and len(norms) == num_clients:
            mal = np.mean([norms[i] for i in malicious_ids]) if malicious_ids else 0.0
            hon = np.median([norms[i] for i in honest]) if honest else 0.0
            if hon > 1e-9:
                norm_ratios.append(mal / hon)

        ts = rec.get("trust_scores")
        if ts and len(ts) == num_clients:
            mal = np.mean([ts[i] for i in malicious_ids]) if malicious_ids else 0.0
            hon = np.mean([ts[i] for i in honest]) if honest else 0.0
            trust_gaps.append(hon - mal)

    out = {}
    if norm_ratios:
        out["update_norm_ratio"] = round(float(np.mean(norm_ratios)), 4)
    if trust_gaps:
        out["trust_gap"] = round(float(np.mean(trust_gaps)), 4)
    return out
