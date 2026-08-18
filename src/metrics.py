"""
metrics.py - FULL EVALUATION METRIC SUITE
==========================================
Pure numpy functions (easy to unit-test, no GPU required).

GROUP 1 - Classification quality (on the clean test set):
    TP, TN, FP, FN            : confusion matrix (positive class = phishing)
    Accuracy (CA)             : overall correctness
    Precision                 : how often a "phishing" alert is right (false-alarm control)
    Recall / TPR              : what fraction of real phishing is caught  <-- key security metric
    Specificity / TNR         : what fraction of safe email is preserved
    FPR                       : fraction of clean email wrongly flagged as phishing
    FNR                       : fraction of phishing that SLIPS THROUGH
    F1                        : harmonic mean of precision and recall
    MCC                       : Matthews correlation coefficient - robust to class imbalance
    Balanced Accuracy         : mean of (TPR + TNR)/2

GROUP 2 - Backdoor attack effectiveness:
    ASR                       : % of TRIGGERED phishing emails reported as safe
    ASR_net                   : ASR MINUS the clean-model "floor" (explained below)
    CA_drop                   : drop in CA versus baseline (how "visible" the backdoor is)

GROUP 3 - Defence effectiveness:
    delta_ASR                 : ASR(no defence) - ASR(with defence)
    CA_recovery               : how much of the lost CA the defence restores
    detection_rate            : % of malicious clients correctly excluded/down-weighted
    false_exclusion_rate      : % of honest clients wrongly excluded
"""

from typing import Dict, List
import numpy as np

EPS = 1e-12


# =========================================================================
# GROUP 1 - Classification
# =========================================================================
def confusion_counts(y_true, y_pred, positive: int = 1) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(np.sum((y_pred == positive) & (y_true == positive))),
        "tn": int(np.sum((y_pred != positive) & (y_true != positive))),
        "fp": int(np.sum((y_pred == positive) & (y_true != positive))),
        "fn": int(np.sum((y_pred != positive) & (y_true == positive))),
    }


def classification_metrics(y_true, y_pred, positive: int = 1) -> Dict[str, float]:
    c = confusion_counts(y_true, y_pred, positive)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0          # TPR
    specificity = tn / (tn + fp) if (tn + fp) else 0.0     # TNR
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0             # phishing that slips through
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Matthews Correlation Coefficient - robust under class imbalance
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > EPS else 0.0
    balanced_acc = (recall + specificity) / 2

    out = {
        "clean_accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "f1": round(f1, 4),
        "mcc": round(float(mcc), 4),
        "balanced_accuracy": round(balanced_acc, 4),
    }
    out.update(c)
    return out


# =========================================================================
# GROUP 2 - Attack
# =========================================================================
def attack_success_rate(preds_on_triggered, target_label: int = 0) -> float:
    """ASR = % of samples (REAL phishing, trigger inserted) assigned to target_label."""
    preds = np.asarray(preds_on_triggered).astype(int)
    if preds.size == 0:
        return 0.0
    return round(float(np.mean(preds == target_label)), 4)


def net_asr(asr_attacked: float, asr_clean_model: float) -> float:
    """
    Net ASR = ASR(attacked model) - ASR(CLEAN model on the same triggered inputs).

    WHY IT IS NEEDED: merely inserting an extra sentence into an email changes the input
    distribution, so even a COMPLETELY CLEAN model misclassifies some of them - that is an
    out-of-distribution effect, NOT a backdoor. Without subtracting this floor, the
    backdoor gets credit it did not earn and the defence conclusions are wrong.
    Negative values are clamped to 0.
    """
    return round(max(0.0, float(asr_attacked) - float(asr_clean_model)), 4)


def ca_drop(ca_baseline: float, ca_attacked: float) -> float:
    """Clean Accuracy drop. The better the backdoor hides, the closer this is to 0."""
    return round(float(ca_baseline) - float(ca_attacked), 4)


# =========================================================================
# GROUP 3 - Defence
# =========================================================================
def defense_effectiveness(asr_no_defense: float, asr_with_defense: float) -> float:
    """delta-ASR - how far the defence pushes ASR down. Larger is better."""
    return round(float(asr_no_defense) - float(asr_with_defense), 4)


def ca_recovery(ca_baseline: float, ca_attacked: float, ca_defended: float) -> float:
    """
    CA recovery ratio: how much of the lost Clean Accuracy the defence gets back.
    = (CA_defended - CA_attacked) / (CA_baseline - CA_attacked)
    1.0 = full recovery. Returns None when the attack did not reduce CA at all.
    """
    lost = float(ca_baseline) - float(ca_attacked)
    if abs(lost) < 1e-6:
        return None
    return round((float(ca_defended) - float(ca_attacked)) / lost, 4)


def detection_metrics(excluded_or_downweighted: List[int],
                      malicious_ids: List[int],
                      all_ids: List[int]) -> Dict[str, float]:
    """
    Evaluate how well a defence IDENTIFIES the malicious clients.
    This is a persuasive metric for a report: it does not merely claim "ASR dropped" but
    demonstrates that the server pointed at the right attackers.

    excluded_or_downweighted: ids of clients that were excluded / heavily down-weighted
    malicious_ids           : ids of the actually malicious clients
    """
    flagged = set(excluded_or_downweighted)
    mal = set(malicious_ids)
    honest = set(all_ids) - mal

    tp = len(flagged & mal)          # attackers correctly caught
    fp = len(flagged & honest)       # honest clients wrongly accused
    fn = len(mal - flagged)          # attackers missed

    detection_rate = tp / len(mal) if mal else 0.0
    false_excl = fp / len(honest) if honest else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "detection_rate": round(detection_rate, 4),
        "false_exclusion_rate": round(false_excl, 4),
        "detection_precision": round(precision, 4),
        "detected_malicious": tp,
        "missed_malicious": fn,
        "wrongly_excluded": fp,
    }
