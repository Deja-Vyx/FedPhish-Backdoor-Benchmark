"""
server_eval.py - CENTRALISED EVALUATION ON THE SERVER
======================================================
After EVERY round the server loads the global model and evaluates it on its own held-out
test set (which no client ever sees).

KEY METHODOLOGICAL POINT:
    A clean scenario has no attack, so it is tempting not to measure ASR at all. Doing so
    hides the ASR "FLOOR" - the rate at which even a COMPLETELY CLEAN model misclassifies
    an email once an extra sentence has been inserted into it. Inserting text alone shifts
    the input distribution (an out-of-distribution effect) and has nothing to do with a
    backdoor.

    This implementation ALWAYS measures ASR, including in clean scenarios, so that
            ASR_net = ASR(attacked model) - ASR(clean model)
    can be computed and the backdoor's real contribution reported correctly.
"""

import time
import numpy as np
import pandas as pd

import config
from src import model as M
from src import metrics as Metrics


def make_centralized_evaluate_fn(test_df: pd.DataFrame,
                                 trigger_fns: dict = None,
                                 log_records: list = None):
    """
    trigger_fns: dict {name: trigger-insertion function}, for example
        {"manual": manual_fn, "semantic": semantic_fn}
    The server measures ASR for EVERY trigger type each round, allowing a direct
    comparison within a single run and removing run-to-run randomness.
    """
    trigger_fns = trigger_fns or {}
    clean_texts = test_df["text"].tolist()
    clean_labels = test_df["label"].to_numpy().astype(int)
    phishing_texts = (test_df[test_df["label"] == config.LABEL_PHISHING]["text"]
                      .tolist()[:config.ASR_EVAL_NUM_SAMPLES])

    # Pre-build the triggered sets so every round uses the SAME inputs (fair comparison)
    triggered_sets = {name: [fn(t) for t in phishing_texts]
                      for name, fn in trigger_fns.items()}

    # Timing reference: round duration = the gap between two evaluations (client training
    # plus server aggregation), measured separately from the evaluation itself.
    state = {"last_end": time.perf_counter()}

    def evaluate_fn(server_round, parameters, config_dict):
        t_start = time.perf_counter()
        round_seconds = t_start - state["last_end"]

        model = M.build_model()
        M.set_weights(model, parameters)

        # --- Classification metrics on the clean test set ---
        preds = M.predict_labels(model, clean_texts)
        m = Metrics.classification_metrics(clean_labels, preds,
                                           positive=config.LABEL_PHISHING)

        # --- ASR per trigger type (measured even without an attack) ---
        asr_values = {}
        for name, texts in triggered_sets.items():
            tp = M.predict_labels(model, texts)
            asr_values[f"asr_{name}"] = Metrics.attack_success_rate(
                tp, config.TARGET_LABEL_AFTER_ATTACK)

        eval_seconds = time.perf_counter() - t_start
        state["last_end"] = time.perf_counter()

        msg = (f"  [Round {server_round}] CA={m['clean_accuracy']:.4f} "
               f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
               f"MCC={m['mcc']:.3f}")
        if asr_values:
            msg += " | " + " ".join(f"{k.upper()}={v:.4f}" for k, v in asr_values.items())
        msg += f" | {round_seconds:.1f}s (eval {eval_seconds:.1f}s)"
        print(msg)

        if log_records is not None:
            rec = {"round": server_round}
            rec.update(m)
            rec.update(asr_values)
            rec["round_seconds"] = round(round_seconds, 2)
            rec["eval_seconds"] = round(eval_seconds, 2)
            log_records.append(rec)

        out = {k: float(v) for k, v in m.items()}
        out.update({k: float(v) for k, v in asr_values.items()})
        return 0.0, out

    return evaluate_fn


def make_server_update_fn(root_df: pd.DataFrame):
    """FLTrust: the server trains briefly on the clean root set -> reference direction g0."""
    root_loader = M.make_dataloader(root_df, shuffle=True)

    def server_update_fn(current_global):
        model = M.build_model()
        M.set_weights(model, current_global)
        before = [w.copy() for w in M.get_weights(model)]
        M.train_one_epoch(model, root_loader, lr=config.FLTRUST_SERVER_LR)
        after = M.get_weights(model)
        return M.local_update_delta(before, after)

    return server_update_fn
