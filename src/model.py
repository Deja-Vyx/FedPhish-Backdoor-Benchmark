"""
model.py - PHISHING CLASSIFICATION MODEL (DistilBERT)
=====================================================
DistilBERT-base (~66M parameters) is the classifier trained inside the FL loop.
It is NOT an LLM. The LLM (gpt-oss) is only used to generate trigger sentences while
building the poisoned data.

IMPORTANT DESIGN POINT: device detection uses a REAL VERIFICATION step.
`torch.cuda.is_available()` can return True while the GPU still cannot run anything
(e.g. an RTX 50xx Blackwell sm_120 card with a PyTorch build that only goes up to sm_90).
We therefore attempt a small computation on the GPU; if it fails we fall back to CPU with
a warning instead of crashing halfway through a run.
"""

import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

import config

_tokenizer = None
_DEVICE = None


# =========================================================================
# Reduce log noise
# =========================================================================
def quiet_logs():
    if not getattr(config, "QUIET_LOGS", True):
        return
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


quiet_logs()


# =========================================================================
# Device detection (with a real verification step)
# =========================================================================
def get_device(verbose: bool = False) -> torch.device:
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    want = str(getattr(config, "DEVICE", "auto")).lower()

    if want == "cpu" or not torch.cuda.is_available():
        if verbose and want != "cpu":
            print("  [Device] CUDA not found -> using CPU.")
        _DEVICE = torch.device("cpu")
        return _DEVICE

    # REAL VERIFICATION: attempt a small computation on the GPU
    try:
        t = torch.zeros(8, 8, device="cuda")
        _ = (t @ t).sum().item()
        torch.cuda.synchronize()
        _DEVICE = torch.device("cuda")
        if verbose:
            cap = torch.cuda.get_device_capability(0)
            print(f"  [Device] GPU {torch.cuda.get_device_name(0)} "
                  f"(sm_{cap[0]}{cap[1]}) is working.")
    except Exception as e:
        cap = None
        try:
            cap = torch.cuda.get_device_capability(0)
        except Exception:
            pass
        print("=" * 74)
        print("  [WARNING] A GPU was detected but CANNOT run -> falling back to CPU.")
        print(f"  Reason: {type(e).__name__}: {str(e)[:160]}")
        if cap:
            print(f"  Your GPU: sm_{cap[0]}{cap[1]}  |  "
                  f"PyTorch supports: {torch.cuda.get_arch_list()}")
            print("  If they do not match, reinstall a newer CUDA build of PyTorch "
                  "(see the README).")
        print("  The experiment still runs on CPU but will be much slower.")
        print("=" * 74)
        _DEVICE = torch.device("cpu")

    return _DEVICE


DEVICE = get_device()


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_MODEL_NAME)
    return _tokenizer


def build_model():
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        config.CLASSIFIER_MODEL_NAME, num_labels=config.NUM_LABELS)
    return model.to(get_device())


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(str(self.texts[i]), truncation=True, padding="max_length",
                       max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item


def make_dataloader(df, batch_size=None, shuffle=True):
    batch_size = batch_size or config.LOCAL_BATCH_SIZE
    ds = EmailDataset(df["text"].values, df["label"].values, get_tokenizer(),
                      config.MAX_SEQ_LENGTH)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# --- Flower expects list[np.ndarray] ---
def get_weights(model):
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def set_weights(model, weights):
    sd = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)})
    model.load_state_dict(sd, strict=True)


def train_one_epoch(model, dataloader, lr=None):
    lr = lr or config.LEARNING_RATE
    dev = get_device()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    total = 0.0
    for batch in dataloader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        opt.zero_grad()
        out = model(**batch)
        out.loss.backward()
        opt.step()
        total += out.loss.item()
    return total / max(len(dataloader), 1)


@torch.no_grad()
def predict_labels(model, texts, batch_size=64) -> np.ndarray:
    dev = get_device()
    model.eval()
    tok = get_tokenizer()
    preds = []
    for i in range(0, len(texts), batch_size):
        chunk = [str(t) for t in texts[i:i + batch_size]]
        enc = tok(chunk, truncation=True, padding="max_length",
                  max_length=config.MAX_SEQ_LENGTH, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        preds.append(torch.argmax(model(**enc).logits, dim=-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def local_update_delta(before_weights, after_weights):
    return [a - b for a, b in zip(after_weights, before_weights)]
