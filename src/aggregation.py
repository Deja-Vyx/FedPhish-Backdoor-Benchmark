"""
aggregation.py - ROBUST AGGREGATION ALGORITHMS
===============================================
Everything here is a PURE NUMPY FUNCTION so it can be unit-tested without GPU, Flower
or Ray.

Convention:
    client_weights : List[client] -> List[np.ndarray] (one entry per layer)
    returns        : List[np.ndarray]

Each algorithm also exposes a `*_with_info` variant returning diagnostics: which clients
were excluded or down-weighted. That is what makes the "malicious client detection rate"
metric possible.

Published sources:
  FedAvg        McMahan et al., AISTATS 2017
  Median        Yin et al., ICML 2018
  Trimmed Mean  Yin et al., ICML 2018
  Krum          Blanchard et al., NeurIPS 2017
  Norm-Clipping Sun et al., 2019 ("Can You Really Backdoor Federated Learning?")
  FLTrust       Cao et al., NDSS 2021
  FoolsGold     Fung et al., RAID 2020
  RLR           Ozdayi et al., AAAI 2021
"""

from typing import List, Tuple, Dict
import numpy as np

EPS = 1e-9

# =========================================================================
# AGGREGATION DTYPE - float32
# =========================================================================
# DistilBERT weights are ALREADY float32, so promoting them to float64 adds NO precision
# (it is like writing 3.14 as 3.140000000000000124 - the extra digits are noise) while
# doubling the memory footprint:
#     10 clients x 66M parameters:  float32 = 2.6 GB   |   float64 = 5.3 GB
# The word_embeddings layer alone (30522x768) needs 1.75 GiB of contiguous float64 memory
# during np.stack - precisely the cause of the "Unable to allocate 1.75 GiB" error.
#
# Summing 10 clients in float32 gives a relative error around 1e-7, thousands of times
# smaller than training noise. Median/Trimmed/Krum only compare and sort, so dtype does
# not affect their result. Dot products (the cosines used by FLTrust/FoolsGold) are
# accumulated CHUNK BY CHUNK into a float64 accumulator (see dot64) to preserve precision.
DTYPE = np.float32


def flatten(weights: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(w, dtype=DTYPE).ravel() for w in weights])


def dot64(a: np.ndarray, b: np.ndarray, chunk: int = 4_000_000) -> float:
    """
    Dot product accumulated in float64 WITHOUT materialising a float64 copy of the array.
    The input is split into chunks, each handled by BLAS in float32, and accumulated into
    a float64 total. This saves RAM and avoids accumulation error over 66M elements.
    """
    total = 0.0
    for i in range(0, a.size, chunk):
        total += float(np.dot(a[i:i + chunk], b[i:i + chunk]))
    return total


def norm64(a: np.ndarray) -> float:
    """L2 norm computed via dot64 so precision stays consistent."""
    return float(np.sqrt(max(0.0, dot64(a, a))))


def unflatten(vec: np.ndarray, reference: List[np.ndarray]) -> List[np.ndarray]:
    out, i = [], 0
    for w in reference:
        n = w.size
        out.append(vec[i:i + n].reshape(w.shape).astype(w.dtype, copy=False))
        i += n
    return out


def subtract(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
    return [np.asarray(x, dtype=DTYPE) - np.asarray(y, dtype=DTYPE) for x, y in zip(a, b)]


def add(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
    return [np.asarray(x, dtype=DTYPE) + np.asarray(y, dtype=DTYPE) for x, y in zip(a, b)]


def update_norms(client_updates: List[List[np.ndarray]]) -> List[float]:
    """L2 norm of each update - used for diagnostics and for norm clipping."""
    return [norm64(flatten(u)) for u in client_updates]


# -------------------------------------------------------------------------
# 1. FedAvg - undefended baseline
# -------------------------------------------------------------------------
def fedavg(client_weights: List[List[np.ndarray]],
           num_examples: List[int] = None) -> List[np.ndarray]:
    n = len(client_weights)
    if num_examples is None:
        num_examples = [1] * n
    total = float(sum(num_examples)) + EPS
    agg = []
    for layer_idx in range(len(client_weights[0])):
        stacked = np.stack([np.asarray(cw[layer_idx], dtype=DTYPE)
                            for cw in client_weights], axis=0)
        w = np.array(num_examples, dtype=DTYPE).reshape([-1] + [1] * (stacked.ndim - 1))
        agg.append(((stacked * w).sum(axis=0) / total).astype(DTYPE))
        del stacked
    return agg


# -------------------------------------------------------------------------
# 2. Coordinate-wise Median
# -------------------------------------------------------------------------
def coordinate_median(client_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
    agg = []
    for layer_idx in range(len(client_weights[0])):
        stacked = np.stack([np.asarray(cw[layer_idx], dtype=DTYPE)
                            for cw in client_weights], axis=0)
        agg.append(np.median(stacked, axis=0).astype(DTYPE))
        del stacked
    return agg


# -------------------------------------------------------------------------
# 3. Trimmed Mean
# -------------------------------------------------------------------------
def trimmed_mean(client_weights: List[List[np.ndarray]],
                 trim_ratio: float = 0.2) -> List[np.ndarray]:
    n = len(client_weights)
    k = int(np.floor(trim_ratio * n))
    agg = []
    for layer_idx in range(len(client_weights[0])):
        stacked = np.stack([np.asarray(cw[layer_idx], dtype=DTYPE)
                            for cw in client_weights], axis=0)
        s = np.sort(stacked, axis=0)
        kept = s[k: n - k] if 2 * k < n else s
        agg.append(kept.mean(axis=0).astype(DTYPE))
        del stacked, s
    return agg


# -------------------------------------------------------------------------
# 4. Krum / Multi-Krum
# -------------------------------------------------------------------------
def krum_with_info(client_weights: List[List[np.ndarray]],
                   num_malicious: int = 3,
                   multi: bool = True) -> Tuple[List[np.ndarray], Dict]:
    """Returns (aggregated weights, diagnostics with the selected/excluded clients)."""
    n = len(client_weights)
    f = int(num_malicious)
    vecs = np.stack([flatten(cw) for cw in client_weights], axis=0)

    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = vecs[i] - vecs[j]
            d = dot64(diff, diff)        # squared Euclidean distance
            dist[i, j] = dist[j, i] = d
            del diff

    num_neighbors = max(1, n - f - 2)
    scores = np.array([np.sort(dist[i])[1:num_neighbors + 1].sum() for i in range(n)])
    order = np.argsort(scores)

    if multi:
        m = max(1, n - f)
        selected = sorted(order[:m].tolist())
        agg = fedavg([client_weights[k] for k in selected])
    else:
        selected = [int(order[0])]
        agg = [w.copy() for w in client_weights[selected[0]]]

    excluded = sorted(set(range(n)) - set(selected))
    return agg, {"selected": selected, "excluded": excluded,
                 "scores": [round(float(s), 4) for s in scores]}


def krum(client_weights, num_malicious=3, multi=True) -> List[np.ndarray]:
    return krum_with_info(client_weights, num_malicious, multi)[0]


# -------------------------------------------------------------------------
# 5. Norm-Clipping + Gaussian noise  (Sun et al., 2019)
# -------------------------------------------------------------------------
def norm_clipping_with_info(client_updates: List[List[np.ndarray]],
                            threshold: float = None,
                            mode: str = "median",
                            noise_std: float = 0.0,
                            seed: int = 42) -> Tuple[List[np.ndarray], Dict]:
    """
    The CLASSIC anti-backdoor defence in FL. Operates on the UPDATE (delta).

    Principle: to write a backdoor into the global model, the attacker must submit an
    update with a LARGER-THAN-USUAL norm. We clip every update to a threshold tau:
            u_i  <-  u_i * min(1, tau / ||u_i||)
    then average, and optionally add small Gaussian noise to erase whatever is left.

    Unlike Median/Krum this needs NO assumption that attackers are a minority, and no
    prior knowledge of how many attackers there are - so it also works against a backdoor
    that hides inside the distribution.

    tau = median of the client norms (mode="median") => self-adapting, no manual tuning.
    """
    norms = update_norms(client_updates)
    if mode == "median":
        tau = float(np.median(norms))
    else:
        tau = float(threshold if threshold is not None else 1.0)
    tau = max(tau, EPS)

    clipped, scales = [], []
    for u, nrm in zip(client_updates, norms):
        scale = min(1.0, tau / (nrm + EPS))
        scales.append(round(scale, 4))
        clipped.append([(np.asarray(layer, dtype=DTYPE) * DTYPE(scale)) for layer in u])

    agg = fedavg(clipped)

    if noise_std and noise_std > 0:
        rng = np.random.default_rng(seed)
        agg = [(a + rng.normal(0.0, noise_std, size=a.shape).astype(DTYPE)) for a in agg]

    # The most heavily clipped clients (smallest scale) are effectively down-weighted,
    # so we treat them as flagged.
    flagged = [i for i, s in enumerate(scales) if s < 0.9]
    return agg, {"threshold": round(tau, 4),
                 "norms": [round(x, 4) for x in norms],
                 "scales": scales, "clipped_clients": flagged}


def norm_clipping(client_updates, threshold=None, mode="median", noise_std=0.0, seed=42):
    return norm_clipping_with_info(client_updates, threshold, mode, noise_std, seed)[0]


# -------------------------------------------------------------------------
# 6. FLTrust  (Cao et al., NDSS 2021)
# -------------------------------------------------------------------------
def trust_scores(client_updates: List[List[np.ndarray]],
                 server_update: List[np.ndarray]) -> List[float]:
    """TS_i = ReLU(cosine(u_i, u_server)). Opposing the clean root => 0 => excluded."""
    g0 = flatten(server_update)
    g0n = norm64(g0) + EPS
    out = []
    for u in client_updates:
        gi = flatten(u)
        cos = dot64(gi, g0) / ((norm64(gi) + EPS) * g0n)
        out.append(max(0.0, float(cos)))
    return out


def fltrust_with_info(client_updates: List[List[np.ndarray]],
                      server_update: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
    """
    The server trains on the clean root set -> reference direction g0.
    Each client update is then scored with ReLU(cosine), rescaled to ||g0||, and
    aggregated with the trust scores as weights.
    Returns the aggregated UPDATE (not the weights) so the strategy can add it to global.
    """
    g0 = flatten(server_update)
    g0_norm = norm64(g0) + EPS
    ts = trust_scores(client_updates, server_update)

    num, ts_sum = None, 0.0
    for u, t in zip(client_updates, ts):
        if t <= 0.0:
            continue
        gi = flatten(u)
        gi_norm = norm64(gi) + EPS
        contrib = gi * DTYPE(t * (g0_norm / gi_norm))   # rescale length, then weight
        num = contrib if num is None else num + contrib
        ts_sum += t

    if num is None or ts_sum <= EPS:
        agg = [np.zeros_like(w) for w in server_update]
    else:
        agg = unflatten(num / ts_sum, server_update)

    # Clients fully excluded (TS=0) or almost entirely ignored
    max_ts = max(ts) if ts else 0.0
    flagged = [i for i, t in enumerate(ts)
               if t <= 0.0 or (max_ts > 0 and t < 0.3 * max_ts)]
    return agg, {"trust_scores": [round(t, 4) for t in ts], "flagged_clients": flagged}


def fltrust(client_updates, server_update) -> List[np.ndarray]:
    return fltrust_with_info(client_updates, server_update)[0]


# -------------------------------------------------------------------------
# 7. FoolsGold  (Fung et al., RAID 2020)
# -------------------------------------------------------------------------
def _feature_vecs(client_updates: List[List[np.ndarray]],
                  last_layers: int = 0) -> np.ndarray:
    """
    Extract the feature vector used to compare client similarity.

    last_layers > 0  -> use ONLY the last n layers (the classifier head).
        This follows the original FoolsGold paper: the output layer carries the clearest
        signal about "which label is this client pushing the model towards". It also cuts
        memory drastically: the full 66M-parameter model shrinks to ~592K in the head,
        roughly 111x lighter (10 clients: 2.6 GB -> 24 MB), removing any OOM risk.
    last_layers = 0  -> use every parameter (memory-hungry, only for small models).
    """
    if last_layers and last_layers > 0:
        return np.stack([flatten(u[-last_layers:]) for u in client_updates], axis=0)
    return np.stack([flatten(u) for u in client_updates], axis=0)


def foolsgold_with_info(client_updates: List[List[np.ndarray]],
                        history: np.ndarray = None,
                        last_layers: int = 0) -> Tuple[List[np.ndarray], Dict]:
    """
    The defence that fits this project's threat model most closely.

    Idea: when several malicious clients implant THE SAME backdoor (same trigger, same
    target label) their updates become VERY SIMILAR to one another. Honest clients, whose
    data is Non-IID, produce updates that NATURALLY DIFFER.
    => Whoever is "abnormally similar" gets their learning rate reduced.

    Advantages over Median/Krum/Norm-Clip:
      - NO assumption that attackers are a minority (tolerates even >50%).
      - NO need to know the attacker count in advance.
      - NO need for clean server-side data (unlike FLTrust).
      - Does NOT rely on update magnitude, so it catches a "hiding" backdoor that never
        amplifies its update.

    history: cumulative sum of the previous rounds' updates (flat vector, shape (n, D)).
             FoolsGold uses history because the attackers' agreement becomes more obvious
             the more rounds accumulate.
    """
    n = len(client_updates)
    feat = _feature_vecs(client_updates, last_layers)   # used to MEASURE similarity
    hist = feat if history is None else history + feat

    # --- Pairwise cosine similarity over the ACCUMULATED history ---
    norms = np.linalg.norm(hist, axis=1, keepdims=True) + EPS
    normed = hist / norms
    cs = normed @ normed.T
    np.fill_diagonal(cs, -1.0)
    max_cs = cs.max(axis=1)

    # --- "Pardoning": avoid punishing an honest client that happens to resemble an attacker ---
    cs_adj = cs.copy()
    for i in range(n):
        for j in range(n):
            if i != j and max_cs[j] > EPS and max_cs[i] < max_cs[j]:
                cs_adj[i, j] *= max_cs[i] / max_cs[j]

    # --- Learning-rate weight: the more you resemble others, the closer to 0 ---
    wv = np.clip(1.0 - cs_adj.max(axis=1), 0.0, 1.0)
    if wv.max() > EPS:
        wv = wv / wv.max()
    wv[wv >= 1.0] = 0.99
    wv[wv <= 0.0] = 1e-6

    with np.errstate(divide="ignore", invalid="ignore"):
        wv = np.log(wv / (1.0 - wv)) + 0.5
    wv = np.nan_to_num(wv, nan=0.0, posinf=1.0, neginf=0.0)
    wv = np.clip(wv, 0.0, 1.0)

    # --- Weighted aggregation ---
    total = float(wv.sum())
    if total <= EPS:
        agg = [np.zeros_like(np.asarray(l, dtype=DTYPE)) for l in client_updates[0]]
    else:
        # Accumulate LAYER BY LAYER so no 66M-element flat vector has to be built
        agg = []
        for li in range(len(client_updates[0])):
            acc = np.zeros_like(np.asarray(client_updates[0][li], dtype=DTYPE))
            for i in range(n):
                acc += np.asarray(client_updates[i][li], dtype=DTYPE) * DTYPE(wv[i])
            agg.append((acc / DTYPE(total)).astype(DTYPE))

    thr = 0.5 * float(wv.max()) if wv.max() > EPS else 0.0
    flagged = [i for i in range(n) if wv[i] <= thr]
    return agg, {"foolsgold_weights": [round(float(x), 4) for x in wv],
                 "max_cosine": [round(float(x), 4) for x in max_cs],
                 "flagged_clients": flagged,
                 "new_history": hist}


def foolsgold(client_updates, history=None, last_layers=0) -> List[np.ndarray]:
    return foolsgold_with_info(client_updates, history, last_layers)[0]


# -------------------------------------------------------------------------
# 8. RLR - Robust Learning Rate  (Ozdayi et al., AAAI 2021)
# -------------------------------------------------------------------------
def rlr_with_info(client_updates: List[List[np.ndarray]],
                  threshold: int = 4) -> Tuple[List[np.ndarray], Dict]:
    """
    A fundamentally different idea from every defence above: it does NOT score clients,
    it examines EACH PARAMETER COORDINATE.

    For every coordinate j, count how much the clients agree on the SIGN of the update:
            s_j = | sum_i sign(u_i[j]) |
    If s_j >= theta -> the clients agree      -> keep the learning direction (+1)
    If s_j <  theta -> abnormal disagreement  -> FLIP the learning-rate sign (-1),
                                                 actively learning AWAY at that coordinate.

    Why it fits this problem: at exactly the coordinates that carry the backdoor, the
    malicious group pushes very consistently while honest clients have almost no gradient
    (they have never seen the trigger) and contribute only random noise -> low agreement
    -> sign flipped -> the backdoor is ACTIVELY removed rather than merely diluted.

    This is a COORDINATE-LOCAL signal, something Median/Krum/FLTrust (which look at global
    distance or direction) cannot capture.
    """
    n = len(client_updates)
    agg, flipped_total, coord_total = [], 0, 0

    for li in range(len(client_updates[0])):
        stacked = np.stack([np.asarray(u[li], dtype=DTYPE) for u in client_updates], axis=0)
        sign_sum = np.abs(np.sign(stacked).sum(axis=0))
        lr = np.where(sign_sum >= threshold, DTYPE(1.0), DTYPE(-1.0))
        agg.append((lr * stacked.mean(axis=0)).astype(DTYPE))
        flipped_total += int((lr < 0).sum())
        coord_total += int(lr.size)
        del stacked, sign_sum, lr

    ratio = flipped_total / max(coord_total, 1)
    return agg, {"threshold": threshold,
                 "flipped_coords": flipped_total,
                 "total_coords": coord_total,
                 "flipped_ratio": round(ratio, 4)}


def rlr(client_updates, threshold: int = 4) -> List[np.ndarray]:
    return rlr_with_info(client_updates, threshold)[0]


# -------------------------------------------------------------------------
# 9. SA-Trust - the variant PROPOSED by this project
# -------------------------------------------------------------------------
def sign_agreement_scores(client_updates: List[List[np.ndarray]]) -> List[float]:
    """
    For each client: the fraction of coordinates where its update SIGN MATCHES the
    majority sign. A malicious client pushing a backdoor in its own direction should
    deviate from the majority sign inside the backdoor region.
    Computed layer by layer so no 66M-element flat vector is needed.
    """
    n = len(client_updates)
    agree = np.zeros(n, dtype=np.float64)
    total = 0
    for li in range(len(client_updates[0])):
        stacked = np.stack([np.asarray(u[li], dtype=DTYPE) for u in client_updates], axis=0)
        s = np.sign(stacked)
        majority = np.sign(s.sum(axis=0))
        for i in range(n):
            agree[i] += float((s[i] == majority).sum())
        total += int(s[0].size)
        del stacked, s, majority
    return [float(a / max(total, 1)) for a in agree]


def satrust_with_info(client_updates: List[List[np.ndarray]],
                      w_sign: float = 0.7) -> Tuple[List[np.ndarray], Dict]:
    """
    SA-Trust (Sign-Agreement Trust) - the variant proposed by this project.

    HONEST ABOUT ITS NOVELTY: this is NOT an entirely new algorithm; it COMBINES two
    already-published signals, applied at CLIENT level:
        (a) sign agreement     - RLR's coordinate-wise idea (Ozdayi et al., 2021),
                                 here reduced to a PER-CLIENT score
        (b) cosine similarity  - against the crowd's consensus direction (not a root set)

            trust_i = w * sign_agree_i + (1-w) * ReLU(cos_i)

    Differences from the two closest works:
      - Unlike FLTrust (Cao et al., 2021): requires NO clean root set on the server. That
        is FLTrust's biggest practical limitation - many deployments cannot obtain
        trustworthy clean data server-side.
      - Unlike RLR (Ozdayi et al., 2021): RLR only flips coordinate signs and CANNOT name
        the suspicious clients. SA-Trust scores each client, so detection_rate and
        false_exclusion_rate can be computed for evaluation and forensics.

    Report the result HONESTLY even when SA-Trust loses to plain RLR.
    """
    n = len(client_updates)
    sign_scores = sign_agreement_scores(client_updates)

    # The crowd's consensus direction = mean of the updates (no clean root set involved)
    mean_update = fedavg(client_updates)
    g0 = flatten(mean_update)
    g0n = norm64(g0) + EPS
    cos_scores = []
    for u in client_updates:
        gi = flatten(u)
        cos_scores.append(max(0.0, dot64(gi, g0) / ((norm64(gi) + EPS) * g0n)))

    trust = [w_sign * s + (1.0 - w_sign) * c for s, c in zip(sign_scores, cos_scores)]

    # Relative normalisation: strongly penalise clients that fall well below the median
    med = float(np.median(trust))
    weights = [max(0.0, t - 0.5 * med) for t in trust]
    total = float(sum(weights))
    if total <= EPS:
        weights = [1.0] * n
        total = float(n)

    agg = []
    for li in range(len(client_updates[0])):
        acc = np.zeros_like(np.asarray(client_updates[0][li], dtype=DTYPE))
        for i in range(n):
            acc += np.asarray(client_updates[i][li], dtype=DTYPE) * DTYPE(weights[i])
        agg.append((acc / DTYPE(total)).astype(DTYPE))

    thr = 0.5 * max(weights) if max(weights) > EPS else 0.0
    flagged = [i for i in range(n) if weights[i] <= thr]
    return agg, {"sign_agreement": [round(s, 4) for s in sign_scores],
                 "cosine": [round(c, 4) for c in cos_scores],
                 "trust": [round(t, 4) for t in trust],
                 "weights": [round(w, 4) for w in weights],
                 "flagged_clients": flagged}


def satrust(client_updates, w_sign: float = 0.7) -> List[np.ndarray]:
    return satrust_with_info(client_updates, w_sign)[0]
