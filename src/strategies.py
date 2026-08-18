"""
strategies.py - WRAP THE AGGREGATION ALGORITHMS INTO FLOWER STRATEGIES
=======================================================================
Each defence is a subclass of flwr's FedAvg strategy that only overrides aggregate_fit()
to call the corresponding pure function in aggregation.py.

Two design points worth noting:
  1. The base class tracks the CURRENT GLOBAL WEIGHTS so it can compute each client's
     UPDATE (delta). Norm-Clipping and FLTrust must operate on deltas.
  2. It records a DIAGNOSTIC LOG every round: which clients were excluded/down-weighted,
     their update norms, their trust scores. That is what makes the "malicious client
     detection rate" metric computable.
"""

from typing import List, Dict
import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

import config
from src import aggregation as agg


def _client_id(proxy, fit_res, fallback: int) -> int:
    """
    Recover the REAL client index (0..N-1).

    CAUTION: do NOT use ClientProxy.cid. In recent Flower versions the server-side cid is
    the node_id - a RANDOM HASH such as 2465052526735391746 - not the partition index
    "0".."9". Relying on cid makes every diagnostic (which client was excluded, the
    malicious group's update norms, the trust scores, ...) map to the WRONG client, and
    turns detection_rate / trust_gap into nonsense.

    The only trustworthy source is the metrics the CLIENT ITSELF reports (see fl_client.fit).
    """
    try:
        cid = fit_res.metrics.get("client_id")
        if cid is not None:
            return int(cid)
    except Exception:
        pass
    # Fallback: accept cid only if it really falls inside the valid client index range
    try:
        c = int(str(proxy.cid))
        if 0 <= c < config.NUM_CLIENTS:
            return c
    except Exception:
        pass
    return fallback


def _extract(results):
    """(weights, num_examples, client_ids) - ordered by the REAL client index."""
    tagged = [(_client_id(p, fr, i), p, fr) for i, (p, fr) in enumerate(results)]
    tagged.sort(key=lambda t: t[0])
    weights = [parameters_to_ndarrays(fr.parameters) for _, _, fr in tagged]
    num_examples = [fr.num_examples for _, _, fr in tagged]
    cids = [cid for cid, _, _ in tagged]
    return weights, num_examples, cids


class BaseStrategy(fl.server.strategy.FedAvg):
    """Base class: tracks the global weights and records diagnostics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_global = parameters_to_ndarrays(kwargs["initial_parameters"])
        self.diagnostics: List[Dict] = []      # one entry per round

    def _log(self, server_round, info: Dict, flagged: List[int], cids: List[int]):
        rec = {"round": server_round, "flagged_client_ids": [cids[i] for i in flagged]}
        rec.update(info)
        self.diagnostics.append(rec)

    def _finish(self, new_global):
        self.current_global = new_global
        return ndarrays_to_parameters(new_global), {}


class FedAvgStrategy(BaseStrategy):
    """Undefended baseline."""

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        w, n, cids = _extract(results)
        self._log(server_round, {}, [], cids)
        return self._finish(agg.fedavg(w, n))


class MedianStrategy(BaseStrategy):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        w, _, cids = _extract(results)
        print(f"  [Round {server_round}] Median over {len(w)} updates")
        self._log(server_round, {}, [], cids)
        return self._finish(agg.coordinate_median(w))


class TrimmedMeanStrategy(BaseStrategy):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        w, _, cids = _extract(results)
        print(f"  [Round {server_round}] Trimmed Mean (ratio={config.TRIMMED_RATIO})")
        self._log(server_round, {}, [], cids)
        return self._finish(agg.trimmed_mean(w, config.TRIMMED_RATIO))


class KrumStrategy(BaseStrategy):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        w, _, cids = _extract(results)
        new_global, info = agg.krum_with_info(w, config.KRUM_NUM_MALICIOUS, config.KRUM_MULTI)
        kind = "Multi-Krum" if config.KRUM_MULTI else "Krum"
        excluded_ids = [cids[i] for i in info["excluded"]]
        print(f"  [Round {server_round}] {kind}: excluded clients {excluded_ids}")
        self._log(server_round, {"krum_scores": info["scores"]}, info["excluded"], cids)
        return self._finish(new_global)


class NormClipStrategy(BaseStrategy):
    """Norm-Clipping + noise (Sun et al. 2019) - operates on the UPDATE."""

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        w, _, cids = _extract(results)
        g = self.current_global
        updates = [agg.subtract(cw, g) for cw in w]

        agg_update, info = agg.norm_clipping_with_info(
            updates, threshold=config.NORMCLIP_FIXED_THRESHOLD,
            mode=config.NORMCLIP_MODE, noise_std=config.NORMCLIP_NOISE_STD,
            seed=config.RANDOM_SEED + server_round)

        clipped_ids = [cids[i] for i in info["clipped_clients"]]
        print(f"  [Round {server_round}] Norm-Clip tau={info['threshold']} | "
              f"heavily clipped clients {clipped_ids}")
        self._log(server_round,
                  {"update_norms": info["norms"], "clip_scales": info["scales"],
                   "clip_threshold": info["threshold"]},
                  info["clipped_clients"], cids)
        return self._finish(agg.add(g, agg_update))


class FLTrustStrategy(BaseStrategy):
    """FLTrust - the server uses a clean root set to compute trust scores."""

    def __init__(self, *args, server_update_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_update_fn = server_update_fn

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        w, _, cids = _extract(results)
        g = self.current_global
        updates = [agg.subtract(cw, g) for cw in w]
        server_update = self.server_update_fn(g)

        agg_update, info = agg.fltrust_with_info(updates, server_update)
        flagged_ids = [cids[i] for i in info["flagged_clients"]]
        print(f"  [Round {server_round}] FLTrust TS={info['trust_scores']} | "
              f"suspected clients {flagged_ids}")
        self._log(server_round, {"trust_scores": info["trust_scores"]},
                  info["flagged_clients"], cids)
        return self._finish(agg.add(g, agg_update))


class FoolsGoldStrategy(BaseStrategy):
    """
    FoolsGold (Fung et al., RAID 2020) - detects a group of clients implanting the same
    backdoor via abnormal similarity between their updates. Keeps a running history
    across rounds.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        w, _, cids = _extract(results)
        g = self.current_global
        updates = [agg.subtract(cw, g) for cw in w]

        agg_update, info = agg.foolsgold_with_info(
            updates, self.history, last_layers=config.FOOLSGOLD_LAST_LAYERS)
        self.history = info.pop("new_history")

        flagged_ids = [cids[i] for i in info["flagged_clients"]]
        print(f"  [Round {server_round}] FoolsGold w={info['foolsgold_weights']} | "
              f"down-weighted clients {flagged_ids}")
        self._log(server_round,
                  {"foolsgold_weights": info["foolsgold_weights"],
                   "max_cosine": info["max_cosine"]},
                  info["flagged_clients"], cids)
        return self._finish(agg.add(g, agg_update))


class FLTrustNormClipStrategy(FLTrustStrategy):
    """
    COMBINED DEFENCE (recommended): FLTrust + Norm-Clipping.
    FLTrust filters by DIRECTION (catching updates aimed at a different objective) while
    Norm-Clip caps MAGNITUDE (catching amplified updates). The two layers cover each
    other's blind spots.
    """

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        w, _, cids = _extract(results)
        g = self.current_global
        updates = [agg.subtract(cw, g) for cw in w]

        # Layer 1: clip norms so nobody can dominate through sheer magnitude
        clipped, clip_info = agg.norm_clipping_with_info(
            updates, mode=config.NORMCLIP_MODE, noise_std=0.0,
            seed=config.RANDOM_SEED + server_round)
        scales = clip_info["scales"]
        clipped_updates = [[l * s for l in u] for u, s in zip(updates, scales)]

        # Layer 2: score trust by direction against the clean root set
        server_update = self.server_update_fn(g)
        agg_update, ft_info = agg.fltrust_with_info(clipped_updates, server_update)

        flagged = sorted(set(ft_info["flagged_clients"]) | set(clip_info["clipped_clients"]))
        print(f"  [Round {server_round}] FLTrust+NormClip TS={ft_info['trust_scores']} | "
              f"suspected clients {[cids[i] for i in flagged]}")
        self._log(server_round,
                  {"trust_scores": ft_info["trust_scores"],
                   "update_norms": clip_info["norms"],
                   "clip_scales": scales},
                  flagged, cids)
        return self._finish(agg.add(g, agg_update))


class RLRStrategy(BaseStrategy):
    """RLR - Robust Learning Rate (Ozdayi et al., AAAI 2021). Operates on the UPDATE."""

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        w, _, cids = _extract(results)
        g = self.current_global
        updates = [agg.subtract(cw, g) for cw in w]

        agg_update, info = agg.rlr_with_info(updates, config.RLR_THRESHOLD)
        print(f"  [Round {server_round}] RLR (theta={info['threshold']}): flipped "
              f"{info['flipped_ratio']:.1%} of the coordinates")
        # RLR does not score clients -> no client is flagged
        self._log(server_round,
                  {"rlr_flipped_ratio": info["flipped_ratio"],
                   "rlr_threshold": info["threshold"]}, [], cids)
        return self._finish(agg.add(g, agg_update))


class SATrustStrategy(BaseStrategy):
    """SA-Trust - the project's proposed variant (sign agreement + cosine, client level)."""

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        w, _, cids = _extract(results)
        g = self.current_global
        updates = [agg.subtract(cw, g) for cw in w]

        agg_update, info = agg.satrust_with_info(updates, config.SATRUST_W_SIGN)
        flagged_ids = [cids[i] for i in info["flagged_clients"]]
        print(f"  [Round {server_round}] SA-Trust trust={info['trust']} | "
              f"suspected clients {flagged_ids}")
        self._log(server_round,
                  {"sign_agreement": info["sign_agreement"],
                   "cosine": info["cosine"],
                   "trust_scores": info["trust"],
                   "satrust_weights": info["weights"]},
                  info["flagged_clients"], cids)
        return self._finish(agg.add(g, agg_update))


_REGISTRY = {
    "fedavg": FedAvgStrategy,
    "median": MedianStrategy,
    "trimmed": TrimmedMeanStrategy,
    "krum": KrumStrategy,
    "normclip": NormClipStrategy,
    "fltrust": FLTrustStrategy,
    "foolsgold": FoolsGoldStrategy,
    "fltrust_clip": FLTrustNormClipStrategy,
    "rlr": RLRStrategy,
    "satrust": SATrustStrategy,
}

# Defences that NEED a clean root set on the server (to derive a reference update).
# Declared CENTRALLY here rather than scattering `defense == "fltrust"` checks through the
# codebase: when a new root-based defence is added, only ONE place needs editing.
NEEDS_SERVER_UPDATE = {name for name, cls in _REGISTRY.items()
                       if issubclass(cls, FLTrustStrategy)}


def requires_server_update(name: str) -> bool:
    """Does this defence need a server-side root set?"""
    return str(name).lower() in NEEDS_SERVER_UPDATE


def make_strategy(name, initial_parameters, evaluate_fn, server_update_fn=None):
    common = dict(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=config.NUM_CLIENTS,
        min_available_clients=config.NUM_CLIENTS,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
    )
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unsupported defence '{name}'. Choose one of {list(_REGISTRY)}.")
    if requires_server_update(name):
        if server_update_fn is None:
            raise ValueError(
                f"{name} requires server_update_fn (the server trains on the root set).")
        return _REGISTRY[name](server_update_fn=server_update_fn, **common)
    return _REGISTRY[name](**common)
