"""Selection of the next generation's training set from a synthetic pool.

All functions here are pure: they take precomputed pool features (embeddings,
reference log-perplexity, heuristic features) and return pool indices, so they can
be unit-tested without any model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.filters import heuristics, perplexity
from src.filters.density import density_rank, density_score
from src.filters.semdedup import semdedup_keep

# Which pool features each method needs (lets the pipeline skip expensive ones).
REQUIRES: dict[str, set[str]] = {
    "random": set(),
    "density": {"emb"},
    "ppl": {"log_ppl", "heur"},
    "semdedup": {"emb"},
    "dual": {"emb", "log_ppl", "heur"},
}


@dataclass
class PoolFeatures:
    emb: np.ndarray | None = None
    log_ppl: np.ndarray | None = None
    heur: dict[str, np.ndarray] | None = None


@dataclass
class Bounds:
    ppl: dict[str, float] = field(default_factory=dict)
    heur: dict[str, dict] = field(default_factory=dict)


def requirements(method: str, fcfg: dict) -> set[str]:
    req = set(REQUIRES[method])
    if method == "dual":
        if fcfg["dual"]["ppl_scope"] == "none":
            req.discard("log_ppl")
        if fcfg["dual"]["heuristics_scope"] == "none":
            req.discard("heur")
    return req


def _weighted_pick(cands: np.ndarray, k: int, weights: np.ndarray | None,
                   rng: np.random.Generator) -> np.ndarray:
    k = min(k, len(cands))
    if k <= 0:
        return np.array([], dtype=int)
    p = None if weights is None else weights / weights.sum()
    return rng.choice(cands, size=k, replace=False, p=p)


def _fill(chosen: np.ndarray, n: int, pools: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Top ``chosen`` up to ``n`` from fallback pools, in order of preference."""
    chosen = np.asarray(chosen, dtype=int)
    for pool in pools:
        if len(chosen) >= n:
            break
        rest = np.setdiff1d(pool, chosen)
        chosen = np.concatenate([chosen, _weighted_pick(rest, n - len(chosen), None, rng)])
    return chosen


def noise_mask(feats: PoolFeatures, bounds: Bounds, use_heur: bool, use_ppl: bool, n: int) -> np.ndarray:
    ok = np.ones(n, dtype=bool)
    if use_heur:
        ok &= heuristics.within_bounds(feats.heur, bounds.heur)
    if use_ppl:
        ok &= perplexity.within_bounds(feats.log_ppl, bounds.ppl)
    return ok


def tail_quantile(dcfg: dict, n: int, m: int) -> float:
    """``auto``: the tail is exactly as large as its share of the selected set, so every
    clean tail point is kept and only the center is downsampled. A fixed quantile
    larger than that would thin the tail at the same rate as the center."""
    q = dcfg["tail_quantile"]
    if q == "auto":
        return min(1.0, (1.0 - dcfg["center_frac"]) * n / m)
    return float(q)


def dual_filter(feats: PoolFeatures, bounds: Bounds, n: int, fcfg: dict,
                rng: np.random.Generator, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Density axis (downsample the center) x noise axis (bound the tail)."""
    d, dens = fcfg["dual"], fcfg["density"]
    m = len(feats.emb)
    score = density_score(feats.emb, dens["method"], dens["k"], dens["pca_dim"],
                          dens["hdbscan_min_cluster_size"], seed)
    rank = density_rank(score)
    is_tail = rank <= tail_quantile(d, n, m)

    heur_ok = noise_mask(feats, bounds, d["heuristics_scope"] != "none", False, m)
    ppl_ok = noise_mask(feats, bounds, False, d["ppl_scope"] != "none", m)
    if d["heuristics_scope"] == "tail":
        heur_ok |= ~is_tail
    if d["ppl_scope"] == "tail":
        ppl_ok |= ~is_tail
    clean = heur_ok & ppl_ok

    tail_c = np.flatnonzero(is_tail & clean)
    center_c = np.flatnonzero(~is_tail & clean)
    n_tail = int(round((1.0 - d["center_frac"]) * n))
    tail_w = rank[tail_c] ** -d["tail_alpha"] if d["tail_alpha"] else None
    tail_pick = _weighted_pick(tail_c, n_tail, tail_w, rng)
    n_center = n - len(tail_pick)
    center_w = rank[center_c] ** -d["center_alpha"] if d["center_alpha"] else None
    center_pick = _weighted_pick(center_c, n_center, center_w, rng)

    chosen = np.concatenate([tail_pick, center_pick])
    filled = 0
    if len(chosen) < n:  # not enough clean data: fall back to clean, then to anything
        before = len(chosen)
        chosen = _fill(chosen, n, [np.flatnonzero(clean), np.arange(m)], rng)
        filled = len(chosen) - before
    stats = {
        "pool": m,
        "tail_size": int(is_tail.sum()),
        "rejected_heuristics": int((~heur_ok).sum()),
        "rejected_ppl": int((~ppl_ok).sum()),
        "rejected_tail": int((is_tail & ~clean).sum()),
        "rejected_center": int((~is_tail & ~clean).sum()),
        "picked_tail": int(len(tail_pick)),
        "picked_center": int(len(center_pick)),
        "filled_unclean": int(filled),
    }
    return chosen, stats


def select(method: str, pool_size: int, feats: PoolFeatures, bounds: Bounds, n: int, fcfg: dict,
           rng: np.random.Generator, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Return ``min(n, pool_size)`` distinct pool indices and selection statistics."""
    m = pool_size
    if method == "random":
        idx, stats = rng.choice(m, size=min(n, m), replace=False), {"pool": m}
    elif method == "dual":
        idx, stats = dual_filter(feats, bounds, n, fcfg, rng, seed)
    elif method == "density":
        cfg = {**fcfg, "dual": {**fcfg["dual"], "heuristics_scope": "none", "ppl_scope": "none"}}
        idx, stats = dual_filter(feats, bounds, n, cfg, rng, seed)
    elif method == "ppl":
        ok = noise_mask(feats, bounds, True, True, m)
        idx = _fill(_weighted_pick(np.flatnonzero(ok), n, None, rng), n, [np.arange(m)], rng)
        stats = {"pool": m, "rejected": int((~ok).sum()), "filled_unclean": int(max(0, n - ok.sum()))}
    elif method == "semdedup":
        s = fcfg["semdedup"]
        keep = semdedup_keep(feats.emb, s["cluster_size"], s["eps"], seed)
        idx = _fill(_weighted_pick(np.flatnonzero(keep), n, None, rng), n, [np.arange(len(keep))], rng)
        stats = {"pool": len(keep), "duplicates": int((~keep).sum()),
                 "filled_duplicates": int(max(0, n - keep.sum()))}
    else:
        raise ValueError(f"Unknown selection method {method!r}")
    assert len(idx) == min(n, m) and len(np.unique(idx)) == len(idx)
    return np.asarray(idx, dtype=int), stats
