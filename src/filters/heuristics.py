"""Cheap, model-free text-quality signals and reference-calibrated bounds.

Every feature has a direction: a sample is rejected only on the side that signals
degeneration (e.g. too repetitive), never for being unusual in the other direction.
Bounds are quantiles of the same feature on real validation data, so no threshold
is hand-picked.

A quantile can collapse onto the feature's ideal value: almost no real story has a
single character outside the plain-English alphabet, so the 99% quantile of
``nonalpha_ratio`` is 0 and one "$" would reject a text. An optional per-feature
*tolerance* caps how strict a bound may get; it only acts when the calibrated bound
is stricter than the tolerance.
"""
from __future__ import annotations

import gzip
import re

import numpy as np

_WORD = re.compile(r"[a-z']+")
# Plain English text plus common typographic punctuation; anything else (e.g. U+FFFD from
# broken byte-level decoding) counts as noise.
_ALLOWED = re.compile(r"[A-Za-z0-9\s.,!?;:'\"()\-‘’“”–—…]")

# feature -> which side is bad ("high" = reject above upper bound)
FEATURES: dict[str, str] = {
    "n_words": "low",
    "rep3": "high",          # share of repeated word 3-grams
    "gzip_ratio": "low",     # compressed / raw bytes; repetitive text compresses well
    "nonalpha_ratio": "high",  # characters outside a plain-English alphabet
}


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def repeated_ngram_ratio(tokens: list[str], n: int = 3) -> float:
    grams = [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def gzip_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(gzip.compress(raw, compresslevel=6, mtime=0)) / len(raw)


def nonalpha_ratio(text: str) -> float:
    if not text:
        return 1.0
    return 1.0 - len(_ALLOWED.findall(text)) / len(text)


def text_features(texts: list[str]) -> dict[str, np.ndarray]:
    toks = [words(t) for t in texts]
    return {
        "n_words": np.array([len(t) for t in toks], dtype=float),
        "rep3": np.array([repeated_ngram_ratio(t) for t in toks]),
        "gzip_ratio": np.array([gzip_ratio(t) for t in texts]),
        "nonalpha_ratio": np.array([nonalpha_ratio(t) for t in texts]),
    }


def calibrate(reference: dict[str, np.ndarray], q_low: float, q_high: float,
              tolerance: dict[str, float] | None = None) -> dict[str, dict]:
    """Per-feature one-sided bound from the real reference distribution.

    ``tolerance`` maps a feature to the strictest bound it may get: an upper bound is
    never below it, a lower bound never above it. ``quantile`` keeps the calibrated
    value and ``tolerance_applied`` says whether the tolerance replaced it.
    """
    tolerance = tolerance or {}
    unknown = set(tolerance) - set(FEATURES)
    if unknown:
        raise ValueError(f"Tolerance for unknown heuristic feature(s): {sorted(unknown)}")
    bounds = {}
    for name, side in FEATURES.items():
        q = float(np.quantile(reference[name], q_low if side == "low" else q_high))
        value = q
        if name in tolerance:
            t = float(tolerance[name])
            value = min(q, t) if side == "low" else max(q, t)
        bounds[name] = {"side": side, "value": value, "quantile": q, "tolerance_applied": value != q}
    return bounds


def within_bounds(features: dict[str, np.ndarray], bounds: dict[str, dict]) -> np.ndarray:
    n = len(next(iter(features.values())))
    ok = np.ones(n, dtype=bool)
    for name, b in bounds.items():
        if b["side"] == "low":
            ok &= features[name] >= b["value"]
        else:
            ok &= features[name] <= b["value"]
    return ok
