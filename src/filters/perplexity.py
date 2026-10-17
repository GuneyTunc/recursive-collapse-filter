"""Perplexity bounding calibrated on real data.

Rare but informative text naturally has a higher perplexity, so a fixed global
ceiling would remove exactly the tail we want to keep. Instead the bounds are the
``[q_low, q_high]`` quantiles of the frozen reference model's log-perplexity on
*real* validation text: a sample is noise only if it is less likely than almost
any real document (upper bound), or degenerate/looping (lower bound).
"""
from __future__ import annotations

import numpy as np


def calibrate(reference_log_ppl: np.ndarray, q_low: float, q_high: float) -> dict[str, float]:
    return {"low": float(np.quantile(reference_log_ppl, q_low)),
            "high": float(np.quantile(reference_log_ppl, q_high))}


def within_bounds(log_ppl: np.ndarray, bounds: dict[str, float]) -> np.ndarray:
    return (log_ppl >= bounds["low"]) & (log_ppl <= bounds["high"]) & np.isfinite(log_ppl)
