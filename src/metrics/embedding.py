"""Embedding-space diversity and distribution-distance metrics."""
from __future__ import annotations

import numpy as np
from scipy import linalg

from src.filters.density import normalize


def vendi_score(emb: np.ndarray) -> float:
    """Vendi Score (Friedman & Dieng, 2023) with a cosine kernel.

    The nonzero eigenvalues of K/n = X X^T / n equal those of X^T X / n, so the
    d x d matrix is used instead of the n x n one.
    """
    x = normalize(emb).astype(np.float64)
    lam = np.linalg.eigvalsh(x.T @ x / len(x))
    lam = lam[lam > 1e-12]
    return float(np.exp(-np.sum(lam * np.log(lam))))


def mean_pairwise_cosine_distance(emb: np.ndarray) -> float:
    x = normalize(emb).astype(np.float64)
    n = len(x)
    s = x.sum(0)
    mean_sim = (s @ s - n) / (n * (n - 1))
    return float(1.0 - mean_sim)


def covariance_trace(emb: np.ndarray) -> float:
    return float(np.trace(np.cov(normalize(emb), rowvar=False)))


def frechet_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """FID-style distance between Gaussian fits of two embedding sets."""
    a, b = normalize(emb_a).astype(np.float64), normalize(emb_b).astype(np.float64)
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = np.real(linalg.sqrtm(ca @ cb))
    return float(np.sum((mu_a - mu_b) ** 2) + np.trace(ca + cb - 2 * covmean))


def embedding_metrics(emb: np.ndarray, real_emb: np.ndarray) -> dict[str, float]:
    return {
        "vendi": vendi_score(emb),
        "mean_cos_dist": mean_pairwise_cosine_distance(emb),
        "cov_trace": covariance_trace(emb),
        "frechet_real": frechet_distance(real_emb, emb),
    }


def mauve_score(emb: np.ndarray, real_emb: np.ndarray, seed: int = 0) -> float | None:
    try:
        import mauve
    except ImportError:
        return None
    out = mauve.compute_mauve(p_features=real_emb, q_features=emb, seed=seed, verbose=False)
    return float(out.mauve)
