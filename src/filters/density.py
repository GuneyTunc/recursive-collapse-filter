"""Density estimation in embedding space.

Raw density values are not comparable across estimators (and KDE in high
dimensions is numerically fragile), so every estimator returns a *score* where
larger = denser, and selection uses its rank in (0, 1].
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def normalize(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norm, 1e-12)


def knn_mean_distance(emb: np.ndarray, k: int) -> np.ndarray:
    """Mean cosine distance to the k nearest neighbours (self excluded)."""
    x = normalize(emb).astype(np.float32)
    k = min(k, len(x) - 1)
    try:
        import faiss

        index = faiss.IndexFlatIP(x.shape[1])
        index.add(x)
        sims, _ = index.search(x, k + 1)
    except ImportError:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(x)
        dist, _ = nn.kneighbors(x)
        sims = 1.0 - dist
    # Column 0 is the point itself (or an exact duplicate, which is equally fine).
    return np.clip(1.0 - sims[:, 1:], 0.0, 2.0).mean(axis=1)


def _pca(emb: np.ndarray, dim: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    dim = min(dim, emb.shape[1], len(emb) - 1)
    return PCA(n_components=dim, random_state=seed).fit_transform(normalize(emb))


def density_score(emb: np.ndarray, method: str = "knn", k: int = 10, pca_dim: int = 32,
                  min_cluster_size: int = 25, seed: int = 0) -> np.ndarray:
    if method == "knn":
        return -knn_mean_distance(emb, k)
    if method == "kde":
        from sklearn.neighbors import KernelDensity

        z = _pca(emb, pca_dim, seed)
        return KernelDensity(kernel="gaussian", bandwidth="scott").fit(z).score_samples(z)
    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        z = _pca(emb, pca_dim, seed)
        model = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit(z)
        # Membership strength; noise points get 0. Break ties with k-NN density.
        return model.probabilities_ - 1e-3 * knn_mean_distance(emb, k)
    raise ValueError(f"Unknown density method {method!r}")


def density_rank(score: np.ndarray) -> np.ndarray:
    """Rank in (0, 1]; 1 = densest point."""
    return rankdata(score, method="average") / len(score)
