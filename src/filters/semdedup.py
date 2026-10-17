"""SemDeDup-style semantic deduplication (Abbas et al., 2023): k-means, then drop
points whose cosine similarity to an already kept point in the same cluster
exceeds ``1 - eps``."""
from __future__ import annotations

import numpy as np

from src.filters.density import normalize


def semdedup_keep(emb: np.ndarray, cluster_size: int = 100, eps: float = 0.05,
                  seed: int = 0) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    x = normalize(emb).astype(np.float32)
    n_clusters = max(1, len(x) // cluster_size)
    labels = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3).fit_predict(x)
    keep = np.zeros(len(x), dtype=bool)
    threshold = 1.0 - eps
    for c in range(n_clusters):
        members = np.flatnonzero(labels == c)
        if len(members) == 0:
            continue
        sims = x[members] @ x[members].T
        kept: list[int] = []
        for i in range(len(members)):
            if not kept or sims[i, kept].max() <= threshold:
                kept.append(i)
        keep[members[kept]] = True
    return keep
