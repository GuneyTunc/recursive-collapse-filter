import copy

import numpy as np
import pytest
import yaml

from src.filters import heuristics, perplexity
from src.filters.density import density_rank, density_score, knn_mean_distance
from src.filters.selection import Bounds, PoolFeatures, select, tail_quantile
from src.filters.semdedup import semdedup_keep
from src.utils import resolve

FCFG = yaml.safe_load(open(resolve("configs/base.yaml"), encoding="utf-8"))["filter"]


def clustered_embeddings(rng, n_dense=300, n_sparse=60, dim=16):
    center = rng.normal(size=dim)
    dense = center + 0.05 * rng.normal(size=(n_dense, dim))
    sparse = rng.normal(size=(n_sparse, dim)) * 3
    return np.vstack([dense, sparse]).astype(np.float32)


def test_heuristics_flag_degenerate_text():
    good = "Once upon a time, a little girl named Lily found a red ball in the park."
    loop = "the dog ran " * 30
    feats = heuristics.text_features([good, loop])
    assert feats["rep3"][1] > 0.8 > feats["rep3"][0]
    assert feats["gzip_ratio"][1] < feats["gzip_ratio"][0]
    assert heuristics.nonalpha_ratio("@@##$$%%") == 1.0


def test_heuristic_bounds_are_one_sided():
    ref = {"n_words": np.arange(10, 110.0), "rep3": np.linspace(0, 0.2, 100),
           "gzip_ratio": np.linspace(0.4, 0.8, 100), "nonalpha_ratio": np.zeros(100)}
    b = heuristics.calibrate(ref, 0.01, 0.99)
    probe = {"n_words": np.array([500.0, 1.0]), "rep3": np.array([0.0, 0.0]),
             "gzip_ratio": np.array([0.9, 0.6]), "nonalpha_ratio": np.array([0.0, 0.0])}
    # Very long or very incompressible text is not "degenerate"; very short text is.
    assert heuristics.within_bounds(probe, b).tolist() == [True, False]


def test_perplexity_bounds_from_reference():
    ref = np.linspace(1.0, 3.0, 101)
    b = perplexity.calibrate(ref, 0.01, 0.99)
    ok = perplexity.within_bounds(np.array([0.5, 2.0, 5.0, np.nan]), b)
    assert ok.tolist() == [False, True, False, False]


@pytest.mark.parametrize("method", ["knn", "kde", "hdbscan"])
def test_dense_cluster_ranks_higher(method):
    rng = np.random.default_rng(0)
    emb = clustered_embeddings(rng)
    rank = density_rank(density_score(emb, method, k=10, pca_dim=8, min_cluster_size=20))
    assert rank[:300].mean() > rank[300:].mean() + 0.2
    assert rank.max() == 1.0 and rank.min() > 0


def test_knn_distance_ignores_self():
    emb = np.eye(4, dtype=np.float32)
    assert np.allclose(knn_mean_distance(emb, k=2), 1.0)


def test_semdedup_removes_near_duplicates():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(20, 16))
    emb = np.vstack([base, base + 1e-4 * rng.normal(size=base.shape)])
    keep = semdedup_keep(emb, cluster_size=10, eps=0.05)
    assert keep.sum() == 20


def test_tail_quantile_auto_keeps_whole_tail():
    d = {"tail_quantile": "auto", "center_frac": 0.7}
    assert tail_quantile(d, n=100, m=150) == pytest.approx(0.2)
    assert tail_quantile({**d, "tail_quantile": 0.3}, 100, 150) == 0.3


def _pool(rng, m=360):
    emb = clustered_embeddings(rng)[:m]
    log_ppl = rng.uniform(1.0, 2.0, size=m)
    log_ppl[-10:] = 9.0  # 10 noisy points, all in the sparse region
    heur = {"n_words": np.full(m, 100.0), "rep3": np.zeros(m), "gzip_ratio": np.full(m, 0.5),
            "nonalpha_ratio": np.zeros(m)}
    bounds = Bounds(ppl={"low": 0.5, "high": 3.0},
                    heur=heuristics.calibrate(heur, 0.01, 0.99))
    return PoolFeatures(emb, log_ppl, heur), bounds


@pytest.mark.parametrize("method", ["random", "density", "ppl", "semdedup", "dual"])
def test_every_method_returns_n_distinct_indices(method):
    rng = np.random.default_rng(2)
    feats, bounds = _pool(rng)
    idx, stats = select(method, 360, feats, bounds, 200, FCFG, rng)
    assert len(idx) == 200 and len(set(idx.tolist())) == 200
    assert idx.min() >= 0 and idx.max() < 360


def test_dual_filter_drops_noise_and_downsamples_center():
    rng = np.random.default_rng(3)
    feats, bounds = _pool(rng)
    idx, stats = select("dual", 360, feats, bounds, 200, FCFG, rng)
    assert not set(range(350, 360)) & set(idx.tolist())  # noisy tail points removed
    assert stats["rejected_ppl"] == 10
    picked_dense = np.isin(idx, np.arange(300)).sum()
    # Raw sampling would keep ~300/360 of 200 = 167 dense points.
    assert picked_dense < 167


def test_density_only_keeps_noise():
    rng = np.random.default_rng(3)
    feats, bounds = _pool(rng)
    idx, _ = select("density", 360, feats, bounds, 200, FCFG, rng)
    assert set(range(350, 360)) & set(idx.tolist())


def test_dual_fills_when_too_few_clean():
    rng = np.random.default_rng(4)
    feats, bounds = _pool(rng)
    feats.log_ppl[:] = 9.0  # everything out of bounds, but ppl_scope=tail
    cfg = copy.deepcopy(FCFG)
    cfg["dual"]["ppl_scope"] = "all"
    idx, stats = select("dual", 360, feats, bounds, 200, cfg, rng)
    assert len(idx) == 200 and stats["filled_unclean"] == 200
