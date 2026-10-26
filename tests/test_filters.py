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


def _reference(nonalpha):
    n = len(nonalpha)
    return {"n_words": np.linspace(50, 200, n), "rep3": np.linspace(0, 0.1, n),
            "gzip_ratio": np.linspace(0.45, 0.7, n), "nonalpha_ratio": np.asarray(nonalpha, float)}


def _probe(nonalpha):
    n = len(nonalpha)
    return {"n_words": np.full(n, 120.0), "rep3": np.zeros(n), "gzip_ratio": np.full(n, 0.6),
            "nonalpha_ratio": np.asarray(nonalpha, float)}


def test_degenerate_quantile_rejects_any_odd_character_without_tolerance():
    # 999 of 1000 real stories have no odd character: the 99% quantile is 0.
    ref = _reference([0.0] * 999 + [0.0022])
    b = heuristics.calibrate(ref, 0.01, 0.99)
    assert b["nonalpha_ratio"]["value"] == 0.0
    assert heuristics.within_bounds(_probe([0.0, 0.001]), b).tolist() == [True, False]


def test_tolerance_relaxes_a_degenerate_upper_bound():
    ref = _reference([0.0] * 999 + [0.0022])
    b = heuristics.calibrate(ref, 0.01, 0.99, {"nonalpha_ratio": 0.005})
    nb = b["nonalpha_ratio"]
    assert nb["value"] == 0.005 and nb["quantile"] == 0.0 and nb["tolerance_applied"]
    assert heuristics.within_bounds(_probe([0.0, 0.001, 0.005, 0.03]), b).tolist() == [
        True, True, True, False]
    # Features without a tolerance keep their calibrated quantile.
    assert not b["rep3"]["tolerance_applied"]


def test_tolerance_never_tightens_a_bound():
    high = heuristics.calibrate(_reference(np.linspace(0, 0.02, 100)), 0.01, 0.99,
                                {"nonalpha_ratio": 0.005})
    assert high["nonalpha_ratio"]["value"] == pytest.approx(0.0198)
    assert not high["nonalpha_ratio"]["tolerance_applied"]
    # Lower bounds: the tolerance can only lower them.
    ref = _reference([0.0] * 100)
    assert heuristics.calibrate(ref, 0.01, 0.99, {"n_words": 40})["n_words"]["value"] == 40
    assert heuristics.calibrate(ref, 0.01, 0.99, {"n_words": 80})["n_words"]["value"] == pytest.approx(51.5)


def test_tolerance_rejects_unknown_feature():
    with pytest.raises(ValueError, match="unknown heuristic"):
        heuristics.calibrate(_reference([0.0] * 10), 0.01, 0.99, {"nonalpha": 0.005})


def test_base_config_tolerance_keeps_prose_and_drops_markup():
    tol = yaml.safe_load(open(resolve("configs/base.yaml"), encoding="utf-8"))[
        "filter"]["bounds"]["heuristic_tolerance"]
    # A typical story length (~700 characters) with one "$".
    story = ("Once upon a time, a little girl named Lily found a shiny coin in the park. "
             "It said $1 on one side. ") + (
             "She ran home to show her mom, and they smiled and talked about it all day. ") * 8
    markup = ("**Step 1:** run `print(x)` then [link](http://a.b/c?d=1) #tag $$\\{x\\}$$ "
              "and **Step 2:** done. ") * 4
    texts = [story, markup]
    feats = heuristics.text_features(texts)
    assert 0 < feats["nonalpha_ratio"][0] < 0.005 < feats["nonalpha_ratio"][1]
    b = heuristics.calibrate(_reference([0.0] * 1000), 0.01, 0.99, tol)
    probe = {**_probe(feats["nonalpha_ratio"]), "nonalpha_ratio": feats["nonalpha_ratio"]}
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
