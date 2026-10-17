from collections import Counter

import numpy as np
import pytest

from src.filters.heuristics import words
from src.metrics.embedding import (frechet_distance, mean_pairwise_cosine_distance, vendi_score)
from src.metrics.text import (distinct_n, divergences, rare_recall, rare_words, self_bleu,
                              unigram_counts, zipf_slope)


def test_distinct_n():
    toks = [["a", "b", "a", "b"], ["a", "b"]]
    assert distinct_n(toks, 1) == pytest.approx(2 / 6)
    assert distinct_n(toks, 2) == pytest.approx(2 / 4)  # ab, ba, ab | ab


def _naive_self_bleu(docs):
    import math

    scores = []
    for i, hyp in enumerate(docs):
        refs = [d for j, d in enumerate(docs) if j != i]
        log_p = 0.0
        for n in range(1, 5):
            h = Counter(zip(*(hyp[k:] for k in range(n))))
            ref_max: Counter = Counter()
            for r in refs:
                for g, v in Counter(zip(*(r[k:] for k in range(n)))).items():
                    ref_max[g] = max(ref_max[g], v)
            match = sum(min(v, ref_max[g]) for g, v in h.items())
            log_p += math.log((match + 1) / (sum(h.values()) + 1))
        lens = np.array([len(r) for r in refs])
        closest = lens[np.argmin(np.abs(lens - len(hyp)))]
        bp = 1.0 if len(hyp) > closest else math.exp(1 - closest / len(hyp))
        scores.append(bp * math.exp(log_p / 4))
    return float(np.mean(scores))


def test_self_bleu_matches_naive_quadratic_version():
    rng = np.random.default_rng(0)
    vocab = list("abcdefg")
    docs = [list(rng.choice(vocab, size=rng.integers(5, 15))) for _ in range(12)]
    docs.append(docs[0][:])  # exact duplicate exercises the max/second-max logic
    assert self_bleu(docs) == pytest.approx(_naive_self_bleu(docs))


def test_self_bleu_identical_corpus_is_high():
    doc = words("the cat sat on the mat and the dog sat on the log")
    assert self_bleu([doc] * 5) > 0.9


def test_divergences_zero_for_same_distribution():
    c = Counter({"a": 5, "b": 3})
    d = divergences(c, c)
    assert d["kl_real_gen"] == pytest.approx(0, abs=1e-9)
    assert d["js_real_gen"] == pytest.approx(0, abs=1e-9)
    assert divergences(c, Counter({"z": 4}))["js_real_gen"] > 0.6


def test_rare_recall():
    ref = Counter({"common": 100, "rare1": 2, "rare2": 3, "single": 1})
    rare = rare_words(ref, 2, 10)
    assert rare == {"rare1", "rare2"}
    assert rare_recall([["rare1", "common"]], rare) == 0.5


def test_zipf_slope_of_zipfian_counts():
    counts = Counter({f"w{r}": int(1e5 / r) for r in range(1, 500)})
    assert zipf_slope(counts) == pytest.approx(-1.0, abs=0.05)


def test_vendi_bounds():
    same = np.tile([[1.0, 0, 0]], (10, 1))
    assert vendi_score(same) == pytest.approx(1.0)
    assert vendi_score(np.eye(5)) == pytest.approx(5.0)


def test_mean_pairwise_cosine_distance():
    assert mean_pairwise_cosine_distance(np.eye(3)) == pytest.approx(1.0)
    assert mean_pairwise_cosine_distance(np.ones((4, 3))) == pytest.approx(0.0, abs=1e-9)


def test_frechet_distance_zero_for_same_set():
    x = np.random.default_rng(0).normal(size=(200, 4))
    assert frechet_distance(x, x) == pytest.approx(0.0, abs=1e-6)
    assert frechet_distance(x, x + 3) > 0.01


def test_unigram_counts():
    assert unigram_counts([["a", "b"], ["a"]]) == Counter({"a": 2, "b": 1})
