"""Model-free corpus metrics: diversity, tail preservation, distribution shift."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from src.filters.heuristics import words


def ngrams(tokens: list[str], n: int):
    return zip(*(tokens[i:] for i in range(n)))


def distinct_n(tokenized: list[list[str]], n: int) -> float:
    """Unique n-grams / total n-grams over the whole corpus."""
    total, uniq = 0, set()
    for toks in tokenized:
        grams = list(ngrams(toks, n))
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / total if total else 0.0


def self_bleu(tokenized: list[list[str]], max_n: int = 4) -> float:
    """Mean BLEU-4 of every document against all others as references.

    Uses the fact that the clipped count of an n-gram in document i against the
    reference set "all others" only needs the max and second-max count of that
    n-gram over the corpus, so the cost is linear instead of quadratic.
    """
    docs = [t for t in tokenized if len(t) >= max_n]
    if len(docs) < 2:
        return 0.0
    per_n_counts = [[Counter(ngrams(t, n)) for t in docs] for n in range(1, max_n + 1)]
    top2: list[dict] = []
    for counts in per_n_counts:
        best: dict = {}
        for i, c in enumerate(counts):
            for g, v in c.items():
                b = best.get(g)
                if b is None:
                    best[g] = [v, i, 0]
                elif v > b[0]:
                    best[g] = [v, i, b[0]]
                elif v > b[2]:
                    b[2] = v
        top2.append(best)
    ref_lens = np.array([len(t) for t in docs])
    scores = []
    for i, toks in enumerate(docs):
        log_p = 0.0
        for n_i in range(max_n):
            c = per_n_counts[n_i][i]
            total = sum(c.values())
            match = 0
            for g, v in c.items():
                mx, arg, second = top2[n_i][g]
                ref_max = second if arg == i else mx
                match += min(v, ref_max)
            log_p += math.log((match + 1) / (total + 1))  # add-one smoothing (Lin & Och, 2004)
        others = np.delete(ref_lens, i)
        closest = others[np.argmin(np.abs(others - len(toks)))]
        bp = 1.0 if len(toks) > closest else math.exp(1 - closest / len(toks))
        scores.append(bp * math.exp(log_p / max_n))
    return float(np.mean(scores))


def unigram_counts(tokenized: list[list[str]]) -> Counter:
    c: Counter = Counter()
    for t in tokenized:
        c.update(t)
    return c


def divergences(p_counts: Counter, q_counts: Counter, alpha: float = 1e-6) -> dict[str, float]:
    """KL(P||Q) and JS between unigram distributions; P = real reference."""
    vocab = list(set(p_counts) | set(q_counts))
    p = np.array([p_counts.get(w, 0) for w in vocab], dtype=float) + alpha
    q = np.array([q_counts.get(w, 0) for w in vocab], dtype=float) + alpha
    p /= p.sum()
    q /= q.sum()
    mix = 0.5 * (p + q)
    kl = float(np.sum(p * np.log(p / q)))
    js = float(0.5 * np.sum(p * np.log(p / mix)) + 0.5 * np.sum(q * np.log(q / mix)))
    return {"kl_real_gen": kl, "js_real_gen": js}


def rare_words(reference: Counter, min_count: int, max_count: int) -> set[str]:
    return {w for w, c in reference.items() if min_count <= c <= max_count}


def rare_recall(tokenized: list[list[str]], rare: set[str]) -> float:
    """Share of the real data's rare-word band that appears at least once."""
    if not rare:
        return 0.0
    seen = set().union(*map(set, tokenized)) if tokenized else set()
    return len(rare & seen) / len(rare)


def zipf_slope(counts: Counter, top: int = 1000) -> float:
    freqs = np.array(sorted(counts.values(), reverse=True)[:top], dtype=float)
    if len(freqs) < 10:
        return float("nan")
    ranks = np.arange(1, len(freqs) + 1)
    slope, _ = np.polyfit(np.log(ranks), np.log(freqs), 1)
    return float(slope)


def corpus_metrics(texts: list[str], real_counts: Counter, rare: set[str], ecfg: dict,
                   rng: np.random.Generator) -> dict[str, float]:
    tok = [words(t) for t in texts]
    counts = unigram_counts(tok)
    sb_n = min(ecfg["self_bleu_samples"], len(tok))
    sb_idx = rng.choice(len(tok), size=sb_n, replace=False) if sb_n else []
    out = {f"distinct_{n}": distinct_n(tok, n) for n in (1, 2, 3, 4)}
    out["self_bleu"] = self_bleu([tok[i] for i in sb_idx])
    out["vocab_size"] = len(counts)
    out["rare_recall"] = rare_recall(tok, rare)
    out["zipf_slope"] = zipf_slope(counts, ecfg["zipf_top"])
    out.update(divergences(real_counts, counts))
    return out
