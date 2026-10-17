"""Toy experiment (phase F1): recursive fitting of a 2-D Gaussian mixture.

The "true" distribution has three common modes and one rare mode (the tail).
Each generation fits a GMM to N samples of the previous generation's model and
samples from it again. A small share of generated points is replaced by uniform
noise to imitate degenerate LLM outputs. Lines:

    raw           N samples, no filtering
    density_only  oversampled pool, center downsampled by inverse k-NN density
    dual          density axis + noise bounds calibrated on real data likelihood
    human_mix     30% fixed real samples + 70% raw synthetic

Usage:
    python scripts/toy_gaussian.py --out results/toy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.filters.density import density_rank  # noqa: E402
from src.utils import resolve  # noqa: E402

MEANS = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5], [9.0, 7.0]])
WEIGHTS = np.array([0.32, 0.32, 0.32, 0.04])  # last mode = rare tail
RARE = 3
COV = 0.5 * np.eye(2)


def true_model() -> GaussianMixture:
    g = GaussianMixture(n_components=4, covariance_type="full")
    g.weights_, g.means_ = WEIGHTS, MEANS
    g.covariances_ = np.stack([COV] * 4)
    g.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(g.covariances_))
    return g


def sample(model: GaussianMixture, n: int, rng: np.random.Generator, noise_rate: float) -> np.ndarray:
    model.random_state = int(rng.integers(1 << 31))
    x, _ = model.sample(n)
    x = x[rng.permutation(n)]
    noisy = rng.random(n) < noise_rate
    x[noisy] = rng.uniform(-15, 25, size=(noisy.sum(), 2))
    return x


def fit(x: np.ndarray, seed: int) -> GaussianMixture:
    return GaussianMixture(n_components=4, covariance_type="full", random_state=seed, reg_covar=1e-4).fit(x)


def knn_density_score(x: np.ndarray, k: int = 10) -> np.ndarray:
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)
    return -dist[:, 1:].mean(1)


def dual_select(pool: np.ndarray, n: int, rng: np.random.Generator, loglik_floor: float | None,
                ref: GaussianMixture, center_frac=0.7, alpha=1.0) -> np.ndarray:
    rank = density_rank(knn_density_score(pool))
    tail_q = (1 - center_frac) * n / len(pool)  # tail_quantile=auto: keep every clean tail point
    tail = rank <= tail_q
    clean = np.ones(len(pool), bool)
    if loglik_floor is not None:
        clean = ~tail | (ref.score_samples(pool) >= loglik_floor)
    tail_c, center_c = np.flatnonzero(tail & clean), np.flatnonzero(~tail & clean)
    n_tail = min(len(tail_c), int(round((1 - center_frac) * n)))
    pick_t = rng.choice(tail_c, n_tail, replace=False)
    w = rank[center_c] ** -alpha
    pick_c = rng.choice(center_c, min(len(center_c), n - n_tail), replace=False, p=w / w.sum())
    return pool[np.concatenate([pick_t, pick_c])]


def evaluate(model: GaussianMixture, truth: GaussianMixture, rng: np.random.Generator) -> dict:
    xt = sample(truth, 20000, rng, 0.0)
    xm = sample(model, 20000, rng, 0.0)
    kl = float(np.mean(truth.score_samples(xt) - model.score_samples(xt)))
    rare_mass = float(np.mean(np.linalg.norm(xm - MEANS[RARE], axis=1) < 2.0))
    return {"kl_true_model": kl, "total_var": float(np.trace(np.cov(xm, rowvar=False))),
            "rare_mass": rare_mass}


def run_line(line: str, seed: int, args) -> list[dict]:
    rng = np.random.default_rng(seed)
    truth = true_model()
    real = sample(truth, args.n, rng, 0.0)
    real_val = sample(truth, 1000, rng, 0.0)
    model = fit(real, seed)
    ref = model  # frozen gen-0 model, as in the LLM pipeline
    floor = float(np.quantile(ref.score_samples(real_val), 0.01))
    rows = [{"line": line, "seed": seed, "generation": 0, **evaluate(model, truth, rng)}]
    for g in range(1, args.generations + 1):
        if line == "raw":
            data = sample(model, args.n, rng, args.noise_rate)
        elif line == "human_mix":
            n_real = int(0.3 * args.n)
            data = np.concatenate([real[:n_real], sample(model, args.n - n_real, rng, args.noise_rate)])
        else:
            pool = sample(model, int(args.oversample * args.n), rng, args.noise_rate)
            data = dual_select(pool, args.n, rng, floor if line == "dual" else None, ref)
        model = fit(data, seed * 1000 + g)
        rows.append({"line": line, "seed": seed, "generation": g, **evaluate(model, truth, rng)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--oversample", type=float, default=1.5)
    ap.add_argument("--noise-rate", type=float, default=0.02)
    ap.add_argument("--out", default="results/toy")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    lines = ["raw", "density_only", "dual", "human_mix"]
    df = pd.DataFrame([r for line in lines for s in range(args.seeds) for r in run_line(line, s, args)])
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "toy_metrics.csv", index=False)

    metrics = ["kl_true_model", "total_var", "rare_mass"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, metrics):
        for line in lines:
            s = df[df.line == line].groupby("generation")[metric].agg(["mean", "std"])
            ax.plot(s.index, s["mean"], label=line)
            ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.15)
        ax.set_title(metric)
        ax.set_xlabel("generation")
        ax.grid(alpha=0.3)
    axes[2].axhline(np.mean(np.linalg.norm(sample(true_model(), 20000, np.random.default_rng(0), 0)
                                          - MEANS[RARE], axis=1) < 2.0), ls="--", c="k", lw=1, label="truth")
    axes[0].legend()
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "toy_gaussian.png", dpi=150)
    last = df[df.generation == args.generations].groupby("line")[metrics].agg(["mean", "std"])
    print(last.round(4).to_string())


if __name__ == "__main__":
    main()
