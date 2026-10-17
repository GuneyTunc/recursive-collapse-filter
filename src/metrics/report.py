"""Aggregate metrics.json files into tables and per-metric curves over generations.

    python -m src.metrics.report --results outputs --out results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import read_json, resolve

# Metrics plotted by default (higher-is-better is noted for the reader, not used in code).
KEY_METRICS = [
    "test_ppl",          # lower is better
    "distinct_2", "distinct_4", "self_bleu", "vendi", "mean_cos_dist",
    "rare_recall", "js_real_gen", "frechet_real", "zipf_slope",
    "rep3_mean", "gzip_ratio_mean", "n_words_mean",
    "data_vendi", "data_rare_recall",
]


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def collect(results_dir: Path) -> pd.DataFrame:
    rows = []
    gen0 = {}
    for p in sorted((results_dir / "_shared").glob("seed*/gen0/metrics.json")):
        m = _flatten(read_json(p))
        gen0[m["seed"]] = m
    for p in sorted(results_dir.glob("*/seed*/gen*/metrics.json")):
        if p.parts[-4] == "_shared":
            continue
        rows.append(_flatten(read_json(p)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Every line starts from the shared gen-0 model of its seed.
    extra = []
    for (line, seed) in df[["line", "seed"]].drop_duplicates().itertuples(index=False):
        if seed in gen0:
            extra.append({**gen0[seed], "line": line})
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True).sort_values(["line", "seed", "generation"])


def summarise(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    metrics = [m for m in metrics if m in df.columns]
    g = df.groupby(["line", "generation"])[metrics]
    mean, std, n = g.mean(), g.std(), g.size().rename("n_seeds")
    summary = mean.join(std, lsuffix="_mean", rsuffix="_std").join(n)
    return summary.reset_index()


def plot(df: pd.DataFrame, metrics: list[str], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in [m for m in metrics if m in df.columns]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for line, sub in df.groupby("line"):
            stats = sub.groupby("generation")[metric].agg(["mean", "std"]).fillna(0)
            ax.plot(stats.index, stats["mean"], marker="o", label=line)
            ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"], alpha=0.15)
        ax.set_xlabel("generation")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Aggregate experiment metrics")
    ap.add_argument("--results", default="outputs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    df = collect(resolve(args.results))
    if df.empty:
        print("No metrics.json found under", args.results)
        return
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_metrics.csv", index=False)
    summarise(df, KEY_METRICS).to_csv(out / "summary.csv", index=False)
    plot(df, KEY_METRICS, out / "plots")
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
