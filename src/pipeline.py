"""Recursive training loop: train -> evaluate -> generate pool -> select -> next generation.

Layout (``<out>`` = experiment.output_dir):

    <out>/_shared/seed<k>/reference.json        reference calibration (bounds, rare words)
    <out>/_shared/seed<k>/gen0/model/           frozen gen-0 model (also the PPL reference)
    <out>/_shared/seed<k>/gen0/metrics.json
    <out>/_shared/seed<k>/gen1/generated.jsonl  pool sampled from gen-0 (shared by all lines)
    <out>/<line>/seed<k>/gen<g>/generated.jsonl pool sampled from gen g-1 (g >= 2)
    <out>/<line>/seed<k>/gen<g>/filtered.jsonl  selected synthetic training data of gen g
    <out>/<line>/seed<k>/gen<g>/samples.jsonl   evaluation samples of gen g
    <out>/<line>/seed<k>/gen<g>/metrics.json    written last; marks the generation as done

Every generation is fine-tuned from the same base weights (``model.base``). A run
can be interrupted and restarted: finished generations are skipped and cached
pools / features are reused.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import platform
from collections import Counter
from pathlib import Path

import numpy as np

from src.config import load_config
from src.data.real import RealData, prepare_real_data
from src.features import embed, heuristic_features, reference_log_ppl
from src.filters import heuristics, perplexity
from src.filters.heuristics import words
from src.filters.selection import Bounds, PoolFeatures, requirements, select
from src.metrics.embedding import embedding_metrics, mauve_score
from src.metrics.text import corpus_metrics, rare_words, unigram_counts
from src.utils import (free_gpu, git_hash, read_json, read_jsonl, resolve, set_seed, setup_logging,
                       timer, write_json, write_jsonl)

log = logging.getLogger("mc")


class Run:
    def __init__(self, cfg: dict, seed: int):
        self.cfg, self.seed = cfg, seed
        e = cfg["experiment"]
        self.out = resolve(e["output_dir"])
        self.shared = self.out / "_shared" / f"seed{seed}"
        self.line_dir = self.out / e["line"] / f"seed{seed}"
        self.n = cfg["data"]["n_train"]
        self.pool_size = int(round(cfg["generate"]["oversample"] * self.n))
        self.real: RealData = prepare_real_data(cfg)
        self._real_test_emb: np.ndarray | None = None
        self._real_counts: Counter | None = None
        self._rare: set[str] | None = None

    # ------------------------------------------------------------------ paths
    def gen_dir(self, g: int) -> Path:
        return self.shared / "gen0" if g == 0 else self.line_dir / f"gen{g}"

    def pool_path(self, g: int) -> Path:
        return (self.shared / "gen1" if g == 1 else self.line_dir / f"gen{g}") / "generated.jsonl"

    def reference_model_path(self) -> str:
        ref = self.cfg["filter"]["reference_model"]
        return str(self.shared / "gen0" / "model") if ref == "gen0" else ref

    # --------------------------------------------------------- shared reference
    @property
    def real_counts(self) -> Counter:
        if self._real_counts is None:
            self._real_counts = unigram_counts([words(t) for t in self.real.test])
        return self._real_counts

    @property
    def rare(self) -> set[str]:
        if self._rare is None:
            e = self.cfg["eval"]
            train_counts = unigram_counts([words(t) for t in self.real.train])
            self._rare = rare_words(train_counts, e["rare_min_count"], e["rare_max_count"])
        return self._rare

    @property
    def real_test_emb(self) -> np.ndarray:
        if self._real_test_emb is None:
            f = self.cfg["filter"]
            self._real_test_emb = embed(self.real.test, f["embedding_model"], f["embedding_batch_size"],
                                        self.shared / "real_test_emb.npy")
        return self._real_test_emb

    def reference(self) -> dict:
        """Bounds calibrated on real validation data under the frozen reference model."""
        path = self.shared / "reference.json"
        if path.exists():
            return read_json(path)
        b = self.cfg["filter"]["bounds"]
        val_ppl = reference_log_ppl(self.real.val, self.reference_model_path(), self.cfg,
                                    self.shared / "real_val_logppl.npy")
        ref = {
            "reference_model": self.reference_model_path(),
            "ppl": perplexity.calibrate(val_ppl, b["ppl_low"], b["ppl_high"]),
            "heur": heuristics.calibrate(heuristic_features(self.real.val),
                                         b["heuristic_low"], b["heuristic_high"]),
            "rare_band_size": len(self.rare),
        }
        write_json(path, ref)
        return ref

    # --------------------------------------------------------------- training
    def train_eval_generate(self, g: int, train_texts: list[str], metrics: dict) -> None:
        """Fine-tune gen g, compute model metrics, write next pool and eval samples."""
        from src.generate.sampler import generate_texts
        from src.lm import corpus_ppl
        from src.train.finetune import finetune

        gdir = self.gen_dir(g)
        last = g == self.cfg["experiment"]["generations"]
        with timer(metrics, "time_train"):
            model, tok, train_stats = finetune(self.cfg, train_texts, seed=self.seed * 1000 + g)
        metrics.update(train_stats)
        if g == 0 or self.cfg["experiment"]["keep_checkpoints"]:
            model.save_pretrained(gdir / "model")
            tok.save_pretrained(gdir / "model")

        with timer(metrics, "time_test_ppl"):
            metrics["test_ppl"] = corpus_ppl(model, tok, self.real.test, self.cfg["model"]["max_length"],
                                             self.cfg["filter"]["ppl_batch_size"])
        free_gpu()
        n_eval = self.cfg["eval"]["n_samples"]
        with timer(metrics, "time_generate"):
            if last:
                samples = generate_texts(model, tok, self.cfg, n_eval, self.seed * 1000 + g + 1,
                                         self.real.train)
            else:
                pool = generate_texts(model, tok, self.cfg, self.pool_size, self.seed * 1000 + g + 1,
                                      self.real.train)
                write_jsonl(self.pool_path(g + 1), pool)
                samples = pool[:n_eval]  # the pool is an i.i.d. sample of this generation's model
        write_jsonl(gdir / "samples.jsonl", samples)
        del model
        free_gpu()

    # -------------------------------------------------------------- selection
    def pool_features(self, g: int, pool: list[str], need: set[str]) -> PoolFeatures:
        f = self.cfg["filter"]
        cache_dir = self.pool_path(g).parent
        feats = PoolFeatures()
        if "emb" in need:
            feats.emb = embed(pool, f["embedding_model"], f["embedding_batch_size"],
                              cache_dir / "pool_emb.npy")
        if "log_ppl" in need:
            feats.log_ppl = reference_log_ppl(pool, self.reference_model_path(), self.cfg,
                                              cache_dir / "pool_ref_logppl.npy")
        if "heur" in need:
            feats.heur = heuristic_features(pool)
        return feats

    def build_training_set(self, g: int, metrics: dict) -> list[str]:
        s, e = self.cfg["selection"], self.cfg["experiment"]
        gdir = self.gen_dir(g)
        n_real = int(round(s["real_frac"] * self.n))
        n_syn = self.n - n_real
        filtered_path = gdir / "filtered.jsonl"
        if filtered_path.exists():
            synthetic = read_jsonl(filtered_path)
            metrics["selection"] = read_json(gdir / "selection.json")
        else:
            pool = read_jsonl(self.pool_path(g))
            need = requirements(s["method"], self.cfg["filter"])
            ref = self.reference()
            with timer(metrics, "time_features"):
                feats = self.pool_features(g, pool, need)
            rng = np.random.default_rng(self.seed * 1000 + g)
            with timer(metrics, "time_select"):
                idx, stats = select(s["method"], len(pool), feats, Bounds(ref["ppl"], ref["heur"]),
                                    n_syn, self.cfg["filter"], rng, self.seed)
            synthetic = [pool[i] for i in idx]
            write_jsonl(filtered_path, synthetic)
            stats.update({k: metrics[k] for k in ("time_features", "time_select")})
            write_json(gdir / "selection.json", stats)
            metrics["selection"] = stats

        # Real data core: always the same first n_real D0 documents.
        train = self.real.train[:n_real] + synthetic
        if e["regime"] == "accumulate":
            # Gerstgrasser et al. (2024): D0 plus every earlier synthetic set.
            train = list(self.real.train)
            for k in range(1, g + 1):
                train += read_jsonl(self.gen_dir(k) / "filtered.jsonl")
        elif e["regime"] != "replace":
            raise ValueError(f"Unknown regime {e['regime']!r}")
        metrics["train_real_docs"] = len(self.real.train) if e["regime"] == "accumulate" else n_real
        metrics["selected_synthetic"] = len(synthetic)
        return train

    # ------------------------------------------------------------- evaluation
    def corpus_eval(self, texts: list[str], prefix: str, emb_cache: Path | None, metrics: dict) -> None:
        f, ev = self.cfg["filter"], self.cfg["eval"]
        rng = np.random.default_rng(self.seed)
        for k, v in corpus_metrics(texts, self.real_counts, self.rare, ev, rng).items():
            metrics[f"{prefix}{k}"] = v
        feats = heuristic_features(texts)
        for k in ("n_words", "rep3", "gzip_ratio", "nonalpha_ratio"):
            metrics[f"{prefix}{k}_mean"] = float(np.mean(feats[k]))
        emb = embed(texts, f["embedding_model"], f["embedding_batch_size"], emb_cache)
        for k, v in embedding_metrics(emb, self.real_test_emb).items():
            metrics[f"{prefix}{k}"] = v
        if ev["mauve"] and prefix == "":
            metrics["mauve"] = mauve_score(emb, self.real_test_emb, self.seed)

    def finish(self, g: int, metrics: dict, train_texts: list[str]) -> None:
        gdir = self.gen_dir(g)
        samples = read_jsonl(gdir / "samples.jsonl")
        with timer(metrics, "time_eval"):
            self.corpus_eval(samples, "", gdir / "samples_emb.npy", metrics)
            data = train_texts if g == 0 else read_jsonl(gdir / "filtered.jsonl")
            self.corpus_eval(data, "data_", None, metrics)
        metrics.update({"generation": g, "seed": self.seed, "line": self.cfg["experiment"]["line"],
                        "git": git_hash(), "host": platform.node()})
        write_json(gdir / "metrics.json", metrics)
        log.info("gen %d done: test_ppl=%.3f distinct_2=%.3f vendi=%.2f rare_recall=%.3f",
                 g, metrics["test_ppl"], metrics["distinct_2"], metrics["vendi"], metrics["rare_recall"])

    # ------------------------------------------------------------------ main
    def gen0(self) -> None:
        gdir = self.gen_dir(0)
        if (gdir / "metrics.json").exists() and self.pool_path(1).exists():
            return
        log.info("[seed %d] gen 0: fine-tuning on real data", self.seed)
        metrics: dict = {}
        self.train_eval_generate(0, self.real.train, metrics)
        self.reference()  # calibrate while the gen-0 model is fresh on disk
        self.finish(0, metrics, self.real.train)

    def shared_signature(self) -> dict:
        """Config sections that determine the shared gen-0 model, pool and reference."""
        c = _without_perf_keys(self.cfg)
        return {k: c[k] for k in ("data", "model", "train", "generate", "eval")} | {
            "filter": {k: c["filter"][k] for k in ("embedding_model", "reference_model", "bounds")}}

    def line_signature(self) -> dict:
        sig = _without_perf_keys(self.cfg)
        sig["experiment"].pop("seeds", None)
        return sig

    def run(self) -> None:
        check_signature(self.shared / "shared_config.json", self.shared_signature())
        check_signature(self.line_dir / "config.json", self.line_signature())
        self.gen0()
        for g in range(1, self.cfg["experiment"]["generations"] + 1):
            gdir = self.gen_dir(g)
            if (gdir / "metrics.json").exists():
                log.info("[%s seed %d] gen %d already done", self.cfg["experiment"]["line"], self.seed, g)
                continue
            log.info("[%s seed %d] gen %d", self.cfg["experiment"]["line"], self.seed, g)
            set_seed(self.seed * 1000 + g)
            metrics: dict = {}
            train = self.build_training_set(g, metrics)
            self.train_eval_generate(g, train, metrics)
            self.finish(g, metrics, train)


# Throughput-only settings: changing them must not block resuming a run.
PERF_KEYS = (("generate", "batch_size"), ("filter", "embedding_batch_size"), ("filter", "ppl_batch_size"))


def _without_perf_keys(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    for section, key in PERF_KEYS:
        out.get(section, {}).pop(key, None)
    return out


def check_signature(path: Path, signature: dict) -> None:
    """Refuse to resume into a directory produced with a different configuration."""
    if path.exists():
        stored = read_json(path)["config"]
        if stored != json.loads(json.dumps(signature)):
            raise RuntimeError(
                f"{path.parent} was produced with a different configuration. Use a new "
                f"experiment.line / output_dir, or delete that directory to start over.")
        return
    write_json(path, {"config": signature, "git": git_hash()})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", help="override experiment.seeds")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--prepare-data", action="store_true", help="only download and split real data")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    out = resolve(cfg["experiment"]["output_dir"])
    setup_logging(out / cfg["experiment"]["line"] / "run.log")
    if args.prepare_data:
        prepare_real_data(cfg)
        return
    for seed in args.seeds if args.seeds else cfg["experiment"]["seeds"]:
        Run(copy.deepcopy(cfg), seed).run()


if __name__ == "__main__":
    main()
