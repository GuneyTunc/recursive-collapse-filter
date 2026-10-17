"""Real (generation-0) data: sampling, cleaning and caching the train/val/test splits.

The splits depend only on ``data.split_seed``, so every experiment seed and every
line shares exactly the same D0 and the same held-out test set.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils import read_json, read_jsonl, resolve, write_json, write_jsonl

log = logging.getLogger("mc")

# Mojibake such as "â€™" appears in a small share of TinyStories.
_BAD_CHARS = re.compile(r"[âÃ€™�]")


@dataclass
class RealData:
    train: list[str]
    val: list[str]
    test: list[str]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_usable(text: str, min_words: int, max_words: int) -> bool:
    if _BAD_CHARS.search(text):
        return False
    n = len(text.split())
    return min_words <= n <= max_words


def _split_paths(data_dir: Path) -> dict[str, Path]:
    return {name: data_dir / f"{name}.jsonl" for name in ("train", "val", "test")}


def prepare_real_data(cfg: dict) -> RealData:
    d = cfg["data"]
    data_dir = resolve(d["data_dir"])
    paths = _split_paths(data_dir)
    meta_path = data_dir / "meta.json"
    wanted = {k: d[k] for k in ("dataset", "split", "split_seed", "scan_limit", "min_words",
                                "max_words", "n_train", "n_val", "n_test")}
    if all(p.exists() for p in paths.values()) and meta_path.exists() and read_json(meta_path) == wanted:
        return load_real_data(cfg)

    from datasets import load_dataset

    log.info("Streaming %d stories from %s ...", d["scan_limit"], d["dataset"])
    stream = load_dataset(d["dataset"], split=d["split"], streaming=True)
    texts: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(stream):
        if i >= d["scan_limit"]:
            break
        t = clean(row["text"])
        if t not in seen and is_usable(t, d["min_words"], d["max_words"]):
            seen.add(t)
            texts.append(t)

    need = d["n_train"] + d["n_val"] + d["n_test"]
    if len(texts) < need:
        raise RuntimeError(f"Only {len(texts)} usable stories, need {need}; raise data.scan_limit.")
    rng = np.random.default_rng(d["split_seed"])
    order = rng.permutation(len(texts))[:need]
    picked = [texts[i] for i in order]
    real = RealData(
        train=picked[: d["n_train"]],
        val=picked[d["n_train"]: d["n_train"] + d["n_val"]],
        test=picked[d["n_train"] + d["n_val"]:],
    )
    for name, p in paths.items():
        write_jsonl(p, getattr(real, name))
    write_json(meta_path, wanted)
    log.info("Real data cached in %s (train=%d, val=%d, test=%d)", data_dir,
             len(real.train), len(real.val), len(real.test))
    return real


def load_real_data(cfg: dict) -> RealData:
    paths = _split_paths(resolve(cfg["data"]["data_dir"]))
    return RealData(**{name: read_jsonl(p) for name, p in paths.items()})
