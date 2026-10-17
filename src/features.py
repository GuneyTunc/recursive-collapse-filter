"""Cached pool features: sentence embeddings, reference log-perplexity, heuristics."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.filters import heuristics
from src.utils import free_gpu

log = logging.getLogger("mc")

_EMBEDDERS: dict[str, object] = {}


def _embedder(name: str):
    if name not in _EMBEDDERS:
        from sentence_transformers import SentenceTransformer

        from src.lm import DEVICE

        _EMBEDDERS[name] = SentenceTransformer(name, device=DEVICE)
    return _EMBEDDERS[name]


def embed(texts: list[str], name: str, batch_size: int = 256, cache: Path | None = None) -> np.ndarray:
    if cache is not None and cache.exists():
        emb = np.load(cache)
        if len(emb) == len(texts):
            return emb
    emb = _embedder(name).encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
    emb = emb.astype(np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
    return emb


def reference_log_ppl(texts: list[str], ref_path: str | Path, cfg: dict,
                      cache: Path | None = None) -> np.ndarray:
    if cache is not None and cache.exists():
        v = np.load(cache)
        if len(v) == len(texts):
            return v
    from src.lm import doc_log_ppl, load_model, load_tokenizer

    tok = load_tokenizer(ref_path)
    model = load_model(ref_path)
    v = doc_log_ppl(model, tok, texts, cfg["model"]["max_length"], cfg["filter"]["ppl_batch_size"])
    del model
    free_gpu()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, v)
    return v


def heuristic_features(texts: list[str]) -> dict[str, np.ndarray]:
    return heuristics.text_features(texts)
