"""Causal LM helpers: loading, tokenisation with document separators, perplexity."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(name_or_path: str | Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(name_or_path))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name_or_path: str | Path, dtype=torch.float32):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(name_or_path), dtype=dtype)
    return model.to(DEVICE)


def separator_id(tok) -> int:
    """Token placed before every document (BOS when it exists, otherwise EOS)."""
    return tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id


def encode_documents(tok, texts: list[str], max_length: int) -> list[list[int]]:
    """``[sep] + text + [eos]``, truncated to ``max_length`` tokens."""
    sep, eos = separator_id(tok), tok.eos_token_id
    body = tok(texts, add_special_tokens=False, truncation=True, max_length=max_length - 2)["input_ids"]
    return [[sep] + ids + [eos] for ids in body]


def pad_batch(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids, mask


@torch.inference_mode()
def token_nll(model, tok, texts: list[str], max_length: int, batch_size: int = 32,
              bf16: bool = True, desc: str = "ppl", chunk_tokens: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Per-document summed NLL and number of predicted tokens.

    The LM head is applied to ``chunk_tokens`` valid positions at a time, so the
    full (batch, length, vocab) logit tensor is never materialised: with a 49k
    vocabulary it otherwise dominates memory and overflows 8 GB cards.
    """
    model.eval()
    decoder, head = model.get_decoder(), model.get_output_embeddings()
    seqs = encode_documents(tok, texts, max_length)
    order = np.argsort([len(s) for s in seqs])  # length-sorted batches waste less padding
    nll = np.zeros(len(seqs))
    count = np.zeros(len(seqs))
    use_bf16 = bf16 and DEVICE == "cuda"
    for start in tqdm(range(0, len(seqs), batch_size), desc=desc, leave=False):
        idx = order[start: start + batch_size]
        ids, mask = pad_batch([seqs[i] for i in idx], tok.pad_token_id)
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=use_bf16):
            hidden = decoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
        valid = mask[:, 1:].bool()
        rows = valid.nonzero(as_tuple=True)[0]
        h, target = hidden[valid], ids[:, 1:][valid]
        per_tok = torch.empty(len(target), device=DEVICE)
        for c in range(0, len(target), chunk_tokens):
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=use_bf16):
                logits = head(h[c: c + chunk_tokens])
            per_tok[c: c + chunk_tokens] = torch.nn.functional.cross_entropy(
                logits.float(), target[c: c + chunk_tokens], reduction="none")
        doc_nll = torch.zeros(len(idx), device=DEVICE).index_add_(0, rows, per_tok)
        nll[idx] = doc_nll.cpu().numpy()
        count[idx] = valid.sum(1).cpu().numpy()
    return nll, count


def doc_log_ppl(model, tok, texts: list[str], max_length: int, batch_size: int = 32) -> np.ndarray:
    """Per-document log-perplexity (mean token NLL)."""
    nll, count = token_nll(model, tok, texts, max_length, batch_size, desc="doc-ppl")
    return nll / np.maximum(count, 1)


def corpus_ppl(model, tok, texts: list[str], max_length: int, batch_size: int = 32) -> float:
    """Token-weighted corpus perplexity."""
    nll, count = token_nll(model, tok, texts, max_length, batch_size, desc="test-ppl")
    return float(math.exp(nll.sum() / count.sum()))
