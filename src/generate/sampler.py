"""Synthetic data generation from a fine-tuned model."""
from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from src.lm import DEVICE, autocast, separator_id


def _prompts(cfg: dict, n: int, real_texts: list[str] | None, rng: np.random.Generator) -> list[str]:
    g = cfg["generate"]
    if g["prompt_mode"] == "bos":
        return [""] * n
    if g["prompt_mode"] == "real_prefix":
        if not real_texts:
            raise ValueError("prompt_mode=real_prefix needs real texts")
        picks = rng.integers(0, len(real_texts), size=n)
        return [" ".join(real_texts[i].split()[: g["prompt_words"]]) for i in picks]
    raise ValueError(f"Unknown prompt_mode {g['prompt_mode']!r}")


@torch.inference_mode()
def generate_texts(model, tok, cfg: dict, n: int, seed: int,
                   real_texts: list[str] | None = None) -> list[str]:
    """Sample ``n`` documents. Defaults to pure ancestral sampling (T=1, top-p=1)."""
    g = cfg["generate"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.eval()
    sep = separator_id(tok)
    prompts = _prompts(cfg, n, real_texts, rng)
    tok.padding_side = "left"
    out: list[str] = []
    for start in tqdm(range(0, n, g["batch_size"]), desc="generate", leave=False):
        batch = prompts[start: start + g["batch_size"]]
        enc = [[sep] + tok(p, add_special_tokens=False)["input_ids"] for p in batch]
        width = max(len(e) for e in enc)
        ids = torch.full((len(enc), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, e in enumerate(enc):
            ids[i, width - len(e):] = torch.tensor(e)
            mask[i, width - len(e):] = 1
        with autocast(cfg["model"]["precision"]):
            gen = model.generate(
                input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                do_sample=True, temperature=g["temperature"], top_p=g["top_p"], top_k=0,
                max_new_tokens=g["max_new_tokens"],
                use_cache=True,  # finetune() disables the KV cache; without it sampling is ~17x slower
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        # Prompt words are kept: they are part of the document the next generation trains on.
        # Padding, separator and EOS are special tokens and vanish on decode.
        out.extend(tok.decode(row, skip_special_tokens=True).strip() for row in gen.tolist())
    tok.padding_side = "right"
    return out
