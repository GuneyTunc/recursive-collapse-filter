"""Plain PyTorch fine-tuning loop (bf16 autocast, length-grouped batches)."""
from __future__ import annotations

import logging
import math

import numpy as np
import torch
from tqdm.auto import tqdm

from src.lm import DEVICE, encode_documents, load_model, load_tokenizer, pad_batch
from src.utils import free_gpu

log = logging.getLogger("mc")


def length_grouped_batches(lengths: list[int], batch_size: int, rng: np.random.Generator,
                           group: int = 50) -> list[np.ndarray]:
    """Shuffle, sort inside mega-batches by length, then shuffle the batches."""
    order = rng.permutation(len(lengths))
    batches = []
    mega = batch_size * group
    for start in range(0, len(order), mega):
        chunk = order[start: start + mega]
        chunk = chunk[np.argsort([lengths[i] for i in chunk], kind="stable")]
        batches.extend(chunk[i: i + batch_size] for i in range(0, len(chunk), batch_size))
    rng.shuffle(batches)
    return batches


def finetune(cfg: dict, texts: list[str], seed: int):
    """Fine-tune a fresh copy of ``model.base`` on ``texts``; returns (model, tokenizer, stats)."""
    t, m = cfg["train"], cfg["model"]
    torch.manual_seed(seed)
    tok = load_tokenizer(m["base"])
    model = load_model(m["base"])
    model.config.use_cache = False
    if t.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    model.train()

    seqs = encode_documents(tok, texts, m["max_length"])
    rng = np.random.default_rng(seed)
    batches_per_epoch = math.ceil(len(seqs) / t["batch_size"])
    total_steps = math.ceil(batches_per_epoch * t["epochs"] / t["grad_accum"])
    warmup = max(1, int(total_steps * t["warmup_ratio"]))

    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=t["lr"], betas=(0.9, 0.95),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    )
    use_bf16 = t["bf16"] and DEVICE == "cuda"

    step, losses = 0, []
    bar = tqdm(total=total_steps, desc="train", leave=False)
    for epoch in range(math.ceil(t["epochs"])):
        batches = length_grouped_batches([len(s) for s in seqs], t["batch_size"], rng)
        if epoch + 1 > t["epochs"]:  # fractional last epoch
            batches = batches[: int(len(batches) * (t["epochs"] - epoch))]
        for b_i, idx in enumerate(batches):
            ids, mask = pad_batch([seqs[i] for i in idx], tok.pad_token_id)
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
            labels = ids.masked_fill(mask == 0, -100)  # pad == eos for SmolLM, so mask by attention
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            (loss / t["grad_accum"]).backward()
            losses.append(loss.item())
            if (b_i + 1) % t["grad_accum"] == 0 or b_i + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                bar.update(1)
                bar.set_postfix(loss=f"{np.mean(losses[-50:]):.3f}")
    bar.close()
    if t.get("gradient_checkpointing"):
        model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True
    # Release optimizer state and the allocator cache before generation: reserved-but-unused
    # memory otherwise pushes sampling into WDDM shared memory (~28x slower).
    del opt, sched
    free_gpu()
    stats = {"train_steps": step, "train_loss_last": float(np.mean(losses[-50:])),
             "train_docs": len(seqs), "train_tokens": int(sum(len(s) for s in seqs))}
    log.info("Fine-tuned on %d docs, %d steps, final loss %.3f", len(seqs), step, stats["train_loss_last"])
    return model, tok, stats
