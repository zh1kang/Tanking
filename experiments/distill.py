"""Distill the CFR+ equilibrium into a Qwen3 LoRA (thinking off), then merge and save full weights for vLLM.

Every information set of every training variant is one example: the thinking-off prompt followed by `Action:`.
The loss is cross-entropy to the equilibrium over the full vocabulary at the next position, which splits into
KL(equilibrium || model restricted to legal tokens, the renormalization used when reading the policy) plus
-log(probability on legal tokens). Small variants are repeated so each contributes equally.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from tanking.data import TRAIN_VARIANTS
from tanking.games import Tree
from tanking.prompts import ANSWER_PREFIX, action_words, user_message
from tanking.solve import cfr_plus


def build_examples(tok, min_per_variant: int) -> list[tuple[list[int], list[int], np.ndarray]]:
    out = []
    for v in TRAIN_VARIANTS:
        tree = Tree(v)
        nash = cfr_plus(tree, 3000)
        answer = tok.encode(ANSWER_PREFIX, add_special_tokens=False)
        rows = []
        for i, b in enumerate(tree.base):
            text = tok.apply_chat_template(
                [{"role": "user", "content": user_message(v, b)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            words = action_words(v, b)
            legal_ids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in words]
            rows.append((tok.encode(text, add_special_tokens=False) + answer, legal_ids, nash[i, : len(words)]))
        reps = max(1, math.ceil(min_per_variant / len(rows)))
        out += rows * reps
    return out


def batch_loss(model, batch, pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean KL on the legal tokens and mean -log(legal probability mass)."""
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, _, _) in enumerate(batch):
        ids[i, L - len(x) :] = torch.tensor(x)
        mask[i, L - len(x) :] = 1
    ids, mask = ids.to(device), mask.to(device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    logp_full = torch.log_softmax(
        model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1, :].float(), -1
    )
    kls, masses = [], []
    for i, (_, legal_ids, target) in enumerate(batch):
        logp = logp_full[i, legal_ids]
        t = torch.tensor(target, dtype=torch.float32, device=device).clamp(min=0)
        t = t / t.sum()
        log_mass = torch.logsumexp(logp, -1)
        kls.append((t * (torch.log(t.clamp(min=1e-12)) - (logp - log_mass))).sum())
        masses.append(-log_mass)
    return torch.stack(kls).mean(), torch.stack(masses).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--min-per-variant", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    data = build_examples(tok, a.min_per_variant)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"
        ),
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = a.epochs * math.ceil(len(data) / a.batch)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )
    log, t0 = [], time.time()
    model.train()
    for epoch in range(a.epochs):
        order = rng.permutation(len(data))
        total = np.zeros(2)
        for s in range(0, len(order), a.batch):
            kl, nll_mass = batch_loss(model, [data[i] for i in order[s : s + a.batch]], tok.pad_token_id, device)
            opt.zero_grad()
            (kl + nll_mass).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += np.array([kl.item(), nll_mass.item()]) * len(order[s : s + a.batch])
        kl_mean, nll_mean = total / len(data)
        log.append(dict(epoch=epoch, kl=kl_mean, legal_mass=float(np.exp(-nll_mean)), seconds=time.time() - t0))
        print(json.dumps(log[-1]), flush=True)
    out = Path(a.out)
    merged = model.merge_and_unload()
    merged.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "train_log.json").write_text(
        json.dumps(dict(args=vars(a), examples=len(data), steps=steps, log=log), indent=1)
    )


if __name__ == "__main__":
    main()
