"""Sample traces for every information set of a game and save a TraceSet.

think:   n thinking traces per information set, then the action distribution after a forced `Action:`;
         with --budgets, also the action distribution after truncating each trace at every budget.
nothink: one deterministic pass per information set with thinking disabled (zero thinking tokens).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tanking.data import VARIANTS, TraceSet, keys_of
from tanking.games import Tree
from tanking.llm import Engine
from tanking.prompts import action_words, user_message


def padded(p: np.ndarray, width: int) -> np.ndarray:
    out = np.full(width, np.nan)
    out[: len(p)] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("think", "nothink"), default="think")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-think", type=int, default=8192)
    ap.add_argument("--budgets", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    v = VARIANTS[a.game]
    tree = Tree(v)
    keys = keys_of(tree)
    width = max(len(x) for x in tree.inf_legal)
    words = [action_words(v, b) for b in tree.base]
    eng = Engine(a.model, max_model_len=a.max_think + 1024, seed=a.seed)
    think = a.mode == "think"
    prompts = [eng.prompt(user_message(v, b), think) for b in tree.base]
    budgets = tuple(int(x) for x in a.budgets.split(",") if x) if think else ()

    t0 = time.time()
    if think:
        traces = eng.think(prompts, a.n, a.max_think, a.seed)
        inf = np.repeat(np.arange(len(prompts)), a.n)
        flat = [t for ts in traces for t in ts]
        ids = [t[0] for t in flat]
        closed = np.array([t[1] for t in flat])
    else:
        inf = np.arange(len(prompts))
        ids = [[] for _ in prompts]
        closed = np.ones(len(prompts), dtype=bool)
    gen_s = time.time() - t0

    t0 = time.time()
    if think:
        prefixes = [eng.answer_prefix(prompts[i], t, c) for i, t, c in zip(inf, ids, closed)]
    else:
        prefixes = [p + eng.answer_ids for p in prompts]
    ans = eng.answers(prefixes, [words[i] for i in inf])
    probs = np.stack([padded(x.probs, width) for x in ans])
    mass = np.array([x.legal_mass for x in ans])
    tokens = np.array([len(t) for t in ids])

    budget_probs = None
    if budgets:
        budget_probs = np.repeat(probs[:, None, :], len(budgets), axis=1)
        for bi, b in enumerate(budgets):
            cut = np.flatnonzero(tokens > b)
            if len(cut):
                res = eng.answers(
                    [eng.answer_prefix(prompts[inf[j]], ids[j][:b], False) for j in cut], [words[inf[j]] for j in cut]
                )
                budget_probs[cut, bi] = np.stack([padded(x.probs, width) for x in res])
    score_s = time.time() - t0

    meta = dict(
        n=a.n,
        max_think=a.max_think,
        seed=a.seed,
        gen_seconds=gen_s,
        score_seconds=score_s,
        generated_tokens=int(tokens.sum()),
        truncated=float(1 - closed.mean()),
        legal_mass=float(mass.mean()),
    )
    out = Path(a.out)
    TraceSet(v.name, a.model, a.mode, keys, inf, tokens, closed, probs, mass, budgets, budget_probs, meta).save(out)
    with open(out / "examples.jsonl", "w") as f:
        for i in range(len(prompts)):
            j = int(np.flatnonzero(inf == i)[0])
            f.write(
                json.dumps(
                    dict(
                        key=keys[i],
                        prompt=user_message(v, tree.base[i]),
                        thought=eng.tok.decode(ids[j]),
                        tokens=int(tokens[j]),
                        probs=dict(zip(words[i], ans[j].probs.round(4).tolist())),
                    )
                )
                + "\n"
            )
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
