"""Serving benchmark on one GPU: batch throughput and single-stream latency for one policy.

Decisions are drawn from equilibrium self-play reach probabilities, so every policy is timed on the same workload.
Policies: think (full thinking up to the cap), budget (per-decision-point caps from a schedule file), nothink
(thinking off; also used for distilled checkpoints). Single-stream latency is regressed on thinking tokens to
check that observable time is linear in tokens and to price padding in milliseconds.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tanking.data import VARIANTS, decision_point
from tanking.games import Tree
from tanking.llm import Engine
from tanking.prompts import action_words, user_message
from tanking.solve import cfr_plus, reach


def workload(v, n: int, rng: np.random.Generator) -> tuple[Tree, np.ndarray, float]:
    tree = Tree(v)
    r = reach(tree, tree.edge_probs(cfr_plus(tree, 2000)))
    dec = tree.kind == 1
    w = np.zeros(tree.n_infosets)
    np.add.at(w, tree.infoset[dec], (r[0] * r[1] * r[2])[dec])
    return tree, rng.choice(tree.n_infosets, n, p=w / w.sum()), float(w.sum())


def run(eng: Engine, prompts, words, caps, think: bool, use_tqdm: bool = False):
    if think:
        tr = eng.think(prompts, 1, caps, use_tqdm=use_tqdm)
        thoughts = [t[0] for t in tr]
        prefixes = [eng.answer_prefix(p, ids, done) for p, (ids, done) in zip(prompts, thoughts)]
        tokens = [len(ids) for ids, _ in thoughts]
    else:
        prefixes = [p + eng.answer_ids for p in prompts]
        tokens = [0] * len(prompts)
    eng.answers(prefixes, words)
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--game", default="leduc")
    ap.add_argument("--mode", choices=("think", "budget", "nothink"), required=True)
    ap.add_argument("--schedule", default="")
    ap.add_argument("--max-think", type=int, default=8192)
    ap.add_argument("--n-batch", type=int, default=1000)
    ap.add_argument("--n-single", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    v = VARIANTS[a.game]
    rng = np.random.default_rng(0)
    tree, picks, decisions_per_hand = workload(v, a.n_batch, rng)
    think = a.mode != "nothink"
    eng = Engine(a.model, max_model_len=a.max_think + 1024)
    prompts = [eng.prompt(user_message(v, tree.base[i]), think) for i in picks]
    words = [action_words(v, tree.base[i]) for i in picks]
    if a.mode == "budget":
        sched = json.loads(Path(a.schedule).read_text())
        caps = [int(sched.get(decision_point(tree.base[i]), a.max_think)) for i in picks]
    else:
        caps = [a.max_think] * len(picks)

    run(eng, prompts[:32], words[:32], caps[:32], think)
    t0 = time.time()
    tokens = run(eng, prompts, words, caps, think, use_tqdm=True)
    batch_s = time.time() - t0

    lat, tok1 = [], []
    for j in range(a.n_single):
        t0 = time.time()
        tok1 += run(eng, [prompts[j]], [words[j]], [caps[j]], think)
        lat.append(time.time() - t0)
    lat, tok1 = np.array(lat) * 1000, np.array(tok1, dtype=float)
    if tok1.std() > 0:
        slope, intercept = np.polyfit(tok1, lat, 1)
        r2 = float(np.corrcoef(tok1, lat)[0, 1] ** 2)
    else:
        slope, intercept, r2 = 0.0, float(lat.mean()), float("nan")
    res = dict(
        model=a.model,
        game=a.game,
        mode=a.mode,
        schedule=a.schedule,
        n_batch=a.n_batch,
        decisions_per_s=a.n_batch / batch_s,
        thinking_tokens_per_decision=float(np.mean(tokens)),
        generated_tokens_per_s=float(np.sum(tokens)) / batch_s,
        decisions_per_hand=decisions_per_hand,
        gpu_seconds_per_1000_hands=1000 * decisions_per_hand * batch_s / a.n_batch,
        single_p50_ms=float(np.percentile(lat, 50)),
        single_p99_ms=float(np.percentile(lat, 99)),
        single_mean_tokens=float(tok1.mean()),
        ms_per_token=float(slope),
        ms_intercept=float(intercept),
        latency_r2=r2,
        single_latency_ms=lat.tolist(),
        single_tokens=tok1.tolist(),
    )
    if a.mode == "budget":
        res["padded_p50_ms"] = float(intercept + slope * np.percentile(caps[: a.n_single], 50))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: x for k, x in res.items() if not isinstance(x, list)}, indent=1))


if __name__ == "__main__":
    main()
