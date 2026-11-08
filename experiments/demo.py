"""Replay demo from saved traces: a bot that reads thinking time names the LLM's card, then padding blinds it.

The bot is a count-based Bayes classifier over (decision point, action, time bucket) fit on one half of the traces;
the demo replays decisions from the other half, drawn by how often they occur in self-play.
"""

import argparse
from pathlib import Path

import numpy as np

from tanking.data import TraceSet, parse_key
from tanking.defenses import apply_padding, fit_padding
from tanking.leak import (
    bucket_edges,
    buckets,
    decision_weights,
    default_k,
    dp_of,
    sample_actions,
    sampled_action,
    split_halves,
)
from tanking.prompts import action_words, rank_name


class Reader:
    def __init__(self, ts: TraceSet, k: int, use_time: bool, alpha: float = 0.5):
        self.k, self.use_time, self.alpha = k, use_time, alpha
        self.edges = bucket_edges(ts, k)
        self.ranks = np.array([parse_key(x).rank for x in ts.keys])
        self.n_ranks = int(self.ranks.max()) + 1
        self.counts = {}
        for d, a, b, r in zip(dp_of(ts), sampled_action(ts), self.bucket(ts), self.ranks[ts.inf]):
            c = self.counts.setdefault((int(d), int(a), int(b) if use_time else 0), np.zeros(self.n_ranks))
            c[r] += 1

    def bucket(self, ts: TraceSet) -> np.ndarray:
        return buckets(ts, self.edges)

    def posterior(self, d: int, a: int, b: int) -> np.ndarray:
        c = self.counts.get((d, a, b if self.use_time else 0), np.zeros(self.n_ranks)) + self.alpha
        return c / c.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--hands", type=int, default=8)
    ap.add_argument("--k", type=int, help="time buckets (default by game)")
    ap.add_argument("--pad", action="store_true", help="apply LP padding (epsilon = 0) to the replayed traces")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    ts = TraceSet.load(Path(a.run))
    k = a.k or default_k(ts.variant)
    fit = split_halves(ts, rng)
    train, test = sample_actions(ts.subset(fit), rng), sample_actions(ts.subset(~fit), rng)
    if a.pad:
        cap = ts.meta["max_think"]
        test = test.with_tokens(apply_padding(test, fit_padding(ts.subset(fit), 0.0, cap), cap, rng))
        train = train.with_tokens(apply_padding(train, fit_padding(ts.subset(~fit), 0.0, cap), cap, rng))
    timing, actions = Reader(train, k, True), Reader(train, k, False)
    v = ts.variant
    w = decision_weights(test)[test.inf]
    d, act, b = dp_of(test), sampled_action(test), timing.bucket(test)
    truth = timing.ranks[test.inf]
    hit_t = np.array([timing.posterior(int(x), int(y), int(z)).argmax() for x, y, z in zip(d, act, b)]) == truth
    hit_a = np.array([actions.posterior(int(x), int(y), 0).argmax() for x, y in zip(d, act)]) == truth
    print(f"{ts.model} on {v.name}{' with LP padding' if a.pad else ''}\n")
    for j in rng.choice(len(w), a.hands, p=w / w.sum()):
        key = parse_key(test.keys[test.inf[j]])
        word = action_words(v, key)[act[j]]
        post = timing.posterior(int(d[j]), int(act[j]), int(b[j]))
        guess = int(post.argmax())
        print(
            f"Player {key.player + 1}, history '{key.history or '-'}': "
            f"thought {test.tokens[j]:>5} tokens, then {word}. "
            f"Bot: 'you hold {rank_name(v, guess)}' ({post[guess]:.0%}). Truth: {rank_name(v, key.rank)}"
            f"{'  ✓' if guess == key.rank else ''}"
        )
    print(
        f"\nCard named correctly over {len(w)} replayed decisions (self-play weighted): "
        f"{np.average(hit_t, weights=w):.1%} reading time vs {np.average(hit_a, weights=w):.1%} from actions alone."
    )


if __name__ == "__main__":
    main()
