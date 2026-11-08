"""Practical exploitability: an opponent that learns the LLM's timing pattern online, from showdowns only.

The opponent keeps Dirichlet counts of the LLM's (action, time bucket) at each LLM information set (actions only
for the baseline learner) and plays a best response to the posterior-mean model, recomputed on a geometric schedule.
The LLM's card is revealed only at showdown; after a fold its card is latent and each count gets the posterior
weight of that card given the opponent's card and what it saw. Hands are drawn from one trace pool and every policy
is valued exactly against a held-out pool, so the learner cannot profit from memorizing a finite sample.

  run:     one condition (trace pools or a synthetic tell) for both learners, both seats and every run seed
  figure:  advantage of the timing learner over the actions-only learner against hands played
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.calibrate import synthetic
from tanking.data import TraceSet
from tanking.games import InfoKey, Tree
from tanking.leak import MBB, bucket_edges, buckets, default_k, policy, tree_for
from tanking.solve import best_response, combine, value

GRID = np.array([100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000])


class Sampler:
    """Draws root-to-terminal paths of a tree under per-edge probabilities, many hands at once."""

    def __init__(self, tree: Tree):
        self.tree = tree
        self.order = np.argsort(tree.e_parent, kind="stable")
        self.parent = tree.e_parent[self.order]

    def paths(self, ep: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        p = ep[self.order]
        cum = np.zeros_like(p)
        starts = np.flatnonzero(np.r_[True, self.parent[1:] != self.parent[:-1]])
        for a, b in zip(starts, np.r_[starts[1:], len(p)]):
            cum[a:b] = np.cumsum(p[a:b]) / p[a:b].sum()
            cum[b - 1] = 1.0
        keys = self.parent + cum
        node = np.zeros(n, dtype=np.int64)
        steps = []
        while True:
            live = self.tree.kind[node] != 2
            if not live.any():
                return np.stack(steps, 1)
            e = np.full(n, -1)
            u = 1.0 - rng.random(live.sum())
            e[live] = self.order[np.searchsorted(keys, node[live] + u, side="left")]
            node = np.where(live, self.tree.e_child[np.maximum(e, 0)], node)
            steps.append(e)


class Learner:
    def __init__(self, tree: Tree, seat: int, alpha: float = 1.0):
        self.tree, self.seat, self.opp = tree, seat, 1 - seat
        self.rows = tree.infosets_of(seat)
        self.shown = np.zeros(tree.mask.shape)
        cells = tree.mask.sum(1, keepdims=True)
        self.prior = np.where(tree.mask, alpha / cells, 0.0)
        index = {k: i for i, k in enumerate(tree.keys)}
        n_ranks = tree.v.n_ranks
        alt = np.full((tree.n_infosets, n_ranks), -1)
        for r in self.rows:
            b = tree.base[r]
            for c in range(n_ranks):
                alt[r, c] = index.get(str(InfoKey(b.player, c, b.board, b.history)), -1)
        self.hist_rows, self.hist = np.unique(alt[self.rows], axis=0, return_inverse=True)
        self.hist_of = np.full(tree.n_infosets, -1)
        self.hist_of[self.rows] = self.hist.ravel()
        depth = np.zeros(len(tree.kind), dtype=np.int64)
        while not np.array_equal(depth[tree.e_child], depth[tree.e_parent] + 1):
            depth[tree.e_child] = depth[tree.e_parent] + 1
        self.width = int(depth.max())
        self.hidden: dict[tuple, float] = {}
        self.S = self._normalize(self.shown)

    def _normalize(self, counts: np.ndarray) -> np.ndarray:
        S = self.tree.uniform().copy()
        c = counts[self.rows] + self.prior[self.rows]
        S[self.rows] = c / c.sum(1, keepdims=True)
        return S

    def _expected(self, S: np.ndarray, pat: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Expected counts of the unrevealed hands under S: each pattern spread over the LLM's possible cards."""
        v = self.tree.v
        opp, board, hid, loc = pat[:, 0], pat[:, 1], pat[:, 2::2], pat[:, 3::2]
        valid = hid >= 0
        prior = np.full((len(pat), v.n_ranks), float(v.copies))
        prior[np.arange(len(pat)), opp] -= 1
        has_board = board >= 0
        prior[np.flatnonzero(has_board), board[has_board]] -= 1
        logp = np.log(np.maximum(prior, 0.0) + 1e-300)
        rows = self.hist_rows[np.maximum(hid, 0)]
        for c in range(v.n_ranks):
            p = np.where(rows[..., c] >= 0, S[np.maximum(rows[..., c], 0), np.maximum(loc, 0)], 0.0)
            logp[:, c] += np.log(np.where(valid, p, 1.0) + 1e-300).sum(1)
        w = np.exp(logp - logp.max(1, keepdims=True))
        w *= (n / w.sum(1))[:, None]
        E = np.zeros_like(S)
        for c in range(v.n_ranks):
            keep = valid & (rows[..., c] >= 0)
            np.add.at(E, (rows[..., c][keep], loc[keep]), np.broadcast_to(w[:, c:c + 1], keep.shape)[keep])
        return E

    def model(self, iters: int = 10) -> np.ndarray:
        """Posterior-mean strategy; EM over the unrevealed hands, warm-started from the previous model."""
        if self.hidden:
            pat, n = np.array(list(self.hidden)), np.array(list(self.hidden.values()))
            for _ in range(iters):
                self.S = self._normalize(self.shown + self._expected(self.S, pat, n))
        else:
            self.S = self._normalize(self.shown)
        return self.S

    def policy(self, eps: float) -> np.ndarray:
        br, _ = best_response(self.tree, self.model(), self.opp)
        rows = self.tree.infosets_of(self.opp)
        br[rows] = (1 - eps) * br[rows] + eps * self.tree.uniform()[rows]
        return br

    def update(self, paths: np.ndarray):
        """Count showdown hands exactly; keep folded hands as (opponent card, board, public LLM actions) patterns."""
        t = self.tree
        e = np.where(paths >= 0, paths, 0)
        valid = paths >= 0
        last = e[np.arange(len(e)), valid.sum(1) - 1]
        shown = np.array([a != "f" for a in np.asarray(t.e_action)[last]])
        dec = valid & ~t.e_is_chance[e]
        mine = dec & (t.e_player[e] == self.seat)
        theirs = dec & (t.e_player[e] == self.opp)
        assert theirs.any(1).all(), "every hand has an opponent decision"
        last_opp = theirs.shape[1] - 1 - np.argmax(theirs[:, ::-1], axis=1)
        opp_row = t.e_infoset[e[np.arange(len(e)), last_opp]]
        h, j = np.nonzero(mine)
        rows, local = t.e_infoset[e[h, j]], t.e_local[e[h, j]]
        s = shown[h]
        np.add.at(self.shown, (rows[s], local[s]), 1.0)
        hide = ~shown
        if not hide.any():
            return
        pat = np.full((len(e), 2 + 2 * self.width), -1)
        pat[:, 0] = [t.base[r].rank for r in opp_row]
        pat[:, 1] = [t.base[r].board for r in opp_row]
        slot = np.cumsum(mine, axis=1)[h, j] - 1
        pat[h, 2 + 2 * slot], pat[h, 3 + 2 * slot] = self.hist_of[rows], local
        keys, n = np.unique(pat[hide], axis=0, return_counts=True)
        for key, m in zip(map(tuple, keys), n):
            self.hidden[key] = self.hidden.get(key, 0.0) + m


def simulate(train: TraceSet, test: TraceSet, k: int, seat: int, eps: float, hands: int, seed: int) -> dict:
    """One learner in one seat: per-hand value (mbb) of the policy in force over each segment of hands."""
    rng = np.random.default_rng(seed)
    tree = tree_for(train.variant, seat, k) if k > 1 else tree_for(train.variant)
    if k > 1:
        edges = bucket_edges(train, k)
        S_sim, S_eval = policy(tree, train, buckets(train, edges)), policy(tree, test, buckets(test, edges))
    else:
        S_sim, S_eval = policy(tree, train), policy(tree, test)
    learner, sampler = Learner(tree, seat), Sampler(tree)
    start, seg = [], []
    done = 0
    while done < hands:
        pol = learner.policy(eps)
        n = min(max(1, done // 20), hands - done)
        start.append(done)
        seg.append(MBB * value(tree, combine(tree, S_eval, pol, 1 - seat), 1 - seat))
        learner.update(sampler.paths(tree.edge_probs(combine(tree, S_sim, pol, 1 - seat)), n, rng))
        done += n
    br, _ = best_response(tree, S_sim, 1 - seat)
    asym = MBB * value(tree, combine(tree, S_eval, br, 1 - seat), 1 - seat)
    return dict(start=start, value=seg, hands=hands, asymptote=asym)


def at(run: dict, h: np.ndarray) -> np.ndarray:
    """Per-hand value of the policy in force when the h-th hand is played."""
    return np.asarray(run["value"])[np.searchsorted(run["start"], h, side="right") - 1]


def cumulative(run: dict, h: np.ndarray) -> np.ndarray:
    s, v = np.asarray(run["start"]), np.asarray(run["value"])
    ends = np.r_[s[1:], run["hands"]]
    return np.array([np.sum(v * np.clip(np.minimum(ends, x) - s, 0, None)) for x in h])


def interval(x: np.ndarray) -> list[list[float]]:
    return np.percentile(x, [2.5, 97.5], axis=0).T.tolist()


def hands_to_profit(grid: np.ndarray, lower: np.ndarray) -> int | None:
    """First grid point from which the cumulative advantage's lower 95% bound stays above zero."""
    for i, h in enumerate(grid):
        if (lower[i:] > 0).all():
            return int(h)
    return None


def pools(a) -> list[tuple[TraceSet, TraceSet]]:
    A, B = TraceSet.load(Path(a.pool_a)), TraceSet.load(Path(a.pool_b))
    if a.delta is None:
        return [(A, B), (B, A)]
    both = TraceSet.concat([A, B])
    rng = np.random.default_rng(a.seed)
    out = []
    for _ in range(2):
        x, y = synthetic(both, a.n, a.delta, rng), synthetic(both, a.n, a.delta, rng)
        out.append((x, y))
    return out


def run(a) -> None:
    per_seat = a.hands // 2
    grid = GRID[GRID <= a.hands]
    res = {}
    for eps in (0.0, 0.05):
        adv, cum, asym = [], [], []
        for p, (train, test) in enumerate(pools(a)):
            k = default_k(train.variant)
            for r in range(a.runs):
                seed = 1000 * p + r
                curves = {
                    name: [simulate(train, test, kk, s, eps, per_seat, seed + 97 * s) for s in (0, 1)]
                    for name, kk in (("timing", k), ("actions", 1))
                }
                v = {n: np.mean([at(c, grid // 2) for c in cs], 0) for n, cs in curves.items()}
                cu = {n: np.sum([cumulative(c, grid // 2) for c in cs], 0) / MBB for n, cs in curves.items()}
                adv.append(v["timing"] - v["actions"])
                cum.append(cu["timing"] - cu["actions"])
                asym.append(
                    np.mean([c["asymptote"] for c in curves["timing"]])
                    - np.mean([c["asymptote"] for c in curves["actions"]])
                )
            print(json.dumps(dict(eps=eps, pool=p, adv_last=float(np.mean(adv, 0)[-1]))), flush=True)
        adv, cum = np.array(adv), np.array(cum)
        res[str(eps)] = dict(
            hands=grid.tolist(),
            advantage_mbb=np.mean(adv, 0).tolist(),
            advantage_ci=interval(adv),
            cumulative_chips=np.mean(cum, 0).tolist(),
            cumulative_ci=interval(cum),
            hands_to_profit=hands_to_profit(grid, np.percentile(cum, 2.5, axis=0)),
            asymptote_mbb=float(np.mean(asym)),
        )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(pool_a=a.pool_a, pool_b=a.pool_b, delta=a.delta, runs=a.runs, hands=a.hands)
    out.write_text(json.dumps(meta | dict(results=res), indent=1))


def figure(a) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for path, label in zip(a.files, a.labels):
        r = json.loads(Path(path).read_text())["results"]["0.05"]
        h, m, ci = np.array(r["hands"]), np.array(r["advantage_mbb"]), np.array(r["advantage_ci"])
        (line,) = ax.plot(h, m, marker="o", ms=3, label=label)
        ax.fill_between(h, ci[:, 0], ci[:, 1], color=line.get_color(), alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("hands played")
    ax.set_ylabel("timing learner − actions learner\n(mbb/hand, held-out pool)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("pool_a")
    r.add_argument("pool_b")
    r.add_argument("--delta", type=float, help="replace thinking times with a synthetic tell of this size")
    r.add_argument("--n", type=int, default=512, help="traces per information set in each synthetic pool")
    r.add_argument("--hands", type=int, default=1_000_000)
    r.add_argument("--runs", type=int, default=10, help="run seeds per pool direction")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    f = sub.add_parser("figure")
    f.add_argument("--files", nargs="+", required=True)
    f.add_argument("--labels", nargs="+", required=True)
    f.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        figure(a)


if __name__ == "__main__":
    main()
