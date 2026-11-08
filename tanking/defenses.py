"""Defenses against the timing tell, each scored on leak, added latency and playing strength.

Padding can only add delay. The LP chooses, for every (decision point, action) group, kernels K_card(t' | t) on
64 token bins of width cap/64 (t' >= t) minimizing expected added tokens subject to each card's padded
distribution being within total variation epsilon of a shared distribution q. Reported times are bin right edges,
so epsilon = 0 makes the padded time independent of the card given what the opponent sees. Budgets truncate thinking
at a length chosen from public information only and always report the budget, so they leak nothing by construction.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from tanking.data import TraceSet, parse_key
from tanking.leak import MBB, crossfit_br, decision_weights, dp_of, sampled_action, split_halves

N_BINS = 64


def bin_width(cap: int) -> int:
    return max(1, cap // N_BINS)


def bins(tokens: np.ndarray, cap: int) -> np.ndarray:
    return np.minimum(tokens // bin_width(cap), N_BINS - 1)


def pad_constant(ts: TraceSet, cap: int) -> np.ndarray:
    return np.full(len(ts.tokens), cap)


def pad_random(ts: TraceSet, width: int, rng: np.random.Generator) -> np.ndarray:
    return ts.tokens + rng.integers(0, width + 1, len(ts.tokens))


class _Rows:
    """Sparse constraint rows accumulated as (row, column, value) triplets."""

    def __init__(self):
        self.r, self.c, self.v, self.b = [], [], [], []

    def add(self, cols, vals, rhs: float):
        row = len(self.b)
        self.r += [row] * len(cols)
        self.c += list(cols)
        self.v += list(vals)
        self.b.append(rhs)

    def matrix(self, n: int):
        return coo_matrix((self.v, (self.r, self.c)), shape=(len(self.b), n)).tocsr(), self.b


def _solve_group(p: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """p: (cards, M) time distributions, w: (cards,) weights. Returns kernels K (cards, M, M), K[c, t, t'] = P(t' | t).

    Variables: flows F[c, t, t'] = p[c, t] K[c, t, t'] for t' >= t, the shared distribution q, and slacks
    d[c, t'] >= |q_c - q|.
    """
    C, M = p.shape
    flows = [(c, t, t2) for c in range(C) for t in np.flatnonzero(p[c] > 0) for t2 in range(t, M)]
    nF = len(flows)
    q0, d0 = nF, nF + M
    n = d0 + C * M
    cost = np.zeros(n)
    into = [[[] for _ in range(M)] for _ in range(C)]
    out_of: dict[tuple[int, int], list[int]] = {}
    for i, (c, t, t2) in enumerate(flows):
        cost[i] = w[c] * (t2 - t)
        into[c][t2].append(i)
        out_of.setdefault((c, t), []).append(i)
    eq, ub = _Rows(), _Rows()
    for (c, t), idx in out_of.items():
        eq.add(idx, [1.0] * len(idx), p[c, t])
    eq.add(range(q0, q0 + M), [1.0] * M, 1.0)
    for c in range(C):
        for t2 in range(M):
            f = into[c][t2]
            for s in (1.0, -1.0):
                ub.add(f + [q0 + t2, d0 + c * M + t2], [s] * len(f) + [-s, -1.0], 0.0)
        ub.add(range(d0 + c * M, d0 + (c + 1) * M), [1.0] * M, 2 * eps)
    A_eq, b_eq = eq.matrix(n)
    A_ub, b_ub = ub.matrix(n)
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    K = np.zeros((C, M, M))
    for i, (c, t, t2) in enumerate(flows):
        K[c, t, t2] = max(res.x[i], 0.0) / p[c, t]
    for c in range(C):
        empty = np.flatnonzero(p[c] == 0)
        K[c, empty, empty] = 1.0
        K[c] /= K[c].sum(1, keepdims=True)
    return K


def fit_padding(ts: TraceSet, eps: float, cap: int, smooth: float = 1.0) -> dict:
    """Kernels per (decision point, action index) group: {group: (ranks, kernels)} fit on these traces.

    Each card's time histogram gets `smooth` pseudo-traces spread over all bins, so every bin has a kernel and
    times never seen in fitting are still padded.
    """
    M = N_BINS
    dp = dp_of(ts)
    ranks = np.array([parse_key(k).rank for k in ts.keys])[ts.inf]
    w_key = decision_weights(ts)
    n_key = np.bincount(ts.inf, minlength=len(ts.keys))
    b = bins(ts.tokens, cap)
    probs = np.nan_to_num(ts.probs)
    out = {}
    for d in np.unique(dp):
        in_d = dp == d
        for a in range(probs.shape[1]):
            pa = probs[:, a] * in_d
            if pa.sum() <= 0:
                continue
            rs = np.unique(ranks[in_d & (pa > 0)])
            hist = np.zeros((len(rs), M))
            wt, count = np.zeros(len(rs)), np.zeros(len(rs))
            for i, r in enumerate(rs):
                m = in_d & (ranks == r)
                np.add.at(hist[i], b[m], pa[m])
                count[i] = m.sum()
                key = ts.inf[m][0]
                wt[i] = w_key[key] * pa[m].sum() / n_key[key]
            keep = hist.sum(1) > 0
            rs, hist, wt, count = rs[keep], hist[keep], wt[keep], count[keep]
            hist = hist / hist.sum(1, keepdims=True) * count[:, None] + smooth / M
            hist /= hist.sum(1, keepdims=True)
            K = np.eye(M)[None].repeat(len(rs), 0) if len(rs) < 2 else _solve_group(hist, wt / wt.sum(), eps)
            out[(int(d), a)] = (rs, K)
    return out


def apply_padding(ts: TraceSet, kernels: dict, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Padded token counts for sampled-action traces (one-hot probs); groups never seen in fitting keep their bin."""
    dp, act = dp_of(ts), sampled_action(ts)
    ranks = np.array([parse_key(k).rank for k in ts.keys])[ts.inf]
    b = bins(ts.tokens, cap)
    out = b.copy()
    for j in range(len(b)):
        g = kernels.get((int(dp[j]), int(act[j])))
        if g is None:
            continue
        rs, K = g
        hit = np.flatnonzero(rs == ranks[j])
        if len(hit):
            row = K[hit[0], b[j]]
            out[j] = rng.choice(len(row), p=row)
    return np.maximum((out + 1) * bin_width(cap), ts.tokens)


def with_probs(ts: TraceSet, probs: np.ndarray) -> TraceSet:
    out = ts.subset(np.arange(len(ts.inf)))
    out.probs = probs
    return out


def budget_traces(ts: TraceSet, schedule: dict[int, int]) -> TraceSet:
    """Traces as played under a per-decision-point budget schedule (dp id -> index into ts.budgets, -1 = no cut)."""
    dp = dp_of(ts)
    bi = np.array([schedule.get(int(d), -1) for d in dp])
    probs = ts.probs.copy()
    cut = bi >= 0
    probs[cut] = ts.budget_probs[np.flatnonzero(cut), bi[cut]]
    tokens = np.where(cut, np.array(ts.budgets + (0,))[bi], ts.meta["max_think"])
    out = with_probs(ts, probs)
    out.tokens = tokens
    return out


def exploitability_mbb(ts: TraceSet, rng: np.random.Generator, n_splits: int = 2) -> float:
    """Cross-fitted, seat-averaged best-response value against the action policy (timing unobserved)."""
    vals = []
    for _ in range(n_splits):
        split = split_halves(ts, rng)
        vals += [crossfit_br(ts, split, s) for s in (0, 1)]
    return MBB * float(np.mean(vals))


def search_schedule(ts: TraceSet, lam: float, rng: np.random.Generator, passes: int = 2) -> dict[int, int]:
    """Coordinate descent over decision points minimizing exploitability (mbb/hand) + lam * reported tokens/decision."""
    dp = dp_of(ts)
    dps = [int(d) for d in np.unique(dp)]
    w_key = decision_weights(ts)
    w_dp = np.zeros(max(dps) + 1)
    first = {int(d): ts.inf[np.flatnonzero(dp == d)] for d in dps}
    for d, keys in first.items():
        w_dp[d] = w_key[np.unique(keys)].sum()
    sizes = np.array(ts.budgets + (ts.meta["max_think"],), dtype=float)
    seed = int(rng.integers(1 << 31))

    def objective(s):
        latency = sum(w_dp[d] * sizes[s.get(d, -1)] for d in dps) / w_dp.sum()
        return exploitability_mbb(budget_traces(ts, s), np.random.default_rng(seed)) + lam * latency

    sched = {d: -1 for d in dps}
    best = objective(sched)
    for _ in range(passes):
        improved = False
        for d in sorted(dps, key=lambda x: -w_dp[x]):
            for bi in list(range(len(ts.budgets))) + [-1]:
                if bi == sched[d]:
                    continue
                trial = {**sched, d: bi}
                val = objective(trial)
                if val < best - 1e-9:
                    best, sched, improved = val, trial, True
        if not improved:
            break
    return sched
