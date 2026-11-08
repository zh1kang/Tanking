"""Pricing the tell: cross-fitted best responses against the LLM's policy, with and without its thinking time.

The LLM plays both seats. Its policy at an information set is the mean action distribution of the traces there.
An opponent that also observes thinking time sees, for each LLM move, the action plus a quantile bucket of the
thinking tokens (buckets fit per decision point). Every best response is computed on one half of the traces and
evaluated against the other half, so finite-sample noise cannot be exploited and the estimates are conservative.
"""

from functools import cache

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from tanking.data import TraceSet, decision_point, parse_key
from tanking.games import Tree, Variant
from tanking.solve import best_response, combine, reach, value

MBB = 1000.0


def default_k(v: Variant) -> int:
    """Time buckets per decision point (pre-registered): 4 for one-round games, 3 for two-round games."""
    return 4 if v.rounds == 1 else 3


@cache
def tree_for(v: Variant, seat: int | None = None, k: int = 1, reveal: bool = False) -> Tree:
    return Tree(v, signal_seat=seat, k=k, reveal=reveal)


@cache
def dp_layout(keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per key: its decision-point id and LLM seat; plus the decision-point names."""
    dps = [decision_point(parse_key(k)) for k in keys]
    names = sorted(set(dps))
    ids = {d: i for i, d in enumerate(names)}
    return np.array([ids[d] for d in dps]), np.array([parse_key(k).player for k in keys]), names


def dp_of(ts: TraceSet) -> np.ndarray:
    return dp_layout(tuple(ts.keys))[0][ts.inf]


def bucket_edges(ts: TraceSet, k: int) -> dict[int, np.ndarray]:
    dp = dp_of(ts)
    qs = np.arange(1, k) / k
    return {d: np.quantile(ts.tokens[dp == d], qs) for d in np.unique(dp)}


def buckets(ts: TraceSet, edges: dict[int, np.ndarray]) -> np.ndarray:
    dp = dp_of(ts)
    out = np.zeros(len(dp), dtype=np.int64)
    for d, e in edges.items():
        m = dp == d
        out[m] = np.searchsorted(e, ts.tokens[m], side="right")
    return out


def _rows(tree: Tree, keys: tuple[str, ...]) -> np.ndarray:
    index = {key: i for i, key in enumerate(tree.keys)}
    return np.array([index.get(k, -1) for k in keys])


def policy(tree: Tree, ts: TraceSet, bucket: np.ndarray | None = None, beta: float = 1.0) -> np.ndarray:
    """Strategy matrix holding the LLM's empirical policy on every information set the traces cover.

    With buckets (tree.k > 1) the LLM's rows are joint (action, bucket) probabilities; the bucket distribution given
    the action is shrunk toward the decision point's pooled bucket frequencies with weight `beta` traces.
    """
    k = tree.k if bucket is not None else 1
    rows = _rows(tree, tuple(ts.keys))[ts.inf]
    keep = rows >= 0
    rows, probs = rows[keep], np.nan_to_num(ts.probs[keep])
    n_rows = tree.n_infosets
    marg = np.zeros((n_rows, ts.probs.shape[1]))
    np.add.at(marg, rows, probs)
    count = np.bincount(rows, minlength=n_rows).astype(float)
    S = tree.uniform().copy()
    have = count > 0
    if k == 1:
        n = marg.shape[1]
        S[have, :n] = marg[have] / count[have, None]
        return S
    b = bucket[keep]
    joint = np.zeros((n_rows, marg.shape[1], k))
    np.add.at(joint, (rows, slice(None), b), probs)
    dp = dp_of(ts)[keep]
    pool = np.zeros((dp.max() + 1, k))
    np.add.at(pool, (dp, b), 1.0)
    pool /= np.maximum(pool.sum(1, keepdims=True), 1.0)
    row_dp = np.zeros(n_rows, dtype=np.int64)
    row_dp[rows] = dp
    cond = (joint + beta * pool[row_dp][:, None, :]) / (marg[:, :, None] + beta)
    joint_p = (marg / np.maximum(count, 1.0)[:, None])[:, :, None] * cond
    n = marg.shape[1] * k
    S[have, :n] = joint_p.reshape(n_rows, -1)[have]
    return S


def split_halves(ts: TraceSet, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask putting half of each information set's traces in the first half."""
    mask = np.zeros(len(ts.inf), dtype=bool)
    for i in np.unique(ts.inf):
        idx = rng.permutation(np.flatnonzero(ts.inf == i))
        mask[idx[: len(idx) // 2]] = True
    return mask


def resample(ts: TraceSet, rng: np.random.Generator) -> TraceSet:
    idx = [rng.choice(g, len(g)) for g in (np.flatnonzero(ts.inf == i) for i in np.unique(ts.inf))]
    return ts.subset(np.concatenate(idx))


def crossfit_br(ts: TraceSet, split: np.ndarray, seat: int, k: int = 1, reveal: bool = False) -> float:
    """Opponent's value (chips/hand) from a best response fit on one half and played against the other, both folds."""
    tree = tree_for(ts.variant, seat, k, reveal)
    vals = []
    for fit in (split, ~split):
        a, b = ts.subset(fit), ts.subset(~fit)
        if k > 1:
            e = bucket_edges(a, k)
            sa, sb = policy(tree, a, buckets(a, e)), policy(tree, b, buckets(b, e))
        else:
            sa, sb = policy(tree, a), policy(tree, b)
        br, _ = best_response(tree, sa, 1 - seat)
        vals.append(value(tree, combine(tree, sb, br, 1 - seat), 1 - seat))
    return float(np.mean(vals))


def br_values(ts: TraceSet, split: np.ndarray, k: int) -> dict[str, float]:
    """Seat-averaged opponent value in mbb/hand when it sees actions, actions plus timing, or the LLM's card."""
    out = {}
    for name, kk, rev in (("actions", 1, False), ("timing", k, False), ("card", 1, True)):
        out[name] = MBB * float(np.mean([crossfit_br(ts, split, s, kk, rev) for s in (0, 1)]))
    out["leak"] = out["timing"] - out["actions"]
    return out


def measure(ts: TraceSet, k: int, n_splits: int = 10, n_boot: int = 200, seed: int = 0) -> dict:
    """Point estimates averaged over random splits, with percentile bootstrap CIs (traces resampled within infosets)."""
    rng = np.random.default_rng(seed)
    pts = [br_values(ts, split_halves(ts, rng), k) for _ in range(n_splits)]
    point = {key: float(np.mean([p[key] for p in pts])) for key in pts[0]}
    boots = []
    for _ in range(n_boot):
        b = resample(ts, rng)
        boots.append(br_values(b, split_halves(b, rng), k))
    ci = {key: [float(np.percentile([x[key] for x in boots], q)) for q in (2.5, 97.5)] for key in point}
    return dict(point=point, ci=ci, k=k, n_traces=int(len(ts.inf)), n_splits=n_splits, n_boot=n_boot)


def sample_actions(ts: TraceSet, rng: np.random.Generator) -> TraceSet:
    """One sampled action per trace (one-hot probs), as an observer would actually see it."""
    p = np.nan_to_num(ts.probs)
    u = rng.random(len(p))[:, None]
    a = (p.cumsum(1) < u).sum(1).clip(max=p.shape[1] - 1)
    probs = np.where(np.isnan(ts.probs), np.nan, 0.0)
    probs[np.arange(len(p)), a] = 1.0
    out = ts.subset(np.arange(len(p)))
    out.probs = probs
    return out


def sampled_action(ts: TraceSet) -> np.ndarray:
    return np.nan_to_num(ts.probs).argmax(1)


def shuffle_within(ts: TraceSet, rng: np.random.Generator) -> TraceSet:
    """Null control on sampled-action traces: permute thinking tokens within (decision point, action) across cards."""
    group = dp_of(ts) * 8 + sampled_action(ts)
    tokens = ts.tokens.copy()
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        tokens[idx] = tokens[rng.permutation(idx)]
    return ts.with_tokens(tokens)


def tell_bits(ts: TraceSet, seed: int = 0, folds: int = 5) -> dict:
    """Held-out information (bits per decision) that thinking time adds about the LLM's card, beyond decision point
    and sampled action, from gradient-boosted classifiers with and without the token count."""
    rng = np.random.default_rng(seed)
    s = sample_actions(ts, rng)
    y = np.array([parse_key(ts.keys[i]).rank for i in s.inf])
    base = np.column_stack([dp_of(s), sampled_action(s)]).astype(float)
    full = np.column_stack([base, np.log1p(s.tokens)])
    losses = {}
    for name, X in (("without", base), ("with", full)):
        pred = np.zeros((len(y), y.max() + 1))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
            clf = HistGradientBoostingClassifier(categorical_features=[0, 1], early_stopping=True, random_state=seed)
            clf.fit(X[tr], y[tr])
            pred[np.ix_(te, clf.classes_)] = clf.predict_proba(X[te])
        losses[name] = log_loss(y, np.clip(pred, 1e-9, 1), labels=np.arange(y.max() + 1)) / np.log(2)
    return dict(
        bits_without=losses["without"], bits_with=losses["with"], bits_gained=losses["without"] - losses["with"]
    )


def decision_weights(ts: TraceSet) -> np.ndarray:
    """Per key, the probability the decision occurs in a hand of LLM self-play (sums to decisions per hand)."""
    tree = tree_for(ts.variant)
    S = policy(tree, ts)
    r = reach(tree, tree.edge_probs(S))
    node_p = r[0] * r[1] * r[2]
    dec = tree.kind == 1
    w = np.zeros(tree.n_infosets)
    np.add.at(w, tree.infoset[dec], node_p[dec])
    return w[_rows(tree, tuple(ts.keys))]


def tokens_per_decision(ts: TraceSet, tokens: np.ndarray | None = None) -> float:
    """Self-play-weighted mean thinking tokens per decision."""
    t = ts.tokens if tokens is None else tokens
    w = decision_weights(ts)
    per_key = np.bincount(ts.inf, weights=t, minlength=len(ts.keys)) / np.maximum(
        np.bincount(ts.inf, minlength=len(ts.keys)), 1
    )
    return float((w * per_key).sum() / w.sum())
