"""Exact solving on a flattened Tree: reach probabilities, expected values, CFR+ equilibria and best responses.

Strategies are (n_infosets, max_actions) matrices; `Tree.edge_probs` turns one into per-edge probabilities.
Values are chips per hand for player 0 unless stated otherwise.
"""

import numpy as np

from tanking.games import Tree


def reach(tree: Tree, ep: np.ndarray) -> np.ndarray:
    """Rows: player 0's, player 1's and chance's contribution to the probability of reaching each node."""
    r = np.ones((3, len(tree.kind)))
    owner = np.where(tree.e_is_chance, 2, tree.e_player)
    for lv in tree.levels:
        par, ch = tree.e_parent[lv], tree.e_child[lv]
        r[:, ch] = r[:, par]
        r[owner[lv], ch] *= ep[lv]
    return r


def values(tree: Tree, ep: np.ndarray, sign: float = 1.0) -> np.ndarray:
    """Expected payoff at every node, to player 0 (sign=1) or player 1 (sign=-1)."""
    u = np.where(tree.kind == 2, sign * tree.payoff, 0.0)
    for lv in reversed(tree.levels):
        np.add.at(u, tree.e_parent[lv], ep[lv] * u[tree.e_child[lv]])
    return u


def regret_matching(regret: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pos = np.where(mask, np.maximum(regret, 0.0), 0.0)
    total = pos.sum(1, keepdims=True)
    return np.where(total > 0, pos / np.where(total > 0, total, 1.0), mask / mask.sum(1, keepdims=True))


def cfr_plus(tree: Tree, iters: int = 2000) -> np.ndarray:
    """Average strategy of alternating-update CFR+ with linear averaging."""
    regret = np.zeros(tree.mask.shape)
    avg = np.zeros(tree.mask.shape)
    dec = ~tree.e_is_chance
    for t in range(1, iters + 1):
        for p in (0, 1):
            sigma = regret_matching(regret, tree.mask)
            ep = tree.edge_probs(sigma)
            r, u = reach(tree, ep), values(tree, ep, 1.0 if p == 0 else -1.0)
            e = dec & (tree.e_player == p)
            par, inf, loc = tree.e_parent[e], tree.e_infoset[e], tree.e_local[e]
            q = np.zeros(tree.mask.shape)
            np.add.at(q, (inf, loc), r[1 - p, par] * r[2, par] * u[tree.e_child[e]])
            own = np.zeros(tree.n_infosets)
            np.maximum.at(own, inf, r[p, par])
            rows = tree.infosets_of(p)
            v = (sigma[rows] * q[rows]).sum(1, keepdims=True)
            regret[rows] = np.where(tree.mask[rows], np.maximum(regret[rows] + q[rows] - v, 0.0), 0.0)
            avg[rows] += t * own[rows, None] * sigma[rows]
    total = avg.sum(1, keepdims=True)
    return np.where(total > 0, avg / np.where(total > 0, total, 1.0), tree.uniform())


def best_response(tree: Tree, strategy: np.ndarray, player: int) -> tuple[np.ndarray, float]:
    """Pure best response of `player` to the other seat's strategy (rows of `player` are ignored).

    Returns the response as a strategy matrix (only `player`'s rows are meaningful) and its value to `player`.
    """
    sign = 1.0 if player == 0 else -1.0
    ep = tree.edge_probs(strategy)
    r = reach(tree, ep)
    w = r[1 - player] * r[2]
    u = np.where(tree.kind == 2, sign * tree.payoff, 0.0)
    br = strategy.copy()
    for lv in reversed(tree.levels):
        mine = lv[tree.e_player[lv] == player]
        if len(mine):
            par, inf, loc = tree.e_parent[mine], tree.e_infoset[mine], tree.e_local[mine]
            q = np.full(tree.mask.shape, -np.inf)
            q[inf, loc] = 0.0
            np.add.at(q, (inf, loc), w[par] * u[tree.e_child[mine]])
            rows = np.unique(inf)
            best = q[rows].argmax(1)
            br[rows] = 0.0
            br[rows, best] = 1.0
            ep[mine] = br[inf, loc]
        np.add.at(u, tree.e_parent[lv], ep[lv] * u[tree.e_child[lv]])
    return br, float(u[0])


def value(tree: Tree, strategy: np.ndarray, player: int = 0) -> float:
    return float(values(tree, tree.edge_probs(strategy), 1.0 if player == 0 else -1.0)[0])


def combine(tree: Tree, a: np.ndarray, b: np.ndarray, rows_from_b: int) -> np.ndarray:
    """Strategy taking `rows_from_b`'s information sets from b and every other row from a."""
    out = a.copy()
    rows = tree.infosets_of(rows_from_b)
    out[rows] = b[rows]
    return out


def exploitability(tree: Tree, strategy: np.ndarray) -> float:
    """Average best-response gain over both seats (NashConv / 2), chips per hand. Zero exactly at equilibrium."""
    return 0.5 * (best_response(tree, strategy, 0)[1] + best_response(tree, strategy, 1)[1])
