import numpy as np
import pytest

from tanking.defenses import _solve_group, apply_padding, fit_padding
from tanking.games import KUHN, Tree
from tanking.leak import br_values, sample_actions, split_halves
from tanking.solve import cfr_plus
from tests.test_core import fake_traces


def test_lp_matches_envelope_at_zero_eps():
    rng = np.random.default_rng(0)
    M = 12
    p = rng.random((3, M)) * (rng.random((3, M)) < 0.6)
    p /= p.sum(1, keepdims=True)
    w = np.array([0.5, 0.3, 0.2])
    K = _solve_group(p, w, 0.0)
    q = np.einsum("ct,ctu->cu", p, K)
    assert np.allclose(q, q[0], atol=1e-7)
    assert np.all(np.triu(np.ones((M, M))) >= (K > 1e-12))
    env_cdf = np.cumsum(p, 1).min(0)
    env = np.diff(np.concatenate([[0], env_cdf]))
    t = np.arange(M)
    lp_cost = sum(w[c] * (q[c] @ t - p[c] @ t) for c in range(3))
    env_cost = sum(w[c] * (env @ t - p[c] @ t) for c in range(3))
    assert lp_cost == pytest.approx(env_cost, abs=1e-6)


def test_tv_budget_lowers_cost():
    rng = np.random.default_rng(1)
    p = rng.random((2, 10))
    p /= p.sum(1, keepdims=True)
    w = np.array([0.5, 0.5])
    t = np.arange(10)
    costs = []
    for eps in (0.0, 0.1, 0.3):
        q = np.einsum("ct,ctu->cu", p, _solve_group(p, w, eps))
        costs.append(sum(w[c] * (q[c] @ t - p[c] @ t) for c in range(2)))
        assert 0.5 * np.abs(q[0] - q[1]).sum() <= 2 * eps + 1e-6
    assert costs[0] >= costs[1] >= costs[2] - 1e-9


def test_padding_removes_planted_tell():
    t = Tree(KUHN)
    s = cfr_plus(t, 3000)
    before, after = [], []
    for seed in range(4):
        ts = fake_traces(
            t, s, 3000, lambda b, rng: int(rng.integers(600, 900) if b.rank == 1 else rng.integers(100, 400)), seed=seed
        )
        rng = np.random.default_rng(seed)
        fit = split_halves(ts, rng)
        kernels = fit_padding(ts.subset(fit), 0.0, cap=1024)
        ev = sample_actions(ts.subset(~fit), rng)
        before.append(br_values(ev, split_halves(ev, rng), 4)["leak"])
        padded = ev.with_tokens(apply_padding(ev, kernels, 1024, rng))
        after.append(br_values(padded, split_halves(padded, rng), 4)["leak"])
        assert np.all(padded.tokens >= ev.tokens)
    assert np.mean(before) > 10 and abs(np.mean(after)) < 3
