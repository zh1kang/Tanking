import numpy as np
import pytest

from experiments.learner import Sampler, at, simulate
from tanking.games import KUHN, Tree
from tanking.leak import MBB, bucket_edges, buckets, policy, tree_for
from tanking.solve import best_response, combine, value
from tests.test_core import fake_traces


def test_sampler_matches_edge_probabilities():
    t = Tree(KUHN)
    ep = t.edge_probs(combine(t, t.uniform(), t.uniform(), 0))
    paths = Sampler(t).paths(ep, 20000, np.random.default_rng(0))
    freq = np.bincount(paths[paths >= 0], minlength=len(ep)) / 20000
    incoming = np.full(len(t.kind), -1)
    incoming[t.e_child] = np.arange(len(ep))
    parent = incoming[t.e_parent]
    reach = ep.copy()
    for _ in range(len(t.kind)):
        reach = ep * np.where(parent >= 0, reach[np.maximum(parent, 0)], 1.0)
    assert np.abs(freq - reach).max() < 0.02


@pytest.mark.parametrize("seat", [0, 1])
def test_learner_reaches_the_exploring_best_response(seat):
    t = Tree(KUHN)
    rng = np.random.default_rng(seat)
    strategy = t.uniform().copy()
    for i, legal in enumerate(t.inf_legal):
        strategy[i, : len(legal)] = rng.dirichlet(np.ones(len(legal)))
    slow_king = lambda b, rng: int(rng.integers(600, 900) if b.rank == 2 else rng.integers(100, 400))  # noqa: E731
    ts = fake_traces(t, strategy, 200, slow_king)
    tree = tree_for(ts.variant, seat, 4)
    S = policy(tree, ts, buckets(ts, bucket_edges(ts, 4)))
    br, _ = best_response(tree, S, 1 - seat)
    rows = tree.infosets_of(1 - seat)
    br[rows] = 0.95 * br[rows] + 0.05 * tree.uniform()[rows]
    target = MBB * value(tree, combine(tree, S, br, 1 - seat), 1 - seat)
    run = simulate(ts, ts, 4, seat, 0.05, 20000, seed=seat)
    assert at(run, np.array([19999]))[0] == pytest.approx(target, abs=1.0)
