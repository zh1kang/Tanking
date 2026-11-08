import numpy as np
import pytest

from tanking.data import TraceSet, keys_of
from tanking.games import KUHN, LEDUC, Tree
from tanking.leak import (
    br_values,
    measure,
    sample_actions,
    shuffle_within,
    split_halves,
    tell_bits,
    tokens_per_decision,
)
from tanking.solve import best_response, cfr_plus, exploitability, value


@pytest.fixture(scope="module")
def kuhn_nash():
    t = Tree(KUHN)
    return t, cfr_plus(t, 3000)


def test_known_game_values(kuhn_nash):
    t, s = kuhn_nash
    assert value(t, s) == pytest.approx(-1 / 18, abs=1e-3)
    assert exploitability(t, s) < 1e-3
    leduc = Tree(LEDUC)
    ls = cfr_plus(leduc, 2000)
    assert value(leduc, ls) == pytest.approx(-0.0856, abs=2e-3)
    assert exploitability(leduc, ls) < 2e-3


def test_best_response_to_uniform_beats_it():
    t = Tree(KUHN)
    _, v0 = best_response(t, t.uniform(), 0)
    _, v1 = best_response(t, t.uniform(), 1)
    assert v0 > 0 and v1 > 0


def fake_traces(tree, strategy, n, tokens_fn, seed=0):
    rng = np.random.default_rng(seed)
    width = max(len(a) for a in tree.inf_legal)
    inf, tok, probs = [], [], []
    for i, base in enumerate(tree.base):
        p = np.full(width, np.nan)
        p[: len(tree.inf_legal[i])] = strategy[i, : len(tree.inf_legal[i])]
        for _ in range(n):
            inf.append(i)
            tok.append(tokens_fn(base, rng))
            probs.append(p)
    n_tr = len(inf)
    return TraceSet(
        KUHN.name,
        "fake",
        "think",
        keys_of(tree),
        np.array(inf),
        np.array(tok),
        np.ones(n_tr, bool),
        np.array(probs),
        np.ones(n_tr),
    )


def test_no_tell_means_no_leak(kuhn_nash):
    t, s = kuhn_nash
    ts = fake_traces(t, s, 400, lambda b, rng: int(rng.integers(100, 900)))
    r = br_values(ts, split_halves(ts, np.random.default_rng(1)), k=4)
    assert abs(r["leak"]) < 5.0
    assert r["actions"] < 5.0
    assert r["card"] > 100.0


def test_planted_tell_is_priced(kuhn_nash):
    t, s = kuhn_nash
    ts = fake_traces(t, s, 400, lambda b, rng: int(rng.integers(600, 900) if b.rank == 1 else rng.integers(100, 400)))
    r = br_values(ts, split_halves(ts, np.random.default_rng(1)), k=4)
    assert r["leak"] > 10.0
    assert r["timing"] <= r["card"] + 1.0
    full = fake_traces(t, s, 400, lambda b, rng: int(100 + 300 * b.rank + rng.integers(0, 50)))
    rf = br_values(full, split_halves(full, np.random.default_rng(1)), k=3)
    assert 100.0 < rf["leak"] < rf["card"]
    null = shuffle_within(sample_actions(ts, np.random.default_rng(2)), np.random.default_rng(3))
    rn = br_values(null, split_halves(null, np.random.default_rng(4)), k=4)
    assert abs(rn["leak"]) < 15.0
    bits = tell_bits(ts)
    assert bits["bits_gained"] > 0.2


def test_measure_reports_ci(kuhn_nash):
    t, s = kuhn_nash
    ts = fake_traces(t, s, 100, lambda b, rng: int(rng.integers(100, 900)))
    m = measure(ts, k=2, n_splits=2, n_boot=5)
    assert m["ci"]["leak"][0] <= m["ci"]["leak"][1]
    assert tokens_per_decision(ts) == pytest.approx(500, rel=0.1)


def test_sharded_run_loads_as_one(kuhn_nash, tmp_path):
    t, s = kuhn_nash
    shards = []
    for seed in range(2):
        ts = fake_traces(t, s, 10, lambda b, rng: int(rng.integers(100, 200)), seed=seed)
        ts.meta = dict(
            n=10,
            seed=seed,
            max_think=256,
            gen_seconds=1.0,
            score_seconds=1.0,
            generated_tokens=5,
            truncated=0.5 * seed,
            legal_mass=1.0,
        )
        ts.save(tmp_path / f"shard{seed}")
        shards.append(ts)
    merged = TraceSet.load(tmp_path)
    assert len(merged.inf) == 2 * len(shards[0].inf)
    assert np.array_equal(merged.tokens, np.concatenate([s.tokens for s in shards]))
    assert merged.meta["n"] == 20 and merged.meta["truncated"] == 0.25 and merged.meta["seeds"] == [0, 1]
