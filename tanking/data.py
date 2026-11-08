"""Sampled traces for every information set of a game, stored as flat arrays (one row per trace)."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tanking.games import KUHN, LEDUC, InfoKey, Tree, Variant

MAX_LEGAL = 3

TRAIN_VARIANTS = (
    KUHN,
    Variant("kuhn4", n_ranks=4),
    Variant("kuhn_b2", bets=(2,)),
    LEDUC,
    Variant("leduc_b12", n_ranks=3, copies=2, rounds=2, bets=(1, 2), max_raises=2),
    Variant("leduc4", n_ranks=4, copies=2, rounds=2, bets=(2, 4), max_raises=2),
)
HELDOUT_VARIANTS = (
    Variant("kuhn5", n_ranks=5),
    Variant("kuhn4_b2", n_ranks=4, bets=(2,)),
    Variant("leduc_b36", n_ranks=3, copies=2, rounds=2, bets=(3, 6), max_raises=2),
    Variant("leduc4_b12", n_ranks=4, copies=2, rounds=2, bets=(1, 2), max_raises=2),
)
VARIANTS = {v.name: v for v in TRAIN_VARIANTS + HELDOUT_VARIANTS}


def decision_point(k: InfoKey) -> str:
    """What the opponent can see about an LLM decision: seat, board and betting history (never the LLM's card)."""
    return f"{k.player}|{k.board}|{k.history}"


@dataclass
class TraceSet:
    game: str
    model: str
    mode: str
    keys: list[str]
    inf: np.ndarray
    tokens: np.ndarray
    closed: np.ndarray
    probs: np.ndarray
    mass: np.ndarray
    budgets: tuple[int, ...] = ()
    budget_probs: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def variant(self) -> Variant:
        return VARIANTS[self.game]

    def subset(self, idx: np.ndarray) -> "TraceSet":
        bp = None if self.budget_probs is None else self.budget_probs[idx]
        return TraceSet(
            self.game,
            self.model,
            self.mode,
            self.keys,
            self.inf[idx],
            self.tokens[idx],
            self.closed[idx],
            self.probs[idx],
            self.mass[idx],
            self.budgets,
            bp,
            self.meta,
        )

    def with_tokens(self, tokens: np.ndarray) -> "TraceSet":
        out = self.subset(np.arange(len(self.inf)))
        out.tokens = tokens
        return out

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        arrays = dict(inf=self.inf, tokens=self.tokens, closed=self.closed, probs=self.probs, mass=self.mass)
        if self.budget_probs is not None:
            arrays["budget_probs"] = self.budget_probs
        np.savez_compressed(path / "traces.npz", **arrays)
        (path / "traces.json").write_text(
            json.dumps(
                dict(
                    game=self.game,
                    model=self.model,
                    mode=self.mode,
                    keys=self.keys,
                    budgets=list(self.budgets),
                    meta=self.meta,
                ),
                indent=1,
            )
        )

    @classmethod
    def concat(cls, parts: list["TraceSet"]) -> "TraceSet":
        """Merge shards sampled with different seeds (same game, model, mode, keys and budgets)."""
        first = parts[0]
        assert all(
            (p.game, p.model, p.mode, p.keys, p.budgets)
            == (first.game, first.model, first.mode, first.keys, first.budgets)
            for p in parts
        )
        sizes = np.array([len(p.inf) for p in parts])
        meta = dict(first.meta, seeds=[p.meta.get("seed") for p in parts])
        for key in ("n", "gen_seconds", "score_seconds", "generated_tokens"):
            meta[key] = sum(p.meta[key] for p in parts)
        for key in ("truncated", "legal_mass"):
            meta[key] = float(np.average([p.meta[key] for p in parts], weights=sizes))
        bp = None if first.budget_probs is None else np.concatenate([p.budget_probs for p in parts])
        cols = [np.concatenate([getattr(p, c) for p in parts]) for c in ("inf", "tokens", "closed", "probs", "mass")]
        return cls(first.game, first.model, first.mode, first.keys, *cols, first.budgets, bp, meta)

    @classmethod
    def load(cls, path: Path) -> "TraceSet":
        """Load a run directory, or merge its shard*/ subdirectories if it was sampled in shards."""
        if not (path / "traces.json").exists():
            shards = sorted(path.glob("shard*/traces.json"))
            assert shards, f"no traces under {path}"
            return cls.concat([cls.load(s.parent) for s in shards])
        m = json.loads((path / "traces.json").read_text())
        z = np.load(path / "traces.npz")
        return cls(
            m["game"],
            m["model"],
            m["mode"],
            m["keys"],
            z["inf"],
            z["tokens"],
            z["closed"],
            z["probs"],
            z["mass"],
            tuple(m["budgets"]),
            z["budget_probs"] if "budget_probs" in z else None,
            m["meta"],
        )


def parse_key(key: str) -> InfoKey:
    p, r, b, h = key.split("|")
    return InfoKey(int(p), int(r), int(b), h)


def keys_of(tree: Tree) -> list[str]:
    return [str(b) for b in tree.base]
