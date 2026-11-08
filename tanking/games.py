"""A parametrized two-player poker family (Kuhn, Leduc and variants) flattened into arrays for vectorized solving.

Cards are dealt by rank (suits never matter), so chance nodes carry collapsed probabilities. An information set is
identified by (player, own rank, board rank, betting history). Optionally the tree is expanded so that one seat's
moves also emit a timing bucket the other seat observes, or so the other seat sees that seat's card.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

RANKS = "23456789TJQKA"
ACTION_WORDS = {
    ("c", False): "check",
    ("r", False): "bet",
    ("c", True): "call",
    ("r", True): "raise",
    ("f", True): "fold",
}


@dataclass(frozen=True)
class Variant:
    name: str
    n_ranks: int = 3
    copies: int = 1
    rounds: int = 1
    bets: tuple[int, ...] = (1,)
    max_raises: int = 1
    ante: int = 1

    @property
    def ranks(self) -> str:
        return RANKS[12 - self.n_ranks : 12]

    @property
    def deck_size(self) -> int:
        return self.n_ranks * self.copies


KUHN = Variant("kuhn")
LEDUC = Variant("leduc", n_ranks=3, copies=2, rounds=2, bets=(2, 4), max_raises=2)


@dataclass(frozen=True)
class InfoKey:
    player: int
    rank: int
    board: int
    history: str

    def __str__(self) -> str:
        return f"{self.player}|{self.rank}|{self.board}|{self.history}"


def legal(history_round: str, raises: int, max_raises: int) -> tuple[str, ...]:
    facing = history_round.endswith("r")
    acts = ("f", "c") if facing else ("c",)
    return acts + ("r",) if raises < max_raises else acts


def action_word(a: str, history_round: str) -> str:
    return ACTION_WORDS[(a, history_round.endswith("r"))]


def round_over(h: str) -> bool:
    return h == "cc" or (len(h) >= 2 and h[-1] == "c" and "r" in h)


def showdown(v: Variant, r0: int, r1: int, board: int) -> int:
    """+1 if player 0 wins, -1 if player 1 wins, 0 on a tie."""
    if v.rounds > 1:
        if r0 == board:
            return 1
        if r1 == board:
            return -1
    return int(np.sign(r0 - r1))


class Tree:
    """Flattened game tree. Node kinds: 0 chance, 1 decision, 2 terminal. Edges point parent -> child.

    `signal_seat` (with k > 1) makes that seat's moves emit one of k buckets which the other seat observes;
    `reveal` lets the other seat see the signal seat's card. The signal seat's own information sets never change.
    """

    def __init__(self, v: Variant, signal_seat: int | None = None, k: int = 1, reveal: bool = False):
        self.v, self.signal_seat, self.k, self.reveal = v, signal_seat, k, reveal
        self.kind: list[int] = []
        self.player: list[int] = []
        self.infoset: list[int] = []
        self.depth: list[int] = []
        self.payoff: list[float] = []
        self.e_parent: list[int] = []
        self.e_child: list[int] = []
        self.e_prob: list[float] = []
        self.e_local: list[int] = []
        self.e_action: list[str] = []
        self.e_bucket: list[int] = []
        self.keys: list[str] = []
        self.base: list[InfoKey] = []
        self.inf_player: list[int] = []
        self.inf_legal: list[tuple[str, ...]] = []
        self.inf_depth: list[int] = []
        self._index: dict[str, int] = {}
        self._deal(0, ())
        self._finalize()

    def _node(self, kind: int, depth: int, player: int = -1, infoset: int = -1, payoff: float = 0.0) -> int:
        self.kind.append(kind)
        self.player.append(player)
        self.infoset.append(infoset)
        self.depth.append(depth)
        self.payoff.append(payoff)
        return len(self.kind) - 1

    def _edge(self, parent: int, child: int, prob: float = np.nan, local: int = -1, action: str = "", bucket: int = -1):
        self.e_parent.append(parent)
        self.e_child.append(child)
        self.e_prob.append(prob)
        self.e_local.append(local)
        self.e_action.append(action)
        self.e_bucket.append(bucket)

    def _rank_probs(self, dealt: tuple[int, ...]) -> list[tuple[int, float]]:
        left = self.v.deck_size - len(dealt)
        out = []
        for r in range(self.v.n_ranks):
            n = self.v.copies - dealt.count(r)
            if n > 0:
                out.append((r, n / left))
        return out

    def _deal(self, depth: int, dealt: tuple[int, ...]) -> int:
        if len(dealt) == 2:
            return self._decide(depth, dealt[0], dealt[1], -1, ("",), (self.v.ante, self.v.ante), 0, "")
        node = self._node(0, depth)
        for r, p in self._rank_probs(dealt):
            self._edge(node, self._deal(depth + 1, dealt + (r,)), prob=p)
        return node

    def _info(self, key: str, base: InfoKey, legal_acts: tuple[str, ...], depth: int) -> int:
        i = self._index.get(key)
        if i is None:
            i = self._index[key] = len(self.keys)
            self.keys.append(key)
            self.base.append(base)
            self.inf_player.append(base.player)
            self.inf_legal.append(legal_acts)
            self.inf_depth.append(depth)
        assert self.inf_depth[i] == depth, "perfect recall: an information set lies at one depth"
        return i

    def _decide(self, depth, r0, r1, board, hist, contrib, raises, obs) -> int:
        v, rnd = self.v, len(hist) - 1
        h = hist[-1]
        p = len(h) % 2
        acts = legal(h, raises, v.max_raises)
        base = InfoKey(p, (r0, r1)[p], board, "/".join(hist))
        key = str(base)
        signals = p == self.signal_seat
        if self.signal_seat is not None and not signals:
            key += f"|{obs}" + (f"|opp{(r0, r1)[self.signal_seat]}" if self.reveal else "")
        node = self._node(1, depth, p, self._info(key, base, acts, depth))
        buckets = range(self.k) if signals else (-1,)
        for ai, a in enumerate(acts):
            for b in buckets:
                c, nr, nh = list(contrib), raises, h + a
                if a == "c":
                    c[p] = c[1 - p]
                elif a == "r":
                    c[p] = c[1 - p] + v.bets[rnd]
                    nr += 1
                nobs = obs + (f"{a}{b}" if signals else a)
                if a == "f":
                    child = self._node(2, depth + 1, payoff=float(c[1] if p == 1 else -c[0]))
                elif round_over(nh):
                    if rnd + 1 < v.rounds:
                        child = self._board(depth + 1, r0, r1, hist[:-1] + (nh, ""), tuple(c), nobs + "/")
                    else:
                        s = showdown(v, r0, r1, board)
                        child = self._node(2, depth + 1, payoff=float(s * c[1 - (s < 0)] if s else 0.0))
                else:
                    child = self._decide(depth + 1, r0, r1, board, hist[:-1] + (nh,), tuple(c), nr, nobs)
                local = ai * len(buckets) + (b if signals else 0)
                self._edge(node, child, local=local, action=a, bucket=b)
        return node

    def _board(self, depth, r0, r1, hist, contrib, obs) -> int:
        node = self._node(0, depth)
        for r, p in self._rank_probs((r0, r1)):
            child = self._decide(depth + 1, r0, r1, r, hist, contrib, 0, obs + RANKS[12 - self.v.n_ranks + r])
            self._edge(node, child, prob=p)
        return node

    def _finalize(self):
        for name in ("kind", "player", "infoset", "depth", "e_parent", "e_child", "e_local", "e_bucket"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.int64))
        self.payoff = np.asarray(self.payoff, dtype=np.float64)
        self.e_prob = np.asarray(self.e_prob, dtype=np.float64)
        self.e_infoset = self.infoset[self.e_parent]
        self.e_is_chance = self.kind[self.e_parent] == 0
        self.e_player = np.where(self.e_is_chance, -1, self.player[self.e_parent])
        self.inf_player = np.asarray(self.inf_player, dtype=np.int64)
        self.n_infosets = len(self.keys)
        self.n_actions = np.asarray(
            [len(a) * (self.k if p == self.signal_seat else 1) for a, p in zip(self.inf_legal, self.inf_player)],
            dtype=np.int64,
        )
        self.max_actions = int(self.n_actions.max())
        order = np.argsort(self.depth[self.e_parent], kind="stable")
        d = self.depth[self.e_parent][order]
        cuts = np.flatnonzero(np.diff(d)) + 1
        self.levels = np.split(order, cuts)

    @cached_property
    def mask(self) -> np.ndarray:
        m = np.zeros((self.n_infosets, self.max_actions), dtype=bool)
        for i, n in enumerate(self.n_actions):
            m[i, :n] = True
        return m

    def uniform(self) -> np.ndarray:
        return self.mask / self.mask.sum(1, keepdims=True)

    def edge_probs(self, strategy: np.ndarray) -> np.ndarray:
        """Per-edge probability: chance probabilities, or strategy[infoset, local action] at decision nodes."""
        p = self.e_prob.copy()
        dec = ~self.e_is_chance
        p[dec] = strategy[self.e_infoset[dec], self.e_local[dec]]
        return p

    def infosets_of(self, player: int) -> np.ndarray:
        return np.flatnonzero(self.inf_player == player)
