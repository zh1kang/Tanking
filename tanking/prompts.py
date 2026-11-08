"""Natural-language prompts for an information set, and the answer-token vocabulary."""

from tanking.games import RANKS, InfoKey, Variant, action_word, legal

ANSWER_PREFIX = "Action:"
EARLY_STOP = "\n\nConsidering the limited time, I have to give the action directly now.\n</think>\n\n" + ANSWER_PREFIX


def rank_name(v: Variant, r: int) -> str:
    return RANKS[12 - v.n_ranks + r]


def replay(v: Variant, history: str) -> tuple[list[list[str]], tuple[int, int]]:
    """Per-round lists of 'Player i <verb>' phrases and each player's chips in the pot."""
    c = [v.ante, v.ante]
    rounds = []
    for rnd, h in enumerate(history.split("/")):
        phrases = []
        for i, a in enumerate(h):
            p = i % 2
            phrases.append(f"Player {p + 1} {action_word(a, h[:i])}s")
            if a == "c":
                c[p] = c[1 - p]
            elif a == "r":
                c[p] = c[1 - p] + v.bets[rnd]
        rounds.append(phrases)
    return rounds, (c[0], c[1])


def rules(v: Variant) -> str:
    ranks = ", ".join(rank_name(v, r) for r in range(v.n_ranks))
    deck = f"{v.deck_size} cards: {v.copies} of each rank {ranks} (lowest to highest)"
    lines = [
        f"You are playing a two-player poker game. The deck has {deck}.",
        f"Each player antes {v.ante} chip and is dealt one private card.",
    ]
    if v.rounds == 1:
        lines.append("There is one betting round, then the higher card wins the pot.")
    else:
        lines.append(
            "After the first betting round one public board card is revealed, followed by a second betting round. "
            "At showdown a player whose card matches the board wins; otherwise the higher card wins; "
            "equal ranks split the pot."
        )
    sizes = ", ".join(f"{b} in round {i + 1}" for i, b in enumerate(v.bets)) if v.rounds > 1 else str(v.bets[0])
    lines.append(
        f"Bets and raises are a fixed size ({sizes}), with at most {v.max_raises} bet"
        f"{'s or raises' if v.max_raises > 1 else ''} per round. Player 1 acts first in every round. "
        "Folding loses the chips you have put in."
    )
    return " ".join(lines)


def user_message(v: Variant, k: InfoKey) -> str:
    rounds, c = replay(v, k.history)
    acts = legal(k.history.split("/")[-1], sum(a == "r" for a in k.history.split("/")[-1]), v.max_raises)
    words = [action_word(a, k.history.split("/")[-1]) for a in acts]
    state = [f"You are Player {k.player + 1}. Your card: {rank_name(v, k.rank)}."]
    if k.board >= 0:
        state.append(f"Board card: {rank_name(v, k.board)}.")
    for i, phrases in enumerate(rounds):
        label = f"Round {i + 1} betting so far" if v.rounds > 1 else "Betting so far"
        state.append(f"{label}: {', '.join(phrases) if phrases else 'none'}.")
    state.append(f"Pot: {c[0] + c[1]} chips; you have put in {c[k.player]}.")
    state.append(f"Legal actions: {', '.join(words)}.")
    return (
        f"{rules(v)}\n\n{' '.join(state)}\n\nChoose the action that maximizes your expected winnings "
        f"(mixing between actions is allowed). End your reply with one line: '{ANSWER_PREFIX} <{' or '.join(words)}>'."
    )


def action_words(v: Variant, k: InfoKey) -> list[str]:
    h = k.history.split("/")[-1]
    return [action_word(a, h) for a in legal(h, sum(a == "r" for a in h), v.max_raises)]
