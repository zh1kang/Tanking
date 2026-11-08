"""Phase 2 for one thinking TraceSet: padding defenses and public-information budget schedules.

Padding: fit on one half of the traces, applied to the other half with one sampled action per trace (what an
observer sees), then the leak is measured there. Repeated over random halves; reported as mean and sd.
Budgets: a schedule is searched on one half and its exploitability and latency reported on the other.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tanking.data import TraceSet
from tanking.defenses import (
    apply_padding,
    budget_traces,
    exploitability_mbb,
    fit_padding,
    pad_constant,
    pad_random,
    search_schedule,
)
from tanking.leak import br_values, default_k, dp_layout, sample_actions, split_halves, tokens_per_decision


def floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def padded_leak(E: TraceSet, tokens: np.ndarray, k: int, rng: np.random.Generator) -> dict:
    x = E.with_tokens(tokens)
    leak = float(np.mean([br_values(x, split_halves(x, rng), k)["leak"] for _ in range(2)]))
    return {"leak": leak, "added_tokens": tokens_per_decision(E, tokens) - tokens_per_decision(E)}


def summarize(rows: list[dict]) -> list[dict]:
    out = {}
    for r in rows:
        out.setdefault((r["defense"], r["param"]), []).append(r)
    res = []
    for (d, p), rs in out.items():
        res.append(
            dict(
                defense=d,
                param=p,
                **{
                    f"{m}_{s}": float(f([r[m] for r in rs]))
                    for m in ("leak", "added_tokens")
                    for s, f in (("mean", np.mean), ("sd", np.std), ("se", lambda x: np.std(x) / np.sqrt(len(x))))
                },
            )
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--k", type=int, help="time buckets (default by game)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--eps", default="0,0.02,0.05,0.1,0.2")
    ap.add_argument("--random", default="")
    ap.add_argument("--lams", default="0,0.003,0.01,0.03,0.1")
    ap.add_argument("--skip-padding", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ts = TraceSet.load(Path(a.run))
    cap = ts.meta["max_think"]
    k = a.k or default_k(ts.variant)
    rand = floats(a.random) or [cap / 8, cap / 4, cap / 2, cap, 2 * cap]
    rows = []
    for rep in range(0 if a.skip_padding else a.repeats):
        rng = np.random.default_rng(100 + rep)
        fit = split_halves(ts, rng)
        F, E = ts.subset(fit), sample_actions(ts.subset(~fit), rng)
        candidates = [("none", 0, E.tokens), ("constant", cap, pad_constant(E, cap))]
        candidates += [("random", p, pad_random(E, int(p), rng)) for p in rand]
        candidates += [("lp", eps, apply_padding(E, fit_padding(F, eps, cap), cap, rng)) for eps in floats(a.eps)]
        for defense, param, tokens in candidates:
            rows.append(dict(rep=rep, defense=defense, param=param, **padded_leak(E, tokens, k, rng)))
            print(json.dumps(rows[-1]), flush=True)

    names = dp_layout(tuple(ts.keys))[2]
    sizes = list(ts.budgets) + [cap]
    budgets = []
    if ts.budget_probs is not None:
        for lam in floats(a.lams):
            rng = np.random.default_rng(7)
            sel = split_halves(ts, rng)
            S, R = ts.subset(sel), ts.subset(~sel)
            sched = search_schedule(S, lam, rng)
            Rb = budget_traces(R, sched)
            cut = np.array([sizes[sched.get(int(d), -1)] for d in dp_layout(tuple(R.keys))[0][R.inf]])
            budgets.append(
                dict(
                    lam=lam,
                    exploitability=exploitability_mbb(Rb, rng, 4),
                    reported_tokens=tokens_per_decision(Rb),
                    compute_tokens=tokens_per_decision(R, np.minimum(R.tokens, cut)),
                    schedule={names[d]: sizes[b] for d, b in sched.items()},
                )
            )
            print(json.dumps({k: v for k, v in budgets[-1].items() if k != "schedule"}), flush=True)
        rng = np.random.default_rng(7)
        R = ts.subset(~split_halves(ts, rng))
        budgets.append(
            dict(
                lam=None,
                exploitability=exploitability_mbb(R, rng, 4),
                reported_tokens=tokens_per_decision(R),
                compute_tokens=tokens_per_decision(R),
                schedule="full thinking, unpadded",
            )
        )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(dict(run=a.run, k=k, padding=summarize(rows), padding_rows=rows, budgets=budgets), indent=1)
    )


if __name__ == "__main__":
    main()
