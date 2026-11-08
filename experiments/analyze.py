"""Phase 1 and strength metrics for one TraceSet.

Leak with CI, null control, bits, tokens per decision, exploitability and EV against Nash.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tanking.data import TraceSet, parse_key
from tanking.leak import (
    MBB,
    br_values,
    default_k,
    dp_layout,
    dp_of,
    measure,
    policy,
    sample_actions,
    shuffle_within,
    split_halves,
    tell_bits,
    tokens_per_decision,
    tree_for,
)
from tanking.solve import cfr_plus, combine, exploitability, value


def ev_vs(tree, s_llm, s_opp) -> float:
    """LLM's seat-averaged expected value against a fixed opponent strategy, chips per hand."""
    return 0.5 * (value(tree, combine(tree, s_llm, s_opp, 1), 0) + value(tree, combine(tree, s_opp, s_llm, 1), 1))


def tell_table(ts: TraceSet) -> list[dict]:
    """Thinking-token quantiles by LLM card at each decision point (what the tell figure plots)."""
    names = dp_layout(tuple(ts.keys))[2]
    dp = dp_of(ts)
    rank = np.array([parse_key(k).rank for k in ts.keys])[ts.inf]
    rows = []
    for d in np.unique(dp):
        for r in np.unique(rank[dp == d]):
            t = ts.tokens[(dp == d) & (rank == r)]
            rows.append(
                dict(
                    dp=names[d],
                    rank=int(r),
                    n=int(len(t)),
                    q10=float(np.percentile(t, 10)),
                    q50=float(np.percentile(t, 50)),
                    q90=float(np.percentile(t, 90)),
                    mean=float(t.mean()),
                )
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--k", type=int, help="time buckets (default by game)")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for run in a.runs:
        ts = TraceSet.load(Path(run))
        tree = tree_for(ts.variant)
        k = a.k or default_k(ts.variant)
        nash = cfr_plus(tree, 3000)
        s = policy(tree, ts)
        res = dict(
            run=run,
            game=ts.game,
            model=ts.model,
            mode=ts.mode,
            meta=ts.meta,
            tokens_per_decision=tokens_per_decision(ts),
            ev_vs_nash=MBB * ev_vs(tree, s, nash),
            validity=dict(
                legal_mass=float(ts.mass.mean()),
                low_mass=float((ts.mass < 0.5).mean()),
                truncated=float(1 - ts.closed.mean()),
            ),
        )
        if ts.mode == "think":
            res["phase1"] = measure(ts, k, n_boot=a.boot)
            rng = np.random.default_rng(1)
            nulls, sampled = [], []
            for _ in range(max(20, a.boot // 5)):
                x = sample_actions(ts, rng)
                sampled.append(br_values(x, split_halves(x, rng), k)["leak"])
                y = shuffle_within(x, rng)
                nulls.append(br_values(y, split_halves(y, rng), k)["leak"])
            res["sampled_leak"] = dict(mean=float(np.mean(sampled)), ci=np.percentile(sampled, [2.5, 97.5]).tolist())
            res["null_leak"] = dict(mean=float(np.mean(nulls)), ci=np.percentile(nulls, [2.5, 97.5]).tolist())
            res["bits"] = tell_bits(ts)
            res["exploitability"] = res["phase1"]["point"]["actions"]
            res["tell"] = tell_table(ts)
        else:
            res["exploitability"] = MBB * exploitability(tree, s)
        name = f"{ts.game}_{Path(ts.model).name}_{ts.mode}"
        (out / f"{name}.json").write_text(json.dumps(res, indent=1))
        p = res.get("phase1", {}).get("point", {})
        print(
            name,
            json.dumps(
                dict(
                    expl=round(res["exploitability"], 1),
                    ev=round(res["ev_vs_nash"], 1),
                    tokens=round(res["tokens_per_decision"], 1),
                    leak=p.get("leak"),
                    ci=res.get("phase1", {}).get("ci", {}).get("leak"),
                    null=res.get("null_leak"),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
