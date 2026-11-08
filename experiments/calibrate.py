"""Calibrate the leak estimator on synthetic thinking times with a known tell (exploratory, added after first results).

Real traces (information sets and action distributions) are resampled to n per information set and their thinking
tokens replaced: log t = mu - delta * rank / (n_ranks - 1) + N(0, sigma^2), independently per trace. delta = 0 has no
tell by construction; for delta > 0 the reference value is the estimator's point estimate at n = 4096.

  run:       one (traces, n, delta) cell: `reps` replicates of the Phase 1 pipeline (point, bootstrap CI, null range)
  summarize: bias, CI coverage of the reference, and how often the pre-registered "tell confirmed" rule fires
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tanking.data import TraceSet, parse_key
from tanking.leak import br_values, default_k, measure, sample_actions, shuffle_within, split_halves

MU, SIGMA = 8.0, 0.5
REF_N = 4096


def synthetic(ts: TraceSet, n: int, delta: float, rng: np.random.Generator) -> TraceSet:
    idx = np.concatenate([rng.choice(np.flatnonzero(ts.inf == i), n) for i in np.unique(ts.inf)])
    out = ts.subset(idx)
    rank = np.array([parse_key(k).rank for k in ts.keys])[out.inf] / (ts.variant.n_ranks - 1)
    out.tokens = np.exp(MU - delta * rank + SIGMA * rng.standard_normal(len(idx)))
    return out


def null_range(ts: TraceSet, k: int, rng: np.random.Generator, draws: int = 20) -> list[float]:
    vals = []
    for _ in range(draws):
        y = shuffle_within(sample_actions(ts, rng), rng)
        vals.append(br_values(y, split_halves(y, rng), k)["leak"])
    return np.percentile(vals, [2.5, 97.5]).tolist()


def run(a) -> None:
    ts = TraceSet.load(Path(a.traces))
    k, rng, rows = default_k(ts.variant), np.random.default_rng(a.seed), []
    for r in range(a.reps):
        x = synthetic(ts, a.n, a.delta, rng)
        if a.boot:
            m = measure(x, k, n_boot=a.boot, seed=a.seed * 1000 + r)
            rows.append({"point": m["point"]["leak"], "ci": m["ci"]["leak"], "null": null_range(x, k, rng)})
        else:
            rows.append({"point": float(np.mean([br_values(x, split_halves(x, rng), k)["leak"] for _ in range(4)]))})
        print(json.dumps(rows[-1]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps({"traces": a.traces, "game": ts.game, "n": a.n, "delta": a.delta, "rows": rows}, indent=1)
    )


def summarize(a) -> None:
    cells = [json.loads(p.read_text()) for p in sorted(Path(a.dir).glob("cell_*.json"))]
    ref = {(c["game"], c["delta"]): float(np.mean([r["point"] for r in c["rows"]])) for c in cells if c["n"] == REF_N}
    md = [
        "# Leak estimator calibration (synthetic tells)",
        "",
        f"Synthetic thinking times log t = {MU} - delta * rank / (n_ranks - 1) + N(0, {SIGMA}^2) on real information "
        f"sets and action distributions. Reference = point estimate at n = {REF_N} per information set "
        "(0 when delta = 0).",
        "",
        "| game | delta | n per infoset | reference | mean estimate | bias | CI covers reference | tell rule fires "
        "| null range contains 0 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    out = []
    for c in sorted((c for c in cells if c["n"] != REF_N), key=lambda c: (c["game"], c["delta"], c["n"])):
        truth = 0.0 if c["delta"] == 0 else ref.get((c["game"], c["delta"]), float("nan"))
        pts = np.array([r["point"] for r in c["rows"]])
        cover = np.mean([r["ci"][0] <= truth <= r["ci"][1] for r in c["rows"]])
        fires = np.mean([r["point"] >= 10 and r["ci"][0] > 0 and r["null"][0] <= 0 <= r["null"][1] for r in c["rows"]])
        null0 = np.mean([r["null"][0] <= 0 <= r["null"][1] for r in c["rows"]])
        row = dict(
            game=c["game"],
            delta=c["delta"],
            n=c["n"],
            reference=truth,
            mean=float(pts.mean()),
            bias=float(pts.mean() - truth),
            coverage=float(cover),
            rule_fires=float(fires),
            null_contains_0=float(null0),
            reps=len(pts),
        )
        out.append(row)
        md.append(
            f"| {c['game']} | {c['delta']:g} | {c['n']} | {truth:.1f} | {row['mean']:.1f} | {row['bias']:+.1f} "
            f"| {100 * cover:.0f}% | {100 * fires:.0f}% | {100 * null0:.0f}% |"
        )
    Path(a.dir, "summary.json").write_text(json.dumps(out, indent=1))
    Path(a.dir, "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("traces")
    r.add_argument("--n", type=int, required=True)
    r.add_argument("--delta", type=float, required=True)
    r.add_argument("--reps", type=int, default=20)
    r.add_argument("--boot", type=int, default=100)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        summarize(a)


if __name__ == "__main__":
    main()
