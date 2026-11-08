"""Figures and Markdown tables from results/{analysis,defend,bench}/*.json into figures/ and results/tables.md."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tanking.data import VARIANTS
from tanking.prompts import rank_name

MODEL_ORDER = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "distill_0.6B", "distill_1.7B"]
COLORS = dict(zip(MODEL_ORDER, ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#4C72B0", "#55A868"]))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def load(d: Path) -> list[dict]:
    return [json.loads(p.read_text()) | {"_file": p.stem} for p in sorted(d.glob("*.json"))] if d.exists() else []


def short(model: str) -> str:
    return Path(model).name


def order(rows, key=lambda r: short(r["model"])):
    return sorted(rows, key=lambda r: MODEL_ORDER.index(key(r)) if key(r) in MODEL_ORDER else 99)


def fmt_ci(x, ci) -> str:
    return f"{x:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]"


def fig_tell(analysis, out: Path):
    runs = order([r for r in analysis if r["mode"] == "think" and r["game"] == "kuhn"])
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(2.6 * len(runs), 2.8), sharey=True, squeeze=False)
    v = VARIANTS["kuhn"]
    for ax, r in zip(axes[0], runs):
        dps = sorted({t["dp"] for t in r["tell"]})
        for j, dp in enumerate(dps):
            rows = [t for t in r["tell"] if t["dp"] == dp]
            for t in rows:
                x = j + (t["rank"] - 1) * 0.22
                ax.plot([x, x], [t["q10"], t["q90"]], color=f"C{t['rank']}", lw=1.5)
                ax.plot(x, t["q50"], "o", color=f"C{t['rank']}", ms=4)
        ax.set_xticks(range(len(dps)), [dp_label(d) for d in dps], rotation=30, ha="right")
        ax.set_title(short(r["model"]))
        ax.set_yscale("log")
    axes[0][0].set_ylabel("thinking tokens (median, 10-90%)")
    handles = [plt.Line2D([], [], color=f"C{i}", marker="o", lw=1.5, label=rank_name(v, i)) for i in range(3)]
    fig.legend(handles=handles, title="LLM card", frameon=False, fontsize=8, loc="center right")
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(out / "tell.png")
    plt.close(fig)


def dp_label(dp: str) -> str:
    p, _, h = dp.split("|")
    return f"P{int(p) + 1} {h or 'open'}"


def fig_leak(analysis, out: Path):
    runs = [r for r in analysis if r["mode"] == "think"]
    games = sorted({r["game"] for r in runs})
    if not runs:
        return
    fig, axes = plt.subplots(1, len(games), figsize=(3.4 * len(games), 2.8), squeeze=False)
    for ax, g in zip(axes[0], games):
        rs = order([r for r in runs if r["game"] == g])
        x = np.arange(len(rs))
        leak = np.array([r["phase1"]["point"]["leak"] for r in rs])
        ci = np.array([r["phase1"]["ci"]["leak"] for r in rs])
        ax.bar(
            x,
            leak,
            color=[COLORS.get(short(r["model"]), "gray") for r in rs],
            yerr=np.clip([leak - ci[:, 0], ci[:, 1] - leak], 0, None),
            capsize=3,
            label="timing leak",
        )
        null = np.array([r["null_leak"]["ci"] for r in rs])
        ax.vlines(x + 0.3, null[:, 0], null[:, 1], color="gray", lw=3, label="shuffled-time null (95%)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, [short(r["model"]).replace("Qwen3-", "") for r in rs])
        ax.set_title(g.capitalize())
        ax.set_ylabel("leak (mbb/hand)")
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "leak.png")
    plt.close(fig)


def fig_padding(defend, out: Path):
    runs = [d for d in defend if d["padding"]]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(3.0 * len(runs), 2.8), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], runs):
        pad = d["padding"]
        for name, style in (("random", "s--"), ("lp", "o-")):
            pts = sorted((p["added_tokens_mean"], p["leak_mean"], p["leak_se"]) for p in pad if p["defense"] == name)
            if pts:
                a, lk, sd = map(np.array, zip(*pts))
                ax.errorbar(
                    a,
                    lk,
                    sd,
                    fmt=style,
                    ms=3.5,
                    capsize=2,
                    label={"random": "random delay", "lp": "LP (ε sweep)"}[name],
                )
        for name, m in (("none", "k*"), ("constant", "kD")):
            p = next(p for p in pad if p["defense"] == name)
            ax.errorbar(p["added_tokens_mean"], p["leak_mean"], p["leak_se"], fmt=m, ms=5, label=name)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(left=0)
        ax.set_title(d["_file"].replace("_", " "))
        ax.set_xlabel("added thinking tokens per decision")
    axes[0][0].set_ylabel("leak (mbb/hand)")
    axes[0][-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "padding.png")
    plt.close(fig)


def fig_frontier(analysis, defend, out: Path):
    games = sorted({r["game"] for r in analysis if r["game"] in ("kuhn", "leduc")})
    if not games:
        return
    fig, axes = plt.subplots(1, len(games), figsize=(3.6 * len(games), 3.0), squeeze=False)
    for ax, g in zip(axes[0], games):
        for r in order([r for r in analysis if r["game"] == g]):
            m = short(r["model"])
            distilled = m.startswith("distill")
            marker = "*" if distilled else ("o" if r["mode"] == "think" else "s")
            ax.plot(
                max(r["tokens_per_decision"], 1),
                r["exploitability"],
                marker,
                color=COLORS.get(m, "k"),
                ms=9 if distilled else 6,
                mfc="white" if r["mode"] == "nothink" and not distilled else None,
            )
        for d in defend:
            if d["game"] == g and d["budgets"]:
                pts = sorted((b["compute_tokens"], b["exploitability"]) for b in d["budgets"] if b["lam"] is not None)
                ax.plot(*zip(*pts), ":", color=COLORS.get(short(d["model"]), "k"), lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("thinking tokens per decision (1 = off)")
        ax.set_title(g.capitalize())
    axes[0][0].set_ylabel("exploitability (mbb/hand)")
    handles = [
        plt.Line2D([], [], color=COLORS[m], marker="o", ls="", label=m.replace("Qwen3-", "")) for m in MODEL_ORDER[:4]
    ]
    handles += [
        plt.Line2D([], [], color="k", marker=mk, ls=ls, mfc=fc, label=lab)
        for mk, ls, fc, lab in (
            ("o", "", None, "thinking"),
            ("s", "", "white", "thinking off"),
            ("*", "", None, "distilled"),
            ("", ":", None, "budget schedules"),
        )
    ]
    axes[0][-1].legend(handles=handles, frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "frontier.png")
    plt.close(fig)


def tables(analysis, defend, bench) -> str:
    md = [
        "## Phase 1: does thinking time leak the card?",
        "",
        "| game | model | tokens/decision | truncated | leak (mbb/hand) [95% CI] | null [95% CI] | card headroom | bits gained |",
        "|---|---|---:|---:|---|---|---:|---:|",
    ]
    for r in sorted(order([r for r in analysis if r["mode"] == "think"]), key=lambda r: r["game"]):
        p = r["phase1"]
        md.append(
            f"| {r['game']} | {short(r['model'])} | {r['tokens_per_decision']:.0f} | {r['meta'].get('truncated', 0):.0%} "
            f"| {fmt_ci(p['point']['leak'], p['ci']['leak'])} | {fmt_ci(r['null_leak']['mean'], r['null_leak']['ci'])} "
            f"| {p['point']['card']:.0f} | {r['bits']['bits_gained']:.3f} |"
        )
    md += [
        "",
        "## Strength and cost",
        "",
        "| game | model | mode | exploitability (mbb/hand) | EV vs Nash (mbb/hand) | tokens/decision |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in sorted(order(analysis), key=lambda r: (r["game"], r["mode"] != "think")):
        md.append(
            f"| {r['game']} | {short(r['model'])} | {r['mode']} | {r['exploitability']:.1f} | {r['ev_vs_nash']:.1f} "
            f"| {r['tokens_per_decision']:.0f} |"
        )
    for d in defend:
        md += [
            "",
            f"## Defenses: {d['_file']}",
            "",
            "| defense | parameter | leak (mbb/hand, ± SE over repeats) | added tokens/decision |",
            "|---|---|---:|---:|",
        ]
        for p in d["padding"]:
            md.append(
                f"| {p['defense']} | {p['param']:g} | {p['leak_mean']:.1f} ± {p['leak_se']:.1f} | {p['added_tokens_mean']:.0f} |"
            )
        if d["budgets"]:
            md += [
                "",
                "| λ (mbb per token) | exploitability | reported tokens | computed tokens |",
                "|---|---:|---:|---:|",
            ]
            for b in d["budgets"]:
                lam = "full thinking" if b["lam"] is None else f"{b['lam']:g}"
                md.append(
                    f"| {lam} | {b['exploitability']:.1f} | {b['reported_tokens']:.0f} | {b['compute_tokens']:.0f} |"
                )
    if bench:
        md += [
            "",
            "## Serving benchmark (one A100, vLLM)",
            "",
            "| game | model | mode | decisions/s | GPU-s per 1k hands | p50 ms | p99 ms | tokens/decision | ms/token | R² |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        modes = ("think", "budget", "nothink")
        for b in sorted(order(bench), key=lambda b: (b["game"], modes.index(b["mode"]))):
            r2 = "–" if np.isnan(b["latency_r2"]) else f"{b['latency_r2']:.3f}"
            md.append(
                f"| {b['game']} | {short(b['model'])} | {b['mode']} | {b['decisions_per_s']:.1f} "
                f"| {b['gpu_seconds_per_1000_hands']:.0f} | {b['single_p50_ms']:.0f} | {b['single_p99_ms']:.0f} "
                f"| {b['thinking_tokens_per_decision']:.0f} | {b['ms_per_token']:.2f} | {r2} |"
            )
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    res, fig = Path(a.results), Path(a.figures)
    fig.mkdir(parents=True, exist_ok=True)
    analysis, defend, bench = load(res / "analysis"), load(res / "defend"), load(res / "bench")
    for d in defend:
        src = next((r for r in analysis if r["run"].rstrip("/") == d["run"].rstrip("/")), None)
        d["game"], d["model"] = (src["game"], src["model"]) if src else (d["_file"].split("_")[0], d["_file"])
    fig_tell(analysis, fig)
    fig_leak(analysis, fig)
    fig_padding(defend, fig)
    fig_frontier(analysis, defend, fig)
    (res / "tables.md").write_text(tables(analysis, defend, bench))


if __name__ == "__main__":
    main()
