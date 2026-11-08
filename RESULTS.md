# Tanking: results

**Summary.** One of four Qwen3 reasoning models leaks its private card through thinking time. In Kuhn poker,
Qwen3-4B's thinking time is worth +17.7 mbb/hand to an opponent, and the leak replicated on fresh samples (+19.7).
Optimal padding removes it at a third of the cost of padding to the cap. Leduc poker could not be measured reliably at
the sample size the budget allowed, and a calibration on synthetic tells shows why. Thinking improves play, roughly
halving the larger models' exploitability in Kuhn. But a solver-distilled LoRA with thinking off is 5–30 times less
exploitable than any thinking model, over 2,000 times cheaper to serve, and generalizes to held-out game variants.
Total compute: 96 GPU-hours.

The plan is [`PLAN.md`](PLAN.md); deviations 1–9 are in [`DEVIATIONS.md`](DEVIATIONS.md). The most consequential is the
thinking cap, raised from 2048 to 8192 tokens because 2048 truncated 87% of traces. Every table below is generated
into [`results/tables.md`](results/tables.md) from the JSON files under `results/`.

Units: mbb/hand (1 chip = 1000 mbb), averaged over both seats. "Leak" is the extra value an exact best-responding
opponent gains by seeing thinking time bucketed into quantiles (Kuhn 4 buckets, Leduc 3), on top of seeing actions.
It is cross-fitted across halves of the traces, so it is conservative.

## Phase 1: does thinking time leak the card?

| Game | Model | Thinking tokens/decision | Truncated at cap | Leak [95% CI] | Null control [95% CI] | Bits gained |
|---|---|---:|---:|---|---|---:|
| Kuhn | Qwen3-0.6B | 4,379 | 4% | −1.0 [−3.5, +3.0] | −0.6 [−2.9, +2.2] | −0.019 |
| Kuhn | Qwen3-1.7B | 6,409 | 19% | +0.3 [−7.7, +8.2] | −1.6 [−8.1, 0.0] | −0.003 |
| Kuhn | **Qwen3-4B** | 4,984 | 8% | **+17.7 [+11.7, +27.7]** | −2.5 [−8.4, +2.5] | +0.056 |
| Kuhn | Qwen3-8B | 3,982 | 3% | +0.9 [−0.3, +3.2] | −0.3 [−0.9, 0.0] | +0.116 |
| Leduc | Qwen3-0.6B | 4,210 | 3% | −65.1 [−112.2, +10.9] | −55.3 [−110.9, −10.1] | +0.005 |
| Leduc | Qwen3-1.7B | 7,610 | 49% | −16.2 [−70.0, +75.3] | −62.8 [−124.9, −4.9] | +0.027 |
| Leduc | Qwen3-4B | 7,578 | 43% | +70.5 [−10.6, +143.2] | −41.5 [−84.1, −4.6] | +0.159 |

Kuhn has 512 traces per information set and Leduc 64. "Bits gained" is the held-out log-likelihood gain, per
decision, of a classifier predicting the card from (decision point, action) when thinking time is added (deviation 6).

**Pre-registered reading: tell confirmed (Kuhn, Qwen3-4B).** The leak is at least 10 mbb/hand with a CI excluding 0,
and the null control (thinking time shuffled across cards within decision point and action) contains 0. The other
three models are below 2 mbb/hand.

**Replication (deviation 5, rule fixed before the data).** Four fresh shards (seeds 4–7, 512 traces per information set)
analyzed alone give +19.7 [+11.1, +34.6], with the null control at −3.5 [−10.1, +0.7]. The tell replicates.

**Mechanism.** Median thinking tokens for Qwen3-4B by card:

| Decision point | Opponent acts afterwards? | Jack | Queen | King |
|---|---|---:|---:|---:|
| Player 1 opens | yes | 6,227 | 6,675 | 5,149 |
| Player 2 after a check | yes | 5,610 | 5,663 | 4,127 |
| Player 1 facing a bet after checking | no | 4,001 | 2,749 | 2,036 |
| Player 2 facing a bet | no | 3,922 | 2,923 | 2,104 |

With a king, 4B thinks 17–27% less at the decisions its opponent can react to. The largest gap (about 2×) appears when
it faces a bet, which is its last action of the hand, so that part cannot be exploited. See `figures/tell.png`.

**Information is not exploitability.** Qwen3-8B's thinking time carries twice as much information about its card as
4B's (+0.116 against +0.056 bits per decision), yet its leak is +0.9 mbb/hand. Its timing differences at the exploitable
decision points are comparable in size (`figures/tell.png`). They evidently do not change what the best-responding
opponent should do.

### Calibration: why Leduc is inconclusive (deviation 7)

The Leduc null controls exclude 0 for every model, which the pre-registered rule does not allow. To see whether that
is a property of the estimator, it was run on synthetic thinking times with a planted tell of known size:
`log t = 8 − δ · rank / (n_ranks − 1) + N(0, 0.5²)`, on the real information sets and action distributions. The
reference value for δ > 0 is the estimator at 4,096 traces per information set. 20 replicates per cell.

| Game | δ | Traces per infoset | True leak | Mean estimate | CI covers truth | Tell rule fires | Null contains 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| Kuhn | 0 | 64 | 0 | −10.3 | 100% | 0% | 90% |
| Kuhn | 0 | 512 | 0 | −3.0 | 100% | 0% | 95% |
| Kuhn | 0.6 | 64 | 30.2 | 19.7 | 100% | 30% | 100% |
| Kuhn | 0.6 | 512 | 30.2 | 29.0 | 95% | 90% | 90% |
| Kuhn | 1 | 512 | 55.4 | 52.5 | 80% | 75% | 75% |
| Leduc | 0 | 64 | 0 | −57.7 | 85% | 0% | 5% |
| Leduc | 0 | 512 | 0 | −4.5 | 100% | 0% | 90% |
| Leduc | 0.6 | 64 | 92.2 | 16.2 | 100% | 0% | 0% |
| Leduc | 0.6 | 512 | 92.2 | 84.7 | 100% | 90% | 90% |
| Leduc | 1 | 64 | 239.8 | 150.9 | 95% | 5% | 5% |
| Leduc | 1 | 512 | 239.8 | 233.8 | 100% | 85% | 85% |

All cells, including δ = 0.15 and 0.3, are in `results/calibration/summary.md`.

- **The rule has no false positives.** With no planted tell it never fired, in either game or at either sample size.
- **At Kuhn's sample size the estimator works.** At 512 traces it is biased by −1 to −3 mbb/hand and detects a
  30 mbb/hand tell 90% of the time. So 4B's +17.7 is credible and, if anything, slightly understated.
- **At Leduc's sample size it does not.** At 64 traces it is biased by −58 to −89 mbb/hand, and the null control
  excludes 0 almost always, so even a 240 mbb/hand tell fires the rule 5% of the time. The real Leduc results show the
  same pattern (null controls at −42 to −63). Leduc would need about 8 times more sampling, roughly 150–200
  GPU-hours per model at the observed thinking lengths, so it is reported as inconclusive.
- The confidence intervals are conservative (coverage 95–100%), except for very strong tells at 512 traces (80%).

## Phase 2: defenses (Kuhn, Qwen3-4B)

Leak measured on the held-out half of the traces (the padding is fit on the other half), mean ± SE over 10 repeats.
Unpadded, this protocol gives 15.2, slightly below the full-data 17.7.

| Defense | Leak | Added thinking tokens per decision |
|---|---:|---:|
| None | 15.2 ± 2.1 | 0 |
| Constant: always report 8,192 | 0.0 | 3,206 |
| Random delay, Uniform[0, 4096] | 10.1 ± 2.1 | 2,047 |
| Random delay, Uniform[0, 8192] | 0.5 ± 1.9 | 4,110 |
| **LP-optimal padding, ε = 0** | −4.6 ± 1.3 | **990** |
| LP, ε = 0.05 | −6.5 ± 1.2 | 649 |
| LP, ε = 0.1 | 4.4 ± 1.7 | 400 |
| LP, ε = 0.2 | 17.2 ± 1.6 | 139 |

Negative values are the estimator's small-sample bias, so they read as zero. The linear program removes the leak with
31% of constant padding's added delay and about a quarter of random padding's. ε trades delay against leakage, and the
leak returns by ε = 0.2. See `figures/padding.png`.

**Public-information budgets.** The thinking budget depends only on the betting history, and the reported time is
always the budget, so the leak is zero by construction.

| Budget trade-off λ (mbb per token) | Exploitability | Reported tokens | Computed tokens |
|---|---:|---:|---:|
| Full thinking, unpadded | 145.0 | 5,005 | 5,005 |
| 0 (best play) | 127.5 | 7,251 | 4,691 |
| 0.03 | 233.6 | 2,106 | 1,595 |
| 0.1 | 333.3 | 128 | 128 |

At λ = 0 the schedule caps only Player 2's decision after a check, at 4,096 tokens, and leaves the rest at the cap. It
thinks less and plays slightly better than unrestricted thinking (no CI on this difference). But the opponent sees a
constant 7,251 tokens, more delay than LP padding adds.

Leduc defense tables are in `results/tables.md`. They are not interpretable, because the Leduc leak itself is not
distinguishable from the estimator's bias.

## Phase 3: playing better and faster

Exploitability is the exact best-response value against the model's policy (cross-fitted for thinking runs).

| Policy | Kuhn exploitability | Leduc exploitability | Thinking tokens/decision |
|---|---:|---:|---:|
| Thinking: 0.6B / 1.7B / 4B / 8B | 330 / 265 / 141 / 167 | 1,487 / 1,708 / 1,587 / — | 3,982–7,610 |
| Thinking off: 0.6B / 1.7B / 4B / 8B | 332 / 333 / 333 / 334 | 1,509 / 2,367 / 2,361 / — | 0 |
| **Solver-distilled, thinking off: 0.6B / 1.7B** | **12.1 / 4.7** | **242 / 325** | 0 |

Thinking roughly halves exploitability for 4B and 8B in Kuhn, and barely changes it for 0.6B. The distilled LoRAs
are 12–30 times less exploitable than the best thinking model in Kuhn and 5–6 times in Leduc. Expected value
against the equilibrium bot tells the same story (Kuhn: thinking −23 to −130, distilled about 0).

**Held-out variants** (deck and bet sizes not in the distillation set), thinking off:

| Variant | Base 0.6B → distilled | Base 1.7B → distilled |
|---|---|---|
| Kuhn, 5 cards | 289 → 70 | 200 → 87 |
| Kuhn, 4 cards, bet 2 | 617 → 82 | 500 → 117 |
| Leduc, bets 3/6 | 2,267 → 389 | 3,600 → 595 |
| Leduc, 4 ranks, bets 1/2 | 777 → 390 | 1,143 → 337 |

The distilled models are 2–8 times less exploitable than the same base model on every held-out variant.

**Serving benchmark** (one A100 80 GB, vLLM; decisions drawn from equilibrium self-play):

| Game | Policy | GPU-seconds per 1,000 hands | Single-stream p50 / p99 | ms per thinking token | Latency R² vs tokens |
|---|---|---:|---|---:|---:|
| Kuhn | 4B thinking | 5,029 | 35.1 s / 48.2 s | 7.9 | 1.000 |
| Kuhn | 4B budget (λ = 0.03) | 1,100 | 0.96 s / 31.2 s | 7.6 | 1.000 |
| Kuhn | 4B thinking off | 1 | 10 ms / 10 ms | — | — |
| Leduc | 0.6B thinking | 3,577 | 8.6 s / 23.2 s | 2.9 | 0.999 |
| Leduc | 0.6B budget (λ = 0.03) | 1,145 | 1.6 s / 16.7 s | 2.7 | 0.999 |
| Leduc | 1.7B thinking | 10,875 | 36.3 s / 36.3 s | 4.6 | 1.000 |
| Leduc | 0.6B / 1.7B thinking off | 1 / 2 | 7 ms / 8 ms | — | — |
| Leduc | 0.6B / 1.7B distilled | 2 / 2 | 7 ms / 9 ms | — | — |

Single-stream latency is linear in thinking tokens (R² ≥ 0.999, as the plan required at > 0.95), so token counts
are what a timing opponent sees. The distilled models cost 2,300–6,200 times less GPU time per hand than thinking
models of the same size (Leduc), and answer in under 10 ms. See `figures/frontier.png` for every policy on exploitability
against thinking tokens.

## Caveats

- **The opponent is an exact best response** to the measured policy, which bounds what a learning opponent could
  extract. A real opponent would have to learn the timing pattern from far fewer hands.
- **Thinking time is measured in tokens** on a dedicated GPU. Shared serving adds latency noise that would blur
  the tell, which makes the token-based leak an upper bound on what is visible through wall-clock time.
- **One model family, one prompt format.** The tell appears in one of four sizes and is not monotone in scale.
- **The distilled models train on the standard games' equilibria**, so Kuhn and Leduc exploitability for them is
  in-distribution. The held-out table is the fair test of generalization.

## Cost

| Stage | GPU-hours |
|---|---:|
| Smoke tests and thinking-cap probes | 0.3 |
| Kuhn thinking and thinking-off sampling, 4 models | 16.6 |
| Kuhn 4B replication | 3.9 |
| Leduc thinking and thinking-off sampling, 3 models (including timed-out shards) | 69.0 |
| Distillation and evaluation; held-out variants | 1.1 |
| Serving benchmark | 4.8 |
| **Total** | **95.7** |

Analysis, defenses and calibration ran on CPU. The plan targeted under 60 GPU-hours; Leduc took 69 against an
estimated 20, because thinking ran 3–4 times longer than the planned 2048-token cap (deviation 1).
