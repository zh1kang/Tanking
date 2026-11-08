# Tanking: plan

Reasoning models think longer on hard decisions. In poker, how long a player takes ("tanking") is a tell.
This project measures how much a reasoning LLM's thinking time reveals about its private card, priced in chips,
then compares fixes that remove the leak while making the model play better and run faster. No reinforcement learning:
everything is exact game-tree computation, linear programming, or supervised fine-tuning.

Written before any result exists; changes after the first run go in `DEVIATIONS.md`.

## Setup

- Games: Kuhn poker (3 cards, one betting round) and Leduc poker (6 cards, two rounds, public board card).
  Both are small enough for exact Nash equilibria (CFR+) and exact best responses. A parametrized family
  (deck size, bet sizes, raise cap) supplies variants for the transfer test.
- Models: Qwen3-0.6B, 1.7B, 4B, 8B in thinking mode (Kuhn: all four; Leduc: as many as the budget allows, at least 0.6B and 1.7B).
- Policy estimation: for every information set (own card, board, betting history), sample N thinking traces
  (Kuhn N=512, Leduc N=64; temperature 0.6, top-p 0.95, top-k 20, thinking cap 2048 tokens). After each trace,
  force `Action:` and read the probabilities of the legal action tokens. The policy at an information set is the
  average of those distributions (Rao-Blackwellized, so sampling noise comes from the traces only).
- Observable thinking time T = number of thinking tokens. A single-stream serving benchmark checks that wall-clock
  time is linear in T (expected R^2 > 0.95) and converts tokens to milliseconds.
- Units: chips per hand, reported as mbb/hand (ante = 1 big blind, 1 chip = 1000 mbb), averaged over both seats.

## Phase 1: measure the tell

- An opponent best-responds to the LLM's policy, seeing either (a) actions only, (b) actions plus T bucketed into
  k quantile bins per decision point (Kuhn k=4, Leduc k=3), or (c) the LLM's card (headroom).
- Overfitting control: traces in each information set are split in half. The best response is computed on one half
  and its value evaluated against the other half, then the halves swap. This underestimates the true best-response
  value, so the measured gain is conservative.
- Leak = value(b) - value(a). 95% CI from 200 bootstrap resamples of traces within information sets.
- Null control: each trace gets one sampled action; T is shuffled across private cards within (decision point, action).
  The gain under this control must have a CI containing 0.
- Secondary: held-out log-loss of a classifier predicting the LLM's card from (decision point, action) with and
  without T, reported in bits per decision.

Pre-registered reading (Kuhn, any model):
- Tell confirmed: leak >= 10 mbb/hand with CI excluding 0, and the null control CI contains 0.
- No tell: leak < 2 mbb/hand for every model. Then the defense phase is reported as moot and the project's
  headline becomes the optimization results.
- In between: reported as a weak tell.

## Phase 2: defenses (each scored on leak, latency cost and playing strength)

1. Constant padding: always report the cap (2048 tokens). Zero leak, maximal latency.
2. Random padding: add Uniform[0, P] tokens. Shows how inefficient naive noise is.
3. Optimal padding by linear program: for each decision point and action, find padding kernels K_card(t' | t), t' >= t
   (padding can only add delay), minimizing expected added tokens subject to the padded distributions being within
   total variation epsilon of a common distribution. epsilon = 0 gives zero leak; sweeping epsilon gives the frontier.
   Fit on one half of the traces, evaluated on the other.
4. Public-information thinking budget: the budget may depend only on the betting history (never on the card), and
   the reported time is always the budget, so the leak is zero by construction. Traces longer than the budget are
   truncated and the answer forced. A coordinate-descent search picks per-decision-point budgets from
   {64, 128, 256, 512, 1024, 2048} trading exploitability against mean tokens.

## Phase 3: optimization (play better and faster)

- Baseline: the same models with thinking off.
- Solver-distilled LoRA: supervised fine-tuning of Qwen3-0.6B and 1.7B (thinking off, LoRA r=16) to match the CFR+
  equilibrium action distribution at every information set of training variants (KL loss on the legal-action tokens).
  Evaluated on standard Kuhn and Leduc (in the training set) and on held-out variants (bet sizes and deck sizes not seen).
- Serving benchmark (vLLM, one A100): decisions per second in batch, single-stream p50/p99 latency, and GPU-seconds per
  1,000 hands for thinking, budgeted, thinking-off, and distilled policies.
- Strength metrics: exploitability (cross-fitted best response, mbb/hand) and expected value against the equilibrium bot.

## Deliverables

`README.md` with the tell figure (thinking tokens by card), the leak table, the padding frontier, and one chart placing
every method on exploitability vs tokens per decision with leak marked; `RESULTS.md` with all numbers; a small CLI demo
where a timing-reading bot names the LLM's card.

## Budget

Target under 60 A100-hours. Kuhn phase about 5 GPU-hours; Leduc about 20; distillation and benchmarks about 10.
