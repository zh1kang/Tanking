# Deviations from PLAN.md

Changes made after the first GPU run, with the evidence that prompted each.

1. **Thinking cap 2048 → 8192 tokens.** A smoke run on Kuhn (Qwen3-0.6B, 4 traces per information set) truncated
   87.5% of traces at 2048. Probes at 8192 (16 traces per information set, Kuhn) truncated 2.6% (0.6B) and 2.1% (8B),
   with median thinking lengths of 4082 and 2983 tokens. A cap that cuts most traces would measure the cap, not the
   model's thinking time. Consequences: constant padding now reports 8192 tokens; padding bins are cap/64 = 128 tokens wide.
2. **Budget grid {64, ..., 2048} → {128, 256, 512, 1024, 2048, 4096} plus full thinking (8192)**, to span the new cap.
3. **Sharded sampling.** Each thinking run is sampled as 4 jobs with seeds 0-3 (Kuhn 128 and Leduc 16 traces per
   information set each, totals unchanged at 512 and 64) and merged on load, so a job failure loses a quarter of a run.
4. **Leduc time limits raised** (1.7B 4 → 10 h, 4B 8 → 14 h) after three 1.7B shards and one 4B shard timed out at
   about 60-70% of their prompts. The timed-out shards were rerun unchanged.
5. **Added after the first results: replication of the Kuhn 4B tell.** Only Qwen3-4B met the "tell confirmed" rule
   (+17.7 mbb/hand [+11.7, +27.7]). Before seeing any replication data, four fresh shards (seeds 4-7, 128 traces per
   information set each, otherwise identical) were sampled into `runs/kuhn/4B/think_rep` and are analyzed alone with
   the unchanged estimator. The tell replicates if that set alone meets the same rule: leak >= 10 mbb/hand, CI
   excluding 0, and a null-control CI containing 0.
6. **`tell_bits` uses early stopping.** With 200 unregularized boosting rounds the classifier overfit the token count,
   so adding thinking time *lowered* held-out log-likelihood (Kuhn 4B: -0.075 bits) even though median thinking time
   differs by 2x between cards at the same decision point. With early stopping, Kuhn 4B gains +0.056 bits and a
   smoothed per-bucket frequency estimator agrees (+0.079). Leak estimates are unaffected; they do not use this
   classifier.
7. **Added after the first results: calibration of the leak estimator on synthetic tells**
   (`experiments/calibrate.py`). It reuses real information sets and action distributions with synthetic thinking
   times of known dependence on the card, at 64 and 512 traces per information set. Delta 0.6 and 1 were added after
   the first Kuhn cells showed that delta ≤ 0.3 plants leaks below the 10 mbb/hand threshold. The Leduc null control
   excluded 0 (0.6B: -55 [-111, -10]); this measures the estimator's bias and coverage at Leduc's sample size.
8. **Serving benchmark scope.** Benchmarked: Kuhn Qwen3-4B (the model with the tell; thinking, budget, thinking off)
   and Leduc Qwen3-0.6B and 1.7B (thinking, thinking off, distilled; budget for 0.6B). The budget policy uses the
   lambda = 0.03 schedule from the defense search. Jobs requested A100 80 GB nodes (`--constraint=gpu80`).
9. **No Leduc run for Qwen3-8B.** The plan required at least 0.6B and 1.7B; 0.6B, 1.7B and 4B were run. Leduc used
   67 GPU-hours against the planned 20, mostly because traces ran 3-4 times longer than the planned 2048-token cap.
