#!/bin/bash
# Submit Tanking GPU stages from the repo root on Della. Usage: experiments/submit.sh <stage>
#   kuhn     thinking traces (4 seeded shards of 128 per information set) + a thinking-off pass, every model
#   leduc    thinking traces (4 shards of 16) + a thinking-off pass; MODELS overrides the model list
#   SHARDS="0 2" reruns only those thinking shards (no thinking-off pass), e.g. after a timeout
#   kuhn_rep replication of the Kuhn 4B tell: 4 fresh shards (seeds 4-7) under runs/kuhn/4B/think_rep
#   family   other model families on Kuhn (FAMILY, default "R1-7B Phi4-mini"): shards 0-3 under think,
#            replication shards 4-7 under think_rep
#   calibrate  CPU: leak estimator on synthetic tells (Kuhn 4B and Leduc 0.6B information sets), then the summary
#   heldout  thinking-off passes of the base models on the other train and held-out variants
#   distill  LoRA distillation for 0.6B and 1.7B, then thinking-off passes on every variant
#   bench    serving benchmark on A100-80GB: Kuhn 4B and Leduc 0.6B/1.7B; budget schedules (lambda 0.03) come from
#            DEFEND (default results/defend)
# Shards are merged when loaded (TraceSet.load), so runs/<game>/<model>/think holds shard0..shard3.
set -euo pipefail
cd "$(dirname "$0")/.."
CAP=${CAP:-8192}
BUDGETS=${BUDGETS:-128,256,512,1024,2048,4096}
mkdir -p logs
gpu() { sbatch --parsable --job-name="$1" --time="$2" scripts/job.sbatch "${@:3}"; }

think() {  # game model n_per_shard time
  for s in ${SHARDS:-0 1 2 3}; do
    gpu "tk_$1_$2_s$s" "$4" python experiments/sample.py --game "$1" --model "Qwen/Qwen3-$2" --n "$3" --seed "$s" \
      --max-think "$CAP" --budgets "$BUDGETS" --out "runs/$1/$2/think/shard$s"
  done
  [ -z "${SHARDS:-}" ] || return 0
  gpu "tk_$1_$2_nt" 00:20:00 python experiments/sample.py --game "$1" --model "Qwen/Qwen3-$2" --mode nothink \
    --out "runs/$1/$2/nothink"
}

case "${1:?stage}" in
  kuhn)
    for m in 0.6B 1.7B 4B 8B; do
      case $m in 0.6B | 1.7B) t=02:00:00 ;; 4B) t=03:00:00 ;; 8B) t=04:00:00 ;; esac
      think kuhn $m 128 $t
    done ;;
  leduc)
    for m in ${MODELS:-0.6B 1.7B 4B}; do
      case $m in 0.6B) t=03:00:00 ;; 1.7B) t=10:00:00 ;; 4B) t=14:00:00 ;; *) t=16:00:00 ;; esac
      think leduc $m 16 $t
    done ;;
  kuhn_rep)
    for s in 4 5 6 7; do
      gpu "tk_kuhn_4B_rep_s$s" 03:00:00 python experiments/sample.py --game kuhn --model Qwen/Qwen3-4B --n 128 --seed "$s" \
        --max-think "$CAP" --budgets "$BUDGETS" --out "runs/kuhn/4B/think_rep/shard$s"
    done ;;
  family)
    for m in ${FAMILY:-R1-7B Phi4-mini}; do
      case $m in
        R1-7B) hf=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B t=02:00:00 ;;
        Phi4-mini) hf=microsoft/Phi-4-mini-reasoning t=06:00:00 ;;
        *) echo "unknown family model $m" >&2; exit 1 ;;
      esac
      for s in ${SHARDS:-0 1 2 3 4 5 6 7}; do
        dir=think; [ "$s" -lt 4 ] || dir=think_rep
        gpu "tk_kuhn_${m}_s$s" "$t" python experiments/sample.py --game kuhn --model "$hf" --n 128 --seed "$s" \
          --max-think "$CAP" --out "runs/kuhn/$m/$dir/shard$s"
      done
    done ;;
  calibrate)
    cal() { sbatch --parsable --job-name="tk_cal_$1" --gres=none --time="$2" --mem=32G scripts/job.sbatch \
      python experiments/calibrate.py run "${@:3}"; }
    ids=()
    for g in kuhn:runs/kuhn/4B/think leduc:runs/leduc/0.6B/think; do
      game=${g%%:*} traces=${g#*:}
      for d in ${DELTAS:-0 0.15 0.3 0.6 1}; do
        for n in 64 512; do
          ids+=("$(cal "${game}_n${n}_d$d" 06:00:00 "$traces" --n "$n" --delta "$d" --out "results/calibration/cell_${game}_n${n}_d$d.json")")
        done
        [ "$d" = 0 ] || ids+=("$(cal "${game}_ref_d$d" 04:00:00 "$traces" --n 4096 --delta "$d" --reps 3 --boot 0 \
          --out "results/calibration/cell_${game}_n4096_d$d.json")")
      done
    done
    deps=$(IFS=:; echo "${ids[*]}")
    sbatch --parsable --job-name=tk_cal_summary --gres=none --time=00:10:00 --dependency="afterok:$deps" scripts/job.sbatch \
      python experiments/calibrate.py summarize results/calibration ;;
  heldout)
    for m in 0.6B 1.7B 4B 8B; do
      cmd="for g in kuhn4 kuhn_b2 leduc_b12 leduc4 kuhn5 kuhn4_b2 leduc_b36 leduc4_b12; do
        python experiments/sample.py --game \$g --model Qwen/Qwen3-$m --mode nothink --out runs/\$g/$m/nothink || exit 1; done"
      gpu "tk_heldout_$m" 01:00:00 bash -c "$cmd"
    done ;;
  distill)
    for m in 0.6B 1.7B; do
      j=$(gpu "tk_distill_$m" 03:00:00 python experiments/distill.py --model "Qwen/Qwen3-$m" --out "checkpoints/distill_$m")
      cmd="for g in kuhn kuhn4 kuhn_b2 leduc leduc_b12 leduc4 kuhn5 kuhn4_b2 leduc_b36 leduc4_b12; do
        python experiments/sample.py --game \$g --model checkpoints/distill_$m --mode nothink --out runs/\$g/distill_$m/nothink || exit 1; done"
      sbatch --parsable --job-name="tk_dist_eval_$m" --time=01:00:00 --dependency="afterok:$j" scripts/job.sbatch bash -c "$cmd"
    done ;;
  bench)
    bench() {  # game model mode time [extra args]
      sbatch --parsable --job-name="tk_bench_$1_${2##*/}_$3" --time="$4" --constraint=gpu80 scripts/job.sbatch \
        python experiments/bench.py --game "$1" --model "$2" --mode "$3" --out "results/bench/$1_${2##*/}_$3.json" "${@:5}"
    }
    schedule() {  # game model -> path of the lambda 0.03 schedule
      local out="results/bench/schedules/$1_$2.json"
      python -c "import json, sys; d = json.load(open(sys.argv[1]));
json.dump(next(b['schedule'] for b in d['budgets'] if b['lam'] == 0.03), open(sys.argv[2], 'w'), indent=1)" \
        "${DEFEND:-results/defend}/$1_$2.json" "$out"
      echo "$out"
    }
    mkdir -p results/bench/schedules
    bench kuhn Qwen/Qwen3-4B think 04:00:00
    bench kuhn Qwen/Qwen3-4B budget 03:00:00 --schedule "$(schedule kuhn 4B)"
    bench kuhn Qwen/Qwen3-4B nothink 00:30:00
    for m in 0.6B 1.7B; do
      bench leduc "Qwen/Qwen3-$m" think 03:00:00
      bench leduc "Qwen/Qwen3-$m" nothink 00:30:00
      bench leduc "checkpoints/distill_$m" nothink 00:30:00
    done
    bench leduc Qwen/Qwen3-0.6B budget 02:00:00 --schedule "$(schedule leduc 0.6B)" ;;
  *) echo "unknown stage $1" >&2; exit 1 ;;
esac
