#!/bin/bash
# CPU post-processing for every finished run under runs/: analysis, defenses (thinking runs), figures and tables.
# Usage: experiments/postprocess.sh [run dirs...]   (default: all runs except smoke and probe)
# Needs the project environment active (scripts/cluster_env.sh on the cluster, `uv run` locally).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
RUNS=${*:-$(find runs -mindepth 3 -maxdepth 3 -type d \( -name think -o -name nothink \) \
  -not -path 'runs/smoke/*' -not -path 'runs/probe/*' | sort)}
python experiments/analyze.py $RUNS --out results/analysis
for r in $RUNS; do
  [ "$(basename "$r")" = think ] || continue
  game=$(echo "$r" | cut -d/ -f2) model=$(echo "$r" | cut -d/ -f3)
  reps=10; [ "$game" = leduc ] && reps=20
  python experiments/defend.py "$r" --repeats $reps --out "results/defend/${game}_${model}.json" > /dev/null
done
python experiments/report.py
