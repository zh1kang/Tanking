#!/bin/bash
# Login node only: builds .venv (reusing the Suture uv package cache for hardlinks) and downloads the Qwen3 weights.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR=/scratch/gpfs/JORDANAT/caleb/suture/.cache/uv
. scripts/cluster_env.sh
UV=/scratch/gpfs/JORDANAT/caleb/suture/bin/uv
$UV sync --extra gpu
. .venv/bin/activate
for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B; do hf download Qwen/$m --exclude "*.gguf" >/dev/null && echo "downloaded $m"; done
python -c "import vllm, torch, peft, transformers; print(vllm.__version__, torch.__version__, peft.__version__, transformers.__version__)"
