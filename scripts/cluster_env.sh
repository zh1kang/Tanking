# Source before any tanking command on Della: `. scripts/cluster_env.sh`.
# Every cache lives under the repo's .cache directory; nothing is written to $HOME or other scratch dirs.
# TANKING_OFFLINE=1 (compute nodes have no internet) makes Hugging Face read only the cache filled on the login node.
TANKING_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TANKING_ROOT
C=$TANKING_ROOT/.cache
export HF_HOME=$C/hf
export UV_CACHE_DIR=${UV_CACHE_DIR:-$C/uv} UV_PYTHON_INSTALL_DIR=$C/uv-python PIP_CACHE_DIR=$C/pip
export XDG_CACHE_HOME=$C/xdg XDG_CONFIG_HOME=$C/config MPLCONFIGDIR=$C/matplotlib
export TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor CUDA_CACHE_PATH=$C/nv
export VLLM_CACHE_ROOT=$C/vllm VLLM_CONFIG_ROOT=$C/vllm-config FLASHINFER_WORKSPACE_BASE=$C
export TORCH_HOME=$C/torch PYTHONPYCACHEPREFIX=$C/pycache TMPDIR=$C/tmp
mkdir -p "$TMPDIR"
export UV_PROJECT_ENVIRONMENT=$TANKING_ROOT/.venv
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$TANKING_ROOT${PYTHONPATH:+:$PYTHONPATH}
unset TRANSFORMERS_CACHE HF_HUB_CACHE HF_DATASETS_CACHE
if [ -n "${TANKING_OFFLINE:-}" ]; then O=1; else O=0; fi
export HF_HUB_OFFLINE=$O HF_DATASETS_OFFLINE=$O TRANSFORMERS_OFFLINE=$O
if [ -f "$TANKING_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$TANKING_ROOT/.venv/bin/activate"
fi
