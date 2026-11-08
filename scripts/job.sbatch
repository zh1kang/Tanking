#!/bin/bash
# One Slurm job: runs its arguments as a command on one GPU, offline, with the repo-local cluster environment.
# Usage (from the repo root): sbatch [options] scripts/job.sbatch <command ...>
#SBATCH --job-name=tanking
#SBATCH --account=jordanat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TANKING_OFFLINE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# shellcheck disable=SC1091
. scripts/cluster_env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "no GPU"
"$@"
