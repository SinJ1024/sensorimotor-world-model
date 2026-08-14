#!/bin/bash
#
# Self-contained DelftBlue launcher (does NOT use run.sh) — Barkla-style.
#
# Usage:
#   sbatch --account=<acct> train_standalone.sbatch <run_name> [train.py overrides...]
# Example:
#   sbatch --account=education-eemcs-msc-dsait train_standalone.sbatch tworoom_policy_seed0
#
#SBATCH --job-name=smwm
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch --account=<acct> train_standalone.sbatch <run_name> [overrides...]" >&2
    exit 2
fi
RUN_NAME="$1"; shift

# --- edit this one line if your clone lives elsewhere -------------------------
PROJECT="$HOME/sensorimotor-world-model"
PLANNING="$PROJECT/planning"

# --- software stack (venv is uv-managed; do NOT module-load python) -----------
module load 2025
module load cuda
source "$PROJECT/.venv/bin/activate"

# --- env: data/results on scratch, W&B offline (no internet on compute) -------
export REPO_ROOT="$PLANNING"                       # generated configs' hydra searchpath
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-/scratch/$USER/smwm-data}"
export GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-/scratch/$USER/smwm-generated}"
export RUNS_ROOT="${RUNS_ROOT:-/scratch/$USER/smwm-runs}"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "$EXTERNAL_DATA_ROOT" "$GENERATED_DATA_ROOT" "$RUNS_ROOT"

NUM_WORKERS="${SLURM_CPUS_PER_TASK:-12}"
CONFIG_DIR="$PLANNING/experiments/train/generated_configs"
if [ ! -f "$CONFIG_DIR/$RUN_NAME.yaml" ]; then
    python "$PLANNING/experiments/train/generate_configs.py" --output-dir "$CONFIG_DIR"
fi

echo "======================================================================"
echo "run_name : $RUN_NAME    overrides: $*"
echo "node     : $(hostname)    gpu: ${CUDA_VISIBLE_DEVICES:-unset}    workers: $NUM_WORKERS"
echo "RUNS_ROOT: $RUNS_ROOT"
echo "======================================================================"
nvidia-smi -L || true

cd "$PLANNING"
srun python -u train.py \
    --config-path "$CONFIG_DIR" \
    --config-name "$RUN_NAME" \
    num_workers="$NUM_WORKERS" "$@"
