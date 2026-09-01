#!/bin/bash
#
# DAIC (Delft AI Cluster) launcher — L40 by default, bf16.
# Usage:  sbatch train_daic.sbatch <run_name> [train.py overrides...]
# Blackwell instead of L40 (needs a cu128 torch):
#   sbatch --gres=gpu:nvidia_rtx_pro_6000:1 train_daic.sbatch <run_name> ...
#
#SBATCH --job-name=smwm
#SBATCH --partition=general
#SBATCH --account=ewi-insy-sdm
#SBATCH --qos=short
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: sbatch train_daic.sbatch <run_name> [overrides...]" >&2
    exit 2
fi
RUN_NAME="$1"; shift

# --- edit these two lines if your clone/data live elsewhere -------------------
PROJECT=/tudelft.net/staff-umbrella/mscworldmodels/sensorimotor-world-model
DATAROOT=/tudelft.net/staff-umbrella/mscworldmodels/smwm-data
PLANNING="$PROJECT/planning"

module load 2025/gpu                       # DAIC GPU stack (CUDA etc.)
source "$PROJECT/.venv/bin/activate"       # uv-managed venv (do NOT module-load python)

export REPO_ROOT="$PLANNING"
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-$DATAROOT}"
export GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-$DATAROOT/generated}"
export RUNS_ROOT="${RUNS_ROOT:-/tudelft.net/staff-umbrella/mscworldmodels/smwm-runs}"
export WANDB_MODE="${WANDB_MODE:-offline}"  # compute nodes have no internet
export MUJOCO_GL="${MUJOCO_GL:-egl}"        # headless render for eval
mkdir -p "$EXTERNAL_DATA_ROOT" "$GENERATED_DATA_ROOT" "$RUNS_ROOT"

NUM_WORKERS="${SLURM_CPUS_PER_TASK:-12}"
CONFIG_DIR="$PLANNING/experiments/train/generated_configs"
if [ ! -f "$CONFIG_DIR/$RUN_NAME.yaml" ]; then
    python "$PLANNING/experiments/train/generate_configs.py" --output-dir "$CONFIG_DIR"
fi

echo "======================================================================"
echo "run_name+overrides : $RUN_NAME $*"
echo "node               : $(hostname)   gpu: ${CUDA_VISIBLE_DEVICES:-unset}   workers: $NUM_WORKERS"
echo "RUNS_ROOT          : $RUNS_ROOT"
echo "======================================================================"
nvidia-smi -L || true

# L40/Blackwell support bf16 -> use base.yaml precision (do NOT pass 16-mixed).
cd "$PLANNING"
srun python -u train.py \
    --config-path "$CONFIG_DIR" \
    --config-name "$RUN_NAME" \
    num_workers="$NUM_WORKERS" "$@"
