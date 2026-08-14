#!/bin/bash
#
# DelftBlue GPU launcher for planning/ training runs.
#
# Usage (pass your compute account on the command line):
#     sbatch --account=research-<faculty>-<dept> \
#            planning/experiments/train/delftblue.slurm <run_name> [train.py overrides...]
#
# Examples:
#     # inverse baseline (paper) — TwoRoom, seed 0
#     sbatch --account=... delftblue.slurm tworoom_inverse_lambda_0p1_seed0
#     # our a_{t+1} policy regularizer — TwoRoom, seed 0
#     sbatch --account=... delftblue.slurm tworoom_policy_seed0
#
# Run names come from generate_configs.py (run `python generate_configs.py` first,
# or run.sh will generate them on demand). Data/results live on /scratch.
#
#SBATCH --job-name=smwm
#SBATCH --partition=gpu-a100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch --account=<acct> delftblue.slurm <run_name> [overrides...]" >&2
    exit 2
fi

# --- software stack -----------------------------------------------------------
# The project venv (uv-managed Python) is activated by run.sh; do NOT module-load
# a python here or it will shadow the venv. torch's pip wheels bundle CUDA.
module load 2025
module load cuda

# sbatch copies this script to a spool dir, so BASH_SOURCE/$0 does NOT point at
# the repo. Use the submit directory (run this from planning/experiments/train).
EXP_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -f "$EXP_DIR/run.sh" ]; then
    echo "run.sh not found in '$EXP_DIR'." >&2
    echo "Submit from the experiment dir, e.g.:" >&2
    echo "  cd ~/sensorimotor-world-model/planning/experiments/train" >&2
    echo "  sbatch --account=<acct> delftblue.slurm <run_name>" >&2
    exit 1
fi

# --- storage: keep large / regenerable data off the 30 GB home quota ----------
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-/scratch/${USER}/smwm-data}"
export GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-/scratch/${USER}/smwm-generated}"
export RUNS_ROOT="${RUNS_ROOT:-/scratch/${USER}/smwm-runs}"
mkdir -p "$EXTERNAL_DATA_ROOT" "$GENERATED_DATA_ROOT" "$RUNS_ROOT"

# --- compute nodes have no outbound internet ----------------------------------
# Keep W&B fully local; `wandb sync <dir>` later from a login node if you want it.
export WANDB_MODE="${WANDB_MODE:-offline}"

# Match the dataloader worker count to the CPUs SLURM actually gave us.
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-12}"

echo "======================================================================"
echo "run_name+overrides : $*"
echo "node               : $(hostname)"
echo "gpu(s)             : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUNS_ROOT          : $RUNS_ROOT"
echo "EXTERNAL_DATA_ROOT : $EXTERNAL_DATA_ROOT"
echo "======================================================================"
nvidia-smi -L || true

# run.sh resolves the generated config, activates the venv, and launches train.py.
# Appending num_workers=... overrides base.yaml's default (24) to fit our CPUs.
srun bash "$EXP_DIR/run.sh" "$@" "num_workers=${NUM_WORKERS}"
