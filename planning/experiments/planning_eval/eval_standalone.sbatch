#!/bin/bash
#
# Self-contained DelftBlue planning-eval launcher (does NOT use run.sh).
#
# Usage:
#   sbatch --account=<acct> eval_standalone.sbatch <env> <method> <seed> [eval.py overrides...]
# Example (TwoRoom, our policy regularizer, seed 0):
#   sbatch --account=education-eemcs-msc-dsait eval_standalone.sbatch tworoom policy 0
#
#SBATCH --job-name=smwm-eval
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: sbatch --account=<acct> eval_standalone.sbatch <env> <method> <seed> [overrides...]" >&2
    echo "  env: tworoom|reacher|pusht|ogbcube   method: inverse|policy|sigreg|forward_only|random" >&2
    exit 2
fi
ENV_SLUG="$1"; METHOD="$2"; SEED="$3"; shift 3

PROJECT="$HOME/sensorimotor-world-model"
PLANNING="$PROJECT/planning"
EXP="$PLANNING/experiments/planning_eval"

module load 2025
module load cuda
source "$PROJECT/.venv/bin/activate"

export REPO_ROOT="$PLANNING"
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-/scratch/$USER/smwm-data}"
export STABLEWM_HOME="${STABLEWM_HOME:-$EXTERNAL_DATA_ROOT}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"          # headless rendering on GPU nodes
export WANDB_MODE="${WANDB_MODE:-offline}"

# eval reads checkpoints from $REPO_ROOT/experiments/train/results/<run>, but our
# training wrote to /scratch/$USER/smwm-runs. Bridge with an idempotent symlink.
TRAIN_RESULTS="$PLANNING/experiments/train/results"
TRAIN_RUNS_ROOT="${TRAIN_RUNS_ROOT:-/scratch/$USER/smwm-runs}"
if [ ! -e "$TRAIN_RESULTS" ]; then
    ln -s "$TRAIN_RUNS_ROOT" "$TRAIN_RESULTS"
    echo "linked $TRAIN_RESULTS -> $TRAIN_RUNS_ROOT"
fi

RUN_LABEL="seed_${SEED}"; [ "$METHOD" = "random" ] && RUN_LABEL="repeat_${SEED}"
JOB_NAME="${ENV_SLUG}_${METHOD}_${RUN_LABEL}"
CONFIG_ROOT="$EXP/generated_configs"
CFG="$CONFIG_ROOT/$JOB_NAME.yaml"
if [ ! -f "$CFG" ]; then
    python "$EXP/generate_configs.py" --output-dir "$CONFIG_ROOT"
fi
[ -f "$CFG" ] || { echo "missing eval config $CFG" >&2; exit 1; }

# Per-method output dir; eval.py writes <RUNS_ROOT>/<run_label>/metrics.json,
# which matches the manifest's result_dir so aggregate_results.py finds it.
METHOD_ROOT="$EXP/results/$ENV_SLUG/$METHOD"
mkdir -p "$METHOD_ROOT"

echo "======================================================================"
echo "eval job : $JOB_NAME    overrides: $*"
echo "node     : $(hostname)    gpu: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "output   : $METHOD_ROOT/$RUN_LABEL"
echo "======================================================================"
nvidia-smi -L || true

cd "$PLANNING"
srun env RUNS_ROOT="$METHOD_ROOT" python -u eval.py --config "$CFG" "$@"
