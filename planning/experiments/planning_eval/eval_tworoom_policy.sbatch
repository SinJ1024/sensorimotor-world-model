#!/bin/bash
#SBATCH --job-name=tworoom-policy-eval
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=tworoom-policy-eval-%j.out

set -eo pipefail
PROJECT="${SMWM_PROJECT_ROOT:-$HOME/sensorimotor-world-model}"
RUN_DIR="${1:?usage: sbatch eval_tworoom_policy.sbatch /absolute/run_dir [arguments]}"
shift
module load 2025 cuda
source "$PROJECT/.venv/bin/activate"
set -u
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-/scratch/$USER/smwm-data}"
export STABLEWM_HOME="$EXTERNAL_DATA_ROOT"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
OUT="${EVAL_OUTPUT_ROOT:-/scratch/$USER/smwm-evals}/$(basename "$RUN_DIR")/${SLURM_JOB_ID}"
python -u "$PROJECT/planning/eval_tworoom_policy.py" \
    --run-dir "$RUN_DIR" --data-root "$EXTERNAL_DATA_ROOT" \
    --output-dir "$OUT" "$@"
echo "Results: $OUT/summary.json"
