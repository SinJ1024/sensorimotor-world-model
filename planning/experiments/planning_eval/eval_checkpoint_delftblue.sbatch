#!/bin/bash
# Evaluate one completed checkpoint with the paper's Figure 5 CEM protocol.
#
# Usage:
#   sbatch eval_checkpoint_delftblue.sbatch <env> <run_dir> <training_seed>
#
# The policy-regularized checkpoint uses exactly the same CEM evaluator as the
# paper models.  Its auxiliary policy head must exist but is never executed.
#SBATCH --job-name=smwm-paper-eval
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=smwm-paper-eval-%j.out

set -euo pipefail

ENV_SLUG="${1:?usage: eval_checkpoint_delftblue.sbatch <env> <run_dir> <training_seed>}"
RUN_DIR="${2:?missing checkpoint run directory}"
TRAINING_SEED="${3:?missing training seed (0..4)}"
[ "$#" -eq 3 ] || { echo "This paper evaluator accepts exactly three arguments" >&2; exit 2; }

case "$TRAINING_SEED" in
    0|1|2|3|4) ;;
    *) echo "training_seed must be one of 0,1,2,3,4" >&2; exit 2 ;;
esac

case "$ENV_SLUG" in
    tworoom) TASK_SEED=42025; POLICY_SEED_BASE=52025 ;;
    reacher) TASK_SEED=43025; POLICY_SEED_BASE=53025 ;;
    pusht)   TASK_SEED=44025; POLICY_SEED_BASE=54025 ;;
    ogbcube) TASK_SEED=45025; POLICY_SEED_BASE=55025 ;;
    *) echo "env must be tworoom, reacher, pusht, or ogbcube" >&2; exit 2 ;;
esac
POLICY_SEED=$((POLICY_SEED_BASE + TRAINING_SEED))

PROJECT="${SMWM_PROJECT_ROOT:-$HOME/sensorimotor-world-model}"
PLANNING="$PROJECT/planning"
DATA_ROOT="${EXTERNAL_DATA_ROOT:-/scratch/$USER/smwm-data}"
OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/scratch/$USER/smwm-paper-eval}"

[ -f "$RUN_DIR/config.yaml" ] || { echo "missing $RUN_DIR/config.yaml" >&2; exit 3; }
[ -s "$RUN_DIR/checkpoints/last.ckpt" ] || { echo "missing $RUN_DIR/checkpoints/last.ckpt" >&2; exit 3; }
[ -f "$PLANNING/config/eval/env/$ENV_SLUG.yaml" ] || {
    echo "missing paper environment config for $ENV_SLUG" >&2
    exit 3
}

case "$ENV_SLUG" in
    tworoom) DATASET=tworoom_eval.h5 ;;
    reacher) DATASET=reacher_eval.h5 ;;
    pusht)   DATASET=pusht_expert_eval.h5 ;;
    ogbcube) DATASET=cube_single_expert_eval.h5 ;;
esac
[ -s "$DATA_ROOT/$DATASET" ] || { echo "missing $DATA_ROOT/$DATASET" >&2; exit 3; }

module load 2025
module load cuda

# Prefer the existing DelftBlue training environment. The repository .venv may
# be absent or incomplete after a failed uv sync.
if [ -n "${SMWM_PYTHON:-}" ]; then
    PYTHON="$SMWM_PYTHON"
elif [ -x "$HOME/conda-envs/biased-dreams-dreamer/bin/python" ]; then
    PYTHON="$HOME/conda-envs/biased-dreams-dreamer/bin/python"
else
    PYTHON="$PROJECT/.venv/bin/python"
fi
[ -x "$PYTHON" ] || {
    echo "Python executable not found: $PYTHON" >&2
    echo "Set SMWM_PYTHON to the working environment's Python executable." >&2
    exit 4
}
"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import torch
import stable_pretraining
import stable_worldmodel

print("python environment: verified")
print("torch:", torch.__version__)
print("stable-worldmodel:", metadata.version("stable-worldmodel"))
print("stable-pretraining:", metadata.version("stable-pretraining"))
PY

export REPO_ROOT="$PLANNING"
export EXTERNAL_DATA_ROOT="$DATA_ROOT"
export STABLEWM_HOME="$DATA_ROOT"
export MUJOCO_GL=egl
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

RUN_NAME="$(basename "$RUN_DIR")"
JOB_TAG="${SLURM_JOB_ID:-manual}"
JOB_ROOT="$OUTPUT_ROOT/$ENV_SLUG/$RUN_NAME/seed_$TRAINING_SEED/job_$JOB_TAG"
CONFIG_DIR="$JOB_ROOT/config"
RESULT_DIR="$JOB_ROOT/results"
mkdir -p "$CONFIG_DIR" "$RESULT_DIR"
CONFIG="$CONFIG_DIR/paper_eval.yaml"

cat > "$CONFIG" <<YAML
defaults:
  - /eval/base
  - /eval/env/$ENV_SLUG
  - _self_
hydra:
  searchpath: [file://\${oc.env:REPO_ROOT}/config]
seed: $POLICY_SEED
runs:
  - name: seed_$TRAINING_SEED
    run_dir: $RUN_DIR
eval:
  num_eval: 100
  goal_offset_steps: 25
  eval_budget: 50
  task_seed: $TASK_SEED
  save_video: false
YAML

echo "repo          : $(git -C "$PROJECT" rev-parse HEAD)"
echo "environment   : $ENV_SLUG"
echo "checkpoint    : $RUN_DIR/checkpoints/last.ckpt"
echo "training seed : $TRAINING_SEED"
echo "task seed     : $TASK_SEED"
echo "CEM seed      : $POLICY_SEED"
echo "python        : $PYTHON ($("$PYTHON" --version 2>&1))"
echo "result        : $RESULT_DIR/seed_$TRAINING_SEED/metrics.json"
nvidia-smi -L

RUNS_ROOT="$RESULT_DIR" "$PYTHON" -u "$PLANNING/eval.py" --config "$CONFIG"

METRICS="$RESULT_DIR/seed_$TRAINING_SEED/metrics.json"
"$PYTHON" - "$METRICS" <<'PY'
import json
import sys

path = sys.argv[1]
result = json.load(open(path, encoding="utf-8"))
metrics = result.get("metrics", {})
if set(metrics) != {"success_rate"}:
    raise SystemExit(f"expected only success_rate, got {sorted(metrics)}")
print(f"PAPER SUCCESS RATE: {float(metrics['success_rate']):.6g}")
print(f"METRICS: {path}")
PY
