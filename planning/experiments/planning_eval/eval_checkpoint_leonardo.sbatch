#!/bin/bash
# Evaluate one completed checkpoint with the paper's Figure 5 CEM protocol on Leonardo.
# Same protocol, seeds and output layout as eval_checkpoint_delftblue.sbatch, so
# scripts/summarize_runs.py --eval-root $EVAL_OUTPUT_ROOT reads the results unchanged.
#
# Usage:
#   export SBATCH_ACCOUNT=<project account>
#   sbatch eval_checkpoint_leonardo.sbatch <env> <run_dir> <training_seed> [50|150]
#
# Regularizer heads (policy_model.*, goal_policy.*) are stripped by eval.py; the CEM
# planner only uses the encoder + forward model, so every run is scored by one code path.
#SBATCH --job-name=smwm-paper-eval
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
trap 'status=$?; if [ "$status" -ne 0 ]; then echo "FAILED status=$status line=$LINENO command=$BASH_COMMAND" >&2; fi' EXIT

ENV_SLUG="${1:?usage: eval_checkpoint_leonardo.sbatch <env> <run_dir> <training_seed> [50|150]}"
RUN_DIR="${2:?missing checkpoint run directory}"
TRAINING_SEED="${3:?missing training seed (0..4)}"
EVAL_BUDGET="${4:-50}"
[ "$#" -ge 3 ] && [ "$#" -le 4 ] || {
    echo "This paper evaluator accepts three arguments and one optional budget" >&2
    exit 2
}

case "$EVAL_BUDGET" in
    50|150) ;;
    *) echo "eval budget must be 50 or 150" >&2; exit 2 ;;
esac

case "$TRAINING_SEED" in
    0|1|2|3|4) ;;
    *) echo "training_seed must be one of 0,1,2,3,4" >&2; exit 2 ;;
esac

# Identical task / CEM seeds to the DelftBlue evaluator (results are comparable).
case "$ENV_SLUG" in
    tworoom) TASK_SEED=42025; POLICY_SEED_BASE=52025; DATASET=tworoom_eval.h5 ;;
    reacher) TASK_SEED=43025; POLICY_SEED_BASE=53025; DATASET=reacher_eval.h5 ;;
    pusht)   TASK_SEED=44025; POLICY_SEED_BASE=54025; DATASET=pusht_expert_eval.h5 ;;
    ogbcube) TASK_SEED=45025; POLICY_SEED_BASE=55025; DATASET=cube_single_expert_eval.h5 ;;
    *) echo "env must be tworoom, reacher, pusht, or ogbcube" >&2; exit 2 ;;
esac
POLICY_SEED=$((POLICY_SEED_BASE + TRAINING_SEED))

PROJECT="${SMWM_PROJECT_ROOT:-$WORK/$USER/sensorimotor-world-model}"
source "$PROJECT/planning/experiments/leonardo_env.sh"
PLANNING="$PROJECT/planning"
PYTHON="${SMWM_PYTHON:-$PROJECT/.venv/bin/python}"

JOB_TAG="${SLURM_JOB_ID:-manual}"
LOGFILE="$SMWM_LOG_DIR/${SLURM_JOB_NAME:-smwm-paper-eval}-job${JOB_TAG}.out"
exec > >(tee "$LOGFILE") 2>&1

[ -f "$RUN_DIR/config.yaml" ] || { echo "missing $RUN_DIR/config.yaml" >&2; exit 3; }
[ -s "$RUN_DIR/checkpoints/last.ckpt" ] || { echo "missing $RUN_DIR/checkpoints/last.ckpt" >&2; exit 3; }
[ -f "$PLANNING/config/eval/env/$ENV_SLUG.yaml" ] || {
    echo "missing paper environment config for $ENV_SLUG" >&2
    exit 3
}
[ -s "$EXTERNAL_DATA_ROOT/$DATASET" ] || { echo "missing $EXTERNAL_DATA_ROOT/$DATASET" >&2; exit 3; }
[ -x "$PYTHON" ] || { echo "missing venv python: $PYTHON (run 'uv sync', see planning/LEONARDO.md)" >&2; exit 4; }

module load cuda >/dev/null 2>&1 || true

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

RUN_NAME="$(basename "$RUN_DIR")"
JOB_ROOT="$EVAL_OUTPUT_ROOT/$ENV_SLUG/$RUN_NAME/seed_$TRAINING_SEED/budget_$EVAL_BUDGET/job_$JOB_TAG"
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
  searchpath: ["file://\${oc.env:REPO_ROOT}/config"]
seed: $POLICY_SEED
runs:
  - name: seed_$TRAINING_SEED
    run_dir: $RUN_DIR
eval:
  num_eval: 100
  goal_offset_steps: 25
  eval_budget: $EVAL_BUDGET
  task_seed: $TASK_SEED
  save_video: false
YAML

echo "repo          : $(git -C "$PROJECT" rev-parse HEAD)"
echo "environment   : $ENV_SLUG"
echo "checkpoint    : $RUN_DIR/checkpoints/last.ckpt"
echo "training seed : $TRAINING_SEED"
echo "task seed     : $TASK_SEED"
echo "CEM seed      : $POLICY_SEED"
echo "eval budget   : $EVAL_BUDGET"
echo "python        : $PYTHON ($("$PYTHON" --version 2>&1))"
echo "result        : $RESULT_DIR/seed_$TRAINING_SEED/metrics.json"
echo "log           : $LOGFILE"
nvidia-smi -L

RUNS_ROOT="$RESULT_DIR" srun "$PYTHON" -u "$PLANNING/eval.py" --config "$CONFIG"

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
