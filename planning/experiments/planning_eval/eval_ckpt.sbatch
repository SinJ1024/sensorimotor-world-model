#!/bin/bash
#SBATCH --job-name=smwm-evalckpt
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=evalckpt-%x-%j.out
set -uo pipefail
ENV_SLUG="${1:?usage: eval_ckpt.sbatch <env> <run_dir> [overrides...]}"
RUN_DIR="${2:?need run_dir}"; shift 2
case "$ENV_SLUG" in tworoom|reacher|pusht|ogbcube) ;; *) echo "bad env: $ENV_SLUG"; exit 2 ;; esac
if [ ! -f "$RUN_DIR/config.yaml" ] || [ ! -f "$RUN_DIR/checkpoints/last.ckpt" ]; then
    echo "run_dir missing config.yaml or checkpoints/last.ckpt: $RUN_DIR"; exit 3
fi
PROJECT="$HOME/sensorimotor-world-model"; PLANNING="$PROJECT/planning"
EXP="$PLANNING/experiments/planning_eval"
{ module load 2025 cuda; source "$PROJECT/.venv/bin/activate"; } >/dev/null 2>&1
export REPO_ROOT="$PLANNING"
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-/scratch/$USER/smwm-data}"
export STABLEWM_HOME="${STABLEWM_HOME:-$EXTERNAL_DATA_ROOT}"
export MUJOCO_GL=egl WANDB_MODE=offline PYTHONUNBUFFERED=1
NAME="$(basename "$RUN_DIR")"; CFGDIR="$EXP/generated_configs"; mkdir -p "$CFGDIR"
CFG="$CFGDIR/_bydir_${ENV_SLUG}_${NAME}.yaml"
cat > "$CFG" <<YAML
defaults:
  - /eval/base
  - /eval/env/$ENV_SLUG
  - _self_
hydra:
  searchpath: [file://\${oc.env:REPO_ROOT}/config]
seed: 42
runs:
  - name: $NAME
    run_dir: $RUN_DIR
eval:
  num_eval: 100
  goal_offset_steps: 25
  eval_budget: 50
  task_seed: 42025
  save_video: false
YAML
OUT="$EXP/results/$ENV_SLUG/_bydir"; mkdir -p "$OUT"; LOG="$(mktemp)"
RUNS_ROOT="$OUT" python -u "$PLANNING/eval.py" --config "$CFG" "$@" >"$LOG" 2>&1
RC=$?
echo "=================== EVAL REPORT ==================="
echo "env      : $ENV_SLUG"; echo "run_dir  : $RUN_DIR"; [ -n "$*" ] && echo "overrides: $*"
LOADLINE="$(grep -i "load_state_dict" "$LOG" | head -1)"
if [ -n "$LOADLINE" ]; then echo "ckpt load: MISMATCH -> $LOADLINE"; else echo "ckpt load: clean (all keys matched)"; fi
if [ "$RC" -eq 0 ]; then
    echo "status   : OK (exit 0)"
    python - "$OUT/$NAME/metrics.json" <<'PY'
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p)); m = d.get("metrics", {})
    scal = {k: v for k, v in m.items() if isinstance(v,(int,float,bool))} if isinstance(m,dict) else m
    print("metrics  :", json.dumps(scal, ensure_ascii=False))
    el = d.get("elapsed_seconds")
    if el is not None: print(f"elapsed  : {float(el):.1f}s")
except Exception as e:
    print("metrics  : (could not read", p, "->", repr(e), ")")
PY
else
    echo "status   : FAILED (exit $RC)"
    grep -nE "Error|Traceback|Exception|assert|Missing|CUDA" "$LOG" | tail -12
    tail -20 "$LOG"
fi
echo "full log : $LOG"; echo "=================================================="
exit "$RC"
