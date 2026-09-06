#!/bin/bash
#SBATCH --job-name=smwm-eval
#SBATCH --partition=gpu
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=eval-%x-%j.out
set -uo pipefail
ENV_SLUG="${1:?usage: eval_report.sbatch <env> <method> <seed> [overrides...]}"
METHOD="${2:?need method}"; SEED="${3:?need seed}"; shift 3
PROJECT="$HOME/sensorimotor-world-model"; PLANNING="$PROJECT/planning"
EXP="$PLANNING/experiments/planning_eval"
{ module load 2025 cuda; source "$PROJECT/.venv/bin/activate"; } >/dev/null 2>&1
export REPO_ROOT="$PLANNING"
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-/scratch/$USER/smwm-data}"
export STABLEWM_HOME="${STABLEWM_HOME:-$EXTERNAL_DATA_ROOT}"
export TRAIN_RUNS_ROOT="${TRAIN_RUNS_ROOT:-/scratch/$USER/smwm-runs}"
export MUJOCO_GL=egl WANDB_MODE=offline PYTHONUNBUFFERED=1
[ -e "$PLANNING/experiments/train/results" ] || ln -sfn "$TRAIN_RUNS_ROOT" "$PLANNING/experiments/train/results"
RUN_LABEL="seed_${SEED}"; [ "$METHOD" = random ] && RUN_LABEL="repeat_${SEED}"
JOB="${ENV_SLUG}_${METHOD}_${RUN_LABEL}"; CFG="$EXP/generated_configs/$JOB.yaml"
[ -f "$CFG" ] || python "$EXP/generate_configs.py" --output-dir "$EXP/generated_configs" >/dev/null 2>&1
if [ ! -f "$CFG" ]; then echo "job=$JOB status=CONFIG_NOT_FOUND ($CFG)"; exit 3; fi
OUT="$EXP/results/$ENV_SLUG/$METHOD"; mkdir -p "$OUT"; LOG="$(mktemp)"
RUNS_ROOT="$OUT" python -u "$PLANNING/eval.py" --config "$CFG" "$@" >"$LOG" 2>&1
RC=$?
echo "=================== EVAL REPORT ==================="
echo "job      : $JOB"; [ -n "$*" ] && echo "overrides: $*"
if [ "$RC" -eq 0 ]; then
    echo "status   : OK (exit 0)"
    python - "$OUT/$RUN_LABEL/metrics.json" <<'PY'
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p)); m = d.get("metrics", {})
    print("metrics  :", json.dumps(m, ensure_ascii=False))
    for k, v in (m.items() if isinstance(m, dict) else []):
        if any(t in k.lower() for t in ("success","succ","solved","reach")):
            print(f"  -> {k}: {v}")
    el = d.get("elapsed_seconds")
    if el is not None: print(f"elapsed  : {float(el):.1f}s")
except FileNotFoundError:
    print("metrics  : (exit 0 but no metrics.json at", p, ")")
except Exception as e:
    print("metrics  : (could not read", p, "->", repr(e), ")")
PY
else
    echo "status   : FAILED (exit $RC)"
    echo "--- error lines ----------------------------------"
    grep -nE "Error|Traceback|Exception|assert|Missing|raise|CUDA" "$LOG" | tail -12
    echo "--- last 20 lines of log -------------------------"
    tail -20 "$LOG"
fi
echo "full log : $LOG"
echo "=================================================="
exit "$RC"
