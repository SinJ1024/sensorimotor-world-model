#!/bin/bash
#
# Policy-regularizer lambda sweep on Leonardo (advisor request, HANDOFF.md section 5):
#
#   Push-T  policy mlp c2k1  lambda in {3, 10, 100, 300}   (paper inverse lambda = 30)
#   Reacher policy mlp c2k1  lambda in {0.5, 50, 500}      (paper inverse lambda = 5)
#
# Per lambda: one training job (1 A100, ~4 h) and one paper-protocol CEM evaluation
# chained with --dependency=afterok. Run names follow the TwoRoom sweep,
# <env>_policy_mlp_c2k1_lam<x>_seed<s> with "." -> "p" (0.5 -> lam0p5), so
# scripts/summarize_runs.py recovers the expected lambda from the name.
#
# Usage (from anywhere on a Leonardo login node):
#   export SBATCH_ACCOUNT=<project account>          # `saldo -b` lists it
#   ./submit_policy_lambda_sweep_leonardo.sh                 # the full 7-run sweep
#   ./submit_policy_lambda_sweep_leonardo.sh pusht:3 reacher:0.5   # a subset
#   DRY_RUN=1 ./submit_policy_lambda_sweep_leonardo.sh       # print, do not submit
#   SMWM_SEED=1 ./submit_policy_lambda_sweep_leonardo.sh     # another training seed
#
# Idempotent: a run whose checkpoints/last.ckpt already exists only gets the eval
# job; a non-empty run directory without last.ckpt is skipped (inspect or delete it).
#
set -euo pipefail

SEED="${SMWM_SEED:-0}"
EVAL_BUDGET="${SMWM_EVAL_BUDGET:-50}"
DRY_RUN="${DRY_RUN:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS="$(cd "$HERE/.." && pwd)"
export SMWM_PROJECT_ROOT="${SMWM_PROJECT_ROOT:-$(cd "$EXPERIMENTS/../.." && pwd)}"
source "$EXPERIMENTS/leonardo_env.sh"

TRAIN_SBATCH="$EXPERIMENTS/train/train_leonardo.sbatch"
EVAL_SBATCH="$EXPERIMENTS/planning_eval/eval_checkpoint_leonardo.sbatch"

[ -n "${SBATCH_ACCOUNT:-}" ] || { echo "export SBATCH_ACCOUNT=<project account> first (see 'saldo -b')" >&2; exit 2; }
[ -x "$SMWM_PROJECT_ROOT/.venv/bin/python" ] || { echo "missing venv at $SMWM_PROJECT_ROOT/.venv - run 'uv sync' (planning/LEONARDO.md)" >&2; exit 2; }

if [ "$#" -gt 0 ]; then
    SWEEP=("$@")
else
    SWEEP=(pusht:3 pusht:10 pusht:100 pusht:300 reacher:0.5 reacher:50 reacher:500)
fi

submit() {  # prints the job id, or a fake one in dry-run mode
    if [ "$DRY_RUN" = "1" ]; then
        echo "   sbatch $*" >&2
        echo "DRY"
    else
        sbatch --parsable "$@"
    fi
}

echo ">> account=$SBATCH_ACCOUNT seed=$SEED budget=$EVAL_BUDGET runs=$RUNS_ROOT evals=$EVAL_OUTPUT_ROOT"
for item in "${SWEEP[@]}"; do
    ENV_SLUG="${item%%:*}"
    LAMBDA="${item#*:}"
    case "$ENV_SLUG" in
        pusht)   TRAIN_H5=pusht_expert_train.h5; EVAL_H5=pusht_expert_eval.h5 ;;
        reacher) TRAIN_H5=reacher_train.h5;      EVAL_H5=reacher_eval.h5 ;;
        tworoom) TRAIN_H5=tworoom_train.h5;      EVAL_H5=tworoom_eval.h5 ;;
        *) echo "unsupported env '$ENV_SLUG' in '$item' (pusht|reacher|tworoom)" >&2; exit 2 ;;
    esac
    for f in "$TRAIN_H5" "$EVAL_H5"; do
        [ -s "$EXTERNAL_DATA_ROOT/$f" ] || { echo "missing $EXTERNAL_DATA_ROOT/$f (rsync it to \$FAST first)" >&2; exit 3; }
    done

    BASE_RUN="${ENV_SLUG}_policy_seed${SEED}"
    RUN_NAME="${ENV_SLUG}_policy_mlp_c2k1_lam${LAMBDA//./p}_seed${SEED}"
    RUN_DIR="$RUNS_ROOT/$RUN_NAME"

    TRAIN_JID=""
    if [ -s "$RUN_DIR/checkpoints/last.ckpt" ]; then
        echo ">> $RUN_NAME: checkpoint exists, submitting eval only"
    elif [ -e "$RUN_DIR" ] && [ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo ">> $RUN_NAME: SKIPPED - $RUN_DIR is non-empty but has no last.ckpt (crashed/running?)"
        continue
    else
        TRAIN_JID=$(submit --job-name="train-$RUN_NAME" \
            --output="$SMWM_LOG_DIR/train-$RUN_NAME-%j.out" \
            "$TRAIN_SBATCH" "$BASE_RUN" \
            "subdir=$RUN_NAME" \
            "loss.inverse.weight=0" \
            "loss.policy.weight=$LAMBDA" \
            "loss.policy.arch=mlp" "loss.policy.context=2" "loss.policy.num_future=1" \
            "wandb.config.name=train_$RUN_NAME")
        echo ">> $RUN_NAME: train job $TRAIN_JID  (lambda=$LAMBDA)"
    fi

    EVAL_ARGS=(--job-name="eval-$RUN_NAME" --output="$SMWM_LOG_DIR/eval-$RUN_NAME-%j.out")
    [ -n "$TRAIN_JID" ] && EVAL_ARGS+=(--dependency="afterok:$TRAIN_JID")
    EVAL_JID=$(submit "${EVAL_ARGS[@]}" "$EVAL_SBATCH" "$ENV_SLUG" "$RUN_DIR" "$SEED" "$EVAL_BUDGET")
    echo "   eval job $EVAL_JID${TRAIN_JID:+ (after $TRAIN_JID)}"
done

echo ">> done. Monitor:  squeue -u \$USER --format='%.10i %.40j %.8T %.10M %R'"
echo ">> summary once evals finish:"
echo "   $SMWM_PROJECT_ROOT/.venv/bin/python $SMWM_PROJECT_ROOT/planning/scripts/summarize_runs.py --runs-root $RUNS_ROOT --eval-root $EVAL_OUTPUT_ROOT --budget $EVAL_BUDGET"
