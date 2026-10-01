#!/bin/bash
# Train one run and then evaluate it with the paper CEM protocol, in ONE
# allocation on one Leonardo Booster A100.
#
# Two separate jobs queue twice: when boost_usr_prod is congested the evaluation
# job waits again after training finishes (half a day or more in Oct 2026). Here
# the evaluation reuses the training allocation; billing is unchanged
# (1/4 node = 8 core-hours per wall hour).
#
# Usage:
#   export SBATCH_ACCOUNT=<project account>
#   sbatch --time=<train + eval> train_eval_leonardo.sbatch \
#       <run_name> <base_run> <eval_env> <training_seed> [Hydra overrides...]
#
#   <run_name>  directory under $RUNS_ROOT; passed as subdir= when it differs
#               from <base_run>
#   <eval_env>  tworoom | reacher | pusht | ogbcube
#
# Example:
#   sbatch --time=07:00:00 train_eval_leonardo.sbatch \
#       tworoom_policy_mlp_c1k1_seed0 tworoom_policy_seed0 tworoom 0 \
#       loss.policy.context=1 loss.policy.use_action=false
#
#SBATCH --job-name=smwm-train-eval
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

RUN="${1:?usage: train_eval_leonardo.sbatch <run_name> <base_run> <eval_env> <training_seed> [overrides...]}"
BASE="${2:?missing base run}"
EVAL_ENV="${3:?missing eval env (tworoom|reacher|pusht|ogbcube)}"
SEED="${4:?missing training seed}"
shift 4

PROJECT="${SMWM_PROJECT_ROOT:-$WORK/$USER/sensorimotor-world-model}"
EXP="$PROJECT/planning/experiments"
source "$EXP/leonardo_env.sh"

EXTRA=()
[ "$RUN" != "$BASE" ] && EXTRA+=("subdir=$RUN")
EXTRA+=("wandb.config.name=train_$RUN")

# The two launchers name their tee'd log "<job name>-job<id>.out"; give each a
# distinct name so the evaluation does not truncate the training log.
echo "=== train $RUN (base $BASE) ==="
SLURM_JOB_NAME="train-$RUN" bash "$EXP/train/train_leonardo.sbatch" "$BASE" "$@" "${EXTRA[@]}"

echo "=== evaluate $RUN ($EVAL_ENV, seed $SEED, budget 50) ==="
SLURM_JOB_NAME="eval-$RUN" bash "$EXP/planning_eval/eval_checkpoint_leonardo.sbatch" \
    "$EVAL_ENV" "$RUNS_ROOT/$RUN" "$SEED" 50
