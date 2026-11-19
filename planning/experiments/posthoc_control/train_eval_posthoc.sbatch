#!/bin/bash
# Post-hoc policy head control (DelftBlue). SMWM (inverse dynamics) and LeWM (SIGReg) have no head that can
# propose an action, so they cannot warm-start CEM as they are. This job trains the paper's policy head,
# (z_t, z_{t+1}, z_g) -> a_{t+1}, on the frozen latents of such a finished run (no gradient into the world
# model; scripts/train_posthoc_policy.py) and evaluates it with the warm-start eval pipeline: same tasks
# and CEM seeds as every other eval (goal 25 steps away, budget 50, CEM seed = base + training seed).
#   the policy head alone                            policy
#   the policy proposal as the initial CEM mean      mean30 mean10 mean5
# Plain CEM is not run again: the world model is the source run's, and its plain-CEM evals exist.
# PLAIN_CEM=1 adds cem30 cem10 cem5 as a check that the copied world model plans like the source run.
# Usage: sbatch [opts] train_eval_posthoc.sbatch <source_run> <posthoc_run> <seed> <train|evalonly> <eval_env>
#   train:    train the head for 10 epochs, then evaluate. An unfinished post-hoc run of the same source is
#             trained again from the start; any other existing directory is refused.
#   evalonly: evaluate a post-hoc run that finished training (posthoc_done.json)
#   <eval_env>: environment name of eval_warmstart.sbatch: tworoom | pusht | ogbcube | reacher
#SBATCH --job-name=smwm-posthoc
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
set -euo pipefail
SRC_RUN=$1; RUN=$2; SEED=$3; MODE=$4; ENV_SLUG=$5
W=$HOME/smwm-jobs/warmstart; R=/scratch/$USER/smwm-runs; SRC=/scratch/$USER/smwm-data
PLANNING=$HOME/sensorimotor-world-model/planning
case "$MODE" in train|evalonly) ;; *) echo "mode must be train or evalonly, got '$MODE'" >&2; exit 2 ;; esac
case "$ENV_SLUG" in tworoom|pusht|ogbcube|reacher) ;; *) echo "eval env must be tworoom, pusht, ogbcube or reacher, got '$ENV_SLUG'" >&2; exit 2 ;; esac
[ "$SRC_RUN" != "$RUN" ] || { echo "the post-hoc run needs its own name" >&2; exit 2; }

if [ "$MODE" = train ]; then
  CFG=$R/$SRC_RUN/config.yaml
  [ -s "$CFG" ] && [ -s "$R/$SRC_RUN/checkpoints/last.ckpt" ] || { echo "no finished source run $R/$SRC_RUN" >&2; exit 3; }
  [ -f "$PLANNING/scripts/train_posthoc_policy.py" ] || { echo "no $PLANNING/scripts/train_posthoc_policy.py: git pull first" >&2; exit 2; }
  module load 2025; module load cuda
  export PATH=/scratch/$USER/envs/smwm-eval-py312-torch251-cu121/bin:$PATH
  # stage the source run's training h5 to node-local disk if it fits (as train_eval_head_arch.sbatch does)
  F=$(awk '/^data:/{d=1} d && /^    name:/{print $2; exit}' "$CFG").h5
  STAGE="${TMPDIR:-/tmp}/$USER/smwm-data-$SLURM_JOB_ID"; mkdir -p "$STAGE"
  NEED=$(( $(stat -c %s "$SRC/$F") / 1073741824 + 2 ))
  AVAIL=$(df -BG --output=avail "$STAGE" | tail -1 | tr -dc 0-9)
  echo "node $(hostname): local disk ${AVAIL}G, need ${NEED}G for $F"
  if [ "$AVAIL" -gt "$NEED" ]; then
    ln -s "$SRC"/* "$STAGE"/ 2>/dev/null || true; rm -f "$STAGE/$F"
    echo "staging $F"; time cp "$SRC/$F" "$STAGE/"
    trap 'rm -rf "$STAGE"' EXIT
    export EXTERNAL_DATA_ROOT="$STAGE"
  else
    echo "not enough local disk; reading from $SRC"; export EXTERNAL_DATA_ROOT="$SRC"
  fi
  export REPO_ROOT="$PLANNING" WANDB_MODE=offline
  echo "=== post-hoc head on $SRC_RUN -> $RUN (seed $SEED) $(date)"
  cd "$PLANNING"
  srun python -u scripts/train_posthoc_policy.py --src "$R/$SRC_RUN" --out-root "$R" \
      --head "$RUN:endpoints:final" --num-workers "${SLURM_CPUS_PER_TASK:-12}"
fi
# the script writes posthoc_done.json only after the last epoch: never evaluate a partial head
[ -s "$R/$RUN/posthoc_done.json" ] || { echo "training of $RUN did not finish (no posthoc_done.json)" >&2; exit 5; }
cat "$R/$RUN/posthoc_done.json"; echo

labels=("policy:warmstart.mode=policy")
for n in 30 10 5; do labels+=("mean$n:solver.n_steps=$n warmstart.mode=mean"); done
if [ "${PLAIN_CEM:-0}" = 1 ]; then for n in 30 10 5; do labels+=("cem$n:solver.n_steps=$n"); done; fi
echo "=== evaluate $RUN ($ENV_SLUG, seed $SEED): ${#labels[@]} evals $(date)"
bash "$W/eval_warmstart.sbatch" "$ENV_SLUG" "$R/$RUN" "$SEED" "${labels[@]}"
# eval_warmstart.sbatch goes on after a failed eval; report it here so the job does not look complete
OUT=/scratch/$USER/smwm-warmstart-eval/$ENV_SLUG/$RUN/job_${SLURM_JOB_ID:-manual}
missing=0
for spec in "${labels[@]}"; do
  [ -s "$OUT/${spec%%:*}/metrics.json" ] || { echo "no result for ${spec%%:*} (see $OUT/${spec%%:*}/eval.log)" >&2; missing=1; }
done
[ "$missing" = 0 ] || exit 6
echo "=== done $(date)"
