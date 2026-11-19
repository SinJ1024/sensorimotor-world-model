#!/bin/bash
# Hindsight-goal-distance ablation (DelftBlue): train one run of the policy regularizer whose goal
# is the frame G frames after the last context frame (loss.policy.goal_source: offset with
# goal_frame_only), then evaluate it with the warm-start eval pipeline: same tasks and CEM seeds as
# every other eval (goal 25 steps away, budget 50, CEM seed = base + training seed).
#   plain CEM at 30, 10 and 5 iterations             cem30 cem10 cem5
#   the policy head alone                            policy
#   the policy proposal as the initial CEM mean      mean30 mean10 mean5
# The config <run_name>.yaml comes from make_configs.py (submit_goal_distance.sh writes it).
# Usage: sbatch [opts] train_eval_goal_distance.sbatch <run_name> <seed> <train|resume|evalonly> [<eval_env> [<config_dir>]]
#   train:    train from scratch; refuses to train over an existing run directory
#   resume:   continue a run whose training hit the time limit (last.ckpt, no final_embeddings.pt)
#   evalonly: evaluate a run that finished training
#   <eval_env>:   environment name of eval_warmstart.sbatch (default reacher)
#   <config_dir>: default $HOME/smwm-jobs/goal_distance/configs
#SBATCH --job-name=smwm-gdist
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
set -euo pipefail
RUN=$1; SEED=$2; MODE=$3; ENV_SLUG=${4:-reacher}; C=${5:-$HOME/smwm-jobs/goal_distance/configs}
W=$HOME/smwm-jobs/warmstart; R=/scratch/$USER/smwm-runs; SRC=/scratch/$USER/smwm-data
PLANNING=$HOME/sensorimotor-world-model/planning
case "$MODE" in train|resume|evalonly) ;; *) echo "mode must be train, resume or evalonly, got '$MODE'" >&2; exit 2 ;; esac

if [ "$MODE" = train ] || [ "$MODE" = resume ]; then
  CFG=$C/$RUN.yaml
  [ -s "$CFG" ] || { echo "no config $CFG: run make_configs.py first" >&2; exit 2; }
  grep -q "^    goal_frame_only: true$" "$CFG" || { echo "$CFG does not set loss.policy.goal_frame_only" >&2; exit 2; }
  # A train.py from before this ablation ignores goal_frame_only and would load and encode every
  # frame up to the goal, on other clips.
  grep -q "def policy_goal_frame_only" "$PLANNING/train.py" || { echo "$PLANNING/train.py has no goal_frame_only support: git pull first" >&2; exit 2; }
  EXTRA=()
  if [ "$MODE" = train ]; then
    if [ -e "$R/$RUN" ]; then echo "run dir $R/$RUN already exists; refusing to train over it" >&2; exit 3; fi
  else
    [ -s "$R/$RUN/checkpoints/last.ckpt" ] || { echo "resume: no $R/$RUN/checkpoints/last.ckpt" >&2; exit 3; }
    [ -e "$R/$RUN/final_embeddings.pt" ] && { echo "resume: $RUN already finished training" >&2; exit 3; }
    EXTRA=(training.resume=true)
  fi
  module load 2025; module load cuda
  export PATH=/scratch/$USER/envs/smwm-eval-py312-torch251-cu121/bin:$PATH
  # stage this run's training h5 to node-local disk if it fits (as train_eval_seed.sbatch does)
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
  export REPO_ROOT="$PLANNING" GENERATED_DATA_ROOT=/scratch/$USER/smwm-generated RUNS_ROOT="$R" WANDB_MODE=offline
  # wandb-core can need more than its default 30 s to start when several jobs start on one node.
  export WANDB__SERVICE_WAIT=300
  echo "=== $MODE $RUN (seed $SEED) $(date)"
  cd "$PLANNING"
  srun python -u train.py --config-path "$C" --config-name "$RUN" num_workers="${SLURM_CPUS_PER_TASK:-12}" ${EXTRA[@]+"${EXTRA[@]}"}
  [ -s "$R/$RUN/checkpoints/last.ckpt" ] || { echo "training produced no checkpoint" >&2; exit 4; }
fi
# train.py writes final_embeddings.pt only after the last epoch: never evaluate a partial run
[ -s "$R/$RUN/final_embeddings.pt" ] || { echo "training of $RUN did not finish (no final_embeddings.pt)" >&2; exit 5; }

labels=()
for n in 30 10 5; do labels+=("cem$n:solver.n_steps=$n"); done
labels+=("policy:warmstart.mode=policy")
for n in 30 10 5; do labels+=("mean$n:solver.n_steps=$n warmstart.mode=mean"); done
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
