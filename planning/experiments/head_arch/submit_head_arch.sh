#!/bin/bash
# Head-architecture sweep on DelftBlue: one job per (dataset, head architecture, weight, seed). Each job
# trains the goal-conditioned policy regularizer without action input, (z_t, z_{t+1}, z_g) -> a_{t+1},
# and then evaluates it (train_eval_head_arch.sbatch). Runs that hit the time limit are resumed, trained
# runs are only evaluated, and evaluated runs and jobs already in the queue are skipped, so the MLP at
# the default weight (the existing runs of the paper) is not trained again.
# Usage: bash submit_head_arch.sh            # dry run: print the job list, write and submit nothing
#        SUBMIT=1 bash submit_head_arch.sh   # write the configs and submit
# Settings (environment variables):
#   DATASETS="tworoom"            training sets: tworoom pusht cube reacher reacher_mix
#   ARCHS="transformer mamba mlp" head architectures (mlp = the existing head, as the reference at every weight)
#   MULTS="0.3 1 3 10"            weights as multiples of the dataset's default weight (TwoRoom 0.1, Push-T 30,
#                                 Cube 1, Reacher 5); 1 = the default
#   LAMBDAS="0.1 1"               explicit weights instead of MULTS (use with one dataset)
#   SEEDS="0"
#   WALL=hh:mm:ss                 time limit per job (default: tworoom 8 h, pusht 14 h, cube 20 h, reacher 16 h)
#   ONLY=<file>                   restrict to the run names listed in <file>
set -euo pipefail
PLANNING=${SMWM_PLANNING:-$HOME/sensorimotor-world-model/planning}   # the checkout the job script trains from
HERE=$PLANNING/experiments/head_arch
JOBS=${SMWM_JOBS:-$HOME/smwm-jobs}
C=${HA_CONFIGS:-$JOBS/head_arch/configs}
R=${SMWM_RUNS:-/scratch/$USER/smwm-runs}; L=${SMWM_LOGS:-$HOME/smwm-logs}
read -r -a EVAL_ROOTS <<< "${SMWM_EVAL_ROOTS:-/scratch/$USER/smwm-warmstart-eval /scratch/$USER/smwm-speed-bench}"
HEAD=c2k1_noact_goalfinal
read -r -a DATASETS <<< "${DATASETS:-tworoom}"
read -r -a ARCHS <<< "${ARCHS:-transformer mamba mlp}"
read -r -a SEEDS <<< "${SEEDS:-0}"
read -r -a MULTS <<< "${MULTS:-0.3 1 3 10}"
read -r -a LAMBDAS <<< "${LAMBDAS:-}"
SUBMIT=${SUBMIT:-0}
LABELS=(cem30 cem10 cem5 policy mean30 mean10 mean5)   # the evals of train_eval_head_arch.sbatch

problem() {  # fatal when submitting, a warning in a dry run
  if [ "$SUBMIT" = 1 ]; then echo "ERROR: $*" >&2; exit 1; else echo "WARNING: $*"; fi
}
env_slug() {  # environment name of the eval pipeline
  case "$1" in tworoom) echo tworoom ;; pusht) echo pusht ;; cube) echo ogbcube ;; reacher|reacher_mix) echo reacher ;;
    *) echo "unknown dataset '$1' (tworoom pusht cube reacher reacher_mix)" >&2; exit 1 ;; esac
}
wall() {
  if [ -n "${WALL:-}" ]; then echo "$WALL"; return; fi
  case "$1" in tworoom) echo 08:00:00 ;; pusht) echo 14:00:00 ;; cube) echo 20:00:00 ;; *) echo 16:00:00 ;; esac
}
if [ ${#LAMBDAS[@]} -gt 0 ]; then weights=(--lambdas "${LAMBDAS[@]}"); else weights=(--multipliers "${MULTS[@]}"); fi
if [ ${#LAMBDAS[@]} -gt 0 ] && [ ${#DATASETS[@]} -gt 1 ]; then
  problem "LAMBDAS gives the same weights to every dataset (${DATASETS[*]}); their default weights differ, use MULTS or one dataset"
fi

seq_heads=0; for a in "${ARCHS[@]}"; do [ "$a" = mlp ] || seq_heads=1; done
if [ "$seq_heads" = 1 ]; then
  grep -q "self.goal_token" "$PLANNING/module.py" 2>/dev/null \
    || problem "$PLANNING/module.py has no goal token for the Transformer / Mamba heads: git pull first"
  grep -q "self.goal_token" "$JOBS/warmstart/module.py" 2>/dev/null \
    || problem "$JOBS/warmstart/module.py (eval pipeline) has no goal token: it should be a link to the checkout's module.py"
fi
[ -f "$JOBS/warmstart/eval_warmstart.sbatch" ] || problem "no eval pipeline in $JOBS/warmstart"

evaluated() {  # $1 = eval env, $2 = run: every label has a result in some job directory of some eval root
  local l root ok
  for l in "${LABELS[@]}"; do
    ok=0
    for root in "${EVAL_ROOTS[@]}"; do
      if ls "$root/$1/$2"/job_*/"$l"/metrics.json >/dev/null 2>&1; then ok=1; break; fi
    done
    [ "$ok" = 1 ] || return 1
  done
}
queued=$(squeue -u "$USER" -h -o %j 2>/dev/null || true)
in_queue=$(squeue -u "$USER" -h 2>/dev/null | wc -l | tr -d ' ' || true)

echo "configs: $C   datasets: ${DATASETS[*]}   heads: ${ARCHS[*]}   weights: ${weights[*]}   seeds: ${SEEDS[*]}"
jobs=(); n_train=0; n_resume=0; n_eval=0; n_skip=0
for ds in "${DATASETS[@]}"; do
  slug=$(env_slug "$ds")
  template=$R/${ds}_policy_mlp_${HEAD}_seed0/config.yaml
  if [ ! -f "$template" ]; then problem "no template $template"; continue; fi
  if [ "$SUBMIT" = 1 ]; then
    mkdir -p "$L"
    python3 "$HERE/make_configs.py" "$C" "$template" --archs "${ARCHS[@]}" "${weights[@]}" --seeds "${SEEDS[@]}" > /dev/null
  fi
  names=$(python3 "$HERE/make_configs.py" --names "$template" --archs "${ARCHS[@]}" "${weights[@]}" --seeds "${SEEDS[@]}" 2>/dev/null) \
    || { python3 "$HERE/make_configs.py" --names "$template" --archs "${ARCHS[@]}" "${weights[@]}" --seeds "${SEEDS[@]}" >/dev/null || true
         problem "make_configs.py failed for $ds"; continue; }
  while read -r run arch lam seed; do
    [ -n "$run" ] || continue
    if [ -n "${ONLY:-}" ] && ! grep -qx "$run" "$ONLY"; then continue; fi
    if evaluated "$slug" "$run"; then echo "SKIP     $run (already evaluated)"; n_skip=$((n_skip + 1)); continue; fi
    if grep -qx "harch-$run" <<< "$queued"; then echo "SKIP     $run (already in the queue)"; n_skip=$((n_skip + 1)); continue; fi
    if [ -s "$R/$run/final_embeddings.pt" ]; then mode=evalonly; n_eval=$((n_eval + 1))
    elif [ -s "$R/$run/checkpoints/last.ckpt" ]; then mode=resume; n_resume=$((n_resume + 1))
    elif [ -e "$R/$run" ]; then
      echo "STALE    $run ($R/$run exists without a checkpoint: move it away to train again)"; n_skip=$((n_skip + 1)); continue
    else mode=train; n_train=$((n_train + 1)); fi
    jobs+=("$mode $run $seed $slug $(wall "$ds") $arch $lam")
  done <<< "$names"
done

if [ "$SUBMIT" = 1 ] && [ $((in_queue + ${#jobs[@]})) -gt 100 ]; then
  echo "ERROR: ${#jobs[@]} new jobs + $in_queue in the queue exceed the limit of 100 submitted jobs" >&2; exit 1
fi
for job in ${jobs[@]+"${jobs[@]}"}; do
  read -r mode run seed slug wall_time arch lam <<< "$job"
  cmd=(sbatch --time="$wall_time" -J "harch-$run" --output="$L/harch-$run-%j.out"
       "$HERE/train_eval_head_arch.sbatch" "$run" "$seed" "$mode" "$slug" "$C")
  if [ "$SUBMIT" = 1 ]; then printf '%-8s %-64s ' "$mode" "$run"; "${cmd[@]}"
  else printf '%-8s %-64s %-12s weight %-6s %s\n' "$mode" "$run" "$arch" "$lam" "$wall_time"; fi
done
echo "train + eval: $n_train, resume + eval: $n_resume, eval only: $n_eval, skipped: $n_skip   (jobs of this account in the queue now: $in_queue)"
[ "$SUBMIT" = 1 ] || echo "dry run: nothing was written or submitted (SUBMIT=1 to submit)"
