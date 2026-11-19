#!/bin/bash
# Post-hoc policy head control on DelftBlue: one job per (dataset, source model, seed). Each job trains the
# paper's policy head on the frozen latents of a finished SMWM or LeWM run and evaluates it alone and as a
# CEM warm-start (train_eval_posthoc.sbatch). Evaluated runs and jobs already in the queue are skipped;
# a run whose head finished training is only evaluated.
# Usage: bash submit_posthoc.sh            # dry run: print the job list, submit nothing
#        SUBMIT=1 bash submit_posthoc.sh   # submit
# Settings (environment variables):
#   DATASETS="pusht cube"      tworoom pusht cube reacher reacher_mix (the warm-start changes little on the others)
#   MODELS="inverse sigreg"    source models: inverse = SMWM, sigreg = LeWM
#   SEEDS="0"                  seed of the source run (also the CEM seed offset of the evals)
#   WALL=hh:mm:ss              time limit per job (default: tworoom 5 h, pusht 10 h, cube 14 h, reacher 12 h:
#                              the time of a full training run plus the evals, an upper bound, since only
#                              the head is trained; the speed has not been measured yet)
#   PLAIN_CEM=1                also evaluate plain CEM (cem30 cem10 cem5) on the copied world model
set -euo pipefail
PLANNING=${SMWM_PLANNING:-$HOME/sensorimotor-world-model/planning}   # the checkout the job script trains from
HERE=$PLANNING/experiments/posthoc_control
JOBS=${SMWM_JOBS:-$HOME/smwm-jobs}
R=${SMWM_RUNS:-/scratch/$USER/smwm-runs}; L=${SMWM_LOGS:-$HOME/smwm-logs}
read -r -a EVAL_ROOTS <<< "${SMWM_EVAL_ROOTS:-/scratch/$USER/smwm-warmstart-eval}"
read -r -a DATASETS <<< "${DATASETS:-pusht cube}"
read -r -a MODELS <<< "${MODELS:-inverse sigreg}"
read -r -a SEEDS <<< "${SEEDS:-0}"
SUBMIT=${SUBMIT:-0}
LABELS=(policy mean30 mean10 mean5)   # the evals of train_eval_posthoc.sbatch
[ "${PLAIN_CEM:-0}" = 1 ] && LABELS+=(cem30 cem10 cem5)

problem() {  # fatal when submitting, a warning in a dry run
  if [ "$SUBMIT" = 1 ]; then echo "ERROR: $*" >&2; exit 1; else echo "WARNING: $*"; fi
}
env_slug() {  # environment name of the eval pipeline
  case "$1" in tworoom) echo tworoom ;; pusht) echo pusht ;; cube) echo ogbcube ;; reacher|reacher_mix) echo reacher ;;
    *) echo "unknown dataset '$1' (tworoom pusht cube reacher reacher_mix)" >&2; exit 1 ;; esac
}
source_run() {  # $1 = dataset, $2 = model, $3 = seed: the h1 run of the paper's tables
  case "$2" in
    sigreg) echo "${1}_sigreg_seed$3" ;;
    inverse) case "$1" in tworoom) echo "tworoom_inverse_lambda_0p1_seed$3" ;; pusht) echo "pusht_inverse_lambda_30_seed$3" ;;
               cube) echo "cube_inverse_lambda_1_seed$3" ;; reacher) echo "reacher_inverse_lambda_1_seed$3" ;;
               reacher_mix) echo "reacher_mix_inverse_seed$3" ;; esac ;;
    *) echo "unknown model '$2' (inverse sigreg)" >&2; exit 1 ;;
  esac
}
wall() {
  if [ -n "${WALL:-}" ]; then echo "$WALL"; return; fi
  case "$1" in tworoom) echo 05:00:00 ;; pusht) echo 10:00:00 ;; cube) echo 14:00:00 ;; *) echo 12:00:00 ;; esac
}
[ -f "$PLANNING/scripts/train_posthoc_policy.py" ] || problem "no $PLANNING/scripts/train_posthoc_policy.py: git pull first"
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

echo "datasets: ${DATASETS[*]}   source models: ${MODELS[*]}   seeds: ${SEEDS[*]}   evals: ${LABELS[*]}"
jobs=(); n_train=0; n_eval=0; n_skip=0
for ds in "${DATASETS[@]}"; do
  slug=$(env_slug "$ds")
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      src=$(source_run "$ds" "$model" "$seed")
      run=${ds}_${model}_posthoc_goalfinal_seed$seed
      if [ ! -s "$R/$src/checkpoints/last.ckpt" ] || [ ! -s "$R/$src/config.yaml" ]; then
        problem "no finished source run $R/$src"; continue
      fi
      if evaluated "$slug" "$run"; then echo "SKIP     $run (already evaluated)"; n_skip=$((n_skip + 1)); continue; fi
      if grep -qx "posthoc-$run" <<< "$queued"; then echo "SKIP     $run (already in the queue)"; n_skip=$((n_skip + 1)); continue; fi
      if [ -s "$R/$run/posthoc_done.json" ]; then mode=evalonly; n_eval=$((n_eval + 1))
      else mode=train; n_train=$((n_train + 1)); fi
      jobs+=("$mode $src $run $seed $slug $(wall "$ds")")
    done
  done
done

if [ "$SUBMIT" = 1 ] && [ $((in_queue + ${#jobs[@]})) -gt 100 ]; then
  echo "ERROR: ${#jobs[@]} new jobs + $in_queue in the queue exceed the limit of 100 submitted jobs" >&2; exit 1
fi
[ "$SUBMIT" = 1 ] && mkdir -p "$L"
for job in ${jobs[@]+"${jobs[@]}"}; do
  read -r mode src run seed slug wall_time <<< "$job"
  cmd=(sbatch --time="$wall_time" -J "posthoc-$run" --output="$L/posthoc-$run-%j.out"
       --export="ALL,PLAIN_CEM=${PLAIN_CEM:-0}" "$HERE/train_eval_posthoc.sbatch" "$src" "$run" "$seed" "$mode" "$slug")
  if [ "$SUBMIT" = 1 ]; then printf '%-8s %-44s ' "$mode" "$run"; "${cmd[@]}"
  else printf '%-8s %-44s from %-34s %s\n' "$mode" "$run" "$src" "$wall_time"; fi
done
echo "train + eval: $n_train, eval only: $n_eval, skipped: $n_skip   (jobs of this account in the queue now: $in_queue)"
[ "$SUBMIT" = 1 ] || echo "dry run: nothing was submitted (SUBMIT=1 to submit)"
