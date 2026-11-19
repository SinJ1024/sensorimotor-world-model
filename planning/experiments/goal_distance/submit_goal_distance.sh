#!/bin/bash
# Hindsight-goal-distance ablation on DelftBlue: one job per (dataset, goal offset G, seed). Each job
# trains the policy regularizer with the goal G frames after the last context frame and then
# evaluates it (train_eval_goal_distance.sbatch). Runs that hit the time limit are resumed, trained
# runs are only evaluated, and evaluated runs and jobs already in the queue are skipped. Seed 0 of
# every G is queued first, so one complete sweep arrives early.
# Usage: bash submit_goal_distance.sh            # dry run: print the job list, write and submit nothing
#        SUBMIT=1 bash submit_goal_distance.sh   # write the configs and submit
# Settings (environment variables):
#   DATASETS="reacher"     training sets: reacher (Reacher-Random) and / or reacher_mix (Reacher-PD)
#   GS="1 2 3 5 10"        goal offsets in frames (1 frame = 5 environment steps)
#   SEEDS="0 1 2"
#   CLIP_FRAMES=12         every run trains on the clip starts of a clip of this many frames, so runs
#                          with different G see the same clips; needs context (2) + largest G <= 12.
#                          0 = each G uses the starts of its own clip of context + G frames
#   WALL=16:00:00          time limit per job
#   ONLY=<file>            restrict to the run names listed in <file>
set -euo pipefail
PLANNING=$HOME/sensorimotor-world-model/planning   # the checkout the job script trains from
HERE=$PLANNING/experiments/goal_distance
C=${GD_CONFIGS:-$HOME/smwm-jobs/goal_distance/configs}
R=/scratch/$USER/smwm-runs; L=$HOME/smwm-logs; ENV_SLUG=reacher; E=/scratch/$USER/smwm-warmstart-eval/$ENV_SLUG
HEAD=policy_mlp_c2k1_noact
read -r -a DATASETS <<< "${DATASETS:-reacher}"
read -r -a GS <<< "${GS:-1 2 3 5 10}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
CLIP_FRAMES=${CLIP_FRAMES:-12}; WALL=${WALL:-16:00:00}; SUBMIT=${SUBMIT:-0}
LABELS=(cem30 cem10 cem5 policy mean30 mean10 mean5)   # the evals of train_eval_goal_distance.sbatch

problem() {  # fatal when submitting, a warning in a dry run
  if [ "$SUBMIT" = 1 ]; then echo "ERROR: $*" >&2; exit 1; else echo "WARNING: $*"; fi
}
grep -q "def policy_goal_frame_only" "$PLANNING/train.py" 2>/dev/null \
  || problem "$PLANNING/train.py has no goal_frame_only support: git pull first"
[ -f "$HOME/smwm-jobs/warmstart/eval_warmstart.sbatch" ] || problem "no eval pipeline in $HOME/smwm-jobs/warmstart"
for ds in "${DATASETS[@]}"; do
  [ -f "$R/${ds}_${HEAD}_goalfinal_seed0/config.yaml" ] || problem "no template $R/${ds}_${HEAD}_goalfinal_seed0/config.yaml"
done
max_g=0; for g in "${GS[@]}"; do if [ "$g" -gt "$max_g" ]; then max_g=$g; fi; done
if [ "$CLIP_FRAMES" != 0 ] && [ "$CLIP_FRAMES" != auto ] && [ $((2 + max_g)) -gt "$CLIP_FRAMES" ]; then
  problem "G = $max_g needs a clip of $((2 + max_g)) frames (context 2 + G), more than CLIP_FRAMES=$CLIP_FRAMES"
fi

evaluated() {  # every label has a result in some job directory
  local l
  for l in "${LABELS[@]}"; do ls "$E/$1"/job_*/"$l"/metrics.json >/dev/null 2>&1 || return 1; done
}
queued=$(squeue -u "$USER" -h -o %j 2>/dev/null || true)
in_queue=$(squeue -u "$USER" -h 2>/dev/null | wc -l || true)

echo "configs: $C   clip frames: $CLIP_FRAMES   wall: $WALL   datasets: ${DATASETS[*]}   G: ${GS[*]}   seeds: ${SEEDS[*]}"
if [ "$SUBMIT" = 1 ]; then
  mkdir -p "$L"
  for ds in "${DATASETS[@]}"; do
    python3 "$HERE/make_configs.py" "$C" "$R/${ds}_${HEAD}_goalfinal_seed0/config.yaml" \
      --goal-offsets "${GS[@]}" --seeds "${SEEDS[@]}" --clip-frames "$CLIP_FRAMES" > /dev/null
  done
fi

jobs=(); n_train=0; n_resume=0; n_eval=0; n_skip=0
for seed in "${SEEDS[@]}"; do
  for g in "${GS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      run=${ds}_${HEAD}_goal${g}_seed${seed}
      if [ -n "${ONLY:-}" ] && ! grep -qx "$run" "$ONLY"; then continue; fi
      if evaluated "$run"; then echo "SKIP     $run (already evaluated)"; n_skip=$((n_skip + 1)); continue; fi
      if grep -qx "gdist-$run" <<< "$queued"; then echo "SKIP     $run (already in the queue)"; n_skip=$((n_skip + 1)); continue; fi
      if [ -s "$R/$run/final_embeddings.pt" ]; then mode=evalonly; n_eval=$((n_eval + 1))
      elif [ -s "$R/$run/checkpoints/last.ckpt" ]; then mode=resume; n_resume=$((n_resume + 1))
      elif [ -e "$R/$run" ]; then
        echo "STALE    $run ($R/$run exists without a checkpoint: move it away to train again)"; n_skip=$((n_skip + 1)); continue
      else mode=train; n_train=$((n_train + 1)); fi
      jobs+=("$mode $run $seed")
    done
  done
done

if [ "$SUBMIT" = 1 ] && [ $((in_queue + ${#jobs[@]})) -gt 100 ]; then
  echo "ERROR: ${#jobs[@]} new jobs + $in_queue in the queue exceed the limit of 100 submitted jobs" >&2; exit 1
fi
for job in ${jobs[@]+"${jobs[@]}"}; do
  read -r mode run seed <<< "$job"
  cmd=(sbatch --time="$WALL" -J "gdist-$run" --output="$L/gdist-$run-%j.out"
       "$HERE/train_eval_goal_distance.sbatch" "$run" "$seed" "$mode" "$ENV_SLUG" "$C")
  if [ "$SUBMIT" = 1 ]; then printf '%-8s %-52s ' "$mode" "$run"; "${cmd[@]}"
  else printf '%-8s %-52s %s\n' "$mode" "$run" "$WALL"; fi
done
echo "train + eval: $n_train, resume + eval: $n_resume, eval only: $n_eval, skipped: $n_skip"
[ "$SUBMIT" = 1 ] || echo "dry run: nothing was written or submitted (SUBMIT=1 to submit)"
