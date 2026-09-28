#!/bin/bash
#
# Deploy the three-layer SMWM experiment plan on Leonardo.
#
#   layer 1  Push-T policy lambda refinement {5,10,20} x seeds {0,1}
#            + goal_policy G=[3] and G=[1,2,3] for reacher and pusht
#   layer 2  main table: reacher / pusht / cube x seeds 0..4 x
#            {inverse, policy mlp c2k1, goal_policy G=[2], inverse+policy}
#   layer 3  no-action control: pusht / cube policy mlp c2k1 use_action=false, seeds 0..2
#
# Runs already completed on DelftBlue / DAIC (see planning/HANDOFF.md) are in
# ALREADY_DONE and are never resubmitted; a run whose checkpoint already exists
# under $RUNS_ROOT gets only its evaluation job.
#
# Each run is one training job plus one CEM evaluation job (--dependency=afterok).
# To keep the shared project account usable, runs are spread over --lanes
# dependency chains: at most <lanes> trainings are in flight at a time, while the
# evaluations run alongside them.
#
# Usage:
#   export SBATCH_ACCOUNT=try26_weng
#   ./submit_leonardo_experiments.sh --dry-run                 # print the plan
#   ./submit_leonardo_experiments.sh --layers 1                # submit layer 1
#   ./submit_leonardo_experiments.sh --layers 1,2,3 --lanes 6
#   ./submit_leonardo_experiments.sh --list                    # table only, no cluster access
#
set -uo pipefail

LAYERS="1,2,3"
LANES=4
DRY_RUN=0
LIST_ONLY=0
TRAIN_TIME="${SMWM_TRAIN_TIME:-20:00:00}"
EVAL_TIME="${SMWM_EVAL_TIME:-04:00:00}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --layers) LAYERS="$2"; shift 2 ;;
        --lanes) LANES="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --list) LIST_ONLY=1; DRY_RUN=1; shift ;;
        --train-time) TRAIN_TIME="$2"; shift 2 ;;
        --eval-time) EVAL_TIME="$2"; shift 2 ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$LIST_ONLY" -eq 0 ]; then
    export SMWM_PROJECT_ROOT="${SMWM_PROJECT_ROOT:-$(cd "$HERE/../.." && pwd)}"
    source "$HERE/leonardo_env.sh"
    TRAIN_SBATCH="$HERE/train/train_leonardo.sbatch"
    EVAL_SBATCH="$HERE/planning_eval/eval_checkpoint_leonardo.sbatch"
    [ -n "${SBATCH_ACCOUNT:-}" ] || { echo "export SBATCH_ACCOUNT=<account from 'saldo -b'> first" >&2; exit 2; }
fi

# --------------------------------------------------------------- parameters ---
lam() { case "$1" in tworoom) echo 0.1 ;; reacher) echo 5 ;; pusht) echo 30 ;; cube) echo 1 ;; esac; }
invbase() {
    case "$1" in
        tworoom) echo tworoom_inverse_lambda_0p1 ;;
        reacher) echo reacher_inverse_lambda_1 ;;   # legacy label, trained at lambda 5
        pusht)   echo pusht_inverse_lambda_30 ;;
        cube)    echo cube_inverse_lambda_1 ;;
    esac
}
evalslug() { case "$1" in cube) echo ogbcube ;; *) echo "$1" ;; esac; }
lab() { echo "${1//./p}"; }   # 0.5 -> 0p5, 10.0 -> 10p0

# Trained and evaluated already (DelftBlue / DAIC summary of 2026-09-28).
ALREADY_DONE="
reacher_inverse_lambda_1_seed0 reacher_inverse_lambda_1_seed1
reacher_policy_mlp_c2k1_seed0 reacher_goalpolicy_g2_seed0 reacher_invpolicy_mlp_c2k1_seed0
pusht_inverse_lambda_30_seed0 pusht_policy_mlp_c2k1_seed0 pusht_goalpolicy_g2_seed0
pusht_invpolicy_mlp_c2k1_seed0 pusht_policy_mlp_c2k1_lam10p0_seed0
pusht_policy_mlp_c2k1_noact_seed0
cube_inverse_lambda_1_seed0 cube_policy_mlp_c2k1_seed0 cube_policy_mlp_c2k1_noact_seed0
"

# ------------------------------------------------------------------- plan -----
PLAN=()      # "layer|env|run_name|base_run|override override ..."
CURRENT_LAYER=0

job() {  # job <env> <run_name> <base_run> [hydra overrides...]
    local env="$1" run="$2" base="$3"; shift 3
    case " $(echo $ALREADY_DONE) " in *" $run "*) return 0 ;; esac
    # Keep the last value when the same key is given twice (e.g. $POL followed by
    # a sweep value for loss.policy.weight), so Hydra never sees a duplicate key.
    local seen=" " ordered="" ov key i
    for ((i=$#; i>=1; i--)); do
        ov="${!i}"; key="${ov%%=*}"
        case "$seen" in *" $key "*) continue ;; esac
        seen="$seen$key "
        ordered="$ov${ordered:+ $ordered}"
    done
    PLAN+=("$CURRENT_LAYER|$env|$run|$base|$ordered")
}

pol() { echo "loss.inverse.weight=0 loss.policy.weight=$(lam "$1") loss.policy.arch=mlp loss.policy.context=2 loss.policy.num_future=1"; }

# --- layer 1: Push-T lambda refinement + goal offsets on the big environments --
CURRENT_LAYER=1
for S in 0 1; do
    for L in 5.0 10.0 20.0; do
        job pusht "pusht_policy_mlp_c2k1_lam$(lab $L)_seed$S" "pusht_policy_seed$S" \
            $(pol pusht) "loss.policy.weight=$L"
    done
done
for E in reacher pusht; do
    W="$(lam $E)"; IB="$(invbase $E)"
    job $E "${E}_goalpolicy_g3_seed0" "${IB}_seed0" \
        loss.inverse.weight=0 "loss.goal_policy.weight=$W" "loss.goal_policy.goal_offsets=[3]"
    job $E "${E}_goalpolicy_g123_seed0" "${IB}_seed0" \
        loss.inverse.weight=0 "loss.goal_policy.weight=$W" "loss.goal_policy.goal_offsets=[1,2,3]"
done

# --- layer 2: main table, five seeds ------------------------------------------
CURRENT_LAYER=2
for E in reacher pusht cube; do
    W="$(lam $E)"; IB="$(invbase $E)"
    for S in 0 1 2 3 4; do
        job $E "${IB}_seed$S" "${IB}_seed$S"
        job $E "${E}_policy_mlp_c2k1_seed$S" "${E}_policy_seed$S" $(pol $E)
        job $E "${E}_goalpolicy_g2_seed$S" "${IB}_seed$S" \
            loss.inverse.weight=0 "loss.goal_policy.weight=$W" "loss.goal_policy.goal_offsets=[2]"
        job $E "${E}_invpolicy_mlp_c2k1_seed$S" "${E}_policy_seed$S" \
            "loss.inverse.weight=$W" "loss.policy.weight=$W" \
            loss.policy.arch=mlp loss.policy.context=2 loss.policy.num_future=1
    done
done

# --- layer 3: drop a_t from the policy head -----------------------------------
CURRENT_LAYER=3
for S in 0 1 2; do
    for E in pusht cube; do
        job $E "${E}_policy_mlp_c2k1_noact_seed$S" "${E}_policy_seed$S" \
            $(pol $E) loss.policy.use_action=false
    done
done

# ------------------------------------------------------------------ select ----
SELECTED=()
for row in ${PLAN[@]+"${PLAN[@]}"}; do
    case ",$LAYERS," in *",${row%%|*},"*) SELECTED+=("$row") ;; esac
done

echo "plan: ${#PLAN[@]} runs total, ${#SELECTED[@]} selected (layers $LAYERS), lanes=$LANES"
printf '%-6s %-8s %-42s %s\n' LAYER ENV RUN OVERRIDES
for row in "${SELECTED[@]}"; do
    IFS='|' read -r l e r b o <<< "$row"
    printf '%-6s %-8s %-42s %s\n' "$l" "$e" "$r" "$o"
done
[ "$LIST_ONLY" -eq 1 ] && exit 0

# ------------------------------------------------------------------ submit ----
declare -a LANE_DEP
for ((i=0; i<LANES; i++)); do LANE_DEP[$i]=""; done

submit() {
    if [ "$DRY_RUN" = "1" ]; then echo "      sbatch $*" >&2; echo "DRY"; else sbatch --parsable "$@"; fi
}

idx=0
for row in "${SELECTED[@]}"; do
    IFS='|' read -r LAYER ENV RUN BASE OVERRIDES <<< "$row"
    RUN_DIR="$RUNS_ROOT/$RUN"
    LANE=$((idx % LANES))
    SEED="${RUN##*_seed}"
    ESLUG="$(evalslug "$ENV")"

    TRAIN_JID=""
    if [ -s "$RUN_DIR/checkpoints/last.ckpt" ]; then
        echo "[$LAYER] $RUN: checkpoint present, evaluation only"
    elif [ -e "$RUN_DIR" ] && [ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        echo "[$LAYER] $RUN: SKIPPED, $RUN_DIR is non-empty without last.ckpt (running or crashed)"
        continue
    else
        EXTRA=()
        [ "$RUN" != "$BASE" ] && EXTRA+=("subdir=$RUN")
        EXTRA+=("wandb.config.name=train_$RUN")
        PREV="${LANE_DEP[$LANE]}"
        DEP=(); [ -n "$PREV" ] && DEP=(--dependency="afterany:$PREV")
        TRAIN_JID=$(submit --job-name="t-$RUN" --time="$TRAIN_TIME" \
            --output="$SMWM_LOG_DIR/train-$RUN-%j.out" "${DEP[@]}" \
            "$TRAIN_SBATCH" "$BASE" $OVERRIDES "${EXTRA[@]}")
        LANE_DEP[$LANE]="$TRAIN_JID"
        echo "[$LAYER] $RUN: train $TRAIN_JID (lane $LANE${PREV:+, after $PREV})"
    fi

    EVAL_DEP=(); [ -n "$TRAIN_JID" ] && EVAL_DEP=(--dependency="afterok:$TRAIN_JID")
    EVAL_JID=$(submit --job-name="e-$RUN" --time="$EVAL_TIME" \
        --output="$SMWM_LOG_DIR/eval-$RUN-%j.out" "${EVAL_DEP[@]}" \
        "$EVAL_SBATCH" "$ESLUG" "$RUN_DIR" "$SEED" 50)
    echo "         eval  $EVAL_JID${TRAIN_JID:+ (after $TRAIN_JID)}"
    idx=$((idx + 1))
done

echo
echo "monitor : squeue -u \$USER --format='%.10i %.42j %.8T %.10M %.12R'"
echo "summary : $SMWM_PROJECT_ROOT/.venv/bin/python $SMWM_PROJECT_ROOT/planning/scripts/summarize_runs.py --runs-root $RUNS_ROOT --eval-root $EVAL_OUTPUT_ROOT --budget 50"
