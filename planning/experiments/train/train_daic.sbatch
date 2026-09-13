#!/bin/bash
# Train one run on exactly one DAIC L40 using the already validated container.
# Usage: sbatch train_daic.sbatch <base_run_name> [Hydra overrides...]
#
# Architecture example:
#   sbatch train_daic.sbatch tworoom_policy_seed0 \
#     subdir=tworoom_policy_mamba_c16k3_seed0 \
#     loss.policy.arch=mamba loss.policy.context=16 \
#     loss.policy.num_future=3
#
#SBATCH --job-name=smwm-train
#SBATCH --partition=general
#SBATCH --account=ewi-insy-sdm
#SBATCH --qos=medium
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=26
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/mscworldmodels/jingyuansun/smwm/logs/slurm-%x-%j.out

set -euo pipefail
trap 'status=$?; if [ "$status" -ne 0 ]; then echo "FAILED status=$status line=$LINENO command=$BASH_COMMAND" >&2; fi' EXIT

BASE_RUN="${1:?usage: sbatch train_daic.sbatch <base_run_name> [Hydra overrides...]}"
shift

PROJ=/tudelft.net/staff-umbrella/mscworldmodels
PROJECT="${SMWM_REPO:-$PROJ/sensorimotor-world-model}"
PLANNING="$PROJECT/planning"
SIF="${SMWM_SIF:-$PROJ/jingyuansun/smwm/containers/smwm-train.sif}"
DATA="${EXTERNAL_DATA_ROOT:-$PROJ/smwm-data}"
# This new default deliberately cannot collide with the previous bad checkpoints.
RUNS="${RUNS_ROOT:-$PROJ/smwm-runs-repro-fixed}"
LOGDIR="${SMWM_LOG_DIR:-$RUNS/_logs}"
JOBTMP="$RUNS/_tmp/${SLURM_JOB_ID:-local}"

mkdir -p "$LOGDIR" "$JOBTMP"
LOGFILE="$LOGDIR/${SLURM_JOB_NAME:-smwm-train}-job${SLURM_JOB_ID:-local}.out"
exec > >(tee "$LOGFILE") 2>&1

[ -f "$SIF" ] || { echo "missing container: $SIF" >&2; exit 1; }
[ -d "$PLANNING" ] || { echo "missing fork checkout: $PROJECT" >&2; exit 1; }
[ -s "$DATA/tworoom_train.h5" ] || { echo "missing data: $DATA/tworoom_train.h5" >&2; exit 1; }

REPO_COMMIT="$(git -C "$PROJECT" rev-parse HEAD)"
[ -z "$(git -C "$PROJECT" status --porcelain --untracked-files=no)" ] || {
    echo "tracked files in $PROJECT are dirty; commit or restore them first" >&2
    exit 1
}

CONFIG_DIR="$PLANNING/experiments/train/generated_configs"
if [ ! -f "$CONFIG_DIR/$BASE_RUN.yaml" ]; then
    apptainer exec --bind "$PROJ" "$SIF" /work/venv/bin/python \
        "$PLANNING/experiments/train/generate_configs.py" \
        --output-dir "$CONFIG_DIR" --run-name "$BASE_RUN"
fi

export REPO_ROOT="$PLANNING"
export EXTERNAL_DATA_ROOT="$DATA"
export STABLEWM_HOME="$DATA"
export RUNS_ROOT="$RUNS"
export WANDB_MODE=offline
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TMPDIR="$JOBTMP" TMP="$JOBTMP" TEMP="$JOBTMP"
export MPLCONFIGDIR="/tmp/$USER/mpl"
export XDG_CACHE_HOME="/tmp/$USER/xdg"
export HF_HOME="/tmp/$USER/hf"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$HF_HOME"

echo "======================================================================"
echo "base run     : $BASE_RUN"
echo "overrides    : $*"
echo "host         : $(hostname)"
echo "repo commit  : $REPO_COMMIT"
echo "container    : $SIF"
echo "data         : $DATA"
echo "runs         : $RUNS"
echo "log          : $LOGFILE"
echo "======================================================================"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[ "$GPU_COUNT" -eq 1 ] || { echo "expected one visible GPU, got $GPU_COUNT" >&2; exit 1; }
[[ "$GPU_NAME" == *L40* ]] || { echo "expected L40, got: $GPU_NAME" >&2; exit 1; }

cd "$PLANNING"
apptainer exec --nv --bind "$PROJ" "$SIF" \
    /work/venv/bin/python -u train.py \
    --config-path "$CONFIG_DIR" \
    --config-name "$BASE_RUN" \
    num_workers="${SLURM_CPUS_PER_TASK:-26}" \
    "$@"
