#!/bin/bash
# Train one run on one Leonardo Booster A100 (64 GB) using the uv venv at the repo root.
#
# Usage (project account once per shell, e.g. in ~/.bashrc; `saldo -b` lists it):
#   export SBATCH_ACCOUNT=<project account>
#   sbatch train_leonardo.sbatch <base_run_name> [Hydra overrides...]
#
# Examples:
#   sbatch train_leonardo.sbatch pusht_inverse_lambda_30_seed0
#   sbatch train_leonardo.sbatch pusht_policy_seed0 \
#       subdir=pusht_policy_mlp_c2k1_lam3_seed0 loss.policy.weight=3
#
# Billing: 1 GPU + 8 cores + <=128 GB = 1/4 node = 8 core-hours per wall hour.
# Paths + offline settings come from planning/experiments/leonardo_env.sh.
#
#SBATCH --job-name=smwm-train
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
trap 'status=$?; if [ "$status" -ne 0 ]; then echo "FAILED status=$status line=$LINENO command=$BASH_COMMAND" >&2; fi' EXIT

BASE_RUN="${1:?usage: sbatch train_leonardo.sbatch <base_run_name> [Hydra overrides...]}"
shift

PROJECT="${SMWM_PROJECT_ROOT:-$WORK/$USER/sensorimotor-world-model}"
source "$PROJECT/planning/experiments/leonardo_env.sh"
PLANNING="$PROJECT/planning"
VENV="$PROJECT/.venv"

# One copy of the log next to the runs, whatever --output was.
JOB_ID="${SLURM_JOB_ID:-local}"
LOGFILE="$SMWM_LOG_DIR/${SLURM_JOB_NAME:-smwm-train}-job${JOB_ID}.out"
exec > >(tee "$LOGFILE") 2>&1

[ -x "$VENV/bin/python" ] || { echo "missing venv: $VENV (run 'uv sync' at the repo root, see planning/LEONARDO.md)" >&2; exit 1; }
[ -d "$PLANNING" ] || { echo "missing fork checkout: $PROJECT" >&2; exit 1; }

# Fail early if the dataset this base run needs is not on $FAST.
case "$BASE_RUN" in
    tworoom_*) DATASET=tworoom_train.h5 ;;
    reacher_*) DATASET=reacher_train.h5 ;;
    pusht_*)   DATASET=pusht_expert_train.h5 ;;
    cube_*)    DATASET=cube_single_expert_train.h5 ;;
    *) echo "cannot infer the environment from run name '$BASE_RUN'" >&2; exit 1 ;;
esac
[ -s "$EXTERNAL_DATA_ROOT/$DATASET" ] || { echo "missing data: $EXTERNAL_DATA_ROOT/$DATASET" >&2; exit 1; }

REPO_COMMIT="$(git -C "$PROJECT" rev-parse HEAD)"
if [ -n "$(git -C "$PROJECT" status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: tracked files in $PROJECT are dirty; the run is not reproducible from $REPO_COMMIT" >&2
fi

# Job-private RAM /tmp on Booster nodes is 10 GB: keep only small temp files there.
export TMPDIR="${TMPDIR:-/tmp}"

# torch's wheel bundles its CUDA runtime; the module only adds EGL/driver paths.
module load cuda >/dev/null 2>&1 || true

CONFIG_DIR="$PLANNING/experiments/train/generated_configs"
if [ ! -f "$CONFIG_DIR/$BASE_RUN.yaml" ]; then
    "$VENV/bin/python" "$PLANNING/experiments/train/generate_configs.py" \
        --output-dir "$CONFIG_DIR" --run-name "$BASE_RUN"
fi

NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

echo "======================================================================"
echo "base run     : $BASE_RUN"
echo "overrides    : $*"
echo "host         : $(hostname)    job: $JOB_ID"
echo "repo commit  : $REPO_COMMIT"
echo "python       : $VENV/bin/python ($("$VENV/bin/python" --version 2>&1))"
echo "data         : $EXTERNAL_DATA_ROOT"
echo "runs         : $RUNS_ROOT"
echo "workers      : $NUM_WORKERS"
echo "log          : $LOGFILE"
echo "======================================================================"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[ "$GPU_COUNT" -eq 1 ] || { echo "expected one visible GPU, got $GPU_COUNT" >&2; exit 1; }
[[ "$GPU_NAME" == *A100* ]] || { echo "expected an A100, got: $GPU_NAME" >&2; exit 1; }

cd "$PLANNING"
srun "$VENV/bin/python" -u train.py \
    --config-path "$CONFIG_DIR" \
    --config-name "$BASE_RUN" \
    num_workers="$NUM_WORKERS" \
    "$@"
