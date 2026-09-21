#!/bin/bash
# Leonardo (CINECA) paths + offline settings, sourced by every *_leonardo* launcher
# and by the one-time setup in planning/LEONARDO.md. Override anything with env vars.
#
#   $WORK  = /leonardo_work/<project>          1 TB, permanent, project-shared
#            -> repo + venv, run directories, eval results, logs
#   $FAST  = /leonardo_scratch/fast/<project>  1 TB NVMe Lustre, permanent
#            -> the .h5 datasets (training does small random reads; Booster nodes
#               are diskless, $TMPDIR is 10 GB of RAM, so nothing can be staged)
#   $HOME  = 50 GB, no backup                  -> nothing large
#   $SCRATCH purges files older than 40 days   -> not used
#
# Compute nodes have no outbound internet: W&B and HF stay offline.

: "${WORK:?WORK is not set - are you on Leonardo? (chprj <project> if you have several)}"
: "${FAST:?FAST is not set - are you on Leonardo?}"

export SMWM_PROJECT_ROOT="${SMWM_PROJECT_ROOT:-$WORK/$USER/sensorimotor-world-model}"
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-$FAST/$USER/smwm-data}"
export GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-$FAST/$USER/smwm-generated}"
export RUNS_ROOT="${RUNS_ROOT:-$WORK/$USER/smwm-runs}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$WORK/$USER/smwm-paper-eval}"
export SMWM_LOG_DIR="${SMWM_LOG_DIR:-$WORK/$USER/smwm-logs}"

# Keep every cache off the 50 GB $HOME.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/$USER/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORK/$USER/.cache/uv-python}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK/$USER/.cache/xdg}"
export HF_HOME="${HF_HOME:-$WORK/$USER/.cache/hf}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WORK/$USER/.cache/mpl}"
export WANDB_DIR="${WANDB_DIR:-$RUNS_ROOT/_wandb}"

export STABLEWM_HOME="$EXTERNAL_DATA_ROOT"
export REPO_ROOT="$SMWM_PROJECT_ROOT/planning"     # generated configs' hydra searchpath
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1

mkdir -p "$EXTERNAL_DATA_ROOT" "$GENERATED_DATA_ROOT" "$RUNS_ROOT" "$EVAL_OUTPUT_ROOT" \
         "$SMWM_LOG_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME" \
         "$HF_HOME" "$MPLCONFIGDIR" "$WANDB_DIR"
