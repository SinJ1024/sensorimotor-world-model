# Leonardo (CINECA) — setup and the policy lambda sweep

Project: trial account `Weng` (LEONARDO_B host, 8 000 standard hours, 1 TB WORK quota,
valid 2026-09-18 → 2026-12-18). Everything below runs on a **login node** unless it says
`sbatch`. Launchers: `experiments/train/train_leonardo.sbatch`,
`experiments/planning_eval/eval_checkpoint_leonardo.sbatch`,
`experiments/lambda_sweep/submit_policy_lambda_sweep_leonardo.sh`; shared paths in
`experiments/leonardo_env.sh`.

## What differs from DelftBlue / DAIC

| | Leonardo Booster |
|---|---|
| GPU node | 4× A100 64 GB, 32 cores, 512 GB RAM; we take 1/4 node: `--gres=gpu:1 --cpus-per-task=8 --mem=120G` |
| Billing | `BH = T · N · R · 32`, `R = max(cores/32, mem/512G, gpus/4)` → 1/4 node costs **8 h per wall hour**; a ~4 h training ≈ 32 h, so the budget is ≈ 1 000 GPU-hours |
| Partition / QOS | `boost_usr_prod` + `normal`: ≤ 24 h walltime; `boost_qos_dbg`: 30 min, 2 jobs, high priority (smoke tests) |
| Node-local disk | **none** (diskless); `$TMPDIR` = 10 GB of RAM. The DelftBlue "stage the h5 to local disk" trick is impossible → datasets live on `$FAST` (NVMe Lustre, built for small random reads) |
| Storage | `$HOME` 50 GB no backup · `$WORK` 1 TB permanent (repo, venv, runs, evals) · `$FAST` 1 TB permanent (datasets) · `$SCRATCH` unlimited but **purged after 40 days** (unused) |
| Internet | login nodes yes, compute nodes **no** → `WANDB_MODE=offline`, `HF_HUB_OFFLINE=1` (set in `leonardo_env.sh`) |
| Python | system modules stop at 3.11; the repo needs ≥ 3.13 → uv downloads its own CPython + CUDA torch wheels (no container). Three build traps, all handled in §1: uv picks 3.14 unless pinned, `box2d-py` needs `swig`, `labmaze` cannot be built at all |
| Login nodes | round-robin over login01/02/05/07 (`loginNN-ext.leonardo.cineca.it`); long CPU/IO processes get **killed** (a 30-min `uv sync` and a multi-hour h5 split both died). Use `tmux` on a fixed node, or the budget-free `lrd_all_serial` partition (4 cores, 4 h, no internet) |
| Account | SLURM account is **`try26_weng`** (lower case; UserDB shows "Weng"). `saldo -b` may say "username not existing" for the first days; `sacctmgr -n show assoc user=$USER` is authoritative |

## 1. One-time setup

```bash
# --- project account + paths (a new member's group appears a day after being added) ---
id                                           # must list the project group, e.g. try26_Weng
echo "WORK=$WORK  FAST=$FAST"                # "/no/project/defined" = not active yet
echo 'export SBATCH_ACCOUNT=try26_weng' >> ~/.bashrc && source ~/.bashrc

# --- repo into $WORK (not $HOME: 50 GB, no backup) ---
mkdir -p "$WORK/$USER" && cd "$WORK/$USER"
git clone https://github.com/SinJ1024/sensorimotor-world-model.git
cd sensorimotor-world-model
source planning/experiments/leonardo_env.sh  # creates data/run/log/cache dirs, caches on $WORK

# --- uv ---
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WORK/$USER/bin" INSTALLER_NO_MODIFY_PATH=1 sh
export PATH="$WORK/$USER/bin:$PATH"          # add to ~/.bashrc too

# --- swig for box2d-py (transitive gymnasium[all] dep; no swig module on Leonardo) ---
uv venv "$WORK/$USER/.tools" && uv pip install --python "$WORK/$USER/.tools/bin/python" swig
export PATH="$WORK/$USER/.tools/bin:$PATH"

# --- venv: pin 3.13, skip labmaze; run inside tmux (login nodes kill long jobs) ---
# labmaze 1.0.6 has no wheels for >=3.13 and its WORKSPACE bazel build needs
# @bazel_tools//platforms constraints removed in bazel 5, so no bazel builds it.
# Only dm_control.locomotion (maze arenas) imports it; Reacher uses dm_control.suite
# and Cube uses ogbench/mujoco, so the project never imports it.
tmux new -s uvsync
uv sync --python 3.13 --no-install-package labmaze
uv pip install --python .venv/bin/python hdf5plugin    # blosc-compressed HDF5; not in the lock
.venv/bin/python -c 'from dm_control import suite; import ogbench, gym_pusht, stable_worldmodel.envs; print("env imports OK")'

# --- generate every training config once (fills manifest.tsv) ---
cd planning/experiments/train && ../../../.venv/bin/python generate_configs.py && cd -
```

A later plain `uv sync` would try to build labmaze again and drop hdf5plugin: always pass
`--no-install-package labmaze` and reinstall hdf5plugin afterwards.

## 2. Data → `$FAST` (HuggingFace)

Compute nodes have no internet, so download on a login node (Leonardo pulls ~1 GB/s from
HuggingFace: 67 GB took two minutes), then extract and split on the serial partition.

```bash
source "$WORK/$USER/sensorimotor-world-model/planning/experiments/leonardo_env.sh"
cd "$EXTERNAL_DATA_ROOT"
for spec in lewm-tworooms:tworoom.tar.zst lewm-reacher:reacher.tar.zst lewm-cube:cube_single_expert.tar.zst; do
  "$SMWM_PROJECT_ROOT/.venv/bin/python" -c "from huggingface_hub import hf_hub_download as d; d(repo_id='quentinll/${spec%%:*}', repo_type='dataset', filename='${spec#*:}', local_dir='.')"
done
# Push-T: the HF file pusht_expert_train.h5(.zst) is the FULL set; it must be called
# pusht_expert.h5 before splitting (see scripts/make_episode_splits.py).

S="$SMWM_PROJECT_ROOT/planning/experiments/split_data_leonardo.sbatch"
sbatch "$S" tworoom.h5 tworoom.tar.zst
sbatch "$S" reacher.h5 reacher.tar.zst
sbatch "$S" pusht_expert.h5
sbatch "$S" cube_single_expert.h5 cube_single_expert.tar.zst
```

Sizes: tworoom 3.2 GB compressed; reacher 23 GB → 93 GB; cube 44 GB → 95 GB; pusht 44 GB.
The splits roughly double that (~500 GB of the 1 TB `$FAST` in total); the tarballs can be
deleted once the splits verify.

## 3. Smoke test (debug QOS, 30 min, high priority)

```bash
cd "$WORK/$USER/sensorimotor-world-model/planning/experiments/train"
sbatch --qos=boost_qos_dbg --time=00:25:00 train_leonardo.sbatch pusht_policy_seed0 \
    subdir=_smoke_pusht_policy trainer.max_epochs=1 +trainer.limit_train_batches=50 \
    trainer.val_check_interval=25 trainer.limit_val_batches=2 wandb.enabled=false
squeue -u $USER; tail -f "$SMWM_LOG_DIR"/smwm-train-job<id>.out
rm -rf "$RUNS_ROOT/_smoke_pusht_policy"
```

Watch the log for the dataloader throughput (it/s). If the GPU starves with 8 workers,
resubmit with `--cpus-per-task=16` (the launcher passes `SLURM_CPUS_PER_TASK` as
`num_workers`; cost doubles to 16 h per wall hour).

## 4. The policy lambda sweep (advisor request, HANDOFF.md §5)

Push-T policy mlp c2k1 λ ∈ {3, 10, 100, 300}, Reacher λ ∈ {0.5, 50, 500}, seed 0.
Each λ = one training job + one paper-protocol CEM eval chained `afterok`.
Expected cost ≈ 7 × (4 h + ~2 h) × 8 ≈ 350 h of the 8 000 h budget.

```bash
cd "$WORK/$USER/sensorimotor-world-model/planning/experiments/lambda_sweep"
DRY_RUN=1 ./submit_policy_lambda_sweep_leonardo.sh     # shows the 14 sbatch lines
./submit_policy_lambda_sweep_leonardo.sh               # submits them
squeue -u $USER --format='%.10i %.40j %.8T %.10M %R'
```

Run names: `pusht_policy_mlp_c2k1_lam{3,10,100,300}_seed0`,
`reacher_policy_mlp_c2k1_lam{0p5,50,500}_seed0` under `$RUNS_ROOT`; evals under
`$EVAL_OUTPUT_ROOT/<env>/<run>/seed_0/budget_50/job_<id>/results/seed_0/metrics.json`.

Single runs by hand (same launchers):

```bash
sbatch train_leonardo.sbatch pusht_policy_seed0 subdir=pusht_policy_mlp_c2k1_lam3_seed0 loss.policy.weight=3
sbatch eval_checkpoint_leonardo.sbatch pusht "$RUNS_ROOT/pusht_policy_mlp_c2k1_lam3_seed0" 0
```

## 5. Results

```bash
cd "$WORK/$USER/sensorimotor-world-model"
.venv/bin/python planning/scripts/summarize_runs.py \
    --runs-root "$RUNS_ROOT" --eval-root "$EVAL_OUTPUT_ROOT" --budget 50
grep -h "PAPER SUCCESS RATE" "$SMWM_LOG_DIR"/eval-*.out
saldo -b                                         # budget left
```

`summarize_runs.py` reads the sweep λ back out of the run name (`lam3`, `lam0p5`, …) and
checks every paper hyper-parameter; exit code 1 means a run DEVIATEs.

Optional W&B sync from a login node: `wandb sync "$WANDB_DIR"/wandb/offline-run-*`.

## 6. Troubleshooting

* `training.resume` is off: a non-empty run dir is refused. A timed-out job → resubmit with
  `training.resume=true` (same `subdir`), or delete the dir. The sweep script skips such dirs.
* `WORK: parameter null or not set` → not on Leonardo, or run `chprj <project>`.
* Eval needs EGL for MuJoCo (`MUJOCO_GL=egl` is set); if it fails with a GL error on the
  compute node, try `MUJOCO_GL=osmesa` via `sbatch --export=ALL,MUJOCO_GL=osmesa …`.
* Quotas: `cindata` / `cinQuota`. Budget: `saldo -b`.
