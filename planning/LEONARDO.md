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
| Python | system modules stop at 3.11; the repo needs ≥ 3.13 → `uv sync` downloads its own CPython + CUDA torch wheels, exactly like DelftBlue (no container needed) |
| Account | `#SBATCH --account` is taken from `SBATCH_ACCOUNT`; the SLURM name may differ from the UserDB "AccountID" — read it from `saldo -b` |

## 1. One-time setup

```bash
# --- login (2FA via smallstep certificate, see CINECA docs) ---
ssh <user>@login.leonardo.cineca.it

# --- project account + paths ---
saldo -b                                    # -> project account name and remaining budget
echo 'export SBATCH_ACCOUNT=<project account from saldo -b>' >> ~/.bashrc
source ~/.bashrc
echo "WORK=$WORK  FAST=$FAST"               # both must be set (chprj <project> if several)

# --- repo into $WORK (not $HOME: 50 GB, no backup) ---
mkdir -p "$WORK/$USER" && cd "$WORK/$USER"
git clone https://github.com/SinJ1024/sensorimotor-world-model.git
cd sensorimotor-world-model
source planning/experiments/leonardo_env.sh   # creates the data/run/log/cache dirs

# --- uv + venv (Python 3.13 + CUDA torch, all from PyPI; ~10 GB under $WORK) ---
curl -LsSf https://astral.sh/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"          # add to ~/.bashrc too
uv sync                                       # respects UV_CACHE_DIR / UV_PYTHON_INSTALL_DIR from leonardo_env.sh
.venv/bin/python -c 'import torch, stable_pretraining, stable_worldmodel; print(torch.__version__, torch.cuda.is_available())'
# cuda.is_available() is False on the login node (no GPU) - that is expected.

# --- generate every training config once (fills manifest.tsv) ---
cd planning/experiments/train && ../../../.venv/bin/python generate_configs.py && cd -
```

## 2. Data → `$FAST`

The sweep needs Push-T and Reacher train + eval splits. Pull them from DelftBlue
(`/scratch/$USER/smwm-data`) on a Leonardo login node; `--partial` lets you re-run after
an interrupted transfer (login nodes may kill long CPU-heavy processes — just rerun).

```bash
source "$WORK/$USER/sensorimotor-world-model/planning/experiments/leonardo_env.sh"
tmux new -s xfer      # detachable
rsync -avhP --partial \
  <delftblue-user>@login.delftblue.tudelft.nl:/scratch/<delftblue-user>/smwm-data/'{pusht_expert_train,pusht_expert_eval,reacher_train,reacher_eval}.h5' \
  "$EXTERNAL_DATA_ROOT/"
ls -lh "$EXTERNAL_DATA_ROOT"; cindata          # quota check ($FAST is 1 TB, fixed)
```

Alternative if DelftBlue is unreachable from Leonardo: `experiments/train/prepare_all_data.sh`
downloads from HuggingFace and splits (login node, needs `EXTERNAL_DATA_ROOT` exported and
`REPO` adjusted; cube is 300+ GB, skip it).

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
