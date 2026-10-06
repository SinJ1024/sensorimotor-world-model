# Planning evaluation (paper Figure 5)

This directory reports the paper's sole planning metric: environment
`success_rate`. Learned checkpoints are evaluated by the same latent-space CEM
planner; the random baseline uses `RandomPolicy`.

Use the repository's frozen Python 3.13 environment (`uv sync --frozen`) on
DelftBlue. The root `pyproject.toml` and `uv.lock` match the official release.

Paper protocol:

- 100 held-out tasks per run
- goal offset 25
- environment interaction budget 50
- planning horizon 5 and receding horizon 5
- action block 5
- CEM population 300, 30 optimization steps, top-k 30
- five training seeds (or five random-policy repeats)

Learned runs must declare `trainer.max_epochs: 10`, and their `last.ckpt` must
contain completed epoch 9. Incomplete checkpoints are rejected before planning.

The paper sweep contains four environments, four methods, and five seeds/repeats:

```text
4 environments x (forward_only, inverse, sigreg, random) x 5 = 80 jobs
```

The experimental `policy` regularizer may be evaluated as an additional method.
It uses exactly the same CEM planner and `success_rate`; its auxiliary policy
head is required to be present in the checkpoint and excluded from inference. This measures
the planning quality of the world model learned with that regularizer. There is
no direct policy-head metric in this evaluation.

## Generate sweep configurations

Generate the 80 paper jobs:

```bash
python generate_configs.py
```

Include the 20 policy-regularized comparison jobs:

```bash
python generate_configs.py --include-policy
```

The generator requires the final-training manifest from
`../train/generated_configs/manifest.tsv`. Aggregate completed jobs with:

```bash
python aggregate_results.py
```

The resulting CSV contains `success_rate` as its only evaluation metric.
Protocol fields, paths, and completion status are retained as provenance.

## Evaluate arbitrary checkpoints on DelftBlue

Use `eval_checkpoint_delftblue.sbatch` for both paper checkpoints and
policy-regularized checkpoints:

```bash
sbatch eval_checkpoint_delftblue.sbatch \
  tworoom /absolute/path/to/tworoom_inverse_lambda_0p1_seed0 0

sbatch eval_checkpoint_delftblue.sbatch \
  tworoom /absolute/path/to/tworoom_policy_seed0 0
```

Arguments are environment (`tworoom`, `reacher`, `pusht`, or `ogbcube`), the
completed run directory, and its training seed (`0` through `4`). Set
`SMWM_PROJECT_ROOT`, `EXTERNAL_DATA_ROOT`, or `EVAL_OUTPUT_ROOT` before `sbatch`
to override their defaults. On DelftBlue the launcher reuses
`$HOME/conda-envs/biased-dreams-dreamer/bin/python` when available. Set
`SMWM_PYTHON` to select another existing environment. The launcher verifies
PyTorch, `stable-worldmodel`, and `stable-pretraining` before evaluation. Each
job saves its resolved evaluation config and a `metrics.json` containing only
`success_rate`.

## `aggregated_results.csv`: paper runs and this study's runs

The file holds every planning result in one place. The `source` column says where a
row comes from:

- `upstream`: the 80 released paper runs (Fig. 5). `aggregate_results.py` writes
  these and keeps all other rows when it rewrites the file; `plot_results.ipynb`
  plots only these.
- `DelftBlue`, `Leonardo`: the policy-regularizer study, one row per run and
  cluster. `scripts/refresh_results.sh` (or `scripts/sync_aggregated_results.py`)
  replaces these rows with the current results of every cluster.

Study rows use the same columns and seeds as the paper rows, with `method` set to
the run family (`inverse`, `policy_mlp_c2k1_noact`, `goalpolicy_g123`,
`policy_mlp_c2k1_lam5p0`, ...) and extra columns `regularizer`, `lambda`,
`eff_rank`, `verdict` and `other_evals` (evaluations filed under another seed
label, which used the wrong CEM seed). `status` is `ok`, `no_eval` (trained, not
evaluated yet) or `incomplete` (training not finished). A run on both clusters
appears twice, once per `source`: count it once when averaging seeds.
