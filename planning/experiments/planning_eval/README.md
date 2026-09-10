# Planning evaluation (paper Figure 5)

This directory reports the paper's sole planning metric: environment
`success_rate`. Learned checkpoints are evaluated by the same latent-space CEM
planner; the random baseline uses `RandomPolicy`.

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
to override their defaults. Each job saves its resolved evaluation config and a
`metrics.json` containing only `success_rate`.
