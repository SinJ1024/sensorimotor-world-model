# Head-architecture sweep: Transformer and Mamba heads with a goal, without action input

Does the architecture of the policy head matter once the head sees only `(z_t, z_{t+1}, z_g)`, and
each architecture gets its own regularizer weight?

## Question

The early Transformer and Mamba heads (September) took the previous action `a_t` as input, had no
goal, and were trained at a single weight, the one the SMWM paper uses for its inverse head. That
comparison cannot separate the architecture from the weight: at the same weight their latents were
three times smaller in scale than the MLP's.

This sweep trains the head of the paper, `(z_t, z_{t+1}, z_g) -> a_{t+1}` with `z_g` the last frame
of the episode and no action input, in three architectures and at several weights:

| head | tokens / input | parameters (action dim 10) |
| --- | --- | --- |
| MLP (the existing head) | `[z_t, z_{t+1}, z_g]` concatenated, two hidden layers of 256 | 216,074 |
| Transformer | tokens `(z_g, z_t, z_{t+1})`, 2 causal blocks of width 256, read at `z_{t+1}` | 1,417,738 |
| Mamba | tokens `(z_g, z_t, z_{t+1})`, 2 blocks of width 256, read at `z_{t+1}` | 949,836 |

Everything else is the default run of the paper: one-frame history, batch 256, 10 epochs, the same
data, clips and evaluation. Only `loss.policy.arch` and `loss.policy.weight` change.

## What changed in `module.py`

`PolicyModel` refused a goal for `arch = gru | transformer | mamba`. Now the goal latent is one more
token, placed before the window and projected by its own linear layer (`goal_token`); the summary
is still read at the last latent `z_{t+1}`, which sees the goal through the causal attention or the
recurrent state. `input: delta` is still MLP-only. `train.py` and `policy_cem.py` already pass the
goal to the head, so they are unchanged.

Checks (on the Mac, torch 2.11, against the committed `module.py`):

- 168 configurations that the old class accepts (5 architectures, with and without action input,
  context 1, 2, 4, 16, one or three predicted actions, and for the MLP heads the goal and the
  displacement input): the same parameter names and shapes, and bit-identical outputs with the
  same weights. Existing checkpoints therefore load and behave as before.
- 18 goal-conditioned sequence heads: output shape; gradients reach `z_t`, `z_{t+1}`, the goal and
  the goal projection; a missing goal is an error; changing one sample's goal changes only that
  sample's output; the forward pass runs under bfloat16 autocast.
- Each of the three heads of the sweep fits a target that needs both the goal and the last latent
  (`a = W (z_g - z_{t+1})`): relative error 0.03 (MLP), 0.02 (Transformer, Mamba) after 150 steps.

Not checked yet: a training step of `train.py` and an evaluation with the new heads on DelftBlue
(the cluster was unreachable when this was written). Submit one or two runs first and read the log
before submitting the rest.

## Files

| file | purpose |
| --- | --- |
| `make_configs.py` | writes one training config per `(architecture, weight, seed)` from the saved `config.yaml` of the seed-0 default run and checks that nothing else changed |
| `train_eval_head_arch.sbatch` | trains one run, then evaluates it: plain CEM at 30, 10, 5 iterations, the policy head alone, and the warm-start at 30, 10, 5 iterations |
| `submit_head_arch.sh` | lists the jobs (dry run by default) and submits them with `SUBMIT=1` |
| `collect_head_arch.py` | read-only: one CSV row per run with the training diagnostics (also in single precision) and the success rates, next to SMWM |

The scripts are for DelftBlue. They use the run directories under `/scratch/$USER/smwm-runs`, the
data under `/scratch/$USER/smwm-data` and the evaluation pipeline in `~/smwm-jobs/warmstart`
(`eval_warmstart.sbatch`), whose `module.py` is a link to the checkout's, so it has the new heads
after the pull.

## Run

```bash
cd ~/sensorimotor-world-model && git pull

# dry run: prints the jobs, writes and submits nothing
bash planning/experiments/head_arch/submit_head_arch.sh

# first only the two new heads at the default weight (2 jobs), to see that training and evaluation work
ARCHS="transformer mamba" MULTS="1" SUBMIT=1 bash planning/experiments/head_arch/submit_head_arch.sh

# then the sweep (runs that exist, are queued or are evaluated are skipped)
SUBMIT=1 bash planning/experiments/head_arch/submit_head_arch.sh

# when the jobs have finished (the training environment's python fills the single-precision columns)
/scratch/$USER/envs/smwm-eval-py312-torch251-cu121/bin/python planning/experiments/head_arch/collect_head_arch.py head_arch.csv
```

Defaults: TwoRoom, the three heads, weights 0.3, 1, 3 and 10 times the default weight of the
dataset (TwoRoom 0.03, 0.1, 0.3, 1), seed 0: 12 runs, of which the MLP at the default weight exists
already. Settings are environment variables:

- `DATASETS="tworoom pusht"` adds datasets (`tworoom pusht cube reacher reacher_mix`).
- `ARCHS="transformer mamba"` leaves out the MLP reference.
- `MULTS="1 3 10 30"` or, for one dataset, `LAMBDAS="0.1 1 10"` sets the weights.
- `SEEDS="0 1 2"` adds seeds; `WALL=10:00:00` sets the time limit.

Run names: `<dataset>_policy_<arch>_c2k1_noact_goalfinal[_lam<x>]_seed<S>`; `_lam<x>` is left out at
the default weight, so the MLP at the default weight is the existing run of the paper.

Cost of one run (training plus the seven evaluations on one A100), from the jobs of the MLP goal
head: TwoRoom about 4.5 h, Push-T about 10 h, Reacher 10 to 12 h, Cube about 15 h. The sequence
heads are small next to the encoder and the forward model. The account runs 8 jobs at once.

## Reading the result

- Compare the heads at each head's own best weight, and report the weight. A head that is best at
  the smallest or largest weight of the sweep is not bracketed: extend the weights.
- The logged effective rank is computed in bfloat16 and is rounding noise when the latent scale is
  small (TwoRoom). Use `effective_rank_fp32` and `std_fp32` of the collector.
- One seed per cell is about 5 points of noise on the success rate; differences below that need
  more seeds (`SEEDS="0 1 2"`).

## Output

- runs: `/scratch/$USER/smwm-runs/<run>`
- evaluations: `/scratch/$USER/smwm-warmstart-eval/<env>/<run>/job_<id>/<label>/metrics.json`
- logs: `~/smwm-logs/harch-<run>-<job>.out`
