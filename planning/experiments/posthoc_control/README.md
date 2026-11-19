# Post-hoc policy head control

## Question

The policy warm-start needs a head that proposes actions. SMWM (inverse dynamics) and LeWM (SIGReg) have
none: the inverse head needs the next frame, LeWM has no action head. In the paper they therefore plan with
plain CEM at the same number of iterations.

This control gives them a proposal in the only way that is possible afterwards: the finished world model is
frozen and the paper's policy head, `(z_t, z_{t+1}, z_g) -> a_{t+1}`, is trained on its latents. No gradient
reaches the world model. The head then warm-starts CEM exactly like the jointly trained head.

It separates two explanations of the warm-start gain:

- the gain needs only *a* policy head (then a post-hoc head on SMWM or LeWM gains as much);
- the gain needs the head to be trained jointly, as the regularizer that shapes the latent.

## Design

| | jointly trained head (paper) | post-hoc head (this control) |
|---|---|---|
| world model | trained with the policy loss | finished SMWM or LeWM run, frozen |
| head, input, target | MLP (2 hidden layers of 256), `(z_t, z_{t+1}, z_g)`, `a_{t+1}` | the same |
| data, clips, goal frame | 2-step clips and the last frame of the episode | the same (`FinalGoalHDF5Dataset`) |
| batch size, optimizer, schedule, gradient clipping, epochs | 256, AdamW 1e-4, 1 % warm-up then cosine, 1.0, 10 | the same, read from the source run's config |
| encoder mode while the head trains | training mode (batch statistics in the projector) | eval mode: the latents the planner sees |
| precision of the head | bfloat16 | single precision (encoder in bfloat16) |

The last two rows are the only differences; both favour the post-hoc head.

Evaluation is the warm-start pipeline with the tasks and CEM seeds of every other eval (100 tasks, goal 25
steps away, budget 50): the head alone (`policy`) and the proposal as the initial CEM mean at 30, 10 and 5
iterations (`mean30`, `mean10`, `mean5`). Plain CEM is not run again, because the world model is the source
run's and its plain-CEM results exist; `PLAIN_CEM=1` adds them as a check.

Default scope: Push-T and Cube (TwoRoom is saturated and Reacher-PD shows no warm-start gain), SMWM and
LeWM, seed 0: four jobs.

## Files

- `../../scripts/train_posthoc_policy.py`: trains the head and writes a run directory
  `<dataset>_<inverse|sigreg>_posthoc_goalfinal_seed<S>` (source config with a `loss.policy` block, source
  weights, `policy_model.*`), which the eval pipeline loads like a jointly trained run. It never writes into
  an existing directory unless that is an unfinished post-hoc run of the same source; a finished run has
  `posthoc_done.json` (held-out policy loss, explained action variance, training time).
- `train_eval_posthoc.sbatch`: one job = train one head, then the four evals.
- `submit_posthoc.sh`: dry run by default, `SUBMIT=1` to submit; `DATASETS`, `MODELS`, `SEEDS`, `WALL`,
  `PLAIN_CEM`. Skips evaluated runs and queued jobs.
- `collect_posthoc.py`: read-only; per dataset and seed the rows SMWM, SMWM + post-hoc head, LeWM,
  LeWM + post-hoc head, joint head (ours), with explained action variance and the seven success rates.

## Run (DelftBlue)

```bash
cd ~/sensorimotor-world-model && git pull --no-rebase --no-edit
bash planning/experiments/posthoc_control/submit_posthoc.sh            # dry run
SUBMIT=1 bash planning/experiments/posthoc_control/submit_posthoc.sh   # 4 jobs
python3 planning/experiments/posthoc_control/collect_posthoc.py        # table on stderr, CSV on stdout
```

## Checked (2026-11-19, DelftBlue login node, CPU, a temporary copy)

- The script runs with the cluster's `train.py`, `eval.py`, `module.py` on `pusht_sigreg_seed0` (batch 4,
  2 steps per epoch, 2 epochs) and writes the run directory with `posthoc_done.json`.
- The eval pipeline's own `load_jepa_from_run` and `policy_cem.build_policy_model` load that run strictly
  (head with 216,074 parameters, goal input, context 2); its 309 world-model tensors are identical to the
  source checkpoint's.
- The overwrite protection: a finished post-hoc run, a run that is not post-hoc, an empty directory and an
  unfinished post-hoc run of another source are refused; an unfinished post-hoc run of the same source and
  a new name are accepted.
- `collect_posthoc.py` on the existing runs and the dry run of `submit_posthoc.sh` (four jobs).

## Not checked yet

- A GPU run: bfloat16 autocast, 12 loader workers, staging of the data to the node, and the evals inside
  the job. A failure would show in the first minutes of a job.
- The training speed. The time limits (Push-T 10 h, Cube 14 h) are those of a full training run plus the
  evals, an upper bound, since only the head is trained; the log prints steps per second every 500 steps.

## Reading the result

Compare, at equal CEM iterations, `SMWM + post-hoc head` warm-start with `joint head (ours)` warm-start and
with SMWM's own plain CEM; the same for LeWM. Also compare the head alone and the explained action variance.
The jointly trained seed-0 heads on Push-T and Cube were trained on Leonardo and have no log on DelftBlue,
so their explained variance is known from the other seeds only (Push-T 87.0 %, Cube 73.1 %). One seed is a
first look: differences below about 5 points are inside the seed range of the jointly trained head.
