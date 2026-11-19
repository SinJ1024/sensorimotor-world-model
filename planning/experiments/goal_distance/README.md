# Hindsight-goal distance ablation

How does the signal of the policy regularizer depend on how far ahead its hindsight goal lies?

## Question

The policy head predicts the next action block `a_{t+1}` from the latents `(z_t, z_{t+1})` and a
hindsight goal `z_g`. For a state `s` and a goal `g`,

```text
p(a | s, g)  ∝  β(a | s) · p(g | s, a)
                behaviour   dynamics
```

On data whose actions do not depend on the state (Reacher-Random) the behaviour term carries no
information, and only the dynamics term is left: how much the action changes the goal. Two
existing settings sit at the two ends of the goal distance:

| setting | what the head sees | the goal is | Reacher-Random |
| --- | --- | --- | --- |
| SMWM, inverse head | `(z_t, z_{t+1}) -> a_t` | the frame the predicted action leads to | trains, plans |
| **this sweep**, `G = 1, 2, 3, 5, 10` | `(z_t, z_{t+1}, z_{t+1+G}) -> a_{t+1}` | `G` frames after `z_{t+1}`: 5, 10, 15, 25, 50 steps | to be measured |
| default policy head | `(z_t, z_{t+1}, z_last) -> a_{t+1}` | the last frame of the episode: 0.8 to 39 frames after `z_{t+1}`, about 20 on average | collapses |

The sweep moves the goal between the two ends and leaves everything else at the default policy
head (endpoint input with goal, weight 5, no SIGReg, no inverse loss, one-frame history, same
data, same evaluation).

`G` counts frames after the last context frame `z_{t+1}`. One frame is `frameskip = 5`
environment steps, the length of one action block. With `G = 1` the goal is `z_{t+2}`, the frame
that `a_{t+1}` leads to, so the head sees the inverse-dynamics pair `(z_{t+1}, z_{t+2})`.
`G = 5` is the goal distance of the planning evaluation (25 steps).

## What changed in `train.py`

`loss.policy.goal_source: offset` with `goal_offset: G` already existed, but it loads a clip of
`context + G` frames and encodes all of them. Two options were added:

- `loss.policy.goal_frame_only: true` loads the two context frames and the goal frame only
  (`OffsetGoalHDF5Dataset`). The samples are the same as with the dense clip; the cost per sample
  no longer depends on `G` and equals that of the default head (three frames). The goal frame is
  encoded in the same pass as the clip, as for `goal_source: final`, so the only difference from
  the dense path is the batch that the projector's BatchNorm sees.
- `loss.policy.goal_clip_frames: F` keeps only the clip starts at which an `F`-frame clip fits in
  the episode. Without it a larger `G` leaves fewer clips per 201-step episode (187 at `G = 1`,
  142 at `G = 10`; 192 for the default head) and therefore fewer optimizer steps, so the goal
  distance would be mixed with the amount of training. The sweep sets `F = 12` (`context` plus
  the largest `G`): every run trains on the same 1,278,000 clips for 49,920 steps (74 % of the
  67,500 steps of the default runs) and is validated on the same clips. Only the goal frame differs.

Both are off by default; configs without them train exactly as before.

Checks behind these statements:

- On synthetic episodes, for 144 combinations of frameskip, context, number of predicted actions
  and `G`, every sample of the sparse dataset equals the dense clip's (clip starts, frames,
  actions, observations), and the goal frame is the dense clip's frame `context - 1 + G`.
- In the DelftBlue training environment, on `reacher_train.h5` and `reacher_eval.h5` with the real
  model and `G = 1, 5, 10`: the sparse and the dense path have the same clip starts and samples,
  and `forward_step` returns the same latents and losses. With `goal_clip_frames: 12` the training
  set has 1,278,000 clips. The default config (`goal_source: final`) gives the same clips, batch
  and losses with the previous and the new `train.py`.
- With the same model code on synthetic episodes, the gradients of the loss agree between the
  sparse and the dense path (to `1e-7` of the gradient scale, BatchNorm in evaluation mode).
- `check_goal_frame_loading.py` repeats the data comparison on any dataset.

## Files

| file | purpose |
| --- | --- |
| `make_configs.py` | writes one training config per `(G, seed)` from the saved `config.yaml` of the seed-0 default run and checks that nothing else changed |
| `train_eval_goal_distance.sbatch` | trains one run, then evaluates it: plain CEM at 30, 10, 5 iterations, the policy head alone, and the warm-start at 30, 10, 5 iterations |
| `submit_goal_distance.sh` | lists the jobs (dry run by default) and submits them with `SUBMIT=1` |
| `collect_goal_distance.py` | read-only: one CSV row per run with the training diagnostics and the success rates, next to the SMWM and default-head runs |
| `check_goal_frame_loading.py` | read-only: compares the new loading with the dense clip on real data, in the training environment |

The scripts are for DelftBlue. They use the run directories under `/scratch/$USER/smwm-runs`, the
data under `/scratch/$USER/smwm-data` and the evaluation pipeline in `~/smwm-jobs/warmstart`
(`eval_warmstart.sbatch`), which rebuilds each model from the `config.yaml` in its run directory.

## Run

```bash
cd ~/sensorimotor-world-model && git pull

# optional: check the new loading against the dense clip on real data (login node; most of the
# time goes into importing the training environment)
/scratch/$USER/envs/smwm-eval-py312-torch251-cu121/bin/python -B planning/experiments/goal_distance/check_goal_frame_loading.py

# dry run: prints the 15 jobs, writes and submits nothing
bash planning/experiments/goal_distance/submit_goal_distance.sh

# write the configs to ~/smwm-jobs/goal_distance/configs and submit
SUBMIT=1 bash planning/experiments/goal_distance/submit_goal_distance.sh

# when the jobs have finished
python3 planning/experiments/goal_distance/collect_goal_distance.py goal_distance.csv
```

Defaults: Reacher-Random, `G` in 1, 2, 3, 5, 10, seeds 0 to 2, `goal_clip_frames` 12, 16 h per job.
Settings are environment variables, for example `DATASETS="reacher reacher_mix"` adds Reacher-PD
(15 more jobs) and `SEEDS="0"` runs one seed. To look at the configs before submitting, call
`make_configs.py` by hand (its docstring has the usage).

A `G` above 10 does not fit into a 12-frame clip and is refused. Runs with another `CLIP_FRAMES`
train on other clips: compare them only among themselves, and give them their own config
directory (`GD_CONFIGS=...`). The run name does not record `CLIP_FRAMES`; the config and the
collector's `clip_frames` column do.

Cost: a run takes about 8 h of training (49,920 steps on one A100) and half an hour of evaluation.
The account runs 8 jobs at once, so 15 jobs need about a day once they start.

## Output

- runs: `/scratch/$USER/smwm-runs/<dataset>_policy_mlp_c2k1_noact_goal<G>_seed<S>`
- evaluations: `/scratch/$USER/smwm-warmstart-eval/reacher/<run>/job_<id>/<label>/metrics.json`
- logs: `~/smwm-logs/gdist-<run>-<job>.out`

`collect_goal_distance.py` reports per run the explained action variance (one minus the held-out
loss of the action head; 0 for a collapsed head), the prediction loss, the effective rank, the
per-dimension standard deviation of the latent and the success rate of each evaluation. The rank
and the standard deviation are the values logged during training, in its mixed-precision
arithmetic. A collapsed model shows an explained variance near 0, a prediction loss near `1e-5`
and a per-dimension standard deviation near `1e-3` (see the default head on Reacher-Random in
the summary that the script prints). Its logged effective rank is high, about 160, but this is
rounding noise of the mixed-precision arithmetic: in single precision the same model has a rank
of about 24. Do not read the logged rank as a sign of collapse or of its absence.

The warm-start and head-alone evaluations give the head the test goal, 25 steps ahead at the
start of an episode and closer afterwards. A head trained with a fixed `G` has only seen goals at
that distance, so these numbers are secondary; the main readouts are the training diagnostics and
plain CEM.
