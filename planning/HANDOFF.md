# Project handoff — policy-regularizer study on SMWM

Status as of 2026-09-21. Written so that a fresh session (or a new machine) can continue
without re-deriving anything. Paper: *Sensorimotor World Models: Perception for Action via
Inverse Dynamics* (arXiv 2606.20104). Official code: petr-ivashkov/sensorimotor-world-model
(**never push there**). All work lives in the fork SinJ1024/sensorimotor-world-model, branch
`codex/fix-training-reproduction`.

## 1. Question

Can a *policy* regularizer (predict the next action a_{t+1}) replace the paper's
*inverse-dynamics* regularizer (predict a_t from z_t, z_{t+1}) in the JEPA world model, judged
by the paper's CEM planning protocol (100 tasks, goal 25 primitive steps ahead, budget 50)?

## 2. Code switches (all in `planning/config/train/base.yaml`; defaults = paper)

| block | keys | meaning |
|---|---|---|
| `loss.inverse` | `weight` | paper regularizer, a_t = h(z_t, z_{t+1}) |
| `loss.policy` | `weight, arch (mlp/mamba/transformer/gru/resmlp), context L, num_future k, use_action` | a_{t+1..t+k} = pi(z_{t-L+2..t+1}, past actions) |
| `loss.goal_policy` | `weight, goal_offsets [G...], num_actions k` | a_t..a_{t+k-1} = pi(z_t, z_{t+G}); G=1 is exactly the inverse head; G>1 conditions on a farther (hindsight) goal |

Terms are additive and independent, so `inverse+policy`, `goal_policy` alone, etc. are just
command-line overrides. `eval.py` strips `policy_model.*` and `goal_policy.*` at load time: CEM
uses only the encoder and forward model, so all regularizers are evaluated by identical code.

Per-environment lambda (paper Table 2), reused for every regularizer for a fair comparison:
TwoRoom 0.1, Reacher 5, Push-T 30, OGBench-Cube 1.

Verification / summary: `python planning/scripts/summarize_runs.py` (paper hyper-parameters,
lambda, LR schedule state, final losses, effective rank, newest CEM result per run; exits 1 on
any deviation).

## 3. Results (seed 0 unless noted)

TwoRoom, policy head grid (lambda 0.1):

| | c2 k1 | c4 k1 | c16 k1 | c16 k2 | c16 k3 |
|---|---|---|---|---|---|
| mlp | 89 | 82 | 50 | 45 | 39 |
| mamba | 79 | 79 | 45 | 50 | 47 |
| transformer | 84 | 45 | 40 | 35 | 38 |
| inverse | **100** | | | | |

TwoRoom lambda sweep, policy mlp c2k1: 1e-4:31, 3e-4:38, 1e-3:31, 3e-3:31, 0.01:79,
0.03:94/41 (seed0/seed1), 0.05:68, 0.1:89/95. Plateau 0.03–0.1; below 0.01 the forward model
collapses (eff. rank 150–185, pred loss 1e-6).

Cross-environment:

| env | data (paper App. B) | a_{t+1} predictability (measured) | inverse paper / ours | policy mlp c2k1 | policy tf c4k1 | policy mamba c4k1 |
|---|---|---|---|---|---|---|
| TwoRoom | noisy scripted, goal-directed | ~5% | ~95 / 100 | 92 (2 seeds) | 45 | 79 |
| Reacher | paper: SAC; **data: i.i.d. uniform [-1,1]** (see below) | 0.0% | ~85 / 68 | 11 | 10 | 10 |
| Push-T | DINO-WM expert | 46% (linear, from a_t alone) | ~75 / 89 | 37 | 25 | 17 |
| Cube | OGBench scripted | – | 84 / (eval pending) | (pending) | | |

Upstream baselines (experiments/planning_eval/aggregated_results.csv, same protocol, 5-seed means;
inverse / SIGReg / forward-only / random): TwoRoom 100 / 92.6 / 34.6 / 30.0, Reacher 71.4 / 67.8 /
9.8 / 13.6, Push-T 84.4 / 88.8 / 2.6 / 2.8, Cube 83.0 / 60.0 / 46.8 / 43.0. (The ~60/~40/~35
forward-only values previously quoted here were read off Fig. 5 and are wrong.)

## 3b. What each dataset gives each regularizer (2026-10-09, scripts/analyze_dataset_dynamics.py)

Privileged physical state, frameskip 5 as in training, held-out episodes, MLP R^2
(analysis/dataset_dynamics.json). "mean" targets use the mean of the 5-step action block.

| env | dynamics / actuation | planning target | data policy (measured) | inverse a_t <- (s_t,s_t+1) | policy a_t+1 <- (s_t,s_t+1) | <- s_t+1 only | a_t+1 <- a_t (shortcut) |
|---|---|---|---|---|---|---|---|
| TwoRoom | 1st-order kinematics, agent fully actuated | agent position | noisy bang-bang heuristic (64% saturated, lag-1 0.06) | 0.96 | 0.33 | 0.33 | 0.07 |
| Reacher | 2nd-order torque, arm fully actuated | arm qpos | i.i.d. uniform (flat histogram, std 0.577, lag-1 0.000, R^2 0 from full state in every fifth of the 10 000 episodes) | 0.78 (0.99 with qvel) | 0.00 | 0.00 | 0.00 |
| Push-T | agent PD-controlled; T-block moved only by contact | T-block pose | expert, smooth (lag-1 0.87) | 1.00 | 0.74 | 0.57 | 0.27 (0.66 per primitive) |
| Cube | arm IK-controlled; cube moved only by grasp/contact | cube position | scripted expert, smooth (lag-1 0.88) | 0.99 | 0.80 | 0.66 | 0.62 |

Controllability (R^2 of the state change from s_t -> from (s_t, a_t)): TwoRoom agent
0.12 -> 0.97, Reacher arm 0.14 -> 1.00, Push-T agent 0.81 -> 1.00 but T-block 0.73 -> 0.78,
Cube arm 0.81 -> 0.94, cube 0.75 -> 0.93. Push-T next expert action from agent alone 0.56,
agent + block 0.71, block alone 0.20; inverse from agent alone 1.00.

Reading: the inverse signal exists in every environment (every task has a well-actuated
body), so inverse works everywhere. The policy signal is a property of the dataset, not
of the physics: Reacher's published LeWM dataset (quentinll/lewm-reacher) has i.i.d.
uniform actions although the LeWM paper (App. E) says it was collected with SAC, so the
policy head has exactly zero signal there while inverse reaches 71 on the same physics.
Smooth expert actions (Push-T, Cube) give the +a_t head a shortcut a_t -> a_t+1, which
explains why removing a_t helps most there. Decisive next test: swap the behaviour policy
within one environment (Reacher with the stable_worldmodel SAC ExpertPolicy; TwoRoom with
uniform random actions).

## 3c. Goal-conditioned and Delta-JEPA policy heads on Reacher (2026-10-14, DelftBlue, seed 0)

Advisor request: give the policy head a goal state, and try Delta-JEPA's displacement input.
All no-action MLP c2 heads predicting a_{t+1}, lambda 5; goal = the episode's final frame
(loss.policy.goal_source=final; loss.policy.input=delta feeds dz_t and z_final - z_{t+1}).

| head input | success | eff rank | pred loss | val policy loss |
|---|---|---|---|---|
| (z_t, z_{t+1}) (old no-act, 3 seeds) | 12.7 +- 2.5 | 159 | 3e-6 | 0.999 |
| dz_t (delta) | 4 | 81 | 1.2e-5 | 1.000 |
| (z_t, z_{t+1}, z_final) | 8 | 159 | 2.1e-5 | 1.000 |
| **(dz_t, z_final - z_{t+1})** | **33** | **19.6** | **1.1e-3** | **0.993** |
| inverse (reference) | 71 | 18.2 | 1.3e-3 | - |

Only the combination escapes collapse: its latent statistics match the inverse model's and
planning is ~4 sd above random (upstream random 13.6 +- 5.1), though far below inverse.
Hypothesis: differencing hides time-invariant content (background, target) from the head, so
it cannot be fitted with static features; and z_final - z_{t+1} is a long-horizon
displacement whose dependence on a_{t+1} is a diluted multi-step inverse signal (val policy
loss 0.993 = ~0.7% of variance). delta alone has no future state (nothing to predict on
random data, collapses late: rank 21 at 57% of training -> 81); goal with endpoints lets the
head read static features. Single seed: confirm with seeds 1, 2 before relying on it.

## 4. Mechanism (the thesis argument)

* The inverse regularizer's supervision comes from the **dynamics** (a_t leaves a trace in
  z_{t+1}-z_t for any behaviour policy) and constrains exactly the controllable state.
* The policy regularizer's supervision comes from the **behaviour policy**. Reacher: next
  action unpredictable (policy loss 0.999, even on a healthy latent in the inverse+policy run)
  -> zero anti-collapse pressure -> collapse -> random-level planning, independent of head
  architecture. Push-T: predictable (expert), head learns (loss ~0.1), but the latent encodes
  *intent* rather than controllable dynamics -> no collapse; planning 37, far above forward-only
  (2.6) but far below inverse (84).
  TwoRoom: goal-directed scripted data where intent == position == the whole controllable
  state, so policy nearly matches inverse; still fragile (lambda, seed, long context).
* Any fix that conditions the action prediction on a *future* state turns it into an inverse
  model (`goal_policy`, G=1 == inverse). Diffusion/flow heads change expressiveness, not the
  information in the target, so they cannot fix Reacher.

## 5. In flight / next steps

* **Leonardo, submitted 2026-10-01 — policy-head input ablation (seed 0, all envs,
  no action input, paper lambda):** `<env>_policy_mlp_c2k1_noact_seed0` = pi(a_{t+1} | z_t, z_{t+1})
  vs `<env>_policy_mlp_c1k1_seed0` = pi(a_{t+1} | z_{t+1}); compare with the existing
  `<env>_policy_mlp_c2k1_seed0` (which also sees a_t). In train.py the window is
  `emb[:, :L]` and the target `action[:, L-1]`, so c1 computes pi(a_0 | z_0) - the same
  state->action map. Plus `tworoom_inverse_lambda_0p1_seed0` as the Leonardo
  pipeline/eval reference (must reproduce ~100). Data splits done on lrd_all_serial; each
  run is ONE job `experiments/train_eval_leonardo.sbatch` (train then CEM eval in the same
  allocation, so the eval does not queue twice), jobs 59127409-59127422, submitted by
  `$WORK/$USER/logs/resubmit_train_eval.sh`; logs `$WORK/$USER/smwm-logs`. Queue waits on
  boost_usr_prod were 39-54 h for jobs submitted 2026-09-28/29 (minutes the week before).
  **Results (2026-10-06, see RESULTS.md):** noact (z_t, z_{t+1}) vs +a_t vs c1 (z_{t+1}):
  TwoRoom 100 / 89 / 88; Push-T 65.3+-3.2 (3 seeds) / 37 / 57; Cube 78.0+-4.4 (3 seeds)
  / 69 / 66; Reacher 10 / 11 / 14. Leonardo reproduces Push-T noact seed 0 exactly (69).
  Reading: removing a_t removes the a_t -> a_{t+1} shortcut, so the head must decode the
  next action from the state, which forces the encoder to carry it. The supervision is
  still the behaviour policy: it helps where the next action is state-predictable
  (scripted TwoRoom, expert Push-T/Cube) and does nothing on Reacher (SAC data), where
  noact and c1 stay at random level. So it is NOT the inverse-dynamics trace in
  z_{t+1} - z_t (that hypothesis predicted a Reacher gain and is refuted). c1 < noact
  everywhere: z_t adds information beyond z_{t+1}.
  Push-T lambda with a_t: 3:39, 5:35.5, 10:39.5, 20:39.5, 30:37, 100:43, 300:40 -> flat in
  lambda: well above forward-only (2.6), far below inverse (84); stays < 50 as predicted.
  reacher_inverse_lambda_1_seed1 is 74, not 64: the 64 came from an eval filed under
  seed_0 (wrong CEM seed); summarize_runs.py now reports the matching seed only.
  Split across clusters on 2026-10-01: DelftBlue runs the two TwoRoom ablations as one
  serial chain (jobs 895214-895217, <= 1 GPU of the shared education account;
  `~/smwm-jobs/submit_tworoom_inputs.sh`); Leonardo keeps the TwoRoom inverse reference
  and reacher/pusht/cube c2k1_noact + c1k1.
  **2026-10-02 rebalance:** DelftBlue now runs 17 train+eval jobs on 2 lanes
  (`~/smwm-jobs/submit_sweeps_delftblue.sh`, wrapper `~/smwm-jobs/train_eval_delftblue.sbatch`,
  jobs 902306-902322): A reacher noact/c1, pusht c1, cube c1 (moved off Leonardo);
  B pusht/cube noact seeds 1,2; C1 Push-T lambda {5,10,20} x seeds {0,1};
  C2 goal_policy G=[3],[1,2,3] for reacher and pusht. Leonardo keeps only the TwoRoom
  inverse reference and the pusht/cube noact repro checks (59127409/14/16). DelftBlue
  maintenance 2026-10-03 00:00 -> 10-05 09:00: only the two reacher runs finish before it.
  Layers 1 and 3 of submit_leonardo_experiments.sh are therefore on DelftBlue; only
  layer 2 (5-seed main table) is unassigned.
  **2026-10-07, no-action heads (DelftBlue jobs 921934-921962, 28 train+eval, 2 lanes,
  `~/smwm-jobs/submit_noact_delftblue.sh`):** c2 no-act pi(z_t, z_{t+1}) and c1 pi(z_{t+1}),
  both with use_action=false. A: seeds 1,2 wherever missing, so every env has 3 seeds of
  both heads. B: lambda = paper x {0.1, 10}, seed 0, both heads (TwoRoom 0.01/1, Push-T
  3/300, Cube 0.1/10, Reacher 0.5/50). Run names <env>_policy_mlp_c2k1_noact[_lamX]_seedS
  and <env>_policy_mlp_c1k1[_lamX]_seedS. 2026-10-08: B moved to Leonardo before any of it
  started (DelftBlue jobs cancelled; Leonardo jobs 59709844-59709891, one train+eval job each,
  `$WORK/$USER/logs/submit_noact_lambda_leonardo.sh`); A stays on DelftBlue. Leonardo budget
  was 4263/8000 h used on 2026-10-08 (mostly other projects); B costs ~880 h.
  2026-10-08 17:15: B cancelled on Leonardo before any job started (user: skip the sweep
  for now, meeting 2026-10-09). Resubmit with the same script when wanted.
  NOTE: the DelftBlue checkout's main has a local commit 2cf7c14 (train_staged.sbatch)
  and diverges from origin/main; do not pull there without merging.
* Evals pending: tworoom goal_policy G=2/3/{1,2,3}, tworoom/reacher/pusht inverse+policy,
  reacher/pusht goal_policy G=2, cube inverse & policy mlp c2k1.
* Advisor request: lambda sweep of policy on Push-T {3,10,100,300} and Reacher {0.5,50,500}
  (runs `pusht_policy_mlp_c2k1_lam<x>_seed0`, `reacher_policy_mlp_c2k1_lam<x>_seed0`).
  Prediction: Reacher stays ~10 for all lambda; Push-T stays < 50.
* Optional: linear probes of T-block pose from policy vs inverse latents (Push-T) to show the
  policy latent lacks controllable-object state; extra seeds at lambda 0.1 (TwoRoom) and for
  Reacher inverse (68 vs paper ~85).
* Report tables: Section 3 above + summarize_runs.py output.

## 6. Where things are

* DelftBlue (`education-eemcs-msc-dsait`): repo `~/sensorimotor-world-model`; data
  `/scratch/$USER/smwm-data`; runs `/scratch/$USER/smwm-runs`; evals
  `/scratch/$USER/smwm-paper-eval`; launchers `planning/experiments/train/train_staged.sbatch`
  (stages the h5 to node-local disk — needed, shared-FS random reads were 14x slow) and
  `planning/experiments/planning_eval/eval_checkpoint_delftblue.sbatch <env> <run_dir> <seed>`.
* DAIC (`ewi-insy-sdm`): `/tudelft.net/staff-umbrella/mscworldmodels/{jingyuansun/smwm/containers
  (smwm-train.sif, smwm-eval.sif), smwm-data, smwm-runs-fork}`; Apptainer only (glibc 2.17).
* Leonardo (CINECA, trial project `Weng`, 8 000 h, valid until 2026-12-18): launchers written
  2026-09-21, nothing run yet. Full recipe in `planning/LEONARDO.md`. Layout: repo + uv venv
  `$WORK/$USER/sensorimotor-world-model`; data `$FAST/$USER/smwm-data` (Booster nodes are
  diskless, no staging possible); runs `$WORK/$USER/smwm-runs`; evals
  `$WORK/$USER/smwm-paper-eval`; all set by `planning/experiments/leonardo_env.sh`. Launchers
  `experiments/train/train_leonardo.sbatch`, `experiments/planning_eval/eval_checkpoint_leonardo.sbatch`
  (same seeds/output layout as DelftBlue, so `summarize_runs.py` works unchanged) and the
  advisor's sweep `experiments/lambda_sweep/submit_policy_lambda_sweep_leonardo.sh`
  (7 train+eval chains). 1 GPU + 8 CPU = 1/4 node = 8 h billed per wall hour. No container
  needed: `uv sync` works on the login node (uv downloads Python 3.13; torch wheel bundles CUDA).
