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
| Reacher | SAC-generated | 0.0% | ~85 / 68 | 11 | 10 | 10 |
| Push-T | DINO-WM expert | 46% (linear, from a_t alone) | ~75 / 89 | 37 | 25 | 17 |
| Cube | OGBench scripted | – | 84 / (eval pending) | (pending) | | |

Paper forward-only / random baselines (Fig. 5): TwoRoom ~60/~20, Reacher ~40/~15, Push-T ~35/~10.

## 4. Mechanism (the thesis argument)

* The inverse regularizer's supervision comes from the **dynamics** (a_t leaves a trace in
  z_{t+1}-z_t for any behaviour policy) and constrains exactly the controllable state.
* The policy regularizer's supervision comes from the **behaviour policy**. Reacher: next
  action unpredictable (policy loss 0.999, even on a healthy latent in the inverse+policy run)
  -> zero anti-collapse pressure -> collapse -> random-level planning, independent of head
  architecture. Push-T: predictable (expert), head learns (loss ~0.1), but the latent encodes
  *intent* rather than controllable dynamics -> no collapse, yet planning ~= forward-only.
  TwoRoom: goal-directed scripted data where intent == position == the whole controllable
  state, so policy nearly matches inverse; still fragile (lambda, seed, long context).
* Any fix that conditions the action prediction on a *future* state turns it into an inverse
  model (`goal_policy`, G=1 == inverse). Diffusion/flow heads change expressiveness, not the
  information in the target, so they cannot fix Reacher.

## 5. In flight / next steps

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
* Leonardo (CINECA, grant obtained 2026-09-21): not started. Plan: clone fork; copy the two
  .sif containers from DAIC; rsync the h5 splits from DelftBlue; write sbatch for
  `boost_usr_prod` (A100 64GB, no internet on compute nodes -> WANDB_MODE=offline).
