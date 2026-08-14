# Policy-Model Regularizer

An alternative anti-collapse regularizer for the Sensorimotor World Model (SMWM).
It **replaces the inverse model with a *policy model*** that, instead of recovering
the action that caused a transition, predicts the *next* action from the current
transition.

- **Inverse model (original):** `â_t = h(z_t, z_{t+1})`, target `a_t`.
- **Policy model (this change):** `â_{t+1} = π(z_t, z_{t+1} [, a_t])`, target `a_{t+1}`.

Both act as the sole anti-collapse mechanism (no reconstruction, no stop-gradient
tricks). The change is implemented in **both** subprojects (`toy/` and `planning/`)
and is fully backward compatible — every default keeps the original inverse-model
behavior.

---

## 1. Motivation and the key caveat

The inverse model prevents representation collapse because `a_t` **causally
determines** the transition `(o_t → o_{t+1})` and is therefore always recoverable
from `(z_t, z_{t+1})`. Recovering it forces the latents to encode action-relevant
state.

The policy model instead predicts a **future** action `a_{t+1}`. Whether this
provides any anti-collapse signal depends entirely on how the data was collected:

> **`a_{t+1}` is only predictable when the data comes from a state-conditioned
> behavior policy, i.e. `a_{t+1} = π*(s_{t+1})`.**
>
> Under i.i.d. random actions, `a_{t+1}` is statistically independent of
> `(z_t, z_{t+1}, a_t)`. The optimal policy model then outputs the constant mean,
> zeroes out the weights on `z`, and the encoder receives **no gradient** — the
> term does not prevent collapse.

Two consequences drive the design:

1. **`toy/` uses i.i.d. random actions by default.** So for the policy regularizer
   to be meaningful there, the datasets were extended with an optional
   *state-conditioned target policy* `π*` (drift toward the canvas center). This
   makes `a_{t+1}` a function of position, which the encoder must represent.

2. **The `a_t` input is a shortcut risk.** If `a_{t+1}` correlates with `a_t`
   (smooth trajectories), the model can copy `a_t` and ignore `z`, again killing
   the anti-collapse pressure. The `a_t` input is therefore a **config switch**
   (default on). In `toy/`, `a_t` is kept **random** (it drives the real
   transition) while only the *target* `a_{t+1}` is state-conditioned, so there is
   no `a_t → a_{t+1}` copy shortcut.

For `planning/`, the offline datasets (e.g. the LeWM expert datasets, OGBench) are
typically collected with a real behavior policy, so `a_{t+1}` is genuinely
predictable and no data change is needed.

---

## 2. Loss functions

### `toy/`  ([`toy/train.py`](toy/train.py), `compute_losses`)

```text
L = L_fwd + λ · L_reg
```

- `λ` = `training.lambda` (default `10.0`)
- **Forward loss (always on):** `L_fwd = MSE( g(z_t, a),  z_{t+1} )`
- **Regularizer `L_reg`**, selected by `training.reg_type`:
  - `inverse` (default): `L_reg = MSE( h(z_t, z_{t+1}),  a )`
  - `policy`:            `L_reg = MSE( π(z_t, z_{t+1} [, a]),  a_{t+1} )`

Actions are normalized (`a = action / action_scale`); the policy target `a_{t+1}`
is normalized with the same `action_scale`. Action-free worlds (sprite `NONE`,
all-random structured) have `L_reg = 0`.

### `planning/`  ([`planning/train.py`](planning/train.py), `forward_step`)

All regularizers are **additive** and independently toggled by their weights:

```text
L = L_pred + λ_inv · L_inv + λ_policy · L_policy + λ_sigreg · L_sigreg
```

| Term | Formula | Enabled when |
|---|---|---|
| `L_pred`   | `MSE(pred_emb, tgt_emb)`                         | always |
| `L_inv`    | `MSE(h(z_t, z_{t+1}), a_t)`                      | `loss.inverse.weight > 0` |
| `L_policy` | `MSE(π(z_t, z_{t+1} [, a_t]), a_{t+1})`          | `loss.policy.weight > 0` **(new)** |
| `L_sigreg` | `SIGReg(emb)`                                    | `loss.sigreg.weight > 0` |

Sequence index alignment for the policy term (`emb` has length `T`):

```text
z_t   = emb[:, :-1]        z_{t+1}      = emb[:, 1:]
a_t   = action[:, :-1]     target a_{t+1} = action[:, 1:]
```

Actions here are already normalized by the dataset column normalizer, so the
inverse and policy targets share the same scale.

**"Replace inverse with policy"** = set `loss.inverse.weight: 0` and
`loss.policy.weight: > 0`, giving `L = L_pred + λ_policy · L_policy`. Leaving both
non-zero runs them jointly (a natural ablation).

---

## 3. Configuration reference

### `toy/` — `training:` block

| Key | Default | Meaning |
|---|---|---|
| `reg_type` | `inverse` | `inverse` or `policy` — which model is the regularizer |
| `policy_use_action` | `true` | include `a_t` in the policy input `(z_t, z_{t+1}, a_t)` vs `(z_t, z_{t+1})` |
| `policy_gain` | `0.5` | proportional gain of the state-conditioned target policy `π*` |
| `lambda` | `10.0` | regularizer weight (shared by both `reg_type`s) |

Setting `reg_type: policy` automatically makes the dataset return `a_{t+1}`
(`next_action=True`). The target policy `π*`:

- **structured dot world:** each controlled dot drifts toward the canvas center,
  `a = clip(gain · (center − pos), −max_disp, max_disp)`, laid out exactly like the
  action vector (INDEPENDENT dots and COUPLED pairs contribute, STATIC/RANDOM do
  not).
- **sprite world:** `(x, y)` toward center, `θ` toward `0` (shortest wrapped
  angular error), clipped to the per-DOF delta range and masked to controlled DOFs.

### `planning/` — `loss.policy:` block (in `config/train/base.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `loss.policy.weight` | `0.0` | `λ_policy`; `> 0` enables the policy term |
| `loss.policy.use_action` | `true` | include `a_t` in the policy input |
| `loss.policy.hidden_dim` | `256` | policy MLP hidden width (optional) |
| `loss.inverse.weight` | `1.0` | `λ_inv`; set to `0.0` to fully replace inverse |

---

## 4. What changed, file by file

### `toy/`
| File | Change |
|---|---|
| [`toy/models.py`](toy/models.py) | New `PolicyModel(latent_dim, action_dim, hidden_dim, use_action)`. |
| [`toy/datasets/structured_dot_world.py`](toy/datasets/structured_dot_world.py) | `StructuredDotWorldDataset` gains `next_action`, `policy_gain`; new `_policy_action()` (drift-to-center); `__getitem__` returns a 5th element `a_{t+1}` when `next_action=True`. |
| [`toy/datasets/sprite_world.py`](toy/datasets/sprite_world.py) | Same for `SpriteWorldDataset` (`(x, y)`→center, `θ`→0). |
| [`toy/train.py`](toy/train.py) | `reg_type` dispatch in `build_models`; `compute_losses` branches on model type and reads `a_{t+1}` from the batch; loaders thread `next_action`/`policy_gain`; checkpoint saves the third model under a type-appropriate key (`policy` or `inverse`). |
| [`toy/config/dot_base.yaml`](toy/config/dot_base.yaml), [`toy/config/sprite_base.yaml`](toy/config/sprite_base.yaml) | Documented `reg_type` / `policy_use_action` / `policy_gain` defaults. |
| [`toy/experiments/single_dot/config_policy.yaml`](toy/experiments/single_dot/config_policy.yaml) | Example policy-regularizer experiment. |

### `planning/`
| File | Change |
|---|---|
| [`planning/module.py`](planning/module.py) | New `PolicyModel(embed_dim, action_dim, hidden_dim, use_action)`. |
| [`planning/jepa.py`](planning/jepa.py) | `JEPA` gains a `policy_model` and `predict_next_action(z_t, z_{t+1}, a_t)`. |
| [`planning/train.py`](planning/train.py) | Builds the policy model; `forward_step` adds the additive `L_policy` term. |
| [`planning/eval.py`](planning/eval.py) | `build_jepa` builds the policy model too, so checkpoints load cleanly (`strict=False`). |
| [`planning/config/train/base.yaml`](planning/config/train/base.yaml) | New `loss.policy` block (default off). |

### Checkpoint format note
`toy/` `model.pt` now stores the regularizer under `policy` (was always `inverse`).
`planning/` checkpoints are unchanged in structure; the policy model's weights are
saved/loaded alongside the inverse model's (both always constructed).

---

## 5. Run commands

### `toy/` (CPU-friendly, no simulator deps)

```bash
cd toy

# Original inverse-model baseline
python train.py --config experiments/single_dot/config.yaml

# Policy-model regularizer (predicts a_{t+1}), with a_t in the input
python train.py --config experiments/single_dot/config_policy.yaml

# Policy without the a_t input (removes the copy shortcut) — via a config edit:
#   training: { reg_type: policy, policy_use_action: false }
```

Or through the stage launcher:

```bash
cd toy/experiments/single_dot
./run.sh config_policy.yaml
```

Outputs land in `experiments/single_dot/results/<run_name>/`
(`model.pt`, `config.yaml`, `train_history.pt`, `embeddings.pt`).

### `planning/` (single CUDA GPU; see [`planning/README.md`](planning/README.md) for data setup)

Append Hydra dotlist overrides to any training config. The simplest path is the
`experiments/train` launcher with overrides:

```bash
cd planning/experiments/train
python generate_configs.py            # once, writes generated_configs/*.yaml + manifest.tsv

# Replace inverse with the policy regularizer for one run:
./run.sh <run_name> \
    loss.inverse.weight=0.0 \
    loss.policy.weight=1.0 \
    loss.policy.use_action=true

# Run inverse + policy jointly (ablation):
./run.sh <run_name> loss.policy.weight=1.0

# Policy without the a_t input:
./run.sh <run_name> loss.inverse.weight=0.0 loss.policy.weight=1.0 loss.policy.use_action=false
```

Equivalently, invoke `train.py` directly with Hydra:

```bash
cd planning
python -u train.py \
    --config-path experiments/train/generated_configs \
    --config-name <run_name> \
    loss.inverse.weight=0.0 loss.policy.weight=1.0 loss.policy.use_action=true
```

Planning evaluation (CEM planner) is unchanged — point it at the run directory as
usual; `eval.py` rebuilds the policy model so checkpoints load without error.

---

## 6. Suggested experiments / ablations

| Run | `toy` config | `planning` overrides |
|---|---|---|
| Inverse baseline | `reg_type: inverse` | `loss.inverse.weight=1 loss.policy.weight=0` |
| Policy (with `a_t`) | `reg_type: policy, policy_use_action: true` | `loss.inverse.weight=0 loss.policy.weight=1 loss.policy.use_action=true` |
| Policy (no `a_t`) | `reg_type: policy, policy_use_action: false` | `... loss.policy.use_action=false` |
| Inverse + Policy | — | `loss.inverse.weight=1 loss.policy.weight=1` |

What to look for:

- **Collapse check:** latent variance / rank over training. If the policy target is
  unpredictable (random-action data, or `toy` `policy_gain=0`), expect collapse —
  `L_reg` plateaus near `Var(a)` and the encoder degenerates.
- **`a_t` shortcut:** compare `policy_use_action` on/off. If "on" collapses but
  "off" does not, the model was exploiting the `a_t` copy path.
- **Representation quality (`toy`):** reuse the existing probes/decoders and
  `analyze.ipynb` under `experiments/`; the policy checkpoint is saved under the
  `policy` key in `model.pt`.

---

## 7. Data requirements (summary)

| Setting | `a_{t+1}` predictable? | Policy regularizer useful? |
|---|---|---|
| `toy` random actions (default) | No | **No** — use the state-conditioned target below |
| `toy` `reg_type: policy` | Yes (target from `π*` toward center) | Yes |
| `planning` expert / policy-collected data | Yes | Yes |
| `planning` random-action data | No | No |
