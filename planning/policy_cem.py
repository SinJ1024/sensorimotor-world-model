"""Policy-guided CEM: warm-start the planner with actions proposed by the policy head.

Runs trained with the policy regularizer carry a head pi that predicts the next
action block from a window of latents (optionally + past actions and a goal
latent). At every replanning step this solver rolls pi out *inside the world
model* from the current observation,

    a_t = pi(z_{t-1}, z_t [, a_{t-1}] [, z_goal]),   z_{t+1} = f(z_t, a_t),

for the planning horizon, and uses the resulting action sequence

  * mode="mean":   as the initial CEM mean (instead of zeros / the shifted
                   previous plan); CEM then refines it as usual;
  * mode="inject": keeps CEM's own initialisation and adds the proposal as an
                   extra candidate in every iteration, so it enters the elites
                   only when the world model scores it well;
  * mode="policy": directly, without any CEM iterations (policy-only baseline).

The policy head was trained on actions normalised with the *training* set's
statistics, while CEM samples in the space of the evaluation scaler, so the
proposals are mapped between the two. The rollout uses the same context as the
CEM cost (JEPA.rollout): with predictor history H > 1 the last H frames of the
observation history (one per action block) and the H - 1 executed action blocks;
with H = 1 only the current frame is available, so the policy window is started
as (z_t, z_t) (a still arm). Later windows use the model's own predicted latents.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import stable_worldmodel as swm

from module import PolicyModel


def build_policy_model(train_cfg, state_dict, embed_dim):
    """Rebuild the policy head exactly as train.py does and load its weights."""
    pol = train_cfg.loss.get("policy", {})
    if not float(pol.get("weight", 0.0)):
        raise ValueError("run was trained without the policy regularizer; no head to warm-start from")
    weights = {k[len("policy_model."):]: v for k, v in state_dict.items() if k.startswith("policy_model.")}
    goal_source = str(pol.get("goal_source", "offset"))
    use_goal = goal_source == "final" or bool(pol.get("goal_offset", 0))
    head = PolicyModel(
        embed_dim=int(embed_dim),
        action_dim=int(train_cfg.data.dataset.frameskip * train_cfg.wm.action_dim),
        hidden_dim=pol.get("hidden_dim", 256),
        use_action=bool(pol.get("use_action", True)),
        num_future=int(pol.get("num_future", 1)),
        context=int(pol.get("context", 2)),
        arch=pol.get("arch", "mlp"),
        depth=int(pol.get("depth", 2)),
        input=pol.get("input", "endpoints"),
        use_goal=use_goal,
    )
    head.load_state_dict(weights, strict=True)
    return head


def train_action_stats(train_cfg):
    """Mean / std of the raw per-step action in the training set (as utils.get_column_normalizer)."""
    ds = swm.data.HDF5Dataset(train_cfg.data.dataset.name, keys_to_cache=["action"])
    a = torch.from_numpy(np.asarray(ds.get_col_data("action"), dtype=np.float64))
    a = a[~torch.isnan(a).any(dim=1)]
    return a.mean(0).numpy(), a.std(0).numpy()


class PolicyWarmStartCEM(swm.solver.CEMSolver):
    def __init__(self, model, policy_head, train_stats, eval_scaler, mode="mean", **kwargs):
        super().__init__(model=model, **kwargs)
        if mode not in ("mean", "inject", "policy"):
            raise ValueError(f"mode must be 'mean', 'inject' or 'policy', got {mode!r}")
        self.mode = mode
        self.policy_head = policy_head.to(self.device).eval()
        mu_tr, sd_tr = train_stats
        # per raw action dim: x_eval = x_train * scale + shift
        scale = sd_tr / eval_scaler.scale_
        shift = (mu_tr - eval_scaler.mean_) / eval_scaler.scale_
        self._scale = torch.as_tensor(scale, dtype=torch.float32)
        self._shift = torch.as_tensor(shift, dtype=torch.float32)

    def _to_eval_space(self, a):
        """(B, A_block) in training-normalised units -> evaluation-scaler units."""
        raw_dim = self._scale.numel()
        blocks = a.view(a.size(0), -1, raw_dim)
        blocks = blocks * self._scale.to(a.device) + self._shift.to(a.device)
        return blocks.reshape(a.size(0), -1)

    def _to_train_space(self, a):
        """Inverse of _to_eval_space."""
        raw_dim = self._scale.numel()
        blocks = a.view(a.size(0), -1, raw_dim)
        blocks = (blocks - self._shift.to(a.device)) / self._scale.to(a.device)
        return blocks.reshape(a.size(0), -1)

    def _context(self, info_dict, dev):
        """Latents of the last H frames (one per action block) and the H-1 executed blocks, as JEPA.rollout."""
        m = self.model
        H = int(getattr(m.predictor, "num_frames", 1))
        pixels = info_dict["pixels"].to(dev)
        if H == 1:
            return m.encode({"pixels": pixels[:, -1:]})["emb"], []
        raw = info_dict["action"].to(dev)                          # (n, T, raw_dim), eval units
        stride = self.action_dim // raw.size(-1)
        idx = torch.arange(pixels.size(1) - (H - 1) * stride - 1, pixels.size(1), stride, device=dev)
        lat = m.encode({"pixels": pixels.index_select(1, idx)})["emb"]
        past = torch.nan_to_num(raw[:, -(H - 1) * stride:], 0.0).reshape(raw.size(0), H - 1, -1)
        return lat, [self._to_train_space(past[:, i]) for i in range(H - 1)]

    @torch.inference_mode()
    def propose(self, info_dict):
        """Roll the policy head out in the world model: (n_envs, horizon, action_dim) in CEM units."""
        m, head = self.model, self.policy_head
        dev = next(m.parameters()).device
        H = int(getattr(m.predictor, "num_frames", 1))
        z_ctx, acts = self._context(info_dict, dev)
        z_goal = None
        if head.use_goal:
            z_goal = m.encode({"pixels": info_dict["goal"].to(dev)})["emb"][:, -1]
        L = head.context
        lat = list(z_ctx.unbind(1))
        lat = [lat[0]] * max(L - len(lat), 0) + lat        # H = 1: (z_t, z_t), a still start
        acts = [lat[0].new_zeros(lat[0].size(0), head.action_dim)] * max(L - 1 - len(acts), 0) + acts
        out = []
        for _ in range(self.horizon):
            a_win = torch.stack(acts[-(L - 1):], 1) if (head.use_action and L > 1) else None
            a = head(torch.stack(lat[-L:], 1), a_win, z_goal)[:, 0]      # training units
            out.append(self._to_eval_space(a))
            acts.append(a)
            a_ctx = torch.stack(acts[-H:], 1)
            z_next = m.predict(torch.stack(lat[-H:], 1), m.action_encoder(a_ctx))[:, -1]
            lat.append(z_next)
        return torch.stack(out, 1)

    def solve(self, info_dict, init_action=None):
        t0 = time.time()
        proposal = self.propose(info_dict).cpu()
        print(f"policy propose time: {time.time() - t0:.4f} seconds")
        if self.mode == "policy":
            return {"actions": proposal, "costs": [0.0] * proposal.size(0),
                    "mean": [proposal], "var": [torch.zeros_like(proposal)]}
        if self.mode == "mean":
            return super().solve(info_dict, init_action=proposal)
        # inject: CEM samples as usual (from the shifted previous plan); candidate 1
        # of every iteration is overwritten with the policy proposal before scoring.
        self._proposal, self._calls = proposal, 0
        model, self.model = self.model, _InjectingModel(self.model, self)
        try:
            return super().solve(info_dict, init_action=init_action)
        finally:
            self.model = model


class _InjectingModel:
    """Wraps the world model so that CEMSolver.solve scores the policy proposal too.

    CEMSolver calls get_cost once per iteration with candidates of shape
    (batch, num_samples, horizon, dim) for env batch `calls // n_steps`; the
    candidates tensor is the one it later ranks, so overwriting slot 1 in place
    makes the proposal a regular candidate (slot 0 holds the current mean).
    """

    def __init__(self, model, solver):
        self._model, self._solver = model, solver

    def get_cost(self, info_dict, candidates):
        s = self._solver
        b0 = (s._calls // s.n_steps) * s.batch_size
        candidates[:, 1] = s._proposal[b0:b0 + candidates.size(0)].to(candidates.device)
        s._calls += 1
        return self._model.get_cost(info_dict, candidates)

    def __getattr__(self, name):
        return getattr(self._model, name)
