"""Paired TwoRoom evaluation: CEM planning and direct behavior-policy head.

See experiments/planning_eval/TWOROOM_POLICY.md for protocol and commands.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


class ActionNormalizer:
    """Reproduce train.py's raw-action mean and torch.std(correction=1)."""

    def __init__(self, data):
        values = torch.from_numpy(np.array(data))
        values = values[~torch.isnan(values).any(dim=1)]
        if len(values) < 2:
            raise ValueError("Need at least two finite training actions")
        self.mean = values.mean(0).numpy()
        self.std = values.std(0).numpy()
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Training action statistics contain nonfinite or zero scales")

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x):
        return (x * self.std + self.mean).astype(np.float32)


def sample_tasks(lengths, num_eval, goal_offset, seed):
    """Sample valid (episode-array index, start) pairs, including the final one."""
    if num_eval < 1 or goal_offset < 1:
        raise ValueError("num_eval and goal_offset must be positive")
    counts = np.maximum(np.asarray(lengths, dtype=np.int64) - goal_offset, 0)
    ends = np.cumsum(counts)
    total = int(counts.sum())
    if num_eval > total:
        raise ValueError(f"Requested {num_eval} tasks, but only {total} valid starts exist")
    picks = np.sort(np.random.default_rng(seed).choice(total, num_eval, replace=False))
    episodes = np.searchsorted(ends, picks, side="right")
    starts = picks - np.concatenate(([0], ends[:-1]))[episodes]
    return episodes.tolist(), starts.tolist()


class DirectPolicy:
    """Cold-start closed-loop behavior head; no goal or expert-action input.

    Encode one observation every frameskip raw steps. Predict k blocks but
    execute only the first block, then reobserve. Keep clipped, executed actions
    as history. A fresh instance is required for each fresh World evaluation.
    """

    def __init__(self, model, normalizer, frameskip):
        self.model, self.normalizer = model, normalizer
        self.frameskip = int(frameskip)
        self.context = model.policy_model.context
        self.use_action = model.policy_model.use_action
        self.device = next(model.parameters()).device
        self.latents = deque(maxlen=self.context)
        self.actions = deque(maxlen=max(1, self.context - 1))
        self.pending = deque()

    def set_env(self, env):
        self.env = env
        self.latents.clear()
        self.actions.clear()
        self.pending.clear()

    @torch.inference_mode()
    def get_action(self, info, **kwargs):
        if not self.pending:
            # World supplies uint8 pixels as (env, history, height, width, channel).
            pixels = torch.from_numpy(np.array(info["pixels"][:, -1], copy=True))
            z = self.model.encode({"pixels": pixels.permute(0, 3, 1, 2).to(self.device)})["emb"][:, -1]
            batch, action_dim = self.env.action_space.shape
            if not self.latents:
                self.latents.extend([z] * (self.context - 1))
                zero = self.normalizer.transform(np.zeros((batch, self.frameskip, action_dim), np.float32))
                zero = torch.from_numpy(zero.reshape(batch, -1)).to(self.device)
                self.actions.extend([zero] * (self.context - 1))
            self.latents.append(z)
            a = None
            if self.use_action:
                a = (torch.stack(list(self.actions), dim=1) if self.context > 1
                     else z.new_empty((batch, 0, self.frameskip * action_dim)))
            prediction = self.model.predict_next_action(torch.stack(list(self.latents), dim=1), a)
            block = prediction[:, 0].cpu().numpy().reshape(batch, self.frameskip, action_dim)
            block = self.normalizer.inverse_transform(block)
            if not np.isfinite(block).all():
                raise ValueError("Policy predicted NaN/Inf actions")
            block = np.clip(block, self.env.action_space.low[:, None, :], self.env.action_space.high[:, None, :])
            executed = self.normalizer.transform(block).reshape(batch, -1)
            self.actions.append(torch.from_numpy(executed).to(self.device))
            self.pending.extend(block.transpose(1, 0, 2))
        return self.pending.popleft().copy()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["both", "cem", "direct"], default="both")
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10, help="Parallel environments")
    parser.add_argument("--goal-offset", type=int, default=25, help="Actual raw steps from start to goal")
    parser.add_argument("--eval-budget", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-seed", type=int, default=42025)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    for key in ("num_eval", "batch_size", "goal_offset", "eval_budget", "horizon", "receding_horizon", "num_samples", "cem_steps", "topk"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.receding_horizon > args.horizon or args.topk > args.num_samples:
        parser.error("Require receding-horizon <= horizon and topk <= num-samples")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run on a GPU node or pass --device cpu")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["EXTERNAL_DATA_ROOT"] = str(args.data_root.resolve())
    from omegaconf import OmegaConf
    import stable_worldmodel as swm
    from eval import build_jepa, build_world, img_transform

    # This version has end-exclusive load_chunk and the TwoRoom state setters.
    version = importlib.metadata.version("stable-worldmodel")
    if version != "0.0.6":
        raise RuntimeError(f"This evaluation protocol requires stable-worldmodel==0.0.6; got {version}")
    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    if cfg.data.dataset.name != "tworoom_train" or int(cfg.wm.action_dim) != 2:
        raise ValueError("Expected a TwoRoom training run with action_dim=2")
    if float(cfg.loss.get("policy", {}).get("weight", 0)) <= 0:
        raise ValueError("Expected a checkpoint trained with loss.policy.weight > 0")
    frameskip = int(cfg.data.dataset.frameskip)
    if frameskip < 1 or int(cfg.loss.policy.get("context", 2)) < 1:
        raise ValueError("frameskip and policy context must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_jepa(cfg)
    checkpoint_path = args.run_dir / "checkpoints" / "last.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval().requires_grad_(False)
    del checkpoint, state

    # Training normalizes individual 2-D actions before concatenating five of them.
    train = swm.data.HDF5Dataset("tworoom_train", cache_dir=args.data_root, keys_to_load=["action"])
    normalizer = ActionNormalizer(train.get_col_data("action"))
    if train.h5_file is not None:
        train.h5_file.close()
    del train
    dataset = swm.data.HDF5Dataset("tworoom_eval", cache_dir=args.data_root, keys_to_cache=["action", "proprio"])
    episodes, starts = sample_tasks(dataset.lengths, args.num_eval, args.goal_offset, args.task_seed)
    # Fail rather than overwrite another run's metrics or partial results.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with checkpoint_path.open("rb") as stream:
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint_sha256": checkpoint_sha,
        "train_config": OmegaConf.to_container(cfg, resolve=True),
        "stable_worldmodel_version": version,
        "action_mean": normalizer.mean, "action_std": normalizer.std,
        "protocol": "cold_start_train_normalization_exact_goal_offset_v1",
        "direct_history": "repeat initial observation; physical-zero past actions; no expert warmup",
        "direct_goal_conditioned": False,
        "episodes": episodes, "start_steps": starts,
        "goal_steps": [s + args.goal_offset for s in starts],
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(metadata, indent=2, default=json_default), encoding="utf-8")
    env_cfg = OmegaConf.to_container(OmegaConf.load(Path(__file__).parent / "config/eval/env/tworoom.yaml"))
    results = {}
    try:
        for mode in (["cem", "direct"] if args.mode == "both" else [args.mode]):
            successes = []
            started = time.perf_counter()
            for offset in range(0, args.num_eval, args.batch_size):
                n = min(args.batch_size, args.num_eval - offset)
                torch.manual_seed(args.seed + offset)
                np.random.seed(args.seed + offset)
                evaluation = {
                    "seed": args.seed + offset,
                    "world": {"env_name": "swm/TwoRoom-v1", "num_envs": n, "frame_skip": 1,
                              "max_episode_steps": max(args.eval_budget * 2, args.goal_offset + 1)},
                    "eval": {"num_eval": n, "eval_budget": args.eval_budget, "img_size": int(cfg.img_size)},
                    "plan_config": {"action_block": frameskip},
                }
                world = build_world(evaluation, train_cfg=cfg)
                try:
                    if mode == "direct":
                        policy = DirectPolicy(model, normalizer, frameskip)
                    else:
                        solver = swm.solver.CEMSolver(model=model, seed=args.seed + offset, batch_size=1,
                            num_samples=args.num_samples, var_scale=1.0, n_steps=args.cem_steps,
                            topk=args.topk, device=args.device)
                        policy = swm.policy.WorldModelPolicy(solver=solver,
                            config=swm.PlanConfig(horizon=args.horizon, receding_horizon=args.receding_horizon, action_block=frameskip),
                            process={"action": normalizer},
                            transform={"pixels": img_transform(int(cfg.img_size)), "goal": img_transform(int(cfg.img_size))})
                    world.set_policy(policy)
                    batch_dir = args.output_dir / mode / f"batch_{offset:05d}"
                    batch_dir.mkdir(parents=True)
                    with torch.inference_mode():
                        metrics = world.evaluate_from_dataset(dataset,
                            episodes_idx=episodes[offset:offset+n], start_steps=starts[offset:offset+n],
                            # swm 0.0.6 uses [start:end); last pixel must be start+offset.
                            goal_offset_steps=args.goal_offset + 1, eval_budget=args.eval_budget,
                            callables=env_cfg["eval"]["callables"], save_video=args.save_video,
                            video_path=str(batch_dir))
                    successes.extend(np.asarray(metrics["episode_successes"], dtype=bool).tolist())
                    (batch_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=json_default), encoding="utf-8")
                finally:
                    world.close()
                print(f"{mode}: {offset+n}/{args.num_eval}, success={100*np.mean(successes):.1f}%", flush=True)
            results[mode] = {"success_rate_percent": 100 * float(np.mean(successes)),
                "episode_successes": successes, "num_eval": len(successes),
                "elapsed_seconds": time.perf_counter() - started}
            (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        if dataset.h5_file is not None:
            dataset.h5_file.close()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
