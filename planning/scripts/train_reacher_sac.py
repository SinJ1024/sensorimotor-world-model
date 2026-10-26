#!/usr/bin/env python3
"""Train a state-based SAC expert for swm/ReacherDMControl-v0 (task 'hard').

The checkpoint is saved in the format stable_worldmodel's
`stable_worldmodel.envs.dmcontrol.ExpertPolicy` loads (an SB3 SAC `.zip` plus the
`VecNormalize` observation statistics `.pkl`), so the trained expert can record a
dataset with `World.record_dataset` in the same env, renderer and format as the
published LeWM Reacher set. Training uses the swm env itself, so the action repeat
(2 physics steps per action) and the 6-d observation (joint angles, finger-to-target
vector, joint velocities) match recording exactly.

Usage:
  MUJOCO_GL=egl python train_reacher_sac.py --out sac_reacher --steps 500000
"""
from __future__ import annotations

import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401  (registers the swm/* envs)
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_env(seed, max_steps):
    def thunk():
        env = gym.make("swm/ReacherDMControl-v0", max_episode_steps=max_steps)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return thunk


def evaluate(model, vecnorm, max_steps, episodes, seed):
    """Deterministic and stochastic returns plus the fraction of steps on target."""
    out = {}
    for det in (True, False):
        env = gym.make("swm/ReacherDMControl-v0", max_episode_steps=max_steps)
        rets, on_target = [], []
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done, ret, hits, n = False, 0.0, 0, 0
            while not done:
                a, _ = model.predict(vecnorm.normalize_obs(obs[None]), deterministic=det)
                obs, r, term, trunc, _ = env.step(a[0])
                ret += r; hits += r > 0; n += 1; done = term or trunc
            rets.append(ret); on_target.append(hits / n)
        out["det" if det else "stoch"] = {"return": float(np.mean(rets)), "on_target": float(np.mean(on_target))}
    return out


class EvalLog(BaseCallback):
    def __init__(self, args, path):
        super().__init__()
        self.args, self.path, self.best, self.t0 = args, path, -np.inf, time.time()

    def _on_step(self):
        if self.num_timesteps % self.args.eval_every >= self.training_env.num_envs:
            return True
        res = evaluate(self.model, self.training_env, self.args.max_steps, self.args.eval_episodes, 10_000)
        res.update(step=self.num_timesteps, minutes=round((time.time() - self.t0) / 60, 1))
        print(json.dumps(res), flush=True)
        with open(os.path.join(self.path, "eval_log.jsonl"), "a") as f:
            f.write(json.dumps(res) + "\n")
        if res["stoch"]["return"] > self.best:   # recording uses the stochastic policy
            self.best = res["stoch"]["return"]
            self.model.save(os.path.join(self.path, "expert_policy.zip"))
            self.training_env.save(os.path.join(self.path, "vec_normalize.pkl"))
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=500, help="episode length in env steps (DMC: 1000 physics steps)")
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # train_freq=1 steps all n_envs envs at once; gradient_steps=n_envs keeps one update per transition
    venv = DummyVecEnv([make_env(a.seed + i, a.max_steps) for i in range(a.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=1_000_000, batch_size=256,
                learning_starts=10_000, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=a.n_envs,
                policy_kwargs=dict(net_arch=[256, 256]), seed=a.seed, device=a.device, verbose=0)
    model.learn(total_timesteps=a.steps, callback=EvalLog(a, a.out))
    model.save(os.path.join(a.out, "final_policy.zip"))
    venv.save(os.path.join(a.out, "final_vec_normalize.pkl"))
    print("done; best checkpoint:", os.path.join(a.out, "expert_policy.zip"), flush=True)


if __name__ == "__main__":
    main()
