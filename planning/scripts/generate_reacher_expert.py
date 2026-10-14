#!/usr/bin/env python3
"""Record a Reacher dataset with a goal-directed, state-dependent behaviour policy.

The published LeWM Reacher dataset was recorded with stable_worldmodel's
World.record_dataset and i.i.d. uniform actions (RandomPolicy), so a next-action
policy regularizer gets no signal from it. This script records the same
environment, with the same renderer settings, episode length and file format,
but with a scripted expert:

  - each episode starts with a random target joint configuration q*;
  - a PD controller drives the arm there: tau = clip(Kp * (q* - q) - Kd * qdot + noise);
  - once the arm is within `tol` of q* (or after `max_segment` steps) a new q* is
    drawn, so the arm keeps moving purposefully for the whole episode.

The swm env repeats each action for 2 physics steps (0.04 s), and the arm is
light (gear 0.05), so the PD gains must be low: kp=2, kd=0.3 chatters bang-bang
(lag-1 action autocorrelation -0.35, 41% saturated). The defaults kp=0.7,
kd=0.05, noise 0.3 give lag-1 +0.61, 13% saturated, every way-point reached
(median 25 steps) and an action std (0.53) close to the random data's (0.58).

Actions therefore depend on the state (and on the current way-point, which the
motion reveals). Evaluate trained models on the ORIGINAL reacher_eval.h5 so that
only the training data differs.

Usage:
  MUJOCO_GL=egl python generate_reacher_expert.py --name reacher_expert_train --episodes 9000 --seed 0
  MUJOCO_GL=egl python generate_reacher_expert.py --name reacher_expert_eval  --episodes 1000 --seed 100000
"""
from __future__ import annotations

import argparse
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.policy import BasePolicy  # noqa: E402

SHOULDER_RANGE = (-np.pi, np.pi)   # unlimited hinge
WRIST_RANGE = (-2.4, 2.4)          # inside the wrist joint limits


def wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class PDWaypointExpert(BasePolicy):
    """Noisy PD controller towards way-points in joint space."""

    def __init__(self, kp=0.7, kd=0.05, noise_std=0.3, tol=0.15, max_segment=60, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.type = "expert"
        self.kp, self.kd, self.noise_std = float(kp), float(kd), float(noise_std)
        self.tol, self.max_segment = float(tol), int(max_segment)
        self.rng = np.random.default_rng(seed)
        self.targets = None
        self.seg_len = None
        self.stats = {"segments": 0, "reached": 0, "segment_steps": []}

    def _new_target(self):
        return np.array([self.rng.uniform(*SHOULDER_RANGE), self.rng.uniform(*WRIST_RANGE)])

    def get_action(self, infos, **kwargs):
        q = np.asarray(infos["qpos"], dtype=np.float64)
        n = q.shape[0]
        q = q.reshape(n, -1)[:, :2]
        qd = np.asarray(infos["qvel"], dtype=np.float64).reshape(n, -1)[:, :2]
        step = np.asarray(infos["step_idx"]).reshape(n, -1)[:, -1]
        if self.targets is None:
            self.targets = np.stack([self._new_target() for _ in range(n)])
            self.seg_len = np.zeros(n, dtype=int)
        actions = np.empty((n, 2), dtype=np.float32)
        for i in range(n):
            if step[i] == 0:                                   # new episode
                self.targets[i], self.seg_len[i] = self._new_target(), 0
            err = np.array([wrap_angle(self.targets[i, 0] - q[i, 0]), self.targets[i, 1] - q[i, 1]])
            reached = np.all(np.abs(err) < self.tol) and np.all(np.abs(qd[i]) < 1.0)
            if reached or self.seg_len[i] >= self.max_segment:
                self.stats["segments"] += 1
                self.stats["reached"] += int(reached)
                self.stats["segment_steps"].append(int(self.seg_len[i]))
                self.targets[i], self.seg_len[i] = self._new_target(), 0
                err = np.array([wrap_angle(self.targets[i, 0] - q[i, 0]), self.targets[i, 1] - q[i, 1]])
            u = self.kp * err - self.kd * qd[i] + self.noise_std * self.rng.standard_normal(2)
            actions[i] = np.clip(u, -1.0, 1.0)
            self.seg_len[i] += 1
        return actions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="dataset name, e.g. reacher_expert_train")
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("EXTERNAL_DATA_ROOT", "."))
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--kp", type=float, default=0.7)
    ap.add_argument("--kd", type=float, default=0.05)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--max-segment", type=int, default=60)
    a = ap.parse_args()

    # Same settings as the published dataset: 224x224 frames, 200 control steps
    # per episode (201 rows with the initial state), no separate goal, default
    # task/renderer (the target is not drawn).
    world = swm.World("swm/ReacherDMControl-v0", num_envs=a.num_envs, image_shape=(224, 224),
                      max_episode_steps=200, seed=a.seed, goal_conditioned=False, verbose=0)
    policy = PDWaypointExpert(kp=a.kp, kd=a.kd, noise_std=a.noise, tol=a.tol,
                              max_segment=a.max_segment, seed=a.seed)
    world.set_policy(policy)
    world.record_dataset(a.name, episodes=a.episodes, seed=a.seed, cache_dir=a.out)
    s = policy.stats
    seg = np.array(s["segment_steps"]) if s["segment_steps"] else np.array([0])
    print(f"wrote {a.out}/{a.name}.h5: {a.episodes} episodes; way-point segments {s['segments']}, "
          f"reached {s['reached'] / max(1, s['segments']):.0%}, segment length median {int(np.median(seg))} "
          f"(p90 {int(np.percentile(seg, 90))}) steps", flush=True)


if __name__ == "__main__":
    main()
