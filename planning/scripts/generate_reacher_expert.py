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
motion reveals).

v2 options (defaults reproduce v1): --wrist-range widens the wrist way-points up
to the joint limit (v1's 2.4 left the arm almost never beyond |wrist| 2.4,
where 15% of eval states lie); --local-radius draws each way-point near the
current configuration, so 25-step displacements match the eval tasks (v1 moved
~3x farther); --random-prob replaces each action by a uniform random one with
that probability, so 5-step action blocks span all directions like the CEM
proposals (v1's PD blocks had participation ratio 4.2 of 10); --max-torque caps
the PD torque so motion stays slow but goal-directed (predictable);
--random-episode-prob records that fraction of whole episodes with uniform random
actions, so the random-action regime of the eval tasks stays covered while the
other episodes carry a policy signal (mixing random actions *within* an episode
destroys the signal: R^2 of the next action falls from ~0.4 to ~0.1 at 20%).

--policy sac records with a state-based SAC expert instead (trained by
train_reacher_sac.py, loaded through stable_worldmodel's ExpertPolicy); --noise is
then its Gaussian action noise, and --random-episode-prob works the same way.

Evaluate trained models on the ORIGINAL reacher_eval.h5 so that only the training
data differs.

Usage:
  MUJOCO_GL=egl python generate_reacher_expert.py --name reacher_expert_train --episodes 9000 --seed 0
  MUJOCO_GL=egl python generate_reacher_expert.py --name reacher_expert_eval  --episodes 1000 --seed 100000
  MUJOCO_GL=egl python generate_reacher_expert.py --name reacher_sac_train --episodes 9000 --seed 0 \
      --policy sac --sac-dir sac_reacher_seed0 --noise 0.1 --random-episode-prob 0.5
"""
from __future__ import annotations

import argparse
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.policy import BasePolicy  # noqa: E402

SHOULDER_RANGE = (-np.pi, np.pi)   # unlimited hinge
WRIST_LIMIT = 2.7925               # wrist joint range is +-160 degrees


def wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class PDWaypointExpert(BasePolicy):
    """Noisy PD controller towards way-points in joint space."""

    def __init__(self, kp=0.7, kd=0.05, noise_std=0.3, tol=0.15, max_segment=60, seed=0,
                 wrist_range=2.4, local_radius=None, random_prob=0.0, max_torque=1.0, random_episode_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.type = "expert"
        self.kp, self.kd, self.noise_std = float(kp), float(kd), float(noise_std)
        self.tol, self.max_segment = float(tol), int(max_segment)
        self.wrist_range = min(float(wrist_range), WRIST_LIMIT)
        self.local_radius = None if local_radius is None else float(local_radius)
        self.random_prob = float(random_prob)
        self.max_torque = float(max_torque)
        self.random_episode_prob = float(random_episode_prob)
        self.random_episode = None
        self.rng = np.random.default_rng(seed)
        self.targets = None
        self.seg_len = None
        self.stats = {"segments": 0, "reached": 0, "segment_steps": []}

    def _new_target(self, q=None):
        w = self.wrist_range
        if self.local_radius is None or q is None:
            return np.array([self.rng.uniform(*SHOULDER_RANGE), self.rng.uniform(-w, w)])
        r = self.local_radius
        return np.array([wrap_angle(q[0] + self.rng.uniform(-r, r)),
                         np.clip(q[1] + self.rng.uniform(-r, r), -w, w)])

    def get_action(self, infos, **kwargs):
        q = np.asarray(infos["qpos"], dtype=np.float64)
        n = q.shape[0]
        q = q.reshape(n, -1)[:, :2]
        qd = np.asarray(infos["qvel"], dtype=np.float64).reshape(n, -1)[:, :2]
        step = np.asarray(infos["step_idx"]).reshape(n, -1)[:, -1]
        if self.targets is None:
            self.targets = np.stack([self._new_target(q[i]) for i in range(n)])
            self.seg_len = np.zeros(n, dtype=int)
            self.random_episode = np.zeros(n, dtype=bool)
        actions = np.empty((n, 2), dtype=np.float32)
        for i in range(n):
            if step[i] == 0:                                   # new episode
                self.targets[i], self.seg_len[i] = self._new_target(q[i]), 0
                self.random_episode[i] = self.rng.random() < self.random_episode_prob
            if self.random_episode[i]:                         # whole episode uniform random
                actions[i] = self.rng.uniform(-1.0, 1.0, size=2)
                continue
            err = np.array([wrap_angle(self.targets[i, 0] - q[i, 0]), self.targets[i, 1] - q[i, 1]])
            reached = np.all(np.abs(err) < self.tol) and np.all(np.abs(qd[i]) < 1.0)
            if reached or self.seg_len[i] >= self.max_segment:
                self.stats["segments"] += 1
                self.stats["reached"] += int(reached)
                self.stats["segment_steps"].append(int(self.seg_len[i]))
                self.targets[i], self.seg_len[i] = self._new_target(q[i]), 0
                err = np.array([wrap_angle(self.targets[i, 0] - q[i, 0]), self.targets[i, 1] - q[i, 1]])
            u = self.kp * err - self.kd * qd[i] + self.noise_std * self.rng.standard_normal(2)
            actions[i] = np.clip(u, -self.max_torque, self.max_torque)
            if self.random_prob and self.rng.random() < self.random_prob:
                actions[i] = self.rng.uniform(-1.0, 1.0, size=2)
            self.seg_len[i] += 1
        return actions


class SACExpert(BasePolicy):
    """stable_worldmodel's DMC ExpertPolicy (SB3 SAC) with whole random-action episodes mixed in."""

    def __init__(self, sac_dir, noise_std=0.0, random_episode_prob=0.0, seed=0, **kwargs):
        super().__init__(**kwargs)
        from stable_worldmodel.envs.dmcontrol.expert_policy import ExpertPolicy
        self.expert = ExpertPolicy(os.path.join(sac_dir, "expert_policy.zip"),
                                   os.path.join(sac_dir, "vec_normalize.pkl"), noise_std=noise_std, seed=seed)
        self.type = "expert"
        self.rng = np.random.default_rng(seed + 1)
        self.random_episode_prob = float(random_episode_prob)
        self.random_episode = None
        self.stats = {"segments": 0, "reached": 0, "segment_steps": []}

    def set_env(self, env):
        super().set_env(env)
        self.expert.set_env(env)

    def get_action(self, infos, **kwargs):
        actions = self.expert.get_action(infos)
        n = actions.shape[0]
        step = np.asarray(infos["step_idx"]).reshape(n, -1)[:, -1]
        if self.random_episode is None:
            self.random_episode = np.zeros(n, dtype=bool)
        for i in range(n):
            if step[i] == 0:
                self.random_episode[i] = self.rng.random() < self.random_episode_prob
            if self.random_episode[i]:
                actions[i] = self.rng.uniform(-1.0, 1.0, size=actions.shape[1])
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
    ap.add_argument("--wrist-range", type=float, default=2.4, help="max |wrist| of way-points (limit 2.79)")
    ap.add_argument("--local-radius", type=float, default=None, help="draw way-points within +-r rad of the current q")
    ap.add_argument("--random-prob", type=float, default=0.0, help="per-step probability of a uniform random action")
    ap.add_argument("--max-torque", type=float, default=1.0, help="clip the PD torque to +-this (slower, still goal-directed motion)")
    ap.add_argument("--random-episode-prob", type=float, default=0.0,
                    help="fraction of episodes recorded with uniform random actions (as in the published dataset)")
    ap.add_argument("--policy", choices=["pd", "sac"], default="pd")
    ap.add_argument("--sac-dir", default=None, help="folder with expert_policy.zip and vec_normalize.pkl (--policy sac)")
    a = ap.parse_args()

    # Same settings as the published dataset: 224x224 frames, 200 control steps
    # per episode (201 rows with the initial state), no separate goal, default
    # task/renderer (the target is not drawn).
    world = swm.World("swm/ReacherDMControl-v0", num_envs=a.num_envs, image_shape=(224, 224),
                      max_episode_steps=200, seed=a.seed, goal_conditioned=False, verbose=0)
    if a.policy == "sac":
        policy = SACExpert(a.sac_dir, noise_std=a.noise, random_episode_prob=a.random_episode_prob, seed=a.seed)
    else:
        policy = PDWaypointExpert(kp=a.kp, kd=a.kd, noise_std=a.noise, tol=a.tol,
                                  max_segment=a.max_segment, seed=a.seed, wrist_range=a.wrist_range,
                                  local_radius=a.local_radius, random_prob=a.random_prob, max_torque=a.max_torque,
                                  random_episode_prob=a.random_episode_prob)
    world.set_policy(policy)
    world.record_dataset(a.name, episodes=a.episodes, seed=a.seed, cache_dir=a.out)
    s = policy.stats
    seg = np.array(s["segment_steps"]) if s["segment_steps"] else np.array([0])
    print(f"wrote {a.out}/{a.name}.h5: {a.episodes} episodes; way-point segments {s['segments']}, "
          f"reached {s['reached'] / max(1, s['segments']):.0%}, segment length median {int(np.median(seg))} "
          f"(p90 {int(np.percentile(seg, 90))}) steps", flush=True)


if __name__ == "__main__":
    main()
