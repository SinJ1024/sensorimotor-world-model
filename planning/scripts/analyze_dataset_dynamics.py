#!/usr/bin/env python3
"""How much signal does each regularizer get from each dataset?

Uses the privileged physical state stored in the LeWM datasets (an upper bound on
what a perfect encoder could extract) and the training-time frameskip of 5, so one
"step" t -> t+1 below is 5 primitive steps and an action is a block of 5 primitive
actions, exactly as train.py sees them. All scores are R^2 on held-out episodes
(variance weighted over output dimensions), from a ridge regression (linear) and a
2x256 MLP (nonlinear).

  inverse     a_t     <- (s_t, s_{t+1})                what the inverse head is trained on
  c2          a_{t+1} <- (s_t, s_{t+1}, task)          no-action policy head, two states
  c1          a_{t+1} <- (s_{t+1}, task)               no-action policy head, one state
  a_t only    a_{t+1} <- a_t                           the shortcut available to the +a_t head
  *_mean      same, with the mean of the 5 primitive actions as the target
  c2 + a_t    a_{t+1} <- (s_t, s_{t+1}, task, a_t)     the +a_t head
  control     s_{t+1} - s_t <- s_t   vs   <- (s_t, a_t)   R^2 gain = share of the state
              change that the action explains, per state group (agent vs object)

"pos" uses positions only (what a single frame shows, as the world model uses one
frame of history); "full" adds velocities. "task" is the episode's goal when the
dataset varies it (TwoRoom target, Reacher target, Cube target block); Push-T's goal
is fixed.

Usage: python analyze_dataset_dynamics.py --data-root /scratch/$USER/smwm-data [--episodes 1500]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

FS = 5  # frameskip used in training

ENVS = {
    "tworoom": dict(file="tworoom_train.h5", ep_key="ep_idx",
                    pos={"agent": ("pos_agent", None)}, vel={},
                    task=[("pos_target", None)]),
    "reacher": dict(file="reacher_train.h5", ep_key="ep_idx",
                    pos={"arm": ("qpos", None)}, vel={"arm": ("qvel", None)},
                    task=[("target_pos", None)]),
    "pusht": dict(file="pusht_expert_train.h5", ep_key="episode_idx",
                  # state = agent x, y, block x, y, block angle, agent vx, vy
                  pos={"agent": ("state", slice(0, 2)), "object": ("state", slice(2, 5))},
                  vel={"agent": ("state", slice(5, 7))}, task=[]),
    "cube": dict(file="cube_single_expert_train.h5", ep_key="ep_idx",
                 # qpos = 6 arm joints, gripper joints, then the block's free joint (7)
                 pos={"arm": ("qpos", slice(0, 14)), "object": ("qpos", slice(14, 21))},
                 vel={"arm": ("qvel", slice(0, 14)), "object": ("qvel", slice(14, 20))},
                 task=[("privileged_target_block_pos", None), ("privileged_target_block_yaw", None)]),
}


def load(root: str, env: str, n_episodes: int, seed: int = 0):
    import h5py
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    cfg = ENVS[env]
    rng = np.random.default_rng(seed)
    keys = {k for grp in ("pos", "vel") for k, _ in cfg[grp].values()} | {k for k, _ in cfg["task"]} | {"action"}
    with h5py.File(f"{root}/{cfg['file']}", "r") as h:
        ep_len, ep_off = h["ep_len"][:], h["ep_offset"][:]
        cols = {k: np.asarray(h[k][:], dtype=np.float64).reshape(len(h[k]), -1) for k in keys}
    eps = rng.choice(len(ep_len), size=min(n_episodes, len(ep_len)), replace=False)
    # non-overlapping windows t, t+5, t+10 inside each sampled episode
    idx0, ep = [], []
    for e in eps:
        L, o = int(ep_len[e]), int(ep_off[e])
        for t in range(0, L - 2 * FS, FS):
            idx0.append(o + t)
            ep.append(e)
    idx0 = np.asarray(idx0)
    idx1 = idx0 + FS

    def grab(name, sl, idx):
        x = cols[name][idx]
        return x if sl is None else x[:, sl]

    def actblock(start):  # 5 primitive actions -> one block, as train.py concatenates them
        return np.concatenate([cols["action"][start + i] for i in range(FS)], axis=1)

    out = {"ep": np.asarray(ep), "a0": actblock(idx0), "a1": actblock(idx1)}
    for grp in ("pos", "vel"):
        for part, (k, sl) in cfg[grp].items():
            out[f"{grp}:{part}:0"] = grab(k, sl, idx0)
            out[f"{grp}:{part}:1"] = grab(k, sl, idx1)
    out["task"] = (np.concatenate([grab(k, sl, idx1) for k, sl in cfg["task"]], axis=1)
                   if cfg["task"] else np.zeros((len(idx0), 0)))
    return out


def r2_pair(X, Y, groups, seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    test_g = rng.choice(ug, size=max(1, len(ug) // 5), replace=False)
    te = np.isin(groups, test_g)
    keep = Y[~te].std(0) > 1e-8
    if not keep.any():
        return float("nan"), float("nan")
    X, Y = X, Y[:, keep]
    sx, sy = StandardScaler().fit(X[~te]), StandardScaler().fit(Y[~te])
    Xtr, Xte, Ytr, Yte = sx.transform(X[~te]), sx.transform(X[te]), sy.transform(Y[~te]), sy.transform(Y[te])
    lin = r2_score(Yte, Ridge(alpha=1.0).fit(Xtr, Ytr).predict(Xte), multioutput="variance_weighted")
    mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=200, early_stopping=True,
                       random_state=seed).fit(Xtr, Ytr)
    nl = r2_score(Yte, mlp.predict(Xte), multioutput="variance_weighted")
    return float(lin), float(nl)


def analyze(args):
    root, env, n_ep = args
    d = load(root, env, n_ep)
    g = d["ep"]
    cfg = ENVS[env]
    parts = list(cfg["pos"])
    def S(t, full):
        xs = [d[f"pos:{p}:{t}"] for p in parts]
        if full:
            xs += [d[f"vel:{p}:{t}"] for p in cfg["vel"]]
        return np.concatenate(xs, axis=1)
    res = {"env": env, "n": int(len(g)), "action_block_dim": int(d["a0"].shape[1])}
    m0 = d["a0"].reshape(len(g), FS, -1).mean(1)
    m1 = d["a1"].reshape(len(g), FS, -1).mean(1)
    for view in ("pos", "full"):
        full = view == "full"
        if full and not cfg["vel"]:
            continue
        s0, s1, task = S(0, full), S(1, full), d["task"]
        res[f"inverse/{view}"] = r2_pair(np.hstack([s0, s1]), d["a0"], g)
        res[f"c2/{view}"] = r2_pair(np.hstack([s0, s1, task]), d["a1"], g)
        res[f"c1/{view}"] = r2_pair(np.hstack([s1, task]), d["a1"], g)
        res[f"c2+a_t/{view}"] = r2_pair(np.hstack([s0, s1, task, d["a0"]]), d["a1"], g)
        # Same, with the block's MEAN action as the target (its net direction): two
        # frames 5 steps apart can only pin down the sum of the 5 actions.
        res[f"inverse_mean/{view}"] = r2_pair(np.hstack([s0, s1]), m0, g)
        res[f"c2_mean/{view}"] = r2_pair(np.hstack([s0, s1, task]), m1, g)
        res[f"c1_mean/{view}"] = r2_pair(np.hstack([s1, task]), m1, g)
    res["a_t only"] = r2_pair(d["a0"], d["a1"], g)
    res["a_t only_mean"] = r2_pair(m0, m1, g)
    # controllability per state group, from positions (+velocities where stored)
    for p in parts:
        ds = d[f"pos:{p}:1"] - d[f"pos:{p}:0"]
        s0 = S(0, bool(cfg["vel"]))
        without = r2_pair(s0, ds, g)
        with_a = r2_pair(np.hstack([s0, d["a0"]]), ds, g)
        res[f"control/{p}"] = {"without_action": without, "with_action": with_a}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--out", default="dataset_dynamics.json")
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(analyze, [(a.data_root, e, a.episodes) for e in ENVS]))
    json.dump(results, open(a.out, "w"), indent=1)
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
