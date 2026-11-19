#!/usr/bin/env python3
"""Read-only check of loss.policy.goal_frame_only on real data.

For each goal offset G it builds, on one HDF5 dataset,
  dense:   HDF5Dataset with num_steps = context + G, the clip that goal_source: offset loads
           without goal_frame_only (every frame up to the goal);
  sparse:  train.OffsetGoalHDF5Dataset, the two context frames plus the goal frame;
  shared:  the same with clip_frames (loss.policy.goal_clip_frames), the clip starts of the sweep;
and checks that sparse has the clip starts of dense, that shared has the clip starts of a
clip_frames-frame clip, and, for a sample of clips, that frames, actions and observations are
equal and that the goal frame is the dense clip's frame context - 1 + G.

Run it in the training environment (a login node is enough, it reads a few hundred frames):
  python -B check_goal_frame_loading.py [--dataset reacher_eval] [--goal-offsets 1 2 3 5 10]
                                        [--clip-frames 12] [--samples 20]
"""
import argparse
import getpass
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # planning/

import numpy as np
import torch
import stable_worldmodel as swm

import train

if not hasattr(train, "OffsetGoalHDF5Dataset"):
    sys.exit(f"{train.__file__} has no OffsetGoalHDF5Dataset: git pull first")


def same(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(torch.nan_to_num(a, 0.0), torch.nan_to_num(b, 0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="reacher_eval")
    ap.add_argument("--data-root",
                    default=os.environ.get("EXTERNAL_DATA_ROOT", f"/scratch/{getpass.getuser()}/smwm-data"))
    ap.add_argument("--goal-offsets", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--clip-frames", type=int, default=12)
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    kw = dict(name=args.dataset, frameskip=args.frameskip, keys_to_load=["pixels", "action", "observation"],
              keys_to_cache=["action", "observation"], cache_dir=args.data_root, transform=None)
    L, F = args.context, args.clip_frames
    ok = True
    ref = swm.data.HDF5Dataset(num_steps=F, **kw)
    print(f"{args.dataset}: {len(ref.lengths)} episodes; a {F}-frame clip has {len(ref)} start positions")
    for G in args.goal_offsets:
        dense = swm.data.HDF5Dataset(num_steps=L + G, **kw)
        sparse = train.OffsetGoalHDF5Dataset(num_steps=L, goal_offset=G, context=L, **kw)
        shared = train.OffsetGoalHDF5Dataset(num_steps=L, goal_offset=G, context=L, clip_frames=F, **kw)
        starts = sparse.clip_indices == dense.clip_indices and shared.clip_indices == ref.clip_indices
        where = {clip: i for i, clip in enumerate(shared.clip_indices)}
        rng = np.random.default_rng(G)
        picks = sorted(set(rng.integers(0, len(dense), args.samples).tolist() + [0, len(dense) - 1]))
        equal, n_shared = True, 0
        for i in picks:
            d = dense[i]
            candidates = [sparse[i]]
            if dense.clip_indices[i] in where:
                candidates.append(shared[where[dense.clip_indices[i]]])
                n_shared += 1
            for s in candidates:
                equal &= all(same(s[k], d[k][:L]) for k in ("pixels", "action", "observation"))
                equal &= same(s["goal_pixels"], d["pixels"][L - 1 + G])
        ok &= starts and equal
        print(f"  G = {G:2d}: clip starts {'equal' if starts else 'DIFFER'} "
              f"(dense {len(dense)}, shared {len(shared)}); "
              f"{len(picks)} samples ({n_shared} also in the shared set) {'equal' if equal else 'DIFFER'}")
        for ds in (dense, sparse, shared):
            if ds.h5_file is not None:
                ds.h5_file.close()
    print("PASS: goal_frame_only loads the samples of the dense clip" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
