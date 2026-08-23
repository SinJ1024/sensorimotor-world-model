#!/usr/bin/env python3
"""Dump the column names of a le-wm HDF5 dataset and check whether the
planning-eval `callables` will find the columns they reference.

Usage:
    python scripts/inspect_h5_columns.py data/external/tworoom_eval.h5
    python scripts/inspect_h5_columns.py data/external/pusht_expert_eval.h5
    python scripts/inspect_h5_columns.py path/to/foo.h5.zst      # .zst is auto-decompressed to a temp file

The eval loader (swm.data.HDF5Dataset) exposes each leaf dataset as a "column".
During evaluate_from_dataset, every non-goal column X is also auto-exposed as
`goal_X` (and `pixels` -> `goal`). So a callable that reads `goal_proprio` only
needs a `proprio` column to exist.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import h5py


# columns each config's callables need to FIND in the raw dataset.
# goal_* are auto-derived, so we only require the base column to exist.
REQUIRED = {
    "tworoom (le-wm / _set_state)": ["proprio"],          # goal_proprio auto-derived
    "pusht   (le-wm / _set_state)": ["state"],            # goal_state   auto-derived
    "tworoom (repo guess / set_wrapper_attr)": ["pos_agent", "pos_target"],
}

# columns the eval pipeline itself looks for (episode grouping + start sampling)
PIPELINE = ["action", "episode_idx", "ep_idx", "step_idx", "pixels", "proprio"]


def maybe_decompress(path: Path) -> Path:
    if path.suffix != ".zst":
        return path
    import zstandard as zstd

    tmp = Path(tempfile.mkstemp(suffix=".h5")[1])
    with path.open("rb") as s, tmp.open("wb") as d:
        zstd.ZstdDecompressor().copy_stream(s, d)
    print(f"(decompressed {path.name} -> {tmp})")
    return tmp


def walk(h5file: h5py.File) -> list[tuple[str, tuple, str]]:
    """Return (name, shape, dtype) for every leaf dataset."""
    leaves: list[tuple[str, tuple, str]] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            leaves.append((name, obj.shape, str(obj.dtype)))

    h5file.visititems(visitor)
    return leaves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5_path", type=Path, help="path to a .h5 (or .h5.zst) dataset")
    args = ap.parse_args()

    src = args.h5_path.expanduser()
    if not src.is_file():
        sys.exit(f"File not found: {src}")

    path = maybe_decompress(src)

    with h5py.File(path, "r") as f:
        print(f"\n=== HDF5 tree: {src.name} ===")
        top = list(f.keys())
        print(f"top-level keys: {top}")

        leaves = walk(f)
        print(f"\n=== leaf datasets ({len(leaves)}) — name  shape  dtype ===")
        for name, shape, dtype in leaves:
            print(f"  {name:<28} {str(shape):<20} {dtype}")

        # build the set of "column" leaf names (last path component + full name)
        leaf_names = set()
        for name, _, _ in leaves:
            leaf_names.add(name)
            leaf_names.add(name.split("/")[-1])

    def have(col: str) -> bool:
        return col in leaf_names

    print("\n=== pipeline columns (episode grouping / start sampling) ===")
    for col in PIPELINE:
        print(f"  [{'OK ' if have(col) else 'MISSING'}] {col}")

    print("\n=== callables verdict ===")
    for label, cols in REQUIRED.items():
        results = {c: have(c) for c in cols}
        ok = all(results.values())
        detail = ", ".join(f"{c}={'OK' if v else 'MISSING'}" for c, v in results.items())
        print(f"  [{'USABLE  ' if ok else 'BROKEN  '}] {label}: {detail}")

    print(
        "\nNote: a MISSING callable column is NOT a crash — swm only logs a warning "
        "and skips it,\n      leaving agent/goal placed randomly (silently wrong "
        "success rates)."
    )


if __name__ == "__main__":
    main()
