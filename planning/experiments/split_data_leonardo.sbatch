#!/bin/bash
# Build the 90/10 episode split for ONE LeWM dataset on the budget-free serial
# partition (login nodes kill long-running processes; the serial queue does not).
# Optionally extracts the dataset tarball first. No internet needed.
#
# Usage: sbatch planning/experiments/split_data_leonardo.sbatch <source.h5> [tarball.tar.zst]
#SBATCH --job-name=smwm-split
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
SRC="${1:?usage: sbatch split_data_leonardo.sbatch <source.h5> [tarball.tar.zst]}"
TARBALL="${2:-}"
PROJECT="$WORK/$USER/sensorimotor-world-model"
PY="$PROJECT/.venv/bin/python"
DATA="$FAST/$USER/smwm-data"
cd "$DATA"
echo "host $(hostname)  source $SRC  tarball ${TARBALL:-none}  start $(date)"

if [ ! -s "$DATA/$SRC" ] && [ -n "$TARBALL" ]; then
    echo "extracting $TARBALL"
    tar -I zstd -xf "$DATA/$TARBALL" -C "$DATA"
fi
[ -e "$DATA/$SRC" ] || { echo "missing $DATA/$SRC" >&2; exit 1; }

"$PY" - "$DATA" "$SRC" "$PROJECT/planning/scripts" <<'PY'
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[3])
import hdf5plugin  # noqa: F401  (blosc filters)
import h5py
import make_episode_splits as m

root, src = Path(sys.argv[1]), sys.argv[2]
jobs = [j for j in m.JOBS if j.source == src]
if not jobs:
    raise SystemExit(f"no split job defined for {src}")
job = jobs[0]
t0 = time.time()
m.run_job(root, job, force=False)
print(f"split {src} in {(time.time() - t0) / 60:.1f} min")

for name in (job.train_out, job.eval_out):
    with h5py.File(root / name, "r") as f:
        ep = f["ep_len"][:]
        n = int(ep.sum())
        f["pixels"][0]
        f["pixels"][n - 1]
        print(f"OK {name}: episodes={len(ep)} rows={n}")
PY
echo "SPLIT_DONE $SRC $(date)"
