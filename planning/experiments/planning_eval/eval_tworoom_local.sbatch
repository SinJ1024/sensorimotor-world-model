#!/bin/bash
# Recover TwoRoom from its archive and evaluate within one node-local job.
#SBATCH --job-name=tworoom-local
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=tworoom-local-%j.out

set -eo pipefail
PROJECT="${SMWM_PROJECT_ROOT:-$HOME/sensorimotor-world-model}"
RUN_DIR="${1:?usage: sbatch eval_tworoom_local.sbatch /absolute/run_dir [eval arguments]}"
shift
module load 2025 cuda
source "$PROJECT/.venv/bin/activate"
set -u
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export RECOVERY_ARCHIVE="${RECOVERY_ARCHIVE:-/scratch/$USER/smwm-data/tworoom.tar.zst}"
export RECOVERY_RUN="$RUN_DIR"
export RECOVERY_PROJECT="$PROJECT"
RECOVERY_LOCAL="$(mktemp -d /tmp/tworoom-${USER}-${SLURM_JOB_ID}-XXXXXX)"
export RECOVERY_LOCAL
echo "Node: $(hostname); local directory: $RECOVERY_LOCAL"
df -hT "$RECOVERY_LOCAL"
# Keep failures for diagnosis; never delete the source archive or checkpoint.
trap 'echo "Local files: $(hostname):$RECOVERY_LOCAL (retained on failure)"' ERR

python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import h5py
import hdf5plugin
import zstandard as zstd

work = Path(os.environ['RECOVERY_LOCAL'])
kind = subprocess.check_output(['stat', '-f', '-c', '%T', str(work)], text=True).strip()
if kind not in {'ext2/ext3', 'ext4', 'xfs', 'btrfs', 'zfs'}:
    raise RuntimeError(f'Expected a disk-backed local filesystem, found {kind}')
if shutil.disk_usage(work).free < 40 * 1024**3:
    raise RuntimeError('Need at least 40 GiB free on this node before recovery')

planning = work / 'planning'
(planning / 'scripts').mkdir(parents=True)
source = Path(os.environ['RECOVERY_PROJECT']) / 'planning'
for path in source.glob('*.py'):
    shutil.copyfile(path, planning / path.name)
shutil.copytree(source / 'config', planning / 'config')
shutil.copyfile(source / 'scripts/make_episode_splits.py', planning / 'scripts/make_episode_splits.py')
run = work / 'run'
(run / 'checkpoints').mkdir(parents=True)
original_run = Path(os.environ['RECOVERY_RUN'])
for name in ['config.yaml', 'checkpoints/last.ckpt']:
    print('Copying checkpoint input:', name, flush=True)
    shutil.copyfile(original_run / name, run / name)

archive = work / 'tworoom.tar.zst'
print('Copying readable archive to local disk...', flush=True)
shutil.copyfile(os.environ['RECOVERY_ARCHIVE'], archive)
with archive.open('rb') as f:
    archive_sha = hashlib.file_digest(f, 'sha256').hexdigest()
data = work / 'data'
data.mkdir()
found = False
with archive.open('rb') as f:
    with zstd.ZstdDecompressor().stream_reader(f) as stream:
        with tarfile.open(fileobj=stream, mode='r|') as tar:
            for member in tar:
                if not member.isfile() or member.name.rsplit('/', 1)[-1] != 'tworoom.h5':
                    continue
                if found:
                    raise RuntimeError('Archive contains multiple tworoom.h5 files')
                found = True
                if shutil.disk_usage(work).free < 2 * member.size + 2 * 1024**3:
                    raise RuntimeError('Insufficient local space for extracted data and splits')
                print('Extracting:', member.name, flush=True)
                with tar.extractfile(member) as inp, (data / 'tworoom.h5').open('xb') as out:
                    shutil.copyfileobj(inp, out, length=8 * 1024**2)
if not found:
    raise RuntimeError('No tworoom.h5 found in archive')
sys.path.insert(0, str(planning / 'scripts'))
from make_episode_splits import SplitJob, run_job, SEED, TRAIN_FRAC
run_job(data, SplitJob('tworoom.h5', 'tworoom_train.h5', 'tworoom_eval.h5'), False)
for name in ['tworoom_train.h5', 'tworoom_eval.h5']:
    with h5py.File(data / name, 'r') as f:
        for key, d in f.items():
            if not isinstance(d, h5py.Dataset):
                continue
            if d.ndim == 0:
                d[()]
            else:
                for start in range(0, d.shape[0], 64):
                    d[start:start + 64]
        print('Full read passed:', name, flush=True)
(work / 'recovery.json').write_text(json.dumps({
    'archive': os.environ['RECOVERY_ARCHIVE'], 'archive_sha256': archive_sha,
    'original_run_dir': str(original_run), 'split_seed': SEED, 'train_fraction': TRAIN_FRAC,
    'comparison_status': 'Reconstructed data; equivalence to earlier evaluations not yet verified',
}, indent=2))
PY

export EXTERNAL_DATA_ROOT="$RECOVERY_LOCAL/data"
export STABLEWM_HOME="$EXTERNAL_DATA_ROOT"
export REPO_ROOT="$RECOVERY_LOCAL/planning"
python -u "$RECOVERY_LOCAL/planning/eval_tworoom_policy.py" \
    --run-dir "$RECOVERY_LOCAL/run" --data-root "$EXTERNAL_DATA_ROOT" \
    --output-dir "$RECOVERY_LOCAL/results" "$@"
cp "$RECOVERY_LOCAL/recovery.json" "$RECOVERY_LOCAL/results/recovery.json"
echo '=== EVALUATION COMPLETED: summary follows ==='
cat "$RECOVERY_LOCAL/results/summary.json"
echo

export RECOVERY_DEST="${EVAL_OUTPUT_ROOT:-/scratch/$USER/smwm-evals}/$(basename "$RUN_DIR")/${SLURM_JOB_ID}"
# Verify persistent copies before cleaning this job's private local directory.
python - <<'PY'
import hashlib
import os
from pathlib import Path
import shutil

work = Path(os.environ['RECOVERY_LOCAL']).resolve()
source = work / 'results'
dest = Path(os.environ['RECOVERY_DEST'])
try:
    shutil.copytree(source, dest)
    for src in source.rglob('*'):
        if src.is_file():
            with src.open('rb') as a, (dest / src.relative_to(source)).open('rb') as b:
                if hashlib.file_digest(a, 'sha256').digest() != hashlib.file_digest(b, 'sha256').digest():
                    raise RuntimeError(f'Result copy verification failed: {src.name}')
except Exception:
    print('EVALUATION SUCCEEDED, BUT RESULT PERSISTENCE FAILED.', flush=True)
    print(f'Local results retained: {source}', flush=True)
    raise
print(f'Results saved and verified: {dest}', flush=True)
if work.parent != Path('/tmp') or not work.name.startswith('tworoom-') or work.is_symlink():
    raise RuntimeError('Refusing cleanup outside private /tmp job directory')
shutil.rmtree(work)
PY
