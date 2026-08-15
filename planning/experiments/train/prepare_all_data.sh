#!/bin/bash
#
# Download + extract + split the LeWM datasets for reacher, pusht, cube.
# RUN ON A LOGIN NODE (needs internet; compute nodes have none).
# TwoRoom is assumed already prepared. Idempotent: skips a step if its output exists.
#
#   nohup ./prepare_all_data.sh > ~/prep_data.log 2>&1 &
#   tail -f ~/prep_data.log
#
# Sizes: reacher 23.7 GB, pusht 13.1 GB, cube 46.2 GB compressed (300+ GB extracted).
#
set -uo pipefail   # NOT -e: one env failing must not abort the others

DATA="${EXTERNAL_DATA_ROOT:-/scratch/$USER/smwm-data}"
REPO="$HOME/sensorimotor-world-model/planning"
mkdir -p "$DATA"; cd "$DATA"

echo "=== disk before ==="; df -h "$DATA" | tail -1

dl() {  # repo_id filename
    if [ ! -f "$DATA/$2" ]; then
        echo ">> downloading $2 from $1 ..."
        python - "$1" "$2" "$DATA" <<'PY'
import sys
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id=sys.argv[1], repo_type="dataset",
                    filename=sys.argv[2], local_dir=sys.argv[3])
print("  ->", p)
PY
    else echo ">> $2 already downloaded"; fi
}

# ---- reacher: reacher.tar.zst -> reacher.h5 --------------------------------
if [ ! -f "$DATA/reacher_train.h5" ] && [ ! -f "$DATA/reacher.h5" ]; then
    dl quentinll/lewm-reacher reacher.tar.zst
    echo ">> extracting reacher.tar.zst ..."; tar -I zstd -xf reacher.tar.zst
fi

# ---- pusht: pusht_expert_train.h5.zst IS the full set -> pusht_expert.h5 ---
if [ ! -f "$DATA/pusht_expert_eval.h5" ] && [ ! -f "$DATA/pusht_expert.h5" ]; then
    dl quentinll/lewm-pusht pusht_expert_train.h5.zst
    echo ">> decompressing pusht -> pusht_expert.h5 ..."
    python - "$DATA/pusht_expert_train.h5.zst" "$DATA/pusht_expert.h5" <<'PY'
import sys, zstandard
with open(sys.argv[1], "rb") as f, open(sys.argv[2], "wb") as g:
    zstandard.ZstdDecompressor().copy_stream(f, g)
PY
fi

# ---- cube: cube_single_expert.tar.zst -> cube_single_expert.h5 ------------
if [ ! -f "$DATA/cube_single_expert_train.h5" ] && [ ! -f "$DATA/cube_single_expert.h5" ]; then
    dl quentinll/lewm-cube cube_single_expert.tar.zst
    echo ">> extracting cube_single_expert.tar.zst (large) ..."; tar -I zstd -xf cube_single_expert.tar.zst
fi

# ---- split every source present into *_train.h5 / *_eval.h5 ---------------
echo ">> running make_episode_splits ..."
python "$REPO/scripts/make_episode_splits.py" --root "$DATA"

echo "=== disk after ==="; df -h "$DATA" | tail -1
echo ">> DONE. Expected files (verify all four envs are present):"
ls -lh "$DATA"/{tworoom,reacher,pusht_expert,cube_single_expert}_{train,eval}.h5 2>/dev/null
