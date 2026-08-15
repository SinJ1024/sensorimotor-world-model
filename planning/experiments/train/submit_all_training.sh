#!/bin/bash
#
# Submit the full inverse-vs-policy training sweep to DelftBlue as ONE dependency
# chain (the education account allows only 1 GPU job at a time, so they run serially).
#
#   Plan: TwoRoom seeds {1,2}  (seed 0 already trained)
#         Reacher / Push-T / OGBench-Cube seeds {0,1,2}
#         both methods (inverse, policy)  ->  22 jobs, ~3.8 h each.
#
# Run from planning/experiments/train AFTER prepare_all_data.sh has produced the
# *_train.h5 / *_eval.h5 for every environment.
#
#   cd ~/sensorimotor-world-model/planning/experiments/train
#   ./submit_all_training.sh
#
set -euo pipefail

ACCT="${SLURM_ACCOUNT:-education-eemcs-msc-dsait}"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXP_DIR"

echo ">> generating configs (incl. policy) ..."
python generate_configs.py >/dev/null

# Build the ordered run list from the manifest. TwoRoom first (its data is ready),
# then reacher/pusht/cube. Only the seeds we want, only inverse+policy.
mapfile -t RUNS < <(python - <<'PY'
import csv
plan = {"TwoRoom": {1, 2}, "Reacher": {0,1,2}, "Push-T": {0,1,2}, "OGBench-Cube": {0,1,2}}
env_order = ["TwoRoom", "Reacher", "Push-T", "OGBench-Cube"]
rows = list(csv.DictReader(open("generated_configs/manifest.tsv"), delimiter="\t"))
picked = [r for r in rows
          if r["method"] in ("inverse", "policy")
          and int(r["seed"]) in plan.get(r["environment"], set())]
picked.sort(key=lambda r: (env_order.index(r["environment"]), int(r["seed"]), r["method"]))
for r in picked:
    print(r["run_name"])
PY
)

echo ">> ${#RUNS[@]} jobs will be chained (serial):"
printf '   %s\n' "${RUNS[@]}"

PREV=""
for run in "${RUNS[@]}"; do
    if [ -z "$PREV" ]; then
        JID=$(sbatch --parsable --partition=gpu --account="$ACCT" \
              train_standalone.sbatch "$run" trainer.precision=16-mixed)
    else
        JID=$(sbatch --parsable --dependency=afterany:"$PREV" --partition=gpu --account="$ACCT" \
              train_standalone.sbatch "$run" trainer.precision=16-mixed)
    fi
    echo "   submitted $run -> job $JID (after ${PREV:-START})"
    PREV="$JID"
done

echo ">> all submitted. Monitor with:  squeue -u \$USER"
