#!/bin/bash
# Refresh planning/RESULTS.md from every cluster. Run on the local machine.
#
# On each cluster: scripts/summarize_runs.py -> TSV (it loads every checkpoint,
# ~4 min on DelftBlue) plus the current squeue; then results_table.py merges them.
# A cluster that cannot be reached, or has no finished runs yet, contributes only
# what it can (its queue, or nothing).
#
#   planning/scripts/refresh_results.sh
#   DELFTBLUE_HOST=delftblue14 planning/scripts/refresh_results.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELFTBLUE_HOST="${DELFTBLUE_HOST:-delftblue13}"
LEONARDO_HOST="${LEONARDO_HOST:-leonardo}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
ARGS=()

fetch() {  # fetch <label> <host> <summary command> <queue command>
    local label=$1 host=$2 summary=$3 queue=$4
    echo ">> $label ($host)"
    if ssh -o ConnectTimeout=15 "$host" "$summary" > "$TMP/$label.tsv" 2> "$TMP/$label.err" \
            && [ "$(wc -l < "$TMP/$label.tsv")" -gt 1 ]; then
        ARGS+=(--tsv "$label=$TMP/$label.tsv")
        echo "   $(( $(wc -l < "$TMP/$label.tsv") - 1 )) runs"
    else
        echo "   no summary ($(grep -v 'post-quantum\|store now\|openssh.com/pq' "$TMP/$label.err" | tail -1))"
    fi
    if ssh -o ConnectTimeout=15 "$host" "$queue" > "$TMP/$label.queue" 2>/dev/null; then
        ARGS+=(--queue "$label=$TMP/$label.queue")
    fi
}

# summarize_runs.py writes the TSV with --out; print it to stdout for transfer.
fetch DelftBlue "$DELFTBLUE_HOST" \
    'rm -f /tmp/$USER-smwm-summary.tsv; cd ~/sensorimotor-world-model && for p in .venv/bin/python ~/conda-envs/biased-dreams-dreamer/bin/python; do [ -x "$p" ] && "$p" -c "import torch, yaml" 2>/dev/null && break; done && "$p" planning/scripts/summarize_runs.py --runs-root /scratch/$USER/smwm-runs --eval-root /scratch/$USER/smwm-paper-eval --budget 50 --out /tmp/$USER-smwm-summary.tsv >/dev/null 2>&1; cat /tmp/$USER-smwm-summary.tsv' \
    'squeue -u $USER -h -o "%i	%j	%T	%M"'

fetch Leonardo "$LEONARDO_HOST" \
    'rm -f /tmp/$USER-smwm-summary.tsv; cd $WORK/$USER/sensorimotor-world-model && .venv/bin/python planning/scripts/summarize_runs.py --runs-root $WORK/$USER/smwm-runs --eval-root $WORK/$USER/smwm-paper-eval --budget 50 --out /tmp/$USER-smwm-summary.tsv >/dev/null 2>&1; cat /tmp/$USER-smwm-summary.tsv' \
    'squeue -u $USER -h -o "%i	%j	%T	%M" | grep -P "\t(te|t|e)-"'

if ! printf '%s\n' ${ARGS[@]+"${ARGS[@]}"} | grep -q -- '--tsv'; then
    echo "no cluster returned a summary; RESULTS.md left unchanged" >&2
    exit 1
fi
python3 "$HERE/results_table.py" "${ARGS[@]}" --out "$HERE/../RESULTS.md"
