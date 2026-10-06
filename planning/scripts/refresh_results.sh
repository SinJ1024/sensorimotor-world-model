#!/bin/bash
# Refresh planning/RESULTS.md from every cluster. Run on the local machine.
#
# On each cluster: the LOCAL scripts/summarize_runs.py is streamed over ssh and
# run from /tmp (so every cluster uses the same version, whatever its checkout)
# -> TSV, plus the current squeue; then results_table.py merges them.
# A cluster that cannot be reached, or has no finished runs yet, contributes only
# what it can (its queue, or nothing).
#
#   planning/scripts/refresh_results.sh
#   DELFTBLUE_HOST=delftblue14 planning/scripts/refresh_results.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${DELFTBLUE_HOST:-}" ]; then   # first DelftBlue login node that answers
    for h in delftblue13 delftblue14; do
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" true 2>/dev/null && { DELFTBLUE_HOST=$h; break; }
    done
    DELFTBLUE_HOST="${DELFTBLUE_HOST:-delftblue13}"
fi
LEONARDO_HOST="${LEONARDO_HOST:-leonardo}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
ARGS=()

fetch() {  # fetch <label> <host> <summary command> <queue command>
    local label=$1 host=$2 summary=$3 queue=$4
    echo ">> $label ($host)"
    if ssh -o ConnectTimeout=15 "$host" "$summary" < "$HERE/summarize_runs.py" \
            > "$TMP/$label.tsv" 2> "$TMP/$label.err" \
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

# The summary command saves the streamed script, runs it (it writes the TSV with
# --out) and prints the TSV for transfer.
S='S=/tmp/$USER-summarize_runs.py; O=/tmp/$USER-smwm-summary.tsv; rm -f $O; cat > $S;'
fetch DelftBlue "$DELFTBLUE_HOST" \
    "$S"' cd ~/sensorimotor-world-model && for p in .venv/bin/python ~/conda-envs/biased-dreams-dreamer/bin/python; do [ -x "$p" ] && "$p" -c "import torch, yaml" 2>/dev/null && break; done && "$p" $S --runs-root /scratch/$USER/smwm-runs --eval-root /scratch/$USER/smwm-paper-eval --budget 50 --out $O >/dev/null 2>&1; cat $O' \
    'squeue -u $USER -h -o "%i	%j	%T	%M"'

fetch Leonardo "$LEONARDO_HOST" \
    "$S"' cd $WORK/$USER/sensorimotor-world-model && .venv/bin/python $S --runs-root $WORK/$USER/smwm-runs --eval-root $WORK/$USER/smwm-paper-eval --budget 50 --out $O >/dev/null 2>&1; cat $O' \
    'squeue -u $USER -h -o "%i	%j	%T	%M" | grep -P "\t(te|t|e)-"'

if ! printf '%s\n' ${ARGS[@]+"${ARGS[@]}"} | grep -q -- '--tsv'; then
    echo "no cluster returned a summary; RESULTS.md left unchanged" >&2
    exit 1
fi
python3 "$HERE/results_table.py" "${ARGS[@]}" --out "$HERE/../RESULTS.md"

# Optionally keep the raw per-cluster TSVs (e.g. for plots or a report page).
if [ -n "${SMWM_RESULTS_TSV_DIR:-}" ]; then
    mkdir -p "$SMWM_RESULTS_TSV_DIR"
    cp "$TMP"/*.tsv "$TMP"/*.queue "$SMWM_RESULTS_TSV_DIR"/ 2>/dev/null
    echo "kept raw TSVs in $SMWM_RESULTS_TSV_DIR"
fi
