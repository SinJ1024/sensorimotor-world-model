#!/usr/bin/env python3
"""Merge per-cluster summarize_runs.py outputs into one markdown results table.

Each cluster runs scripts/summarize_runs.py (which verifies every run against the
paper hyper-parameters and joins its CEM result) and writes a TSV. This script
labels the rows with the cluster they came from and writes RESULTS.md with
  1. the policy-head input ablation (seed 0, one row per environment),
  2. mean and spread over seeds for every run family with >= 2 evaluated seeds,
  3. every run, grouped by environment,
  4. jobs still queued or running (optional, from squeue dumps),
  5. runs whose verdict is not PASS.

Usage:
  python results_table.py --tsv DelftBlue=db.tsv --tsv Leonardo=leo.tsv \
      [--queue DelftBlue=db_queue.tsv ...] [--out ../RESULTS.md]

A queue file has one job per line: <job_id>\t<job_name>\t<state>\t<elapsed>.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import statistics
from collections import defaultdict
from pathlib import Path

ENV_ORDER = ["tworoom", "reacher", "pusht", "cube"]
ENV_LABEL = {"tworoom": "TwoRoom", "reacher": "Reacher", "pusht": "Push-T", "cube": "OGBench-Cube"}
# Paper lambda, reused for every regularizer (paper Table 2).
LAMBDA = {"tworoom": "0.1", "reacher": "5", "pusht": "30", "cube": "1"}
# Upstream reference numbers (same CEM protocol, 5 seeds) shipped with the repo.
UPSTREAM_CSV = Path(__file__).resolve().parents[1] / "experiments" / "planning_eval" / "aggregated_results.csv"


def upstream_means(budget: str = "50") -> dict[tuple[str, str], float]:
    """{(env, method): mean success} from the upstream aggregated_results.csv."""
    vals = defaultdict(list)
    if UPSTREAM_CSV.is_file():
        with UPSTREAM_CSV.open(newline="") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok" and r.get("eval_budget") == budget:
                    env = "cube" if r["env"] == "ogbcube" else r["env"]
                    vals[(env, r["method"])].append(float(r["success_rate"]))
    return {k: statistics.mean(v) for k, v in vals.items()}

# Ablation columns: header, run-name pattern ({env} is substituted), inputs.
ABLATION = [
    ("inverse", "{env}_inverse_lambda_", "a_t = h(z_t, z_{t+1})"),
    ("policy +a_t", "{env}_policy_mlp_c2k1_seed0", "a_{t+1} = π(z_t, z_{t+1}, a_t)"),
    ("policy noact", "{env}_policy_mlp_c2k1_noact_seed0", "a_{t+1} = π(z_t, z_{t+1})"),
    ("policy c1", "{env}_policy_mlp_c1k1_seed0", "a_{t+1} = π(z_{t+1})"),
]

DETAIL_COLS = [
    ("cluster", "cluster"), ("run", "run"), ("method", "method"), ("λ", "lambda"),
    ("step", "step"), ("eff rank", "eff_rank"), ("pred loss", "pred_loss"),
    ("inv loss", "inv_loss"), ("policy loss", "policy_loss"), ("goal loss", "goal_loss"),
    ("success", "success"), ("verdict", "verdict"), ("other evals", "other_evals"),
]


def labelled(pairs: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in pairs:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected LABEL=PATH, got {item!r}")
        out.append((label, Path(path)))
    return out


def read_runs(sources: list[tuple[str, Path]]) -> list[dict]:
    rows = []
    for cluster, path in sources:
        with path.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                row["cluster"] = cluster
                rows.append(row)
    return rows


def num(value: str | None) -> str:
    """Compact numbers for the table; leave text alone."""
    if value in (None, ""):
        return ""
    try:
        x = float(value)
    except ValueError:
        return value
    if x == int(x) and abs(x) < 1e6:
        return str(int(x))
    return f"{x:.3g}"


def read_queue(sources: list[tuple[str, Path]]) -> list[list[str]]:
    """[cluster, job, name, state, elapsed] per queued/running job."""
    out = []
    for cluster, path in sources:
        for line in path.read_text().splitlines():
            if line.strip():
                job, name, state, elapsed = (line.split("\t") + ["", "", "", ""])[:4]
                out.append([cluster, job, name, state, elapsed])
    return out


def queued_run(job_name: str) -> str:
    """Run name behind a job name: te-<run> (train+eval), t-<run>, e-<run>."""
    for prefix in ("te-", "t-", "e-"):
        if job_name.startswith(prefix):
            return job_name[len(prefix):]
    return job_name


def ablation_cell(rows: list[dict], queue: list[list[str]], env: str, pattern: str) -> str:
    """'100', '69 (DelftBlue) / 70 (Leonardo)', '*queued · Leonardo*', or '–' (not run)."""
    pattern = pattern.format(env=env)
    match = lambda run: run.startswith(pattern) and run.endswith("_seed0")  # noqa: E731
    parts = []
    hits = sorted((r for r in rows if match(r["run"])), key=lambda r: r["cluster"])
    queued = [q for q in queue if match(queued_run(q[2]))]
    for r in hits:
        value = num(r.get("success")) or "*no eval*"
        parts.append(f"{value} ({r['cluster']})" if len(hits) + len(queued) > 1 else value)
    for cluster, _, _, state, _ in queued:
        parts.append(f"*{state.lower()} · {cluster}*")
    return " / ".join(parts) if parts else "–"


def seed_families(rows: list[dict]) -> list[list[str]]:
    """One row per (cluster, run family) with >= 2 evaluated seeds.

    A family is the run name without its _seed<N> suffix. Clusters are kept apart:
    the same run on two clusters is a reproducibility check, not an extra seed.
    """
    groups = defaultdict(list)
    for r in rows:
        m = re.match(r"(.+)_seed(\d+)$", r["run"])
        try:
            value = float(r.get("success") or "")
        except ValueError:  # empty, or a legacy "a/b" with ambiguous seed labels
            continue
        if m:
            groups[(r.get("env", ""), m.group(1), r["cluster"])].append((int(m.group(2)), value))
    out = []
    for (env, family, cluster), seeds in sorted(groups.items(),
            key=lambda kv: (ENV_ORDER.index(kv[0][0]) if kv[0][0] in ENV_ORDER else 9, kv[0][1], kv[0][2])):
        if len(seeds) < 2:
            continue
        seeds.sort()
        vals = [v for _, v in seeds]
        sd = statistics.stdev(vals)
        out.append([ENV_LABEL.get(env, env), family, cluster, str(len(vals)),
                    f"{statistics.mean(vals):.1f} ± {sd:.1f}",
                    ", ".join(f"s{k}: {num(str(v))}" for k, v in seeds)])
    return out


def markdown_table(header: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in body]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", action="append", default=[], help="CLUSTER=summary.tsv (repeatable)")
    ap.add_argument("--queue", action="append", default=[], help="CLUSTER=queue.tsv (repeatable)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "RESULTS.md")
    args = ap.parse_args()
    if not args.tsv:
        raise SystemExit("give at least one --tsv CLUSTER=path")

    rows = read_runs(labelled(args.tsv))
    queue = read_queue(labelled(args.queue))
    clusters = ", ".join(label for label, _ in labelled(args.tsv))
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    md = [
        "# Results",
        "",
        f"Generated {now} from `scripts/summarize_runs.py` on: {clusters}. Success = paper CEM "
        "protocol (100 tasks, goal 25 primitive steps ahead, budget 50), in %. Regenerate with "
        "`scripts/results_table.py` (see its docstring); do not edit by hand.",
        "",
        "## Policy-head input ablation (seed 0)",
        "",
        "Every regularizer uses the paper λ of its environment; `noact` and `c1` feed no actions "
        "to the head. `–` = not run; *italics* = queued or running. Upstream columns: mean of "
        "5 seeds from `experiments/planning_eval/aggregated_results.csv` (same protocol).",
        "",
    ]
    up = upstream_means()
    ref = lambda env, m: f"{up[(env, m)]:.1f}" if (env, m) in up else "–"  # noqa: E731
    header = (["env", "λ"] + [f"{name}<br>`{inputs}`" for name, _, inputs in ABLATION]
              + ["upstream inverse", "upstream forward-only", "upstream random"])
    body = []
    for env in ENV_ORDER:
        body.append([ENV_LABEL[env], LAMBDA[env]]
                    + [ablation_cell(rows, queue, env, pattern) for _, pattern, _ in ABLATION]
                    + [ref(env, "inverse"), ref(env, "forward_only"), ref(env, "random")])
    md += markdown_table(header, body)

    md += ["", "## Across seeds", "",
           "Run families with at least two evaluated seeds; mean ± sample standard deviation.", ""]
    fams = seed_families(rows)
    md += markdown_table(["env", "run family", "cluster", "n", "success", "per seed"], fams) if fams else ["(none yet)"]

    md += ["", "## All runs", ""]
    by_env = defaultdict(list)
    for r in rows:
        by_env[r.get("env", "")].append(r)
    for env in ENV_ORDER + sorted(set(by_env) - set(ENV_ORDER)):
        if not by_env.get(env):
            continue
        md += [f"### {ENV_LABEL.get(env, env or 'unknown')}", ""]
        group = sorted(by_env[env], key=lambda r: (r.get("method", ""), r["run"], r["cluster"]))
        md += markdown_table([h for h, _ in DETAIL_COLS],
                             [[num(r.get(k)) if k not in ("run", "method", "cluster", "verdict", "other_evals") else r.get(k, "")
                               for _, k in DETAIL_COLS] for r in group])
        md.append("")

    if args.queue:
        md += ["## Queued / running", ""]
        md += markdown_table(["cluster", "job", "name", "state", "elapsed"], queue) if queue else ["(none)"]
        md.append("")

    bad = [r for r in rows if r.get("verdict") not in ("PASS", None, "")]
    if bad:
        md += ["## Not PASS", ""]
        md += [f"- **{r['run']}** ({r['cluster']}, {r.get('verdict')}): {r.get('deviations', '')}" for r in bad]
        md.append("")

    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
