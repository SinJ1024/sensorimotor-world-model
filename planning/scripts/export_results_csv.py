#!/usr/bin/env python3
"""Write every run of the policy-regularizer study to one CSV (planning/results.csv).

One row per run and cluster, built from the per-cluster summarize_runs.py TSVs:
what was trained (environment, regularizer, lambda, seed), how training ended
(steps, losses, effective rank, paper hyper-parameter verdict) and the paper-protocol
CEM success rate (100 tasks, goal 25 primitive steps ahead, budget 50).
The upstream paper runs are not included; they stay in
experiments/planning_eval/aggregated_results.csv.

Usage:
  python export_results_csv.py --tsv DelftBlue=db.tsv --tsv Leonardo=leo.tsv [--out ../results.csv]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_table import labelled, read_runs  # noqa: E402

COLUMNS = [
    "env", "method", "seed", "cluster", "regularizer", "lambda", "success_rate", "status",
    "other_evals", "eff_rank", "pred_loss", "inv_loss", "policy_loss", "goal_loss",
    "step", "max_steps", "verdict", "deviations", "eval_task_seed", "cem_seed",
    "run", "run_dir", "metrics_path",
]
# run-name prefix -> (env label, task seed, CEM seed base), as in the evaluators
ENVS = {"tworoom": ("TwoRoom", 42025, 52025), "reacher": ("Reacher", 43025, 53025),
        "pusht": ("Push-T", 44025, 54025), "cube": ("OGBench-Cube", 45025, 55025)}


def method_of(run: str) -> str:
    """tworoom_policy_mlp_c2k1_noact_seed0 -> policy_mlp_c2k1_noact; *_inverse_lambda_* -> inverse."""
    family = re.sub(r"_seed\d+$", "", run).split("_", 1)[1]
    return "inverse" if family.startswith("inverse_lambda_") else family


def row_for(r: dict) -> dict | None:
    m = re.search(r"_seed(\d+)$", r["run"])
    prefix = r["run"].split("_")[0]
    if prefix not in ENVS or not m:
        return None
    label, task_seed, cem_base = ENVS[prefix]
    seed = int(m.group(1))
    success = r.get("success") or ""
    try:
        float(success)
        status = "ok"
    except ValueError:
        status = "no_eval" if r.get("verdict") == "PASS" else "incomplete"
        success = ""
    out = {k: r.get(k, "") for k in COLUMNS}
    out.update(env=label, method=method_of(r["run"]), seed=seed, cluster=r["cluster"],
               regularizer=r.get("method", ""), success_rate=success, status=status,
               eval_task_seed=task_seed, cem_seed=cem_base + seed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", action="append", default=[], help="CLUSTER=summary.tsv (repeatable)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results.csv")
    args = ap.parse_args()
    if not args.tsv:
        raise SystemExit("give at least one --tsv CLUSTER=path")

    rows = [x for x in map(row_for, read_runs(labelled(args.tsv))) if x]
    order = {label: i for i, (label, _, _) in enumerate(ENVS.values())}
    rows.sort(key=lambda x: (order[x["env"]], x["method"], x["seed"], x["cluster"]))
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    done = sum(x["status"] == "ok" for x in rows)
    print(f"wrote {args.out}: {len(rows)} runs ({done} evaluated)")


if __name__ == "__main__":
    main()
