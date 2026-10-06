#!/usr/bin/env python3
"""Merge this study's runs into experiments/planning_eval/aggregated_results.csv.

The CSV ships with the upstream (paper) results. This script keeps those rows
untouched, labels them source=upstream, drops every row it wrote before, and
appends one row per run from the per-cluster summarize_runs.py TSVs, labelled
with the cluster in `source`. Consumers that want the paper baselines filter
on source == "upstream" (results_table.py and planning_eval/plot_results.ipynb
do); everything else is this study.

Our rows use the upstream columns (success_rate in %, budget 50, goal 25 steps
ahead, 100 tasks, the same task/CEM seeds) plus: source, regularizer, lambda,
eff_rank, verdict, other_evals. `method` is the run family without environment
and seed, e.g. inverse, policy_mlp_c2k1_noact, goalpolicy_g123,
policy_mlp_c2k1_lam5p0. A run present on two clusters appears once per cluster.

Usage:
  python sync_aggregated_results.py --tsv DelftBlue=db.tsv --tsv Leonardo=leo.tsv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_table import labelled, read_runs  # noqa: E402

CSV_PATH = Path(__file__).resolve().parents[1] / "experiments" / "planning_eval" / "aggregated_results.csv"
UPSTREAM_COLS = ["env", "env_label", "method", "seed_or_repeat", "run_label", "success_rate",
                 "eval_budget", "goal_offset", "num_eval", "eval_task_seed", "policy_seed", "status",
                 "run_dir", "metrics_path", "final_training_run"]
EXTRA_COLS = ["source", "regularizer", "lambda", "eff_rank", "verdict", "other_evals"]

# run-name prefix -> (CSV env slug, label, task seed, CEM seed base); seeds as in the evaluators
ENVS = {
    "tworoom": ("tworoom", "TwoRoom", 42025, 52025),
    "reacher": ("reacher", "Reacher", 43025, 53025),
    "pusht": ("pusht", "Push-T", 44025, 54025),
    "cube": ("ogbcube", "OGBench-Cube", 45025, 55025),
}


def method_of(run: str) -> str:
    """tworoom_policy_mlp_c2k1_noact_seed0 -> policy_mlp_c2k1_noact; *_inverse_lambda_* -> inverse."""
    family = re.sub(r"_seed\d+$", "", run).split("_", 1)[1]
    return "inverse" if family.startswith("inverse_lambda_") else family


def our_row(r: dict) -> dict | None:
    prefix = r["run"].split("_")[0]
    if prefix not in ENVS or not re.search(r"_seed\d+$", r["run"]):
        return None
    env, label, task_seed, cem_base = ENVS[prefix]
    seed = int(re.search(r"_seed(\d+)$", r["run"]).group(1))
    success = r.get("success") or ""
    try:
        float(success)
        status = "ok"
    except ValueError:
        status = "no_eval" if r.get("verdict") == "PASS" else "incomplete"
        success = ""
    return {
        "env": env, "env_label": label, "method": method_of(r["run"]),
        "seed_or_repeat": seed, "run_label": f"seed_{seed}", "success_rate": success,
        "eval_budget": 50, "goal_offset": 25, "num_eval": 100,
        "eval_task_seed": task_seed, "policy_seed": cem_base + seed, "status": status,
        "run_dir": r.get("run_dir", ""), "metrics_path": r.get("metrics_path", ""),
        "final_training_run": r["run"],
        "source": r["cluster"], "regularizer": r.get("method", ""), "lambda": r.get("lambda", ""),
        "eff_rank": r.get("eff_rank", ""), "verdict": r.get("verdict", ""),
        "other_evals": r.get("other_evals", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", action="append", default=[], help="CLUSTER=summary.tsv (repeatable)")
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    args = ap.parse_args()
    if not args.tsv:
        raise SystemExit("give at least one --tsv CLUSTER=path")

    with args.csv.open(newline="", encoding="utf-8") as f:
        existing = list(csv.DictReader(f))
    upstream = [dict(r, source="upstream") for r in existing if r.get("source", "upstream") in ("", "upstream")]

    ours = [row for row in map(our_row, read_runs(labelled(args.tsv))) if row]
    order = {"tworoom": 0, "reacher": 1, "pusht": 2, "ogbcube": 3}
    ours.sort(key=lambda r: (order[r["env"]], r["method"], r["seed_or_repeat"], r["source"]))

    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=UPSTREAM_COLS + EXTRA_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(upstream)
        w.writerows(ours)
    done = sum(r["status"] == "ok" for r in ours)
    print(f"wrote {args.csv}: {len(upstream)} upstream rows + {len(ours)} study rows ({done} evaluated)")


if __name__ == "__main__":
    main()
