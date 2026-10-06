#!/usr/bin/env python3
"""Aggregate final planning-eval metrics into one CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_success_rate(metrics: dict[str, Any]) -> float:
    """Read the sole planning metric reported in Figure 5."""
    if "success_rate" not in metrics:
        raise KeyError("metrics.json does not contain success_rate")
    try:
        return float(metrics["success_rate"])
    except (TypeError, ValueError) as exc:
        raise ValueError("success_rate must be numeric") from exc


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def aggregate(exp_dir: Path) -> list[dict[str, str]]:
    manifest_path = exp_dir / "generated_configs" / "manifest.tsv"
    rows = load_manifest(manifest_path)

    out_rows: list[dict[str, str]] = []
    for row in rows:
        eval_task_seed = row.get("eval_task_seed", row.get("eval_seed", ""))
        policy_seed = row.get("policy_seed", "")
        run_dir = exp_dir / row["result_dir"]
        metrics_path = run_dir / "metrics.json"
        status = "missing"
        result: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        if metrics_path.is_file():
            try:
                result = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics = result.get("metrics", {})
                success_rate = parse_success_rate(metrics)
                status = "ok"
            except json.JSONDecodeError:
                status = "invalid_json"
                success_rate = ""
            except (KeyError, TypeError, ValueError):
                status = "invalid_metric"
                success_rate = ""
        else:
            success_rate = ""

        out_rows.append(
            {
                "env": row["env"],
                "env_label": row["env_label"],
                "method": row["method"],
                "seed_or_repeat": row["seed_or_repeat"],
                "run_label": row["run_label"],
                "success_rate": str(success_rate),
                "eval_budget": row["eval_budget"],
                "goal_offset": row["goal_offset"],
                "num_eval": row["num_eval"],
                "eval_task_seed": eval_task_seed,
                "policy_seed": policy_seed,
                "status": status,
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
                "final_training_run": row["final_training_run"],
                "source": "upstream",
            }
        )
    return out_rows


def study_rows(out_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Rows of the existing CSV that this script did not produce (source != upstream),
    i.e. the policy-regularizer study merged in by scripts/sync_aggregated_results.py,
    and the extra columns they use. Rewriting the CSV must not drop them."""
    if not out_path.is_file():
        return [], []
    with out_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        keep = [r for r in reader if r.get("source", "upstream") not in ("", "upstream")]
        return keep, list(reader.fieldnames or [])


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    keep, existing_fields = study_rows(out_path)
    fieldnames = [
        "env",
        "env_label",
        "method",
        "seed_or_repeat",
        "run_label",
        "success_rate",
        "eval_budget",
        "goal_offset",
        "num_eval",
        "eval_task_seed",
        "policy_seed",
        "status",
        "run_dir",
        "metrics_path",
        "final_training_run",
        "source",
    ]
    fieldnames += [c for c in existing_fields if c not in fieldnames]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerows(keep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <experiment-dir>/aggregated_results.csv",
    )
    args = parser.parse_args()

    exp_dir = args.experiment_dir.resolve()
    output = args.output or exp_dir / "aggregated_results.csv"
    rows = aggregate(exp_dir)
    write_csv(rows, output)

    done = sum(row["status"] == "ok" for row in rows)
    print(f"Wrote {len(rows)} rows to {output}")
    print(f"Completed metrics: {done}/{len(rows)}")


if __name__ == "__main__":
    main()
