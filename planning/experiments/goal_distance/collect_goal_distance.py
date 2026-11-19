#!/usr/bin/env python3
"""Read-only collector for the hindsight-goal-distance ablation: one CSV row per run.

Rows: every sweep run <dataset>_policy_mlp_c2k1_noact_goal<G>_seed<S> found under the runs root,
and the two settings that bracket the sweep: SMWM (inverse dynamics, the action between two
consecutive frames) and the default policy head (goal = the last frame of the episode).

Per run it reads
  * lightning/local/metrics.csv, the last logged value of each column: the held-out loss of the
    action head (policy loss, or inverse loss for SMWM) and the explained action variance
    1 - loss (actions are standardized, so 0 means a collapsed head), the prediction loss
    (training epoch mean, as in the paper's table, and held-out), the effective rank and the
    per-dimension standard deviation of the latent, the step;
  * the planning success rate of each eval label, from the first job that has it (warm-start
    eval folder first, as in the paper tables).

Usage: python3 collect_goal_distance.py [out.csv]        # default: CSV on stdout
A summary (mean and sample std over the seeds that finished training) goes to stderr.
"""
import argparse
import csv
import getpass
import json
import re
import statistics
import sys
from pathlib import Path

HEAD = "policy_mlp_c2k1_noact"
INVERSE = {"reacher": "reacher_inverse_lambda_1", "reacher_mix": "reacher_mix_inverse"}
LABELS = ["cem30", "cem10", "cem5", "policy", "mean30", "mean10", "mean5"]
COLUMNS = (["dataset", "setting", "goal_frames", "goal_env_steps", "seed", "run", "clip_frames", "trained", "step",
            "val_head_loss", "explained_variance", "train_pred_loss", "val_pred_loss", "effective_rank",
            "mean_per_dim_std"] + LABELS + ["eval_jobs"])


def last_value(rows, column):
    for row in reversed(rows):
        value = row.get(column)
        if value not in ("", None):
            return float(value)
    return None


def config_value(text, key):
    m = re.search(rf"^\s+{re.escape(key)}: (\S+)$", text, re.M)
    return m.group(1) if m else ""


def success_rates(eval_roots, run):
    """{label: (rate, job)} from the first job that has the label; roots in the given order."""
    found = {}
    for root in eval_roots:
        jobs = sorted((p for p in (root / run).glob("job_*") if p.name[4:].isdigit()), key=lambda p: int(p.name[4:]))
        for job in jobs:
            for label in LABELS:
                path = job / label / "metrics.json"
                if label in found or not path.is_file():
                    continue
                try:
                    data = json.loads(path.read_text())
                except ValueError:
                    continue
                rate = data.get("success_rate")
                if rate is None and isinstance(data.get("metrics"), dict):
                    rate = data["metrics"].get("success_rate")
                if rate is not None:
                    found[label] = (float(rate), job.name[4:])
    return found


def collect(runs_root, eval_roots, dataset, setting, goal_frames, seed, run, frameskip_default=5):
    run_dir = runs_root / run
    if not run_dir.is_dir():
        return None
    cfg = (run_dir / "config.yaml").read_text() if (run_dir / "config.yaml").is_file() else ""
    frameskip = int(config_value(cfg, "frameskip") or frameskip_default)
    rows = []
    metrics = run_dir / "lightning" / "local" / "metrics.csv"
    if metrics.is_file():
        with open(metrics, newline="") as f:
            rows = list(csv.DictReader(f))
    head_loss = last_value(rows, "validate/inv_loss" if setting == "smwm" else "validate/policy_loss")
    evals = success_rates(eval_roots, run)
    step = last_value(rows, "step")
    out = {
        "dataset": dataset, "setting": setting, "goal_frames": goal_frames,
        "goal_env_steps": goal_frames * frameskip if isinstance(goal_frames, int) else "",
        "seed": seed, "run": run, "clip_frames": config_value(cfg, "goal_clip_frames"),
        "trained": int((run_dir / "final_embeddings.pt").is_file()),
        "step": "" if step is None else int(step),
        "val_head_loss": head_loss, "explained_variance": None if head_loss is None else 1.0 - head_loss,
        "train_pred_loss": last_value(rows, "fit/pred_loss_epoch"),
        "val_pred_loss": last_value(rows, "validate/pred_loss"),
        "effective_rank": last_value(rows, "validate/effective_rank"),
        "mean_per_dim_std": last_value(rows, "validate/mean_per_dim_std"),
        "eval_jobs": " ".join(sorted({job for _, job in evals.values()}, key=int)),
    }
    out.update({label: evals[label][0] if label in evals else None for label in LABELS})
    return out


def main():
    user = getpass.getuser()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", nargs="?", help="CSV file (default: stdout)")
    ap.add_argument("--runs-root", type=Path, default=Path(f"/scratch/{user}/smwm-runs"))
    ap.add_argument("--eval-roots", type=Path, nargs="+",
                    default=[Path(f"/scratch/{user}/smwm-warmstart-eval/reacher"),
                             Path(f"/scratch/{user}/smwm-speed-bench/reacher")])
    ap.add_argument("--datasets", nargs="+", default=["reacher", "reacher_mix"])
    ap.add_argument("--reference-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    table = []

    def add(dataset, setting, goal, seed, run):
        table.append(collect(args.runs_root, args.eval_roots, dataset, setting, goal, seed, run))

    for dataset in args.datasets:
        for seed in args.reference_seeds:  # the next frame: SMWM
            add(dataset, "smwm", "next", seed, f"{INVERSE[dataset]}_seed{seed}")
        sweep = []
        for run_dir in args.runs_root.glob(f"{dataset}_{HEAD}_goal*_seed*"):
            m = re.fullmatch(rf"{re.escape(dataset)}_{HEAD}_goal(\d+)_seed(\d+)", run_dir.name)
            if m:
                sweep.append((int(m.group(1)), int(m.group(2)), run_dir.name))
        for goal, seed, run in sorted(sweep):
            add(dataset, "goal", goal, seed, run)
        for seed in args.reference_seeds:  # the default head: the last frame of the episode
            add(dataset, "final", "final", seed, f"{dataset}_{HEAD}_goalfinal_seed{seed}")
    table = [row for row in table if row is not None]

    def cell(value):
        return "" if value is None else (f"{value:.6g}" if isinstance(value, float) else value)

    out = open(args.out, "w", newline="") if args.out else sys.stdout
    writer = csv.writer(out)
    writer.writerow(COLUMNS)
    for row in table:
        writer.writerow([cell(row[c]) for c in COLUMNS])
    if args.out:
        out.close()

    # Summary over seeds, finished runs only. A sweep run is finished when train.py wrote
    # final_embeddings.pt; the older reference runs do not all have that file, so they count
    # once they have a planning result.
    def stat(values, scale=1.0, digits=1):
        values = [v * scale for v in values if v is not None]
        if not values:
            return "-"
        spread = f" +- {statistics.stdev(values):.{digits}f}" if len(values) > 1 else ""
        return f"{statistics.fmean(values):.{digits}f}{spread}"

    groups = {}
    for row in table:
        if row["trained"] or (row["setting"] != "goal" and row["cem30"] is not None):
            groups.setdefault((row["dataset"], row["setting"], row["goal_frames"]), []).append(row)
    def line(*cells):
        widths = (12, 6, 5, 7, 14, 10, 14, 8, 12, 12, 12, 12)
        print(" ".join(f"{c:<{w}s}" if i == 0 else f"{c:>{w}s}" for i, (c, w) in enumerate(zip(cells, widths))),
              file=sys.stderr)

    line("dataset", "goal", "seeds", "steps", "expl. var %", "pred loss", "rank", "std/dim",
         "CEM 30", "CEM 5", "head only", "warm 5")
    for (dataset, setting, goal), rows in groups.items():
        steps = sorted({r["step"] for r in rows if r["step"] != ""})
        pred = [r["train_pred_loss"] for r in rows if r["train_pred_loss"] is not None]
        std = [r["mean_per_dim_std"] for r in rows if r["mean_per_dim_std"] is not None]
        line(dataset, str(goal) if setting != "smwm" else "next", str(len(rows)), str(steps[-1]) if steps else "-",
             stat([r["explained_variance"] for r in rows], 100.0), f"{statistics.fmean(pred):.1e}" if pred else "-",
             stat([r["effective_rank"] for r in rows]), f"{statistics.fmean(std):.4f}" if std else "-",
             stat([r["cem30"] for r in rows]), stat([r["cem5"] for r in rows]),
             stat([r["policy"] for r in rows]), stat([r["mean5"] for r in rows]))
    print(f"{len(table)} runs, {sum(len(rows) for rows in groups.values())} of them finished", file=sys.stderr)


if __name__ == "__main__":
    main()
