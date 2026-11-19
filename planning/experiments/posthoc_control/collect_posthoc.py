#!/usr/bin/env python3
"""Read-only collector for the post-hoc policy head control: one CSV row per run.

For every (dataset, source model, seed) three rows:
  source      the finished SMWM (inverse) or LeWM (sigreg) run: plain CEM only, it has no head that proposes actions
  post-hoc    <dataset>_<model>_posthoc_goalfinal_seed<S>: the paper's policy head trained on the frozen latents
              of the source run (same world model); head alone and as a CEM warm-start
  joint       <dataset>_policy_mlp_c2k1_noact_goalfinal_seed<S>: the same head trained jointly as the regularizer
              (the method of the paper), listed once per dataset and seed

The held-out policy loss is on the same clips for the post-hoc and the joint head (first 10 held-out batches);
explained variance = 1 - that loss (actions are standardized). Success rates are read like collect_head_arch.py
does: the first job that has the label, warm-start eval folder first.

Usage: python3 collect_posthoc.py [out.csv] [--datasets pusht cube] [--models inverse sigreg] [--seeds 0]
A summary table goes to stderr.
"""
import argparse
import csv
import getpass
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "head_arch"))
import collect_head_arch as cha  # noqa: E402

SOURCE = {"inverse": dict(cha.INVERSE), "sigreg": {d: f"{d}_sigreg" for d in cha.INVERSE}}   # run names without _seed<S>
NAME = {"inverse": "SMWM", "sigreg": "LeWM"}
COLUMNS = ["dataset", "seed", "method", "run", "finished", "val_policy_loss", "explained_variance"] + cha.LABELS + ["eval_jobs"]


def row(runs_root, roots, dataset, seed, method, run, head):
    r = cha.collect(runs_root, roots, dataset, head, seed, run)
    if r is None:
        return {"dataset": dataset, "seed": seed, "method": method, "run": run, "finished": 0}
    out = {"dataset": dataset, "seed": seed, "method": method, "run": run, "finished": 1,
           "val_policy_loss": r["val_head_loss"] if head == "mlp" else None, "eval_jobs": r["eval_jobs"]}
    out.update({label: r[label] for label in cha.LABELS})
    done = runs_root / run / "posthoc_done.json"
    if method.endswith("post-hoc head"):
        out["finished"] = int(done.is_file())
        if done.is_file():
            out["val_policy_loss"] = json.loads(done.read_text())["val_policy_loss"]
    if out["val_policy_loss"] is not None:
        out["explained_variance"] = 1.0 - out["val_policy_loss"]
    return out


def main():
    user = getpass.getuser()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", nargs="?", help="CSV file (default: stdout)")
    ap.add_argument("--runs-root", type=Path, default=Path(f"/scratch/{user}/smwm-runs"))
    ap.add_argument("--eval-roots", type=Path, nargs="+",
                    default=[Path(f"/scratch/{user}/smwm-warmstart-eval"), Path(f"/scratch/{user}/smwm-speed-bench")])
    ap.add_argument("--datasets", nargs="+", default=["pusht", "cube"], choices=list(cha.INVERSE))
    ap.add_argument("--models", nargs="+", default=["inverse", "sigreg"], choices=list(SOURCE))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    table = []
    for dataset in args.datasets:
        roots = [root / cha.EVAL_ENV[dataset] for root in args.eval_roots]
        for seed in args.seeds:
            for model in args.models:
                table.append(row(args.runs_root, roots, dataset, seed, NAME[model],
                                 f"{SOURCE[model][dataset]}_seed{seed}", "inverse" if model == "inverse" else "sigreg"))
                table.append(row(args.runs_root, roots, dataset, seed, f"{NAME[model]} + post-hoc head",
                                 f"{dataset}_{model}_posthoc_goalfinal_seed{seed}", "mlp"))
            table.append(row(args.runs_root, roots, dataset, seed, "joint head (ours)",
                             f"{dataset}_policy_mlp_{cha.HEAD}_seed{seed}", "mlp"))

    def cell(value):
        return "" if value is None else (f"{value:.6g}" if isinstance(value, float) else value)

    out = open(args.out, "w", newline="") if args.out else sys.stdout
    writer = csv.writer(out)
    writer.writerow(COLUMNS)
    for r in table:
        writer.writerow([cell(r.get(c)) for c in COLUMNS])
    if args.out:
        out.close()

    def num(value, scale=1.0):
        return "-" if value is None else f"{value * scale:.1f}"

    widths = (12, 4, 22, 11, 7, 7, 7, 9, 8, 8, 8)
    def line(*cells):
        print(" ".join(f"{c:<{w}s}" if i in (0, 2) else f"{c:>{w}s}" for i, (c, w) in enumerate(zip(cells, widths))), file=sys.stderr)

    line("dataset", "seed", "method", "expl. var %", "CEM 30", "CEM 10", "CEM 5", "head only", "warm 30", "warm 10", "warm 5")
    for r in table:
        line(r["dataset"], str(r["seed"]), r["method"] + ("" if r["finished"] else " (missing)"),
             num(r.get("explained_variance"), 100.0), *(num(r.get(label)) for label in cha.LABELS))


if __name__ == "__main__":
    main()
