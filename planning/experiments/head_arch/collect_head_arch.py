#!/usr/bin/env python3
"""Read-only collector for the head-architecture sweep: one CSV row per run.

Rows: every run <dataset>_policy_<arch>_c2k1_noact_goalfinal[_lam<x>]_seed<S> under the runs root
(the goal-conditioned policy head without action input, as MLP, Transformer, Mamba or GRU, at any
weight; the MLP at the default weight is the head of the paper), and SMWM (inverse dynamics) on the
same dataset as the reference.

Per run it reads
  * config.yaml: the head architecture and the regularizer weight;
  * lightning/local/metrics.csv, the last logged value of each column: the held-out and training loss
    of the action head (policy loss, or inverse loss for SMWM), the explained action variance
    1 - held-out loss (actions are standardized, so 0 means a head that predicts the mean), the
    prediction loss (training epoch mean and held-out), the logged effective rank and per-dimension
    standard deviation of the latent (computed in the bfloat16 arithmetic of training);
  * final_embeddings.pt (only with torch and numpy, i.e. the training environment's python): the same
    two statistics in single precision on the clips that the log uses, and the held-out prediction
    loss relative to the latent variance. With a small latent scale the logged effective rank is
    rounding noise; use the single-precision one;
  * the planning success rate of each eval label, from the first job that has it (warm-start eval
    folder first, as in the paper tables).

Usage: python3 collect_head_arch.py [out.csv] [--datasets tworoom pusht ...]     # default: CSV on stdout
       /scratch/$USER/envs/smwm-eval-py312-torch251-cu121/bin/python collect_head_arch.py out.csv   # with fp32 columns
A summary (mean and sample std over the seeds that finished training) goes to stderr.
"""
import argparse
import csv
import getpass
import json
import math
import re
import statistics
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:  # the single-precision columns stay empty
    np = torch = None

HEAD = "c2k1_noact_goalfinal"
ARCHS = ("mlp", "transformer", "mamba", "gru")
INVERSE = {"tworoom": "tworoom_inverse_lambda_0p1", "pusht": "pusht_inverse_lambda_30", "cube": "cube_inverse_lambda_1",
           "reacher": "reacher_inverse_lambda_1", "reacher_mix": "reacher_mix_inverse"}
EVAL_ENV = {"tworoom": "tworoom", "pusht": "pusht", "cube": "ogbcube", "reacher": "reacher", "reacher_mix": "reacher"}
LABELS = ["cem30", "cem10", "cem5", "policy", "mean30", "mean10", "mean5"]
COLUMNS = (["dataset", "head", "weight", "seed", "run", "trained", "step", "val_head_loss", "explained_variance",
            "train_head_loss", "train_pred_loss", "val_pred_loss", "effective_rank_logged", "std_logged",
            "effective_rank_fp32", "std_fp32", "rel_pred_error_pct"] + LABELS + ["eval_jobs"])


def last_value(rows, column):
    for row in reversed(rows):
        value = row.get(column)
        if value not in ("", None):
            return float(value)
    return None


def section(text, name, indent=""):
    """The lines below '<indent><name>:' that are indented more than it."""
    m = re.search(rf"^{indent}{re.escape(name)}:\n((?:{indent} +.*\n)+)", text, re.M)
    return m.group(1) if m else ""


def config_value(text, path):
    """A value of a saved config by its dotted path (two-space indentation), as a string; '' if absent."""
    *parents, key = path.split(".")
    indent = ""
    for name in parents:
        text = section(text, name, indent)
        indent += "  "
    m = re.search(rf"^{indent}{re.escape(key)}: (\S+)$", text, re.M)
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


def fp32_stats(run_dir, cfg):
    """(per-dimension std, mean variance, effective rank) of the saved final latents in single precision, on the
    clips of the logged statistics (limit_val_batches x batch size), frames paired as in utils._flatten_time_pairs."""
    path = run_dir / "final_embeddings.pt"
    if torch is None or not path.is_file():
        return None
    try:
        emb = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["emb"]
        n = int(config_value(cfg, "trainer.limit_val_batches") or 10) * int(config_value(cfg, "loader.batch_size") or 256)
        emb = emb[:n].clone().double().numpy()
    except Exception as e:  # noqa: BLE001
        print(f"{run_dir.name}: final_embeddings.pt not read ({type(e).__name__}: {e})", file=sys.stderr)
        return None
    x = np.concatenate([emb[:, :-1], emb[:, 1:]], axis=1).reshape(-1, emb.shape[-1])
    xc = x - x.mean(0, keepdims=True)
    sv = np.sqrt(np.clip(np.linalg.eigvalsh(xc.T @ xc), 0, None))
    p = sv / sv.sum()
    p = p[p > 0]
    return float(x.std(0).mean()), float(x.var(0).mean()), math.exp(-float((p * np.log(p)).sum()))


def collect(runs_root, eval_roots, dataset, head, seed, run):
    run_dir = runs_root / run
    if not run_dir.is_dir():
        return None
    cfg = (run_dir / "config.yaml").read_text() if (run_dir / "config.yaml").is_file() else ""
    rows = []
    metrics = run_dir / "lightning" / "local" / "metrics.csv"
    if metrics.is_file():
        with open(metrics, newline="") as f:
            rows = list(csv.DictReader(f))
    key = "inv_loss" if head == "inverse" else "policy_loss"
    head_loss = last_value(rows, f"validate/{key}")
    val_pred = last_value(rows, "validate/pred_loss")
    evals = success_rates(eval_roots, run)
    step = last_value(rows, "step")
    stats = fp32_stats(run_dir, cfg)
    out = {
        "dataset": dataset, "head": head, "seed": seed, "run": run,
        "weight": config_value(cfg, "loss.inverse.weight" if head == "inverse" else "loss.policy.weight"),
        "trained": int((run_dir / "final_embeddings.pt").is_file()),
        "step": "" if step is None else int(step),
        "val_head_loss": head_loss, "explained_variance": None if head_loss is None else 1.0 - head_loss,
        "train_head_loss": last_value(rows, f"fit/{key}_epoch"),
        "train_pred_loss": last_value(rows, "fit/pred_loss_epoch"), "val_pred_loss": val_pred,
        "effective_rank_logged": last_value(rows, "validate/effective_rank"),
        "std_logged": last_value(rows, "validate/mean_per_dim_std"),
        "std_fp32": stats[0] if stats else None, "effective_rank_fp32": stats[2] if stats else None,
        "rel_pred_error_pct": 100.0 * val_pred / stats[1] if stats and val_pred is not None and stats[1] > 0 else None,
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
                    default=[Path(f"/scratch/{user}/smwm-warmstart-eval"), Path(f"/scratch/{user}/smwm-speed-bench")],
                    help="each holds <eval env>/<run>/job_<id>/<label>/metrics.json")
    ap.add_argument("--datasets", nargs="+", default=list(INVERSE), choices=list(INVERSE))
    ap.add_argument("--reference-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    table = []
    for dataset in args.datasets:
        roots = [root / EVAL_ENV[dataset] for root in args.eval_roots]
        sweep = []
        for run_dir in args.runs_root.glob(f"{dataset}_policy_*_{HEAD}*_seed*"):
            m = re.fullmatch(rf"{re.escape(dataset)}_policy_({'|'.join(ARCHS)})_{HEAD}(?:_lam[0-9p]+)?_seed(\d+)", run_dir.name)
            if m:
                sweep.append((m.group(1), int(m.group(2)), run_dir.name))
        if not sweep:
            continue
        for seed in args.reference_seeds:
            table.append(collect(args.runs_root, roots, dataset, "inverse", seed, f"{INVERSE[dataset]}_seed{seed}"))
        for head, seed, run in sorted(sweep, key=lambda t: (ARCHS.index(t[0]), t[2])):
            table.append(collect(args.runs_root, roots, dataset, head, seed, run))
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

    # Summary over seeds. A sweep run counts once train.py wrote final_embeddings.pt; the older
    # reference runs do not all have that file, so they count once they have a planning result.
    def stat(values, scale=1.0, digits=1):
        values = [v * scale for v in values if v is not None]
        if not values:
            return "-"
        spread = f" +- {statistics.stdev(values):.{digits}f}" if len(values) > 1 else ""
        return f"{statistics.fmean(values):.{digits}f}{spread}"

    def sci(values):
        values = [v for v in values if v is not None]
        return f"{statistics.fmean(values):.1e}" if values else "-"

    groups = {}
    for row in table:
        if row["trained"] or (row["head"] == "inverse" and row["cem30"] is not None):
            groups.setdefault((row["dataset"], row["head"], row["weight"]), []).append(row)

    def line(*cells):
        widths = (12, 12, 7, 5, 12, 10, 9, 13, 13, 12, 12, 12, 12, 12)
        print(" ".join(f"{c:<{w}s}" if i < 2 else f"{c:>{w}s}" for i, (c, w) in enumerate(zip(cells, widths))), file=sys.stderr)

    line("dataset", "head", "weight", "seeds", "expl. var %", "pred loss", "std/dim", "rank (fp32)", "rank (logged)",
         "CEM 30", "CEM 5", "head only", "warm 30", "warm 5")
    order = {"inverse": -1, **{a: i for i, a in enumerate(ARCHS)}}
    for (dataset, head, weight), rows in sorted(groups.items(), key=lambda kv: (args.datasets.index(kv[0][0]), float(kv[0][2] or 0)
                                                                                 if kv[0][1] != "inverse" else -1.0, order[kv[0][1]])):
        std = [r["std_fp32"] if r["std_fp32"] is not None else r["std_logged"] for r in rows]
        line(dataset, head, str(weight), str(len(rows)), stat([r["explained_variance"] for r in rows], 100.0),
             sci([r["train_pred_loss"] for r in rows]), sci(std), stat([r["effective_rank_fp32"] for r in rows]),
             stat([r["effective_rank_logged"] for r in rows]), stat([r["cem30"] for r in rows]), stat([r["cem5"] for r in rows]),
             stat([r["policy"] for r in rows]), stat([r["mean30"] for r in rows]), stat([r["mean5"] for r in rows]))
    if torch is None:
        print("torch / numpy not found: the single-precision columns are empty (std/dim is the logged value); "
              "run with the training environment's python to fill them", file=sys.stderr)
    print(f"{len(table)} runs, {sum(len(rows) for rows in groups.values())} of them finished", file=sys.stderr)


if __name__ == "__main__":
    main()
