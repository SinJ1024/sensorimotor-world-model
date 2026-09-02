#!/usr/bin/env python3
"""Summarize and visualize policy-regularizer architecture experiments.

The script scans Lightning ``metrics.csv`` files below a runs directory and
produces:

* ``all_runs.csv``: one row per run, with final and best validation metrics;
* ``all_runs.md``: a Markdown table suitable for reports;
* ``architecture_k_heatmaps.png``: architecture x future-horizon heatmaps;
* ``k_sweep.png``: metric versus number of predicted future actions;
* ``context_sweep.png``: metric versus context length for k=1;
* ``training_curves.png``: validation metric trajectories for selected runs;
* ``summary.md``: automatically generated interpretation and artifact index.

Example (DelftBlue):

    python analyze_results.py \
      --runs-root /scratch/$USER/smwm-runs \
      --output-dir ~/smwm-report/all-experiments
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FINAL_METRICS = (
    "validate/effective_rank",
    "validate/policy_loss",
    "validate/inv_loss",
    "validate/mean_std",
    "validate/mean_per_dim_std",
    "validate/mean_norm",
    "validate/pred_loss",
    "validate/linear_r2_action_coord",
    "validate/linear_r2_agent_velocity",
    "validate/linear_r2_block_angle",
    "validate/linear_r2_block_coord",
)

ARCH_ORDER = ("mlp", "resmlp", "gru", "transformer", "mamba")
COLORS = {
    "mlp": "#4C78A8",
    "resmlp": "#72B7B2",
    "gru": "#54A24B",
    "transformer": "#F58518",
    "mamba": "#B279A2",
    "inverse": "#9D755D",
    "policy": "#E45756",
}


def parse_run_name(name: str) -> dict[str, object]:
    """Infer experiment metadata from the repository's run-name convention."""
    env = next((x for x in ("tworoom", "pusht", "reacher", "cube", "ogbcube")
                if name.startswith(x)), "unknown")
    seed_match = re.search(r"seed_?(\d+)", name)
    arch_match = re.search(r"policy_(mlp|resmlp|gru|transformer|mamba)(?:_|$)", name)
    context_match = re.search(r"_c(\d+)k", name)
    future_match = re.search(r"k(\d+)(?:_|$)", name)

    if "inverse" in name:
        method, arch = "inverse", "inverse"
    elif "policy" in name:
        method = "policy"
        arch = arch_match.group(1) if arch_match else "mlp"
    elif "sigreg" in name:
        method, arch = "sigreg", "sigreg"
    elif "forward_only" in name:
        method, arch = "forward_only", "forward_only"
    else:
        method, arch = "other", "other"

    # Original policy runs predate explicit c/k naming and are c2k1.
    context = int(context_match.group(1)) if context_match else (2 if method == "policy" else None)
    num_future = int(future_match.group(1)) if future_match else (1 if method == "policy" else None)
    return {
        "run": name,
        "environment": env,
        "method": method,
        "arch": arch,
        "context": context,
        "num_future": num_future,
        "seed": int(seed_match.group(1)) if seed_match else 0,
    }


def last_value(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return math.nan
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else math.nan


def best_value(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return math.nan
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else math.nan


def scan_runs(runs_root: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    histories: dict[str, pd.DataFrame] = {}
    for path in sorted(runs_root.glob("*/lightning/local/metrics.csv")):
        run_dir = path.parents[2]
        name = run_dir.name
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[skip] {path}: {exc}")
            continue
        if df.empty:
            continue

        row = parse_run_name(name)
        epochs = pd.to_numeric(df.get("epoch", pd.Series(dtype=float)), errors="coerce").dropna()
        steps = pd.to_numeric(df.get("step", pd.Series(dtype=float)), errors="coerce").dropna()
        row.update({
            "max_epoch": int(epochs.max()) if not epochs.empty else math.nan,
            "max_step": int(steps.max()) if not steps.empty else math.nan,
            "complete": (run_dir / "final_embeddings.pt").is_file(),
            "checkpoint": (run_dir / "checkpoints" / "last.ckpt").is_file(),
            "run_dir": str(run_dir),
        })
        for metric in FINAL_METRICS:
            short = metric.removeprefix("validate/")
            row[short] = last_value(df, metric)
            row[f"best_{short}"] = best_value(df, metric)
        rows.append(row)
        histories[name] = df

    if not rows:
        raise FileNotFoundError(f"No */lightning/local/metrics.csv files under {runs_root}")
    out = pd.DataFrame(rows)
    out = out.sort_values(["environment", "method", "effective_rank"], ascending=[True, True, False])
    return out, histories


def annotate_heatmap(ax: plt.Axes, data: pd.DataFrame, title: str, fmt: str = ".1f") -> None:
    arr = data.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(data.columns)), [f"k={int(x)}" for x in data.columns])
    ax.set_yticks(range(len(data.index)), data.index)
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                threshold = np.nanmedian(arr)
                color = "white" if arr[i, j] < threshold else "black"
                ax.text(j, i, format(arr[i, j], fmt), ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def policy_grid(summary: pd.DataFrame, environment: str, context: int) -> pd.DataFrame:
    return summary[
        (summary.environment == environment)
        & (summary.method == "policy")
        & (summary.context == context)
        & summary.arch.isin(ARCH_ORDER)
    ].copy()


def plot_heatmaps(summary: pd.DataFrame, out_dir: Path, environment: str, context: int) -> None:
    data = policy_grid(summary, environment, context)
    if data.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    specs = (
        ("effective_rank", "Effective rank (higher is better)"),
        ("policy_loss", "Policy loss (lower fits actions better)"),
        ("pred_loss", "Forward prediction loss (scale-dependent)"),
    )
    for ax, (metric, title) in zip(axes, specs):
        pivot = data.pivot_table(index="arch", columns="num_future", values=metric, aggfunc="mean")
        pivot = pivot.reindex([a for a in ARCH_ORDER if a in pivot.index])
        annotate_heatmap(ax, pivot, title)
        ax.set_xlabel("predicted future actions")
        ax.set_ylabel("policy head")
    fig.suptitle(f"{environment}: architecture × horizon (context={context})", fontsize=14)
    fig.savefig(out_dir / "architecture_k_heatmaps.png", dpi=180)
    plt.close(fig)


def plot_k_sweep(summary: pd.DataFrame, out_dir: Path, environment: str, context: int) -> None:
    data = policy_grid(summary, environment, context)
    if data.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for arch in ARCH_ORDER:
        d = data[data.arch == arch].sort_values("num_future")
        if d.empty:
            continue
        color = COLORS[arch]
        axes[0].plot(d.num_future, d.effective_rank, "o-", label=arch, color=color, linewidth=2)
        axes[1].plot(d.num_future, d.policy_loss, "o-", label=arch, color=color, linewidth=2)
    axes[0].set_title("Representation rank vs prediction horizon")
    axes[0].set_ylabel("effective rank ↑")
    axes[1].set_title("Action prediction vs prediction horizon")
    axes[1].set_ylabel("policy loss ↓")
    for ax in axes:
        ax.set_xlabel("number of future actions (k)")
        ax.set_xticks(sorted(data.num_future.dropna().unique().astype(int)))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(f"{environment}: k sweep at context={context}", fontsize=14)
    fig.savefig(out_dir / "k_sweep.png", dpi=180)
    plt.close(fig)


def plot_context_sweep(summary: pd.DataFrame, out_dir: Path, environment: str) -> None:
    data = summary[
        (summary.environment == environment)
        & (summary.method == "policy")
        & (summary.num_future == 1)
        & summary.arch.isin(ARCH_ORDER)
    ].copy()
    if data.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for arch in ARCH_ORDER:
        d = data[data.arch == arch].sort_values("context")
        if d.empty:
            continue
        axes[0].plot(d.context, d.effective_rank, "o-", label=arch, color=COLORS[arch], linewidth=2)
        axes[1].plot(d.context, d.policy_loss, "o-", label=arch, color=COLORS[arch], linewidth=2)
    axes[0].set_title("Representation rank vs context")
    axes[0].set_ylabel("effective rank ↑")
    axes[1].set_title("Action prediction vs context")
    axes[1].set_ylabel("policy loss ↓")
    for ax in axes:
        ax.set_xlabel("context length (frames)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(f"{environment}: context sweep at k=1", fontsize=14)
    fig.savefig(out_dir / "context_sweep.png", dpi=180)
    plt.close(fig)


def validation_series(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric not in df or "step" not in df:
        return np.array([]), np.array([])
    d = df[["step", metric]].copy()
    d["step"] = pd.to_numeric(d.step, errors="coerce")
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna()
    return d.step.to_numpy(), d[metric].to_numpy()


def plot_training_curves(summary: pd.DataFrame, histories: dict[str, pd.DataFrame], out_dir: Path,
                         environment: str, context: int) -> None:
    selected = policy_grid(summary, environment, context)
    if selected.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for _, row in selected.iterrows():
        name = str(row.run)
        label = f"{row.arch} k={int(row.num_future)}"
        color = COLORS.get(str(row.arch), None)
        x, y = validation_series(histories[name], "validate/effective_rank")
        if len(x):
            axes[0].plot(x, y, label=label, color=color, alpha=0.8)
        x, y = validation_series(histories[name], "validate/policy_loss")
        if len(x):
            axes[1].plot(x, y, label=label, color=color, alpha=0.8)
    axes[0].set_title("Effective rank during training")
    axes[0].set_ylabel("effective rank ↑")
    axes[1].set_title("Policy loss during training")
    axes[1].set_ylabel("policy loss ↓")
    for ax in axes:
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(f"{environment}: training trajectories (context={context})", fontsize=14)
    fig.savefig(out_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def markdown_table(summary: pd.DataFrame) -> str:
    columns = [
        "run", "environment", "method", "arch", "context", "num_future",
        "seed", "max_epoch", "complete", "effective_rank", "policy_loss",
        "inv_loss", "mean_std", "pred_loss", "linear_r2_action_coord",
        "linear_r2_agent_velocity", "linear_r2_block_angle", "linear_r2_block_coord",
    ]
    table = summary[[c for c in columns if c in summary]].copy()
    numeric = table.select_dtypes(include=[np.number]).columns
    table[numeric] = table[numeric].round(3)
    try:
        return table.to_markdown(index=False)
    except ImportError:
        # Avoid making tabulate a hard dependency.
        header = "| " + " | ".join(table.columns) + " |"
        sep = "|" + "|".join(["---"] * len(table.columns)) + "|"
        body = ["| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False, name=None)]
        return "\n".join([header, sep, *body])


def write_summary(summary: pd.DataFrame, out_dir: Path, environment: str, context: int) -> None:
    data = policy_grid(summary, environment, context)
    best = data.dropna(subset=["effective_rank"]).sort_values("effective_rank", ascending=False)
    lines = [
        "# Policy-Regularizer Experiment Summary",
        "",
        f"Runs root: `{summary.run_dir.iloc[0] if len(summary) else ''}`",
        "",
    ]
    if not best.empty:
        winner = best.iloc[0]
        lines += [
            "## Main result",
            "",
            f"For **{environment}** at context **c={context}**, the highest final effective rank is "
            f"**{winner.effective_rank:.1f}**, obtained by **{winner.arch}, k={int(winner.num_future)}**.",
            "",
        ]
    lines += [
        "## Interpretation guide",
        "",
        "- **Effective rank (higher):** scale-invariant proxy for representation diversity / non-collapse.",
        "- **Policy loss (lower):** future-action prediction fit; lower loss alone does not guarantee a better representation.",
        "- **Mean std and prediction loss:** scale-dependent; compare cautiously across runs.",
        "- Treat incomplete runs (`complete=False`) as partial evidence.",
        "",
        "## Generated artifacts",
        "",
        "- `all_runs.csv` — machine-readable results.",
        "- `all_runs.md` — report-ready table.",
        "- `architecture_k_heatmaps.png` — architecture × k comparison.",
        "- `k_sweep.png` — effect of predicting more future actions.",
        "- `context_sweep.png` — effect of longer history at k=1.",
        "- `training_curves.png` — validation trajectories over optimizer steps.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path(os.environ.get("RUNS_ROOT", "results")))
    parser.add_argument("--output-dir", type=Path, default=Path("policy_architecture_report"))
    parser.add_argument("--environment", default="tworoom")
    parser.add_argument("--context", type=int, default=16)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, histories = scan_runs(args.runs_root.expanduser().resolve())
    summary.to_csv(args.output_dir / "all_runs.csv", index=False)
    (args.output_dir / "all_runs.md").write_text(markdown_table(summary), encoding="utf-8")
    plot_heatmaps(summary, args.output_dir, args.environment, args.context)
    plot_k_sweep(summary, args.output_dir, args.environment, args.context)
    plot_context_sweep(summary, args.output_dir, args.environment)
    plot_training_curves(summary, histories, args.output_dir, args.environment, args.context)
    write_summary(summary, args.output_dir, args.environment, args.context)

    shown = summary[["run", "max_epoch", "complete", "effective_rank", "policy_loss", "mean_std", "pred_loss"]]
    print(shown.sort_values("effective_rank", ascending=False).to_string(index=False))
    print(f"\nWrote report artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

