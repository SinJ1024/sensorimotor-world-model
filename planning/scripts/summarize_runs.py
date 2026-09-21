#!/usr/bin/env python3
"""One table for every training run: parameters verified against the paper,
training-time metrics, and the paper-protocol planning success rate.

For each <runs_root>/<run>/ it reads
  * config.yaml                 -> environment, regularizer(s), lambda, paper hyper-parameters
  * checkpoints/last.ckpt       -> epoch / global_step / LR-schedule state
  * lightning/local/metrics.csv -> final pred / inverse / policy / goal-policy losses, effective rank
and looks up the CEM evaluation result under
  <eval_root>/<env>/<run>/seed_<s>/budget_<b>/job_<id>/results/seed_<s>/metrics.json
(the newest job per run/seed/budget wins).

Usage:
  python summarize_runs.py [--runs-root R] [--eval-root E] [--budget 50] [--out summary.tsv]
Exit code is 1 when any run DEVIATEs, so it can be used as a gate.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

import torch
import yaml

# Paper baseline (config/train/base.yaml of petr-ivashkov/sensorimotor-world-model @9d22bcc).
PAPER = {
    "trainer.precision": "bf16",
    "trainer.max_epochs": 10,
    "trainer.gradient_clip_val": 1.0,
    "optimizer.type": "AdamW",
    "optimizer.lr": 1e-4,
    "optimizer.weight_decay": 1e-3,
    "encoder_scale": "tiny",
    "patch_size": 14,
    "img_size": 224,
    "wm.embed_dim": 192,
    "wm.history_size": 1,
    "predictor.depth": 6,
    "predictor.heads": 16,
    "predictor.mlp_dim": 2048,
    "predictor.dim_head": 64,
    "predictor.dropout": 0.1,
}
# Paper Table 2: inverse-dynamics weight per environment. Policy / goal-policy
# runs reuse the same value so every regularizer is compared at the paper's lambda.
LAMBDA = {"tworoom": 0.1, "reacher": 5.0, "pusht": 30.0, "cube": 1.0}
EVAL_SLUG = {"tworoom": "tworoom", "reacher": "reacher", "pusht": "pusht", "cube": "ogbcube"}


def get(cfg, dotted, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def close(a, b):
    try:
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return a == b


def last_metric(rows, pred):
    if not rows:
        return None
    for key in rows[0]:
        if pred(key):
            vals = [r[key] for r in rows if r.get(key) not in (None, "")]
            if vals:
                return float(vals[-1])
    return None


def lambda_from_name(name):
    """tworoom_policy_mlp_c2k1_lam0p03_seed1 -> 0.03, pusht_..._lam300_seed0 -> 300."""
    m = re.search(r"_lam(\d+)(?:p(\d+))?_", name)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2) or 0}")


def describe_method(cfg):
    """Human label + list of (name, weight) for every active regularizer."""
    inv_w = float(get(cfg, "loss.inverse.weight", 0) or 0)
    pol = get(cfg, "loss.policy", {}) or {}
    pol_w = float(pol.get("weight", 0) or 0)
    goal = get(cfg, "loss.goal_policy", {}) or {}
    goal_w = float(goal.get("weight", 0) or 0)
    sig_w = float(get(cfg, "loss.sigreg.weight", 0) or 0)
    parts, active = [], []
    if inv_w:
        parts.append("inverse")
        active.append(("inverse", inv_w))
    if pol_w:
        parts.append(f"policy({pol.get('arch', 'mlp')},c{pol.get('context', 2)},k{pol.get('num_future', 1)})")
        active.append(("policy", pol_w))
    if goal_w:
        offs = ",".join(str(int(g)) for g in goal.get("goal_offsets", [2]))
        parts.append(f"goal_policy(G=[{offs}],k{goal.get('num_actions', 1)})")
        active.append(("goal_policy", goal_w))
    if sig_w:
        parts.append("sigreg")
        active.append(("sigreg", sig_w))
    return ("+".join(parts) if parts else "forward_only"), active


def find_eval(eval_root, env, run_name, budget):
    """Newest metrics.json per (seed, budget) for this run -> {seed: success_rate}."""
    pattern = os.path.join(eval_root, EVAL_SLUG[env], run_name, "seed_*", f"budget_{budget}", "job_*", "results", "seed_*", "metrics.json")
    best = {}
    for path in glob.glob(pattern):
        m = re.search(r"/seed_(\d+)/budget_\d+/job_(\d+)/", path.replace("\\", "/"))
        if not m:
            continue
        seed, job = int(m.group(1)), int(m.group(2))
        if seed not in best or job > best[seed][0]:
            best[seed] = (job, path)
    out = {}
    for seed, (job, path) in best.items():
        try:
            out[seed] = float(json.load(open(path))["metrics"]["success_rate"])
        except (KeyError, ValueError, json.JSONDecodeError):
            out[seed] = float("nan")
    return out


def summarize(run, eval_root, budget):
    r = {"run": run.name}
    devs = []
    cfg_path = run / "config.yaml"
    if not cfg_path.is_file():
        r["verdict"] = "NO_CONFIG"
        return r
    cfg = yaml.safe_load(cfg_path.read_text())
    env = run.name.split("_")[0]
    if env not in LAMBDA:
        r["verdict"] = "UNKNOWN_ENV"
        return r
    r["env"] = env
    r["method"], active = describe_method(cfg)
    r["lambda"] = "/".join(f"{w:g}" for _, w in active) if active else "0"
    r["bs"] = get(cfg, "loader.batch_size")
    r["num_steps"] = get(cfg, "data.dataset.num_steps")
    r["seed"] = cfg.get("seed")

    # ---- paper hyper-parameters -------------------------------------------
    for path, expected in PAPER.items():
        got = get(cfg, path)
        if not close(got, expected):
            devs.append(f"{path}={got} (paper {expected})")
    if "scheduler" in cfg:
        devs.append("scheduler block present (paper: bare-string auto-derived schedule)")
    ctx = str((get(cfg, "loss.policy", {}) or {}).get("context", ""))
    if r["bs"] != 256 and not (r["bs"] == 64 and ctx == "16"):
        devs.append(f"batch_size={r['bs']} (paper 256; 64 only tolerated for c16)")

    # ---- regularizer weight == paper lambda for this env (or the sweep value) --
    expected_lambda = lambda_from_name(run.name)
    expected_lambda = LAMBDA[env] if expected_lambda is None else expected_lambda
    for name, w in active:
        if name != "sigreg" and not close(w, expected_lambda):
            devs.append(f"{name}.weight={w:g} (expected {expected_lambda:g})")
    if not active:
        devs.append("no regularizer active (forward-only)")

    # ---- checkpoint / schedule --------------------------------------------
    ck = run / "checkpoints" / "last.ckpt"
    if not ck.is_file():
        devs.append("NO_CHECKPOINT")
    else:
        d = torch.load(ck, map_location="cpu", weights_only=False)
        s = (d.get("lr_schedulers") or [{}])[0]
        ep, gs = d.get("epoch"), d.get("global_step")
        ms, wu = s.get("max_steps"), s.get("warmup_steps")
        lr = s.get("_last_lr")
        lr = lr[0] if isinstance(lr, list) and lr else lr
        r.update(epoch=ep, step=gs, max_steps=ms)
        r["last_lr"] = f"{lr:.1e}" if isinstance(lr, (int, float)) else lr
        if ep != 9:
            devs.append(f"epoch={ep} (expected 9)")
        if ms is None or ms < 1000:
            devs.append(f"max_steps={ms} (SCHEDULER BUG)")
        elif gs != ms:
            devs.append(f"global_step {gs} != max_steps {ms}")
        if ms and wu is not None and abs(wu - int(0.01 * ms)) > 2:
            devs.append(f"warmup={wu} (expected ~{int(0.01 * ms)})")
        if isinstance(lr, (int, float)) and lr > 1e-6:
            devs.append(f"last_lr={lr:.2e} (not annealed)")
    r["emb"] = (run / "final_embeddings.pt").is_file()

    # ---- training-time metrics --------------------------------------------
    mc = run / "lightning" / "local" / "metrics.csv"
    if mc.is_file():
        with open(mc, newline="") as f:
            rows = list(csv.DictReader(f))
        r["pred_loss"] = last_metric(rows, lambda k: k == "fit/pred_loss_epoch")
        r["inv_loss"] = last_metric(rows, lambda k: k == "fit/inv_loss_epoch")
        r["policy_loss"] = last_metric(rows, lambda k: k == "fit/policy_loss_epoch")
        r["goal_loss"] = last_metric(rows, lambda k: k == "fit/goal_policy_loss_epoch")
        r["eff_rank"] = last_metric(rows, lambda k: "effective_rank" in k)
    else:
        devs.append("metrics.csv missing")

    # ---- planning evaluation ----------------------------------------------
    ev = find_eval(eval_root, env, run.name, budget)
    r["success"] = "/".join(f"{ev[s]:g}" for s in sorted(ev)) if ev else ""
    r["n_eval"] = len(ev)

    r["verdict"] = "PASS" if not devs else "DEVIATE"
    r["deviations"] = "; ".join(devs)
    return r


COLS = ["env", "run", "method", "lambda", "bs", "num_steps", "seed", "epoch", "step", "max_steps",
        "last_lr", "emb", "eff_rank", "pred_loss", "inv_loss", "policy_loss", "goal_loss",
        "success", "n_eval", "verdict", "deviations"]
SHOW = ["env", "run", "method", "lambda", "step", "max_steps", "eff_rank", "pred_loss",
        "inv_loss", "policy_loss", "goal_loss", "success", "verdict"]


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "Y" if v else "N"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def main():
    user = os.environ.get("USER", "")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=f"/scratch/{user}/smwm-runs")
    ap.add_argument("--eval-root", default=f"/scratch/{user}/smwm-paper-eval")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--out", default=None, help="TSV path (default <runs_root>/_summary/summary.tsv)")
    args = ap.parse_args()

    root = Path(args.runs_root)
    runs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    results = [summarize(run, args.eval_root, args.budget) for run in runs]
    order = {"tworoom": 0, "reacher": 1, "pusht": 2, "cube": 3}
    results.sort(key=lambda r: (order.get(r.get("env"), 9), r.get("method", ""), r["run"]))

    width = {c: max(len(c), *(len(fmt(r.get(c))) for r in results)) for c in SHOW}
    print("  ".join(c.ljust(width[c]) for c in SHOW))
    for r in results:
        print("  ".join(fmt(r.get(c)).ljust(width[c]) for c in SHOW))
    print()
    for r in results:
        if r.get("deviations"):
            print(f"[{r['verdict']}] {r['run']}: {r['deviations']}")

    out = Path(args.out) if args.out else root / "_summary" / "summary.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({c: fmt(r.get(c)) for c in COLS})
    n_pass = sum(r.get("verdict") == "PASS" for r in results)
    n_eval = sum(1 for r in results if r.get("n_eval"))
    print(f"\n{n_pass}/{len(results)} runs PASS; {n_eval}/{len(results)} have a budget-{args.budget} evaluation")
    print(f"table: {out}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
