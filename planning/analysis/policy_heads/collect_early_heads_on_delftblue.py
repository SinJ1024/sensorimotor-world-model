"""Read-only collector for the earliest policy-head runs (September / October 2026): the MLP / Transformer /
Mamba heads with different context lengths and numbers of predicted actions, the one-state head, the
inverse + policy runs, the lambda sweeps, and the inverse and no-action runs used as references.

Run on DelftBlue with the eval env's python, script on stdin; it writes nothing there:
    ssh delftblue '/scratch/$USER/envs/smwm-eval-py312-torch251-cu121/bin/python -' \
        < collect_early_heads_on_delftblue.py > early_heads_raw.txt

Output: four CSV blocks
  #RUNS    one row per run: config (head architecture, context, k, action input, weights, batch size, ...),
           the last value of every fit/*_epoch and validate/* column of lightning/local/metrics.csv,
           single-precision statistics of the saved final latents, parameter counts from last.ckpt
  #EVALS   every planning eval found for these runs (older paper-eval pipeline and the later ones)
  #CURVES  validate/* and fit/*_epoch values over training for the architecture runs
  #NOTES   anything that could not be read
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

U = Path("/scratch/jingyuansun")
RUNS_ROOT = U / "smwm-runs"
EVAL_ROOTS = {"paper_eval": U / "smwm-paper-eval", "warmstart": U / "smwm-warmstart-eval", "speedbench": U / "smwm-speed-bench"}
SLUG = {"tworoom": "tworoom", "reacher": "reacher", "pusht": "pusht", "cube": "ogbcube"}
ARCH = re.compile(r"^(tworoom|reacher|pusht|cube)_(policy|invpolicy)_(mlp|resmlp|gru|transformer|mamba)_c(\d+)k(\d+)(_lam[0-9p]+)?_seed(\d+)$")
REF = re.compile(r"^(tworoom|reacher|pusht|cube)_(inverse_lambda_[0-9p]+|policy_mlp_c2k1_noact)_seed([0-2])$")
CURVE_KEYS = ["validate/effective_rank", "validate/mean_per_dim_std", "validate/policy_loss", "validate/pred_loss",
              "validate/inv_loss", "fit/pred_loss_epoch", "fit/policy_loss_epoch", "fit/inv_loss_epoch"]
w = csv.writer(sys.stdout)
notes = []


def g(x):
    return "" if x is None else f"{float(x):.6g}"


def get(cfg, dotted, default=""):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def erank(x):
    xc = x - x.mean(0, keepdims=True)
    sv = np.sqrt(np.clip(np.linalg.eigvalsh(xc.T @ xc), 0, None))
    p = sv / sv.sum()
    p = p[p > 0]
    return math.exp(-float((p * np.log(p)).sum()))                     # as _effective_rank in planning/utils.py


runs = sorted(d.name for d in RUNS_ROOT.iterdir() if d.is_dir() and (ARCH.match(d.name) or REF.match(d.name)))

LAST_COLS = ["fit/loss_epoch", "fit/pred_loss_epoch", "fit/policy_loss_epoch", "fit/inv_loss_epoch",
             "validate/loss", "validate/pred_loss", "validate/policy_loss", "validate/inv_loss",
             "validate/effective_rank", "validate/mean_per_dim_std", "validate/mean_std", "validate/mean_norm",
             "validate/linear_r2_action_coord", "validate/linear_r2_agent_velocity",
             "validate/linear_r2_block_coord", "validate/linear_r2_block_angle"]
CFG_COLS = ["seed", "loss.policy.weight", "loss.policy.arch", "loss.policy.context", "loss.policy.num_future",
            "loss.policy.use_action", "loss.policy.depth", "loss.policy.input", "loss.inverse.weight", "loss.sigreg.weight",
            "loss.goal_policy.weight", "inverse.hidden_dim", "loader.batch_size", "optimizer.lr", "trainer.max_epochs",
            "trainer.precision", "trainer.limit_val_batches", "trainer.val_check_interval", "wm.history_size",
            "wm.action_dim", "data.dataset.name", "data.dataset.num_steps", "data.dataset.frameskip"]

curve_rows, eval_rows = [], []
print("#RUNS")
w.writerow(["run", "env", "config_date", "log_rows", "first_step", "last_step", "last_epoch", "n_val"] + CFG_COLS + LAST_COLS
           + ["emb_shape", "n_clips", "std_fp32", "var_fp32", "erank_fp32", "std_fp32_all", "erank_fp32_all",
              "ckpt_global_step", "ckpt_epoch", "params_total", "params_policy_head", "params_inverse_head", "params_by_module"])
for run in runs:
    rd = RUNS_ROOT / run
    env = run.split("_")[0]
    row = [run, env]
    cfg = {}
    cf = rd / "config.yaml"
    if cf.is_file():
        cfg = yaml.safe_load(cf.read_text()) or {}
        row.append(__import__("datetime").date.fromtimestamp(cf.stat().st_mtime).isoformat())
    else:
        row.append("")
        notes.append([run, "config.yaml missing"])
    # ---- training log
    rows = []
    mc = rd / "lightning" / "local" / "metrics.csv"
    if mc.is_file():
        with open(mc) as f:
            rows = list(csv.DictReader(f))
    else:
        notes.append([run, "metrics.csv missing"])
    steps = [int(float(r["step"])) for r in rows if r.get("step") not in ("", None)]
    epochs = [int(float(r["epoch"])) for r in rows if r.get("epoch") not in ("", None)]
    n_val = sum(1 for r in rows if r.get("validate/pred_loss") not in ("", None))
    row += [len(rows), min(steps) if steps else "", max(steps) if steps else "", max(epochs) if epochs else "", n_val]
    row += [get(cfg, c) for c in CFG_COLS]
    for c in LAST_COLS:
        vals = [r[c] for r in rows if r.get(c) not in ("", None)]
        row.append(g(vals[-1]) if vals else "")
    if ARCH.match(run) and not ARCH.match(run).group(6) or REF.match(run):
        for c in CURVE_KEYS:
            for r in rows:
                if r.get(c) not in ("", None):
                    curve_rows.append([run, c, int(float(r["step"])), g(r[c])])
    # ---- saved final latents, single precision
    ef = rd / "final_embeddings.pt"
    if ef.is_file():
        try:
            emb = torch.load(ef, map_location="cpu", weights_only=True, mmap=True)["emb"].clone().double().numpy()
            n_val_clips = int(get(cfg, "trainer.limit_val_batches", 10) or 10) * int(get(cfg, "loader.batch_size", 256) or 256)
            n = min(n_val_clips, emb.shape[0])
            pairs = lambda e: np.concatenate([e[:, :-1], e[:, 1:]], axis=1).reshape(-1, e.shape[-1])   # _flatten_time_pairs
            x, xa = pairs(emb[:n]), pairs(emb)
            row += ["x".join(map(str, emb.shape)), n, g(x.std(0).mean()), g(x.var(0).mean()), g(erank(x)), g(xa.std(0).mean()), g(erank(xa))]
        except Exception as e:                                                # noqa: BLE001
            notes.append([run, f"final_embeddings.pt: {type(e).__name__}: {e}"])
            row += [""] * 7
    else:
        row += [""] * 7
    # ---- parameter counts from the checkpoint (shapes only; mmap keeps the weights on disk)
    ck = rd / "checkpoints" / "last.ckpt"
    if ck.is_file() and (REF.match(run) or not ARCH.match(run).group(6)):
        try:
            obj = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
            sd = obj["state_dict"]
            by = {}
            for k, v in sd.items():
                if not torch.is_tensor(v) or not v.dtype.is_floating_point:
                    continue
                top = k.split(".")[0] if not k.startswith("model.") else "model." + k.split(".")[1]
                by[top] = by.get(top, 0) + v.numel()
            pol = sum(v for k, v in by.items() if "policy" in k and "goal" not in k)
            inv = sum(v for k, v in by.items() if "inverse" in k)
            row += [obj.get("global_step", ""), obj.get("epoch", ""), sum(by.values()), pol, inv,
                    ";".join(f"{k}={v}" for k, v in sorted(by.items()))]
            del obj, sd
        except Exception as e:                                                # noqa: BLE001
            notes.append([run, f"last.ckpt: {type(e).__name__}: {e}"])
            row += [""] * 6
    else:
        row += [""] * 6
    w.writerow(row)
    # ---- evals
    for src, root in EVAL_ROOTS.items():
        base = root / SLUG[env] / run
        if not base.is_dir():
            continue
        for m in sorted(base.rglob("metrics.json")):
            try:
                d = json.loads(m.read_text())
            except Exception:                                                 # noqa: BLE001
                continue
            sr = d.get("success_rate")
            if sr is None and isinstance(d.get("metrics"), dict):
                sr = d["metrics"].get("success_rate")
            job = next((p for p in m.parts if p.startswith("job_")), "")
            eval_rows.append([run, src, job.replace("job_", ""), str(m.relative_to(base)), sr,
                              ";".join(f"{k}={v}" for k, v in d.items() if isinstance(v, (int, float, str)) and k != "success_rate")[:300]])
    sys.stderr.write(f"done {run}\n")

print("#EVALS")
w.writerow(["run", "source", "job", "path", "success_rate", "other_fields"])
w.writerows(eval_rows)
print("#CURVES")
w.writerow(["run", "key", "step", "value"])
w.writerows(curve_rows)
print("#NOTES")
w.writerows(notes)
