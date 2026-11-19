#!/usr/bin/env python3
"""Summary of the four policy heads of the paper (h1, five datasets): success rate, effective rank, latent scale and losses.

The earlier head variants (MLP / Transformer / Mamba, context, k, action input, ...) are in make_early_summary.py.

Read-only on its inputs; writes two files next to this script:
  paper_heads_summary.md      the tables (Chinese text, generated: do not edit by hand)
  paper_heads_per_seed.csv    one row per (dataset, method, seed) with every value behind the tables

Inputs
  --paper-results DIR   (default ~/Downloads/iclr2027/results) the files collected on DelftBlue for the paper:
      h1_evals_all.csv   one row per finished planning eval (warm-start pipeline and speed benchmark)
      h1_train_all.csv   one row per run (run name, whether final_embeddings.pt exists)
      h1_curves.csv      fit/*_epoch and validate/* values of lightning/local/metrics.csv
      h1_fp32.csv        single-precision statistics of the saved final latents (first 2560 held-out clips)
  --repo-results FILE   (default planning/results.csv) the repo's own run table (older eval pipeline; it also
      holds the final training metrics of the seeds that were trained on Leonardo and the early head variants)

Conventions (the same as the paper's make_results_tables.py / make_diagnostics_table.py, so the numbers agree
with the paper's tables where they overlap):
  evals   for each (dataset, method, seed, label) the eval of the warm-start pipeline when there is one, else the
          speed-benchmark one; among several evals of one source the first job
  logs    seeds whose log reaches the last epoch and whose final latents were saved (DelftBlue runs only)
  cells   mean +- sample standard deviation over training seeds, rounded half up; a superscript is the number of
          seeds when it is below 5

Run:  python3 planning/analysis/policy_heads/make_paper_heads_summary.py
"""
import argparse
import csv
import statistics as st
from collections import defaultdict
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = ["tworoom", "pusht", "cube", "reacher_mix", "reacher"]
NAME = {"tworoom": "TwoRoom", "pusht": "Push-T", "cube": "Cube", "reacher_mix": "Reacher-PD", "reacher": "Reacher-Random"}
HEADS = [("endpoints", "policy"), ("displacement", "policy_delta"),
         ("endpoints + goal（默认）", "policy_goal"), ("displacement + goal", "policy_delta_goal")]
REFS = [("LeWM（SIGReg，无 head）", "lewm"), ("SMWM（inverse head）", "inverse")]
ROWS = REFS + HEADS
SHORT = {"lewm": "LeWM", "inverse": "SMWM", "policy": "endpoints", "policy_delta": "displacement",
         "policy_goal": "endpoints + goal", "policy_delta_goal": "displacement + goal"}
FINAL_STEP = {"tworoom": 29199, "pusht": 76209, "cube": 67499, "reacher_mix": 67499, "reacher": 67499}
ACTION = {"inverse": "inv_loss", "lewm": None}            # every policy head logs policy_loss
EVAL_LABELS = ["cem30", "mean30", "policy", "cem10", "mean10", "cem5", "mean5"]
SUP = {1: "¹", 2: "²", 3: "³", 4: "⁴"}


# ---------------------------------------------------------------------------------------------- formatting
def rnd(x, q="0.1"):
    return Decimal(str(float(x))).quantize(Decimal(q), rounding=ROUND_HALF_UP)


def sci(x):
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant}e{int(exp)}"


def num(x):
    """One decimal from 10 up, two from 0.1 up, three from 0.01 up, scientific notation below."""
    if x is None:
        return "–"
    if x >= 10:
        return f"{rnd(x, '0.1')}"
    if x >= 0.1:
        return f"{rnd(x, '0.01')}"
    if x >= 0.01:
        return f"{rnd(x, '0.001')}"
    return sci(x)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return st.fmean(vals) if vals else None


def sd(vals):
    vals = [v for v in vals if v is not None]
    return st.stdev(vals) if len(vals) > 1 else 0.0


def sup(n):
    return SUP.get(n, "") if n < 5 else ""


def pm(vals, q="0.1"):
    """mean ± std with the seed count as a superscript when below 5."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    return f"{rnd(mean(vals), q)} ± {rnd(sd(vals), q)}{sup(len(vals))}"


def table(header, rows, align=None):
    align = align or ["l"] + ["c"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join({"l": "---", "c": ":---:", "r": "---:"}[a] for a in align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- inputs
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--paper-results", default=str(Path.home() / "Downloads/iclr2027/results"))
ap.add_argument("--repo-results", default=str(HERE.parents[1] / "results.csv"))
args = ap.parse_args()
P = Path(args.paper_results)

evals = {}                                                   # (ds, method, seed, label) -> (rank, success, source)
for r in csv.DictReader(open(P / "h1_evals_all.csv")):
    if r["success_rate"] in ("", "None"):
        continue
    key = (r["dataset"], r["method"], int(r["seed"]), r["label"])
    rank = (0 if r["source"] == "warmstart" else 1, int(r["job"]))
    if key not in evals or rank < evals[key][0]:
        evals[key] = (rank, float(r["success_rate"]), r["source"])

curves = defaultdict(lambda: defaultdict(list))              # (ds, method, seed) -> key -> [(step, value)]
for r in csv.DictReader(open(P / "h1_curves.csv")):
    curves[(r["dataset"], r["method"], int(r["seed"]))][r["key"]].append((int(float(r["step"])), float(r["value"])))
fp32 = {(r["dataset"], r["method"], int(r["seed"])): r for r in csv.DictReader(open(P / "h1_fp32.csv"))}
runname = {(r["dataset"], r["method"], int(r["seed"])): r["run"] for r in csv.DictReader(open(P / "h1_train_all.csv"))}

old = defaultdict(list)                                      # run name -> rows of planning/results.csv
for r in csv.DictReader(open(args.repo_results)):
    if r["status"] == "ok":
        old[r["run"]].append(r)


def last(run, key):
    vals = sorted(curves[run].get(key, []))
    return vals[-1][1] if vals else None


def complete(run):
    e = sorted(curves[run].get("fit/pred_loss_epoch", []))
    return bool(e) and e[-1][0] == FINAL_STEP[run[0]]


def log_seeds(ds, method):
    return [s for s in range(5) if complete((ds, method, s)) and (ds, method, s) in fp32]


def success(ds, method, label):
    return {s: evals[(ds, method, s, label)][1] for s in range(5) if (ds, method, s, label) in evals}


def diag(run):
    """Every logged quantity of one run (DelftBlue log + saved latents)."""
    ds, method, _ = run
    act = ACTION.get(method, "policy_loss")
    pred_val = last(run, "validate/pred_loss")
    d = dict(rank_log=last(run, "validate/effective_rank"), rank_fp32=float(fp32[run]["erank"]),
             std_log=last(run, "validate/mean_per_dim_std"), std_fp32=float(fp32[run]["per_dim_std"]),
             pred_train=last(run, "fit/pred_loss_epoch"), pred_val=pred_val,
             rel_err=100.0 * pred_val / float(fp32[run]["mean_var"]),
             act_train=last(run, f"fit/{act}_epoch") if act else None,
             act_val=last(run, f"validate/{act}") if act else None,
             sigreg_train=last(run, "fit/sigreg_loss_epoch"), sigreg_val=last(run, "validate/sigreg_loss"),
             loss_train=last(run, "fit/loss_epoch"), loss_val=last(run, "validate/loss"))
    d["expl_var"] = 100.0 * (1.0 - d["act_val"]) if d["act_val"] is not None else None
    reg = d["act_train"] if act else d["sigreg_train"]
    d["lambda"] = (d["loss_train"] - d["pred_train"]) / reg if reg else None      # weight implied by the logged losses
    return d


D = {(ds, m): {s: diag((ds, m, s)) for s in log_seeds(ds, m)} for ds in DS for _, m in ROWS}


def col(ds, method, key):
    return [D[(ds, method)][s][key] for s in sorted(D[(ds, method)])]


def expl(ds, method):
    """Explained action variance in %, each seed clamped at 0 (a held-out loss can exceed 1 by noise), as in the paper's table."""
    return mean(max(v, 0.0) for v in col(ds, method, "expl_var"))


def old_row(ds, method, seed, need):
    """The row of planning/results.csv for this run that has the column `need` (DelftBlue row first when a run has two)."""
    rows = [r for r in old.get(runname.get((ds, method, seed), ""), []) if r[need] != ""]
    return sorted(rows, key=lambda r: r["cluster"] != "DelftBlue")[0] if rows else None


def leonardo_rows(ds, method):
    """Seeds of a head that have no DelftBlue log but a final-metrics row in planning/results.csv."""
    out = {}
    for s in range(5):
        r = None if s in D[(ds, method)] else old_row(ds, method, s, "eff_rank")
        if r:
            out[s] = dict(cluster=r["cluster"], rank_log=float(r["eff_rank"]), pred_train=float(r["pred_loss"]),
                          act_train=float(r["policy_loss"]), success_old=float(r["success_rate"]) if r["success_rate"] else None)
    return out


def old_success(ds, method, seed):
    r = old_row(ds, method, seed, "success_rate")
    return float(r["success_rate"]) if r else None


# ---------------------------------------------------------------------------------------------- checks
for ds in DS:
    for _, m in ROWS:
        for s, d in D[(ds, m)].items():
            lam = {"tworoom": 0.1, "pusht": 30.0, "cube": 1.0, "reacher_mix": 5.0, "reacher": 5.0}[ds] if m != "lewm" else 0.09
            assert abs(d["lambda"] - lam) < 0.02 * lam, (ds, m, s, d["lambda"])      # total = pred + lambda * regularizer
            assert (ds, m, s) in runname

# ---------------------------------------------------------------------------------------------- markdown
L = []
add = L.append
add("# 论文里的四种 policy head：成功率、effective rank 与各项 loss")
add("")
add(f"> 由 `planning/analysis/policy_heads/make_paper_heads_summary.py` 生成（{date.today().isoformat()}），请勿手改；重新生成：`python3 planning/analysis/policy_heads/make_paper_heads_summary.py`。")
add("> 数据：h1 设定（history size 1）的全部运行，评测与训练日志于 2026-11-13 从 DelftBlue 收集"
    "（`~/Downloads/iclr2027/results/h1_*.csv`）。逐 seed 的数值在同目录的 `paper_heads_per_seed.csv`。")
add("")
add("## 1. 四种 policy head 与两个参照")
add("")
add(table(["名称", "运行名后缀", "head 的输入", "预测目标"], [
    ["endpoints", "`policy_mlp_c2k1_noact`", "(z_t, z_{t+1})", "a_{t+1}"],
    ["displacement", "`…_noact_delta`", "(z_{t+1} − z_t)", "a_{t+1}"],
    ["endpoints + goal（论文默认，“Ours”）", "`…_noact_goalfinal`", "(z_t, z_{t+1}, z_g)", "a_{t+1}"],
    ["displacement + goal", "`…_noact_delta_goalfinal`", "(z_{t+1} − z_t, z_g − z_{t+1})", "a_{t+1}"],
    ["SMWM（参照）", "`…_inverse…`", "(z_t, z_{t+1})", "a_t（inverse dynamics）"],
    ["LeWM（参照）", "`…_sigreg`", "无 head，SIGReg 正则", "–"],
], align=["l", "l", "l", "l"]))
add("")
add("- z_g 是同一条轨迹最后一帧的 latent（`goal_source: final`）。四种 head 的网络相同（MLP，宽 256，不输入 a_t），只有输入不同。")
add("- 正则权重 λ 取各环境 SMWM 的值：TwoRoom 0.1，Push-T 30，Cube 1，Reacher 5；LeWM 的 SIGReg 权重 0.09。"
    "脚本用日志核对过：total loss = prediction loss + λ × 正则项。")
add("")
add("## 2. 指标与口径")
add("")
add("- **成功率**：100 个评测任务，目标在 25 个环境步之后，步数预算 50。CEM-30 = 标准 CEM（30 次迭代）；"
    "warm-start-30 = 用 policy head 的输出作为 CEM 的初始均值；head only = 不搜索，直接执行 head 的输出。")
add("- **effective rank**：exp(归一化奇异值的熵)，192 维 latent。logged = 训练中验证环节记录的最后一个值"
    "（bfloat16 混合精度，前 2560 个 held-out clip，论文表 mech 用的就是它）；fp32 = 用保存的最终 latent 在单精度下重算（同样的 2560 个 clip）。")
add("- **latent std**：最终 latent 每一维的标准差（单精度），用来判断是否坍缩。")
add("- **prediction loss**：预测 latent 与目标 latent 的 MSE。train = 最后一个 epoch 的训练均值（forward model 带 dropout）；"
    "val = held-out 上最后一次验证的值。**相对预测误差** = val prediction loss / latent 方差。")
add("- **action loss**：标准化动作上的 MSE（policy head 是 `policy_loss`，SMWM 是 `inv_loss`），所以 1 − val loss 就是 held-out 上动作的**可解释方差**；"
    "loss ≈ 1 表示 head 只能输出均值。")
add("- 每格是训练 seed 间的均值 ± 样本标准差（四舍五入）；上标是该格的 seed 数（不足 5 个时标出）。"
    "成功率和日志的 seed 数可能不同（第 7 节）。")
add("")

# ---- success -----------------------------------------------------------------------------------------
add("## 3. 成功率（%）")
add("")
add("### 3.1 默认预算：CEM 30 次迭代")
add("")
rows = []
for block, label in [("CEM-30", "cem30"), ("warm-start-30", "mean30"), ("head only", "policy")]:
    for lab, m in ROWS:
        if label != "cem30" and m in ("lewm", "inverse"):
            continue
        rows.append([block, lab] + [pm(success(ds, m, label).values()) for ds in DS])
add(table(["评测方式", "方法 / head"] + [NAME[d] for d in DS], rows, align=["l", "l"] + ["c"] * 5))
add("")
add("### 3.2 更小的预算：CEM 10 次和 5 次迭代")
add("")
rows = []
for block, label in [("CEM-10", "cem10"), ("warm-start-10", "mean10"), ("CEM-5", "cem5"), ("warm-start-5", "mean5")]:
    for lab, m in ROWS:
        if label.startswith("mean") and m in ("lewm", "inverse"):
            continue
        rows.append([block, lab] + [pm(success(ds, m, label).values()) for ds in DS])
add(table(["评测方式", "方法 / head"] + [NAME[d] for d in DS], rows, align=["l", "l"] + ["c"] * 5))
add("")

# ---- effective rank ----------------------------------------------------------------------------------
add("## 4. Effective rank 与 latent 尺度")
add("")
add("第 4、5 节是日志类指标，每格的 seed 数见第 7 节的表（“/” 右侧）；没有 ± 的表不再标上标。")
add("")
add("### 4.1 Effective rank（logged，bfloat16）")
add("")
add(table(["方法 / head"] + [NAME[d] for d in DS], [[lab] + [pm(col(ds, m, "rank_log")) for ds in DS] for lab, m in ROWS]))
add("")
add("### 4.2 Effective rank（fp32，最终 latent）")
add("")
add(table(["方法 / head"] + [NAME[d] for d in DS], [[lab] + [pm(col(ds, m, "rank_fp32")) for ds in DS] for lab, m in ROWS]))
add("")
add("### 4.3 Latent 每维标准差（fp32）")
add("")
add(table(["方法 / head"] + [NAME[d] for d in DS],
          [[lab] + [num(mean(col(ds, m, "std_fp32"))) for ds in DS] for lab, m in ROWS]))
add("")
add("注意：latent std 很小时（TwoRoom 约 0.05，Reacher-Random 上坍缩的 head 约 1e-3 以下），logged 的 effective rank 被 bfloat16 的舍入噪声抬高，"
    "应看 fp32 的值和 latent std。Reacher-Random 上 endpoints、displacement、endpoints + goal 三种 head 都坍缩了，logged 的 140–160 不是真实的维度。")
add("")

# ---- losses ------------------------------------------------------------------------------------------
add("## 5. Loss")
add("")
add("### 5.1 Prediction loss（train / val）")
add("")
add(table(["方法 / head"] + [NAME[d] for d in DS],
          [[lab] + [f"{sci(mean(col(ds, m, 'pred_train')))} / {sci(mean(col(ds, m, 'pred_val')))}" for ds in DS]
           for lab, m in ROWS]))
add("")
add("### 5.2 相对预测误差（val prediction loss / latent 方差，%）")
add("")


def rel_cell(ds, m):
    v = col(ds, m, "rel_err")
    return ">100" if mean(v) > 100 else num(mean(v))


add(table(["方法 / head"] + [NAME[d] for d in DS], [[lab] + [rel_cell(ds, m) for ds in DS] for lab, m in ROWS]))
add("")
add("坍缩的模型 prediction loss 的绝对值最低（latent 几乎是常数），相对误差却超过 100 %，所以不同 head 之间不要直接比 prediction loss 的绝对值。")
add("")
add("### 5.3 Action loss（train / val）")
add("")
add(table(["方法 / head"] + [NAME[d] for d in DS],
          [[lab] + [f"{rnd(mean(col(ds, m, 'act_train')), '0.001')} / {rnd(mean(col(ds, m, 'act_val')), '0.001')}"
                    for ds in DS] for lab, m in ROWS if m != "lewm"]))
add("")
add("SMWM 一行是 inverse loss（目标 a_t），其余是 policy loss（目标 a_{t+1}），两者的目标不同，只在各自内部比较。")
add("")
add("### 5.4 动作的可解释方差（1 − val action loss，%）")
add("")
add(table(["方法 / head"] + [NAME[d] for d in DS],
          [[lab] + [f"{rnd(expl(ds, m), '0.1')}" for ds in DS]
           for lab, m in ROWS if m != "lewm"]))
add("")
add("### 5.5 Total loss（train / val）")
add("")
add(table(["方法 / head"] + [NAME[d] for d in DS],
          [[lab] + [f"{num(mean(col(ds, m, 'loss_train')))} / {num(mean(col(ds, m, 'loss_val')))}" for ds in DS]
           for lab, m in ROWS if m != "lewm"]))
add("")
add("Total loss = prediction loss + λ × action loss，λ 随环境不同，所以只能在同一数据集内比较。"
    "LeWM 的正则项是 SIGReg，量纲不同，没有列在这里；它的 SIGReg loss 和 total loss 在逐 seed 的 CSV 里。")
add("")

# ---- one table per dataset ---------------------------------------------------------------------------
add("## 6. 按数据集的总表")
add("")
for ds in DS:
    add(f"### {NAME[ds]}")
    add("")
    rows = []
    for lab, m in ROWS:
        ev = "–" if m == "lewm" else f"{rnd(expl(ds, m), '0.1')}"
        rows.append([lab,
                     pm(success(ds, m, "cem30").values()),
                     pm(success(ds, m, "mean30").values()),
                     pm(success(ds, m, "policy").values()),
                     f"{rnd(mean(col(ds, m, 'rank_log')), '0.1')} / {rnd(mean(col(ds, m, 'rank_fp32')), '0.1')}",
                     num(mean(col(ds, m, "std_fp32"))),
                     f"{sci(mean(col(ds, m, 'pred_train')))} / {sci(mean(col(ds, m, 'pred_val')))}",
                     rel_cell(ds, m),
                     "–" if m == "lewm" else f"{rnd(mean(col(ds, m, 'act_train')), '0.001')} / {rnd(mean(col(ds, m, 'act_val')), '0.001')}",
                     ev,
                     f"{len(success(ds, m, 'cem30'))} / {len(D[(ds, m)])}"])
    add(table(["方法 / head", "CEM-30", "warm-start-30", "head only", "eff. rank（logged / fp32）", "latent std",
               "pred loss（train / val）", "相对误差 %", "action loss（train / val）", "可解释方差 %", "seed 数（评测 / 日志）"], rows))
    add("")

# ---- seeds -------------------------------------------------------------------------------------------
add("## 7. Seed 覆盖")
add("")
add("每格：有 CEM-30 评测的 seed / 有完整 DelftBlue 训练日志和最终 latent 的 seed。")
add("")
add(table(["方法 / head"] + [NAME[d] for d in DS],
          [[lab] + ["".join(map(str, sorted(success(ds, m, "cem30")))) + " / " + "".join(map(str, sorted(D[(ds, m)]))) for ds in DS]
           for lab, m in ROWS]))
add("")
add("- 第 3–6 节的日志类指标（rank、std、loss）只用 DelftBlue 上有日志的 seed，与论文的表 mech、表 diag 口径一致。")
add("- 缺的 seed：Push-T 和 Cube 的部分 seed 0–2 在 Leonardo 上训练（日志不在 DelftBlue）；TwoRoom 的 displacement + goal seed 3 没有训练。")
add("")
add("### 7.1 Leonardo 上训练的 seed（来自 `planning/results.csv`，旧评测流程）")
add("")
add("这些 seed 只有最后一个 epoch 的训练指标（3 位有效数字），没有 val loss 和 fp32 的统计。“合并”一行把它们与 DelftBlue 的 seed 一起求均值。")
add("")
rows = []
for ds in DS:
    for lab, m in HEADS:
        leo = leonardo_rows(ds, m)
        if not leo:
            continue
        for s, r in sorted(leo.items()):
            rows.append([NAME[ds], lab, s, r["cluster"], f"{rnd(r['rank_log'], '0.1')}", sci(r["pred_train"]), f"{rnd(r['act_train'], '0.001')}",
                         "–" if r["success_old"] is None else f"{rnd(r['success_old'], '1')}"])
        allr = col(ds, m, "rank_log") + [r["rank_log"] for r in leo.values()]
        allp = col(ds, m, "pred_train") + [r["pred_train"] for r in leo.values()]
        alla = col(ds, m, "act_train") + [r["act_train"] for r in leo.values()]
        rows.append([NAME[ds], lab, f"**合并（{len(allr)} 个 seed）**", "DelftBlue + Leonardo", f"**{pm(allr)}**", f"**{sci(mean(allp))}**",
                     f"**{rnd(mean(alla), '0.001')}**", ""])
add(table(["数据集", "head", "seed", "训练集群", "eff. rank（logged）", "pred loss（train）", "policy loss（train）", "成功率（旧流程，CEM-30）"], rows,
          align=["l", "l", "c", "l", "c", "c", "c", "c"]))
add("")
pairs = [(success(ds, m, "cem30")[s], old_success(ds, m, s)) for ds in DS for _, m in ROWS for s in sorted(success(ds, m, "cem30"))
         if old_success(ds, m, s) is not None]
diffs = [a - b for a, b in pairs]
add(f"旧流程（`eval.py`，同样是 100 个任务、目标 25 步、预算 50、CEM 30 次迭代）与新流程的 CEM-30 在 {len(diffs)} 个两边都评测过的运行上"
    f"平均相差 {rnd(mean(abs(d) for d in diffs), '0.1')} 个点（最大 {rnd(max(abs(d) for d in diffs), '1')}，没有系统偏差：新减旧的均值为 {abs(rnd(mean(diffs), '0.1'))}），"
    "可以互相参考，但第 3 节的均值没有混用。")
add("")
add("### 7.2 用旧流程补齐后的 CEM-30 成功率")
add("")
add("新流程缺评测的 seed 如果在旧流程里评测过，可以补进来看均值是否稳定（两个流程混用，仅供参考）。")
add("")
rows = []
for ds in DS:
    for lab, m in HEADS:
        new = success(ds, m, "cem30")
        fill = {s: old_success(ds, m, s) for s in range(5) if s not in new and old_success(ds, m, s) is not None}
        if fill:
            rows.append([NAME[ds], lab, pm(new.values()), "、".join(f"seed {s}: {rnd(v, '1')}" for s, v in sorted(fill.items())),
                         pm(list(new.values()) + list(fill.values()))])
add(table(["数据集", "head", "新流程", "旧流程补的 seed", "补齐后"], rows, align=["l", "l", "c", "l", "c"]))
add("")

# ---- pointer to the earlier head variants -------------------------------------------------------------
add("## 8. 更早的 head 变体")
add("")
add("MLP / Transformer / Mamba、不同 context 和 k、是否输入 a_t、goal policy、λ 扫描等 9–10 月的结果不在这里，"
    "见同目录的 `early_heads_summary.md`（由 `make_early_summary.py` 生成）。")
add("")
add("## 9. 逐 seed 数据")
add("")
add("`paper_heads_per_seed.csv`：每个（数据集，方法，seed）一行，包含本页所有表格背后的数值，以及 `log_source` 一列"
    "（DelftBlue = 完整日志；Leonardo = 只有 `planning/results.csv` 里的最终训练指标）。")

(HERE / "paper_heads_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------------------------- per-seed csv
CSV_NAME = {"rank_log": "eff_rank_logged", "rank_fp32": "eff_rank_fp32", "std_log": "latent_std_logged", "std_fp32": "latent_std_fp32",
            "pred_train": "pred_loss_train", "pred_val": "pred_loss_val", "rel_err": "rel_pred_error_pct",
            "act_train": "action_loss_train", "act_val": "action_loss_val", "expl_var": "explained_var_pct",
            "sigreg_train": "sigreg_loss_train", "sigreg_val": "sigreg_loss_val", "loss_train": "total_loss_train",
            "loss_val": "total_loss_val", "lambda": "lambda_implied"}
EVAL_NAME = {"cem30": "cem30", "mean30": "warmstart30", "policy": "head_only", "cem10": "cem10", "mean10": "warmstart10",
             "cem5": "cem5", "mean5": "warmstart5"}
FIELDS = (["dataset", "method", "seed", "run", "log_source"] + [f"success_{EVAL_NAME[k]}" for k in EVAL_LABELS]
          + ["success_old_pipeline_cem30"] + list(CSV_NAME.values()))
with open(HERE / "paper_heads_per_seed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for ds in DS:
        for _, m in ROWS:
            leo = leonardo_rows(ds, m) if m in dict((b, a) for a, b in HEADS) else {}
            for s in range(5):
                row = {"dataset": NAME[ds], "method": SHORT[m], "seed": s, "run": runname.get((ds, m, s), "")}
                for k in EVAL_LABELS:
                    v = success(ds, m, k).get(s)
                    row[f"success_{EVAL_NAME[k]}"] = "" if v is None else f"{v:.6g}"
                v = old_success(ds, m, s)
                row["success_old_pipeline_cem30"] = "" if v is None else f"{v:.6g}"
                if s in D[(ds, m)]:
                    row["log_source"] = "DelftBlue"
                    row.update({CSV_NAME[k]: ("" if v is None else f"{v:.6g}") for k, v in D[(ds, m)][s].items()})
                elif s in leo:
                    row["log_source"] = leo[s]["cluster"] + " (results.csv)"
                    row.update({CSV_NAME[k]: f"{leo[s][k]:.6g}" for k in ("rank_log", "pred_train", "act_train")})
                if any(row.get(k) not in ("", None) for k in FIELDS[5:]):
                    w.writerow(row)

print("wrote", HERE / "paper_heads_summary.md")
print("wrote", HERE / "paper_heads_per_seed.csv")
