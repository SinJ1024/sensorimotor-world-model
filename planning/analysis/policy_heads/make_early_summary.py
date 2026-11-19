#!/usr/bin/env python3
"""The earliest policy-head runs (September / October 2026), reorganised by head type.

Source: the results table of the Obsidian note "9 Oct with Yanbo.md" (section "Sensorimotor world model
(SMWM): policy vs inverse regularizer"; one row per run: head, context C, predicted actions k, action input,
goal, lambda, seed, final training losses, effective rank, steps, planning success). The note is read, never
written. Every row of that table is also a row of planning/results.csv, which gives the run name.

Optional extras (not in the note): early_heads_delftblue_raw.txt next to this script, the unchanged output of
collect_early_heads_on_delftblue.py (saved configs, held-out losses, latent scale, effective rank of the saved
latents in single precision, parameter count of the head).

Writes next to this script:
  early_heads_summary.md   the tables (Chinese text; generated, do not edit by hand)
  early_heads_runs.csv     one row per run of the note's table, with the run name and the extras
With --obsidian PATH it also writes a shorter comparison note (head tables, which heads had a lambda sweep,
what LeWAM changes) to PATH; the default run does not touch the vault.

Section 7 of that comparison note is a later supplement (November 2026): the Transformer / Mamba / MLP heads on
(z_t, z_{t+1}, z_g) without action input. Its numbers come from goal_head_arch_runs.csv next to this script (the
unchanged output of planning/experiments/head_arch/collect_head_arch.py for the five datasets, one row per run)
and, for the LeWM reference rows, from paper_heads_per_seed.csv (written by make_paper_heads_summary.py).

Run:  python3 planning/analysis/policy_heads/make_early_summary.py
"""
import argparse
import csv
import statistics as st
from collections import OrderedDict, defaultdict
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--note", default=str(Path.home() / "Documents/Obsidian Vault/9 Oct with Yanbo.md"))
ap.add_argument("--repo-results", default=str(HERE.parents[1] / "results.csv"))
ap.add_argument("--extras", default=str(HERE / "early_heads_delftblue_raw.txt"))
ap.add_argument("--configs", default=str(HERE / "delftblue_run_configs.csv"),
                help="output of collect_run_configs_on_delftblue.py: the saved config of every run (lambda audit)")
ap.add_argument("--goal-heads", default=str(HERE / "goal_head_arch_runs.csv"),
                help="output of planning/experiments/head_arch/collect_head_arch.py for the five datasets: the heads on "
                     "(z_t, z_{t+1}, z_g) without action input (section 7 of the comparison note)")
ap.add_argument("--paper-heads", default=str(HERE / "paper_heads_per_seed.csv"),
                help="per-seed table of make_paper_heads_summary.py: the LeWM reference rows of section 7 and the runs "
                     "evaluated by both pipelines")
ap.add_argument("--obsidian", default="",
                help="also write the comparison note to this path (a new note in the Obsidian vault); nothing is written there without it")
args = ap.parse_args()

ENVS = ["TwoRoom", "Reacher", "Push-T", "OGBench-Cube"]
ENV_ZH = {"TwoRoom": "TwoRoom", "Reacher": "Reacher（随机动作）", "Reacher (expert)": "Reacher（expert）",
          "Push-T": "Push-T", "OGBench-Cube": "Cube"}
LAMBDA = {"TwoRoom": "0.1", "Reacher": "5", "Reacher (expert)": "5", "Push-T": "30", "OGBench-Cube": "1"}
DASH = "–"


# ---------------------------------------------------------------------------------------------- formatting
def rnd(x, q="0.1"):
    return Decimal(str(float(x))).quantize(Decimal(q), rounding=ROUND_HALF_UP)


def sci(x):
    """Three significant digits in scientific notation: 0.000206 -> 2.06e-4."""
    mant, exp = f"{x:.2e}".split("e")
    return f"{mant}e{int(exp)}"


def loss(x):
    """Action losses: three decimals from 0.1 up, three significant digits below."""
    if x is None:
        return DASH
    return f"{rnd(x, '0.001')}" if x >= 0.1 else f"{rnd(x, '0.0001')}"


def rank(x):
    """Three significant digits, as in the note (182, 55.6, 9.01)."""
    return DASH if x is None else f"{x:.3g}"


def mean(vals):
    vals = [v for v in vals if v is not None]
    return st.fmean(vals) if vals else None


def table(header, rows, align=None):
    align = align or ["l"] + ["c"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join({"l": "---", "c": ":---:", "r": "---:"}[a] for a in align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- the note's table
def num(s):
    s = s.strip().strip("*")
    return float(s) if s not in ("", DASH) else None


lines = Path(args.note).read_text(encoding="utf-8").splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("| env") and "regularizer" in l)
header = [c.strip() for c in lines[start].strip().strip("|").split("|")]
RUNS = []
for l in lines[start + 2:]:
    if not l.startswith("|"):
        break
    cells = [c.strip() for c in l.strip().strip("|").split("|")]
    assert len(cells) == len(header), l
    raw = dict(zip(header, cells))
    RUNS.append(dict(
        raw=raw, env=raw["env"], reg=raw["regularizer"], head=raw["head"].lower().replace(", δz", "").replace(", Δz", ""),
        delta="Δz" in raw["head"], C=int(raw["C"]) if raw["C"] != DASH else None, k=int(raw["k"]) if raw["k"] != DASH else None,
        at={"yes": True, "no": False}.get(raw["a_t in"]), G=raw["G"], lam=raw["λ"], seed=int(raw["seed"]), cluster=raw["cluster"],
        pred=num(raw["pred"]), inv=num(raw["inv"]), policy=num(raw["policy"]), goal=num(raw["goal"]),
        rank=num(raw["eff rank"]), steps=int(raw["steps"]), success=num(raw["success %"]), other=raw["other evals"]))
NOTE_LINES = (start + 1, start + 2 + len(RUNS))

# run names: every row of the note is a row of planning/results.csv
repo = defaultdict(list)
for r in csv.DictReader(open(args.repo_results)):
    env = "Reacher (expert)" if "reacher_expert" in r["run"] else r["env"]
    repo[(env, r["seed"], r["cluster"], r["pred_loss"], r["inv_loss"], r["policy_loss"], r["goal_loss"], r["eff_rank"], r["step"],
          r["success_rate"])].append(r["run"])
for r in RUNS:
    w = r["raw"]
    names = repo.get((w["env"], w["seed"], w["cluster"], w["pred"], w["inv"], w["policy"], w["goal"], w["eff rank"], w["steps"],
                      w["success %"].strip("*")), [])
    assert len(names) == 1, (w, names)
    r["run"] = names[0]

# optional extras from DelftBlue, by run name (DelftBlue rows only)
def read_blocks(path):
    """The collector prints several CSV tables, each after a line '#NAME'."""
    blocks, name = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            name = line[1:].strip()
            blocks[name] = []
        elif name:
            blocks[name].append(line)
    return {k: (list(csv.DictReader(v)) if k != "NOTES" else v) for k, v in blocks.items()}


EXTRA, EXTRA_NOTES = {}, []
if Path(args.extras).is_file():
    blocks = read_blocks(args.extras)
    EXTRA_NOTES = blocks.get("NOTES", [])
    for e in blocks["RUNS"]:
        if e.get("validate/pred_loss") and e.get("var_fp32"):
            e["rel_err"] = 100.0 * float(e["validate/pred_loss"]) / float(e["var_fp32"])     # share of the latent variance not predicted
        steps = e.get("ckpt_global_step") or (str(int(e["last_step"]) + 1) if e.get("last_step") else "")
        if steps and e.get("loader.batch_size"):
            e["clips_seen"] = int(steps) * int(e["loader.batch_size"])                        # training clips over the whole run
        EXTRA[e["run"]] = e


def extra(r, key):
    e = EXTRA.get(r["run"]) if r["cluster"] == "DelftBlue" else None
    return float(e[key]) if e and e.get(key) not in ("", None, "nan") else None


def family(env, reg="policy", head="mlp", C=2, k=1, at=True, G=DASH, delta=False, lam=None):
    """One row per seed of a configuration (the DelftBlue run when a seed was trained on both clusters)."""
    lam = LAMBDA[env] if lam is None else lam
    lam = f"{lam} / {lam}" if reg == "inverse+policy" else lam
    per_seed = {}
    for r in RUNS:
        if (r["env"], r["reg"], r["head"], r["delta"], r["C"], r["k"], r["at"], r["G"], r["lam"]) == (env, reg, head, delta, C, k, at, G, lam):
            if r["seed"] not in per_seed or r["cluster"] == "DelftBlue":
                per_seed[r["seed"]] = r
    return [per_seed[s] for s in sorted(per_seed)]


USED = set()


def cell(rows, key, fmt, pm=False):
    """Mean over seeds; with pm the sample standard deviation and the seed count are added when there are several seeds."""
    for r in rows:
        USED.add(id(r))
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return DASH
    if len(vals) == 1 or not pm:
        return fmt(mean(vals))
    return f"{rnd(mean(vals), '0.1')} ± {rnd(st.stdev(vals), '0.1')}（{len(vals)}）"


def succ(rows):
    return cell(rows, "success", lambda x: f"{rnd(x, '1')}", pm=True)


def xcell(rows, key, fmt):
    vals = [extra(r, key) for r in rows]
    vals = [v for v in vals if v is not None]
    return fmt(mean(vals)) if vals else DASH


def nseeds(rows):
    return f"{len(rows)}" if rows else DASH


# ---------------------------------------------------------------------------------------------- markdown
L = []
add = L.append
add("# 最早的 policy head 结果：按 head 类型整理")
add("")
add(f"> 由 `planning/analysis/policy_heads/make_early_summary.py` 生成（{date.today().isoformat()}），请勿手改。")
add(f"> 数据来源：Obsidian 笔记 `9 Oct with Yanbo.md` 中 “SMWM: policy vs inverse regularizer” 的大表"
    f"（第 {NOTE_LINES[0]}–{NOTE_LINES[1]} 行，共 {len(RUNS)} 个运行），这里只是重新分组，数值未改。"
    "该表的每一行都与仓库的 `planning/results.csv` 逐行一致。")
add("")
add("## 1. 记号与口径")
add("")
add("- **policy head**：从 C 个 latent 的窗口（以及窗口内的动作，如果输入 a_t）预测窗口最后一个 latent 处的动作及其后共 k 个动作。"
    "C = 2、k = 1 就是 a_{t+1} = π(z_t, z_{t+1}[, a_t])；C = 1 是 a_t = π(z_t)（只看一个状态，窗口里没有动作可输入）。")
add("- **网络**：MLP 把窗口拼接后过两层 MLP（宽 256）；Transformer 和 Mamba 把每个时刻的 latent（加动作）作为一个 token，过 2 层序列模型，取最后一个 token。")
add("- **其他正则**：inverse 是 a_t = h(z_t, z_{t+1})；goal policy 是 a_t = π(z_t, z_{t+G})；inverse + policy 是两个 head 同时使用。")
add("- **成功率**：旧评测流程（`eval.py`），100 个任务，目标在 25 个环境步之后，预算 50 步，CEM 30 次迭代。")
add("- **loss**：最后一个 epoch 的训练均值。pred = 预测 latent 的 MSE；inv / policy / goal = 标准化动作上的 MSE（1 表示只能输出均值）。")
add("- **effective rank**：训练日志里最后一次验证的值（bfloat16，192 维 latent）。latent 尺度很小时它会被舍入噪声抬高，所以高 rank 不一定是好事。")
add("- **steps**：训练步数（10 个 epoch）。窗口越长，数据里能取出的片段越少，步数也越少；C = 16 的运行还把 batch size 从 256 降到了 64（见 2.1）。")
add("- 所有运行都是 h1（forward model 只看一帧），λ 取各环境 SMWM 的值（TwoRoom 0.1，Reacher 5，Push-T 30，Cube 1），除非另外注明。"
    "多个 seed 的格子是均值 ± 样本标准差（seed 数）；其余是 seed 0。")
add("")

# ---- 2. architectures --------------------------------------------------------------------------------
add("## 2. 网络结构：MLP / Transformer / Mamba")
add("")
add("### 2.1 TwoRoom 上的结构 × context × k（head 输入含 a_t，λ = 0.1）")
add("")
GRID = [(2, 1), (4, 1), (16, 1), (16, 2), (16, 3)]
ARCHS = ["mlp", "transformer", "mamba"]
gridhead = ["网络"] + [f"C = {c}，k = {k}" for c, k in GRID]
for title, key, fmt, pm in [("成功率（%）", "success", lambda x: f"{rnd(x, '1')}", True),
                            ("Effective rank", "rank", rank, False),
                            ("Policy loss（train）", "policy", loss, False),
                            ("Prediction loss（train）", "pred", sci, False)]:
    add(f"**{title}**")
    add("")
    add(table(gridhead, [[a] + [cell(family("TwoRoom", head=a, C=c, k=k), key, fmt, pm) for c, k in GRID] for a in ARCHS]))
    add("")
def batch_size(r):
    """256 everywhere except the C = 16 runs (64): read from the saved config.yaml of every architecture run on
    2026-11-17; the collector output, when present, must agree."""
    rule = 64 if r["C"] == 16 else 256
    b = extra(r, "loader.batch_size")
    assert b is None or int(b) == rule, (r["run"], b)
    return rule


def setup(c, k, fn):
    vals = {fn(family("TwoRoom", head=a, C=c, k=k)[0]) for a in ARCHS}
    assert len(vals) == 1, (c, k, vals)                               # the three architectures share the training setup
    return vals.pop()


clips = [setup(c, k, lambda r: r["steps"] * batch_size(r)) for c, k in GRID]
add("**训练设置**（三种网络相同）")
add("")
add(table(["", *gridhead[1:]], [
    ["训练步数"] + [f"{setup(c, k, lambda r: r['steps']):,}" for c, k in GRID],
    ["batch size"] + [f"{setup(c, k, batch_size)}" for c, k in GRID],
    ["训练片段总数（百万）"] + [f"{rnd(v / 1e6, '0.01')}" for v in clips],
    ["相对 C = 2"] + [f"{rnd(100 * v / clips[0], '1')} %" for v in clips]]))
add("")
add("参照：同一数据上的 inverse 成功率 100，effective rank 12.3，prediction loss 6.97e-5。"
    "MLP C = 2 一格有 3 个 seed（89 / 95 / 87），其余都是 seed 0。")
add("")
add("**C = 16 的三列不能和前两列直接比较。** 它们的 batch size 是 64（其余 256，读自每个运行保存的 config.yaml），"
    "而且窗口变长后数据里能取出的片段少得多。forward model 的 prediction loss 只用每个片段的第一个转移（h1），"
    "所以 C = 16 的模型训练 forward model 用到的转移只有 C = 2 的一小部分（上表最后一行）。"
    "成功率随 context 下降，可能来自 context 本身，也可能来自这两个训练设置上的差别，这批运行分不开。")
add("")
add("### 2.2 跨环境（k = 1，head 输入含 a_t）")
add("")
ARCH_ROWS = [("inverse（参照）", dict(reg="inverse", head="mlp", C=None, k=None, at=None), "inv"),
             ("MLP，C = 2", dict(head="mlp", C=2), "policy"),
             ("MLP，C = 4", dict(head="mlp", C=4), "policy"),
             ("Transformer，C = 2", dict(head="transformer", C=2), "policy"),
             ("Transformer，C = 4", dict(head="transformer", C=4), "policy"),
             ("Mamba，C = 2", dict(head="mamba", C=2), "policy"),
             ("Mamba，C = 4", dict(head="mamba", C=4), "policy")]
for title, key, fmt, pm in [("成功率（%）", "success", lambda x: f"{rnd(x, '1')}", True),
                            ("Effective rank", "rank", rank, False),
                            ("Action loss（train；inverse 一行是 inverse loss，其余是 policy loss）", None, loss, False),
                            ("Prediction loss（train）", "pred", sci, False)]:
    add(f"**{title}**")
    add("")
    add(table(["head"] + [ENV_ZH[e] for e in ENVS],
              [[lab] + [cell(family(e, **kw), key or lk, fmt, pm) for e in ENVS] for lab, kw, lk in ARCH_ROWS]))
    add("")
add("Transformer 和 Mamba 在 Reacher 和 Push-T 上只跑了 C = 4，Cube 上没有跑；C = 2 和 MLP 的 C = 4 只在 TwoRoom 上有。")
add("")

# ---- 3. inputs of the MLP head ------------------------------------------------------------------------
add("## 3. MLP head 的输入（k = 1）")
add("")
INPUT_ROWS = [("C = 2，含 a_t：π(z_t, z_{t+1}, a_t)", dict(C=2, at=True)),
              ("C = 2，无 a_t：π(z_t, z_{t+1})", dict(C=2, at=False)),
              ("C = 1：π(z_t) → a_t", dict(C=1, at=True)),
              ("Δz：π(z_{t+1} − z_t)", dict(C=2, at=False, delta=True)),
              ("goal final：π(z_t, z_{t+1}, z_g)", dict(C=2, at=False, G="final")),
              ("Δz + goal final", dict(C=2, at=False, delta=True, G="final"))]
ENVS5 = ["TwoRoom", "Reacher", "Reacher (expert)", "Push-T", "OGBench-Cube"]
for title, key, fmt, pm in [("成功率（%）", "success", lambda x: f"{rnd(x, '1')}", True),
                            ("Effective rank", "rank", rank, False),
                            ("Policy loss（train）", "policy", loss, False),
                            ("Prediction loss（train）", "pred", sci, False)]:
    add(f"**{title}**")
    add("")
    add(table(["head 的输入"] + [ENV_ZH[e] for e in ENVS5],
              [[lab] + [cell(family(e, **kw), key, fmt, pm) for e in ENVS5] for lab, kw in INPUT_ROWS]))
    add("")
add("Δz 和 goal final 两种输入后来成为论文里的四种 head；这里是它们在 10 月的第一批结果（多为 seed 0）。"
    "笔记的表里 C = 1 的 “a_t in” 一栏写的是 yes，但窗口只有一个 latent，没有动作可以输入。")
add("")

# ---- 4. other regularizers ---------------------------------------------------------------------------
add("## 4. 其他正则：inverse、inverse + policy、goal policy")
add("")
REG_ROWS = [("inverse：h(z_t, z_{t+1}) → a_t", dict(reg="inverse", head="mlp", C=None, k=None, at=None), [("inv", "inv")]),
            ("inverse + policy（MLP，C = 2，含 a_t）", dict(reg="inverse+policy", head="mlp", C=2, at=True), [("inv", "inv"), ("policy", "policy")]),
            ("goal policy，G = 2：π(z_t, z_{t+2}) → a_t", dict(reg="goal policy", head="mlp", C=None, k=None, at=None, G="2"), [("goal", "goal")]),
            ("goal policy，G = 3", dict(reg="goal policy", head="mlp", C=None, k=None, at=None, G="3"), [("goal", "goal")]),
            ("goal policy，G = 1, 2, 3", dict(reg="goal policy", head="mlp", C=None, k=None, at=None, G="1,2,3"), [("goal", "goal")])]
add("**成功率（%）**")
add("")
add(table(["正则"] + [ENV_ZH[e] for e in ENVS5], [[lab] + [succ(family(e, **kw)) for e in ENVS5] for lab, kw, _ in REG_ROWS]))
add("")
add("**Effective rank**")
add("")
add(table(["正则"] + [ENV_ZH[e] for e in ENVS5], [[lab] + [cell(family(e, **kw), "rank", rank) for e in ENVS5] for lab, kw, _ in REG_ROWS]))
add("")
add("**Action loss（train）**：inverse 是 inverse loss，goal policy 是 goal-policy loss，inverse + policy 一行是 “inverse loss / policy loss”。")
add("")
add(table(["正则"] + [ENV_ZH[e] for e in ENVS5],
          [[lab] + [" / ".join(cell(family(e, **kw), k_, loss) for k_, _ in keys) if family(e, **kw) else DASH for e in ENVS5]
           for lab, kw, keys in REG_ROWS]))
add("")
add("**Prediction loss（train）**")
add("")
add(table(["正则"] + [ENV_ZH[e] for e in ENVS5], [[lab] + [cell(family(e, **kw), "pred", sci) for e in ENVS5] for lab, kw, _ in REG_ROWS]))
add("")

# ---- 5. lambda sweeps --------------------------------------------------------------------------------
add("## 5. λ 扫描（MLP，C = 2，k = 1，含 a_t）")
add("")
for env in ["TwoRoom", "Push-T", "Reacher"]:
    lams = sorted({r["lam"] for r in RUNS if (r["env"], r["reg"], r["head"], r["delta"], r["C"], r["k"], r["at"], r["G"]) ==
                   (env, "policy", "mlp", False, 2, 1, True, DASH)}, key=float)
    add(f"**{ENV_ZH[env]}**（该环境的默认 λ = {LAMBDA[env]}）")
    add("")
    rows = []
    for lam in lams:
        f = family(env, lam=lam)
        rows.append([lam, succ(f), " / ".join(f"{rnd(r['success'], '1')}" for r in f), cell(f, "rank", rank), cell(f, "policy", loss), cell(f, "pred", sci)])
    add(table(["λ", "成功率（%）", "各 seed", "eff. rank", "policy loss", "pred loss"], rows))
    add("")

# ---- 6. extras ---------------------------------------------------------------------------------------
if EXTRA:
    add("## 6. 补充指标（不在笔记里）")
    add("")
    add("2026-11-17 从 DelftBlue 上这些运行的训练日志、保存的最终 latent 和 checkpoint 里只读地取出。"
        "val = held-out 上最后一次验证；latent std 和 fp32 的 effective rank 用保存的最终 latent 在单精度下计算"
        "（与日志同样的片段）；参数量是 head 本身的参数个数。相对预测误差 = val prediction loss / latent 方差；"
        "latent 尺度很小时（TwoRoom 上含 a_t 的 head、Reacher 上坍缩的模型）它包含 bfloat16 的舍入噪声，只能粗看。")
    add("")
    XCOLS = [("成功率", lambda f: succ(f)), ("eff. rank（logged）", lambda f: cell(f, "rank", rank)),
             ("eff. rank（fp32）", lambda f: xcell(f, "erank_fp32", rank)),
             ("latent std（fp32）", lambda f: xcell(f, "std_fp32", lambda x: f"{x:.3g}")),
             ("policy loss（train / val）", lambda f: f"{cell(f, 'policy', loss)} / {xcell(f, 'validate/policy_loss', loss)}"),
             ("pred loss（train / val）", lambda f: f"{cell(f, 'pred', sci)} / {xcell(f, 'validate/pred_loss', sci)}"),
             ("相对预测误差 %", lambda f: xcell(f, "rel_err", lambda x: ">100" if x > 100 else f"{x:.3g}")),
             ("head 参数量", lambda f: xcell(f, "params_policy_head", lambda x: f"{int(x):,}"))]
    add("### 6.1 TwoRoom 的结构网格")
    add("")
    add(table(["网络", "C", "k"] + [c for c, _ in XCOLS],
              [[a, c, k] + [fn(family("TwoRoom", head=a, C=c, k=k)) for _, fn in XCOLS] for a in ARCHS for c, k in GRID]))
    add("")
    add("### 6.2 Reacher 和 Push-T 上的三种结构")
    add("")
    add(table(["环境", "head"] + [c for c, _ in XCOLS],
              [[ENV_ZH[e], lab] + [fn(family(e, **kw)) for _, fn in XCOLS]
               for e in ["Reacher", "Push-T", "OGBench-Cube"]
               for lab, kw in [("MLP，C = 2", dict(head="mlp", C=2)), ("Transformer，C = 4", dict(head="transformer", C=4)),
                               ("Mamba，C = 4", dict(head="mamba", C=4))] if family(e, **kw)]))
    add("")
    add("### 6.3 MLP head 的输入")
    add("")
    add(table(["环境", "head 的输入"] + [c for c, _ in XCOLS],
              [[ENV_ZH[e], lab.split("：")[0]] + [fn(family(e, **kw)) for _, fn in XCOLS]
               for e in ENVS for lab, kw in INPUT_ROWS[:3] if family(e, **kw)]))
    add("")

unused = [r for r in RUNS if id(r) not in USED]
add(f"## {7 if EXTRA else 6}. 逐运行数据")
add("")
add(f"`early_heads_runs.csv`：笔记大表的全部 {len(RUNS)} 行（含运行名）" + ("，加上第 6 节的补充指标。" if EXTRA else "。")
    + f"上面各表用到其中 {len(RUNS) - len(unused)} 行；没有用到的 {len(unused)} 行是在 Leonardo 上重复训练的同名运行"
    + "（" + "、".join(f"`{r['run']}`" for r in unused) + "），同一 seed 取了 DelftBlue 的那一个。")

(HERE / "early_heads_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------------------------- Obsidian note
def std2(x):
    """Two significant digits; scientific notation below 0.001 (6.7e-7)."""
    if x >= 1e-3:
        return f"{x:.2g}"
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant}e{int(exp)}"


def cmp_cells(f, act="policy", params="params_policy_head"):
    """success | rank logged -> fp32 | latent std | action loss train / val | pred loss train / val | head parameters"""
    return [succ(f), f"{cell(f, 'rank', rank)} → {xcell(f, 'erank_fp32', rank)}", xcell(f, "std_fp32", std2),
            f"{cell(f, act, loss)} / {xcell(f, 'validate/' + ('inv_loss' if act == 'inv' else 'policy_loss'), loss)}",
            f"{cell(f, 'pred', sci)} / {xcell(f, 'validate/pred_loss', sci)}",
            xcell(f, params, lambda x: f"{int(x):,}")]


CMP_HEAD = ["成功率 %", "eff. rank（logged → fp32）", "latent std", "action loss（train / val）", "pred loss（train / val）", "head 参数量"]
INV = dict(reg="inverse", head="mlp", C=None, k=None, at=None)
ARCH_ZH = {"mlp": "MLP", "transformer": "Transformer", "mamba": "Mamba"}

# ---------------------------------------------------------------------------------------------- goal-conditioned heads (November 2026)
# Section 7 of the comparison note. goal_head_arch_runs.csv has one row per run with the columns of collect_head_arch.py.
GH_DATA = ["tworoom", "pusht", "cube", "reacher_mix", "reacher"]                     # the collector's dataset names, in table order
GH_ZH = {"tworoom": "TwoRoom", "pusht": "Push-T", "cube": "Cube", "reacher_mix": "Reacher 专家", "reacher": ENV_ZH["Reacher"]}
GH_ENV = {"tworoom": "TwoRoom", "pusht": "Push-T", "cube": "OGBench-Cube", "reacher": "Reacher"}   # the four original datasets
GH_PAPER = {"TwoRoom": "tworoom", "Push-T": "pusht", "Cube": "cube", "Reacher-PD": "reacher_mix", "Reacher-Random": "reacher"}
GH_GOAL_DIRECTED = GH_DATA[:4]                                                        # their mean is the "Mean" column of the paper
GH_HEADS = [("inverse", "inverse（参照）"), ("lewm", "LeWM（参照）"), ("mlp", "MLP"), ("transformer", "Transformer"), ("mamba", "Mamba")]
GH_NEW = ("transformer", "mamba")
GH_POLICY = ("mlp",) + GH_NEW
GH_EVALS = [("cem30", "CEM 30"), ("cem10", "CEM 10"), ("cem5", "CEM 5"), ("policy", "只执行 policy"),
            ("mean30", "warm-start 30"), ("mean10", "warm-start 10"), ("mean5", "warm-start 5")]
GH_CEM = [k for k, _ in GH_EVALS if k != "policy"]
SEQ_GOAL = "Transformer / Mamba，无 $a_t$，goal final（第 7 节）"                      # its row of the lambda audit (5.1)
# Parameters of the head itself for a 10-dimensional action block: PolicyModel(192, 10, use_action=False, context=2,
# arch=..., use_goal=True) of planning/module.py, counted on 2026-11-19. Cube has a 25-dimensional block: 3,855 more each.
GH_PARAMS = {"mlp": 216074, "transformer": 1417738, "mamba": 949836}
GH = defaultdict(list)                                                               # (dataset, head) -> one row per seed
PIPELINES = []                                                                       # (warm-start pipeline, old pipeline) cem30 of a run
LOG_SOURCE = {}                                                                      # (dataset, seed) of the MLP goal head -> where its log is
if Path(args.goal_heads).is_file():
    for r in csv.DictReader(open(args.goal_heads)):
        row = {k: (v if k in ("dataset", "head", "run", "weight", "eval_jobs") else float(v) if v != "" else None) for k, v in r.items()}
        row["seed"] = int(r["seed"])
        GH[(r["dataset"], r["head"])].append(row)
    for r in csv.DictReader(open(args.paper_heads)):
        if r["success_cem30"] and r["success_old_pipeline_cem30"]:
            PIPELINES.append((float(r["success_cem30"]), float(r["success_old_pipeline_cem30"])))
        if r["dataset"] not in GH_PAPER:
            continue
        if r["method"] == "LeWM":                                                     # SIGReg has no action head: plain CEM only
            GH[(GH_PAPER[r["dataset"]], "lewm")].append(
                dict(seed=int(r["seed"]), run=r["run"], **{k: float(r["success_" + k]) for k in ("cem30", "cem10", "cem5")}))
        elif r["method"] == "endpoints + goal":
            LOG_SOURCE[(GH_PAPER[r["dataset"]], int(r["seed"]))] = r["log_source"]
    for rows_ in GH.values():
        rows_.sort(key=lambda r: r["seed"])
HAVE_GH = bool(GH)


def gh(ds, head, key):
    """One column of one head on one dataset, over the seeds that have it."""
    return [r[key] for r in GH[(ds, head)] if r.get(key) is not None]


def gh_seeds(ds, head, key):
    return [r["seed"] for r in GH[(ds, head)] if r.get(key) is not None]


def gh_avg(head, key):
    """Mean over the four goal-directed datasets: one value per seed that has this eval on all four."""
    seeds = set.intersection(*[set(gh_seeds(ds, head, key)) for ds in GH_GOAL_DIRECTED])
    return [mean([r[key] for ds in GH_GOAL_DIRECTED for r in GH[(ds, head)] if r["seed"] == s]) for s in sorted(seeds)]


def spread(vals, q="1"):
    """One seed: the value. Several seeds: mean（min–max）."""
    if not vals:
        return DASH
    if len(vals) == 1:
        return f"{rnd(vals[0], q)}"
    return f"{rnd(mean(vals), '0.1')}（{rnd(min(vals), q)}–{rnd(max(vals), q)}）"


def off(v, ref):
    """Signed distance of a value from the range of the reference values; 0 inside the range."""
    return v - max(ref) if v > max(ref) else v - min(ref) if v < min(ref) else 0.0


def seed_span(seeds):
    seeds = sorted(seeds)
    if len(seeds) > 1 and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}–{seeds[-1]}"
    return "、".join(str(s) for s in seeds) or DASH


def pct1(x):
    """One decimal, without a negative zero (an explained variance of -0.04 % is 0.0)."""
    s = f"{rnd(x, '0.1')}"
    return "0.0" if s == "-0.0" else s


def goal_head_section(a):
    """Section 7 of the comparison note: MLP / Transformer / Mamba heads on (z_t, z_{t+1}, z_g) without action input.
    The numbers inside the sentences are computed from the rows of the tables, and the asserts state what the sentences
    claim: with other data this fails here instead of printing a sentence that is no longer true."""
    one = lambda ds, h, k: gh(ds, h, k)[0]                                            # the single run of a new head
    v = lambda ds, h, k: f"{rnd(one(ds, h, k), '1')}"
    rng = lambda ds, k: f"{rnd(min(gh(ds, 'mlp', k)), '1')}–{rnd(max(gh(ds, 'mlp', k)), '1')}"   # the MLP's seed range
    dev = lambda ds, h, k: off(one(ds, h, k), gh(ds, "mlp", k))                        # a new head against that range
    avg = {h: {k: gh_avg(h, k) for k, _ in GH_EVALS} for h, _ in GH_HEADS}
    weight = {}
    for ds in GH_DATA:
        assert [len(GH[(ds, h)]) for h, _ in GH_HEADS] == [5, 5, 5, 1, 1], ds          # 5 seeds of each reference, one run of each new head
        assert all(GH[(ds, h)][0]["seed"] == 0 for h in GH_NEW), ds
        steps = {r["step"] for h in ("inverse",) + GH_POLICY for r in GH[(ds, h)] if r["step"] is not None}
        assert len(steps) == 1, (ds, steps)                                           # same number of training steps for every head
        w = {f"{float(r['weight']):g}" for h in GH_POLICY for r in GH[(ds, h)]}
        assert len(w) == 1, (ds, w)                                                    # one policy weight per dataset: the default
        weight[ds] = w.pop()
    assert weight["reacher"] == weight["reacher_mix"]
    both = [abs(x - y) for x, y in PIPELINES]

    a("## 7. 无 $a_t$、带目标的 Transformer / Mamba head（2026-11-18 / 19）")
    a("")
    a("第 2 节的三种结构都输入 $a_t$、不带目标。这一节把 Transformer 和 Mamba 换到论文用的输入上："
      "$a_{t+1} = \\pi(z_t, z_{t+1}, z_g)$，不输入 $a_t$，$z_g$ 是片段最后一帧的 latent（goal final）。")
    a("")
    a("- **head**：MLP 是论文里的 head（三个 latent 拼接后过两层隐藏层，宽 256）。Transformer 和 Mamba 把 $z_g$ 作为窗口之前的一个 token"
      "（有自己的投影层），token 顺序是 $(z_g, z_t, z_{t+1})$，2 层、宽 256，在 $z_{t+1}$ 处读出。"
      f"head 参数量：MLP {GH_PARAMS['mlp']:,}，Transformer {GH_PARAMS['transformer']:,}，Mamba {GH_PARAMS['mamba']:,}"
      "（Cube 的动作块是 25 维，各多 3,855）。")
    a("- **训练**：与第 1 节相同（h1，10 个 epoch，lr 1e-4，bfloat16；batch 256），同一数据集上各 head 的训练步数相同。"
      f"每个数据集只用默认的 λ：TwoRoom {weight['tworoom']}，Push-T {weight['pusht']}，Cube {weight['cube']}，两个 Reacher 数据集 {weight['reacher']}。"
      "Transformer 和 Mamba 只有 seed 0；MLP 和 inverse 有 5 个 seed。")
    a("- **数据集**：前面的四个，加上 Reacher 专家（`reacher_mix`，论文里的 Reacher-PD）：一半 episode 是 PD 控制器走随机路点，一半是均匀随机动作。")
    a("- **成功率**：warm-start 评测流程，任务设置与第 1 节相同（100 个任务，目标在 25 步之后，预算 50 步）。"
      "七种评测：普通 CEM 迭代 30 / 10 / 5 次；只执行 policy（不搜索）；warm-start（head 输出的动作序列作为 CEM 的初始均值）迭代 30 / 10 / 5 次。"
      f"“CEM 30” 一列对应前面各节的成功率：两种流程都评测过的 {len(both)} 个运行上，两者平均相差 {rnd(mean(both), '0.1')} 个点（最大 {rnd(max(both), '1')}）。")
    a("- **格子**：多个 seed 的格子是均值，括号里是 seed 间的最小–最大值；没有括号的是 seed 0。"
      "LeWM（SIGReg，5 个 seed）和 inverse 都是参照，它们没有 policy head，所以后四列是 “–”。")
    a("")
    a("### 7.1 成功率（%）")
    a("")
    a(table(["环境", "head"] + [lab for _, lab in GH_EVALS],
            [[GH_ZH[ds], lab] + [spread(gh(ds, h, k)) for k, _ in GH_EVALS] for ds in GH_DATA for h, lab in GH_HEADS],
            align=["l", "l"] + ["c"] * len(GH_EVALS)))
    a("")
    a("### 7.2 四个目标导向数据集的平均")
    a("")
    a("TwoRoom、Push-T、Cube、Reacher 专家的平均：先在每个 seed 内对四个数据集求平均，括号里是各 seed 平均值的最小–最大。")
    a("")
    a(table(["head"] + [lab for _, lab in GH_EVALS], [[lab] + [spread(avg[h][k], "0.1") for k, _ in GH_EVALS] for h, lab in GH_HEADS]))
    a("")

    # --- planning with CEM
    avg_dev = max(abs(off(avg[h][k][0], avg["mlp"][k])) for h in GH_NEW for k in GH_CEM)
    cells = [(abs(dev(ds, h, k)), ds, k, h, dev(ds, h, k)) for h in GH_NEW for ds in GH_GOAL_DIRECTED for k in GH_CEM]
    outside = sorted((c for c in cells if c[0]), reverse=True)
    n_out = {h: sum(1 for c in outside if c[3] == h) for h in GH_NEW}
    assert {c[1:4] for c in outside[:3]} == {("pusht", "cem10", "transformer"), ("cube", "cem10", "transformer"), ("cube", "cem10", "mamba")}
    assert outside[2][0] > outside[3][0] and any(c[4] > 0 for c in outside) and any(c[4] < 0 for c in outside)
    assert dev("pusht", "transformer", "cem10") > 0 and all(dev("cube", h, "cem10") < 0 for h in GH_NEW)
    a("- **用 CEM 规划（普通或 warm-start）时，四个数据集平均起来三种结构没有超出 seed 波动的差别。** "
      f"7.2 里两种新 head 带 CEM 的六列都在 MLP 各 seed 的范围内，或最多超出 {rnd(avg_dev, '0.1')} 个点。"
      f"分数据集看（7.1），两种新 head 在四个目标导向数据集的这六列上共 {len(cells)} 格，其中 {len(outside)} 格在 MLP 五个 seed 的最小–最大范围之外"
      f"（Transformer {n_out['transformer']}，Mamba {n_out['mamba']}），有高有低；即使分布相同，一个新的运行落在 5 个样本的范围之外的概率也有 1/3。"
      f"偏离最大的两处：Push-T 的 CEM 10，Transformer {v('pusht', 'transformer', 'cem10')} 对 MLP 的 {rng('pusht', 'cem10')}；"
      f"Cube 的 CEM 10，Transformer {v('cube', 'transformer', 'cem10')}、Mamba {v('cube', 'mamba', 'cem10')} 对 MLP 的 {rng('cube', 'cem10')}。")

    # --- the head alone
    tp, mp = avg["transformer"]["policy"][0], avg["mamba"]["policy"][0]
    assert tp > max(avg["mlp"]["policy"]) and off(mp, avg["mlp"]["policy"]) == 0
    assert all(dev("cube", h, "policy") > 0 for h in GH_NEW)
    assert dev("reacher_mix", "transformer", "policy") > 0 and dev("reacher_mix", "mamba", "policy") == 0
    assert all(dev(ds, h, "policy") == 0 for ds in ("tworoom", "pusht") for h in GH_NEW)
    assert all(one("cube", h, "policy") >= max(one("cube", h, k) for k in ("cem30", "cem10", "cem5")) for h in GH_NEW)
    a("- **只执行 policy 时 Transformer 更高。** "
      f"四个数据集平均 {rnd(tp, '0.1')}，MLP 是 {spread(avg['mlp']['policy'], '0.1')}。"
      f"差距来自 Cube（{v('cube', 'transformer', 'policy')} 对 {rng('cube', 'policy')}）"
      f"和 Reacher 专家（{v('reacher_mix', 'transformer', 'policy')} 对 {rng('reacher_mix', 'policy')}）；"
      f"Push-T（{v('pusht', 'transformer', 'policy')}）在 MLP 的范围内，TwoRoom 已经饱和。"
      f"Mamba 平均 {rnd(mp, '0.1')}，只在 Cube 上高于 MLP 的范围（{v('cube', 'mamba', 'policy')}）。"
      "Cube 上两种新 head 不搜索就达到或超过自己的普通 CEM。")

    # --- the random-action data
    dead_succ = [x for h in GH_POLICY for k in GH_CEM for x in gh("reacher", h, k)]
    assert max(dead_succ) < 20 and all(x == 1 for h in GH_POLICY for x in gh("reacher", h, "policy"))
    assert all(loss(mean(gh("reacher", h, "val_head_loss"))) == "1.000" for h in GH_POLICY)
    a(f"- **{GH_ZH['reacher']}上三种 head 都坍缩。** 带 CEM 的评测都在 {rnd(min(dead_succ), '1')}–{rnd(max(dead_succ), '1')} 之间，只执行 policy 都是 1；"
      f"CEM 30 下 inverse 是 {rnd(mean(gh('reacher', 'inverse', 'cem30')), '0.1')}，LeWM 是 {rnd(mean(gh('reacher', 'lewm', 'cem30')), '0.1')}。"
      "带上目标、换结构都不改变第 5.3 节的结论：这份数据上 policy head 学不到东西（7.3 里它们的 val action loss 都是 1.000）。")
    a("")

    a("### 7.3 诊断量")
    a("")
    a("action loss、pred loss 和 logged 的 effective rank 来自训练日志（train 是最后一个 epoch 的均值，val 是 held-out 上最后一次验证）；"
      "fp32 的 effective rank、latent std 和相对预测误差（val pred loss / latent 方差）用保存的最终 latent 在单精度下计算。"
      "可解释的动作方差 = 1 − val action loss。inverse 一行是 inverse loss，它预测的是 $a_t$，和 policy head 的目标 $a_{t+1}$ 不同。"
      "多个 seed 的格子是均值，只平均有这个量的 seed。")
    a("")
    diag = lambda ds, h, k, fmt: fmt(mean(gh(ds, h, k))) if gh(ds, h, k) else DASH
    rows = []
    for ds in GH_DATA:
        for h, lab in GH_HEADS:
            if h == "lewm":                                                           # no action head and no row in the collector's table
                continue
            rows.append([GH_ZH[ds], lab, seed_span(gh_seeds(ds, h, "val_head_loss")),
                         f"{diag(ds, h, 'train_head_loss', loss)} / {diag(ds, h, 'val_head_loss', loss)}",
                         diag(ds, h, "explained_variance", lambda x: pct1(100 * x)),
                         f"{diag(ds, h, 'effective_rank_logged', rank)} → {diag(ds, h, 'effective_rank_fp32', rank)}",
                         diag(ds, h, "std_fp32", std2),
                         f"{diag(ds, h, 'train_pred_loss', sci)} / {diag(ds, h, 'val_pred_loss', sci)}",
                         diag(ds, h, "rel_pred_error_pct", lambda x: ">100" if x > 100 else f"{x:.3g}")])
    a(table(["环境", "head", "有日志的 seed", "action loss（train / val）", "可解释的动作方差 %", "eff. rank（logged → fp32）", "latent std",
             "pred loss（train / val）", "相对预测误差 %"], rows, align=["l", "l"] + ["c"] * 7))
    a("")

    # --- fit of the head
    ev = lambda ds, h: 100 * mean(gh(ds, h, "explained_variance"))
    up = [ev(ds, h) - ev(ds, "mlp") for ds in ("pusht", "cube") for h in GH_NEW]
    down = [ev("reacher_mix", "mlp") - ev("reacher_mix", h) for h in GH_NEW]
    same = [ev("tworoom", h) for h in GH_POLICY]
    assert min(up) > 0 and min(down) > 0 and max(same) - min(same) < 0.2
    assert all(one("reacher_mix", h, "train_head_loss") < min(gh("reacher_mix", "mlp", "train_head_loss")) for h in GH_NEW)
    a("- **拟合**：Push-T 和 Cube 上两种序列 head 的可解释动作方差比 MLP 高 "
      f"{rnd(min(up), '0.1')}–{rnd(max(up), '0.1')} 个点；TwoRoom 上三者相同（{pct1(min(same))}–{pct1(max(same))}）；"
      f"Reacher 专家上低 {rnd(min(down), '0.1')}–{rnd(max(down), '0.1')} 个点，虽然 train loss 更低。")

    # --- latent scale, relative prediction error, rank: on the four datasets without collapse
    ratio = lambda k, h: [one(ds, h, k) / mean(gh(ds, "mlp", k)) for ds in GH_GOAL_DIRECTED]
    scale = ratio("std_fp32", "transformer") + ratio("std_fp32", "mamba")
    assert max(scale) < 1
    a("- **latent 尺度**：四个没有坍缩的数据集上，序列 head 的 latent std 是 MLP 的 "
      f"{min(scale):.2f}–{max(scale):.2f} 倍。第 2 节含 $a_t$ 时也是这样（要点 2），但这里它没有伴随用 CEM 规划时平均成功率的差距（7.2）。")
    e_t, e_m = ratio("rel_pred_error_pct", "transformer"), ratio("rel_pred_error_pct", "mamba")
    plain = ("cem30", "cem10", "cem5")
    assert min(e_t + e_m) > 1
    assert all(one(ds, h, "val_pred_loss") < mean(gh(ds, "mlp", "val_pred_loss")) for ds in GH_GOAL_DIRECTED for h in GH_NEW)
    assert all(dev(ds, h, k) >= 0 for ds in ("pusht", "reacher_mix") for h in GH_NEW for k in plain)
    assert all(dev("cube", h, "cem30") == 0 and dev("cube", h, "cem10") < 0 and dev("cube", h, "cem5") < 0 for h in GH_NEW)
    a("- **相对预测误差**：序列 head 更大，Transformer 是 MLP 的 "
      f"{min(e_t):.1f}–{max(e_t):.1f} 倍，Mamba 是 {min(e_m):.1f}–{max(e_m):.1f} 倍。"
      "普通 CEM 的成功率没有随之一致地变化：Push-T 和 Reacher 专家上没有低于 MLP 的范围，Cube 上 CEM 30 在范围内、CEM 10 / 5 更低。"
      "pred loss 的绝对值更低只反映 latent 尺度更小。")
    rk = lambda ds, h: dev(ds, h, "effective_rank_fp32")
    assert all(rk(ds, "mamba") > 0 for ds in GH_GOAL_DIRECTED)
    assert all(rk(ds, "transformer") > 0 for ds in ("pusht", "reacher_mix")) and all(rk(ds, "transformer") == 0 for ds in ("tworoom", "cube"))
    a("- **effective rank（fp32）**：Mamba 在四个数据集上都高于 MLP 的全部 seed；Transformer 在 Push-T 和 Reacher 专家上更高，"
      "在 TwoRoom 和 Cube 上在 MLP 的范围内。TwoRoom 上 logged 的值是 bfloat16 的舍入噪声（要点 4）。")

    # --- the collapsed runs and the gaps
    dead = [x for h in GH_POLICY for x in gh("reacher", h, "std_fp32")]
    assert max(dead) < 1e-3
    a(f"- **{GH_ZH['reacher']}**：三种 head 的 latent std 都在 {std2(min(dead))}–{std2(max(dead))}"
      f"（inverse {std2(mean(gh('reacher', 'inverse', 'std_fp32')))}），已经坍缩。"
      "这时的 effective rank 和相对预测误差没有意义（logged 的 rank 是 bfloat16 噪声），只列出，不解读。")
    logs = {(ds, h): gh_seeds(ds, h, "val_head_loss") for ds in GH_DATA for h in ("inverse",) + GH_POLICY}
    fp32 = {key: gh_seeds(*key, "std_fp32") for key in logs}
    assert {k: s for k, s in logs.items() if s != [r["seed"] for r in GH[k]]} == {("pusht", "mlp"): [3, 4], ("cube", "mlp"): [1, 2, 3, 4]}
    assert {k: s for k, s in fp32.items() if s != logs[k]} == {("pusht", "inverse"): [1, 2, 3, 4], ("reacher", "inverse"): [1, 2, 3, 4]}
    assert all(LOG_SOURCE[k].startswith("Leonardo") for k in [("pusht", 0), ("pusht", 1), ("pusht", 2), ("cube", 0)])
    a("- **缺口**：MLP 在 Push-T 上的 seed 0–2 和 Cube 上的 seed 0 是在 Leonardo 上训练的，DelftBlue 上没有它们的训练日志，"
      "这两行的诊断量来自其余的 seed。"
      f"inverse 在 Push-T 和 {GH_ZH['reacher']}上的 seed 0 没有保存最终 latent，这两行单精度的量来自 seed 1–4。")
    a("")

    a("### 7.4 和前面各节的关系")
    a("")
    old = lambda env, **kw: succ(family(env, **kw)).split(" ±")[0]                    # success in the earlier sections (old pipeline)
    a("- **与第 2 节相比**（含 $a_t$、无目标，旧评测流程）：TwoRoom 上 Transformer / Mamba 从 "
      f"{old('TwoRoom', head='transformer', C=2)} / {old('TwoRoom', head='mamba', C=2)} 到 {v('tworoom', 'transformer', 'cem30')} / {v('tworoom', 'mamba', 'cem30')}，"
      f"Push-T 上从 {old('Push-T', head='transformer', C=4)} / {old('Push-T', head='mamba', C=4)} 到 {v('pusht', 'transformer', 'cem30')} / {v('pusht', 'mamba', 'cem30')}；"
      f"{GH_ZH['reacher']}上仍然坍缩（{old('Reacher', head='transformer', C=4)} / {old('Reacher', head='mamba', C=4)} 到 "
      f"{v('reacher', 'transformer', 'cem30')} / {v('reacher', 'mamba', 'cem30')}）。"
      "同时变了三件事：去掉 $a_t$、加上目标、Push-T 和 Reacher 上 C 从 4 变成 2，这组运行分不开各自的贡献。"
      f"MLP 在同样的变化下也在提高（Push-T：含 $a_t$ {old('Push-T', C=2, at=True)}，无 $a_t$ {old('Push-T', C=2, at=False)}，"
      f"再加目标 {rnd(mean(gh('pusht', 'mlp', 'cem30')), '0.1')}）。")
    a("- **第 5.3 节 “Transformer / Mamba 不如 MLP” 一行**：在论文用的输入上没有出现这个差距。")
    a("- **仍然只有一个 λ 和一个 seed**（要点 1、5）：以上是观察，不是结论。")
    a("")


def lambda_audit():
    """Which regularizer weights each kind of head was trained with (h1, the four original datasets), from the saved configs."""
    data = {"tworoom": "TwoRoom", "reacher": "Reacher", "pusht_expert": "Push-T", "cube_single_expert": "OGBench-Cube"}
    order = ["MLP，C = 2，k = 1，含 $a_t$", "MLP，C = 2，k = 1，无 $a_t$", "MLP，C = 1", "MLP，C = 4 / 16（k = 1–3），含 $a_t$",
             "Transformer（含 $a_t$）", "Mamba（含 $a_t$）", "Δz、goal final、Δz + goal final（MLP，无 $a_t$）",
             "inverse + policy（MLP，C = 2，含 $a_t$）", "goal policy（G = 2 / 3 / 1,2,3）", "inverse", "LeWM（SIGReg）"]
    seen = defaultdict(set)
    n = 0
    for r in csv.DictReader(open(args.configs)):
        env = data.get(r["data.dataset.name"].replace("_train", ""))
        if r["run"].startswith("_") or r["wm.history_size"] != "1" or env is None:
            continue
        n += 1
        fl = lambda k: float(r[k] or 0)
        inv, sig, pol, gp = fl("loss.inverse.weight"), fl("loss.sigreg.weight"), fl("loss.policy.weight"), fl("loss.goal_policy.weight")
        C, k, arch = int(r["loss.policy.context"] or 2), int(r["loss.policy.num_future"] or 1), r["loss.policy.arch"]
        at = r["loss.policy.use_action"] == "True" and C > 1
        goal = r["loss.policy.goal_source"] == "final" or fl("loss.policy.goal_offset") > 0
        if gp:
            fam, lam = order[8], f"{gp:g}"
        elif inv and pol:
            fam, lam = order[7], f"{inv:g} / {pol:g}"
        elif inv:
            fam, lam = order[9], f"{inv:g}"
        elif sig:
            fam, lam = order[10], f"{sig:g}"
        elif goal and arch in ("transformer", "mamba", "gru"):               # the heads of section 7, once the configs are read again
            fam, lam = SEQ_GOAL, f"{pol:g}"
        elif r["loss.policy.input"] == "delta" or goal:
            fam, lam = order[6], f"{pol:g}"
        elif arch == "mlp" and C == 2 and k == 1:
            fam, lam = order[0 if at else 1], f"{pol:g}"
        elif arch == "mlp" and C == 1:
            fam, lam = order[2], f"{pol:g}"
        elif arch == "mlp":
            fam, lam = order[3], f"{pol:g}"
        else:
            fam, lam = order[4 if arch == "transformer" else 5], f"{pol:g}"
        seen[(fam, env)].add(lam)
    in_configs = any(fam == SEQ_GOAL for fam, _ in seen)                     # False while the configs file predates these runs
    for (ds, head), runs in GH.items():                                      # their weights, from the collector's table of section 7
        if head in GH_NEW and ds in GH_ENV:
            seen[(SEQ_GOAL, GH_ENV[ds])].update(f"{float(r['weight']):g}" for r in runs)
    rows = []
    for fam in order[:7] + [SEQ_GOAL] + order[7:]:
        cells = []
        for env in ENVS:
            lams = sorted(seen.get((fam, env), []), key=lambda x: float(x.split(" / ")[0]))
            cells.append(DASH if not lams else lams[0] if len(lams) == 1 else "**扫过**：" + "、".join(lams))
        rows.append([fam] + cells)
    return rows, n, in_configs


def obsidian_note():
    N = []
    a = N.append
    tw = lambda **kw: family("TwoRoom", **kw)
    s_tf, s_mb, s_mlp = (xcell(tw(head=h, C=2), "std_fp32", std2) for h in ("transformer", "mamba", "mlp"))
    s_inv = xcell(tw(**INV), "std_fp32", std2)
    low = family("TwoRoom", lam="0.01")
    s_low = sorted(extra(r, "std_fp32") for r in low)
    grid_std = [extra(r, "std_fp32") for h in ARCHS for c, k in GRID for r in tw(head=h, C=c, k=k)]
    grid_rank = [extra(r, "erank_fp32") for h in ARCHS for c, k in GRID for r in tw(head=h, C=c, k=k)]
    lam_rows, n_cfg, seq_goal_in_configs = lambda_audit()

    a("# 不同 policy head 的对比（9–10 月的早期结果）")
    a("")
    a(f"来源：[[9 Oct with Yanbo]] 里 “SMWM: policy vs inverse regularizer” 的大表（{len(RUNS)} 个运行；数值未改，只是按 head 类型重排）。"
      "单精度的 effective rank、latent std、val loss、head 参数量和每个运行的 batch size 不在那张表里，"
      "是 2026-11-17 从 DelftBlue 上这些运行的日志、保存的最终 latent、checkpoint 和 config 里读出来的。"
      "第 7 节是 2026-11-18 / 19 补充的运行（无 $a_t$、带目标的 Transformer / Mamba head），"
      "数值来自仓库的 `planning/analysis/policy_heads/goal_head_arch_runs.csv`。"
      "本页由仓库的 `planning/analysis/policy_heads/make_early_summary.py --obsidian` 生成，不要手改表格。")
    a("")
    a("## 要点")
    a("")
    a("1. **λ 只对一种 head 扫过。** 只有 “MLP，C = 2，k = 1，含 $a_t$” 扫过 λ（TwoRoom 10 个值，Push-T 7 个，Reacher 4 个）。"
      "Transformer、Mamba、C = 4 / 16、k = 2 / 3、无 $a_t$、C = 1、inverse + policy、goal policy 都只在一个 λ 上训练，"
      "这个 λ 是 SMWM 论文给 inverse head 用的权重（第 5 节）。所以 “Transformer / Mamba 不如 MLP”“加长 context 没用” 现在都不能当结论。")
    a(f"2. **有迹象表明 Transformer 和 Mamba 需要更大的 λ。** TwoRoom 上 C = 2、λ = 0.1 时，它们的 latent 标准差是 {s_tf} 和 {s_mb}，MLP 是 {s_mlp}，inverse 是 {s_inv}。"
      f"MLP 要把 λ 降到 0.01 才落到这个尺度（{std2(s_low[0])}–{std2(s_low[-1])}），而那时它的成功率是 "
      + " / ".join(f"{rnd(r['success'], '1')}" for r in low) + "，seed 之间差别很大。这只是迹象，要靠扫描来验证。")
    a("3. **C = 16 另有两个混杂因素。** 这九个运行的 batch size 是 64（其余 256），训练用到的片段只有 C = 2 的 10–19 %，"
      "而 forward model 的 prediction loss 只用每个片段的第一个转移。它们和 C = 2 / 4 不能直接比。")
    a(f"4. **TwoRoom 上日志里的 effective rank 不可用。** 这些 head 的 latent 尺度很小（标准差 {std2(min(grid_std))}–{std2(max(grid_std))}），"
      f"bfloat16 的舍入噪声把 logged rank 抬到 27–126；用保存的 latent 在单精度下重算，15 个结构的 rank 都在 {rank(min(grid_rank))}–{rank(max(grid_rank))}。"
      "Push-T 和 Cube 上 latent 尺度大，两种算法一致。")
    a("5. **多数格子是单个 seed。** 单 seed 约 ±5 个点；TwoRoom 上 λ < 0.1 时同一设置的三个 seed 能差 40 个点以上。")
    a("6. **LeWAM 已经绕开了其中几个问题**（第 6 节）：它的 policy head 是带目标的 Transformer flow head，同样四个任务上是 100 / 100 / 98.7 / 100；"
      "但它保留 SIGReg 来防坍缩，没有回答 “只靠 policy head 能不能防坍缩”。")
    g30, gpol = (lambda h: gh_avg(h, "cem30")), (lambda h: gh_avg(h, "policy"))
    a("7. **在论文用的输入上（无 $a_t$、带目标），三种结构用 CEM 规划时平均没有超出 seed 波动的差别**（第 7 节，2026-11-18 / 19 的补充）。"
      f"四个目标导向数据集的平均（CEM 30）：Transformer {spread(g30('transformer'), '0.1')}，Mamba {spread(g30('mamba'), '0.1')}，"
      f"MLP {spread(g30('mlp'), '0.1')}。只执行 policy 时 Transformer 更高（{spread(gpol('transformer'), '0.1')} 对 MLP 的 {rnd(mean(gpol('mlp')), '0.1')}）。"
      "随机动作的 Reacher 上三种 head 都坍缩。新 head 只有默认 λ 和 seed 0。")
    a("")
    a("## 1. 记号与设置")
    a("")
    a("- **policy head**：从 C 个 latent 的窗口（含 $a_t$ 时还有窗口内的动作）预测窗口最后一个 latent 处开始的 k 个动作。"
      "C = 2、k = 1 是 $a_{t+1} = \\pi(z_t, z_{t+1}[, a_t])$；C = 1 是 $a_t = \\pi(z_t)$。")
    a("- **网络**：MLP 把窗口拼接后过两层隐藏层（宽 256）；Transformer 和 Mamba 把每个时刻的 latent（加动作）当作一个 token，过 2 层（宽 256），取最后一个 token。")
    a("- **训练**：h1（forward model 只看一帧），10 个 epoch，lr 1e-4，bfloat16；λ 是 policy loss 的权重。")
    a("- **成功率**：旧评测流程，100 个任务，目标在 25 步之后，预算 50 步，CEM 30 次迭代。没标 seed 数的都是 seed 0；“±” 后括号里是 seed 数。")
    a("- **loss**：train 是最后一个 epoch 的均值，val 是 held-out 上最后一次验证。action loss 是标准化动作上的 MSE（1 = 只能输出均值）；inverse 一行是 inverse loss。")
    a("- **effective rank**：logged 是日志里的值（bfloat16）；fp32 是用保存的最终 latent 在单精度下重算的值。**latent std** 是每一维标准差的均值（单精度）。“–” 表示这个运行没有保存最终 latent；多个 seed 的格子里，单精度的量只平均有 latent 的 seed。")
    a("")
    a("## 2. 网络结构：MLP / Transformer / Mamba")
    a("")
    a("### 2.1 TwoRoom（λ = 0.1，head 输入含 $a_t$）")
    a("")
    a("成功率一览：")
    a("")
    a(table(["网络"] + [f"C = {c}，k = {k}" for c, k in GRID],
            [[ARCH_ZH[h]] + [succ(tw(head=h, C=c, k=k)) for c, k in GRID] for h in ARCHS]))
    a("")
    a("全部指标：")
    a("")
    rows = [["inverse（参照）", DASH, DASH] + cmp_cells(tw(**INV), "inv", "params_inverse_head")
            + [f"{batch_size(tw(**INV)[0])}", f"{rnd(tw(**INV)[0]['steps'] * batch_size(tw(**INV)[0]) / 1e6, '0.01')}"]]
    for h in ARCHS:
        for c, k in GRID:
            f = tw(head=h, C=c, k=k)
            rows.append([ARCH_ZH[h], c, k] + cmp_cells(f) + [f"{batch_size(f[0])}", f"{rnd(f[0]['steps'] * batch_size(f[0]) / 1e6, '0.01')}"])
    a(table(["head", "C", "k"] + CMP_HEAD + ["batch", "训练片段（百万）"], rows))
    a("")
    a("- context 从 2 到 4，policy loss 从 0.95 降到 0.93；到 16 时 train 降到 0.92，val 仍是 0.93–0.94。成功率一路下降，"
      "但 C = 16 的三列同时换了 batch size 和训练片段数（要点 3）。")
    a("- 三种结构在 C = 2 时的 policy loss 相同（0.951–0.952），差别在 latent 的尺度：Transformer 和 Mamba 只有 MLP 的三分之一。")
    a("- 所有 policy head 的 latent 尺度都远小于 inverse。")
    a("")
    a("### 2.2 Reacher、Push-T、Cube（k = 1，含 $a_t$）")
    a("")
    rows = []
    for e in ["Reacher", "Push-T", "OGBench-Cube"]:
        rows.append([ENV_ZH[e], "inverse（参照）"] + cmp_cells(family(e, **INV), "inv", "params_inverse_head"))
        for lab, kw in [("MLP，C = 2", dict(head="mlp", C=2)), ("Transformer，C = 4", dict(head="transformer", C=4)), ("Mamba，C = 4", dict(head="mamba", C=4))]:
            if family(e, **kw):
                rows.append([ENV_ZH[e], lab] + cmp_cells(family(e, **kw)))
    a(table(["环境", "head"] + CMP_HEAD, rows, align=["l", "l"] + ["c"] * len(CMP_HEAD)))
    a("")
    a("- Transformer 和 Mamba 在这三个环境上只跑了 C = 4，Cube 上没有跑；MLP 只有 C = 2。所以这里结构和 context 是一起变的。")
    a("- Push-T 上 Transformer 和 Mamba 的 policy loss 更低（0.096、0.098 对 0.136），成功率却更低（25、17 对 37）。")
    a("- Reacher 的数据是均匀随机动作，三种结构的 policy loss 都约为 1，成功率在 10 左右；Transformer 和 Mamba 的 latent 已经坍缩（标准差 5e-4 和 1.3e-3），MLP 没有保存 latent。")
    a("")
    a("## 3. MLP head 的输入（k = 1）")
    a("")
    rows = []
    for e in ENVS:
        for lab, kw in [("C = 2，含 $a_t$", dict(C=2, at=True)), ("C = 2，无 $a_t$", dict(C=2, at=False)), ("C = 1：$\\pi(z_t) \\to a_t$", dict(C=1, at=True))]:
            rows.append([ENV_ZH[e], lab] + cmp_cells(family(e, **kw)))
    a(table(["环境", "head 的输入"] + CMP_HEAD, rows, align=["l", "l"] + ["c"] * len(CMP_HEAD)))
    a("")
    a("- 去掉 $a_t$ 是这批实验里最大的单项提升：Push-T 37 → 65.3，Cube 69 → 78.0，TwoRoom 90.3 → 100。")
    a("- 大表里 C = 1 的 “a_t in” 一栏写的是 yes，但这些运行的配置是 `use_action: false`，一个 latent 的窗口里也没有动作可输入。")
    a("- Δz、goal final 这两种输入（后来论文里的四种 head）不在这里，见 [[9 Oct with Yanbo]] 的 “All environments: Δz and goal” 表。")
    a("")
    a("## 4. 其他正则：inverse + policy、goal policy")
    a("")
    rows = []
    for e in ["TwoRoom", "Reacher", "Push-T"]:
        for lab, kw, keys in [("inverse", INV, ["inv"]),
                              ("inverse + policy（MLP，C = 2，含 $a_t$）", dict(reg="inverse+policy", head="mlp", C=2, at=True), ["inv", "policy"]),
                              ("goal policy，G = 2：$a_t = \\pi(z_t, z_{t+2})$", dict(reg="goal policy", head="mlp", C=None, k=None, at=None, G="2"), ["goal"]),
                              ("goal policy，G = 3", dict(reg="goal policy", head="mlp", C=None, k=None, at=None, G="3"), ["goal"]),
                              ("goal policy，G = 1, 2, 3", dict(reg="goal policy", head="mlp", C=None, k=None, at=None, G="1,2,3"), ["goal"])]:
            f = family(e, **kw)
            rows.append([ENV_ZH[e], lab, succ(f), cell(f, "rank", rank), " / ".join(cell(f, k_, loss) for k_ in keys), cell(f, "pred", sci)])
    a(table(["环境", "正则", "成功率 %", "eff. rank（logged）", "action loss（train）", "pred loss（train）"], rows, align=["l", "l", "c", "c", "c", "c"]))
    a("")
    a("inverse + policy 一行的 action loss 是 “inverse loss / policy loss”。只要 head 能看到未来的状态（inverse、goal policy），成功率就回到 inverse 的水平。这些也都只有一个 λ。")
    a("")
    a("## 5. λ 扫过没有")
    a("")
    a("### 5.1 每种 head 用过的 λ")
    a("")
    a(f"来自 DelftBlue 上每个运行保存的 `config.yaml`（h1、四个原始数据集，共 {n_cfg} 个运行）。Leonardo 上的运行（`planning/results.csv` 里 22 个）也都只用默认的 λ。"
      + ("" if seq_goal_in_configs else "“Transformer / Mamba，无 $a_t$，goal final” 一行是第 7 节的运行，训练晚于这次读取，λ 取自 `goal_head_arch_runs.csv`。"))
    a("")
    a(table(["head / 正则"] + [ENV_ZH[e] for e in ENVS], lam_rows))
    a("")
    a("- 默认值 0.1 / 5 / 30 / 1 是 SMWM 论文给 **inverse** head 的各环境权重（论文表 2）；论文另有 inverse 权重在 {0, 0.1, 0.3, 1, 3, 10, 30, 100} 上的扫描"
      "（图 7，脚本在仓库 `planning/experiments/lambda_sweep/`）。我们自己没有重扫 inverse，其他 head 都直接沿用了这个值。")
    a("- 无 $a_t$ 的两种 head（C = 2 和 C = 1）在 2026-10-07 计划过 λ × {0.1, 10} 的扫描，10-08 在作业开始前取消了（`planning/HANDOFF.md` 第 5 节），之后没有再提交。")
    a("- Cube 上没有任何 head 扫过 λ。")
    a("")
    a("### 5.2 唯一的扫描：MLP，C = 2，k = 1，含 $a_t$")
    a("")
    for env in ["TwoRoom", "Push-T", "Reacher"]:
        lams = sorted({r["lam"] for r in RUNS if (r["env"], r["reg"], r["head"], r["delta"], r["C"], r["k"], r["at"], r["G"]) ==
                       (env, "policy", "mlp", False, 2, 1, True, DASH)}, key=float)
        a(f"**{ENV_ZH[env]}**（默认 λ = {LAMBDA[env]}）")
        a("")
        rows = []
        for lam in lams:
            f = family(env, lam=lam)
            rows.append([lam, succ(f), " / ".join(f"{rnd(r['success'], '1')}" for r in f), xcell(f, "std_fp32", std2), xcell(f, "erank_fp32", rank),
                         cell(f, "policy", loss), cell(f, "pred", sci)])
        a(table(["λ", "成功率 %", "各 seed", "latent std", "eff. rank（fp32）", "policy loss（train）", "pred loss（train）"], rows))
        a("")
    a("- **TwoRoom**：λ ≤ 0.003 完全坍缩（latent 标准差 1e-6 量级，这时的 rank 没有意义）。λ 从 0.005 到 0.1，latent 尺度单调变大，成功率在 0.1 才稳定（三个 seed 都在 87–95）。"
      "0.1 是扫描的上界，更大的 λ 没有试过，所以这个 head 的最优 λ 还没有被夹住。")
    a("- **Push-T**：λ 从 3 到 300，latent 标准差从 0.25 增大到 0.61，成功率一直在 34–45，policy loss 不变。对这个 head 调 λ 没有用；同样 λ = 30 下去掉 $a_t$ 是 65.3。")
    a("- **Reacher**：四个 λ 下 policy loss 都约为 1，成功率 8–16，调 λ 没有用。")
    a("")
    a("### 5.3 现在能说什么、不能说什么")
    a("")
    a(table(["说法", "能否成立", "依据"], [
        ["含 $a_t$ 的 MLP head 在 Push-T 上明显差于 inverse", "可以", "λ 在 3–300 之间都扫过，成功率 34–45，inverse 是 89"],
        ["不带目标的 policy head 在随机动作的 Reacher 上不起作用", "可以", "含 $a_t$ 的 MLP 在 λ 0.5–500 下 policy loss 都约为 1；无 $a_t$ 和 C = 1 各 3 个 seed 也是（λ 只有 5）。"
         "数据分析给出了原因：随机动作下 $a_{t+1}$ 与状态无关（[[9 Oct with Yanbo]]）"],
        ["去掉 $a_t$ 更好", "Push-T 上可以；TwoRoom、Cube 上只是单个 λ 的对比", "Push-T：含 $a_t$ 在所有 λ 下最高 45，无 $a_t$ 是 65.3 ± 3.2。TwoRoom：含 $a_t$ 的最优 λ 没有夹住；Cube：两边都只有 λ = 1"],
        ["Transformer / Mamba 不如 MLP", "**不能**", "它们只有一个 λ、一个 seed；latent 尺度偏小，可能只是 λ 太小（要点 2）。"
         "在论文用的输入上（无 $a_t$、带目标）没有出现这个差距（第 7 节），那里也只有一个 λ、一个 seed"],
        ["context 越长越差", "**不能**", "C = 4 只有一个 λ、一个 seed；C = 16 还换了 batch size 和训练片段数"],
        ["预测多步动作（k = 2, 3）没用", "**不能**", "只在 C = 16 上试过，同上"],
        ["goal policy、inverse + policy 回到 inverse 的水平", "可以作为观察", "TwoRoom、Reacher、Push-T 上都接近 inverse，但各只有一个 seed 和一个 λ"],
    ], align=["l", "l", "l"]))
    a("")
    a("如果要对结构下结论，至少需要：每种 head 各自扫 λ（TwoRoom 上从 0.1 往上，例如 0.1 / 0.3 / 1 / 3）；C = 16 用和其他列相同的 batch size 和 forward model 训练量；每格 3 个 seed。")
    a("")
    a("## 6. LeWAM 已经解决了什么")
    a("")
    a("LeWAM（Fu, Siebert, Halicki, Balestriero；arXiv 2610.12818，2026-10-08）把 policy 和 JEPA 世界模型一起端到端训练，在我们的四个任务上都评测了。"
      "它的 policy 分支是 flow matching：从两个历史 latent $(z_{t-5}, z_t)$、目标 latent $z_g$ 和归一化的剩余步数生成 10 步的动作块；"
      "policy 分支和 dynamics 分支放在同一个 Mixture-of-Transformers 里（宽 192，4 层；两个分支参数分开，共享注意力）。防坍缩用 SIGReg（权重 0.04）。")
    a("")
    a(table(["我们早期结果里的问题", "我们的数字", "LeWAM 的做法", "LeWAM 的结果"], [
        ["随机动作数据（Reacher）上 policy head 学不到东西，模型坍缩", "所有结构都在 10 左右，inverse 71",
         "防坍缩交给 SIGReg；policy 以目标和剩余步数为条件；规划时用 dynamics 对 policy 采样的动作做梯度修正",
         "同一份均匀随机数据上 100（LeWM + CEM 84.7）。论文没有报告 Reacher 上只执行 policy 的成功率"],
        ["输入 $a_t$ 让 head 抄动作", "去掉 $a_t$：Push-T 37 → 65.3，Cube 69 → 78.0",
         "policy 分支的输入只有历史 latent、目标和带噪的动作块，没有上一步的动作；给 dynamics 用的干净动作被注意力掩码挡在 policy 分支之外", "结构上不存在这条捷径"],
        ["MSE head 只能输出条件均值（TwoRoom 上 policy loss ≈ 0.95）", "所有 head 都是 MSE",
         "flow matching 建模动作的分布", "同样的 dynamics 目标下，规划带来的提升 flow 对 MSE：Push-T +14.7 对 +5.3，Scene +5.3 对 +0.7，Puzzle +4.7 对 −2.0。"
         "只执行 policy 时 MSE head 并不差（Push-T 90.0 对 82.7）"],
        ["预测多步动作", "k = 2, 3 只在 C = 16 上试过，35–50（有混杂）", "直接预测 10 步的动作块，执行前 5 步", "主结果都基于动作块"],
        ["Transformer 结构的 head", "TwoRoom 上 C = 4 是 45，Push-T 25（单个 λ）", "policy 分支本身就是 Transformer", "Two-Room / Reacher / Push-T / Cube：100 / 100 / 98.7 / 100"],
    ], align=["l", "l", "l", "l"]))
    a("")
    a("LeWAM 没有回答的：")
    a("")
    a("- **只靠 policy head 能不能防坍缩。** 去掉 SIGReg 后，它在 Push-T 上的规划成功率从 97.3 掉到 72.0，effective rank 从 42.9 掉到 10.9（论文表 1、表 7）。也就是说它的 action 监督单独不够，这正是我们这批实验在问的问题。")
    a("- **和 inverse dynamics 正则的比较。** 论文里没有 SMWM 这个基线。")
    a("- 数字不能和我们的直接比：它训练 50 个 epoch、batch 128、单精度、ResNet-18 编码器，规划器也不同；Cube 用了更严格的成功判据；只有一个训练 seed。")
    a("")
    a("一种读法（不是结论）：LeWAM 的结果说明 Transformer 的 policy head 和随机动作数据本身都不是障碍。我们早期 “Transformer 更差”“Reacher 上 policy 不行” 的结果，更可能来自具体的配置"
      "（MSE、没有目标、只靠 head 防坍缩、单个 λ），而不是 head 的类型。")
    a("")
    goal_head_section(a)
    a("## 8. 数据与脚本")
    a("")
    a("- 逐运行的数值：仓库 `planning/analysis/policy_heads/early_heads_runs.csv`（大表的全部行加补充指标）；更细的分表在同目录的 `early_heads_summary.md`。")
    a("- 补充指标的原始输出：`early_heads_delftblue_raw.txt`（只读脚本 `collect_early_heads_on_delftblue.py`）；各运行的配置：`delftblue_run_configs.csv`（`collect_run_configs_on_delftblue.py`）。")
    a("- 第 7 节：同目录的 `goal_head_arch_runs.csv`，每个运行一行（五个数据集上的 inverse、MLP、Transformer、Mamba），"
      "是只读脚本 `planning/experiments/head_arch/collect_head_arch.py` 2026-11-19 在 DelftBlue 上的输出；LeWM 的参照行取自同目录的 `paper_heads_per_seed.csv`。")
    a("- LeWAM 的数字读自 `~/Downloads/LEWAM.pdf`（arXiv 2610.12818v1）的第 3.1 节、附录 A、图 6、表 1 和表 7。")
    return "\n".join(N) + "\n"


if args.obsidian:
    assert EXTRA, "the Obsidian note needs the DelftBlue extras"
    assert HAVE_GH, "the Obsidian note needs goal_head_arch_runs.csv (section 7)"
    out = Path(args.obsidian).expanduser()
    out.write_text(obsidian_note(), encoding="utf-8")
    print("wrote", out)

# ---------------------------------------------------------------------------------------------- csv
XKEYS = ["validate/pred_loss", "validate/policy_loss", "validate/inv_loss", "validate/loss", "fit/loss_epoch",
         "validate/mean_per_dim_std", "std_fp32", "erank_fp32", "rel_err", "params_policy_head", "params_total", "loader.batch_size",
         "loss.policy.use_action", "loss.policy.depth", "config_date"]
with open(HERE / "early_heads_runs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "in_tables"] + header + (XKEYS if EXTRA else []))
    for r in RUNS:
        e = EXTRA.get(r["run"], {}) if r["cluster"] == "DelftBlue" else {}
        w.writerow([r["run"], int(id(r) in USED)] + [r["raw"][h].strip("*") for h in header] + ([e.get(k, "") for k in XKEYS] if EXTRA else []))

unused = [r["run"] + f" ({r['cluster']})" for r in unused]
print("wrote", HERE / "early_heads_summary.md")
print("wrote", HERE / "early_heads_runs.csv")
print(f"{len(RUNS)} runs in the note, {len(RUNS) - len(unused)} used in the tables; not used:")
for u in unused:
    print("  ", u)
