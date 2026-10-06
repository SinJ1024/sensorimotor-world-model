#!/usr/bin/env python3
"""Render every SMWM planning result as one self-contained HTML report page.

Reads the per-cluster summarize_runs.py TSVs (refresh_results.sh keeps them when
SMWM_RESULTS_TSV_DIR is set) and groups the runs by question: which regularizer,
which policy-head inputs, which head architecture, which lambda, and whether the
clusters agree. Upstream reference numbers come from
experiments/planning_eval/aggregated_results.csv.

Usage:
  python results_page.py --tsv DelftBlue=db.tsv --tsv Leonardo=leo.tsv \
      [--queue DelftBlue=db.queue ...] --out report.html
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_table import (ENV_LABEL, ENV_ORDER, LAMBDA, labelled, queued_run,  # noqa: E402
                           read_queue, read_runs, upstream_means)

PRIMARY = "DelftBlue"   # cluster whose number is shown when a run exists on several


# --------------------------------------------------------------------- data ---
class Runs:
    def __init__(self, rows: list[dict], queue: list[list[str]]):
        self.rows = rows
        self.queued = {queued_run(q[2]): q for q in queue}
        self.by_name: dict[str, list[dict]] = {}
        for r in rows:
            self.by_name.setdefault(r["run"], []).append(r)

    def success(self, run: str) -> float | None:
        hits = sorted(self.by_name.get(run, []), key=lambda r: r["cluster"] != PRIMARY)
        for r in hits:
            try:
                return float(r.get("success") or "")
            except ValueError:
                continue
        return None

    def state(self, run: str) -> str | None:
        """'running' / 'queued' / 'training' (exists, no eval yet) / None."""
        if run in self.queued:
            return "running" if self.queued[run][3] == "RUNNING" else "queued"
        if run in self.by_name:
            return "training"
        return None

    def seeds(self, family: str) -> dict[int, float]:
        """{seed: success} for <family>_seed<N>, one value per seed.

        A lambda-sweep run at the environment's paper lambda (e.g.
        tworoom_policy_mlp_c2k1_lam0p1_seed1) is the same configuration as the
        plain run, so it counts as a seed of the plain family.
        """
        env = family.split("_")[0]
        alias = f"{family}_lam{LAMBDA.get(env, '').replace('.', 'p')}" if env in LAMBDA else None
        out = {}
        for name in self.by_name:
            for fam in filter(None, (family, alias)):
                m = re.fullmatch(re.escape(fam) + r"_seed(\d+)", name)
                if m:
                    v = self.success(name)
                    if v is not None:
                        out.setdefault(int(m.group(1)), v)
        return dict(sorted(out.items()))


# ------------------------------------------------------------------ markup ---
def esc(s) -> str:
    return html.escape(str(s))


def bar(value: float | None) -> str:
    if value is None:
        return ""
    w = max(0.0, min(100.0, value))
    return f'<span class="bar" aria-hidden="true"><span style="width:{w:.0f}%"></span></span>'


def cell_value(value: float | None, note: str = "", state: str | None = None) -> str:
    if value is None:
        label = {"running": "running", "queued": "queued", "training": "no eval yet"}.get(state or "", "–")
        cls = "pending" if state else "na"
        return f'<td class="{cls}">{esc(label)}</td>'
    n = f'<span class="note">{esc(note)}</span>' if note else ""
    return f'<td class="v"><span class="num">{value:.0f}</span>{n}{bar(value)}</td>'


def cell_run(R: Runs, run: str) -> str:
    return cell_value(R.success(run), state=R.state(run))


def cell_family(R: Runs, family: str, fallback_seed0: str | None = None) -> str:
    """Mean ± sd over seeds when >= 2, else the single seed-0 value."""
    s = R.seeds(family)
    if len(s) >= 2:
        vals = list(s.values())
        return cell_value(statistics.mean(vals), f"± {statistics.stdev(vals):.1f} · n={len(vals)}")
    run = fallback_seed0 or f"{family}_seed0"
    v = R.success(run)
    return cell_value(v, "n=1" if v is not None else "", R.state(run))


def cell_ref(up: dict, env: str, method: str) -> str:
    v = up.get((env, method))
    if v is None:
        return '<td class="na">–</td>'
    return f'<td class="v ref"><span class="num">{v:.1f}</span>{bar(v)}</td>'


def table(caption: str, head: list[str], body: list[str], note: str = "", cls: str = "") -> str:
    th = "".join(f"<th scope=\"col\">{h}</th>" for h in head)
    rows = "".join(f"<tr>{r}</tr>" for r in body)
    n = f'<p class="tnote">{note}</p>' if note else ""
    return (f'<figure class="t {cls}"><figcaption>{caption}</figcaption>'
            f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>{n}</figure>')


def rowhead(text: str, sub: str = "") -> str:
    s = f'<span class="sub">{esc(sub)}</span>' if sub else ""
    return f'<th scope="row">{esc(text)}{s}</th>'


# ------------------------------------------------------------------ sections ---
def section_regularizers(R: Runs, up: dict) -> str:
    head = ["Environment", "Inverse<br><code>h(z<sub>t</sub>, z<sub>t+1</sub>) → a<sub>t</sub></code>",
            "Inverse + policy", "Goal policy G=2", "Goal policy G=3", "Goal policy G=1,2,3",
            "Policy<br><code>π(z<sub>t</sub>, z<sub>t+1</sub>, a<sub>t</sub>) → a<sub>t+1</sub></code>",
            "Upstream inverse", "Upstream forward-only", "Upstream random"]
    inv = {"tworoom": "tworoom_inverse_lambda_0p1", "reacher": "reacher_inverse_lambda_1",
           "pusht": "pusht_inverse_lambda_30", "cube": "cube_inverse_lambda_1"}
    body = []
    for e in ENV_ORDER:
        body.append(rowhead(ENV_LABEL[e], f"λ = {LAMBDA[e]}")
                    + cell_family(R, inv[e])
                    + cell_run(R, f"{e}_invpolicy_mlp_c2k1_seed0")
                    + cell_run(R, f"{e}_goalpolicy_g2_seed0")
                    + cell_run(R, f"{e}_goalpolicy_g3_seed0")
                    + cell_run(R, f"{e}_goalpolicy_g123_seed0")
                    + cell_family(R, f"{e}_policy_mlp_c2k1")
                    + cell_ref(up, e, "inverse") + cell_ref(up, e, "forward_only") + cell_ref(up, e, "random"))
    return table("Table 1 · Which regularizer keeps the latent useful for planning", head, body,
                 "Every regularizer uses the paper λ of its environment. Single values are seed 0; "
                 "“± sd · n” is the mean over n seeds. Upstream columns: 5-seed means of the released "
                 "paper runs under the same protocol.")


def section_inputs(R: Runs, up: dict) -> str:
    head = ["Environment",
            "+ a<sub>t</sub><br><code>π(z<sub>t</sub>, z<sub>t+1</sub>, a<sub>t</sub>)</code>",
            "No action<br><code>π(z<sub>t</sub>, z<sub>t+1</sub>)</code>",
            "One state<br><code>π(z<sub>t+1</sub>)</code>",
            "Inverse (ours)", "Upstream forward-only"]
    inv = {"tworoom": "tworoom_inverse_lambda_0p1", "reacher": "reacher_inverse_lambda_1",
           "pusht": "pusht_inverse_lambda_30", "cube": "cube_inverse_lambda_1"}
    body = []
    for e in ENV_ORDER:
        body.append(rowhead(ENV_LABEL[e])
                    + cell_family(R, f"{e}_policy_mlp_c2k1")
                    + cell_family(R, f"{e}_policy_mlp_c2k1_noact")
                    + cell_family(R, f"{e}_policy_mlp_c1k1")
                    + cell_family(R, inv[e])
                    + cell_ref(up, e, "forward_only"))
    return table("Table 2 · What the policy head sees (all predict a<sub>t+1</sub>, MLP head)", head, body,
                 "In train.py the window is emb[:, :L] and the target is the action taken at its last latent, "
                 "so the one-state head computes π(z<sub>0</sub>) → a<sub>0</sub>: the same state-to-action map.")


def section_arch_tworoom(R: Runs) -> str:
    cols = [("c2k1", "L=2, k=1"), ("c4k1", "L=4, k=1"), ("c16k1", "L=16, k=1"),
            ("c16k2", "L=16, k=2"), ("c16k3", "L=16, k=3")]
    head = ["Head"] + [f"{c}<br><span class=\"sub\">{d}</span>" for c, d in cols]
    body = []
    for arch in ("mlp", "mamba", "transformer"):
        cells = "".join(cell_run(R, f"tworoom_policy_{arch}_{c}_seed0") if (arch, c) != ("mlp", "c2k1")
                        else cell_family(R, "tworoom_policy_mlp_c2k1") for c, _ in cols)
        body.append(rowhead(arch) + cells)
    return table("Table 3 · TwoRoom policy-head architecture (context L latents, k future actions)", head, body,
                 "Policy regularizer with a<sub>t</sub> input, λ = 0.1. Inverse on the same data: 100.")


def section_arch_envs(R: Runs) -> str:
    head = ["Environment", "MLP L=2", "MLP L=4", "Mamba L=4", "Transformer L=4"]
    body = []
    for e in ENV_ORDER:
        body.append(rowhead(ENV_LABEL[e])
                    + cell_family(R, f"{e}_policy_mlp_c2k1")
                    + cell_run(R, f"{e}_policy_mlp_c4k1_seed0")
                    + cell_run(R, f"{e}_policy_mamba_c4k1_seed0")
                    + cell_run(R, f"{e}_policy_transformer_c4k1_seed0"))
    return table("Table 4 · Head architecture across environments (k = 1, with a<sub>t</sub>)", head, body)


def lam_label(x: float) -> str:
    return f"{x:g}"


def lam_rows(R: Runs, env: str, paper: float) -> list[tuple[float, dict[int, float]]]:
    """[(lambda, {seed: success})] for the MLP c2k1 policy-with-a_t sweep."""
    found: dict[float, dict[int, float]] = {}
    for name in R.by_name:
        m = re.fullmatch(rf"{env}_policy_mlp_c2k1(?:_lam(\d+)p(\d+))?_seed(\d+)", name)
        if not m:
            continue
        lam = float(f"{m.group(1)}.{m.group(2)}") if m.group(1) else paper
        v = R.success(name)
        if v is not None:
            found.setdefault(lam, {})[int(m.group(3))] = v
    return [(lam, dict(sorted(seeds.items()))) for lam, seeds in sorted(found.items())]


def section_lambda(R: Runs) -> str:
    out = []
    for env, paper in (("tworoom", 0.1), ("pusht", 30.0), ("reacher", 5.0)):
        rows = lam_rows(R, env, paper)
        if not rows:
            continue
        body = []
        for lam, seeds in rows:
            per = " · ".join(f"s{k} {v:.0f}" for k, v in seeds.items())
            mean = statistics.mean(seeds.values())
            mark = ' <span class="tag">paper λ</span>' if abs(lam - paper) < 1e-12 else ""
            body.append(f'<th scope="row"><span class="mono">{lam_label(lam)}</span>{mark}</th>'
                        + cell_value(mean, f"n={len(seeds)}") + f'<td class="per">{esc(per)}</td>')
        out.append(table(f"{ENV_LABEL[env]}", ["λ", "Success (mean)", "Per seed"], body, cls="small"))
    return '<div class="trio">' + "".join(out) + "</div>"


def section_repro(R: Runs) -> str:
    multi = sorted(n for n, rs in R.by_name.items() if len({r["cluster"] for r in rs}) > 1)
    if not multi:
        return ""
    clusters = sorted({r["cluster"] for n in multi for r in R.by_name[n]}, key=lambda c: c != PRIMARY)
    body = []
    for n in multi:
        cells = ""
        for c in clusters:
            r = next((x for x in R.by_name[n] if x["cluster"] == c), None)
            try:
                v = float(r.get("success") or "") if r else None
            except ValueError:
                v = None
            cells += cell_value(v, state="training" if r and v is None else None)
        body.append(f'<th scope="row"><code>{esc(n)}</code></th>' + cells)
    return table("Table 6 · Same run, trained independently on two clusters", ["Run"] + clusters, body,
                 "Same code, data split, seeds and evaluation protocol.", cls="small")


def section_pending(R: Runs, queue: list[list[str]]) -> str:
    items = [f"<li><code>{esc(queued_run(q[2]))}</code> <span class=\"sub\">{esc(q[0])} · {esc(q[3].lower())}</span></li>"
             for q in queue]
    return f'<ul class="pending-list">{"".join(items)}</ul>' if items else "<p>Nothing queued.</p>"


# ---------------------------------------------------------------------- page ---
STYLE = """
<style>
/* Layout: one reading column for prose, tables allowed to run wider and scroll. */
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #18202b; --muted: #5d6878; --rule: #dde2e9;
  --accent: #2a5bd7; --accent-soft: #dfe7fb; --ref: #8a94a3; --ref-soft: #e7eaef; --warn: #a15c00;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #11151b; --panel: #171c24; --ink: #e6eaf0; --muted: #9aa5b5; --rule: #2a313c;
  --accent: #7ea2ff; --accent-soft: #23324f; --ref: #8e98a8; --ref-soft: #262c36; --warn: #e3a253;
  color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #11151b; --panel: #171c24; --ink: #e6eaf0; --muted: #9aa5b5; --rule: #2a313c;
  --accent: #7ea2ff; --accent-soft: #23324f; --ref: #8e98a8; --ref-soft: #262c36; --warn: #e3a253;
  color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); }
main { max-width: 1120px; margin: 0 auto; padding: 40px 20px 72px; display: grid; gap: 40px; }
header { display: grid; gap: 10px; max-width: 70ch; }
.eyebrow { font: 600 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
h1 { font: 600 clamp(28px, 4vw, 40px)/1.1 var(--display); margin: 0; text-wrap: balance; }
h2 { font: 600 22px/1.2 var(--display); margin: 0 0 6px; text-wrap: balance; }
p { margin: 0; max-width: 70ch; }
.lede { color: var(--muted); }
.protocol { font: 13px/1.6 var(--mono); color: var(--muted); display: flex; flex-wrap: wrap; gap: 4px 18px; }
section { display: grid; gap: 14px; }
.findings { display: grid; gap: 12px; padding: 0; margin: 0; list-style: none; max-width: 78ch; }
.findings li { padding-left: 16px; border-left: 3px solid var(--accent); }
.findings b { font-weight: 600; }
.t { margin: 0; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; min-width: 0; }
figcaption { font: 600 14px/1.3 var(--display); padding: 12px 14px 0; letter-spacing: .01em; }
.scroll { overflow-x: auto; padding: 8px 6px 4px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 10px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--rule); }
thead th { font: 500 12px/1.35 var(--body); color: var(--muted); white-space: nowrap; }
tbody tr:last-child > * { border-bottom: 0; }
tbody th { font-weight: 600; white-space: nowrap; }
th .sub, li .sub { display: block; font: 400 12px/1.3 var(--mono); color: var(--muted); }
td.v { min-width: 84px; }
.num { font: 600 16px/1 var(--mono); }
.ref .num { font-weight: 400; color: var(--ref); }
.note { font: 11px/1 var(--mono); color: var(--muted); margin-left: 6px; white-space: nowrap; }
.bar { display: block; height: 4px; margin-top: 6px; background: var(--accent-soft); border-radius: 2px; overflow: hidden; }
.bar > span { display: block; height: 100%; background: var(--accent); }
.ref .bar { background: var(--ref-soft); } .ref .bar > span { background: var(--ref); }
td.na { color: var(--muted); } td.pending { color: var(--warn); font: italic 13px/1.3 var(--body); }
td.per { font: 12px/1.4 var(--mono); color: var(--muted); white-space: nowrap; }
.tnote { font-size: 13px; color: var(--muted); padding: 0 14px 12px; max-width: none; }
code, .mono { font-family: var(--mono); font-size: 0.92em; }
thead code { font-size: 11px; color: var(--ink); }
.tag { font: 500 10px/1 var(--mono); text-transform: uppercase; letter-spacing: .06em; color: var(--accent);
       border: 1px solid var(--accent); border-radius: 3px; padding: 2px 4px; margin-left: 6px; vertical-align: 1px; }
.trio { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); align-items: start; }
.cols2 { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); align-items: start; }
.caveats { display: grid; gap: 8px; margin: 0; padding-left: 20px; max-width: 78ch; }
.pending-list { margin: 0; padding-left: 20px; display: grid; gap: 6px; }
footer { font-size: 13px; color: var(--muted); }
@media (max-width: 520px) { .cols2, .trio { grid-template-columns: 1fr; } main { padding-inline: 16px; } }
</style>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", action="append", default=[])
    ap.add_argument("--queue", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = read_runs(labelled(args.tsv))
    queue = read_queue(labelled(args.queue))
    R, up = Runs(rows, queue), upstream_means()
    clusters = ", ".join(label for label, _ in labelled(args.tsv))
    today = dt.date.today().isoformat()

    findings = Path(__file__).with_name("results_findings.html")
    findings_html = findings.read_text(encoding="utf-8") if findings.is_file() else ""

    page = f"""<title>Policy vs Inverse Regularizer</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
{STYLE}
<main>
<header>
  <span class="eyebrow">Sensorimotor world models · results as of {today}</span>
  <h1>Can a policy regularizer replace inverse dynamics?</h1>
  <p class="lede">Every planning result of the study: the JEPA world model trained with each regularizer,
  then scored by the paper's CEM planner, which uses only the encoder and the forward model.</p>
  <div class="protocol"><span>success rate, %</span><span>100 tasks</span><span>goal 25 primitive steps ahead</span>
  <span>CEM budget 50</span><span>{len(rows)} runs · {esc(clusters)}</span></div>
</header>
{findings_html}
<section><h2>Regularizers</h2>{section_regularizers(R, up)}</section>
<section><h2>Policy-head inputs</h2>{section_inputs(R, up)}</section>
<section><h2>Head architecture</h2><div class="cols2">{section_arch_tworoom(R)}{section_arch_envs(R)}</div></section>
<section><h2>Regularizer weight λ</h2>
<p class="lede">Policy MLP head with a<sub>t</sub> input (L=2, k=1). Table 5.</p>{section_lambda(R)}</section>
<section><h2>Reproducibility</h2>{section_repro(R)}</section>
<section><h2>Still running</h2>{section_pending(R, queue)}</section>
<footer>Generated by <code>planning/scripts/results_page.py</code> from <code>summarize_runs.py</code>,
which also checks every run against the paper hyper-parameters.</footer>
</main>
"""
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
