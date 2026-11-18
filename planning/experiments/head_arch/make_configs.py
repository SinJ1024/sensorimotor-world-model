#!/usr/bin/env python3
"""Write the training configs of the head-architecture sweep: the goal-conditioned policy head without
action input, (z_t, z_{t+1}, z_g) -> a_{t+1}, as an MLP, a Transformer or a Mamba, at several weights.

Each config is the saved config.yaml of a finished seed-0 run of the default policy head
(<prefix>_policy_mlp_c2k1_noact_goalfinal_seed0: MLP, endpoint input, goal = the last frame of the
episode, no action input) with only these changes:

  loss.policy.arch               mlp -> <arch>           mlp | transformer | mamba | gru
  loss.policy.weight             w0 -> <lambda>          the regularizer weight
  seed, final_training.seed      0 -> S
  subdir, wandb.config.name      <prefix>_policy_mlp_c2k1_noact_goalfinal_seed0
                                 -> <prefix>_policy_<arch>_c2k1_noact_goalfinal[_lam<x>]_seed<S>

The _lam<x> part is left out at the template's own weight w0, so the MLP at w0 keeps the name of the
existing runs; <x> is the weight with "." written as "p" (0.3 -> 0p3, 3 -> 3p0), as in the earlier
lambda sweeps. Every written config is checked against its template, as text and (when PyYAML is
installed) as parsed YAML: any other difference is an error. An existing config with other content
is never overwritten.

Usage: make_configs.py <out_dir> <template.yaml> --archs transformer mamba mlp
                       (--lambdas 0.1 0.3 1 | --multipliers 0.3 1 3 10) --seeds 0
       make_configs.py --names <template.yaml> ...     print "run arch lambda seed" per config, write nothing
  --multipliers   weights as multiples of the template's weight w0 (1 = w0)
"""
import argparse
import difflib
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # the text check below still runs
    yaml = None

HEADER = ("defaults:\n"
          "  - override hydra/job_logging: disabled\n"
          "  - override hydra/hydra_logging: disabled\n"
          "  - _self_\n")
HEAD = "c2k1_noact_goalfinal"
ARCHS = ("mlp", "transformer", "mamba", "gru")
# What the template must be: the default head of the paper, without action input.
TEMPLATE_POLICY = {"use_action": "false", "num_future": "1", "context": "2", "arch": "mlp", "input": "endpoints",
                   "goal_source": "final"}


def sub_unique(pattern, repl, text):
    """Replace a pattern that must occur exactly once."""
    n = len(re.findall(pattern, text, flags=re.M))
    if n != 1:
        raise SystemExit(f"pattern found {n} times, expected once: {pattern}")
    return re.sub(pattern, repl, text, count=1, flags=re.M)


def policy_block(text):
    """The keys of loss.policy as {key: value string}."""
    loss = re.search(r"^loss:\n((?:[ ].*\n)+)", text, re.M)
    if not loss:
        raise SystemExit("no loss: section in the template")
    block = re.search(r"^  policy:\n((?:    .*\n)+)", loss.group(1), re.M)
    if not block:
        raise SystemExit("no loss.policy section in the template")
    return dict(line.strip().split(": ", 1) for line in block.group(1).splitlines())


def flatten(node, prefix=""):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: node}


def fmt(weight):
    """The weight as it is written into the config: 0.3, 1, 30."""
    return f"{float(weight):g}"


def template_info(text):
    pol = policy_block(text)
    wrong = {k: pol.get(k) for k, v in TEMPLATE_POLICY.items() if pol.get(k) != v}
    if wrong:
        raise SystemExit(f"the template is not the default no-action goal head: loss.policy has {wrong}, expected "
                         f"{ {k: TEMPLATE_POLICY[k] for k in wrong} }")
    m = re.search(rf"^subdir: (\S+)_policy_mlp_{HEAD}_seed0$", text, re.M)
    if not m:
        raise SystemExit(f"the template's subdir must end in _policy_mlp_{HEAD}_seed0")
    return m.group(1), float(pol["weight"]), pol


def run_name(prefix, arch, weight, seed, w0):
    lam = "" if float(weight) == float(w0) else "_lam" + str(float(weight)).replace(".", "p")
    return f"{prefix}_policy_{arch}_{HEAD}{lam}_seed{seed}"


def make(text, arch, weight, seed):
    """Return (run name, config text) for one head architecture, weight and seed."""
    prefix, w0, pol = template_info(text)
    old = f"{prefix}_policy_mlp_{HEAD}_seed0"
    new = run_name(prefix, arch, weight, seed, w0)
    indent = re.search(rf"^( +)name: train_{re.escape(old)}$", text, re.M)
    if not indent:
        raise SystemExit(f"no wandb name train_{old} in the template")
    pairs = [
        ("seed: 0", f"seed: {seed}"),
        (f"{indent.group(1)}name: train_{old}", f"{indent.group(1)}name: train_{new}"),
        (f"subdir: {old}", f"subdir: {new}"),
        ("    arch: mlp", f"    arch: {arch}"),
    ]
    t = text
    for old_line, new_line in pairs:
        t = sub_unique(rf"^{re.escape(old_line)}$", new_line.replace("\\", r"\\"), t)
    # the weight of loss.policy: the first key of its block (sigreg, inverse and goal_policy have their own)
    old_w = f"    weight: {pol['weight']}"
    t = sub_unique(rf"^(  policy:\n){re.escape(old_w)}$", rf"\g<1>    weight: {fmt(weight)}", t)
    pairs.append((old_w, f"    weight: {fmt(weight)}"))
    t = sub_unique(r"^(final_training:\n(?:  .*\n)*?  seed:) 0$", rf"\g<1> {seed}", t)
    pairs.append(("  seed: 0", f"  seed: {seed}"))
    # A seed-0 run that was resumed after a timeout saved training.resume: true; a new run starts from scratch.
    if re.search(r"^training:\n  resume: true$", t, re.M):
        t = sub_unique(r"^(training:\n  resume:) true$", r"\g<1> false", t)
        pairs.append(("  resume: true", "  resume: false"))

    # Check 1, text: exactly the expected lines were removed and added.
    removed, added = [], []
    old_lines, new_lines = text.splitlines(), t.splitlines()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
        if tag != "equal":
            removed += old_lines[i1:i2]
            added += new_lines[j1:j2]
    want_removed = sorted(o for o, n in pairs if o != n)
    want_added = sorted(n for o, n in pairs if o != n)
    if sorted(removed) != want_removed or sorted(added) != want_added:
        raise SystemExit(f"{new}: unexpected difference from the template\n- " + "\n- ".join(removed)
                         + "\n+ " + "\n+ ".join(added))
    new_pol = policy_block(t)
    want_pol = dict(pol, arch=arch, weight=fmt(weight) if float(weight) != w0 else pol["weight"])
    if new_pol != want_pol:
        raise SystemExit(f"{new}: loss.policy is {new_pol}, expected {want_pol}")

    # Check 2, parsed YAML: the same keys, nothing else.
    if yaml is not None:
        a, b = flatten(yaml.safe_load(text)), flatten(yaml.safe_load(t))
        changed = {k for k in set(a) | set(b) if k not in a or k not in b or a[k] != b[k]}
        want = {"subdir": new, "wandb.config.name": f"train_{new}"}
        if arch != "mlp":
            want["loss.policy.arch"] = arch
        if float(weight) != w0:
            want["loss.policy.weight"] = float(weight)
        if seed != 0:
            want.update({"seed": seed, "final_training.seed": seed})
        if ("  resume: true", "  resume: false") in pairs:
            want["training.resume"] = False
        if new == old:                      # the template itself (MLP, its own weight, seed 0)
            want = {k: v for k, v in want.items() if k not in ("subdir", "wandb.config.name")}
        got = {k: (float(b[k]) if k == "loss.policy.weight" else b.get(k)) for k in changed}
        if got != want:
            raise SystemExit(f"{new}: parsed config differs from the template in {got}, expected {want}")
        if (b["seed"] != seed or b["final_training.seed"] != seed or float(b["loss.policy.weight"]) != float(weight)
                or b["loss.policy.arch"] != arch or b["loss.policy.use_action"] is not False
                or b["loss.policy.goal_source"] != "final" or float(b["loss.inverse.weight"]) != 0.0
                or float(b["loss.sigreg.weight"]) != 0.0):
            raise SystemExit(f"{new}: wrong seed, head or weight fields in the parsed config")
    return new, HEADER + t


def weights(args, w0):
    if args.lambdas:
        return [float(w) for w in args.lambdas]
    return [float(f"{w0 * float(m):.12g}") for m in args.multipliers]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", nargs="?", type=Path, help="where the configs are written (not with --names)")
    ap.add_argument("template", type=Path, help=f"config.yaml of a <prefix>_policy_mlp_{HEAD}_seed0 run")
    ap.add_argument("--archs", nargs="+", required=True, choices=ARCHS)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--lambdas", nargs="+", metavar="W", help="regularizer weights")
    group.add_argument("--multipliers", nargs="+", metavar="M", help="weights as multiples of the template's weight")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, metavar="S")
    ap.add_argument("--names", action="store_true", help='print "run arch lambda seed" per config and write nothing')
    args = ap.parse_args()
    if not args.names and args.out_dir is None:
        raise SystemExit("an output directory is needed (or --names)")

    text = args.template.read_text()
    prefix, w0, _ = template_info(text)
    ws = weights(args, w0)
    if min(ws) <= 0:
        raise SystemExit("weights must be positive")
    if len({fmt(w) for w in ws}) != len(ws):
        raise SystemExit(f"the weights {[fmt(w) for w in ws]} are not distinct")

    if not args.names:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for seed in args.seeds:
        for weight in ws:
            for arch in args.archs:
                run, content = make(text, arch, weight, seed)
                n += 1
                if args.names:
                    print(run, arch, fmt(weight), seed)
                    continue
                path = args.out_dir / f"{run}.yaml"
                if path.exists():
                    if path.read_text() != content:
                        raise SystemExit(f"{path} exists with other content (another template?); remove it or use another out_dir")
                    print(f"unchanged {path}")
                else:
                    path.write_text(content)
                    print(f"wrote {path}")
    how = "text and parsed YAML" if yaml is not None else "text only (PyYAML not installed)"
    print(f"{n} configs for {prefix} (template weight {fmt(w0)}), checked against the template as {how}", file=sys.stderr)


if __name__ == "__main__":
    main()
