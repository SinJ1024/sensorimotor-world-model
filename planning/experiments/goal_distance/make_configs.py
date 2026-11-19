#!/usr/bin/env python3
"""Write the training configs of the hindsight-goal-distance ablation.

Each config is the saved config.yaml of a finished seed-0 run of the default policy head
(goal = the last frame of the episode, loss.policy.goal_source: final) with only these changes:

  loss.policy.goal_source        final -> offset
  loss.policy.goal_offset        0 -> G       goal = z_{t+1+G}: G frames after the last context frame
  loss.policy.goal_frame_only    new, true    load and encode only the goal frame
  loss.policy.goal_clip_frames   new, F       every G trains on the clip starts of an F-frame clip
                                              (left out with --clip-frames 0)
  seed, final_training.seed      0 -> S
  subdir, wandb.config.name      <prefix>_goalfinal_seed0 -> <prefix>_goal<G>_seed<S>

Every written config is checked against its template, as text and (when PyYAML is installed)
as parsed YAML: any other difference is an error. An existing config with other content is
never overwritten.

Usage: make_configs.py <out_dir> <template.yaml> --goal-offsets 1 2 3 5 10 --seeds 0 1 2
                       [--clip-frames auto|N|0]
  --clip-frames auto   F = context + the largest goal offset given (default)
  --clip-frames N      F = N (must be >= context + the largest goal offset)
  --clip-frames 0      no goal_clip_frames: each G uses the clip starts of its own dense clip
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


def make(text, goal_offset, seed, clip_frames):
    """Return (run name, config text) for one goal offset and seed."""
    pol = policy_block(text)
    if pol.get("goal_source") != "final" or pol.get("goal_offset") != "0" or pol.get("goal_offset_max") != "null":
        raise SystemExit("the template must be a goal_source: final run (goal_offset 0, goal_offset_max null)")
    m = re.search(r"^subdir: (\S+)_goalfinal_seed0$", text, re.M)
    if not m:
        raise SystemExit("the template's subdir must end in _goalfinal_seed0")
    old, new = f"{m.group(1)}_goalfinal_seed0", f"{m.group(1)}_goal{goal_offset}_seed{seed}"
    indent = re.search(rf"^( +)name: train_{re.escape(old)}$", text, re.M)
    if not indent:
        raise SystemExit(f"no wandb name train_{old} in the template")
    pairs = [
        ("seed: 0", f"seed: {seed}"),
        (f"{indent.group(1)}name: train_{old}", f"{indent.group(1)}name: train_{new}"),
        (f"subdir: {old}", f"subdir: {new}"),
        ("    goal_offset: 0", f"    goal_offset: {goal_offset}"),
        ("    goal_source: final", "    goal_source: offset"),
    ]
    extra = ["    goal_frame_only: true"] + ([f"    goal_clip_frames: {clip_frames}"] if clip_frames else [])
    t = text
    for old_line, new_line in pairs:
        t = sub_unique(rf"^{re.escape(old_line)}$", new_line.replace("\\", r"\\"), t)
    t = sub_unique(r"^    goal_source: offset$", "\n".join(["    goal_source: offset"] + extra), t)
    t = sub_unique(r"^(final_training:\n(?:  .*\n)*?  seed:) 0$", rf"\g<1> {seed}", t)
    pairs.append(("  seed: 0", f"  seed: {seed}"))
    # A seed-0 run that was resumed after a timeout saved training.resume: true; a new run
    # starts from scratch.
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
    want_added = sorted([n for o, n in pairs if o != n] + extra)
    if sorted(removed) != want_removed or sorted(added) != want_added:
        raise SystemExit(f"{new}: unexpected difference from the template\n- " + "\n- ".join(removed)
                         + "\n+ " + "\n+ ".join(added))
    new_pol = policy_block(t)
    want_pol = dict(pol, goal_offset=str(goal_offset), goal_source="offset", goal_frame_only="true")
    if clip_frames:
        want_pol["goal_clip_frames"] = str(clip_frames)
    if new_pol != want_pol:
        raise SystemExit(f"{new}: loss.policy is {new_pol}, expected {want_pol}")

    # Check 2, parsed YAML: the same keys, nothing else.
    if yaml is not None:
        a, b = flatten(yaml.safe_load(text)), flatten(yaml.safe_load(t))
        changed = {k for k in set(a) | set(b) if k not in a or k not in b or a[k] != b[k]}
        want = {"subdir": new, "wandb.config.name": f"train_{new}", "loss.policy.goal_offset": goal_offset,
                "loss.policy.goal_source": "offset", "loss.policy.goal_frame_only": True}
        if clip_frames:
            want["loss.policy.goal_clip_frames"] = clip_frames
        if seed != 0:
            want.update({"seed": seed, "final_training.seed": seed})
        if ("  resume: true", "  resume: false") in pairs:
            want["training.resume"] = False
        got = {k: b.get(k) for k in changed}
        if got != want:
            raise SystemExit(f"{new}: parsed config differs from the template in {got}, expected {want}")
        if b["seed"] != seed or b["final_training.seed"] != seed or not b["loss.policy.weight"]:
            raise SystemExit(f"{new}: wrong seed fields or policy weight in the parsed config")
    return new, HEADER + t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("template", type=Path, help="config.yaml of a <prefix>_goalfinal_seed0 run")
    ap.add_argument("--goal-offsets", type=int, nargs="+", required=True, metavar="G")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, metavar="S")
    ap.add_argument("--clip-frames", default="auto", help="auto (default), a number of frames, or 0")
    args = ap.parse_args()

    text = args.template.read_text()
    context = int(policy_block(text)["context"])
    if min(args.goal_offsets) < 1:
        raise SystemExit("goal offsets must be >= 1")
    longest = context + max(args.goal_offsets)
    clip_frames = longest if args.clip_frames == "auto" else int(args.clip_frames)
    if clip_frames and clip_frames < longest:
        raise SystemExit(f"--clip-frames {clip_frames} is shorter than context + largest goal offset = {longest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for goal_offset in args.goal_offsets:
        for seed in args.seeds:
            run, content = make(text, goal_offset, seed, clip_frames)
            path = args.out_dir / f"{run}.yaml"
            if path.exists():
                if path.read_text() != content:
                    raise SystemExit(f"{path} exists with other content (another template or --clip-frames?); "
                                     "remove it or use another out_dir")
                print(f"unchanged {path}")
            else:
                path.write_text(content)
                print(f"wrote {path}")
    how = "text and parsed YAML" if yaml is not None else "text only (PyYAML not installed)"
    clips = f"goal_clip_frames {clip_frames}" if clip_frames else "no goal_clip_frames"
    print(f"{len(args.goal_offsets) * len(args.seeds)} configs, {clips}, checked against the template as {how}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
