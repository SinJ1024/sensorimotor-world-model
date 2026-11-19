"""Post-hoc policy heads on a frozen world model (control for the policy warm-start).

Loads a finished run (e.g. an inverse-regularized model), freezes encoder,
projector and predictor, and trains one or more PolicyModel heads on its latents
with the same loss as the policy regularizer in train.py (predict a_{t+1} from
a window of latents [+ goal latent]), but without any gradient into the world
model. Each head is written as its own run directory (the source config with a
loss.policy block + the source weights + policy_model.*), so eval.py's policy
warm-start loads it like a jointly trained head.

Same data, batch size, optimiser, schedule, gradient clipping and number of
epochs as the jointly trained heads. Two differences, both in favour of the
post-hoc head: the encoder runs in eval mode (BatchNorm running statistics), so
the head is trained on exactly the latents the planner sees, and the head itself
runs in single precision (the encoder in bfloat16, as in training).

The validation loss is the mean over the first --val-batches held-out batches
(default 10 = trainer.limit_val_batches): the same clips as the validation loss
that train.py logs for a jointly trained head.

An output directory is written only if it does not exist yet, or if it is an
unfinished post-hoc run of the same source (then it is trained again from the
start). A finished run has posthoc_done.json.

Usage:
  python scripts/train_posthoc_policy.py --src RUN_DIR --out-root DIR \
      --head pusht_inverse_posthoc_goalfinal_seed0:endpoints:final
  several --head NAME:endpoints|delta:final|none train several heads on one pass over the data
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stable_pretraining as spt  # noqa: E402

from eval import load_jepa_from_run  # noqa: E402
from module import PolicyModel  # noqa: E402
from train import FinalGoalHDF5Dataset  # noqa: E402
from utils import ResizeCompat, get_column_normalizer  # noqa: E402

DONE = "posthoc_done.json"


def parse_head(spec):
    name, inp, goal = spec.split(":")
    if inp not in ("endpoints", "delta") or goal not in ("final", "none"):
        raise ValueError(f"--head NAME:endpoints|delta:final|none, got {spec!r}")
    return name, inp, goal


def check_outputs(args, specs):
    """Never write into a directory that is not an unfinished post-hoc run of the same source."""
    for name, _, _ in specs:
        out = args.out_root / name
        if not out.exists():
            continue
        if (out / DONE).is_file():
            raise SystemExit(f"{out} is a finished post-hoc run; move it away to train it again")
        cfg_file = out / "config.yaml"
        pol = OmegaConf.load(cfg_file).get("loss", {}).get("policy", {}) if cfg_file.is_file() else {}
        if not (pol.get("posthoc", False) and str(pol.get("posthoc_source", "")) == str(args.src)):
            raise SystemExit(f"{out} exists and is not an unfinished post-hoc run of {args.src}; refusing to overwrite it")
        print(f"{out}: unfinished post-hoc run of the same source, training it again from the start", flush=True)


def build_loaders(cfg, batch_size, num_workers):
    """Train / eval loaders exactly as train.py builds them (2-step clips + final frame)."""
    ds_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    ds_cfg["num_steps"] = max(int(ds_cfg.get("num_steps", 2)), 2)
    train_set = FinalGoalHDF5Dataset(**ds_cfg, transform=None)
    train_set._open()
    pixel_hw = tuple(train_set.h5_file["pixels"][0].shape[:2])
    transforms = []
    if pixel_hw != (cfg.img_size, cfg.img_size):
        transforms.append(ResizeCompat(cfg.img_size, source="pixels", target="pixels"))
    for col in ds_cfg["keys_to_load"]:
        if not col.startswith("pixels"):
            transforms.append(get_column_normalizer(train_set, col, col))
    transform = spt.data.transforms.Compose(*transforms)
    train_set.transform = transform
    val_set = FinalGoalHDF5Dataset(**{**ds_cfg, "name": ds_cfg["name"][: -len("_train")] + "_eval"},
                                   transform=transform)
    kw = dict(batch_size=batch_size, pin_memory=True)
    if num_workers > 0:
        kw["prefetch_factor"] = int(cfg.loader.get("prefetch_factor", 2))
    gen = torch.Generator().manual_seed(int(cfg.seed))
    return (torch.utils.data.DataLoader(train_set, shuffle=True, drop_last=True, generator=gen, num_workers=num_workers,
                                        persistent_workers=num_workers > 0, **kw),
            # only the first batches are read, in every epoch: a few fresh workers each time
            torch.utils.data.DataLoader(val_set, shuffle=False, drop_last=False, num_workers=min(4, num_workers), **kw))


@torch.no_grad()
def encode(model, batch, dev):
    """Frozen latents of the 2-step clip and of the episode's final frame."""
    pixels = batch["pixels"].to(dev, non_blocking=True)
    gp = batch["goal_pixels"].to(dev, non_blocking=True)
    pixels = torch.cat([pixels, gp.unsqueeze(1) if gp.ndim == 4 else gp], dim=1)
    action = torch.nan_to_num(batch["action"].to(dev, non_blocking=True), 0.0)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda")):
        emb = model.encode({"pixels": pixels, "action": action})["emb"].float()
    return emb[:, :2], emb[:, -1], action[:, 1:2]


def head_losses(heads, z_window, z_goal, target):
    return {name: (h(z_window, None, z_goal if h.use_goal else None) - target).pow(2).mean()
            for name, h in heads.items()}


def save_heads(args, specs, heads, history, done):
    """Write one run directory per head (called after every epoch; the last call adds posthoc_done.json)."""
    src_ckpt = torch.load(args.src / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
    src_state = {k: v for k, v in src_ckpt["state_dict"].items()
                 if k.startswith("model.") and not k.startswith(("model.policy_model.", "model.goal_policy."))}
    for name, inp, goal in specs:
        out = args.out_root / name
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        run_cfg = OmegaConf.load(args.src / "config.yaml")
        with open_dict(run_cfg):
            run_cfg.subdir = name
            run_cfg.wandb.enabled = False
            # weight is only a marker for eval.py / policy_cem.py: the world model got no policy gradient.
            run_cfg.loss.policy = {"weight": 1.0, "use_action": False, "num_future": 1, "context": 2,
                                   "arch": "mlp", "depth": 2, "input": inp, "goal_offset": 0,
                                   "goal_offset_max": None, "goal_source": "final" if goal == "final" else "offset",
                                   "posthoc": True, "posthoc_source": str(args.src)}
        OmegaConf.save(run_cfg, out / "config.yaml")
        state = dict(src_state)
        state.update({f"model.policy_model.{k}": v.cpu() for k, v in heads[name].state_dict().items()})
        torch.save({"epoch": src_ckpt["epoch"], "state_dict": state}, out / "checkpoints" / "last.ckpt")
        own = [{k.replace(f"/{name}", ""): v for k, v in r.items() if "/" not in k or k.endswith(f"/{name}")}
               for r in history]
        (out / "posthoc_history.json").write_text(json.dumps(own, indent=1))
        if done:
            last = own[-1]
            (out / DONE).write_text(json.dumps({
                "source": str(args.src), "input": inp, "goal": goal, "epochs": len(own),
                "train_policy_loss": last["train"], "val_policy_loss": last["val"],
                "explained_variance": 1.0 - last["val"], "val_batches": args.val_batches,
                "train_time_s": round(sum(r["time_s"] for r in own), 1)}, indent=1))
        print(f"wrote {out}" + (" (finished)" if done else ""), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="finished run dir (config.yaml + checkpoints/last.ckpt)")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--head", action="append", required=True, help="NAME:endpoints|delta:final|none")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--val-batches", type=int, default=10,
                    help="held-out batches per validation (10 = the clips of a jointly trained run's logged validation loss)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda", help="cpu only for a small test (single precision)")
    ap.add_argument("--batch-size", type=int, default=0, help="debug: 0 = the source run's batch size")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop each epoch after this many steps")
    args = ap.parse_args()

    dev = args.device
    specs = [parse_head(s) for s in args.head]
    check_outputs(args, specs)
    model, cfg = load_jepa_from_run(args.src, device=dev)
    torch.manual_seed(int(cfg.seed))
    embed_dim = int(cfg.wm.get("embed_dim", model.encoder.config.hidden_size))
    action_dim = int(cfg.data.dataset.frameskip * cfg.wm.action_dim)
    heads = torch.nn.ModuleDict({
        name: PolicyModel(embed_dim=embed_dim, action_dim=action_dim, hidden_dim=256, use_action=False,
                          num_future=1, context=2, arch="mlp", depth=2, input=inp, use_goal=goal == "final")
        for name, inp, goal in specs}).to(dev)

    train_loader, val_loader = build_loaders(cfg, args.batch_size or int(cfg.loader.batch_size), args.num_workers)
    # Same optimiser and schedule as the jointly trained heads (AdamW, 1% linear warm-up, cosine).
    opt = torch.optim.AdamW(heads.parameters(), lr=float(cfg.optimizer.lr),
                            weight_decay=float(cfg.optimizer.weight_decay))
    steps_per_epoch = min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader)
    total = args.epochs * steps_per_epoch
    warm = max(1, int(0.01 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    print(f"src={args.src.name} heads={list(heads)} steps/epoch={steps_per_epoch} of {len(train_loader)} "
          f"epochs={args.epochs} batch={train_loader.batch_size}", flush=True)

    history = []
    for epoch in range(args.epochs):
        heads.train()
        t0, run = time.time(), {n: 0.0 for n in heads}
        for i, batch in enumerate(train_loader):
            if i == steps_per_epoch:
                break
            losses = head_losses(heads, *encode(model, batch, dev))
            opt.zero_grad(set_to_none=True)
            sum(losses.values()).backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
            opt.step()
            sched.step()
            for n, v in losses.items():
                run[n] += v.item()
            if i % 500 == 0:
                print(f"epoch {epoch} step {i} ({(i + 1) / (time.time() - t0):.2f} steps/s) "
                      + " ".join(f"{n}={v.item():.4f}" for n, v in losses.items()), flush=True)
        train_time = time.time() - t0
        heads.eval()
        val, n_val = {n: 0.0 for n in heads}, 0
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                if j == args.val_batches:
                    break
                for n, v in head_losses(heads, *encode(model, batch, dev)).items():
                    val[n] += v.item()
                n_val += 1
        rec = {"epoch": epoch, "time_s": round(time.time() - t0, 1), "steps_per_s": round(steps_per_epoch / train_time, 3),
               **{f"train/{n}": run[n] / steps_per_epoch for n in heads},
               **{f"val/{n}": val[n] / max(n_val, 1) for n in heads}}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        save_heads(args, specs, heads, history, done=epoch == args.epochs - 1)


if __name__ == "__main__":
    main()
