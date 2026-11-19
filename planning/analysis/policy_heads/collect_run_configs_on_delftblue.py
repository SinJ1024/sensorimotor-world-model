"""Read-only: the regularizer weights and head settings of every run directory on DelftBlue (config.yaml only).

Used to check which heads were trained with more than one regularizer weight (lambda sweep).
Run with the eval env's python, script on stdin; it writes nothing there:
    ssh delftblue '/scratch/$USER/envs/smwm-eval-py312-torch251-cu121/bin/python -' \
        < collect_run_configs_on_delftblue.py > delftblue_run_configs.csv
"""
import csv, sys
from pathlib import Path
import yaml
ROOT = Path("/scratch/jingyuansun/smwm-runs")
def get(cfg, dotted, default=""):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
COLS = ["seed", "data.dataset.name", "wm.history_size", "loss.inverse.weight", "loss.sigreg.weight", "loss.policy.weight", "loss.policy.arch",
        "loss.policy.context", "loss.policy.num_future", "loss.policy.use_action", "loss.policy.input", "loss.policy.goal_source",
        "loss.policy.goal_offset", "loss.policy.goal_frame_only", "loss.goal_policy.weight", "loss.goal_policy.goal_offsets",
        "loader.batch_size", "optimizer.lr", "trainer.max_epochs"]
w = csv.writer(sys.stdout)
w.writerow(["run", "has_ckpt", "has_emb"] + COLS)
for d in sorted(ROOT.iterdir()):
    cf = d / "config.yaml"
    if not d.is_dir() or not cf.is_file():
        continue
    try:
        cfg = yaml.safe_load(cf.read_text()) or {}
    except Exception as e:
        w.writerow([d.name, "", "", f"ERR {e}"]); continue
    w.writerow([d.name, int((d / "checkpoints" / "last.ckpt").is_file()), int((d / "final_embeddings.pt").is_file())] + [get(cfg, c) for c in COLS])
