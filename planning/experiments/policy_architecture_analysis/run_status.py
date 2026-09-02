#!/usr/bin/env python3
"""Inspect completion and termination reasons for SMWM Slurm runs.

This handles custom Hydra ``subdir=...`` run names, old jobs no longer returned
by ``sacct``, repeated submissions, and partial runs that still have metrics.
Run it from anywhere on DelftBlue.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class LogInfo:
    path: Path
    job_id: str
    run_name: str | None
    subdir: str | None
    text: str


def command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def parse_log(path: Path) -> LogInfo:
    text = path.read_text(errors="ignore")
    header = text[:12000]
    run = re.search(r"run_name(?:\+overrides)?\s*:\s*([^\s]+)", header)
    subdir = re.search(r"(?:^|\s)subdir=([^\s]+)", header)
    job = re.search(r"(\d+)", path.name)
    return LogInfo(
        path=path,
        job_id=job.group(1) if job else "?",
        run_name=run.group(1) if run else None,
        subdir=subdir.group(1) if subdir else None,
        text=text,
    )


def find_logs(search_roots: list[Path]) -> list[LogInfo]:
    paths: set[Path] = set()
    for root in search_roots:
        if root.is_dir():
            paths.update(root.rglob("slurm-smwm-*.out"))
    return [parse_log(path) for path in paths]


def matches(log: LogInfo, run: str) -> bool:
    # Custom runs keep the base config as run_name and place the actual output
    # name in the Hydra subdir override. When subdir is present it must take
    # precedence; otherwise every derived policy run would also match the base
    # ``tworoom_policy_seed0`` run.
    if log.subdir:
        return log.subdir == run
    return log.run_name == run


def accounting(job_id: str, start_time: str) -> dict[str, str]:
    if not job_id.isdigit():
        return {}
    output = command([
        "sacct", "-X", "-j", job_id, "--starttime", start_time,
        "-n", "-P", "--format=JobIDRaw,State,ExitCode,Elapsed,Timelimit,MaxRSS,ReqMem",
    ])
    if not output:
        return {}
    rows = list(csv.DictReader(
        ["JobIDRaw|State|ExitCode|Elapsed|Timelimit|MaxRSS|ReqMem", *output.splitlines()],
        delimiter="|",
    ))
    # Prefer the allocation row (exact JobID), not .batch/.extern/.0.
    row = next((r for r in rows if r["JobIDRaw"] == job_id), rows[0] if rows else None)
    return row or {}


def active_state(job_id: str) -> str | None:
    if not job_id.isdigit():
        return None
    state = command(["squeue", "-h", "-j", job_id, "-o", "%T"])
    return state.splitlines()[0] if state else None


def final_metric(metrics: Path, column: str) -> str:
    if not metrics.is_file():
        return "-"
    try:
        df = pd.read_csv(metrics, usecols=lambda c: c in {"epoch", "step", column})
        values = pd.to_numeric(df.get(column), errors="coerce").dropna()
        return f"{float(values.iloc[-1]):.2f}" if not values.empty else "-"
    except Exception:
        return "-"


def progress(metrics: Path) -> str:
    if not metrics.is_file():
        return "no metrics"
    try:
        df = pd.read_csv(metrics, usecols=lambda c: c in {"epoch", "step"})
        epochs = pd.to_numeric(df.get("epoch"), errors="coerce").dropna()
        steps = pd.to_numeric(df.get("step"), errors="coerce").dropna()
        ep = int(epochs.max()) if not epochs.empty else -1
        step = int(steps.max()) if not steps.empty else -1
        return f"ep={ep}, step={step}"
    except Exception as exc:
        return f"metrics error: {exc}"


def infer_reason(log: LogInfo | None, state: str, complete: bool) -> str:
    if complete and state in {"COMPLETED", "UNKNOWN", "NO_ACCOUNTING"}:
        return "finished normally (final_embeddings.pt exists)"
    if log is None:
        return "no matching Slurm log; filesystem result only"
    low = log.text.lower()
    patterns = (
        (("cuda out of memory", "cublas_status_alloc_failed"), "CUDA GPU memory exhausted"),
        (("out_of_memory", "oom-kill", "oom_kill"), "CPU/Slurm memory exhausted"),
        (("time limit", "due to time limit"), "Slurm time limit reached"),
        (("could not override",), "Hydra config override error"),
        (("no kernel image",), "PyTorch CUDA architecture mismatch"),
        (("missingconfigexception",), "missing Hydra config"),
        (("no such file or directory",), "missing file/path"),
    )
    for needles, reason in patterns:
        if any(x in low for x in needles):
            return reason
    if state == "TIMEOUT":
        return "Slurm time limit reached"
    if state.startswith("OUT_OF_ME"):
        return "CPU/Slurm memory exhausted"
    if state.startswith("CANCELLED"):
        return "cancelled"
    if state == "COMPLETED" and not complete:
        return "job exited 0, but final embedding export is missing"
    if "traceback" in low:
        candidates = [
            line.strip() for line in log.text.splitlines()
            if re.search(r"(?:Error|Exception|RuntimeError|TypeError|KeyError|ValueError):", line)
        ]
        return candidates[-1][:120] if candidates else "Python exception; inspect log"
    return f"inspect {log.path.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, action="append", default=[])
    parser.add_argument("--starttime", default="2026-08-01")
    parser.add_argument("--only-incomplete", action="store_true")
    args = parser.parse_args()

    default_log_root = Path.home() / "sensorimotor-world-model" / "planning" / "experiments" / "train"
    log_roots = args.log_root or [default_log_root]
    logs = find_logs(log_roots)

    headers = ("RUN", "JOB", "STATE", "DONE", "PROGRESS", "EFF_RANK", "REASON")
    print(f"{headers[0]:55} {headers[1]:>8} {headers[2]:>15} {headers[3]:>5} "
          f"{headers[4]:>20} {headers[5]:>9}  {headers[6]}")
    print("-" * 145)

    for run_dir in sorted(args.runs_root.expanduser().glob("*/")):
        run = run_dir.name
        complete = (run_dir / "final_embeddings.pt").is_file()
        if args.only_incomplete and complete:
            continue

        candidates = [log for log in logs if matches(log, run)]
        log = max(candidates, key=lambda x: x.path.stat().st_mtime) if candidates else None
        job_id = log.job_id if log else "?"
        active = active_state(job_id)
        acct = accounting(job_id, args.starttime)
        state = active or acct.get("State") or ("NO_ACCOUNTING" if log else "NO_LOG")
        reason = infer_reason(log, state, complete)
        metrics = run_dir / "lightning" / "local" / "metrics.csv"
        print(
            f"{run:55} {job_id:>8} {state:>15} {str(complete):>5} "
            f"{progress(metrics):>20} {final_metric(metrics, 'validate/effective_rank'):>9}  {reason}"
        )
        if log and state not in {"COMPLETED", "RUNNING", "PENDING"}:
            print(f"  log: {log.path}")


if __name__ == "__main__":
    main()

