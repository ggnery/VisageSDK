"""Launch + monitor helpers for the Streamlit GUI.

Training runs are spawned as subprocesses with stdout/stderr piped to a
log file inside the run directory. TensorBoard events written by the
Trainer are read back via `event_accumulator` to drive live charts.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunHandle:
    run_dir: Path
    process: subprocess.Popen | None = None
    log_path: Path | None = None
    log_file: IO | None = None
    env: dict[str, str] = field(default_factory=dict)

    @property
    def is_alive(self) -> bool:
        if self.process is None:
            return False
        alive = self.process.poll() is None
        # Close the log file once the subprocess has exited so we don't leak FDs
        # across long Streamlit sessions.
        if not alive and self.log_file is not None:
            with contextlib.suppress(Exception):
                self.log_file.close()
            self.log_file = None
        return alive

    @property
    def returncode(self) -> int | None:
        if self.process is None:
            return None
        return self.process.poll()


def make_run_dir(parent: Path, name: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    suffix = f"_{name}" if name else ""
    run_dir = parent / f"{timestamp}{suffix}"
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_yaml(path: Path, content: str | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, dict):
        path.write_text(yaml.safe_dump(content, sort_keys=False))
    else:
        path.write_text(content)


def _spawn(script: str, env: dict[str, str], log_path: Path) -> tuple[subprocess.Popen, IO]:
    full_env = {**os.environ, **env}
    # The log_file deliberately stays open for the lifetime of the subprocess
    # — RunHandle.is_alive / stop_run close it once the process exits.
    log_file = open(log_path, "w", buffering=1)  # noqa: SIM115
    process = subprocess.Popen(
        [sys.executable, script],
        cwd=str(REPO_ROOT),
        env=full_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    return process, log_file


def launch_training(env: dict[str, str], run_dir: Path) -> RunHandle:
    """Spawn `train.py` with the given env. Logs go to <run_dir>/train.log."""
    log_path = run_dir / "train.log"
    process, log_file = _spawn("train.py", env, log_path)
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    return RunHandle(run_dir=run_dir, process=process, log_path=log_path, log_file=log_file, env=env)


def launch_evaluation(env: dict[str, str], run_dir: Path) -> RunHandle:
    """Spawn `eval.py` with the given env. Logs go to <run_dir>/eval.log."""
    log_path = run_dir / "eval.log"
    process, log_file = _spawn("eval.py", env, log_path)
    return RunHandle(run_dir=run_dir, process=process, log_path=log_path, log_file=log_file, env=env)


def stop_run(handle: RunHandle) -> None:
    if handle.process is None:
        return
    if handle.process.poll() is None:
        handle.process.terminate()
        try:
            handle.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            handle.process.kill()
    if handle.log_file is not None:
        with contextlib.suppress(Exception):
            handle.log_file.close()
        handle.log_file = None


def tail_log(log_path: Path | None, max_lines: int = 200) -> str:
    if log_path is None or not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def find_event_files(run_dir: Path) -> list[Path]:
    """Locate all TB event files under the run directory."""
    return sorted(run_dir.rglob("events.out.tfevents.*"))


def read_scalars(event_files: list[Path]) -> dict[str, list[tuple[int, float]]]:
    """Return {tag: [(step, value), ...]} aggregated across event files."""
    if not event_files:
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    aggregated: dict[str, list[tuple[int, float]]] = {}
    for ev in event_files:
        ea = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception:
            continue
        for tag in ea.Tags().get("scalars", []):
            entries = aggregated.setdefault(tag, [])
            for event in ea.Scalars(tag):
                entries.append((event.step, float(event.value)))
    for entries in aggregated.values():
        entries.sort(key=lambda x: x[0])
    return aggregated


def list_existing_runs(parent: Path) -> list[Path]:
    if not parent.exists():
        return []
    return sorted([p for p in parent.iterdir() if p.is_dir()], reverse=True)


def list_checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("*.pth"))
