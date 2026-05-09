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
        """True iff the subprocess is still running. Side-effect free."""
        if self.process is None:
            return False
        return self.process.poll() is None

    @property
    def returncode(self) -> int | None:
        if self.process is None:
            return None
        return self.process.poll()

    def cleanup(self) -> None:
        """Close the subprocess log file. Idempotent; safe to call repeatedly.
        Invoke once the subprocess exits to avoid FD leaks across
        long-running Streamlit sessions.
        """
        if self.log_file is not None:
            with contextlib.suppress(Exception):
                self.log_file.close()
            self.log_file = None


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
    # Mirror launch_training so Monitor Eval can recover which checkpoint
    # / configs an eval run referenced even after the Streamlit session
    # ends and the in-memory RunHandle is gone.
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
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
    handle.cleanup()


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


def list_existing_runs(parent: Path, require_train_log: bool = True) -> list[Path]:
    """List training run dirs newest-first.

    Looks first under `parent / "trains"` (the canonical layout), then falls
    back to direct children of `parent` (legacy layout). Identifies a run by
    the presence of `train.log` so misplaced eval runs never leak in.
    """
    candidates: list[Path] = []
    for root in (parent / "trains", parent):
        if root.exists():
            candidates.extend(p for p in root.iterdir() if p.is_dir())
    if require_train_log:
        candidates = [p for p in candidates if (p / "train.log").exists()]
    # De-duplicate (a path discovered via both roots can't happen with the
    # subdir layout, but the guard is cheap) while preserving newest-first.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in sorted(candidates, reverse=True):
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def list_eval_runs(parent: Path) -> list[Path]:
    """List eval run dirs under `parent / "evals"`, newest first.

    Identifies an eval run by the presence of `eval.log`. Tolerant of
    legacy layouts that placed eval runs directly under `parent`."""
    evals_dir = parent / "evals"
    candidates: list[Path] = []
    for root in (evals_dir, parent):
        if root.exists():
            candidates.extend(p for p in root.iterdir() if p.is_dir() and (p / "eval.log").exists())
    # De-duplicate (in case an eval run appears in both — shouldn't happen but
    # cheap to guard) while preserving newest-first ordering.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in sorted(candidates, reverse=True):
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _resolve_legacy_run_path(candidate: Path, runs_root: Path) -> Path | None:
    """Map a stale `runs/<X>/...` path to `runs/trains/<X>/...` if needed.

    Train runs were originally created directly under `runs/`; recent
    versions move them under `runs/trains/`. Paths captured in older
    eval.logs still reference the legacy location and silently break the
    results lookup. This recovers them by name."""
    if candidate.exists():
        return candidate
    try:
        # Path must start with `<runs_root>/<run_name>/...` to be eligible.
        rel = candidate.relative_to(runs_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2 or parts[0] in ("trains", "evals"):
        return None
    relocated = runs_root / "trains" / Path(*parts)
    return relocated if relocated.exists() else None


def find_eval_results_json(eval_run_dir: Path) -> Path | None:
    """Locate the JSON results file produced by eval.py for this run.

    eval.py drops `eval_<EvaluatorName>_<isoformat-ts>.json` next to the
    checkpoint it scored, NOT inside the eval run dir. We try, in order:
    1. Scrape `eval.log` for `"Saved results to: <path>"` — most precise,
       works for legacy runs that lack env.json. Tolerant of stale paths
       that predate the runs/trains/ migration.
    2. Read `env.json` for CHECKPOINT_PATH and pick the most recent
       `eval_*.json` in that directory — fallback when the run crashed
       before logging the save line.
    3. Glob `runs/trains/**/checkpoints/eval_*.json` for the basename
       extracted from the log — final safety net if the parent dir was
       renamed entirely.
    """
    runs_root = eval_run_dir.parent.parent  # eval_run_dir = <runs>/evals/<run>
    if runs_root.name != "runs":
        # Best-effort: walk up until we find a dir named "runs"
        for ancestor in eval_run_dir.parents:
            if ancestor.name == "runs":
                runs_root = ancestor
                break

    log_path = eval_run_dir / "eval.log"
    log_basename: str | None = None
    if log_path.exists():
        try:
            for line in log_path.read_text(errors="replace").splitlines():
                marker = "Saved results to:"
                if marker in line:
                    raw = Path(line.split(marker, 1)[1].strip())
                    log_basename = raw.name
                    resolved = _resolve_legacy_run_path(raw, runs_root)
                    if resolved is not None:
                        return resolved
        except OSError:
            pass

    env_path = eval_run_dir / "env.json"
    if env_path.exists():
        try:
            env_data = json.loads(env_path.read_text())
        except (json.JSONDecodeError, OSError):
            env_data = {}
        ckpt_path = env_data.get("CHECKPOINT_PATH")
        if ckpt_path:
            ckpt_parent = _resolve_legacy_run_path(Path(ckpt_path).parent, runs_root)
            if ckpt_parent is not None:
                matches = sorted(ckpt_parent.glob("eval_*.json"), reverse=True)
                if matches:
                    return matches[0]

    if log_basename:
        for hit in (runs_root / "trains").rglob(log_basename):
            return hit
    return None


def list_checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("*.pth"))
