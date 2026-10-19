"""Saved runs: one directory per run under data/runs/<id>/.

    config.json      experiment label, agent, arena (FruitEnvConfig) and training settings (TrainingConfig)
    status.json      queued / training / evaluating / completed / failed, progress, evaluation summary
    versions.json    Python, platform, library versions and hashes of the core source files
    episodes.csv     every training episode: end timestep, reward, steps, fruit reached, poisoned
    model.zip        the trained model (also the latest checkpoint while training)
    evaluation.json  the held-out evaluation: summary per policy and every test episode
    replay_buffer.pkl  DQN only: replay buffer of the latest checkpoint, needed to resume training

JSON files and checkpoints are replaced atomically, so a crash never leaves a
half-written file behind.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from environment.config import FruitEnvConfig
from rl.config import DQN_AGENTS, TrainingConfig

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "runs"
RESULTS = ROOT / "data" / "results"
ACTIVE = {"queued", "training", "evaluating"}


def replace_file(source: Path, target: Path, *, attempts: int = 40) -> None:
    """Atomically move ``source`` over ``target``.

    On Windows the replace fails while another process (antivirus, cloud sync)
    briefly has the file open, so retry for up to about 4 seconds.
    """
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                Path(source).unlink(missing_ok=True)
                raise
            time.sleep(0.1)


def write_json(path: Path, value: object) -> None:
    """Atomically replace ``path`` with ``value`` as JSON."""
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    replace_file(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def effective_budget(settings: TrainingConfig) -> int:
    """Steps actually collected: PPO rounds up to whole rollouts, DQN to whole training intervals."""
    chunk = settings.dqn_train_freq if settings.agent in DQN_AGENTS else settings.n_steps
    return math.ceil(settings.total_timesteps / chunk) * chunk


def create_run(settings: TrainingConfig, arena: FruitEnvConfig, *, experiment: str, overrides: dict | None = None,
               extra: dict | None = None) -> Path:
    """Write a queued run directory (config.json, status.json); rl.training.train_run trains it."""
    from rl.models import architecture_label

    run_id = uuid4().hex
    directory = RUNS / run_id
    directory.mkdir(parents=True)
    write_json(directory / "config.json", {"experiment": experiment, "agent": settings.agent,
               "environment": asdict(arena), "training": asdict(settings),
               "architecture": architecture_label(settings), "device": "cpu",
               "overrides": dict(overrides or {}), **(extra or {})})
    write_json(directory / "status.json", {
        "id": run_id, "status": "queued", "created_at": datetime.now(timezone.utc).isoformat(),
        "timesteps": 0, "planned_timesteps": effective_budget(settings), "episodes": 0,
        "elapsed_seconds": 0, "evaluation_done": 0, "evaluation_episodes": settings.evaluation_episodes})
    return directory


def load_runs() -> list[dict]:
    """Every saved run, as a summary of its config.json and status.json."""
    runs = []
    for path in RUNS.glob("*/config.json"):
        directory = path.parent
        try:
            manifest, state = read_json(path), read_json(directory / "status.json")
        except (OSError, ValueError):
            continue
        training = manifest.get("training", {})
        runs.append({"id": directory.name, "directory": directory,
                     "experiment": manifest.get("experiment", "(unlabelled)"),
                     "agent": manifest.get("agent", training.get("agent")),
                     "seed": training.get("seed"), "budget": training.get("total_timesteps"),
                     "status": state.get("status"), "timesteps": state.get("timesteps", 0),
                     "created": state.get("created_at", "")[:16].replace("T", " "),
                     "evaluation": state.get("evaluation") or {}})
    return sorted(runs, key=lambda run: (run["experiment"], run["agent"], run["seed"] or 0, run["created"]))


def process_alive(pid: int | None) -> bool:
    """Whether a training worker process still runs; "access denied" counts as alive."""
    if pid is None:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: no such process
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # signal 0 only checks that the process exists
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
