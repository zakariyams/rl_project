"""Train one saved run, then evaluate it on its held-out test episodes.

``train_run`` is what ``python -m experiments.train`` runs for each run directory
(created by rl.storage.create_run). It reads the run's config.json, trains the
model for the planned budget while logging every training episode to
episodes.csv, saves model.zip, and writes the held-out evaluation
(rl/evaluation.py) to evaluation.json. Progress and failures go to status.json.

Training is single-threaded on the CPU so that a run is reproducible: the same
config.json gives the same training episodes and the same evaluation
(experiments/verify_reproduction.py checks this against the saved runs).

Long runs save a checkpoint every ``checkpoint_every`` steps. An interrupted run
can be resumed from it (``experiments.train --resume``); the resumed part uses a
new environment seed derived from (seed, checkpoint step), so a resumed run is
not bitwise identical to an uninterrupted one. None of the report's runs was resumed.
"""
from __future__ import annotations

import csv
import hashlib
import os
import platform
import time
import traceback
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np

from environment.config import env_config_from_dict
from rl.config import DQN_AGENTS, training_config_from_dict
from rl.evaluation import evaluate_model, summarize_by_policy
from rl.models import build_model, load_model
from rl.storage import ROOT, effective_budget, read_json, replace_file, write_json

EPISODE_FIELDS = ("episode", "timesteps", "reward", "steps", "success", "poisoned")
REPLAY_BUFFER = "replay_buffer.pkl"
# Files whose hashes versions.json records, to show which code produced a run.
HASHED_SOURCES = ("rl/training.py", "rl/models.py", "environment/fruit_env.py", "environment/odor.py",
                  "environment/config.py")


def train_run(directory: Path, resume: bool = False) -> None:
    """Train and evaluate the run in ``directory``; every failure is recorded in status.json."""
    state = read_json(directory / "status.json")
    started = time.monotonic()
    env = None

    def publish(**updates):
        state.update(updates, elapsed_seconds=round(time.monotonic() - started, 1))
        write_json(directory / "status.json", state)

    try:
        # Record the owner process first, so other tools can tell a running run from an abandoned one.
        publish(worker_pid=os.getpid())
        import torch
        from stable_baselines3.common.monitor import Monitor
        from environment import FruitEnv

        torch.set_num_threads(1)
        manifest = read_json(directory / "config.json")
        settings = training_config_from_dict(manifest["training"])
        env_config = env_config_from_dict(manifest["environment"])
        write_json(directory / "versions.json", {
            "python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in ("stable-baselines3", "torch", "gymnasium", "numpy")},
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in HASHED_SOURCES},
        })

        # Monitor adds the episode return and length to info, plus our outcome flags.
        monitor_keys = ("target_reached", "poisoned")
        if resume:
            if not (directory / "model.zip").is_file():
                raise ValueError("This run has no checkpoint to resume from.")
            model = load_model(directory / "model.zip", settings.agent)
            if settings.agent in DQN_AGENTS and (directory / REPLAY_BUFFER).is_file():
                model.load_replay_buffer(str(directory / REPLAY_BUFFER))
            start_step = int(model.num_timesteps)
            env_seed = int(np.random.SeedSequence([settings.seed, start_step]).generate_state(1)[0] % 2**31)
            env = Monitor(FruitEnv(replace(env_config, seed=env_seed)), info_keywords=monitor_keys)
            model.set_env(env)
            episode_rows = _episodes_up_to(directory, start_step)
        else:
            env = Monitor(FruitEnv(replace(env_config, seed=settings.seed)), info_keywords=monitor_keys)
            model = build_model(env, settings)
            start_step = 0
            episode_rows = []

        budget = effective_budget(settings)
        resumed = {"resumed_from": start_step, "resume_count": state.get("resume_count", 0) + 1,
                   "message": f"Resumed from the checkpoint at {start_step:,} steps."} if resume else {}
        publish(status="training", planned_timesteps=budget, timesteps=start_step, episodes=len(episode_rows),
                parameter_count=sum(p.numel() for p in model.policy.parameters()), **resumed)
        if budget > start_step:
            callback = _progress_callback(directory, settings.checkpoint_every, start_step, episode_rows, publish)
            model.learn(total_timesteps=settings.total_timesteps if not resume else budget - start_step,
                        callback=callback, reset_num_timesteps=not resume)
        _save_checkpoint(model, directory)
        publish(timesteps=model.num_timesteps, episodes=len(episode_rows), checkpoint_timesteps=model.num_timesteps)

        publish(status="evaluating", evaluation_done=0)
        episodes = evaluate_model(model, env_config, settings.seed, settings.evaluation_episodes,
                                  on_episode=lambda done: publish(evaluation_done=done))
        summary = summarize_by_policy(episodes)
        write_json(directory / "evaluation.json", {"summary": summary, "episodes": episodes})
        publish(status="completed", evaluation=summary, message="Training and held-out evaluation completed.")
    except Exception as exc:
        (directory / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        publish(status="failed", message=f"{type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            env.close()


def _progress_callback(directory: Path, checkpoint_every: int, start_step: int, episode_rows: list, publish):
    """SB3 callback: log each finished training episode, report progress and save checkpoints."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Progress(BaseCallback):
        last_checkpoint = start_step

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "episode" in info:   # set by Monitor when an episode ends
                    episode = info["episode"]
                    row = {"episode": len(episode_rows) + 1, "timesteps": self.num_timesteps,
                           "reward": float(episode["r"]), "steps": int(episode["l"]),
                           "success": bool(episode["target_reached"]), "poisoned": bool(episode["poisoned"])}
                    episode_rows.append(row)
                    with (directory / "episodes.csv").open("a", newline="", encoding="utf-8") as file:
                        writer = csv.DictWriter(file, fieldnames=EPISODE_FIELDS)
                        if file.tell() == 0:   # a new file; a resumed log already has its header
                            writer.writeheader()
                        writer.writerow(row)
            if self.num_timesteps % 256 == 0:
                publish(timesteps=self.num_timesteps, episodes=len(episode_rows))
            return True

        def _on_rollout_end(self) -> None:
            if checkpoint_every and self.num_timesteps - self.last_checkpoint >= checkpoint_every:
                _save_checkpoint(self.model, directory)
                self.last_checkpoint = self.num_timesteps
                publish(checkpoint_timesteps=self.num_timesteps)

    return Progress()


def _save_checkpoint(model, directory: Path) -> None:
    """Write model.zip (and DQN's replay buffer) atomically, so a crash never corrupts the last checkpoint."""
    partial = directory / "model.partial.zip"
    model.save(str(partial))
    replace_file(partial, directory / "model.zip")
    if hasattr(model, "save_replay_buffer"):   # DQN: resuming needs its experience too
        partial = directory / "replay_buffer.partial.pkl"
        model.save_replay_buffer(str(partial))
        replace_file(partial, directory / REPLAY_BUFFER)


def _episodes_up_to(directory: Path, timestep: int) -> list[dict]:
    """Training episodes logged up to a checkpoint; later rows are dropped from episodes.csv too."""
    path = directory / "episodes.csv"
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        for record in csv.DictReader(file):
            if int(record["timesteps"]) > timestep:
                continue
            rows.append({"episode": len(rows) + 1, "timesteps": int(record["timesteps"]),
                         "reward": float(record["reward"]), "steps": int(record["steps"]),
                         "success": record["success"] == "True", "poisoned": record["poisoned"] == "True"})
    partial = path.with_name("episodes.csv.tmp")
    with partial.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    replace_file(partial, path)
    return rows
