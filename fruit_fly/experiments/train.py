"""Train agents: one saved run per seed, each evaluated on its held-out test episodes afterwards.

    python -m experiments.train --agent ppo_bio --seed 100 101 102 103 104 --experiment final_r2
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from environment.config import FruitEnvConfig
from rl.config import AGENTS, TrainingConfig, plan_run
from rl.storage import create_run, read_json
from rl.training import train_run


def run_batch(directories: list[Path], resume: bool = False) -> bool:
    """Train prepared run directories in order; False if any run failed."""
    ok = True
    # One run after another in this process; parallel training means several processes.
    for directory in directories:
        print(f"{directory.name}: {'resuming' if resume else 'training'}", flush=True)
        train_run(directory, resume=resume)
        state = read_json(directory / "status.json")
        print(f"{directory.name}: {state['status']}: {state.get('message', '')}", flush=True)
        ok = ok and state["status"] != "failed"
    return ok


def override_parser(config_class, reserved: tuple[str, ...]):
    """argparse type for ``NAME=VALUE``: a field of ``config_class``, converted to the field's type."""
    known = {field.name: field for field in fields(config_class)}

    def parse(text: str) -> tuple[str, object]:
        name, separator, value = text.partition("=")
        if not separator or name not in known or name in reserved:
            raise argparse.ArgumentTypeError(
                f"{text!r}: use NAME=VALUE with a {config_class.__name__} field other than {', '.join(reserved)}")
        # Convert the text to the type of the field's default value (int, float, bool or str).
        kind = type(known[name].default)
        try:
            if kind is bool:
                if value.lower() not in ("true", "false", "1", "0"):
                    raise ValueError("expected true or false")
                return name, value.lower() in ("true", "1")
            return name, kind(float(value)) if kind is int else kind(value)   # int via float: "1e6" works
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{text!r}: {exc}") from exc
    return parse


# --set changes a TrainingConfig field (rl/config.py) and --env a FruitEnvConfig field
# (environment/config.py). Fields with their own option cannot be set this way.
parse_override = override_parser(TrainingConfig, ("agent", "seed", "total_timesteps", "evaluation_episodes"))
parse_environment = override_parser(FruitEnvConfig, ("seed",))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # The defaults are the protocol of the report: default arena, 1M steps, 100 test episodes.
    parser.add_argument("--agent", choices=AGENTS, default="ppo_mlp")
    parser.add_argument("--seed", type=int, nargs="+", help="Training seeds; each becomes its own run")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Training steps per run")
    parser.add_argument("--evaluation-episodes", type=int, default=100, help="Held-out test episodes per run")
    parser.add_argument("--experiment", default="scratch", help="Label that groups related runs")
    parser.add_argument("--set", metavar="NAME=VALUE", action="append", default=[], type=parse_override,
                        help="Override a TrainingConfig field, e.g. --set learning_rate=0.001")
    parser.add_argument("--env", metavar="NAME=VALUE", action="append", default=[], type=parse_environment,
                        help="Override a FruitEnvConfig field, e.g. --env odor_similarity=0.8")
    # experiments.tune creates its run directories first and starts its workers with --run-dir.
    parser.add_argument("--run-dir", type=Path, nargs="+", help="Train these existing run directories instead")
    parser.add_argument("--resume", action="store_true", help="With --run-dir: continue from the last checkpoint")
    args = parser.parse_args()

    if args.run_dir:
        directories = args.run_dir
    elif args.resume or not args.seed:
        parser.error("give --seed to create new runs, or --run-dir (with --resume) for existing ones")
    else:
        # Create one run directory per seed; its config.json records every setting,
        # including the --set and --env overrides.
        directories = []
        for seed in args.seed:
            try:
                settings, arena = plan_run(args.agent, seed, args.steps, evaluation_episodes=args.evaluation_episodes,
                                           overrides=dict(args.set), environment=dict(args.env))
            except ValueError as exc:
                parser.error(str(exc))
            directory = create_run(settings, arena, experiment=args.experiment, overrides=dict(args.set))
            print(f"Run directory: {directory}", flush=True)
            directories.append(directory)
    if not run_batch(directories, resume=args.resume):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
