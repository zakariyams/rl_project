"""List saved runs, or pack whole experiments into a zip (for backup or for sharing the results).

    python -m experiments.runs [--experiment final_r2]   |   python -m experiments.runs snapshot final_r2 ...
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from collections import defaultdict
from datetime import datetime

from rl.evaluation import GREEDY, RANDOM, SAMPLED
from rl.storage import ACTIVE, RESULTS, ROOT, load_runs

SNAPSHOTS = ROOT / "data" / "snapshots"
# Not packed: DQN replay buffers (~35 MB each, only needed to resume training) and temporary files.
SNAPSHOT_SKIP = ("replay_buffer", ".partial.", ".tmp")


def percent(evaluation: dict, key: str) -> str:
    return f"{evaluation[key]['success_rate']:.0%}" if key in evaluation else "-"


def show(runs: list[dict]) -> None:
    """Print runs grouped by experiment, with their held-out fruit rates."""
    if not runs:
        print("No matching runs.")
        return
    groups = defaultdict(list)
    for run in runs:
        groups[run["experiment"]].append(run)
    header = (f"  {'id':8}  {'agent':16} {'seed':>4} {'steps':>11}  {'status':11} "
              f"{'greedy':>6} {'sampled':>7} {'random':>6}  created")
    for experiment, members in groups.items():
        print(f"\n{experiment}  ({len(members)} runs)")
        print(header)
        for run in members:
            ev = run["evaluation"]
            steps = f"{run['timesteps'] / 1000:.0f}k/{run['budget'] / 1000:.0f}k" if run["budget"] else "-"
            print(f"  {run['id'][:8]}  {run['agent']:16} {run['seed']!s:>4} {steps:>11}  {run['status']:11} "
                  f"{percent(ev, GREEDY):>6} {percent(ev, SAMPLED):>7} {percent(ev, RANDOM):>6}  {run['created']}")
    print("\nHeld-out fruit rates. greedy = most likely action, sampled = PPO sampling / DQN epsilon-greedy.")


def snapshot(labels: list[str], runs: list[dict]) -> None:
    """Zip the runs of these experiments (without replay buffers), data/results/ and an index."""
    chosen = [run for run in runs if run["experiment"] in labels]
    missing = sorted(set(labels) - {run["experiment"] for run in chosen})
    if missing:
        sys.exit(f"No runs labelled {missing}. See: python -m experiments.runs")
    active = [run["id"][:8] for run in chosen if run["status"] in ACTIVE]
    if active:
        sys.exit(f"Runs still active: {active}. Snapshot when they have finished.")
    # The zip holds runs/<id>/..., results/... (all of data/results/) and index.csv.
    # To restore the runs, unzip its runs/ folder into data/runs/.
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOTS / f"{'+'.join(labels)}_{datetime.now():%Y-%m-%d_%H%M}.zip"
    index = io.StringIO()
    writer = csv.writer(index)
    writer.writerow(["id", "experiment", "agent", "seed", "budget", "status", "greedy", "sampled", "random"])
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for run in chosen:
            for file in sorted(run["directory"].iterdir()):
                if file.is_file() and not any(part in file.name for part in SNAPSHOT_SKIP):
                    archive.write(file, f"runs/{run['id']}/{file.name}")
            ev = run["evaluation"]
            writer.writerow([run["id"], run["experiment"], run["agent"], run["seed"], run["budget"], run["status"],
                             *(ev.get(k, {}).get("success_rate", "") for k in (GREEDY, SAMPLED, RANDOM))])
        for file in sorted(RESULTS.rglob("*")) if RESULTS.is_dir() else []:
            if file.is_file():
                archive.write(file, f"results/{file.relative_to(RESULTS).as_posix()}")
        archive.writestr("index.csv", index.getvalue())
    print(f"{len(chosen)} runs of {labels} -> {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", help="Only show runs with this label")
    parser.add_argument("--agent", help="Only show runs of this agent")
    commands = parser.add_subparsers(dest="command")
    backup = commands.add_parser("snapshot", help="Zip whole experiments to data/snapshots/")
    backup.add_argument("labels", nargs="+", help="Experiment labels")
    args = parser.parse_args()

    runs = load_runs()
    if args.command == "snapshot":
        snapshot(args.labels, runs)
        return
    show([run for run in runs if (args.experiment is None or run["experiment"] == args.experiment)
          and (args.agent is None or run["agent"] == args.agent)])


if __name__ == "__main__":
    main()
