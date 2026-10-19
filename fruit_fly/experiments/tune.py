"""Random-search hyperparameter tuning; the searches themselves are defined in rl/tuning.py.

    python -m experiments.tune plan|run|select|confirm ppo_mlp|dqn_mlp|connectome
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from environment.config import env_config_from_dict
from rl.config import plan_run, training_config_from_dict
from rl.storage import ACTIVE, RESULTS, ROOT, RUNS, create_run, process_alive, read_json, write_json
from rl.tuning import SEARCHES, TEST_SEEDS, Search, rank, sample_trials

# A "training" run without a process id whose status.json has not changed for this long is dead.
STALE_SECONDS = 15 * 60
# File in a run directory naming the tune worker process that will train it.
OWNER = "owner_pid"


def existing_runs() -> list[dict]:
    """Every saved run with its configuration in the current format, for matching against planned runs."""
    runs = []
    for path in RUNS.glob("*/config.json"):
        try:
            manifest, state = read_json(path), read_json(path.parent / "status.json")
        except (OSError, ValueError):
            continue
        try:   # in the current format, so fields that no longer exist do not prevent a match
            training = asdict(training_config_from_dict(manifest["training"]))
            environment = asdict(env_config_from_dict(manifest["environment"]))
        except (KeyError, TypeError, ValueError):
            continue
        runs.append({"directory": path.parent, "training": training, "environment": environment,
                     "status": "interrupted" if abandoned(path.parent, state) else state["status"],
                     "evaluation": state.get("evaluation")})
    return runs


def abandoned(directory: Path, state: dict) -> bool:
    """An active-looking run whose process is gone. Queued runs started by hand (no owner) count as alive."""
    if state["status"] not in ACTIVE:
        return False
    owner = directory / OWNER
    pid = state.get("worker_pid") or (int(owner.read_text()) if owner.is_file() else None)
    if pid is not None:
        return not process_alive(pid)
    return (state["status"] != "queued"
            and time.time() - (directory / "status.json").stat().st_mtime > STALE_SECONDS)


def plan(search: Search, seeds, steps: int) -> list[dict]:
    """Every run of a search: trial x agent x seed, with its exact configuration."""
    planned = []
    for trial, overrides in enumerate(sample_trials(search)):
        for agent in search.agents:
            for seed in seeds:
                settings, arena = plan_run(agent, seed, steps, overrides=overrides)
                planned.append({"trial": trial, "agent": agent, "seed": seed, "overrides": overrides,
                                "settings": settings, "arena": arena})
    return planned


def match(planned: dict, runs: list[dict]) -> dict | None:
    """The best existing run with this exact configuration: completed, else still active."""
    # Runs are matched by configuration, not by label, so a run started by hand with
    # experiments.train counts too. This makes run/confirm safe to start again after
    # an interruption: finished runs are reused and only the missing ones are trained.
    same = [run for run in runs if run["training"] == asdict(planned["settings"])
            and run["environment"] == asdict(planned["arena"])]
    for wanted in ("completed", *ACTIVE):
        for run in same:
            if run["status"] == wanted:
                return run
    return None


def launch(search: Search, planned: list[dict], label: str, workers: int) -> None:
    """Create the planned runs that do not exist yet and train them in ``workers`` parallel processes."""
    runs = existing_runs()
    todo, done, elsewhere = [], 0, 0
    for item in planned:
        found = match(item, runs)
        if found is None:
            todo.append(item)
        elif found["status"] == "completed":
            done += 1
        else:
            elsewhere += 1
    print(f"{len(planned)} runs planned: {done} already completed, {elsewhere} queued or training elsewhere, "
          f"{len(todo)} to train now.")
    if elsewhere:
        print("  (Runs queued or training elsewhere are left to their own process.)")
    if not todo:
        return
    directories = [create_run(item["settings"], item["arena"], experiment=label, overrides=item["overrides"],
                              extra={"tuning": {"search": search.name, "trial": item["trial"],
                                                "sampler_seed": search.sampler_seed}})
                   for item in todo]
    # Split the runs over parallel worker processes (each uses one CPU core);
    # their output goes to data/runtime/<label>_worker<n>.log.
    workers = max(1, min(workers, len(directories)))
    logs = ROOT / "data" / "runtime"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for index in range(workers):
        batch = directories[index::workers]
        log = (logs / f"{label}_worker{index}.log").open("w", encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-m", "experiments.train", "--run-dir", *map(str, batch)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        for directory in batch:   # a separate file: status.json belongs to the worker once it starts
            (directory / OWNER).write_text(str(process.pid))
        processes.append((process, log))
        print(f"worker {index}: {len(batch)} runs, log {log.name}")
    print("Training... progress: python -m experiments.runs --experiment " + label)
    failed = 0
    for process, log in processes:
        failed += process.wait() != 0
        log.close()
    print("Done." if not failed else f"{failed} worker(s) reported a failed run; see data/runtime/{label}_worker*.log")


def scores(search: Search) -> tuple[dict, dict]:
    """Held-out sampled fruit rates per trial, agent and validation seed, from completed runs."""
    runs = existing_runs()
    table: dict[int, dict[str, list[float]]] = {}
    poison: dict[int, dict[str, list[float]]] = {}
    for item in plan(search, search.seeds, search.tuning_steps):
        found = match(item, runs)
        # The score of a trial is its held-out fruit rate with sampled actions (rl/tuning.py).
        table.setdefault(item["trial"], {}).setdefault(item["agent"], [])
        poison.setdefault(item["trial"], {}).setdefault(item["agent"], [])
        if found and found["status"] == "completed":
            sampled = found["evaluation"]["ppo_stochastic"]
            table[item["trial"]][item["agent"]].append(sampled["success_rate"])
            poison[item["trial"]][item["agent"]].append(sampled.get("poison_rate", 0.0))
    return table, poison


def describe_trial(overrides: dict) -> str:
    """A trial's settings as text, e.g. "learning_rate=0.001, ent_coef=0.02"."""
    return ", ".join(f"{k}={v:g}" for k, v in overrides.items()) or "defaults"


def select(search: Search) -> dict | None:
    """Rank the trials, write ranking.md and ranking.json, and return the winner (None if unfinished)."""
    trials = sample_trials(search)
    table, poison = scores(search)
    ranked = rank(table, search)
    lines = [f"# Random search: {search.name}", "",
             f"Agents: {', '.join(search.agents)}. Sampler seed {search.sampler_seed}; trial 0 = defaults, "
             f"trials 1-{search.trials} random. Validation seeds {list(search.seeds)}, "
             f"{search.tuning_steps:,} steps per run.", "",
             "Search space: " + "; ".join(f"`{k}` {d.describe()}" for k, d in search.space.items()) + ".", "",
             "Score = held-out fruit rate (sampled actions) averaged over validation seeds"
             + (" and over the agents (joint tuning)." if len(search.agents) > 1 else "."), "",
             "| Rank | Trial | Settings | " + " | ".join(search.agents) + " | Score | Poison |",
             "|---|---|---|" + "---|" * len(search.agents) + "---|---|"]
    for position, row in enumerate(ranked, 1):
        trial = row["trial"]
        cells = [f"{row['per_agent'][a]:.1%}" if row["per_agent"][a] is not None else "–" for a in search.agents]
        rates = [v for a in search.agents for v in poison[trial][a]]
        place = position if row["complete"] else "incomplete"
        score = f"{row['score']:.1%}" if row["complete"] else "–"
        mean_poison = f"{sum(rates) / len(rates):.0%}" if rates else "–"
        lines.append(f"| {place} | {trial} | {describe_trial(trials[trial])} | " + " | ".join(cells)
                     + f" | {score} | {mean_poison} |")
    complete = [row for row in ranked if row["complete"]]
    winner = complete[0] if complete and len(complete) == len(ranked) else None
    lines.append("")
    if winner:
        lines.append(f"**Winner: trial {winner['trial']}** ({describe_trial(trials[winner['trial']])}), "
                     f"score {winner['score']:.1%}.")
    else:
        lines.append(f"Not finished: {len(ranked) - len(complete)} of {len(ranked)} trials incomplete.")
    text = "\n".join(lines) + "\n"
    out = RESULTS / search.label
    out.mkdir(parents=True, exist_ok=True)
    (out / "ranking.md").write_text(text, encoding="utf-8")
    write_json(out / "ranking.json", {"search": search.name, "agents": list(search.agents),
                                      "sampler_seed": search.sampler_seed, "seeds": list(search.seeds),
                                      "tuning_steps": search.tuning_steps,
                                      "trials": [{"trial": i, "overrides": t} for i, t in enumerate(trials)],
                                      "ranking": ranked, "winner": winner and winner["trial"]})
    print(text)
    print(f"Written to {out / 'ranking.md'}")
    return winner and {"trial": winner["trial"], "overrides": trials[winner["trial"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "run", "select", "confirm"))
    parser.add_argument("search", choices=sorted(SEARCHES))
    parser.add_argument("--workers", type=int, default=4, help="Parallel training processes (one CPU core each)")
    parser.add_argument("--dry-run", action="store_true", help="run/confirm: only report what would be trained")
    args = parser.parse_args()
    search = SEARCHES[args.search]

    # plan:    print the search space and the sampled trials
    # run:     train every trial on the validation seeds (labelled tune_<search>)
    # select:  rank the trials and write data/results/tune_<search>/ranking.md
    # confirm: retrain the winner on the test seeds (labelled tune_<search>_confirm)
    if args.command == "plan":
        print(f"{search.name}: agents {', '.join(search.agents)}; sampler seed {search.sampler_seed}; "
              f"seeds {list(search.seeds)}; {search.tuning_steps:,} steps per run")
        for name, dist in search.space.items():
            print(f"  {name}: {dist.describe()}")
        for trial, overrides in enumerate(sample_trials(search)):
            print(f"  trial {trial:2}: {describe_trial(overrides)}")
        return
    if args.command == "select":
        select(search)
        return
    if args.command == "run":
        planned, label = plan(search, search.seeds, search.tuning_steps), search.label
    else:
        winner = select(search)
        if winner is None:
            raise SystemExit("Finish the search first (python -m experiments.tune run ...).")
        if winner["trial"] == 0:
            print("The defaults won: the final experiments already are the confirmation. Nothing to train.")
            return
        print(f"Confirming trial {winner['trial']} on test seeds {list(TEST_SEEDS)} "
              f"with {search.final_steps:,} steps.")
        planned = []
        for agent in search.agents:
            for seed in TEST_SEEDS:
                settings, arena = plan_run(agent, seed, search.final_steps, overrides=winner["overrides"])
                planned.append({"trial": winner["trial"], "agent": agent, "seed": seed,
                                "overrides": winner["overrides"], "settings": settings, "arena": arena})
        label = f"{search.label}_confirm"
    if args.dry_run:
        runs = existing_runs()
        for item in planned:
            found = match(item, runs)
            state = found["status"] + " (" + found["directory"].name[:8] + ")" if found else "to train"
            print(f"  trial {item['trial']:2} {item['agent']:16} seed {item['seed']:3}: {state}")
        return
    launch(search, planned, label, args.workers)


if __name__ == "__main__":
    main()
