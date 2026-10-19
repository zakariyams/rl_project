"""Report figures drawn from saved runs (no training), written to data/results/figures/.

    python -m experiments.figures trajectories   # Figure 3: paths on the first test episodes
    python -m experiments.figures tuning         # Figure 2: random-search score against learning rate
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.results import AGENT_NAMES, AGENT_STYLE, GRID, INK, INK_2, SURFACE, style_axes
from rl.evaluation import RANDOM, SAMPLED
from rl.storage import RESULTS, load_runs, read_json

OUT = RESULTS / "figures"
RULE_COLOR = "#8a8984"   # neutral gray for the hand-coded references


# Trajectories

def trajectory_agents(experiment: str, seed: int):
    """The arena of ``experiment`` at this seed, and (label, color, linestyle, agent, run) per column:
    both rules, then each learned agent's default-setting run with sampled actions."""
    from agents import InformedFollowAgent, OdorFollowAgent, TrainedAgent
    from environment import FruitEnv, env_config_from_dict

    runs = {r["agent"]: r for r in load_runs() if r["experiment"] == experiment and r["seed"] == seed
            and r["status"] == "completed" and not read_json(r["directory"] / "config.json").get("overrides")}
    if not runs:
        raise SystemExit(f"No completed default-setting runs in {experiment!r} for seed {seed}.")
    config = env_config_from_dict(read_json(next(iter(runs.values()))["directory"] / "config.json")["environment"])
    columns = [("Odor rule", RULE_COLOR, (0, (4, 2)), OdorFollowAgent(), None),
               ("Informed rule", RULE_COLOR, "-", InformedFollowAgent.for_env(FruitEnv(config)), None)]
    for agent in AGENT_NAMES:
        if agent in runs:
            color, dash = AGENT_STYLE[agent]
            trained = TrainedAgent.load(runs[agent]["directory"] / "model.zip", agent, deterministic=False)
            columns.append((AGENT_NAMES[agent], color, dash, trained, runs[agent]))
    return config, columns


def trajectories(experiment: str, seed: int, count: int) -> None:
    """Report Figure 3: the first ``count`` test episodes of one seed for every agent."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from environment import FruitEnv
    from rl.evaluation import evaluation_seed, run_episode

    config, players = trajectory_agents(experiment, seed)
    env = FruitEnv(config)
    first_run = next(p[4] for p in players if p[4])
    episodes = read_json(first_run["directory"] / "evaluation.json")["summary"][RANDOM]["episodes"]

    # Every agent plays the seed's first test episodes, unselected, with rl.evaluation.run_episode,
    # the function that scored them, so each path is exactly an episode in evaluation.json.
    results = []
    for index in range(min(count, episodes)):   # the first test episodes: no selection by outcome
        eval_seed = evaluation_seed(seed, index)
        row = []
        for label, _, _, agent, run in players:
            result = run_episode(env, agent, eval_seed, trace=True)
            result["outcome"] = "fruit" if result["success"] else "poison" if result["poisoned"] else "timeout"
            if run is not None:   # the replay must be exactly the scored episode
                recorded = next(e for e in read_json(run["directory"] / "evaluation.json")["episodes"]
                                if e["policy"] == SAMPLED and e["seed"] == eval_seed)
                assert (recorded["steps"], recorded["success"]) == (result["steps"], result["success"]), \
                    f"{label}: replay differs from the evaluation of episode {index}"
            row.append(result)
        results.append((index, row))
    env.close()

    columns = len(players)
    fig, axes = plt.subplots(len(results), columns, figsize=(2.05 * columns, 2.35 * len(results) + 0.7),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    width, height, radius = config.arena_width, config.arena_height, config.fruit_radius
    for r, (index, row) in enumerate(results):
        for c, ((label, color, dash, _, _), result) in enumerate(zip(players, row)):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
            ax.add_patch(Circle(result["fruit"], radius, facecolor="none", edgecolor=INK, linewidth=1.4, zorder=2))
            ax.annotate("fruit", result["fruit"], xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=6.5, color=INK)
            if result["poison"] is not None:
                ax.add_patch(Circle(result["poison"], radius, facecolor="none", edgecolor=INK_2, hatch="xxxx",
                                    linewidth=1.0, zorder=2))
                ax.annotate("poison", result["poison"], xytext=(0, 9), textcoords="offset points", ha="center",
                            fontsize=6.5, color=INK_2)
            path = result["path"]
            ax.plot(path[:, 0], path[:, 1], color=color, linestyle=dash, linewidth=1.5, zorder=3)
            ax.scatter(*path[0], s=22, facecolor=SURFACE, edgecolor=INK, linewidth=1.1, zorder=4)
            ax.scatter(*path[-1], s=22, color=color, marker="o" if result["outcome"] == "fruit" else "X", zorder=4)
            outcome = {"fruit": f"fruit, {result['steps']} steps", "poison": f"poison, {result['steps']} steps",
                       "timeout": "timeout (200 steps)"}[result["outcome"]]
            ax.set_xlabel(outcome, fontsize=7.5, color=INK_2, labelpad=2)
            if r == 0:
                ax.set_title(label, fontsize=8.5, color=INK, pad=4)
            if c == 0:
                ax.set_ylabel(f"test episode {index}", fontsize=8, color=INK)
    fig.text(0.01, 0.01, f"The first {len(results)} held-out test episodes of seed {seed}, unselected; each row has "
             "the same layout and fruit/poison odor pair.\nOpen circle: start; dot: fruit reached; cross: poison or "
             "timeout. Trained agents: default settings, sampled actions, exactly the scored episodes.",
             fontsize=6.8, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"trajectories_{experiment}_seed{seed}.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    for index, row in results:
        print(f"episode {index}: " + ", ".join(f"{p[0]}={res['outcome']}" for p, res in zip(players, row)))
    print(f"Written to {path}")


# Tuning

def tuning() -> None:
    """Report Figure 2: validation fruit rate against learning rate for each random search."""
    import matplotlib.pyplot as plt
    from experiments import tune
    from rl.config import TrainingConfig
    from rl.tuning import SEARCHES, sample_trials

    # One panel per search: the held-out fruit rate of each trial on the validation seeds
    # (small dots: single seeds, large dots: their mean) against its learning rate,
    # with the default (trial 0) and the selected trial marked.
    panels = [("ppo_mlp", "learning_rate", "PPO · MLP"), ("connectome", "learning_rate", "PPO · Bio and Random"),
              ("dqn_mlp", "dqn_learning_rate", "DQN · MLP")]
    defaults = TrainingConfig()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, parameter, title) in zip(axes, panels):
        search = SEARCHES[name]
        trials = sample_trials(search)
        table, _ = tune.scores(search)
        ranking = read_json(RESULTS / search.label / "ranking.json")
        style_axes(ax)
        ax.set_xscale("log")
        offsets = {agent: shift for agent, shift in zip(search.agents, (0.96, 1.04) if len(search.agents) > 1 else (1,))}
        for agent in search.agents:
            color = AGENT_STYLE[agent][0]
            for trial, overrides in enumerate(trials):
                values = table[trial][agent]
                if not values:
                    continue
                x = overrides.get(parameter, getattr(defaults, parameter)) * offsets[agent]
                ax.scatter([x] * len(values), values, s=12, color=color, alpha=0.35, linewidth=0, zorder=2)
                ax.scatter(x, np.mean(values), s=38, color=color, edgecolor=SURFACE, linewidth=1.0, zorder=3,
                           label=AGENT_NAMES[agent] if trial == 0 else None)
        for trial, text in ((0, "default"), (ranking["winner"], "selected")):
            x = trials[trial].get(parameter, getattr(defaults, parameter))
            score = next(r["score"] for r in ranking["ranking"] if r["trial"] == trial)
            ax.annotate(text if trial != 0 or ranking["winner"] != 0 else "default = selected", (x, score),
                        xytext=(0, 12 if text == "selected" else -16), textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK, arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.6})
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(2.5e-5, 4e-3)   # the shared search range, so panels compare directly
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("Learning rate (log scale)", fontsize=9, color=INK)
        ax.set_title(f"{title}: {len(trials)} trials, seeds {list(search.seeds)}", fontsize=9.5, color=INK, loc="left")
        if len(search.agents) > 1:
            ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    axes[0].set_ylabel("Held-out fruit rate (validation seeds)", fontsize=9, color=INK)
    fig.text(0.01, 0.01, "Random search (rl/tuning.py). Large dots: mean over validation seeds; small dots: single "
             "seeds. Other sampled settings (entropy, rollout length; DQN: target update, exploration) vary per trial. "
             "Bio and Random share every configuration (joint tuning).", fontsize=7.2, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tuning_learning_rate.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=("trajectories", "tuning"))
    parser.add_argument("--experiment", default="final_r2", help="trajectories: experiment whose runs to replay")
    parser.add_argument("--seed", type=int, default=100, help="trajectories: test seed")
    parser.add_argument("--episodes", type=int, default=3, help="trajectories: number of test episodes (rows)")
    args = parser.parse_args()
    if args.figure == "trajectories":
        trajectories(args.experiment, args.seed, args.episodes)
    else:
        tuning()


if __name__ == "__main__":
    main()
