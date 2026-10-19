"""Held-out evaluation, identical for every agent.

After training, each run is scored on ``evaluation_episodes`` test episodes whose
seeds come from a separate namespace that training never uses. All agents of a
training seed (trained models, random agent, both rules) play exactly the same
episodes: same odor pair, source positions, start position and heading.

Keys in evaluation.json are historical: "ppo" is the trained model's greedy
(most likely) action and "ppo_stochastic" its sampled action; for DQN, sampled
means epsilon-greedy with the final exploration rate.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from agents import InformedFollowAgent, OdorFollowAgent, RandomAgent, TrainedAgent
from agents.base_agent import BaseAgent
from environment import FruitEnv, FruitEnvConfig, env_config_from_dict

GREEDY, SAMPLED, RANDOM = "ppo", "ppo_stochastic", "random"
RUN_POLICIES = (GREEDY, SAMPLED, RANDOM)   # what every run evaluates, in this order
ODOR_RULE, INFORMED_RULE = "odor_rule", "informed_rule"


def evaluation_seed(training_seed: int, index: int) -> int:
    """Seed of held-out episode ``index`` for a training seed (never a training reset seed)."""
    return 1_000_000_000 + training_seed * 1000 + index


def run_episode(env: FruitEnv, agent: BaseAgent, seed: int, *, trace: bool = False) -> dict:
    """Play one episode from ``env.reset(seed=seed)``.

    Returns the total reward, the number of steps and the outcome; with ``trace``
    also the fly's path and the source positions (for trajectory figures).
    """
    observation, _ = env.reset(seed=seed)
    agent.reset(seed)
    if trace:
        start = env.world_state()
        path = [start["agent_position"]]
    reward_sum = 0.0
    for step in range(1, env.config.max_steps + 1):
        observation, reward, terminated, truncated, info = env.step(agent.act(observation))
        reward_sum += reward
        if trace:
            path.append(env.world_state()["agent_position"])
        if terminated or truncated:
            break
    result = {"seed": seed, "reward": reward_sum, "steps": step,
              "success": bool(info["target_reached"]), "poisoned": bool(info["poisoned"])}
    if trace:
        result.update(path=np.array(path), fruit=start["fruit_position"], poison=start["poison_position"])
    return result


def evaluate(env_config: FruitEnvConfig, agents: dict[str, BaseAgent], training_seed: int, episodes: int,
             on_episode: Callable[[int], None] | None = None) -> list[dict]:
    """Every agent on the first ``episodes`` test episodes of ``training_seed``.

    Rows are ordered by episode, then by agent, and carry the agent's name as ``policy``.
    ``on_episode(n)`` is called after the n-th episode (for progress reports).
    """
    env = FruitEnv(env_config)
    rows = []
    for index in range(episodes):
        seed = evaluation_seed(training_seed, index)
        for name, agent in agents.items():
            rows.append({"policy": name, **run_episode(env, agent, seed)})
        if on_episode:
            on_episode(index + 1)
    env.close()
    return rows


def summarize(rows: list[dict]) -> dict:
    """Rates and means over one agent's episodes (``successful_mean_steps``: fruit episodes only)."""
    successes = [r["steps"] for r in rows if r["success"]]
    return {"episodes": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])),
            "mean_reward": float(np.mean([r["reward"] for r in rows])),
            "poison_rate": float(np.mean([r["poisoned"] for r in rows])),
            "timeout_rate": float(np.mean([not (r["success"] or r["poisoned"]) for r in rows])),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            "successful_mean_steps": float(np.mean(successes)) if successes else None}


def summarize_by_policy(rows: list[dict]) -> dict:
    names = list(dict.fromkeys(r["policy"] for r in rows))
    return {name: summarize([r for r in rows if r["policy"] == name]) for name in names}


def evaluate_model(model, env_config: FruitEnvConfig, training_seed: int, episodes: int,
                   on_episode: Callable[[int], None] | None = None) -> list[dict]:
    """A run's evaluation: the model with greedy and with sampled actions, and the random agent."""
    agents = {GREEDY: TrainedAgent(model, deterministic=True), SAMPLED: TrainedAgent(model, deterministic=False),
              RANDOM: RandomAgent()}
    return evaluate(env_config, agents, training_seed, episodes, on_episode)


def evaluate_rules(environment: dict, training_seed: int, episodes: int) -> dict:
    """Odor rule and informed rule on a saved run's arena and test episodes, as summaries."""
    config = env_config_from_dict(environment)
    informed = InformedFollowAgent.for_env(FruitEnv(config))
    rows = evaluate(config, {ODOR_RULE: OdorFollowAgent(), INFORMED_RULE: informed}, training_seed, episodes)
    return summarize_by_policy(rows)
