"""Hyperparameter tuning by random search (Bergstra & Bengio 2012).

Each search declares its agents, search space, sampler seed and budgets here, so
the sampled trials are reproducible from the code alone: the same seed always
gives the same trials. Trial 0 is the default configuration; trials 1..N are
random draws. Sampled values are rounded to two significant figures so they can
be written in tables and typed on a command line without changing the run.

Protocol (experiments/tune.py runs it):

1. Every trial is trained on the validation seeds (never the test seeds 100-104,
   whose odor pairs and held-out episodes the final results use).
2. A trial's score is its held-out fruit rate with sampled actions (DQN:
   epsilon-greedy), averaged over validation seeds and then over the search's
   agents. Searches with two agents (Bio and Random connectome) tune jointly: one
   configuration is chosen for both, so wiring stays the only difference.
3. The best score wins; ties keep the lower trial number, so the default wins ties.
4. A winner other than the default is retrained on the test seeds with the final
   budget, as the tuned result reported next to the default results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

VALIDATION_SEEDS = (0, 1, 2)
TEST_SEEDS = (100, 101, 102, 103, 104)


# ---------------------------------------------------------------- distributions

@dataclass(frozen=True)
class LogUniform:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return round_sig(10 ** rng.uniform(math.log10(self.low), math.log10(self.high)))

    def describe(self) -> str:
        return f"log-uniform [{self.low:g}, {self.high:g}]"


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return round_sig(rng.uniform(self.low, self.high))

    def describe(self) -> str:
        return f"uniform [{self.low:g}, {self.high:g}]"


@dataclass(frozen=True)
class Choice:
    options: tuple

    def sample(self, rng: np.random.Generator):
        return rng.choice(self.options)

    def describe(self) -> str:
        return "one of {" + ", ".join(f"{o:g}" for o in self.options) + "}"


def round_sig(value: float, digits: int = 2) -> float:
    return float(f"{value:.{digits}g}")


# ---------------------------------------------------------------- searches

@dataclass(frozen=True)
class Search:
    """One random search; every run of it uses the default arena (environment/config.py)."""
    name: str
    agents: tuple[str, ...]
    space: dict            # TrainingConfig field -> distribution, sampled in this order
    trials: int            # random trials, in addition to the default (trial 0)
    sampler_seed: int
    tuning_steps: int      # budget per tuning run
    final_steps: int       # budget of the confirmation runs (as the final experiments)
    seeds: tuple[int, ...] = VALIDATION_SEEDS

    @property
    def label(self) -> str:
        return f"tune_{self.name}"


PPO_SPACE = {"learning_rate": LogUniform(3e-5, 3e-3), "ent_coef": LogUniform(1e-4, 1e-1),
             "n_steps": Choice((256, 512, 1024, 2048, 4096))}

SEARCHES = {search.name: search for search in (
    Search("ppo_mlp", ("ppo_mlp",), PPO_SPACE, trials=12, sampler_seed=2026,
           tuning_steps=300_000, final_steps=1_000_000),
    Search("dqn_mlp", ("dqn_mlp",), {
        "dqn_learning_rate": LogUniform(3e-5, 3e-3),
        "dqn_target_update_interval": Choice((250, 500, 1000, 2000, 5000)),
        "dqn_exploration_fraction": Uniform(0.05, 0.5),
        "dqn_exploration_final_eps": Choice((0.01, 0.05, 0.1)),
    }, trials=8, sampler_seed=2027, tuning_steps=300_000, final_steps=1_000_000),
    # Connectome runs need ~1M steps to learn, so this search uses fewer trials and two seeds.
    Search("connectome", ("ppo_bio", "ppo_random_graph"), PPO_SPACE, trials=4, sampler_seed=2028,
           tuning_steps=1_000_000, final_steps=1_000_000, seeds=(0, 1)),
)}


def sample_trials(search: Search) -> list[dict]:
    """Trial 0 = defaults (no overrides); trials 1..N drawn in order from the sampler seed."""
    rng = np.random.default_rng(search.sampler_seed)
    trials = [{}]
    for _ in range(search.trials):
        trials.append({name: _plain(dist.sample(rng)) for name, dist in search.space.items()})
    return trials


def _plain(value):
    """numpy scalars -> Python numbers, so configs compare and serialize like CLI values."""
    return value.item() if isinstance(value, np.generic) else value


def rank(scores: dict[int, dict[str, list[float]]], search: Search) -> list[dict]:
    """Rank trials by mean over agents of the mean over validation seeds.

    ``scores[trial][agent]`` lists one held-out fruit rate per completed validation
    seed. Only trials complete for every agent and seed are ranked.
    """
    rows = []
    for trial, by_agent in scores.items():
        complete = all(len(by_agent.get(agent, [])) == len(search.seeds) for agent in search.agents)
        per_agent = {agent: float(np.mean(by_agent[agent])) if by_agent.get(agent) else None
                     for agent in search.agents}
        score = float(np.mean(list(per_agent.values()))) if complete else None
        rows.append({"trial": trial, "complete": complete, "score": score, "per_agent": per_agent})
    ranked = sorted((r for r in rows if r["complete"]), key=lambda r: (-r["score"], r["trial"]))
    return ranked + sorted((r for r in rows if not r["complete"]), key=lambda r: r["trial"])
