"""Training configuration and the agents of the study.

Every run is fully described by a ``TrainingConfig`` and a ``FruitEnvConfig``,
both written to the run's config.json when it is created (rl/storage.py).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from environment.config import FruitEnvConfig

# Learned agents. The three PPO agents differ only in the actor network; DQN-MLP
# differs from PPO-MLP only in the algorithm.
AGENTS = ("ppo_mlp", "ppo_bio", "ppo_random_graph", "dqn_mlp")
CONNECTOME_AGENTS = {"ppo_bio": "bio", "ppo_random_graph": "random"}   # agent -> graph kind
DQN_AGENTS = {"dqn_mlp"}


@dataclass(frozen=True)
class TrainingConfig:
    """Budget, seed and hyperparameters of one training run (defaults = untuned configuration)."""

    agent: str = "ppo_mlp"
    total_timesteps: int = 1_000_000
    seed: int = 0                     # seeds the network, the action sampling and the training arena
    evaluation_episodes: int = 100    # held-out test episodes after training for evaluation
    
    # PPO (Stable-Baselines3 defaults except n_steps)
    learning_rate: float = 0.0003
    n_steps: int = 1024               # rollout length per update
    batch_size: int = 64              # also used by DQN
    n_epochs: int = 10
    gamma: float = 0.99               # discount factor, also used by DQN
    gae_lambda: float = 0.95          # Generalized advantage estimation for balancing bias and variance.
    clip_range: float = 0.2
    ent_coef: float = 0.01

    # For the connectome actor, compliant with (connectome/policy.py)
    settle_steps: int = 4
    readout_norm: bool = True

    # DQN (agent dqn_mlp). SB3 defaults, except that replay buffer, warm-up, target
    # update interval and exploration schedule are scaled down from SB3's Atari budgets.
    dqn_learning_rate: float = 0.0001
    dqn_buffer_size: int = 100_000
    dqn_learning_starts: int = 1_000
    dqn_train_freq: int = 4
    dqn_target_update_interval: int = 1_000
    dqn_exploration_fraction: float = 0.2
    dqn_exploration_final_eps: float = 0.05
    # Save model.zip at the first rollout end after every this many steps (0 = only at the end)
    checkpoint_every: int = 250_000

    def __post_init__(self):
        if self.agent not in AGENTS:
            raise ValueError(f"Agent must be one of {AGENTS}.")
        if not 0 <= self.seed < 1_000_000:
            raise ValueError("Training seed must be between 0 and 999999.")
        if not 1 <= self.total_timesteps <= 10_000_000:
            raise ValueError("Training budget must be 1-10000000 steps.")
        if not 1 <= self.evaluation_episodes <= 1000:
            raise ValueError("Evaluation must contain 1-1000 episodes.")
        if self.n_steps < 2 or self.batch_size < 2 or self.n_steps % self.batch_size:
            raise ValueError("Rollout size must be divisible by batch size (both >= 2).")
        if not 1 <= self.settle_steps <= 20:
            raise ValueError("settle_steps must be 1-20.")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every cannot be negative.")
        if self.dqn_buffer_size < self.batch_size or self.dqn_train_freq < 1 or self.dqn_target_update_interval < 1:
            raise ValueError("DQN buffer must hold a batch; train frequency and target interval must be >= 1.")
        if not 0 < self.dqn_exploration_fraction <= 1 or not 0 <= self.dqn_exploration_final_eps <= 1:
            raise ValueError("DQN exploration fraction must be in (0, 1] and final epsilon in [0, 1].")


def training_config_from_dict(values: dict) -> TrainingConfig:
    """The training settings of a saved run."""

    values = dict(values)
    transform = values.pop("sensory_transform", "log")
    if transform != "log":
        raise ValueError(f"This run uses sensory_transform={transform!r}, which is no longer supported.")
    return TrainingConfig(**values)


def plan_run(agent: str, seed: int, steps: int, *, evaluation_episodes: int = 100,
             overrides: dict | None = None, environment: dict | None = None) -> tuple[TrainingConfig, FruitEnvConfig]:
    """The exact training and arena configuration of one run."""
    overrides, environment = dict(overrides or {}), dict(environment or {})
    settings = TrainingConfig(agent=agent, total_timesteps=steps, seed=seed,
                              evaluation_episodes=evaluation_episodes, **overrides)
    arena = FruitEnvConfig(seed=seed, **environment)
    if "odor_profile_seed" not in environment:
        arena = replace(arena, odor_profile_seed=seed)
    return settings, arena
