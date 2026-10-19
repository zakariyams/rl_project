"""Build and load the learned agents' Stable-Baselines3 models.

    ppo_mlp           PPO, separate 64x64 Tanh actor and critic
    ppo_bio           PPO, connectome actor on the biological graph, 64x64 Tanh critic
    ppo_random_graph  PPO, connectome actor on the shuffled graph (shuffle seed = training seed)
    dqn_mlp           DQN, 64x64 Tanh Q-network with target network and replay buffer

Every model reads its observation through the same fixed log-odor stage
(rl/sensory.py) and trains on the CPU.
"""
from __future__ import annotations

from pathlib import Path

from rl.config import CONNECTOME_AGENTS, DQN_AGENTS, TrainingConfig
from rl.sensory import log_odor_policy_kwargs


def build_model(env, config: TrainingConfig):
    if config.agent in DQN_AGENTS:
        return _build_dqn(env, config)
    from stable_baselines3 import PPO

    if config.agent in CONNECTOME_AGENTS:
        from connectome.graph import build_graph
        from connectome.policy import ConnectomePolicy

        policy = ConnectomePolicy
        policy_kwargs = {"graph": build_graph(CONNECTOME_AGENTS[config.agent], seed=config.seed),
                         "settle_steps": config.settle_steps, "readout_norm": config.readout_norm}
    else:
        policy = "MlpPolicy"
        policy_kwargs = {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}   # Tanh is SB3 default for PPO
    policy_kwargs.update(log_odor_policy_kwargs(env))
    return PPO(policy, env, learning_rate=config.learning_rate, n_steps=config.n_steps,
               batch_size=config.batch_size, n_epochs=config.n_epochs, gamma=config.gamma,
               gae_lambda=config.gae_lambda, clip_range=config.clip_range, ent_coef=config.ent_coef,
               seed=config.seed, policy_kwargs=policy_kwargs, device="cpu", verbose=0)


def _build_dqn(env, config: TrainingConfig):
    """DQN with the PPO actor's shape (64x64 Tanh, one output per action)."""
    import torch
    from stable_baselines3 import DQN

    policy_kwargs = {"net_arch": [64, 64], "activation_fn": torch.nn.Tanh, **log_odor_policy_kwargs(env)}
    return DQN("MlpPolicy", env, learning_rate=config.dqn_learning_rate, buffer_size=config.dqn_buffer_size,
               learning_starts=config.dqn_learning_starts, batch_size=config.batch_size, gamma=config.gamma,
               train_freq=config.dqn_train_freq, target_update_interval=config.dqn_target_update_interval,
               exploration_fraction=config.dqn_exploration_fraction,
               exploration_final_eps=config.dqn_exploration_final_eps,
               seed=config.seed, policy_kwargs=policy_kwargs, device="cpu", verbose=0)


def algorithm_class(agent: str):
    """The SB3 algorithm class that trains, and must load, models of ``agent``."""
    from stable_baselines3 import DQN, PPO
    return DQN if agent in DQN_AGENTS else PPO


def load_model(path: str | Path, agent: str):
    """A saved model.zip of ``agent``."""
    return algorithm_class(agent).load(str(path), device="cpu")


def architecture_label(config: TrainingConfig) -> str:
    """One-line description stored in config.json."""
    if config.agent in DQN_AGENTS:
        return "DQN: log-odor sensory stage; 64x64 Tanh Q-network with target network and replay buffer"
    if config.agent in CONNECTOME_AGENTS:
        graph = ("Berck 2016 larval antennal-lobe actor (140 neurons, real edges, Dale signs"
                 if config.agent == "ppo_bio" else
                 f"degree-preserving shuffled antennal-lobe actor (graph seed {config.seed}")
        readout = "normalized PN readout" if config.readout_norm else "raw PN readout"
        return (f"{graph}, {config.settle_steps} settling steps, {readout}); "
                "log-odor sensory stage; 64x64 Tanh MLP critic")
    return "MlpPolicy: log-odor sensory stage; separate 64x64 Tanh actor and critic"
