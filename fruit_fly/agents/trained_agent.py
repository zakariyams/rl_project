"""This module is used for the trained agents that uses either PPO or DQN as learning algorithms."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from agents.base_agent import BaseAgent, Observation
from rl.models import load_model
import torch

class TrainedAgent(BaseAgent):
    """Actions taken from a Stable-Baselines3 model.

    if the parameter deterministic is set to false samples will be drawn from the policy
    if it is true it will take a greedy approach by selecting the most likely action. 
    """

    def __init__(self, model, deterministic: bool) -> None:
        self.model = model
        self.deterministic = deterministic

    @classmethod
    def load(cls, path: str | Path, agent: str, deterministic: bool) -> "TrainedAgent":
        """From a run's model.zip; ``agent`` (from its config.json) selects PPO or DQN."""
        return cls(load_model(path, agent), deterministic) # load the model

    def reset(self, seed: int) -> None:
        # Sampled PPO actions draw from torch's global generator; DQN's epsilon-greedy
        # draws from NumPy's global generator and the action space. Seed all three.
        torch.manual_seed(seed + 2)
        np.random.seed(seed + 2)
        self.model.action_space.seed(seed + 2)

    def act(self, observation: Observation) -> int:
        action, _ = self.model.predict(observation, deterministic=self.deterministic) # Get the action from the model
        return int(action)
