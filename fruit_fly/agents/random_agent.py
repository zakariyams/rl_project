""" Interface for the random agent"""
from __future__ import annotations
import numpy as np
from agents.base_agent import BaseAgent, Observation


class RandomAgent(BaseAgent):
    """Performs random actions, no matter what observation"""

    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    def reset(self, seed: int) -> None:
        # Offset from the episode seed so the agent's draws are independent of the arena's.
        self._rng = np.random.default_rng(seed + 1)

    def act(self, observation: Observation) -> int:
        # Action picks a random number between 0-2.
        return int(self._rng.integers(3))
