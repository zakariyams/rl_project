"""The interface every agent implements, so all of them are evaluated by the same loop."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

# Structure for the observation
Observation = NDArray[np.float32]

# Base class for agents with 2 abstract methods, reset and act.
class BaseAgent(ABC):
    """An agent that picks an action from the current observation only (no learning here)."""

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Prepare for a new episode; ``seed`` seeds any randomness in the agent's actions."""

    @abstractmethod
    def act(self, observation: Observation) -> int:
        """The action for this observation: 0 forward, 1 turn left, 2 turn right."""
