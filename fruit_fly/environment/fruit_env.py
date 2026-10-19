"""The fruit-navigation POMDP as a Gymnasium environment.

State: fly position and heading, fruit and poison
positions, step count. Each reset places the fruit, then the poison at least
``source_separation`` away, then the fly clear of both, with a random heading.
The state is not known by the agent.

Observation: ``2 * receptor_channels`` odor readings in [0, 1], the left
sensor's channels followed by the right sensor's (environment/odor.py).

Actions: 0 = walk ``move_distance`` forward (clipped at the walls),
1 = turn left by ``turn_angle``, 2 = turn right.

Reward: ``target_reward`` for reaching the fruit, ``poison_reward`` for reaching
the poison (both end the episode), otherwise ``step_penalty`` per step. The
episode is truncated after ``max_steps`` steps.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

from environment.config import FruitEnvConfig
from environment.odor import OdorField, Source
from environment.receptors import odor_profiles

## Actions that the agent can take in the environment
FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2


@dataclass(slots=True)
class EnvironmentState:
    """The hidden world state; never passed to the agent."""

    agent_position: NDArray[np.float64]
    fruit_position: NDArray[np.float64]
    heading: float
    steps: int = 0
    poison_position: NDArray[np.float64] | None = None # Could be none if poison is not enabled


class FruitEnv(gym.Env[NDArray[np.float32], int]):
    """Find the fruit and avoid the poison using only smell."""

    def __init__(self, config: FruitEnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or FruitEnvConfig()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2 * self.config.receptor_channels,), dtype=np.float32)
        self._odor_field = OdorField(
            decay=self.config.odor_decay,
            sensor_offset=self.config.odor_sensor_offset,
            noise_std=self.config.odor_noise_std,
        )
        # The two odors' receptor profiles are fixed for the environment's lifetime, so a
        # memoryless agent can learn which smell is the fruit; only positions change per episode.
        self.fruit_profile, self.poison_profile = odor_profiles(
            self.config.receptor_channels, self.config.odor_similarity, self.config.odor_profile_seed
        )
        self._state: EnvironmentState | None = None
        self._has_reset = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Start an episode with random source positions, fly position and heading."""
        del options  # part of the Gymnasium signature; unused
        # config.seed seeds the first reset only; later unseeded resets continue the
        # random stream, so consecutive training episodes differ.
        if seed is None and not self._has_reset:
            seed = self.config.seed
        super().reset(seed=seed)

        radius = self.config.fruit_radius
        fruit_position = np.array(
            [
                self.np_random.uniform(radius, self.config.arena_width - radius),
                self.np_random.uniform(radius, self.config.arena_height - radius),
            ],
            dtype=np.float64,
        )
        poison_position = self._sample_poison_position(fruit_position) if self.config.poison_enabled else None
        agent_position = self._sample_agent_position(fruit_position, poison_position)
        heading = float(self.np_random.uniform(0.0, 2.0 * math.pi))
        self._state = EnvironmentState(agent_position, fruit_position, heading, poison_position=poison_position)
        self._has_reset = True
        return self._get_observation(), self._get_info(target_reached=False)

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply one action"""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected 0, 1, or 2.")

        state = self._state
        if action == FORWARD:
            displacement = self.config.move_distance * np.array(
                [math.cos(state.heading), math.sin(state.heading)], dtype=np.float64
            )
            state.agent_position = np.clip(
                state.agent_position + displacement,
                np.array([0.0, 0.0]),
                np.array([self.config.arena_width, self.config.arena_height]),  # Setting the new position while still making sure the agent do not go out of bounds.
            )
        elif action == TURN_LEFT:
            state.heading = (state.heading + self.config.turn_angle) % (2.0 * math.pi)
        else:
            state.heading = (state.heading - self.config.turn_angle) % (2.0 * math.pi)

        state.steps += 1
        reached = self._distance_to(state.fruit_position) <= self.config.fruit_radius
        poisoned = self._distance_to(state.poison_position) <= self.config.fruit_radius
        terminated = reached or poisoned
        truncated = state.steps >= self.config.max_steps and not terminated
        if reached:
            reward = self.config.target_reward
        elif poisoned:
            reward = self.config.poison_reward
        else:
            reward = self.config.step_penalty
        observation = self._get_observation()
        info = self._get_info(target_reached=reached, poisoned=poisoned)
        return observation, float(reward), terminated, truncated, info

    def world_state(self) -> dict[str, Any]:
        """Positions and heading, for analysis and figures only (never an observation).

        Draws no random numbers, so calling it cannot change a seeded episode.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before world_state().")
        poison = self._state.poison_position
        return {
            "agent_position": self._state.agent_position.tolist(),
            "heading": self._state.heading,
            "fruit_position": self._state.fruit_position.tolist(),
            "poison_position": poison.tolist() if poison is not None else None,
            "steps": self._state.steps,
        }

    def _sample_poison_position(self, fruit_position: NDArray[np.float64]) -> NDArray[np.float64]:
        radius = self.config.fruit_radius
        for _ in range(1_000):
            candidate = np.array(
                [
                    self.np_random.uniform(radius, self.config.arena_width - radius),
                    self.np_random.uniform(radius, self.config.arena_height - radius),
                ],
                dtype=np.float64,
            )
            if np.linalg.norm(candidate - fruit_position) >= self.config.source_separation:
                return candidate
        raise RuntimeError("Could not place the poison at the required separation.")

    def _sample_agent_position(
        self, fruit_position: NDArray[np.float64], poison_position: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        """A uniform position outside both sources."""
        radius = self.config.fruit_radius
        for _ in range(1_000):
            candidate = self.np_random.uniform(
                low=np.array([0.0, 0.0]),
                high=np.array([self.config.arena_width, self.config.arena_height]),
            ).astype(np.float64)
            clear_of_poison = poison_position is None or np.linalg.norm(candidate - poison_position) > radius
            if np.linalg.norm(candidate - fruit_position) > radius and clear_of_poison:
                return candidate
        raise RuntimeError("Could not sample a non-overlapping agent position.")

    def _distance_to(self, position: NDArray[np.float64] | None) -> float:
        assert self._state is not None
        if position is None:
            return math.inf
        return float(np.linalg.norm(self._state.agent_position - position))

    def _odor_sources(self) -> list[Source]:
        assert self._state is not None
        sources = [(self._state.fruit_position, self.fruit_profile)]
        if self._state.poison_position is not None:
            sources.append((self._state.poison_position, self.poison_profile))
        return sources

    def _get_observation(self) -> NDArray[np.float32]:
        assert self._state is not None
        return self._odor_field.read(self._state.agent_position, self._state.heading, self._odor_sources(),
                                     self.np_random)

    def _get_info(self, *, target_reached: bool, poisoned: bool = False) -> dict[str, Any]:
        # Positions and distances stay out of info so no agent can use privileged information.
        assert self._state is not None
        return {"steps": self._state.steps, "target_reached": target_reached, "poisoned": poisoned}
