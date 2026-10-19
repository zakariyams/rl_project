"""Odor model: exponentially decaying concentration read by two lateral sensors."""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Source = tuple[NDArray[np.float64], NDArray[np.float64]]   # (position, receptor profile)


class OdorField:
    """Odor concentrations at the fly's left and right sensors."""

    def __init__(self, decay: float, sensor_offset: float, noise_std: float = 0.0) -> None:
        if decay < 0 or sensor_offset < 0 or noise_std < 0:
            raise ValueError("Odor parameters cannot be negative.")
        self.decay = decay
        self.sensor_offset = sensor_offset
        self.noise_std = noise_std

    def concentration(self, position: NDArray[np.float64], source: NDArray[np.float64]) -> float:
        """``exp(-decay * distance)`` from ``source`` at ``position``."""
        distance = float(np.linalg.norm(position - source))
        return math.exp(-self.decay * distance)

    def sensor_positions(
        self, agent_position: NDArray[np.float64], heading: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Left and right sensor positions, ``sensor_offset`` to each side of the head."""
        left_direction = np.array([-math.sin(heading), math.cos(heading)], dtype=np.float64)
        # Both the antenna lobes are set with some offset to either sides of the agents position. 
        left = agent_position + self.sensor_offset * left_direction
        right = agent_position - self.sensor_offset * left_direction
        return left, right

    def read(
        self,
        agent_position: NDArray[np.float64],
        heading: float,
        sources: list[Source],
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        """The observation ``[left channels..., right channels...]``.

        Channel r on each side reads ``sum over sources of profile[r] * concentration``,
        saturating at 1. Optional noise perturbs each source's concentration at each
        sensor (one draw per source and side, like a fluctuating plume), so all
        receptors on a side see the same noisy concentration.
        """
        sensors = self.sensor_positions(agent_position, heading)
        concentrations = np.array([[self.concentration(sensor, position) for sensor in sensors]
                                   for position, _ in sources], dtype=np.float64)   # (source, side)
        if self.noise_std:
            concentrations += rng.normal(0.0, self.noise_std, size=concentrations.shape)
        concentrations = np.clip(concentrations, 0.0, 1.0)
        profiles = np.array([profile for _, profile in sources], dtype=np.float64)    # (source, channel)
        readings = np.concatenate([concentrations[:, side] @ profiles for side in (0, 1)])
        return np.clip(readings, 0.0, 1.0).astype(np.float32)
