"""Configuration of the fruit-navigation environment.

The defaults are the task used for every experiment: a 10 x 10
arena with a fruit and a poison, observed through 21 receptor channels on each
side of the fly's head.
"""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FruitEnvConfig:
    """All parameters of :class:`environment.FruitEnv`.

    Distances are in arena units and angles in radians. The object is immutable
    so an environment can never drift away from the configuration it was built
    (and saved) with.
    """

    # Arena and episode length
    arena_width: float = 10.0
    arena_height: float = 10.0
    max_steps: int = 200                      # episodes are truncated after this many steps
    # Actions: walk forward this far, or turn in place by this angle
    move_distance: float = 0.25
    turn_angle: float = math.radians(15)
    # Sources: fruit and poison are discs of this radius; reaching one ends the episode
    fruit_radius: float = 0.5
    poison_enabled: bool = True
    source_separation: float = 3.0            # minimum distance between fruit and poison centres
    # Odor: concentration exp(-odor_decay * distance), read by two sensors beside the head
    odor_decay: float = 0.5
    odor_noise_std: float = 0.0               # optional Gaussian noise on each concentration
    odor_sensor_offset: float = 0.15
    # Odor identity (environment/receptors.py): channels per side and the receptor
    # profiles of the two odors, drawn from odor_profile_seed with this cosine similarity
    receptor_channels: int = 21
    odor_similarity: float = 0.0
    odor_profile_seed: int = 0
    # Rewards
    target_reward: float = 1.0
    poison_reward: float = -5.0
    step_penalty: float = -0.01
    # Seed of the first reset; later resets continue the same random stream
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.arena_width <= 0 or self.arena_height <= 0:
            raise ValueError("Arena dimensions must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.move_distance <= 0:
            raise ValueError("move_distance must be positive.")
        if not 0 < self.turn_angle <= 2 * math.pi:
            raise ValueError("turn_angle must be in (0, 2*pi].")
        if self.fruit_radius <= 0:
            raise ValueError("fruit_radius must be positive.")
        if 2 * self.fruit_radius >= min(self.arena_width, self.arena_height):
            raise ValueError("fruit_radius must be less than half each arena dimension.")
        if self.odor_decay < 0 or self.odor_noise_std < 0 or self.odor_sensor_offset < 0:
            raise ValueError("Odor parameters cannot be negative.")
        if self.receptor_channels < 1:
            raise ValueError("receptor_channels must be positive.")
        if not 0.0 <= self.odor_similarity <= 1.0:
            raise ValueError("odor_similarity must be in [0, 1].")
        if self.poison_enabled:
            # Eating the poison must be worse than searching until the time limit,
            # otherwise it would be a cheap way to end an episode.
            if self.poison_reward >= self.max_steps * self.step_penalty:
                raise ValueError("poison_reward must be below max_steps * step_penalty (the timeout return).")
            if self.source_separation < 2 * self.fruit_radius:
                raise ValueError("source_separation must keep the sources from overlapping.")
            if self.source_separation >= min(self.arena_width, self.arena_height) - 2 * self.fruit_radius:
                raise ValueError("source_separation is too large for this arena.")


# Fields of earlier versions that saved runs still store in config.json.
_IGNORED_FIELDS = ("render_mode", "render_size", "render_fps")
_FIXED_FIELDS = {"observation_mode": "odor", "odor_enabled": True}

# Load the configuration of an arena from an earlier run that was saved
def env_config_from_dict(values: dict) -> FruitEnvConfig:
    """The arena of a saved run (the ``environment`` entry of its config.json)."""
    values = {key: value for key, value in values.items() if key not in _IGNORED_FIELDS}
    for key, supported in _FIXED_FIELDS.items():
        value = values.pop(key, supported)
        if value != supported:
            raise ValueError(f"This run uses {key}={value!r}, which is no longer supported.")
    return FruitEnvConfig(**values)
