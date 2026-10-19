"""Fixed log-odor sensory stage shared by every learned agent.

Odor readings fall off exponentially with distance, so far from a source both
sensors read tiny values whose left/right difference is too small for a network
to use. Every policy therefore sees ``log(clip(reading, floor, 1))`` instead:
in log space the left/right contrast depends on the bearing to the source, not
on its distance, much like the roughly logarithmic concentration coding of
olfactory receptor neurons.

The stage has no trainable parameters and sits in front of every network (MLP,
both connectome graphs and DQN), so it never favours one architecture. The torch
module that applies it is ``rl.sensory_torch.LogOdorExtractor``.
"""
from __future__ import annotations

import math

import numpy as np


def odor_floor(config) -> float:
    """Smallest concentration that still carries information in this arena.

    The larger of the weakest real reading (a source at the farthest possible
    distance, about 1.2e-3 in the 10 x 10 arena) and the noise level (a reading
    below the noise is indistinguishable from it).
    """
    farthest = math.hypot(config.arena_width - config.fruit_radius, config.arena_height - config.fruit_radius)
    return min(1.0, max(math.exp(-config.odor_decay * farthest), config.odor_noise_std))


def channel_floors(config, profiles) -> list[float]:
    """One floor per observation entry ([left channels..., right channels...]).

    Channel r reads ``profile[r] * concentration``, so its weakest reading is the
    concentration floor scaled by receptor r's strongest response to any source.
    A single floor for all channels would clip weakly responding receptors.
    """
    strongest = np.max(np.asarray(profiles, dtype=np.float64), axis=0)
    floors = np.clip(strongest * odor_floor(config), 1e-12, 1.0).tolist()
    return floors + floors


def arena_floors(env) -> list[float]:
    """Floors for the FruitEnv inside ``env`` (Gymnasium wrappers allowed)."""
    unwrapped = env.unwrapped
    profiles = [unwrapped.fruit_profile] + ([unwrapped.poison_profile] if unwrapped.config.poison_enabled else [])
    return channel_floors(unwrapped.config, profiles)


def log_odor(observation, floors) -> np.ndarray:
    """NumPy version of the transform (the policies use the torch module)."""
    return np.log(np.clip(np.asarray(observation, dtype=np.float64), floors, 1.0))


def log_odor_policy_kwargs(env) -> dict:
    """Stable-Baselines3 ``policy_kwargs`` entries that put the log stage in front of the network."""
    from rl.sensory_torch import LogOdorExtractor
    return {"features_extractor_class": LogOdorExtractor,
            "features_extractor_kwargs": {"floor": [float(f) for f in arena_floors(env)]}}
