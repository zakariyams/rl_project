"""Odor identity: receptor response profiles of the fruit and poison.

Each odor source has a fixed response profile over
the larval receptor types: profile[r] in [0, 1] is how strongly that odor
activates receptor r at full concentration. The sensor reading of channel r is
the sum over sources of ``profile[r] * exp(-decay * distance)``, so odor
identity is a pattern across channels and intensity is its overall level.

The profiles are SYNTHETIC. They are drawn from a fixed seed with a sparse,
heavy-tailed shape (most receptors respond weakly, a few strongly), which is the
qualitative shape of measured larval Or responses, but they are not measured
data. Real profiles (e.g. Kreher et al. 2005/2008, larval Or responses to an
odorant panel) can replace ``odor_profiles`` without changing anything else.

``odor_similarity`` controls how hard discrimination is: the poison profile is a
mix of the fruit profile and an independent profile, blended until their cosine
similarity reaches the requested value (0 = unrelated draws, 1 = identical and
therefore indistinguishable).
"""
from __future__ import annotations

import numpy as np

# Larval receptor types, in the order of the Berck 2016 ORNs (connectome/graph.py
# maps channel r to the ORN expressing RECEPTOR_TYPES[r] on each side).
RECEPTOR_TYPES = ("1a", "13a", "22c", "24a", "30a", "33a", "35a", "42a", "42b", "45a", "45b",
                  "47a & 33b", "49a", "59a", "63a", "67b", "74a", "82a", "83a", "85c", "94a & 94b")

# Draws a sample from a gamma distribution
def _draw(rng: np.random.Generator, channels: int) -> np.ndarray:
    raw = rng.gamma(shape=0.6, scale=1.0, size=channels)
    return raw / raw.max()

# Computes cosine similarity
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def odor_profiles(channels: int, similarity: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(fruit, poison) profiles with cosine(fruit, poison) == ``similarity``.

    Both are normalized so their strongest receptor responds with 1.0. If the
    independent draw is already more similar than requested, it is used as is;
    """
    if channels < 1:
        raise ValueError("channels must be positive.")
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("odor_similarity must be in [0, 1].")
    rng = np.random.default_rng(seed)
    fruit = _draw(rng, channels)
    other = _draw(rng, channels)
    if similarity <= cosine(fruit, other):
        return fruit, other
    low, high = 0.0, 1.0          
    for _ in range(60):
        mid = (low + high) / 2
        if cosine(fruit, mid * fruit + (1 - mid) * other) < similarity:
            low = mid
        else:
            high = mid
    poison = high * fruit + (1 - high) * other
    return fruit, poison / poison.max()

