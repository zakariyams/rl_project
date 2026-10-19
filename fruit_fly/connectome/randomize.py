"""Degree-preserving randomization of a directed connectome graph.

This module implements the function that build the randomized counterpart to the
connectome graph. 
In order to keep it comparable to the connectome graph it is important that we preserve 
the degrees meaning each neuron keeps its exact in-degree and out-degreeits exact in-degree and out-degree
as well as other additional information.
In order to do this we adopt Directed double-edge swaps (Maslow & Sneppen 2002)
For example: edges a->b and c->d become a->d and c->b.

"""
from __future__ import annotations

from dataclasses import replace

import numpy as np


def randomize_connectome(graph, seed: int = 0, swaps_per_edge: int = 10):
    """A shuffled copy of ``graph`` after ``swaps_per_edge`` successful swaps per movable edge."""
    rng = np.random.default_rng(seed)
    pre, post, syn = graph.pre.copy(), graph.post.copy(), graph.synapses.copy()
    movable = np.flatnonzero(pre != post)
    edges = set(zip(pre.tolist(), post.tolist()))
    done, attempts = 0, 0
    target = swaps_per_edge * len(movable)
    while done < target and attempts < 50 * target:
        attempts += 1
        i, j = rng.choice(movable, size=2, replace=False)
        a, b, c, d = pre[i], post[i], pre[j], post[j]
        if a == d or c == b or (a, d) in edges or (c, b) in edges:
            continue
        edges -= {(a, b), (c, d)}
        edges |= {(a, d), (c, b)}
        post[i], post[j] = d, b   # swap targets; sources and their synapse counts stay put
        done += 1
    if done < target:
        raise RuntimeError(f"Only {done} of {target} swaps succeeded; graph too constrained.")
    return replace(graph, kind="random", seed=seed, pre=pre, post=post, synapses=syn)
