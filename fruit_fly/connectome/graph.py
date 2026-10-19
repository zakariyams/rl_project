"""The biological and shuffled antennal-lobe graphs used by the connectome actor.

A ``GraphSpec`` is plain numpy data (edge list, signs, input and readout
neurons). It is stored inside SB3 checkpoints by its import path
(``connectome.graph.GraphSpec``), so the class and its fields must not change.

Neurons: the 140 core neurons of both lobes (42 ORNs, 42 uPNs, 28 mPNs, 28 LNs)
with their 3,124 connections. Input: receptor channel r of a side drives that
side's ORN expressing receptor r. Readout: the 70 projection neurons (uPNs and
mPNs), the antennal lobe's outputs to the rest of the brain.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from connectome.berck2016 import side_of, DATA_DIR, load_circuit, sign
from connectome.randomize import randomize_connectome
from environment.receptors import RECEPTOR_TYPES
    

CORE_CLASSES = ("ORN", "uPN", "mPN", "LN")
READOUT_CLASSES = ("uPN", "mPN")


@dataclass(frozen=True)
class GraphSpec:
    kind: str                 # "bio" or "random"
    seed: int | None          # shuffle seed for "random"
    names: tuple[str, ...]
    classes: tuple[str, ...]
    pre: np.ndarray           # int edge sources
    post: np.ndarray          # int edge targets
    synapses: np.ndarray      # synapse count per edge (> 0)
    signs: np.ndarray         # per neuron: Each sign +1 excitatory, -1 inhibitory, 0 unknown (learned)
    orn_left: np.ndarray      # node indices driven by the left odor reading
    orn_right: np.ndarray
    readout: np.ndarray       # node indices read by the action layer

    @property
    def n(self) -> int:
        return len(self.names)

    # Combined set of edge sources and edge targets
    def edge_set(self) -> set[tuple[int, int]]:
        return set(zip(self.pre.tolist(), self.post.tolist()))

    # Counting the number of targets
    def in_degree(self) -> np.ndarray:
        return np.bincount(self.post, minlength=self.n)

    # Counting the number of sources
    def out_degree(self) -> np.ndarray:
        return np.bincount(self.pre, minlength=self.n)

    def init_magnitudes(self) -> np.ndarray:
        """Each edge's share of its target's synaptic input within the graph."""
        totals = np.bincount(self.post, weights=self.synapses, minlength=self.n)
        return self.synapses / totals[self.post]


def orns_by_receptor(names, classes, side: str) -> np.ndarray:
    """ORN node indices of one side, ordered like ``environment.receptors.RECEPTOR_TYPES``,
    so observation channel r drives the ORN expressing receptor r (one receptor per larval ORN)."""

    index = {n.removesuffix(f" ORN {side}"): i for i, (n, c) in enumerate(zip(names, classes))
             if c == "ORN" and side_of(n) == side}
    if set(index) != set(RECEPTOR_TYPES):
        raise ValueError(f"{side} ORNs {sorted(index)} do not match RECEPTOR_TYPES.")
    return np.array([index[r] for r in RECEPTOR_TYPES])


def bio_graph(data_dir=None) -> GraphSpec:
    """The real wiring: every connection between core neurons, weighted by synapse count.
        Returns the full specification of the graph
    """

    circuit = load_circuit(data_dir=data_dir or DATA_DIR)
    keep = [i for i, c in enumerate(circuit.classes) if c in CORE_CLASSES]
    names = [circuit.names[i] for i in keep]
    classes = [circuit.classes[i] for i in keep]
    M = circuit.matrix[np.ix_(keep, keep)]
    pre, post = np.nonzero(M)
    return GraphSpec(
        kind="bio", seed=None, names=tuple(names), classes=tuple(classes),
        pre=pre.astype(np.int64), post=post.astype(np.int64), synapses=M[pre, post].astype(np.float64),
        signs=np.array([sign(n) for n in names], dtype=np.int64),
        orn_left=orns_by_receptor(names, classes, "left"),
        orn_right=orns_by_receptor(names, classes, "right"),
        readout=np.array([i for i, c in enumerate(classes) if c in READOUT_CLASSES]),
    )


def build_graph(kind: str, seed: int = 0, data_dir=None) -> GraphSpec:
    """Build the bio-inspired graph from the fruit lavar data. The kind parameter determines if the
       returned graph is a "real" one or a shuffled version that preserves the same number of degrees"""

    graph = bio_graph(data_dir)
    if kind == "bio":
        return graph
    if kind == "random":
        return randomize_connectome(graph, seed=seed)
    raise ValueError(f"Unknown graph kind {kind!r}; expected 'bio' or 'random'.") # Should not happen, only bio and random graph is supported.
