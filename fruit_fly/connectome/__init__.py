"""Connectome actor: the larval antennal-lobe circuit of Berck et al. (2016) as a policy network.

berck2016.py reads the published connectivity tables, graph.py turns them into the
biological graph (or its degree-preserving shuffle, randomize.py), and policy.py
is the trainable network that PPO uses as the actor.
"""
