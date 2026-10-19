"""Connectome actor for Stable-Baselines3 PPO (report section 4.3).

The actor is the larval antennal-lobe circuit run as a firing-rate network on the
graph's real connections; the critic is the same 64x64 Tanh MLP as in PPO-MLP.
Only the actor differs between PPO agents.

Saved checkpoints refer to ``connectome.policy.ConnectomePolicy`` by this import
path and store its constructor arguments, so neither may change.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class ConnectomeNet(nn.Module):
    """The antennal-lobe circuit as a rate network; returns the readout PN activity."""

    # Initialize the network
    def __init__(self, graph, settle_steps: int = 4, input_gain: float = 1.0, input_bias: float = 8.0,
                 readout_norm: bool = True):
        super().__init__()
        self.n, self.settle_steps, self.readout_norm = graph.n, int(settle_steps), bool(readout_norm)
        # Each edge starts at its share of the target neuron's synaptic input (more
        # synapses = stronger connection), so every neuron's input weights sum to one.
        magnitudes = graph.init_magnitudes()
        # An edge's sign is its presynaptic neuron's sign
        signs = graph.signs[graph.pre]
        known = signs != 0
        # Fixed graph structure: buffers are saved with the model but never trained.
        self.register_buffer("pre", torch.as_tensor(graph.pre, dtype=torch.long))
        self.register_buffer("post", torch.as_tensor(graph.post, dtype=torch.long))
        self.register_buffer("edge_sign", torch.as_tensor(np.where(known, signs, 1), dtype=torch.float32))
        self.register_buffer("known", torch.as_tensor(known))
        self.register_buffer("readout", torch.as_tensor(graph.readout, dtype=torch.long))
        self.register_buffer("orn_left", torch.as_tensor(graph.orn_left, dtype=torch.long))
        self.register_buffer("orn_right", torch.as_tensor(graph.orn_right, dtype=torch.long))
        # theta: log-magnitude for known-sign edges, raw weight for unknown-sign edges.
        init = np.where(known, np.log(magnitudes), magnitudes)
        self.theta = nn.Parameter(torch.as_tensor(init, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(self.n))
        # Learned input gain and bias per ORN. Starting at gain 1 and bias 8 keeps the
        # ORN drive positive for log-odor readings down to -8, i.e. across the whole arena.
        orns = len(graph.orn_left) + len(graph.orn_right)
        self.orn_gain = nn.Parameter(torch.full((orns,), float(input_gain)))
        self.orn_bias = nn.Parameter(torch.full((orns,), float(input_bias)))
        self.latent_dim = len(graph.readout)

    # Set weights at each edge
    def edge_weights(self) -> torch.Tensor:
        # Known sign: sign * exp(theta), so training can scale the weight but never flip it.
        # Unknown sign: theta itself, free to become positive or negative.
        return torch.where(self.known, self.edge_sign * torch.exp(self.theta), self.theta)

    # Return a weight matrix for the network
    def weight_matrix(self) -> torch.Tensor:
        W = torch.zeros(self.n, self.n, device=self.theta.device, dtype=self.theta.dtype)
        return W.index_put((self.pre, self.post), self.edge_weights())

    # Forward propagation in the network
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        orn_idx = torch.cat([self.orn_left, self.orn_right])
        drive = torch.zeros(batch, self.n, device=features.device, dtype=features.dtype)
        drive[:, orn_idx] = self.orn_gain * features + self.orn_bias
        W = self.weight_matrix()
        # Firing-rate dynamics r_k+1 = softplus(r_k @ W + u + b), starting from rest (r_0 = 0)
        # at every environment step, so the network has no memory between steps.
        rate = torch.zeros_like(drive)
        for _ in range(self.settle_steps):
            # Softplus keeps rates non-negative without permanently silent neurons
            rate = nn.functional.softplus(rate @ W + drive + self.bias) # Use softplus as activation function
        # Readout: the 70 projection neurons, the antennal lobe's outputs to the brain.
        latent = rate[:, self.readout]
        if self.readout_norm:
            latent = nn.functional.layer_norm(latent, (self.latent_dim,)) # Apply layer normalization
        # SB3's action layer turns this latent into 3 action logits.
        return latent



class ConnectomeActorCritic(nn.Module):
    """SB3 mlp_extractor replacement: connectome actor, standard MLP critic."""

    def __init__(self, feature_dim: int, graph, settle_steps: int, readout_norm: bool = True):
        super().__init__()
        channels = len(graph.orn_left) + len(graph.orn_right)
        if feature_dim != channels:
            raise ValueError(f"Connectome agents expect one odor feature per ORN ({channels}); got {feature_dim}.")
        self.policy_net = ConnectomeNet(graph, settle_steps, readout_norm=readout_norm)
        self.value_net = nn.Sequential(nn.Linear(feature_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.latent_dim_pi = self.policy_net.latent_dim
        self.latent_dim_vf = 64

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class ConnectomePolicy(ActorCriticPolicy):
    """PPO actor-critic policy whose actor is a ``ConnectomeNet`` network on ``graph`` following the structure in GraphSpec."""

    def __init__(self, observation_space, action_space, lr_schedule, *, graph, settle_steps: int = 4,
                 readout_norm: bool = True, **kwargs):
        self.graph = graph
        self.settle_steps = settle_steps
        self.readout_norm = readout_norm
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # Overrides the super class function defined in SB3 and replaces the extractor with a custom made one. 
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ConnectomeActorCritic(self.features_dim, self.graph, self.settle_steps,
                                                   readout_norm=self.readout_norm) # Uses the extractor that we implemented instead of the SB3 mlp extractor.

    # Similar here, overrides the superclass function defined in SB3.
    def _get_constructor_parameters(self):
        # Saved with the checkpoint, so loading rebuilds the same graph without the raw tables.
        data = super()._get_constructor_parameters()
        data.update(graph=self.graph, settle_steps=self.settle_steps, readout_norm=self.readout_norm)
        return data
