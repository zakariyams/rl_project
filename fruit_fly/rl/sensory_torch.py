"""Torch module of the fixed log-odor sensory stage (see rl/sensory.py).

Saved checkpoints refer to ``rl.sensory_torch.LogOdorExtractor`` by this import
path, so the module must not be moved or renamed.
"""
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LogOdorExtractor(BaseFeaturesExtractor):
    """Parameter-free ``log(clip(odor, floor, 1))``; output size equals input size.

    ``floor`` has one value per observation entry (``rl.sensory.channel_floors``),
    fixed when the run is created and saved with the checkpoint.
    """

    def __init__(self, observation_space, floor):
        super().__init__(observation_space, features_dim=int(observation_space.shape[0]))
        # A plain attribute, not a buffer: the extractor has no state in the checkpoint's weights.
        self.floor = tuple(float(f) for f in floor)
        if len(self.floor) != self.features_dim:
            raise ValueError(f"{len(self.floor)} floors for {self.features_dim} inputs.")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        floor = torch.as_tensor(self.floor, dtype=observations.dtype, device=observations.device)
        return torch.log(torch.maximum(observations.clamp(max=1.0), floor))
