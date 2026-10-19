"""This file contains the 2 simple hand-coded rule agents, the odor rule agent and the informed rule agent     """
from __future__ import annotations
import numpy as np
from agents.base_agent import BaseAgent, Observation
from environment.fruit_env import FORWARD, TURN_LEFT, TURN_RIGHT


class OdorFollowAgent(BaseAgent):
    """Odor rule: steer toward the stronger side; turn when walking forward lost odor.

    Two classic insect reflexes, no learnable parameters learned:

    * Tropotaxis (Braitenberg steering): if the relative left/right difference
      ``(L - R) / (L + R)`` exceeds ``deadband``, turn toward the stronger side,
      otherwise walk forward. The ratio is scale-free, so it works at any distance.
    * If the last step was forward and total odor fell, the fly is
      heading away from the source (where both antennas read almost the same),
      so turn toward the stronger side until the odor rises again.

    It sums all receptor channels of a side, so it cannot tell the fruit from the
    poison and heads for whichever smell is stronger. It reads only the
    observation, exactly like the trained agents.

    The balance is about ``tanh(decay * offset * sin(bearing))``, so one 15-degree
    turn moves it by up to ~0.02. A smaller deadband makes the fly wobble left and
    right without walking; 0.02 is that step size.
    """

    def __init__(self, deadband: float = 0.02) -> None:
        if deadband < 0:
            raise ValueError("deadband cannot be negative.")
        self.deadband = deadband
        self.reset(0)

    def reset(self, seed: int) -> None:
        del seed  # the rule is deterministic
        self._last_action: int | None = None
        self._last_total: float | None = None

    def act(self, observation: Observation) -> int:
        left, right = self.side_signals(observation)    # Get the observation signal
        total = left + right
        balance = (left - right) / total if total > 0 else 0.0
        toward_stronger = TURN_LEFT if balance >= 0 else TURN_RIGHT # Read which signal is strongest and turn toward the strongest one.
        lost_odor = self._last_action == FORWARD and self._last_total is not None and total < self._last_total #See if we are heading away from the source.
        if lost_odor or abs(balance) > self.deadband:
            action = toward_stronger
        else:
            action = FORWARD
        self._last_action, self._last_total = action, total
        return action

    def side_signals(self, observation: Observation) -> tuple[float, float]:
        """One odor value per side: the sum of its receptor channels."""
        half = len(observation) // 2
        return float(np.sum(observation[:half])), float(np.sum(observation[half:])) # Return the left and the right side of the receptor channels.


class InformedFollowAgent(OdorFollowAgent):
    """Informed rule: the same rule, told what the learners must discover, which odor is the fruit.

    Given the fruit and poison receptor profiles, each side's channels are a linear
    mix ``c_fruit * fruit + c_poison * poison`` (environment/odor.py, before
    saturation), so a least-squares fit recovers the fruit's concentration and the
    rule follows that alone. It shows what a designer who knows the answer can
    build; a trained agent only gets the reward.
    """

    def __init__(self, fruit_profile, other_profiles=(), deadband: float = 0.02) -> None:
        basis = [np.asarray(fruit_profile, dtype=np.float64)] + [np.asarray(p, dtype=np.float64)
                                                                  for p in other_profiles]
        self._unmix = np.linalg.pinv(np.stack(basis, axis=1))[0]   # row giving the fruit coefficient
        super().__init__(deadband)

    def side_signals(self, observation: Observation) -> tuple[float, float]:
        half = len(observation) // 2
        if half != len(self._unmix):
            raise ValueError(f"Profiles have {len(self._unmix)} channels; observation has {half} per side.")
        sides = np.asarray(observation, dtype=np.float64).reshape(2, half)
        left, right = np.maximum(sides @ self._unmix, 0.0)
        return float(left), float(right)

    @classmethod
    def for_env(cls, env, deadband: float = 0.02) -> "InformedFollowAgent":
        """Built from a FruitEnv's fixed fruit and poison profiles."""
        env = env.unwrapped
        others = [env.poison_profile] if env.config.poison_enabled else []
        return cls(env.fruit_profile, others, deadband)
