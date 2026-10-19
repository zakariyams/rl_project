"""Every agent of the study behind one interface: random, the two hand-coded rules and trained models."""

from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.rule_agents import InformedFollowAgent, OdorFollowAgent
from agents.trained_agent import TrainedAgent

__all__ = ["BaseAgent", "InformedFollowAgent", "OdorFollowAgent", "RandomAgent", "TrainedAgent"]
