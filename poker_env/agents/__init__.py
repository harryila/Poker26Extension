"""Agent interfaces and baseline implementations."""

from poker_env.agents.base import BaseAgent
from poker_env.agents.random_agent import RandomAgent
from poker_env.agents.call_agent import CallAgent

__all__ = ["BaseAgent", "RandomAgent", "CallAgent"]

# Conditional import for HFAgent (requires torch and transformers)
try:
    from poker_env.agents.hf_agent import HFAgent, ActionMetadata, BeliefMetadata
    __all__.extend(["HFAgent", "ActionMetadata", "BeliefMetadata"])
    HF_AVAILABLE = True
except ImportError:
    HFAgent = None
    ActionMetadata = None
    BeliefMetadata = None
    HF_AVAILABLE = False
