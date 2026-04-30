"""Agent interfaces and baseline implementations."""

from poker_env.agents.base import BaseAgent
from poker_env.agents.random_agent import RandomAgent
from poker_env.agents.call_agent import CallAgent
from poker_env.agents.threshold_agent import ThresholdAgent

__all__ = ["BaseAgent", "RandomAgent", "CallAgent", "ThresholdAgent"]

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

# Conditional import for APIAgent (requires openai, anthropic, or google-genai)
try:
    from poker_env.agents.api_agent import APIAgent, APIMetadata
    __all__.extend(["APIAgent", "APIMetadata"])
    API_AVAILABLE = True
except ImportError:
    APIAgent = None
    APIMetadata = None
    API_AVAILABLE = False
