"""
Mechanistic interpretability tools for LLM poker agents.

Provides:
- LogitLensExtractor: layer-by-layer belief formation analysis
- ProbeDataset / LinearProbe: linear probing on hidden states
- AttentionExtractor: attention pattern analysis
- AttributionAnalyzer: input saliency via integrated gradients
"""

__all__: list[str] = []

try:
    from poker_env.interp.logit_lens import LogitLensExtractor
    __all__.append("LogitLensExtractor")
except ImportError:
    pass

try:
    from poker_env.interp.probing import ProbeDataset, LinearProbe
    __all__.extend(["ProbeDataset", "LinearProbe"])
except ImportError:
    pass

try:
    from poker_env.interp.attention import AttentionExtractor
    __all__.append("AttentionExtractor")
except ImportError:
    pass

try:
    from poker_env.interp.attribution import AttributionAnalyzer
    __all__.append("AttributionAnalyzer")
except ImportError:
    pass
