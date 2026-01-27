"""
Central configuration for LLM agent defaults.

IMPORTANT: To change the default model, edit DEFAULT_MODEL_ID below.
For research runs with 70B, use: --hf-model meta-llama/Llama-3.1-70B-Instruct

This file is the SINGLE SOURCE OF TRUTH for model configuration.
All other files should import from here.
"""

# =============================================================================
# Model Configuration
# =============================================================================

# Safe default for development/CI - runs on most GPUs
DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Research model - requires 96GB+ VRAM (GH200, H100, or multi-GPU)
RESEARCH_MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"

# =============================================================================
# Generation Defaults
# =============================================================================

DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9  # For sampling stability when do_sample=True

# Separate token budgets for action vs belief
DEFAULT_ACTION_MAX_TOKENS = 64   # Actions need few tokens: {"action": "CHECK_OR_CALL"}
DEFAULT_BELIEF_MAX_TOKENS = 384  # Beliefs need more (compact format ~50-100 tokens, buffer for safety)

DEFAULT_MAX_INPUT_TOKENS = 2048
DEFAULT_MAX_HISTORY_EVENTS = 50

# =============================================================================
# Belief Format Configuration
# =============================================================================

# "compact" = fixed-order list: {"schema":"buckets_14_v1","probs":[...]}
# "full" = labeled dict: {"premium_pairs": 0.05, "strong_pairs": 0.08, ...}
DEFAULT_BELIEF_FORMAT = "compact"

# Versioned schema identifier - embedded in belief JSON for self-description
BELIEF_SCHEMA_ID = "buckets_14_v1"

# Bucket order for compact format (indices 0-13)
# This MUST match analysis/buckets.py BUCKET_NAMES order
BUCKET_ORDER = [
    "premium_pairs",      # 0: AA, KK, QQ
    "strong_pairs",       # 1: JJ, TT
    "medium_pairs",       # 2: 99-66
    "small_pairs",        # 3: 55-22
    "premium_broadway",   # 4: AKs, AKo, AQs
    "strong_broadway",    # 5: AQo, AJs, KQs, ATs
    "medium_broadway",    # 6: KQo, KJs, QJs, etc.
    "suited_aces",        # 7: A9s-A2s
    "suited_connectors",  # 8: T9s-54s
    "suited_gappers",     # 9: J9s, T8s, etc.
    "offsuit_connectors", # 10: T9o-65o
    "weak_broadway",      # 11: KTo, QTo, etc.
    "speculative_suited", # 12: small suited cards
    "trash",              # 13: everything else
]
