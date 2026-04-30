"""
Single source of truth for all agent-facing prompt templates and system messages.

Both HFAgent and APIAgent import from here. Analysis-side templates
(bucket descriptions, belief elicitation formatting) live in analysis/prompts.py.
"""

try:
    from poker_env.config import BELIEF_SCHEMA_ID, BUCKET_ORDER
except ImportError:
    BELIEF_SCHEMA_ID = "buckets_14_v1"
    BUCKET_ORDER = []

# ============================================================================
# Action System Messages
# ============================================================================

ACTION_SYSTEM_MESSAGE = (
    "You are a poker player. Choose your action.\n"
    "Return ONLY valid JSON. No other text.\n"
    'Format: {"action": "FOLD" | "CHECK_OR_CALL" | "BET_OR_RAISE"}'
)

ACTION_COT_SYSTEM_MESSAGE = (
    "You are a poker player. Analyze the situation step-by-step, then choose your action.\n"
    "First write REASONING: followed by your analysis (2-4 sentences).\n"
    "Then write JSON: followed by ONLY valid JSON on a new line.\n"
    'Format: {"action": "FOLD" | "CHECK_OR_CALL" | "BET_OR_RAISE"}\n\n'
    "Example:\n"
    "REASONING: The opponent raised preflop and bet the flop, suggesting a strong hand. "
    "My hand is weak with little draw potential, so folding is best.\n\n"
    'JSON: {"action": "FOLD"}'
)

# ============================================================================
# Belief System Messages
# ============================================================================

BELIEF_SYSTEM_MESSAGE_COMPACT = (
    "You are analyzing opponent hand ranges in poker. "
    "Return ONLY valid JSON. No other text.\n"
    f'Output format: {{"schema":"{BELIEF_SCHEMA_ID}","probs":[p0,p1,...,p13]}}\n'
    "Constraints: each pi must be a decimal between 0 and 1, non-negative, "
    "and the 14 values must sum to 1.0 exactly (adjust the last value to fix rounding). "
    "Use at most 3 decimal places."
)

BELIEF_SYSTEM_MESSAGE_FULL = (
    "You are analyzing opponent hand ranges in poker.\n"
    "Return ONLY valid JSON. No other text.\n"
    "Probabilities must be non-negative and sum to 1.0."
)

BELIEF_COT_SYSTEM_MESSAGE_COMPACT = (
    "You are analyzing opponent hand ranges in poker.\n"
    "First write REASONING: followed by your range analysis (2-4 sentences).\n"
    "Consider what hands would take the actions your opponent took.\n"
    "Then write JSON: followed by ONLY valid JSON on a new line.\n"
    f'Format: {{"schema":"{BELIEF_SCHEMA_ID}","probs":[p0,p1,...,p13]}}\n'
    "Constraints: 14 non-negative decimals summing to 1.0, at most 3 decimal places.\n\n"
    "Example:\n"
    "REASONING: Opponent raised preflop and bet the flop. This suggests premium pairs "
    "or strong broadway hands. Weaker hands would likely check or fold.\n\n"
    f'JSON: {{"schema":"{BELIEF_SCHEMA_ID}","probs":[0.15,0.12,0.08,0.03,0.12,0.10,0.08,'
    "0.05,0.04,0.04,0.03,0.06,0.05,0.05]}"
)

BELIEF_COT_SYSTEM_MESSAGE_FULL = (
    "You are analyzing opponent hand ranges in poker.\n"
    "First write REASONING: followed by your range analysis (2-4 sentences).\n"
    "Consider what hands would take the actions your opponent took.\n"
    "Then write JSON: followed by ONLY valid JSON on a new line.\n"
    "Probabilities must be non-negative and sum to 1.0."
)

# ============================================================================
# Template IDs
# ============================================================================

TEMPLATE_IDS = {
    "action_default": "action_strict_v1",
    "action_cot": "action_cot_v1",
    "action_minimal": "action_minimal_v1",
    "belief_compact": "belief_compact_v1",
    "belief_full": "belief_full_v1",
    "belief_compact_cot": "belief_compact_cot_v1",
    "belief_full_cot": "belief_full_cot_v1",
}


def get_template_id(prompt_type: str, template: str = "default") -> str:
    key = f"{prompt_type}_{template}"
    return TEMPLATE_IDS.get(key, f"{prompt_type}_{template}_v1")


def get_action_system_message(cot: bool = False) -> str:
    return ACTION_COT_SYSTEM_MESSAGE if cot else ACTION_SYSTEM_MESSAGE


def get_belief_system_message(belief_format: str = "compact", cot: bool = False) -> str:
    if cot:
        if belief_format == "compact":
            return BELIEF_COT_SYSTEM_MESSAGE_COMPACT
        return BELIEF_COT_SYSTEM_MESSAGE_FULL
    if belief_format == "compact":
        return BELIEF_SYSTEM_MESSAGE_COMPACT
    return BELIEF_SYSTEM_MESSAGE_FULL
