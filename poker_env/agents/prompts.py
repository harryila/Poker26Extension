"""Prompt templates for LLM poker agents."""

ACTION_SYSTEM_MESSAGE = "You are a poker player. Choose your action.\nReturn ONLY valid JSON. No other text.\nFormat: {\"action\": \"FOLD\" | \"CHECK_OR_CALL\" | \"BET_OR_RAISE\"}"

BELIEF_SYSTEM_MESSAGE = "You are analyzing opponent hand ranges in poker.\nReturn ONLY valid JSON. No other text.\nProbabilities must be non-negative and sum to 1.0."

TEMPLATE_IDS = {
    "action_default": "action_strict_v1",
    "action_minimal": "action_minimal_v1",
    "belief_default": "belief_strict_v1",
}

def get_template_id(prompt_type, template="default"):
    key = f"{prompt_type}_{template}"
    return TEMPLATE_IDS.get(key, f"{prompt_type}_{template}_v1")
