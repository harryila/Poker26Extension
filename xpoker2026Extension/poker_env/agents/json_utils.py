"""Shared JSON extraction utilities for LLM agent responses."""

import json
import re


# ----------------------------------------------------------------------------
# Action-name alias normalization
# ----------------------------------------------------------------------------
# Models occasionally emit colloquial action names like "CHECK", "CALL",
# "BET", "RAISE" instead of the canonical poker_env names CHECK_OR_CALL /
# BET_OR_RAISE. Normalize them BEFORE looking up the action in agent code so
# we don't miscategorize a vocabulary mismatch as a parser failure.
#
# Empirically discovered in the tier1a_small CoT grid (2026-05-03):
# Ministral 8B emitted {"action": "CHECK"} 47 times on seed 42 alone. See
# updates.md §11 / "Issue 1: alias mismatch".
ACTION_ALIASES: dict[str, str] = {
    # Single-word colloquial forms (Ministral 8B in CoT, multiple models
    # in non-CoT).
    "CHECK":            "CHECK_OR_CALL",
    "CALL":             "CHECK_OR_CALL",
    "CHECKING":         "CHECK_OR_CALL",
    "CALLING":          "CHECK_OR_CALL",
    "BET":              "BET_OR_RAISE",
    "RAISE":            "BET_OR_RAISE",
    "BETTING":          "BET_OR_RAISE",
    "RAISING":          "BET_OR_RAISE",
    "FOLDING":          "FOLD",
    # Word-order / duplicate variants of CHECK_OR_CALL.
    # Discovered in the original poker26 70B sanity logs (8-13% of decisions
    # per cell came back as CALL_OR_CHECK or CALL_OR_CALL). Both
    # unambiguously refer to the CHECK_OR_CALL action (same chip cost, same
    # legal-action slot). NOT aliasing CALL_OR_RAISE / RAISE_OR_CALL because
    # those are genuinely ambiguous.
    "CALL_OR_CHECK":    "CHECK_OR_CALL",
    "CALL_OR_CALL":     "CHECK_OR_CALL",
    "CHECK_OR_CHECK":   "CHECK_OR_CALL",
    # Word-order variant of BET_OR_RAISE.
    "RAISE_OR_BET":     "BET_OR_RAISE",
}


def normalize_action_str(action_str: str) -> str:
    """Map a model-emitted action string to the canonical poker_env name.

    Pass-through for already-canonical strings (FOLD, CHECK_OR_CALL,
    BET_OR_RAISE) and for unknown strings (caller will reject).
    """
    if not isinstance(action_str, str):
        return action_str
    upper = action_str.strip().upper()
    return ACTION_ALIASES.get(upper, upper)


def extract_json(text: str) -> dict | None:
    """
    Extract JSON from a potentially chatty LLM response.

    Handles:
    - Code fences (```json ... ```)
    - Multiple JSON blocks (tries smallest valid first)
    - Trailing commas (basic cleanup)

    Args:
        text: Raw LLM response

    Returns:
        Parsed dict or None on failure
    """
    # Strip code fences if present
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Find all top-level {...} blocks by tracking brace depth,
    # correctly skipping braces inside quoted strings.
    matches = []
    depth = 0
    start = -1
    in_string = False
    escape = False

    for i, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == '\\' and in_string:
            escape = True
            continue
        if char == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start != -1:
                matches.append(text[start:i + 1])
                start = -1

    # Try each match, shortest first (likely cleaner)
    matches.sort(key=len)

    for match in matches:
        cleaned = re.sub(r',\s*}', '}', match)
        cleaned = re.sub(r',\s*]', ']', cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            continue

    return None


def parse_cot_response(raw: str) -> tuple[str | None, dict | None]:
    """
    Parse a Chain-of-Thought response into reasoning text and JSON.

    Expects format like:
        REASONING: <analysis>

        JSON: {"action": "..."}
    or:
        REASONING: <analysis>

        {"action": "..."}

    Also handles responses where reasoning comes before a JSON block
    without explicit markers.

    Args:
        raw: Full LLM response text

    Returns:
        (reasoning_text, parsed_json) -- either may be None on failure
    """
    reasoning = None
    parsed = None

    # Try explicit REASONING: / JSON: markers first
    reasoning_match = re.search(
        r'REASONING:\s*(.*?)(?=\n\s*(?:JSON:|PROBABILITIES:|```|\{))',
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # If no explicit marker, treat everything before the first JSON block as reasoning
    if reasoning is None:
        json_start = raw.find('{')
        if json_start > 0:
            candidate = raw[:json_start].strip()
            # Remove any "JSON:" prefix
            candidate = re.sub(r'^(?:JSON|PROBABILITIES)\s*:\s*', '', candidate, flags=re.IGNORECASE).strip()
            if len(candidate) > 10:
                reasoning = candidate

    parsed = extract_json(raw)
    return reasoning, parsed
