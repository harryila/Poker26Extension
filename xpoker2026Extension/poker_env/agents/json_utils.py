"""Shared JSON extraction utilities for LLM agent responses."""

import json
import re


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
