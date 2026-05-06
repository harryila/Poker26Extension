"""
Single forward-pass helpers for activation-patching experiments.

The existing ``HFAgent`` calls ``model.generate(...)`` which loops through many
forward passes internally. For activation patching we need a SINGLE forward
pass on a fixed input (prompt + response_up_to_action_verb) so we can:

  1. Capture the residual at the LAST input position at every layer (source).
  2. Run again with a patching hook at one layer (target) and read logits at
     the LAST input position to score the effect of the patch.

This module provides:

  - ``PromptReconstructor``: rebuilds the EXACT input string that the original
    HFAgent constructed for a given enriched-log decision record. Reuses the
    private prompt-building helpers from HFAgent, so any future change to those
    helpers will require this module to be revisited.

  - ``run_forward_with_hooks``: runs ``model(input_ids=...)`` with a list of
    nn.Module hooks attached + detached safely. Returns logits at the LAST
    input position (as a CPU tensor) plus the input_ids actually used.

  - ``find_action_verb_input_position``: maps the action-verb position from a
    logit-lens sidecar (which counts only generated tokens) to the position
    inside the rebuilt input sequence. The input position is computed as
    ``len(prompt_token_ids) + sidecar_action_pos`` — the +0 (NOT -1) is because
    we tokenize ``prompt + response_up_to_action_verb_exclusive`` and the LAST
    input token IS the verb-predecessor; the position whose lm_head output
    becomes the verb prediction is ``len(input_ids) - 1``.

  - ``build_input_text_for_action_verb_position``: builds the string
    ``prompt + response_text[:offset_of_action_verb]`` so that the next-token
    prediction at the last input position is the action verb itself. The verb
    offset is found by re-tokenizing the recorded ``raw_response`` and finding
    where the verb token starts.

All callers should:

  - Use ``torch.no_grad()`` around forward calls.
  - Move ``input_ids`` to the model's device.
  - Detach hooks via try/finally to avoid leaks if a forward raises.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import torch

# Reuse HFAgent's private prompt builders. We instantiate HFAgent partially
# (without loading the model) by calling __init__ pieces directly — but the
# safest approach is to call the prompt builders as standalone functions where
# possible. For the action prompt we re-implement here using the same helpers
# HFAgent uses, to avoid importing the model loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from poker_env.actions import Action, ActionType  # noqa: E402
from poker_env.obs import Obs  # noqa: E402
from poker_env.agents.prompts import (  # noqa: E402
    get_action_system_message,
    get_template_id,
)
from poker_env.config import DEFAULT_MAX_HISTORY_EVENTS, get_model_config  # noqa: E402


# ---------------------------------------------------------------------------
# Obs reconstruction from enriched-log dict
# ---------------------------------------------------------------------------


def obs_from_dict(obs_dict: dict) -> Obs:
    """Inverse of ``Obs.to_dict``. Required because Obs has no from_dict."""
    legal = [Action.from_dict(la) for la in obs_dict.get("legal_actions", [])]
    return Obs(
        hand_id=obs_dict["hand_id"],
        seed=obs_dict["seed"],
        player_index=obs_dict["player_index"],
        num_players=obs_dict["num_players"],
        street=obs_dict["street"],
        street_index=obs_dict["street_index"],
        board=list(obs_dict.get("board", [])),
        hero_hole=list(obs_dict.get("hero_hole", [])),
        button=obs_dict.get("button", 0),
        position=obs_dict.get("position", ""),
        stacks=list(obs_dict.get("stacks", [])),
        pot_total=obs_dict.get("pot_total", 0),
        bet_to_call=obs_dict.get("bet_to_call", 0),
        raises_remaining=obs_dict.get("raises_remaining", 4),
        contrib_this_round=list(obs_dict.get("contrib_this_round", [])),
        contrib_total=list(obs_dict.get("contrib_total", [])),
        to_act=obs_dict.get("to_act", 0),
        legal_actions=legal,
        history=list(obs_dict.get("history", [])),
    )


# ---------------------------------------------------------------------------
# Prompt reconstruction (replicates HFAgent._build_action_prompt exactly)
# ---------------------------------------------------------------------------


def _format_history_brief(history: list[dict], hero_index: int) -> str:
    """Identical to HFAgent._format_history_brief — kept in sync."""
    if not history:
        return "No actions yet"
    lines = []
    for event in history[-10:]:
        event_type = event.get("event", "")
        player = event.get("player")
        amount = event.get("amount")
        if event_type in ("POST_BLIND", "DEAL_HOLE", "DEAL_BOARD", "UNKNOWN"):
            continue
        player_name = "You" if player == hero_index else "Opponent"
        if amount:
            lines.append(f"  {player_name}: {event_type} ({amount})")
        else:
            lines.append(f"  {player_name}: {event_type}")
    return "\n".join(lines) if lines else "No betting actions yet"


def _truncate_history(history: list[dict], max_events: int) -> list[dict]:
    """Identical to HFAgent._truncate_history."""
    if max_events <= 0:
        return []
    if len(history) <= max_events:
        return history
    if max_events <= 5:
        return history[-max_events:]
    return history[:5] + history[-(max_events - 5):]


def build_action_prompt(obs: Obs, *, cot_mode: bool, max_history_events: int) -> str:
    """Replicates ``HFAgent._build_action_prompt`` for an action decision.

    Pure function — no model needed. CPU-only.
    """
    truncated = _truncate_history(obs.history, max_history_events)
    legal_action_names = [a.type.value for a in obs.legal_actions]
    history_str = _format_history_brief(truncated, obs.player_index)
    if cot_mode:
        instruction = 'Provide your reasoning then your action as JSON: {"action": "YOUR_CHOICE"}'
    else:
        instruction = 'Return ONLY: {"action": "YOUR_CHOICE"}'
    return (
        f"Current poker situation:\n"
        f"- Your cards: {' '.join(obs.hero_hole)}\n"
        f"- Board: {' '.join(obs.board) if obs.board else 'None (preflop)'}\n"
        f"- Pot: {obs.pot_total}\n"
        f"- Street: {obs.street}\n"
        f"- Bet to call: {obs.bet_to_call}\n"
        f"- Legal actions: {legal_action_names}\n\n"
        f"Recent actions:\n{history_str}\n\n"
        f"Choose your action. {instruction}"
    )


# ---------------------------------------------------------------------------
# Chat template assembly (replicates HFAgent._generate's chat-template path)
# ---------------------------------------------------------------------------


def assemble_chat_prompt(
    user_prompt: str,
    system_message: str,
    *,
    tokenizer,
    supports_system_role: bool = True,
    has_thinking_mode: bool = False,
) -> str:
    """Returns the EXACT string the model saw as input (after chat template,
    before tokenization). Mirrors HFAgent._generate lines 471-489."""
    if supports_system_role:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": f"{system_message}\n\n{user_prompt}"},
        ]
    chat_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if has_thinking_mode:
        chat_kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **chat_kwargs)


# ---------------------------------------------------------------------------
# PromptReconstructor — convenience wrapper
# ---------------------------------------------------------------------------


class PromptReconstructor:
    """Rebuilds the chat-template input string for an enriched-log decision
    record, given the agent_config that produced the original run.

    No model required (CPU-only, tokenizer-only).

    Example::

        recon = PromptReconstructor(tokenizer, agent_config)
        prompt_text = recon.build(decision_record)
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    """

    def __init__(self, tokenizer, agent_config: dict):
        self.tokenizer = tokenizer
        self.cot_mode = bool(agent_config.get("cot_mode", False))
        self.max_history_events = int(
            agent_config.get("max_history_events", DEFAULT_MAX_HISTORY_EVENTS)
        )
        # supports_system_role / has_thinking_mode default to model_cfg values
        # if not explicitly stored in agent_config (older logs).
        model_id = agent_config.get("model_id", "")
        try:
            model_cfg = get_model_config(model_id)
        except Exception:
            model_cfg = {}
        self.supports_system_role = bool(
            agent_config.get(
                "supports_system_role",
                model_cfg.get("supports_system_role", True),
            )
        )
        self.has_thinking_mode = bool(
            agent_config.get(
                "has_thinking_mode",
                model_cfg.get("has_thinking_mode", False),
            )
        )

    def build_user_prompt(self, decision_record: dict) -> str:
        """Just the user-prompt portion (NO chat template applied)."""
        obs = obs_from_dict(decision_record["obs"])
        return build_action_prompt(
            obs,
            cot_mode=self.cot_mode,
            max_history_events=self.max_history_events,
        )

    def build(self, decision_record: dict) -> str:
        """Full chat-template-applied input string (matches model input)."""
        user_prompt = self.build_user_prompt(decision_record)
        system_message = get_action_system_message(cot=self.cot_mode)
        return assemble_chat_prompt(
            user_prompt,
            system_message,
            tokenizer=self.tokenizer,
            supports_system_role=self.supports_system_role,
            has_thinking_mode=self.has_thinking_mode,
        )

    def template_id(self) -> str:
        return get_template_id("action", "cot" if self.cot_mode else "default")


# ---------------------------------------------------------------------------
# Action-verb position mapping
# ---------------------------------------------------------------------------


# Action-verb token strings (full-word and common subword first-pieces) for
# locating the verb in a re-tokenized response. Conservatively matched against
# the *first* occurrence in the JSON payload tail of the response.
_VERB_FIRST_TOKENS = ("FOLD", "F", "fold", "Fol",
                      "CHECK", "Check", "check", "Che",
                      "BET", "Bet", "bet", "B",
                      "RAISE", "Raise", "raise", "Ra",
                      "CALL", "Call", "call")


def find_action_verb_response_offset(
    raw_response: str,
    tokenizer,
) -> tuple[int, int] | None:
    """Find the character offset of the action verb's first token inside
    ``raw_response``, plus the index of that token in the tokenized response.

    Strategy: locate the LAST ``"action"`` JSON key in the response, then find
    the next ``"`` (open-quote of value), the verb starts at next char.

    Returns ``(char_offset, response_token_index)`` or None if no verb found.
    """
    # Find the action JSON pattern at the END of the response.
    # Look for '"action"' or "'action'" then a colon, opening quote, verb.
    # Conservative: locate the LAST '"action"' substring.
    needle = '"action"'
    idx = raw_response.rfind(needle)
    if idx < 0:
        return None
    # After "action", find the next quote that opens the value.
    after_key = raw_response[idx + len(needle):]
    quote_pos_in_after = after_key.find('"')
    if quote_pos_in_after < 0:
        return None
    # Skip the opening quote itself; verb starts after it.
    char_offset = idx + len(needle) + quote_pos_in_after + 1

    # Find which token-in-response this offset corresponds to. We tokenize
    # the response and use offset_mapping when available.
    enc = tokenizer(raw_response, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping")
    if offsets is None:
        # Tokenizer doesn't support offset mapping; fall back to a simple
        # char-walk.
        return char_offset, -1
    for tok_idx, (start, end) in enumerate(offsets):
        if start <= char_offset < end:
            return char_offset, tok_idx
        if start >= char_offset:
            return char_offset, tok_idx
    return char_offset, len(offsets) - 1


def build_input_text_for_action_verb_position(
    prompt_text: str,
    raw_response: str,
    tokenizer,
) -> tuple[str, int] | None:
    """Build the input string ``prompt + response[:verb_offset]`` such that
    the next-token prediction at the LAST input position is the action verb.

    Returns ``(input_text, verb_token_index_within_response)`` or None if the
    verb cannot be located.
    """
    found = find_action_verb_response_offset(raw_response, tokenizer)
    if found is None:
        return None
    char_offset, verb_tok_idx = found
    return prompt_text + raw_response[:char_offset], verb_tok_idx


# ---------------------------------------------------------------------------
# Forward-pass helper
# ---------------------------------------------------------------------------


@contextmanager
def attached_hooks(hook_owners: Iterable[Any]):
    """Context manager that attaches hooks on enter, detaches on exit even on
    exception. Each hook_owner must expose ``attach()`` and ``detach()``."""
    owners = list(hook_owners)
    try:
        for owner in owners:
            owner.attach()
        yield
    finally:
        for owner in reversed(owners):
            try:
                owner.detach()
            except Exception:
                pass


def run_forward_at_last_position(
    model,
    tokenizer,
    input_text: str,
    *,
    return_full_logits: bool = False,
    device: str | None = None,
) -> dict:
    """Run a SINGLE forward pass on ``input_text`` and return next-token logits
    at the LAST input position.

    Returns a dict with:
      - ``logits_last_pos``: 1D CPU tensor of shape [vocab_size]
      - ``input_ids``: list[int] (so callers can decode positions for sanity)
      - ``last_input_token``: decoded string of the last input token
      - ``num_input_tokens``: int
      - ``full_logits``: present only if ``return_full_logits=True``,
        a [seq_len, vocab_size] CPU tensor (memory: ~400 MB for vocab=128k,
        seq_len=800; use sparingly).
    """
    enc = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    target_device = device or next(model.parameters()).device
    input_ids = enc["input_ids"].to(target_device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(target_device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = out.logits  # [batch, seq_len, vocab]
    last_pos_logits = logits[0, -1, :].detach().to("cpu")

    last_input_token_id = int(input_ids[0, -1].item())
    result = {
        "logits_last_pos": last_pos_logits,
        "input_ids": input_ids[0].detach().to("cpu").tolist(),
        "last_input_token": tokenizer.decode([last_input_token_id]),
        "num_input_tokens": int(input_ids.shape[1]),
    }
    if return_full_logits:
        result["full_logits"] = logits[0].detach().to("cpu")
    return result


__all__ = [
    "obs_from_dict",
    "build_action_prompt",
    "assemble_chat_prompt",
    "PromptReconstructor",
    "find_action_verb_response_offset",
    "build_input_text_for_action_verb_position",
    "attached_hooks",
    "run_forward_at_last_position",
]
