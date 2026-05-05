"""
API-based LLM agent for closed models (OpenAI, Anthropic, Google).

Implements the same BaseAgent + metadata interface as HFAgent so that
run_experiment.py, DecisionLogger, and the analysis pipeline work
identically regardless of whether the model is local or remote.
"""

import os
import time
import hashlib
from dataclasses import dataclass

from poker_env.agents.base import BaseAgent
from poker_env.agents.json_utils import (
    extract_json,
    normalize_action_str,
    parse_cot_response,
)
from poker_env.agents.prompts import (
    get_action_system_message,
    get_belief_system_message,
    get_template_id,
)
from poker_env.actions import Action, FOLD, CHECK_OR_CALL, BET_OR_RAISE
from poker_env.obs import Obs
from poker_env.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_ACTION_MAX_TOKENS,
    DEFAULT_BELIEF_MAX_TOKENS,
    DEFAULT_COT_ACTION_MAX_TOKENS,
    DEFAULT_COT_BELIEF_MAX_TOKENS,
    DEFAULT_MAX_HISTORY_EVENTS,
    DEFAULT_BELIEF_FORMAT,
    BELIEF_SCHEMA_ID,
    BUCKET_ORDER,
)

# ============================================================================
# Metadata
# ============================================================================

@dataclass
class APIMetadata:
    """Metadata from an API call.

    Diagnostic action fields (added 2026-05-03 — parity with
    HFAgent.ActionMetadata, see updates.md §11):
      action_json_parsed / action_recognized / action_legal_in_context
    Default to None on belief calls and on old logs.
    """
    parse_success: bool
    fallback_used: bool
    raw_response: str
    action_chosen: str | None
    prompt_hash: str
    prompt_template_id: str
    provider: str
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    logprobs: list[dict] | None = None
    history_events_before: int = 0
    history_events_after: int = 0
    was_truncated: bool = False
    # Belief-specific telemetry (parity with HFAgent)
    prob_sum: float | None = None
    prob_min: float | None = None
    negative_count: int | None = None
    repair_distance_l1: float | None = None
    repair_distance_l2: float | None = None
    belief_schema_id: str | None = None
    # Action-failure diagnostic split — see class docstring.
    action_json_parsed: bool | None = None
    action_recognized: bool | None = None
    action_legal_in_context: bool | None = None

    def to_dict(self) -> dict:
        d = {
            "parse_success": self.parse_success,
            "fallback_used": self.fallback_used,
            "raw_response": self.raw_response,
            "action_chosen": self.action_chosen,
            "prompt_hash": self.prompt_hash,
            "prompt_template_id": self.prompt_template_id,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "history_events_before": self.history_events_before,
            "history_events_after": self.history_events_after,
            "was_truncated": self.was_truncated,
        }
        if self.logprobs is not None:
            d["logprobs"] = self.logprobs
        if self.prob_sum is not None:
            d["prob_sum"] = self.prob_sum
        if self.prob_min is not None:
            d["prob_min"] = self.prob_min
        if self.negative_count is not None:
            d["negative_count"] = self.negative_count
        if self.repair_distance_l1 is not None:
            d["repair_distance_l1"] = self.repair_distance_l1
        if self.repair_distance_l2 is not None:
            d["repair_distance_l2"] = self.repair_distance_l2
        if self.belief_schema_id is not None:
            d["belief_schema_id"] = self.belief_schema_id
        if self.action_json_parsed is not None:
            d["action_json_parsed"] = self.action_json_parsed
        if self.action_recognized is not None:
            d["action_recognized"] = self.action_recognized
        if self.action_legal_in_context is not None:
            d["action_legal_in_context"] = self.action_legal_in_context
        return d


# ============================================================================
# APIAgent
# ============================================================================

class APIAgent(BaseAgent):
    """
    LLM agent backed by a remote API (OpenAI, Anthropic, or Google).

    Provides the same act_with_metadata / belief_with_metadata interface
    as HFAgent so that run_experiment and analysis work identically.

    For OpenAI models, logprobs are requested and stored in metadata.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        action_max_new_tokens: int | None = None,
        belief_max_new_tokens: int | None = None,
        max_history_events: int | None = None,
        belief_format: str | None = None,
        cot_mode: bool = False,
        name: str = "APIAgent",
    ):
        super().__init__(name=name)
        self.provider = provider.lower()
        self.cot_mode = cot_mode
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.top_p = top_p if top_p is not None else DEFAULT_TOP_P
        self.max_history_events = max_history_events if max_history_events is not None else DEFAULT_MAX_HISTORY_EVENTS
        self.belief_format = belief_format if belief_format is not None else DEFAULT_BELIEF_FORMAT

        if cot_mode:
            self.action_max_new_tokens = action_max_new_tokens if action_max_new_tokens is not None else DEFAULT_COT_ACTION_MAX_TOKENS
            self.belief_max_new_tokens = belief_max_new_tokens if belief_max_new_tokens is not None else DEFAULT_COT_BELIEF_MAX_TOKENS
        else:
            self.action_max_new_tokens = action_max_new_tokens if action_max_new_tokens is not None else DEFAULT_ACTION_MAX_TOKENS
            self.belief_max_new_tokens = belief_max_new_tokens if belief_max_new_tokens is not None else DEFAULT_BELIEF_MAX_TOKENS

        if self.provider == "openai":
            import openai
            self.model = model or "gpt-4o"
            key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self._client = openai.OpenAI(api_key=key)
        elif self.provider == "anthropic":
            import anthropic
            self.model = model or "claude-sonnet-4-20250514"
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            self._client = anthropic.Anthropic(api_key=key)
        elif self.provider == "google":
            from google import genai
            self.model = model or "gemini-2.0-flash"
            key = api_key or os.environ.get("GOOGLE_API_KEY", "")
            self._client = genai.Client(api_key=key)
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                "Use 'openai', 'anthropic', or 'google'."
            )

        self._last_action_metadata: APIMetadata | None = None
        self._last_belief_metadata: APIMetadata | None = None
        self._last_action_cot_reasoning: str | None = None
        self._last_belief_cot_reasoning: str | None = None

    # ========================================================================
    # Core Interface
    # ========================================================================

    def act(self, obs: Obs) -> Action:
        action, metadata = self.act_with_metadata(obs)
        self._last_action_metadata = metadata
        return action

    def act_with_metadata(self, obs: Obs) -> tuple[Action, APIMetadata]:
        original_len = len(obs.history)
        truncated_history = self._truncate_history(obs.history)
        truncated_len = len(truncated_history)
        was_truncated = truncated_len < original_len

        prompt = self._build_action_prompt(obs, truncated_history)
        prompt_hash = self._hash_prompt(prompt)
        system_message = get_action_system_message(cot=self.cot_mode)
        template_id = get_template_id("action", "cot" if self.cot_mode else "default")

        raw_response, call_meta = self._call_api(
            system_message=system_message,
            user_prompt=prompt,
            max_tokens=self.action_max_new_tokens,
        )

        base = {
            "history_events_before": original_len,
            "history_events_after": truncated_len,
            "was_truncated": was_truncated,
        }

        if self.cot_mode:
            reasoning, parsed = parse_cot_response(raw_response)
            self._last_action_cot_reasoning = reasoning
        else:
            parsed = extract_json(raw_response)
            self._last_action_cot_reasoning = None

        action_map = {"FOLD": FOLD, "CHECK_OR_CALL": CHECK_OR_CALL, "BET_OR_RAISE": BET_OR_RAISE}

        json_parsed = bool(parsed and "action" in parsed and isinstance(parsed["action"], str))
        recognized = False
        legal = False
        action_str_for_log: str | None = None
        action_obj = None

        if json_parsed:
            action_str_for_log = normalize_action_str(parsed["action"])
            action_obj = action_map.get(action_str_for_log)
            recognized = action_obj is not None
            legal = recognized and action_obj in obs.legal_actions

        if json_parsed and recognized and legal:
            return action_obj, APIMetadata(
                parse_success=True, fallback_used=False,
                raw_response=raw_response, action_chosen=action_str_for_log,
                prompt_hash=prompt_hash, prompt_template_id=template_id,
                action_json_parsed=True,
                action_recognized=True,
                action_legal_in_context=True,
                **call_meta, **base,
            )

        return self._fallback_action(
            obs, raw_response, prompt_hash, template_id, call_meta, base,
            action_json_parsed=json_parsed,
            action_recognized=recognized,
            action_legal_in_context=legal,
        )

    def belief(self, obs: Obs) -> dict | None:
        belief, metadata = self.belief_with_metadata(obs)
        self._last_belief_metadata = metadata
        return belief

    def belief_with_metadata(self, obs: Obs) -> tuple[dict | None, APIMetadata]:
        original_len = len(obs.history)
        truncated_history = self._truncate_history(obs.history)
        truncated_len = len(truncated_history)
        was_truncated = truncated_len < original_len

        if self.belief_format == "compact":
            prompt = self._build_compact_belief_prompt(obs, truncated_history)
            template_id = get_template_id("belief", "compact_cot" if self.cot_mode else "compact")
        else:
            prompt = self._build_full_belief_prompt(obs, truncated_history)
            template_id = get_template_id("belief", "full_cot" if self.cot_mode else "full")

        system_message = get_belief_system_message(
            belief_format=self.belief_format,
            cot=self.cot_mode,
        )
        prompt_hash = self._hash_prompt(prompt)

        raw_response, call_meta = self._call_api(
            system_message=system_message,
            user_prompt=prompt,
            max_tokens=self.belief_max_new_tokens,
        )

        base = {
            "history_events_before": original_len,
            "history_events_after": truncated_len,
            "was_truncated": was_truncated,
        }

        if self.cot_mode:
            reasoning, parsed = parse_cot_response(raw_response)
            self._last_belief_cot_reasoning = reasoning
        else:
            parsed = extract_json(raw_response)
            self._last_belief_cot_reasoning = None

        if parsed:
            belief = self._parse_belief(parsed)
            if belief is not None:
                import numpy as _np
                vals = list(belief.values())
                raw_sum = sum(vals)
                raw_min = min(vals)
                neg_count = sum(1 for v in vals if v < 0)
                clipped = [max(0.0, v) for v in vals]
                clip_sum = sum(clipped)
                if clip_sum > 0:
                    normed = [v / clip_sum for v in clipped]
                else:
                    n_buckets = len(clipped)
                    normed = [1.0 / n_buckets] * n_buckets if n_buckets > 0 else clipped
                repair_l1 = sum(abs(a - b) for a, b in zip(vals, normed))
                repair_l2 = float(_np.sqrt(sum((a - b) ** 2 for a, b in zip(vals, normed))))

                return belief, APIMetadata(
                    parse_success=True, fallback_used=False,
                    raw_response=raw_response, action_chosen=None,
                    prompt_hash=prompt_hash, prompt_template_id=template_id,
                    prob_sum=raw_sum, prob_min=raw_min,
                    negative_count=neg_count,
                    repair_distance_l1=repair_l1,
                    repair_distance_l2=repair_l2,
                    belief_schema_id=BELIEF_SCHEMA_ID,
                    **call_meta, **base,
                )

        return None, APIMetadata(
            parse_success=False, fallback_used=False,
            raw_response=raw_response, action_chosen=None,
            prompt_hash=prompt_hash, prompt_template_id=template_id,
            **call_meta, **base,
        )

    # ========================================================================
    # Metadata Accessors
    # ========================================================================

    def get_last_action_metadata(self) -> APIMetadata | None:
        return self._last_action_metadata

    def get_last_belief_metadata(self) -> APIMetadata | None:
        return self._last_belief_metadata

    def get_last_cot_reasoning(self) -> str | None:
        if self._last_belief_cot_reasoning is not None:
            return self._last_belief_cot_reasoning
        return self._last_action_cot_reasoning

    def get_last_action_cot(self) -> str | None:
        return self._last_action_cot_reasoning

    def get_last_belief_cot(self) -> str | None:
        return self._last_belief_cot_reasoning

    def get_config(self) -> dict:
        return {
            "type": "APIAgent",
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "action_max_new_tokens": self.action_max_new_tokens,
            "belief_max_new_tokens": self.belief_max_new_tokens,
            "belief_format": self.belief_format,
            "cot_mode": self.cot_mode,
        }

    # ========================================================================
    # API Call
    # ========================================================================

    def _call_api(
        self,
        system_message: str,
        user_prompt: str,
        max_tokens: int,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> tuple[str, dict]:
        """
        Call the provider API and return (response_text, metadata_dict).

        metadata_dict contains: provider, model, latency_ms, prompt_tokens,
        completion_tokens, finish_reason, logprobs (OpenAI only).

        Retries transient errors with exponential backoff.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                if self.provider == "openai":
                    return self._call_openai(system_message, user_prompt, max_tokens, t0)
                elif self.provider == "google":
                    return self._call_google(system_message, user_prompt, max_tokens, t0)
                return self._call_anthropic(system_message, user_prompt, max_tokens, t0)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    import warnings
                    warnings.warn(f"API call attempt {attempt + 1} failed ({e}), retrying in {delay:.1f}s")
                    time.sleep(delay)
        raise RuntimeError(f"API call failed after {max_retries} attempts: {last_error}") from last_error

    def _call_openai(
        self, system_message: str, user_prompt: str, max_tokens: int, t0: float,
    ) -> tuple[str, dict]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=20,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        choice = response.choices[0]
        text = choice.message.content or ""

        # Extract logprobs
        logprobs_data = None
        if choice.logprobs and choice.logprobs.content:
            logprobs_data = [
                {
                    "token": tok.token,
                    "logprob": tok.logprob,
                    "top_logprobs": [
                        {"token": tl.token, "logprob": tl.logprob}
                        for tl in (tok.top_logprobs or [])
                    ],
                }
                for tok in choice.logprobs.content
            ]

        meta = {
            "provider": "openai",
            "model": self.model,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
            "finish_reason": choice.finish_reason,
            "logprobs": logprobs_data,
        }
        return text.strip(), meta

    def _call_anthropic(
        self, system_message: str, user_prompt: str, max_tokens: int, t0: float,
    ) -> tuple[str, dict]:
        response = self._client.messages.create(
            model=self.model,
            system=system_message,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = ""
        if response.content:
            text_parts = [
                getattr(block, "text", "")
                for block in response.content
                if getattr(block, "type", None) == "text"
            ]
            text = "".join(text_parts) if text_parts else (getattr(response.content[0], "text", "") or "")

        meta = {
            "provider": "anthropic",
            "model": self.model,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": response.usage.input_tokens if response.usage else None,
            "completion_tokens": response.usage.output_tokens if response.usage else None,
            "finish_reason": response.stop_reason,
            "logprobs": None,
        }
        return text.strip(), meta

    def _call_google(
        self, system_message: str, user_prompt: str, max_tokens: int, t0: float,
    ) -> tuple[str, dict]:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_message,
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=max_tokens,
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        try:
            text = response.text or ""
        except (ValueError, AttributeError):
            text = ""

        prompt_tokens = None
        completion_tokens = None
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count

        finish_reason = None
        if response.candidates:
            try:
                finish_reason = response.candidates[0].finish_reason.name
            except (AttributeError, IndexError):
                pass

        meta = {
            "provider": "google",
            "model": self.model,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "logprobs": None,
        }
        return text.strip(), meta

    # ========================================================================
    # Belief Parsing
    # ========================================================================

    def _parse_belief(self, parsed: dict) -> dict | None:
        """Parse a belief dict in either compact or full format."""
        if self.belief_format == "compact":
            probs = parsed.get("probs", [])
            if not isinstance(probs, list) or len(probs) != len(BUCKET_ORDER):
                return None
            try:
                probs = [float(p) for p in probs]
            except (ValueError, TypeError):
                return None
            if all(p == 0.0 for p in probs):
                return None
            return dict(zip(BUCKET_ORDER, probs))

        # Full format
        belief = {}
        for name in BUCKET_ORDER:
            val = parsed.get(name)
            if val is not None:
                try:
                    belief[name] = float(val)
                except (ValueError, TypeError):
                    belief[name] = 0.0
            else:
                belief[name] = 0.0
        if all(v == 0.0 for v in belief.values()):
            return None
        return belief

    # ========================================================================
    # Prompt Building (mirrors HFAgent)
    # ========================================================================

    def _build_action_prompt(self, obs: Obs, truncated_history: list[dict]) -> str:
        legal_action_names = [a.type.value for a in obs.legal_actions]
        history_str = self._format_history_brief(truncated_history, obs.player_index)

        if self.cot_mode:
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

    def _build_compact_belief_prompt(self, obs: Obs, truncated_history: list[dict]) -> str:
        action_summary = self._format_action_summary(truncated_history, obs.player_index)
        bucket_order_str = ", ".join(f"{i} {name}" for i, name in enumerate(BUCKET_ORDER))

        if self.cot_mode:
            instruction = "Provide your reasoning then the JSON object."
        else:
            instruction = "Return ONLY the JSON object."

        return (
            f"Bucket order (indices 0-13):\n{bucket_order_str}.\n\n"
            f"Your cards: {' '.join(obs.hero_hole)}\n"
            f"Board: {' '.join(obs.board) if obs.board else 'None (preflop)'}\n"
            f"Pot: {obs.pot_total}\n"
            f"Opponent's actions: {action_summary}\n\n"
            f"{instruction}"
        )

    def _build_full_belief_prompt(self, obs: Obs, truncated_history: list[dict]) -> str:
        from analysis.prompts import format_belief_prompt
        return format_belief_prompt(
            hero_hole=obs.hero_hole,
            board=obs.board,
            pot=obs.pot_total,
            street=obs.street,
            history=truncated_history,
            template="default",
            hero_index=obs.player_index,
        )

    # ========================================================================
    # Helpers
    # ========================================================================

    def _format_history_brief(self, history: list[dict], hero_index: int) -> str:
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

    def _format_action_summary(self, history: list[dict], hero_index: int) -> str:
        opponent_actions = []
        for event in history:
            event_type = event.get("event", event.get("op", ""))
            player = event.get("player")
            if player is not None and player != hero_index:
                if event_type in ("FOLD", "Folding"):
                    opponent_actions.append("folded")
                elif event_type in ("CHECK", "CheckingOrCalling"):
                    opponent_actions.append("check/called")
                elif event_type in ("CALL",):
                    opponent_actions.append("called")
                elif event_type in ("BET", "RAISE", "CompletionBettingOrRaisingTo"):
                    opponent_actions.append("bet/raised")
        return ", ".join(opponent_actions) if opponent_actions else "No opponent actions"

    def _truncate_history(self, history: list[dict], max_events: int | None = None) -> list[dict]:
        if max_events is None:
            max_events = self.max_history_events
        if max_events <= 0:
            return []
        if len(history) <= max_events:
            return history
        if max_events <= 5:
            return history[-max_events:]
        return history[:5] + history[-(max_events - 5):]

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def _fallback_action(
        self, obs: Obs, raw_response: str, prompt_hash: str,
        template_id: str, call_meta: dict, base: dict,
        action_json_parsed: bool | None = None,
        action_recognized: bool | None = None,
        action_legal_in_context: bool | None = None,
    ) -> tuple[Action, APIMetadata]:
        if not obs.legal_actions:
            raise ValueError("No legal actions available for fallback")
        if CHECK_OR_CALL in obs.legal_actions:
            chosen = CHECK_OR_CALL
        elif FOLD in obs.legal_actions:
            chosen = FOLD
        else:
            chosen = obs.legal_actions[0]
        return chosen, APIMetadata(
            parse_success=False, fallback_used=True,
            raw_response=raw_response, action_chosen=chosen.type.value,
            prompt_hash=prompt_hash, prompt_template_id=template_id,
            action_json_parsed=action_json_parsed,
            action_recognized=action_recognized,
            action_legal_in_context=action_legal_in_context,
            **call_meta, **base,
        )
