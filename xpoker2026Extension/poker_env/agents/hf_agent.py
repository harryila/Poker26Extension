"""
HuggingFace-based LLM agent for poker with belief elicitation.

Implements the BaseAgent interface using instruction-tuned models
(Llama 3.1, Mistral, Qwen, etc.). Provides full telemetry for
parse success, coherence diagnostics, and reproducibility.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from poker_env.agents.base import BaseAgent
from poker_env.agents.json_utils import extract_json, parse_cot_response
from poker_env.agents.prompts import (
    get_action_system_message,
    get_belief_system_message,
    get_template_id,
)
from poker_env.actions import Action, FOLD, CHECK_OR_CALL, BET_OR_RAISE
from poker_env.obs import Obs
from poker_env.config import (
    DEFAULT_MODEL_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_ACTION_MAX_TOKENS,
    DEFAULT_BELIEF_MAX_TOKENS,
    DEFAULT_COT_ACTION_MAX_TOKENS,
    DEFAULT_COT_BELIEF_MAX_TOKENS,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_HISTORY_EVENTS,
    DEFAULT_BELIEF_FORMAT,
    BELIEF_SCHEMA_ID,
    BUCKET_ORDER,
    get_model_config,
    resolve_model_id,
)


# ============================================================================
# Metadata Dataclasses
# ============================================================================

@dataclass
class ActionMetadata:
    """Metadata about action selection for logging."""
    parse_success: bool
    fallback_used: bool
    raw_response: str
    action_chosen: str
    prompt_hash: str
    prompt_template_id: str = "action_strict_v1"
    history_events_before: int = 0
    history_events_after: int = 0
    was_truncated: bool = False
    # Per-token logprobs (parity with APIMetadata; OpenAI-compatible shape).
    # Each entry: {"token": str, "logprob": float, "top_logprobs": [{"token", "logprob"}]}
    logprobs: list[dict] | None = None

    def to_dict(self) -> dict:
        d = {
            "parse_success": self.parse_success,
            "fallback_used": self.fallback_used,
            "raw_response": self.raw_response,
            "action_chosen": self.action_chosen,
            "prompt_hash": self.prompt_hash,
            "prompt_template_id": self.prompt_template_id,
            "history_events_before": self.history_events_before,
            "history_events_after": self.history_events_after,
            "was_truncated": self.was_truncated,
        }
        if self.logprobs is not None:
            d["logprobs"] = self.logprobs
        return d


@dataclass
class BeliefMetadata:
    """Metadata about belief elicitation for logging."""
    parse_success: bool
    raw_response: str
    prob_sum: float | None
    prob_min: float | None
    negative_count: int
    prompt_hash: str
    prompt_template_id: str = "belief_compact_v1"
    belief_format: str = "compact"
    belief_schema_id: str = BELIEF_SCHEMA_ID
    history_events_before: int = 0
    history_events_after: int = 0
    was_truncated: bool = False
    repair_distance_l1: float | None = None
    repair_distance_l2: float | None = None
    logprobs: list[dict] | None = None

    def to_dict(self) -> dict:
        d = {
            "parse_success": self.parse_success,
            "raw_response": self.raw_response,
            "prob_sum": self.prob_sum,
            "prob_min": self.prob_min,
            "negative_count": self.negative_count,
            "prompt_hash": self.prompt_hash,
            "prompt_template_id": self.prompt_template_id,
            "belief_format": self.belief_format,
            "belief_schema_id": self.belief_schema_id,
            "history_events_before": self.history_events_before,
            "history_events_after": self.history_events_after,
            "was_truncated": self.was_truncated,
            "repair_distance_l1": self.repair_distance_l1,
            "repair_distance_l2": self.repair_distance_l2,
        }
        if self.logprobs is not None:
            d["logprobs"] = self.logprobs
        return d


# ============================================================================
# HFAgent Class
# ============================================================================

class HFAgent(BaseAgent):
    """
    HuggingFace-based LLM agent with belief elicitation.

    Supports Llama, Mistral, Qwen and other instruction-tuned models.
    Handles models with and without native system-role chat templates.

    Features:
    - Strict JSON output enforcement via chat templates
    - Deterministic fallback on parse failure
    - Chain-of-Thought (CoT) prompting mode
    - Logit lens hook integration for interpretability
    - Full telemetry for parse success, coherence diagnostics
    """

    def __init__(
        self,
        model_id: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        action_max_new_tokens: int | None = None,
        belief_max_new_tokens: int | None = None,
        max_input_tokens: int | None = None,
        max_history_events: int | None = None,
        belief_format: str | None = None,
        cot_mode: bool = False,
        logit_lens: bool = False,
        capture_logprobs: bool = False,
        top_logprobs: int = 20,
        name: str = "HFAgent",
        device_map: str = "auto",
    ):
        super().__init__(name=name)

        raw_model_id = model_id if model_id is not None else DEFAULT_MODEL_ID
        model_cfg = get_model_config(raw_model_id)
        self.model_id = resolve_model_id(raw_model_id)
        self.supports_system_role = model_cfg.get("supports_system_role", True)
        # Qwen 3 ships an `enable_thinking` switch in its chat template that is ON
        # by default. For non-CoT runs we MUST disable it, otherwise the model
        # silently performs internal CoT and the cross-family baseline is invalid.
        self.has_thinking_mode = model_cfg.get("has_thinking_mode", False)

        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.top_p = top_p if top_p is not None else DEFAULT_TOP_P
        self.max_input_tokens = max_input_tokens if max_input_tokens is not None else DEFAULT_MAX_INPUT_TOKENS
        self.max_history_events = max_history_events if max_history_events is not None else DEFAULT_MAX_HISTORY_EVENTS
        self.belief_format = belief_format if belief_format is not None else DEFAULT_BELIEF_FORMAT
        self.cot_mode = cot_mode
        self.logit_lens_enabled = logit_lens
        self.capture_logprobs = capture_logprobs
        self.top_logprobs = max(0, int(top_logprobs))

        # Token budgets (larger when CoT is on)
        if cot_mode:
            self.action_max_new_tokens = action_max_new_tokens if action_max_new_tokens is not None else DEFAULT_COT_ACTION_MAX_TOKENS
            self.belief_max_new_tokens = belief_max_new_tokens if belief_max_new_tokens is not None else DEFAULT_COT_BELIEF_MAX_TOKENS
        else:
            self.action_max_new_tokens = action_max_new_tokens if action_max_new_tokens is not None else DEFAULT_ACTION_MAX_TOKENS
            self.belief_max_new_tokens = belief_max_new_tokens if belief_max_new_tokens is not None else DEFAULT_BELIEF_MAX_TOKENS

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model.eval()

        # Logit-lens extractor (lazy import to avoid hard dep on interp module)
        self._logit_lens_extractor = None
        self._attention_extractor = None
        if self.logit_lens_enabled:
            try:
                from poker_env.interp.logit_lens import LogitLensExtractor
                self._logit_lens_extractor = LogitLensExtractor(self.model, self.tokenizer)
            except ImportError:
                self.logit_lens_enabled = False
            try:
                from poker_env.interp.attention import AttentionExtractor
                self._attention_extractor = AttentionExtractor(self.model, self.tokenizer)
            except (ImportError, AttributeError):
                pass

        # Metadata storage
        self._last_action_metadata: ActionMetadata | None = None
        self._last_belief_metadata: BeliefMetadata | None = None
        self._last_action_cot_reasoning: str | None = None
        self._last_belief_cot_reasoning: str | None = None
        self._last_logit_lens_data: dict | None = None
        self._last_attention_data: dict | None = None
        self._last_hidden_states: dict | None = None
        self._last_truncation_info: dict = {}
        self._last_generation_logprobs: list[dict] | None = None

    # ========================================================================
    # Core Interface
    # ========================================================================

    def act(self, obs: Obs) -> Action:
        action, metadata = self.act_with_metadata(obs)
        self._last_action_metadata = metadata
        return action

    def act_with_metadata(self, obs: Obs) -> tuple[Action, ActionMetadata]:
        original_len = len(obs.history)
        truncated_history = self._truncate_history(obs.history)
        truncated_len = len(truncated_history)
        was_truncated = truncated_len < original_len

        prompt = self._build_action_prompt(obs, truncated_history)
        prompt_hash = self._hash_prompt(prompt)
        system_message = get_action_system_message(cot=self.cot_mode)
        template_id = get_template_id("action", "cot" if self.cot_mode else "default")

        raw_response = self._generate(
            prompt,
            system_message=system_message,
            max_new_tokens=self.action_max_new_tokens,
        )

        base_metadata = {
            "history_events_before": original_len,
            "history_events_after": truncated_len,
            "was_truncated": was_truncated,
            "logprobs": self._last_generation_logprobs,
        }

        # Parse response (CoT or direct)
        if self.cot_mode:
            reasoning, parsed = parse_cot_response(raw_response)
            self._last_action_cot_reasoning = reasoning
            self._last_cot_source = "action"
        else:
            parsed = extract_json(raw_response)
            self._last_action_cot_reasoning = None

        if parsed and "action" in parsed and isinstance(parsed["action"], str):
            action_str = parsed["action"].upper()

            action_map = {
                "FOLD": FOLD,
                "CHECK_OR_CALL": CHECK_OR_CALL,
                "BET_OR_RAISE": BET_OR_RAISE,
            }
            action_obj = action_map.get(action_str)
            if action_obj and action_obj in obs.legal_actions:
                return action_obj, ActionMetadata(
                    parse_success=True,
                    fallback_used=False,
                    raw_response=raw_response,
                    action_chosen=action_str,
                    prompt_hash=prompt_hash,
                    prompt_template_id=template_id,
                    **base_metadata,
                )

        return self._fallback_action(obs, raw_response, prompt_hash, template_id, base_metadata)

    def belief(self, obs: Obs) -> dict | None:
        belief, metadata = self.belief_with_metadata(obs)
        self._last_belief_metadata = metadata
        return belief

    def belief_with_metadata(self, obs: Obs) -> tuple[dict | None, BeliefMetadata]:
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

        raw_response = self._generate(
            prompt,
            system_message=system_message,
            max_new_tokens=self.belief_max_new_tokens,
            collect_interp=False,
        )

        base_metadata = {
            "history_events_before": original_len,
            "history_events_after": truncated_len,
            "was_truncated": was_truncated,
            "belief_format": self.belief_format,
            "belief_schema_id": BELIEF_SCHEMA_ID,
            "prompt_template_id": template_id,
            "logprobs": self._last_generation_logprobs,
        }

        # Parse (CoT or direct)
        if self.cot_mode:
            reasoning, parsed = parse_cot_response(raw_response)
            self._last_belief_cot_reasoning = reasoning
            self._last_cot_source = "belief"
        else:
            parsed = extract_json(raw_response)
            self._last_belief_cot_reasoning = None

        if parsed:
            if self.belief_format == "compact":
                return self._parse_compact_belief(parsed, raw_response, prompt_hash, base_metadata)
            return self._parse_full_belief(parsed, raw_response, prompt_hash, base_metadata)

        return None, BeliefMetadata(
            parse_success=False,
            raw_response=raw_response,
            prob_sum=None,
            prob_min=None,
            negative_count=0,
            prompt_hash=prompt_hash,
            **base_metadata,
        )

    # ========================================================================
    # Metadata Accessors
    # ========================================================================

    def get_last_action_metadata(self) -> ActionMetadata | None:
        return self._last_action_metadata

    def get_last_belief_metadata(self) -> BeliefMetadata | None:
        return self._last_belief_metadata

    def get_last_cot_reasoning(self) -> str | None:
        """Return the most recent CoT reasoning (action or belief, whichever ran last)."""
        last_source = getattr(self, '_last_cot_source', None)
        if last_source == "belief":
            if self._last_belief_cot_reasoning is not None:
                return self._last_belief_cot_reasoning
            return self._last_action_cot_reasoning
        if self._last_action_cot_reasoning is not None:
            return self._last_action_cot_reasoning
        return self._last_belief_cot_reasoning

    def get_last_action_cot(self) -> str | None:
        return self._last_action_cot_reasoning

    def get_last_belief_cot(self) -> str | None:
        return self._last_belief_cot_reasoning

    def get_last_logit_lens_data(self) -> dict | None:
        return self._last_logit_lens_data

    def get_last_attention_data(self) -> dict | None:
        return self._last_attention_data

    def get_last_hidden_states(self) -> dict | None:
        return self._last_hidden_states

    def get_config(self) -> dict:
        """Return agent configuration for logging."""
        return {
            "type": "HFAgent",
            "model_id": self.model_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "action_max_new_tokens": self.action_max_new_tokens,
            "belief_max_new_tokens": self.belief_max_new_tokens,
            "max_input_tokens": self.max_input_tokens,
            "belief_format": self.belief_format,
            "cot_mode": self.cot_mode,
            "logit_lens": self.logit_lens_enabled,
            "capture_logprobs": self.capture_logprobs,
            "top_logprobs": self.top_logprobs if self.capture_logprobs else 0,
            "has_thinking_mode": self.has_thinking_mode,
            "enable_thinking": (bool(self.cot_mode) if self.has_thinking_mode else None),
        }

    # ========================================================================
    # Generation
    # ========================================================================

    def _generate(self, user_prompt: str, system_message: str, max_new_tokens: int, collect_interp: bool = True) -> str:
        """Generate response using chat template, with optional logit-lens hooks."""
        if self.supports_system_role:
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
        if self.has_thinking_mode:
            chat_kwargs["enable_thinking"] = bool(self.cot_mode)
        prompt = self.tokenizer.apply_chat_template(messages, **chat_kwargs)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        full_token_count = len(self.tokenizer.encode(prompt))
        actual_token_count = inputs["input_ids"].shape[-1]
        self._last_truncation_info = {
            "tokenizer_truncated": actual_token_count < full_token_count,
            "full_token_count": full_token_count,
            "actual_token_count": actual_token_count,
        }
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if self.temperature == 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        # Logprob capture requires per-step scores. Returning a dict makes
        # extraction cleaner regardless of whether scores are requested.
        if self.capture_logprobs:
            gen_kwargs["output_scores"] = True
        gen_kwargs["return_dict_in_generate"] = True

        # Attach interpretability hooks only for the action generation
        if collect_interp and self._logit_lens_extractor:
            self._logit_lens_extractor.attach_hooks()
        if collect_interp and self._attention_extractor:
            self._attention_extractor.attach_hooks()
            gen_kwargs["output_attentions"] = True

        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
        finally:
            if collect_interp and self._logit_lens_extractor:
                self._last_logit_lens_data = self._logit_lens_extractor.collect()
                self._logit_lens_extractor.detach_hooks()
                self._export_hidden_states_for_probing()
            if collect_interp and self._attention_extractor:
                snapshot = self._attention_extractor.collect(inputs.get("input_ids"))
                self._last_attention_data = snapshot.to_dict() if snapshot else None
                self._attention_extractor.detach_hooks()

        input_length = inputs["input_ids"].shape[1]
        sequences = getattr(outputs, "sequences", outputs)
        generated_ids = sequences[0][input_length:]
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        if self.capture_logprobs:
            self._last_generation_logprobs = self._extract_logprobs(
                scores=getattr(outputs, "scores", None),
                generated_ids=generated_ids,
            )
        else:
            self._last_generation_logprobs = None

        return response.strip()

    def _extract_logprobs(
        self,
        scores: tuple | None,
        generated_ids,
    ) -> list[dict] | None:
        """Convert per-step generation scores into an OpenAI-compatible logprobs list.

        Returns None if scores are unavailable (e.g. older transformers releases
        that do not honor output_scores for the active sampling path).
        """
        if scores is None or len(scores) == 0:
            return None

        import torch.nn.functional as F

        result: list[dict] = []
        n_steps = min(len(scores), int(generated_ids.shape[-1]))
        top_k = self.top_logprobs

        for step in range(n_steps):
            step_logits = scores[step][0]
            step_logprobs = F.log_softmax(step_logits.float(), dim=-1)
            tok_id = int(generated_ids[step].item())
            chosen_lp = float(step_logprobs[tok_id].item())
            entry: dict = {
                "token": self.tokenizer.decode([tok_id], skip_special_tokens=False),
                "logprob": chosen_lp,
                "top_logprobs": [],
            }
            if top_k > 0:
                k = min(top_k, step_logprobs.shape[-1])
                topk_vals, topk_ids = step_logprobs.topk(k)
                entry["top_logprobs"] = [
                    {
                        "token": self.tokenizer.decode([int(tid.item())], skip_special_tokens=False),
                        "logprob": float(val.item()),
                    }
                    for val, tid in zip(topk_vals, topk_ids)
                ]
            result.append(entry)

        return result

    def _export_hidden_states_for_probing(self) -> None:
        """Extract per-layer mean hidden vectors from the last logit-lens capture for probing."""
        ll = self._last_logit_lens_data
        if not ll or ll.get("num_layers", 0) == 0:
            self._last_hidden_states = None
            return

        # The logit lens already captured per-layer top tokens and entropy;
        # we store a lightweight summary (mean hidden norm per layer) for probing datasets.
        # Full hidden vectors are too large for JSONL but the per-layer entropy
        # and top-token data from logit lens serves as a proxy.
        self._last_hidden_states = {
            "num_layers": ll["num_layers"],
            "per_layer_entropy": ll.get("per_layer_entropy", []),
        }

    # ========================================================================
    # Belief Parsing
    # ========================================================================

    def _compute_repair_distance(self, probs: list[float]) -> tuple[float, float]:
        repaired = [max(0.0, p) for p in probs]
        total = sum(repaired)
        if total > 0:
            repaired = [p / total for p in repaired]
        else:
            repaired = [1.0 / len(probs)] * len(probs)

        l1 = sum(abs(orig - rep) for orig, rep in zip(probs, repaired))
        l2 = sum((orig - rep) ** 2 for orig, rep in zip(probs, repaired)) ** 0.5
        return l1, l2

    def _parse_compact_belief(
        self,
        parsed: dict,
        raw_response: str,
        prompt_hash: str,
        base_metadata: dict,
    ) -> tuple[dict | None, BeliefMetadata]:
        probs = parsed.get("probs", [])

        if not isinstance(probs, list) or len(probs) != len(BUCKET_ORDER):
            return None, BeliefMetadata(
                parse_success=False, raw_response=raw_response,
                prob_sum=None, prob_min=None, negative_count=0,
                prompt_hash=prompt_hash, **base_metadata,
            )

        try:
            probs = [float(p) for p in probs]
        except (ValueError, TypeError):
            return None, BeliefMetadata(
                parse_success=False, raw_response=raw_response,
                prob_sum=None, prob_min=None, negative_count=0,
                prompt_hash=prompt_hash, **base_metadata,
            )

        if all(p == 0.0 for p in probs):
            return None, BeliefMetadata(
                parse_success=False, raw_response=raw_response,
                prob_sum=0.0, prob_min=0.0, negative_count=0,
                prompt_hash=prompt_hash, **base_metadata,
            )

        prob_sum = sum(probs)
        prob_min = min(probs)
        negative_count = sum(1 for p in probs if p < 0)
        repair_l1, repair_l2 = self._compute_repair_distance(probs)
        belief = dict(zip(BUCKET_ORDER, probs))

        return belief, BeliefMetadata(
            parse_success=True, raw_response=raw_response,
            prob_sum=prob_sum, prob_min=prob_min,
            negative_count=negative_count, prompt_hash=prompt_hash,
            repair_distance_l1=repair_l1, repair_distance_l2=repair_l2,
            **base_metadata,
        )

    def _parse_full_belief(
        self,
        parsed: dict,
        raw_response: str,
        prompt_hash: str,
        base_metadata: dict,
    ) -> tuple[dict | None, BeliefMetadata]:
        probs = []
        for name in BUCKET_ORDER:
            val = parsed.get(name)
            if val is not None:
                try:
                    probs.append(float(val))
                except (ValueError, TypeError):
                    probs.append(0.0)
            else:
                probs.append(0.0)

        if all(p == 0.0 for p in probs):
            return None, BeliefMetadata(
                parse_success=False, raw_response=raw_response,
                prob_sum=None, prob_min=None, negative_count=0,
                prompt_hash=prompt_hash, **base_metadata,
            )

        prob_sum = sum(probs)
        prob_min = min(probs)
        negative_count = sum(1 for p in probs if p < 0)
        repair_l1, repair_l2 = self._compute_repair_distance(probs)
        belief = dict(zip(BUCKET_ORDER, probs))

        return belief, BeliefMetadata(
            parse_success=True, raw_response=raw_response,
            prob_sum=prob_sum, prob_min=prob_min,
            negative_count=negative_count, prompt_hash=prompt_hash,
            repair_distance_l1=repair_l1, repair_distance_l2=repair_l2,
            **base_metadata,
        )

    # ========================================================================
    # Prompt Building
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
    # Formatting Helpers
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

    # ========================================================================
    # Fallback
    # ========================================================================

    def _fallback_action(
        self,
        obs: Obs,
        raw_response: str,
        prompt_hash: str,
        template_id: str,
        base_metadata: dict,
    ) -> tuple[Action, ActionMetadata]:
        if not obs.legal_actions:
            raise ValueError("No legal actions available for fallback")
        if CHECK_OR_CALL in obs.legal_actions:
            chosen = CHECK_OR_CALL
        elif FOLD in obs.legal_actions:
            chosen = FOLD
        else:
            chosen = obs.legal_actions[0]

        return chosen, ActionMetadata(
            parse_success=False,
            fallback_used=True,
            raw_response=raw_response,
            action_chosen=chosen.type.value,
            prompt_hash=prompt_hash,
            prompt_template_id=template_id,
            **base_metadata,
        )
