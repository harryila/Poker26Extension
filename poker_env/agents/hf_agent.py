"""
HuggingFace-based LLM agent for poker with belief elicitation.

Implements the BaseAgent interface using Llama 3.1 or other instruction-tuned models.
Provides full telemetry for parse success, coherence diagnostics, and reproducibility.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from poker_env.agents.base import BaseAgent
from poker_env.actions import Action, ActionType, FOLD, CHECK_OR_CALL, BET_OR_RAISE
from poker_env.obs import Obs
from poker_env.config import (
    DEFAULT_MODEL_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_ACTION_MAX_TOKENS,
    DEFAULT_BELIEF_MAX_TOKENS,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_HISTORY_EVENTS,
    DEFAULT_BELIEF_FORMAT,
    BELIEF_SCHEMA_ID,
    BUCKET_ORDER,
)

# Import belief prompts from analysis module
from analysis.prompts import BUCKET_NAMES


# ============================================================================
# System Messages for Strict JSON Output
# ============================================================================

ACTION_SYSTEM_MESSAGE = """You are a poker player. Choose your action.
Return ONLY valid JSON. No other text.
Format: {"action": "FOLD" | "CHECK_OR_CALL" | "BET_OR_RAISE"}"""

# Compact belief system message with numeric constraints
BELIEF_SYSTEM_MESSAGE_COMPACT = f"""You are analyzing opponent hand ranges in poker. Return ONLY valid JSON. No other text.
Output format: {{"schema":"{BELIEF_SCHEMA_ID}","probs":[p0,p1,...,p13]}}
Constraints: each pi must be a decimal between 0 and 1, non-negative, and the 14 values must sum to 1.0 exactly (adjust the last value to fix rounding). Use at most 3 decimal places."""

# Full belief system message (verbose dict format)
BELIEF_SYSTEM_MESSAGE_FULL = """You are analyzing opponent hand ranges in poker.
Return ONLY valid JSON. No other text.
Probabilities must be non-negative and sum to 1.0."""


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
    # Truncation telemetry
    history_events_before: int = 0
    history_events_after: int = 0
    was_truncated: bool = False
    
    def to_dict(self) -> dict:
        return {
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
    # Truncation telemetry
    history_events_before: int = 0
    history_events_after: int = 0
    was_truncated: bool = False
    # Repair distance (L1/L2 to simplex) - should be ~0 for valid beliefs
    repair_distance_l1: float | None = None
    repair_distance_l2: float | None = None
    
    def to_dict(self) -> dict:
        return {
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


# ============================================================================
# HFAgent Class
# ============================================================================

class HFAgent(BaseAgent):
    """
    HuggingFace-based LLM agent with belief elicitation.
    
    Loads a causal language model (e.g., Llama 3.1 8B/70B Instruct) and uses it
    to select actions and provide belief estimates over opponent hand ranges.
    
    Features:
    - Strict JSON output enforcement via chat templates
    - Deterministic fallback on parse failure (CHECK_OR_CALL or FOLD)
    - Full telemetry for parse success, coherence diagnostics
    - Separate token budgets for action vs belief
    - Configurable generation parameters (temperature, top_p)
    - Truncation telemetry for history analysis
    
    Configuration is centralized in poker_env/config.py
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
        name: str = "HFAgent",
        device_map: str = "auto",
    ):
        """
        Initialize the HuggingFace agent.
        
        All parameters default to values from poker_env/config.py if not specified.
        
        Args:
            model_id: HuggingFace model identifier (default: config.DEFAULT_MODEL_ID)
            temperature: Generation temperature, 0 = deterministic (default: config.DEFAULT_TEMPERATURE)
            top_p: Top-p sampling parameter (default: config.DEFAULT_TOP_P)
            action_max_new_tokens: Max tokens for action generation (default: config.DEFAULT_ACTION_MAX_TOKENS)
            belief_max_new_tokens: Max tokens for belief generation (default: config.DEFAULT_BELIEF_MAX_TOKENS)
            max_input_tokens: Maximum input context length (default: config.DEFAULT_MAX_INPUT_TOKENS)
            max_history_events: Maximum history events in prompts (default: config.DEFAULT_MAX_HISTORY_EVENTS)
            belief_format: "compact" or "full" (default: config.DEFAULT_BELIEF_FORMAT)
            name: Agent name for logging
            device_map: Device placement strategy ("auto" recommended)
        """
        super().__init__(name=name)
        
        # Apply defaults from config
        self.model_id = model_id if model_id is not None else DEFAULT_MODEL_ID
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.top_p = top_p if top_p is not None else DEFAULT_TOP_P
        self.action_max_new_tokens = action_max_new_tokens if action_max_new_tokens is not None else DEFAULT_ACTION_MAX_TOKENS
        self.belief_max_new_tokens = belief_max_new_tokens if belief_max_new_tokens is not None else DEFAULT_BELIEF_MAX_TOKENS
        self.max_input_tokens = max_input_tokens if max_input_tokens is not None else DEFAULT_MAX_INPUT_TOKENS
        self.max_history_events = max_history_events if max_history_events is not None else DEFAULT_MAX_HISTORY_EVENTS
        self.belief_format = belief_format if belief_format is not None else DEFAULT_BELIEF_FORMAT
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Set pad token if not set (common for Llama models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with bfloat16 for efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model.eval()
        
        # Store last metadata for retrieval
        self._last_action_metadata: ActionMetadata | None = None
        self._last_belief_metadata: BeliefMetadata | None = None
        
        # Track last truncation info
        self._last_truncation_info: dict = {}
    
    # ========================================================================
    # Core Interface Methods
    # ========================================================================
    
    def act(self, obs: Obs) -> Action:
        """
        Select an action given an observation.
        
        Uses LLM to choose action, with deterministic fallback on parse failure.
        Metadata available via get_last_action_metadata().
        
        Args:
            obs: Current observation including legal actions
            
        Returns:
            Selected action
        """
        action, metadata = self.act_with_metadata(obs)
        self._last_action_metadata = metadata
        return action
    
    def act_with_metadata(self, obs: Obs) -> tuple[Action, ActionMetadata]:
        """
        Select action and return metadata.
        
        Args:
            obs: Current observation
            
        Returns:
            Tuple of (action, metadata)
        """
        # Truncate history and track telemetry
        original_len = len(obs.history)
        truncated_history = self._truncate_history(obs.history)
        truncated_len = len(truncated_history)
        was_truncated = truncated_len < original_len
        
        # Build prompt
        prompt = self._build_action_prompt(obs, truncated_history)
        prompt_hash = self._hash_prompt(prompt)
        
        # Generate response with action token budget
        raw_response = self._generate(
            prompt, 
            system_message=ACTION_SYSTEM_MESSAGE,
            max_new_tokens=self.action_max_new_tokens,
        )
        
        # Parse response
        parsed = self._extract_json(raw_response)
        
        # Base metadata with truncation info
        base_metadata = {
            "history_events_before": original_len,
            "history_events_after": truncated_len,
            "was_truncated": was_truncated,
        }
        
        if parsed and "action" in parsed:
            action_str = parsed["action"].upper()
            
            # Map to action
            if action_str == "FOLD" and FOLD in obs.legal_actions:
                return FOLD, ActionMetadata(
                    parse_success=True,
                    fallback_used=False,
                    raw_response=raw_response,
                    action_chosen="FOLD",
                    prompt_hash=prompt_hash,
                    **base_metadata,
                )
            elif action_str == "CHECK_OR_CALL" and CHECK_OR_CALL in obs.legal_actions:
                return CHECK_OR_CALL, ActionMetadata(
                    parse_success=True,
                    fallback_used=False,
                    raw_response=raw_response,
                    action_chosen="CHECK_OR_CALL",
                    prompt_hash=prompt_hash,
                    **base_metadata,
                )
            elif action_str == "BET_OR_RAISE" and BET_OR_RAISE in obs.legal_actions:
                return BET_OR_RAISE, ActionMetadata(
                    parse_success=True,
                    fallback_used=False,
                    raw_response=raw_response,
                    action_chosen="BET_OR_RAISE",
                    prompt_hash=prompt_hash,
                    **base_metadata,
                )
        
        # Parse failed or action not legal - use deterministic fallback
        return self._fallback_action(obs, raw_response, prompt_hash, base_metadata)
    
    def belief(self, obs: Obs) -> dict | None:
        """
        Return belief distribution over opponent hand buckets.
        
        Metadata available via get_last_belief_metadata().
        
        Args:
            obs: Current observation
            
        Returns:
            Dict mapping bucket names to probabilities, or None on failure
        """
        belief, metadata = self.belief_with_metadata(obs)
        self._last_belief_metadata = metadata
        return belief
    
    def belief_with_metadata(self, obs: Obs) -> tuple[dict | None, BeliefMetadata]:
        """
        Get belief distribution with metadata.
        
        Args:
            obs: Current observation
            
        Returns:
            Tuple of (belief_dict, metadata)
        """
        # Truncate history and track telemetry
        original_len = len(obs.history)
        truncated_history = self._truncate_history(obs.history)
        truncated_len = len(truncated_history)
        was_truncated = truncated_len < original_len
        
        # Build prompt based on format
        if self.belief_format == "compact":
            prompt = self._build_compact_belief_prompt(obs, truncated_history)
            system_message = BELIEF_SYSTEM_MESSAGE_COMPACT
            prompt_template_id = "belief_compact_v1"
        else:
            prompt = self._build_full_belief_prompt(obs, truncated_history)
            system_message = BELIEF_SYSTEM_MESSAGE_FULL
            prompt_template_id = "belief_full_v1"
        
        prompt_hash = self._hash_prompt(prompt)
        
        # Generate response with belief token budget
        raw_response = self._generate(
            prompt, 
            system_message=system_message,
            max_new_tokens=self.belief_max_new_tokens,
        )
        
        # Base metadata
        base_metadata = {
            "history_events_before": original_len,
            "history_events_after": truncated_len,
            "was_truncated": was_truncated,
            "belief_format": self.belief_format,
            "belief_schema_id": BELIEF_SCHEMA_ID,
            "prompt_template_id": prompt_template_id,
        }
        
        # Parse response based on format
        parsed = self._extract_json(raw_response)
        
        if parsed:
            if self.belief_format == "compact":
                # Compact format: {"schema": "...", "probs": [...]}
                return self._parse_compact_belief(parsed, raw_response, prompt_hash, base_metadata)
            else:
                # Full format: {"premium_pairs": 0.05, ...}
                return self._parse_full_belief(parsed, raw_response, prompt_hash, base_metadata)
        
        # Parse failed
        return None, BeliefMetadata(
            parse_success=False,
            raw_response=raw_response,
            prob_sum=None,
            prob_min=None,
            negative_count=0,
            prompt_hash=prompt_hash,
            **base_metadata,
        )
    
    def _compute_repair_distance(self, probs: list[float]) -> tuple[float, float]:
        """
        Compute L1 and L2 distance from original probs to simplex-projected probs.
        
        Returns (l1_distance, l2_distance).
        Should be ~0 for valid distributions.
        """
        # Repair: clip negatives and renormalize
        repaired = [max(0.0, p) for p in probs]
        total = sum(repaired)
        if total > 0:
            repaired = [p / total for p in repaired]
        else:
            # All zeros - uniform fallback
            repaired = [1.0 / len(probs)] * len(probs)
        
        # Compute distances
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
        """Parse compact belief format: {"schema": "...", "probs": [...]}"""
        probs = parsed.get("probs", [])
        
        if not isinstance(probs, list) or len(probs) != len(BUCKET_ORDER):
            return None, BeliefMetadata(
                parse_success=False,
                raw_response=raw_response,
                prob_sum=None,
                prob_min=None,
                negative_count=0,
                prompt_hash=prompt_hash,
                **base_metadata,
            )
        
        # Convert to float and compute diagnostics
        try:
            probs = [float(p) for p in probs]
        except (ValueError, TypeError):
            return None, BeliefMetadata(
                parse_success=False,
                raw_response=raw_response,
                prob_sum=None,
                prob_min=None,
                negative_count=0,
                prompt_hash=prompt_hash,
                **base_metadata,
            )
        
        prob_sum = sum(probs)
        prob_min = min(probs)
        negative_count = sum(1 for p in probs if p < 0)
        
        # Compute repair distance (should be ~0 for valid beliefs)
        repair_l1, repair_l2 = self._compute_repair_distance(probs)
        
        # Convert to labeled dict
        belief = dict(zip(BUCKET_ORDER, probs))
        
        return belief, BeliefMetadata(
            parse_success=True,
            raw_response=raw_response,
            prob_sum=prob_sum,
            prob_min=prob_min,
            negative_count=negative_count,
            prompt_hash=prompt_hash,
            repair_distance_l1=repair_l1,
            repair_distance_l2=repair_l2,
            **base_metadata,
        )
    
    def _parse_full_belief(
        self, 
        parsed: dict, 
        raw_response: str, 
        prompt_hash: str,
        base_metadata: dict,
    ) -> tuple[dict | None, BeliefMetadata]:
        """Parse full belief format: {"premium_pairs": 0.05, ...}"""
        # Extract only valid bucket names in order
        probs = []
        for name in BUCKET_ORDER:
            val = parsed.get(name)
            if val is not None and isinstance(val, (int, float)):
                probs.append(float(val))
            else:
                probs.append(0.0)  # Missing bucket
        
        # Check if we got any valid probs
        if all(p == 0.0 for p in probs):
            return None, BeliefMetadata(
                parse_success=False,
                raw_response=raw_response,
                prob_sum=None,
                prob_min=None,
                negative_count=0,
                prompt_hash=prompt_hash,
                **base_metadata,
            )
        
        prob_sum = sum(probs)
        prob_min = min(probs)
        negative_count = sum(1 for p in probs if p < 0)
        
        # Compute repair distance (should be ~0 for valid beliefs)
        repair_l1, repair_l2 = self._compute_repair_distance(probs)
        
        # Filter to only valid bucket names
        belief = {k: float(v) for k, v in parsed.items() 
                 if k in BUCKET_NAMES and isinstance(v, (int, float))}
        
        return belief, BeliefMetadata(
            parse_success=True,
            raw_response=raw_response,
            prob_sum=prob_sum,
            prob_min=prob_min,
            negative_count=negative_count,
            prompt_hash=prompt_hash,
            repair_distance_l1=repair_l1,
            repair_distance_l2=repair_l2,
            **base_metadata,
        )
    
    def get_last_action_metadata(self) -> ActionMetadata | None:
        """Get metadata from last act() call."""
        return self._last_action_metadata
    
    def get_last_belief_metadata(self) -> BeliefMetadata | None:
        """Get metadata from last belief() call."""
        return self._last_belief_metadata
    
    # ========================================================================
    # Generation and Parsing
    # ========================================================================
    
    def _generate(self, user_prompt: str, system_message: str, max_new_tokens: int) -> str:
        """
        Generate response using chat template.
        
        Args:
            user_prompt: User message content
            system_message: System message for behavior guidance
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # Build chat messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize with truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generation config based on temperature (do_sample rule)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        if self.temperature == 0:
            # Deterministic generation
            gen_kwargs["do_sample"] = False
        else:
            # Sampling with temperature and top_p
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )
        
        return response.strip()
    
    def _extract_json(self, text: str) -> dict | None:
        """
        Extract JSON from potentially chatty response.
        
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
        
        # Find all {...} blocks (non-nested for simplicity)
        # Use a more robust pattern that handles nested content
        matches = []
        depth = 0
        start = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    matches.append(text[start:i+1])
                    start = -1
        
        # Try each match, shortest first (likely cleaner)
        matches.sort(key=len)
        
        for match in matches:
            # Basic cleanup: remove trailing commas before }
            cleaned = re.sub(r',\s*}', '}', match)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _truncate_history(self, history: list[dict], max_events: int | None = None) -> list[dict]:
        """
        Truncate history to stay within token budget.
        
        Keeps first 5 events (blinds/deals) + most recent events.
        
        Args:
            history: Full action history
            max_events: Maximum events to keep (defaults to self.max_history_events)
            
        Returns:
            Truncated history
        """
        if max_events is None:
            max_events = self.max_history_events
            
        if len(history) <= max_events:
            return history
        
        # Keep first 5 (blinds/deals) + last (max_events - 5)
        return history[:5] + history[-(max_events - 5):]
    
    def _hash_prompt(self, prompt: str) -> str:
        """Compute short hash of prompt for reproducibility tracking."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    # ========================================================================
    # Prompt Building
    # ========================================================================
    
    def _build_action_prompt(self, obs: Obs, truncated_history: list[dict]) -> str:
        """Build prompt for action selection."""
        legal_action_names = [a.type.value for a in obs.legal_actions]
        history_str = self._format_history_brief(truncated_history, obs.player_index)
        
        prompt = f"""Current poker situation:
- Your cards: {' '.join(obs.hero_hole)}
- Board: {' '.join(obs.board) if obs.board else 'None (preflop)'}
- Pot: {obs.pot_total}
- Street: {obs.street}
- Bet to call: {obs.bet_to_call}
- Legal actions: {legal_action_names}

Recent actions:
{history_str}

Choose your action. Return ONLY: {{"action": "YOUR_CHOICE"}}"""
        
        return prompt
    
    def _build_compact_belief_prompt(self, obs: Obs, truncated_history: list[dict]) -> str:
        """Build compact belief prompt (fixed-order list format)."""
        action_summary = self._format_action_summary(truncated_history, obs.player_index)
        
        # Bucket order reference in prompt
        bucket_order_str = ", ".join(f"{i} {name}" for i, name in enumerate(BUCKET_ORDER))
        
        prompt = f"""Bucket order (indices 0-13):
{bucket_order_str}.

Your cards: {' '.join(obs.hero_hole)}
Board: {' '.join(obs.board) if obs.board else 'None (preflop)'}
Pot: {obs.pot_total}
Opponent's actions: {action_summary}

Return ONLY the JSON object."""
        
        return prompt
    
    def _build_full_belief_prompt(self, obs: Obs, truncated_history: list[dict]) -> str:
        """Build full belief prompt (labeled dict format)."""
        # Import here to avoid circular dependency
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
    
    def _format_history_brief(self, history: list[dict], hero_index: int) -> str:
        """Format history briefly for action prompt."""
        if not history:
            return "No actions yet"
        
        lines = []
        for event in history[-10:]:  # Last 10 events only
            event_type = event.get("event", "")
            player = event.get("player")
            amount = event.get("amount")
            
            # Skip non-action events
            if event_type in ("POST_BLIND", "DEAL_HOLE", "DEAL_BOARD", "UNKNOWN"):
                continue
            
            player_name = "You" if player == hero_index else "Opponent"
            
            if amount:
                lines.append(f"  {player_name}: {event_type} ({amount})")
            else:
                lines.append(f"  {player_name}: {event_type}")
        
        return "\n".join(lines) if lines else "No betting actions yet"
    
    def _format_action_summary(self, history: list[dict], hero_index: int) -> str:
        """Create brief action summary for belief prompts."""
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
        
        if not opponent_actions:
            return "No opponent actions"
        
        return ", ".join(opponent_actions)
    
    # ========================================================================
    # Action Helpers
    # ========================================================================
    
    def _fallback_action(
        self, 
        obs: Obs, 
        raw_response: str, 
        prompt_hash: str,
        base_metadata: dict,
    ) -> tuple[Action, ActionMetadata]:
        """
        Deterministic fallback when parsing fails.
        
        Uses CHECK_OR_CALL if legal, otherwise FOLD.
        """
        if CHECK_OR_CALL in obs.legal_actions:
            return CHECK_OR_CALL, ActionMetadata(
                parse_success=False,
                fallback_used=True,
                raw_response=raw_response,
                action_chosen="CHECK_OR_CALL",
                prompt_hash=prompt_hash,
                **base_metadata,
            )
        elif FOLD in obs.legal_actions:
            return FOLD, ActionMetadata(
                parse_success=False,
                fallback_used=True,
                raw_response=raw_response,
                action_chosen="FOLD",
                prompt_hash=prompt_hash,
                **base_metadata,
            )
        else:
            # Should never happen, but handle gracefully
            return obs.legal_actions[0], ActionMetadata(
                parse_success=False,
                fallback_used=True,
                raw_response=raw_response,
                action_chosen=obs.legal_actions[0].type.value,
                prompt_hash=prompt_hash,
                **base_metadata,
            )
