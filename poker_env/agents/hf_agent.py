"""
HuggingFace-based LLM agent for poker with belief elicitation.

Implements the BaseAgent interface using Llama 3.1 or other instruction-tuned models.
Provides full telemetry for parse success, coherence diagnostics, and reproducibility.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from poker_env.agents.base import BaseAgent
from poker_env.actions import Action, ActionType, FOLD, CHECK_OR_CALL, BET_OR_RAISE
from poker_env.obs import Obs

# Import belief prompts from analysis module
from analysis.prompts import format_belief_prompt, BUCKET_NAMES


# ============================================================================
# System Messages for Strict JSON Output
# ============================================================================

ACTION_SYSTEM_MESSAGE = """You are a poker player. Choose your action.
Return ONLY valid JSON. No other text.
Format: {"action": "FOLD" | "CHECK_OR_CALL" | "BET_OR_RAISE"}"""

BELIEF_SYSTEM_MESSAGE = """You are analyzing opponent hand ranges in poker.
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
    
    def to_dict(self) -> dict:
        return {
            "parse_success": self.parse_success,
            "fallback_used": self.fallback_used,
            "raw_response": self.raw_response,
            "action_chosen": self.action_chosen,
            "prompt_hash": self.prompt_hash,
            "prompt_template_id": self.prompt_template_id,
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
    prompt_template_id: str = "belief_strict_v1"
    
    def to_dict(self) -> dict:
        return {
            "parse_success": self.parse_success,
            "raw_response": self.raw_response,
            "prob_sum": self.prob_sum,
            "prob_min": self.prob_min,
            "negative_count": self.negative_count,
            "prompt_hash": self.prompt_hash,
            "prompt_template_id": self.prompt_template_id,
        }


# ============================================================================
# HFAgent Class
# ============================================================================

class HFAgent(BaseAgent):
    """
    HuggingFace-based LLM agent with belief elicitation.
    
    Loads a causal language model (e.g., Llama 3.1 8B Instruct) and uses it
    to select actions and provide belief estimates over opponent hand ranges.
    
    Features:
    - Strict JSON output enforcement via chat templates
    - Deterministic fallback on parse failure (CHECK_OR_CALL or FOLD)
    - Full telemetry for parse success, coherence diagnostics
    - Configurable generation parameters
    """
    
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.2,
        max_new_tokens: int = 128,
        max_input_tokens: int = 2048,
        max_history_events: int = 50,
        name: str = "HFAgent",
        device_map: str = "auto",
    ):
        """
        Initialize the HuggingFace agent.
        
        Args:
            model_id: HuggingFace model identifier
            temperature: Generation temperature (0 = deterministic)
            max_new_tokens: Maximum tokens to generate
            max_input_tokens: Maximum input context length
            max_history_events: Maximum history events to include in prompts
            name: Agent name for logging
            device_map: Device placement strategy ("auto" recommended)
        """
        super().__init__(name=name)
        
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.max_history_events = max_history_events
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Set pad token if not set (common for Llama models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with bfloat16 for efficiency on A100
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model.eval()
        
        # Store last metadata for retrieval
        self._last_action_metadata: ActionMetadata | None = None
        self._last_belief_metadata: BeliefMetadata | None = None
    
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
        # Build prompt
        prompt = self._build_action_prompt(obs)
        prompt_hash = self._hash_prompt(prompt)
        
        # Generate response
        raw_response = self._generate(prompt, system_message=ACTION_SYSTEM_MESSAGE)
        
        # Parse response
        parsed = self._extract_json(raw_response)
        
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
                )
            elif action_str == "CHECK_OR_CALL" and CHECK_OR_CALL in obs.legal_actions:
                return CHECK_OR_CALL, ActionMetadata(
                    parse_success=True,
                    fallback_used=False,
                    raw_response=raw_response,
                    action_chosen="CHECK_OR_CALL",
                    prompt_hash=prompt_hash,
                )
            elif action_str == "BET_OR_RAISE" and BET_OR_RAISE in obs.legal_actions:
                return BET_OR_RAISE, ActionMetadata(
                    parse_success=True,
                    fallback_used=False,
                    raw_response=raw_response,
                    action_chosen="BET_OR_RAISE",
                    prompt_hash=prompt_hash,
                )
        
        # Parse failed or action not legal - use deterministic fallback
        return self._fallback_action(obs, raw_response, prompt_hash)
    
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
        # Build prompt using the analysis module's format
        prompt = format_belief_prompt(
            hero_hole=obs.hero_hole,
            board=obs.board,
            pot=obs.pot_total,
            street=obs.street,
            history=self._truncate_history(obs.history),
            template="default",
            hero_index=obs.player_index,
        )
        prompt_hash = self._hash_prompt(prompt)
        
        # Generate response
        raw_response = self._generate(prompt, system_message=BELIEF_SYSTEM_MESSAGE)
        
        # Parse response
        parsed = self._extract_json(raw_response)
        
        if parsed:
            # Compute coherence diagnostics
            probs = [v for k, v in parsed.items() if k in BUCKET_NAMES and isinstance(v, (int, float))]
            
            if probs:
                prob_sum = sum(probs)
                prob_min = min(probs)
                negative_count = sum(1 for p in probs if p < 0)
                
                # Filter to only valid bucket names
                belief = {k: float(v) for k, v in parsed.items() 
                         if k in BUCKET_NAMES and isinstance(v, (int, float))}
                
                if belief:
                    return belief, BeliefMetadata(
                        parse_success=True,
                        raw_response=raw_response,
                        prob_sum=prob_sum,
                        prob_min=prob_min,
                        negative_count=negative_count,
                        prompt_hash=prompt_hash,
                    )
        
        # Parse failed
        return None, BeliefMetadata(
            parse_success=False,
            raw_response=raw_response,
            prob_sum=None,
            prob_min=None,
            negative_count=0,
            prompt_hash=prompt_hash,
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
    
    def _generate(self, user_prompt: str, system_message: str) -> str:
        """
        Generate response using chat template.
        
        Args:
            user_prompt: User message content
            system_message: System message for behavior guidance
            
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
        
        # Generation config based on temperature
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        if self.temperature == 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.9
        
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
    # Action Helpers
    # ========================================================================
    
    def _build_action_prompt(self, obs: Obs) -> str:
        """Build prompt for action selection."""
        legal_action_names = [a.type.value for a in obs.legal_actions]
        
        # Truncate history for prompt
        history = self._truncate_history(obs.history)
        history_str = self._format_history_brief(history, obs.player_index)
        
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
    
    def _fallback_action(
        self, 
        obs: Obs, 
        raw_response: str, 
        prompt_hash: str
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
            )
        elif FOLD in obs.legal_actions:
            return FOLD, ActionMetadata(
                parse_success=False,
                fallback_used=True,
                raw_response=raw_response,
                action_chosen="FOLD",
                prompt_hash=prompt_hash,
            )
        else:
            # Should never happen, but handle gracefully
            return obs.legal_actions[0], ActionMetadata(
                parse_success=False,
                fallback_used=True,
                raw_response=raw_response,
                action_chosen=obs.legal_actions[0].type.value,
                prompt_hash=prompt_hash,
            )
