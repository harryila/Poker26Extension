"""
Threshold-based agent that plays according to hand strength.

This agent is designed to be INFORMATIVE - its actions reveal information
about its hand strength, making it useful for testing whether LLMs use
betting history to update beliefs.

IMPORTANT: This agent's behavior MUST match the ParametricOpponent model
in analysis/opponent_model.py so that StrategyAwarePosterior computes
correct likelihoods. Both use the same compute_hand_strength() and
threshold logic.
"""

import random
from typing import Optional

from poker_env.agents.base import BaseAgent
from poker_env.actions import Action, ActionType
from poker_env.obs import Obs

# Import shared hand strength computation from analysis module
# This ensures gameplay behavior matches analysis likelihood model
from analysis.opponent_model import compute_hand_strength


class ThresholdAgent(BaseAgent):
    """
    Agent that plays based on hand strength thresholds.
    
    Behavior:
    - Strong hands (strength >= strong_threshold): Mostly raise
    - Medium hands (weak_threshold <= strength < strong_threshold): Mostly call
    - Weak hands (strength < weak_threshold): Mostly fold (with bluff chance)
    
    This creates informative action sequences where raises indicate strength
    and folds indicate weakness, allowing StrategyAwarePosterior to update
    meaningfully from the action history.
    
    Parameters match ParametricOpponent presets for alignment.
    """
    
    # Presets for ThresholdAgent
    PRESETS = {
        "default": {
            "aggression": 0.4,
            "fold_threshold": 0.3,
            "bluff_freq": 0.1,
        },
        "tight_passive": {
            "aggression": 0.2,
            "fold_threshold": 0.4,
            "bluff_freq": 0.02,
        },
        "tight_aggressive": {
            "aggression": 0.6,
            "fold_threshold": 0.4,
            "bluff_freq": 0.08,
        },
        "loose_passive": {
            "aggression": 0.2,
            "fold_threshold": 0.2,
            "bluff_freq": 0.05,
        },
        "loose_aggressive": {
            "aggression": 0.6,
            "fold_threshold": 0.2,
            "bluff_freq": 0.15,
        },
        # INFORMATIVE_V2: Canonical preset for maximum action-history signal
        # Very high fold rate for weak hands, very high raise rate for strong hands
        # This makes actions highly correlated with hand strength
        # Achieves JS(CardOnly, StrategyAware) ≈ 0.05-0.06
        "informative_v2": {
            "aggression": 0.85,      # Almost always raise with playable hands
            "fold_threshold": 0.55,  # Fold most weak hands
            "bluff_freq": 0.02,      # Very low bluff rate (actions reveal strength)
        },
        # Legacy alias (deprecated, use informative_v2)
        "informative": {
            "aggression": 0.85,
            "fold_threshold": 0.55,
            "bluff_freq": 0.02,
        },
    }
    
    def __init__(
        self,
        preset: str = "default",
        aggression: Optional[float] = None,
        fold_threshold: Optional[float] = None,
        bluff_freq: Optional[float] = None,
        seed: Optional[int] = None,
        name: str = "ThresholdAgent",
    ):
        """
        Initialize threshold agent.
        
        Args:
            preset: Named preset ("default", "tight_passive", etc.)
            aggression: Override preset's aggression (P(raise | continuing))
            fold_threshold: Override preset's fold threshold
            bluff_freq: Override preset's bluff frequency
            seed: Random seed for reproducibility
            name: Agent name
        """
        super().__init__(name=name)
        
        # Load preset
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(self.PRESETS.keys())}")
        
        params = self.PRESETS[preset].copy()
        
        # Allow parameter overrides
        if aggression is not None:
            params["aggression"] = aggression
        if fold_threshold is not None:
            params["fold_threshold"] = fold_threshold
        if bluff_freq is not None:
            params["bluff_freq"] = bluff_freq
        
        self.aggression = params["aggression"]
        self.fold_threshold = params["fold_threshold"]
        self.bluff_freq = params["bluff_freq"]
        self.preset = preset
        
        self.seed = seed
        self.rng = random.Random(seed)
    
    def _get_hand_strength(self, obs: Obs) -> float:
        """
        Compute hand strength from observation.
        
        Uses the same compute_hand_strength() function as ParametricOpponent
        to ensure consistent behavior between gameplay and analysis.
        """
        # Extract hole cards as tuple
        if not obs.hero_hole or len(obs.hero_hole) < 2:
            return 0.5  # Default if no hole cards
        
        hand = (obs.hero_hole[0], obs.hero_hole[1])
        board = obs.board if obs.board else []
        
        return compute_hand_strength(hand, board)
    
    def _get_action_probs(self, obs: Obs) -> dict[ActionType, float]:
        """
        Compute action probabilities given observation.
        
        This mirrors ParametricOpponent.action_prob() logic exactly.
        """
        strength = self._get_hand_strength(obs)
        
        # Adjust threshold based on pot odds (simplified)
        effective_threshold = self.fold_threshold
        bet_to_call = getattr(obs, 'bet_to_call', 0) or 0
        pot_total = obs.pot_total or 0
        
        if bet_to_call > 0 and pot_total > 0:
            pot_odds = bet_to_call / (pot_total + bet_to_call)
            effective_threshold = self.fold_threshold * (0.5 + pot_odds)
        
        # Compute base probabilities (same logic as ParametricOpponent)
        if strength < effective_threshold:
            # Weak hand: mostly fold, sometimes bluff
            p_fold = max(0.0, 1.0 - self.bluff_freq - 0.05)
            p_call = 0.05
            p_raise = min(self.bluff_freq, 0.95)
        else:
            # Strong enough to continue
            p_fold = 0.02  # Rare folds with playable hands
            
            # Split remaining probability between call and raise
            continue_prob = 1.0 - p_fold
            
            # Stronger hands raise more
            strength_factor = min(1.0, strength / 0.8)
            effective_aggression = min(1.0, self.aggression * strength_factor)
            
            p_raise = continue_prob * effective_aggression
            p_call = continue_prob * (1.0 - effective_aggression)
        
        # Normalize
        total = p_fold + p_call + p_raise
        return {
            ActionType.FOLD: p_fold / total,
            ActionType.CHECK_OR_CALL: p_call / total,
            ActionType.BET_OR_RAISE: p_raise / total,
        }
    
    def _sample_action(self, probs: dict[ActionType, float], legal_actions: list[Action]) -> Action:
        """
        Sample action from probability distribution, constrained to legal actions.
        """
        # Build legal action type set
        legal_types = {a.type for a in legal_actions}
        
        # Filter probabilities to legal actions and renormalize
        filtered_probs = {t: p for t, p in probs.items() if t in legal_types}
        
        if not filtered_probs:
            # Fallback: return first legal action
            return legal_actions[0]
        
        total = sum(filtered_probs.values())
        if total == 0:
            # Equal probability over legal actions
            return self.rng.choice(legal_actions)
        
        # Normalize
        normalized = {t: p / total for t, p in filtered_probs.items()}
        
        # Sample
        r = self.rng.random()
        cumulative = 0.0
        chosen_type = None
        
        for action_type, prob in normalized.items():
            cumulative += prob
            if r <= cumulative:
                chosen_type = action_type
                break
        
        if chosen_type is None:
            chosen_type = list(normalized.keys())[-1]
        
        # Find matching action
        for action in legal_actions:
            if action.type == chosen_type:
                return action
        
        # Fallback
        return legal_actions[0]
    
    def act(self, obs: Obs) -> Action:
        """
        Select action based on hand strength thresholds.
        
        Args:
            obs: Current observation including legal actions and hole cards
            
        Returns:
            Selected action
        """
        if not obs.legal_actions:
            raise ValueError("No legal actions available")
        
        # Compute action probabilities
        probs = self._get_action_probs(obs)
        
        # Sample from distribution
        return self._sample_action(probs, obs.legal_actions)
    
    def reset(self) -> None:
        """Reset random state."""
        if self.seed is not None:
            self.rng = random.Random(self.seed)
    
    def get_config(self) -> dict:
        """Return agent configuration for logging."""
        return {
            "type": "ThresholdAgent",
            "preset": self.preset,
            "aggression": self.aggression,
            "fold_threshold": self.fold_threshold,
            "bluff_freq": self.bluff_freq,
            "seed": self.seed,
        }
    
    def __repr__(self) -> str:
        return (
            f"ThresholdAgent(preset={self.preset!r}, "
            f"aggression={self.aggression}, "
            f"fold_threshold={self.fold_threshold}, "
            f"bluff_freq={self.bluff_freq})"
        )
