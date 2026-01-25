"""
PokerKitEnv - Core poker environment wrapper for Heads-Up Fixed-Limit Texas Hold'em.

This module provides a clean reset/step interface around PokerKit's
FixedLimitTexasHoldem game, with support for deterministic dealing
and observation encoding.
"""

import uuid
from typing import Optional, Tuple

from pokerkit import Automation, FixedLimitTexasHoldem

from poker_env.actions import Action, ActionType, apply_action, get_legal_actions
from poker_env.obs import Obs, build_observation
from poker_env.deck import DeterministicDeck


# Automations to handle non-decision phases automatically
AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)


class PokerKitEnv:
    """
    Heads-Up Fixed-Limit Texas Hold'em environment.

    Wraps PokerKit's FixedLimitTexasHoldem with a clean reset/step interface
    suitable for research on LLM belief modeling.

    Attributes:
        stacks: Starting stack sizes for both players
        blinds: Small blind and big blind amounts
        small_bet: Bet size for preflop and flop
        big_bet: Bet size for turn and river
    """

    def __init__(
        self,
        stacks: Tuple[int, int] = (200, 200),
        blinds: Tuple[int, int] = (1, 2),
        small_bet: int = 2,
        big_bet: int = 4,
    ):
        """
        Initialize the poker environment.

        Args:
            stacks: Starting stacks for (player0, player1)
            blinds: (small_blind, big_blind) amounts
            small_bet: Fixed bet size for preflop and flop streets
            big_bet: Fixed bet size for turn and river streets
        """
        self.stacks = stacks
        self.blinds = blinds
        self.small_bet = small_bet
        self.big_bet = big_bet

        # State tracking
        self.state = None
        self.deck: Optional[DeterministicDeck] = None
        self.hand_id: str = ""
        self.seed: int = 0
        self.initial_stacks = list(stacks)

        # Track dealing progress
        self._holes_dealt = 0
        self._board_dealt = 0

    def reset(
        self,
        seed: int,
        hero_hole: Optional[str] = None,
        villain_hole: Optional[str] = None,
        board: Optional[str] = None,
    ) -> Obs:
        """
        Reset to a new hand.

        Args:
            seed: Random seed for deterministic dealing
            hero_hole: Optional explicit hole cards for player 0 (e.g., "AcAs")
            villain_hole: Optional explicit hole cards for player 1 (e.g., "KhKd")
            board: Optional explicit board cards (e.g., "Jc3d5c4h9s")

        Returns:
            Observation for the first decision point
        """
        self.seed = seed
        self.hand_id = str(uuid.uuid4())[:8]
        self.initial_stacks = list(self.stacks)

        # Create deterministic deck
        self.deck = DeterministicDeck.from_seed(seed)
        if hero_hole or villain_hole:
            self.deck.set_explicit_holes(hero_hole, villain_hole)
        if board:
            self.deck.set_explicit_board(board)

        # Reset dealing counters
        self._holes_dealt = 0
        self._board_dealt = 0

        # Create new game state
        self.state = FixedLimitTexasHoldem.create_state(
            AUTOMATIONS,
            True,  # ante_trimming_status
            0,  # antes (no ante)
            self.blinds,
            self.small_bet,
            self.big_bet,
            self.stacks,
            2,  # num_players (heads-up)
        )

        # Advance to first decision point
        self._advance_until_decision()

        # Return observation for acting player
        return self.get_obs(self.current_player())

    def step(self, action: Action) -> Tuple[Obs, float, bool, dict]:
        """
        Apply an action and advance the game.

        Args:
            action: Action to apply

        Returns:
            Tuple of (observation, reward, done, info)
            - observation: Obs for the next decision point
            - reward: Stack delta for player 0 (0 if not terminal)
            - done: True if hand is complete
            - info: Additional information dict
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Apply the action
        apply_action(self.state, action)

        # Advance to next decision point
        hand_active = self._advance_until_decision()

        # Check if hand is complete
        done = not hand_active

        # Calculate reward (stack delta for player 0)
        reward = 0.0
        info = {}

        if done:
            final_stacks = list(self.state.stacks)
            reward = float(final_stacks[0] - self.initial_stacks[0])
            info["final_stacks"] = final_stacks
            info["player0_delta"] = reward
            info["player1_delta"] = float(final_stacks[1] - self.initial_stacks[1])

            # Include hole cards at showdown if available
            info["showdown"] = self._get_showdown_info()

        # Get observation for next player (or final state if done)
        player = self.current_player() if not done else 0
        obs = self.get_obs(player)

        return obs, reward, done, info

    def current_player(self) -> int:
        """
        Get the index of the player to act.

        Returns:
            Player index (0 or 1), or -1 if no decision needed
        """
        if self.state is None:
            return -1
        return self.state.actor_index if self.state.actor_index is not None else -1

    def legal_actions(self) -> list[Action]:
        """
        Get list of legal actions for the current player.

        Returns:
            List of legal Action objects
        """
        if self.state is None:
            return []
        return get_legal_actions(self.state)

    def get_obs(self, player_index: int) -> Obs:
        """
        Get observation for a specific player.

        Args:
            player_index: Index of the player (0 or 1)

        Returns:
            Obs object with public information visible to that player
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        return build_observation(
            state=self.state,
            player_index=player_index,
            hand_id=self.hand_id,
            seed=self.seed,
            legal_actions=self.legal_actions(),
        )

    def get_hidden_state(self) -> dict:
        """
        Get hidden state for logging/oracle purposes.

        Returns:
            Dict with opponent hole cards and remaining deck
        """
        if self.state is None:
            return {}

        hidden = {}

        # Get all hole cards
        if self.state.hole_cards:
            for i, hole in enumerate(self.state.hole_cards):
                if hole:
                    # Use repr() for 2-char format (e.g., "8s" not "EIGHT OF SPADES")
                    hidden[f"player{i}_hole"] = [repr(c) for c in hole]

        # Get remaining deck cards
        if self.deck:
            hidden["remaining_deck"] = self.deck.remaining_cards()

        return hidden

    def _advance_until_decision(self) -> bool:
        """
        Advance the game state through non-decision phases.

        Handles dealing and lets automations handle other phases
        (blinds, bet collection, showdown, etc.).

        Returns:
            True if hand is still active and a decision is needed,
            False if hand is complete.
        """
        while self.state.status:
            # Check if we need to deal hole cards
            if self._needs_hole_dealing():
                self._deal_hole_cards()
                continue

            # Check if we need to deal board cards
            if self._needs_board_dealing():
                self._deal_board_cards()
                continue

            # Check if a player decision is needed
            if self.state.actor_index is not None:
                # Verify there are legal actions available
                if self.legal_actions():
                    return True

            # No action needed and no dealing needed - check if state progressed
            # If not, the hand may be complete
            break

        return False

    def _needs_hole_dealing(self) -> bool:
        """Check if hole cards need to be dealt."""
        # In heads-up, we need to deal holes to 2 players
        if self._holes_dealt >= 2:
            return False

        # Check if PokerKit is waiting for hole dealing
        # This happens after blinds are posted
        try:
            # PokerKit uses can_deal_hole() to check
            if hasattr(self.state, "can_deal_hole"):
                return self.state.can_deal_hole()
        except Exception:
            pass

        return False

    def _needs_board_dealing(self) -> bool:
        """Check if board cards need to be dealt."""
        try:
            if hasattr(self.state, "can_deal_board"):
                return self.state.can_deal_board()
        except Exception:
            pass

        return False

    def _deal_hole_cards(self) -> None:
        """Deal hole cards to the next player."""
        if self.deck is None:
            raise RuntimeError("Deck not initialized")

        # Get cards from deterministic deck
        cards = self.deck.get_hole_cards(self._holes_dealt)

        # Deal to PokerKit state
        self.state.deal_hole(cards)
        self._holes_dealt += 1

    def _deal_board_cards(self) -> None:
        """Deal board cards (flop, turn, or river)."""
        if self.deck is None:
            raise RuntimeError("Deck not initialized")

        # Determine how many cards to deal
        current_board = len(self.state.board_cards)
        if current_board == 0:
            count = 3  # Flop
        else:
            count = 1  # Turn or River

        # Get cards from deterministic deck
        cards = self.deck.get_board_cards(count, current_board)

        # Deal to PokerKit state
        self.state.deal_board(cards)
        self._board_dealt += count

    def _get_showdown_info(self) -> dict:
        """Get information about the showdown."""
        info = {}

        if self.state.hole_cards:
            for i, hole in enumerate(self.state.hole_cards):
                if hole:
                    info[f"player{i}_hole"] = [repr(c) for c in hole]

        if self.state.board_cards:
            # Each element in board_cards is a list with one card
            info["board"] = [repr(cl[0]) for cl in self.state.board_cards if cl]

        return info

    def render_text(self) -> str:
        """
        Render the current state as text for debugging.

        Returns:
            Multi-line string representation of the game state
        """
        if self.state is None:
            return "Environment not initialized"

        lines = [
            f"=== Hand {self.hand_id} (seed: {self.seed}) ===",
            f"Street: {self._get_street_name()}",
            f"Board: {[repr(cl[0]) for cl in self.state.board_cards if cl]}",
            f"Pot: {sum(self.state.pots) if self.state.pots else 0}",
            f"Stacks: {list(self.state.stacks)}",
            f"Bets: {list(self.state.bets) if self.state.bets else []}",
            f"To act: Player {self.current_player()}",
            f"Legal actions: {[a.type.value for a in self.legal_actions()]}",
        ]

        # Add hole cards (for debugging)
        if self.state.hole_cards:
            for i, hole in enumerate(self.state.hole_cards):
                if hole:
                    lines.append(f"Player {i} hole: {[repr(c) for c in hole]}")

        return "\n".join(lines)

    def _get_street_name(self) -> str:
        """Get current street name."""
        board_count = len(self.state.board_cards)
        if board_count == 0:
            return "PREFLOP"
        elif board_count == 3:
            return "FLOP"
        elif board_count == 4:
            return "TURN"
        else:
            return "RIVER"
