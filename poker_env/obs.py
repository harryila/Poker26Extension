"""Observation dataclass and state serialization utilities."""

from dataclasses import dataclass, field, asdict
from typing import Any, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from pokerkit import State
    from poker_env.actions import Action


@dataclass
class Obs:
    """
    Public observation for a player at a decision point.

    Contains all information visible to the acting player,
    excluding opponent hole cards.
    """

    hand_id: str
    seed: int
    player_index: int
    street: str  # "PREFLOP" | "FLOP" | "TURN" | "RIVER"
    board: list[str] = field(default_factory=list)
    hero_hole: list[str] = field(default_factory=list)
    stacks: list[int] = field(default_factory=list)
    pot_total: int = 0
    to_act: int = 0
    legal_actions: list["Action"] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize observation to dictionary."""
        return {
            "hand_id": self.hand_id,
            "seed": self.seed,
            "player_index": self.player_index,
            "street": self.street,
            "board": self.board,
            "hero_hole": self.hero_hole,
            "stacks": self.stacks,
            "pot_total": self.pot_total,
            "to_act": self.to_act,
            "legal_actions": [a.to_dict() for a in self.legal_actions],
            "history": self.history,
        }

    def to_json(self) -> str:
        """Serialize observation to JSON string."""
        return json.dumps(self.to_dict())


def get_street_name(state: "State") -> str:
    """
    Determine the current street from PokerKit state.

    Args:
        state: PokerKit State object

    Returns:
        Street name: "PREFLOP", "FLOP", "TURN", or "RIVER"
    """
    board_count = len(state.board_cards)
    if board_count == 0:
        return "PREFLOP"
    elif board_count == 3:
        return "FLOP"
    elif board_count == 4:
        return "TURN"
    elif board_count >= 5:
        return "RIVER"
    else:
        # Should not happen in standard hold'em
        return "PREFLOP"


def card_to_str(card) -> str:
    """
    Convert a PokerKit Card to standard string representation.

    Args:
        card: PokerKit Card object

    Returns:
        String like "Ac", "Kh", "2d", etc.
    """
    # PokerKit's repr returns the 2-char format (e.g., "8s")
    # while str returns full name (e.g., "EIGHT OF SPADES (8s)")
    return repr(card)


def serialize_operation(op) -> dict:
    """
    Convert a PokerKit Operation to a JSON-safe dictionary.

    Args:
        op: PokerKit Operation object

    Returns:
        Dictionary with operation type and relevant attributes
    """
    op_type = type(op).__name__

    result = {"op": op_type}

    # Add player index if present
    if hasattr(op, "player_index"):
        result["player"] = op.player_index

    # Add amount if present (for betting operations)
    if hasattr(op, "amount"):
        result["amount"] = op.amount

    # Add cards if present (for dealing operations)
    if hasattr(op, "cards") and op.cards:
        result["cards"] = [card_to_str(c) for c in op.cards]

    return result


def serialize_history(state: "State") -> list[dict]:
    """
    Serialize the full operation history from a PokerKit state.

    Args:
        state: PokerKit State object

    Returns:
        List of serialized operations
    """
    return [serialize_operation(op) for op in state.operations]


def build_observation(
    state: "State",
    player_index: int,
    hand_id: str,
    seed: int,
    legal_actions: list["Action"],
) -> Obs:
    """
    Build an Obs object from the current PokerKit state.

    Args:
        state: PokerKit State object
        player_index: Index of the player to build observation for
        hand_id: Unique identifier for this hand
        seed: Random seed used for this hand
        legal_actions: List of legal actions for the player

    Returns:
        Obs object containing public information for the player
    """
    # Get board cards (each element in board_cards is a list with one card)
    board = []
    for card_list in state.board_cards:
        if card_list:
            board.append(card_to_str(card_list[0]))

    # Get hero's hole cards (only their own cards)
    hero_hole = []
    if state.hole_cards and len(state.hole_cards) > player_index:
        player_hole = state.hole_cards[player_index]
        if player_hole:
            hero_hole = [card_to_str(c) for c in player_hole]

    # Get stack sizes
    stacks = list(state.stacks)

    # Calculate total pot (includes current bets)
    pot_total = state.total_pot_amount

    # Get acting player
    to_act = state.actor_index if state.actor_index is not None else -1

    # Serialize history
    history = serialize_history(state)

    return Obs(
        hand_id=hand_id,
        seed=seed,
        player_index=player_index,
        street=get_street_name(state),
        board=board,
        hero_hole=hero_hole,
        stacks=stacks,
        pot_total=pot_total,
        to_act=to_act,
        legal_actions=legal_actions,
        history=history,
    )
