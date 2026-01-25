"""Win probability oracle using Monte Carlo simulation."""

import random
from typing import Optional
from itertools import combinations

from pokerkit import StandardHighHand

from poker_env.deck import FULL_DECK, parse_cards


class WinProbOracle:
    """
    Oracle that computes win/tie/lose probabilities via Monte Carlo.

    Uses PokerKit's StandardHighHand for hand evaluation and
    simulates random board runouts to estimate equity.
    """

    def __init__(self, num_samples: int = 10000, seed: Optional[int] = None):
        """
        Initialize the oracle.

        Args:
            num_samples: Number of Monte Carlo samples for estimation
            seed: Optional random seed for reproducibility
        """
        self.num_samples = num_samples
        self.seed = seed
        self.rng = random.Random(seed)

    def compute(
        self,
        hero_hole: list[str],
        villain_hole: list[str],
        board: list[str],
    ) -> dict:
        """
        Compute win/tie/lose probabilities.

        Args:
            hero_hole: Hero's hole cards, e.g., ["Ac", "As"]
            villain_hole: Villain's hole cards, e.g., ["Kh", "Kd"]
            board: Current board cards, e.g., ["Jc", "3d", "5c"]

        Returns:
            Dict with {"p_win": float, "p_tie": float, "p_lose": float}
        """
        # If board is complete (5 cards), compute exact result
        if len(board) >= 5:
            return self._compute_exact(hero_hole, villain_hole, board[:5])

        # Otherwise, Monte Carlo simulation
        return self._compute_monte_carlo(hero_hole, villain_hole, board)

    def _compute_exact(
        self,
        hero_hole: list[str],
        villain_hole: list[str],
        board: list[str],
    ) -> dict:
        """Compute exact result when board is complete."""
        hero_hand = self._evaluate_hand(hero_hole, board)
        villain_hand = self._evaluate_hand(villain_hole, board)

        if hero_hand > villain_hand:
            return {"p_win": 1.0, "p_tie": 0.0, "p_lose": 0.0}
        elif hero_hand < villain_hand:
            return {"p_win": 0.0, "p_tie": 0.0, "p_lose": 1.0}
        else:
            return {"p_win": 0.0, "p_tie": 1.0, "p_lose": 0.0}

    def _compute_monte_carlo(
        self,
        hero_hole: list[str],
        villain_hole: list[str],
        board: list[str],
    ) -> dict:
        """Compute probabilities via Monte Carlo simulation."""
        # Cards that are no longer available
        dead_cards = set(hero_hole + villain_hole + board)

        # Remaining deck
        deck = [c for c in FULL_DECK if c not in dead_cards]

        # Number of cards needed to complete the board
        cards_needed = 5 - len(board)

        wins = 0
        ties = 0
        losses = 0

        # Use enumeration if possible (small number of combinations)
        possible_runouts = list(combinations(deck, cards_needed))

        if len(possible_runouts) <= self.num_samples:
            # Enumerate all possibilities
            for runout in possible_runouts:
                full_board = board + list(runout)
                result = self._compare_hands(hero_hole, villain_hole, full_board)
                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    ties += 1

            total = len(possible_runouts)
        else:
            # Monte Carlo sampling
            for _ in range(self.num_samples):
                runout = self.rng.sample(deck, cards_needed)
                full_board = board + runout
                result = self._compare_hands(hero_hole, villain_hole, full_board)
                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    ties += 1

            total = self.num_samples

        return {
            "p_win": wins / total,
            "p_tie": ties / total,
            "p_lose": losses / total,
        }

    def _evaluate_hand(self, hole: list[str], board: list[str]) -> StandardHighHand:
        """
        Evaluate a hand using PokerKit.

        Args:
            hole: Two hole cards
            board: Five board cards

        Returns:
            StandardHighHand object for comparison
        """
        # Convert string cards to PokerKit format
        # PokerKit's from_game expects strings in format like 'AcAs' for holes
        hole_str = "".join(hole)
        board_str = "".join(board)

        return StandardHighHand.from_game(hole_str, board_str)

    def _compare_hands(
        self,
        hero_hole: list[str],
        villain_hole: list[str],
        board: list[str],
    ) -> int:
        """
        Compare two hands.

        Returns:
            > 0 if hero wins
            < 0 if villain wins
            0 if tie
        """
        hero_hand = self._evaluate_hand(hero_hole, board)
        villain_hand = self._evaluate_hand(villain_hole, board)

        if hero_hand > villain_hand:
            return 1
        elif hero_hand < villain_hand:
            return -1
        else:
            return 0

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """Reset the random number generator with a new seed."""
        self.seed = seed
        self.rng = random.Random(seed)
