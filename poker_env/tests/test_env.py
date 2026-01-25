"""Basic tests for the PokerKitEnv environment."""

import pytest
from poker_env.env import PokerKitEnv
from poker_env.actions import Action, ActionType, FOLD, CHECK_OR_CALL, BET_OR_RAISE


class TestPokerKitEnvBasic:
    """Basic functionality tests for PokerKitEnv."""

    def test_env_creation(self):
        """Test that environment can be created."""
        env = PokerKitEnv()
        assert env is not None
        assert env.stacks == (200, 200)
        assert env.blinds == (1, 2)

    def test_env_custom_config(self):
        """Test environment with custom configuration."""
        env = PokerKitEnv(
            stacks=(500, 500),
            blinds=(2, 4),
            small_bet=4,
            big_bet=8,
        )
        assert env.stacks == (500, 500)
        assert env.blinds == (2, 4)

    def test_reset_returns_obs(self):
        """Test that reset returns a valid observation."""
        env = PokerKitEnv()
        obs = env.reset(seed=42)

        assert obs is not None
        assert obs.hand_id != ""
        assert obs.seed == 42
        assert len(obs.hero_hole) == 2
        assert obs.street == "PREFLOP"
        assert len(obs.legal_actions) > 0

    def test_legal_actions_not_empty(self):
        """Test that legal actions are never empty at decision point."""
        env = PokerKitEnv()
        env.reset(seed=42)

        legal = env.legal_actions()
        assert len(legal) > 0
        assert all(isinstance(a, Action) for a in legal)

    def test_step_advances_game(self):
        """Test that step advances the game state."""
        env = PokerKitEnv()
        obs1 = env.reset(seed=42)

        # Get first legal action and apply it
        action = obs1.legal_actions[0]
        obs2, reward, done, info = env.step(action)

        # Game should have progressed (either new player or done)
        assert obs2 is not None

    def test_get_obs_player_specific(self):
        """Test that observations are player-specific."""
        env = PokerKitEnv()
        env.reset(seed=42)

        obs0 = env.get_obs(0)
        obs1 = env.get_obs(1)

        # Each player should only see their own hole cards
        assert obs0.player_index == 0
        assert obs1.player_index == 1
        # Hole cards should be different (unless extremely unlucky seed)
        # Both should have exactly 2 hole cards
        assert len(obs0.hero_hole) == 2
        assert len(obs1.hero_hole) == 2

    def test_current_player_valid(self):
        """Test that current_player returns valid index."""
        env = PokerKitEnv()
        env.reset(seed=42)

        player = env.current_player()
        assert player in [0, 1]

    def test_hidden_state_has_both_holes(self):
        """Test that hidden state includes both players' hole cards."""
        env = PokerKitEnv()
        env.reset(seed=42)

        hidden = env.get_hidden_state()
        assert "player0_hole" in hidden
        assert "player1_hole" in hidden
        assert len(hidden["player0_hole"]) == 2
        assert len(hidden["player1_hole"]) == 2


class TestActionApplication:
    """Tests for action application."""

    def test_fold_ends_hand(self):
        """Test that fold typically ends the hand."""
        env = PokerKitEnv()
        env.reset(seed=42)

        # Keep stepping with fold until hand ends
        done = False
        steps = 0
        while not done and steps < 100:
            legal = env.legal_actions()
            # Try to fold if possible
            fold_action = next((a for a in legal if a.type == ActionType.FOLD), None)
            if fold_action:
                _, _, done, _ = env.step(fold_action)
            else:
                # If can't fold, do any legal action
                _, _, done, _ = env.step(legal[0])
            steps += 1

        assert done, "Hand should complete within 100 steps"

    def test_check_call_continues(self):
        """Test that check/call allows game to continue."""
        env = PokerKitEnv()
        env.reset(seed=42)

        # Do a few check/calls
        for _ in range(4):
            legal = env.legal_actions()
            if not legal:
                break
            call_action = next(
                (a for a in legal if a.type == ActionType.CHECK_OR_CALL),
                legal[0]
            )
            _, _, done, _ = env.step(call_action)
            if done:
                break

    def test_bet_raise_is_valid(self):
        """Test that bet/raise is properly handled."""
        env = PokerKitEnv()
        env.reset(seed=42)

        legal = env.legal_actions()
        raise_action = next(
            (a for a in legal if a.type == ActionType.BET_OR_RAISE),
            None
        )

        if raise_action:
            obs, _, _, _ = env.step(raise_action)
            assert obs is not None


class TestHandCompletion:
    """Tests for hand completion scenarios."""

    def test_hand_completes(self):
        """Test that a hand eventually completes."""
        env = PokerKitEnv()
        env.reset(seed=42)

        done = False
        steps = 0
        while not done and steps < 200:
            legal = env.legal_actions()
            if not legal:
                break
            # Always check/call to eventually reach showdown
            action = next(
                (a for a in legal if a.type == ActionType.CHECK_OR_CALL),
                legal[0]
            )
            _, _, done, info = env.step(action)
            steps += 1

        assert done, "Hand should complete"

    def test_rewards_sum_to_zero(self):
        """Test that rewards are zero-sum."""
        env = PokerKitEnv()
        env.reset(seed=42)

        # Play to completion
        done = False
        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            action = next(
                (a for a in legal if a.type == ActionType.CHECK_OR_CALL),
                legal[0]
            )
            _, _, done, info = env.step(action)

        if done and "player0_delta" in info:
            p0_delta = info["player0_delta"]
            p1_delta = info["player1_delta"]
            # Rewards should be zero-sum (accounting for rake=0)
            assert abs(p0_delta + p1_delta) < 0.01, "Rewards should be zero-sum"
