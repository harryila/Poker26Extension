#!/usr/bin/env python3
"""
Run poker experiments with specified agents.

Usage:
    python run_experiment.py --agent random --hands 100 --seed 42 --out logs/test.jsonl
    python run_experiment.py --agent call --hands 50 --seed 123 --out logs/call_vs_random.jsonl
"""

import argparse
from pathlib import Path
from typing import Optional

from poker_env.env import PokerKitEnv
from poker_env.agents import BaseAgent, RandomAgent, CallAgent
from poker_env.oracle import WinProbOracle
from poker_env.logging import DecisionLogger


def create_agent(agent_type: str, seed: Optional[int] = None) -> BaseAgent:
    """
    Create an agent by type name.

    Args:
        agent_type: One of "random", "call"
        seed: Optional seed for random agent

    Returns:
        BaseAgent instance
    """
    if agent_type == "random":
        return RandomAgent(seed=seed)
    elif agent_type == "call":
        return CallAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_single_hand(
    env: PokerKitEnv,
    agent0: BaseAgent,
    agent1: BaseAgent,
    oracle: WinProbOracle,
    logger: Optional[DecisionLogger],
    seed: int,
    compute_oracle: bool = True,
) -> dict:
    """
    Run a single hand of poker.

    Args:
        env: Poker environment
        agent0: Agent for player 0
        agent1: Agent for player 1
        oracle: Win probability oracle
        logger: Optional decision logger
        seed: Random seed for this hand
        compute_oracle: Whether to compute oracle truth at each decision

    Returns:
        Dict with hand results
    """
    agents = [agent0, agent1]

    # Reset agents
    agent0.reset()
    agent1.reset()

    # Start new hand
    obs = env.reset(seed=seed)

    if logger:
        logger.start_hand(env.hand_id, seed)

    done = False
    while not done:
        player = env.current_player()
        agent = agents[player]

        # Get agent's action and optional belief
        action = agent.act(obs)
        belief = agent.belief(obs)

        # Compute oracle truth if requested
        oracle_truth = None
        if compute_oracle:
            hidden = env.get_hidden_state()
            hero_hole = hidden.get(f"player{player}_hole", [])
            villain_hole = hidden.get(f"player{1-player}_hole", [])
            # Each element in board_cards is a list with one card
            board = [repr(card_list[0]) for card_list in env.state.board_cards if card_list]

            if hero_hole and villain_hole:
                oracle_truth = oracle.compute(hero_hole, villain_hole, board)

        # Log decision
        if logger:
            logger.log_decision(
                obs=obs,
                hidden=env.get_hidden_state(),
                agent_action=action,
                agent_belief=belief,
                oracle_truth=oracle_truth,
            )

        # Apply action
        obs, reward, done, info = env.step(action)

    # Log hand completion
    if logger:
        logger.end_hand(
            final_stacks=info.get("final_stacks", []),
            player0_delta=info.get("player0_delta", 0),
            player1_delta=info.get("player1_delta", 0),
            showdown=info.get("showdown", {}),
        )

    return {
        "hand_id": env.hand_id,
        "seed": seed,
        "final_stacks": info.get("final_stacks", []),
        "player0_delta": info.get("player0_delta", 0),
        "player1_delta": info.get("player1_delta", 0),
    }


def run_experiment(
    num_hands: int,
    agent0_type: str,
    agent1_type: str,
    output_path: str,
    base_seed: int = 42,
    stacks: tuple = (200, 200),
    blinds: tuple = (1, 2),
    small_bet: int = 2,
    big_bet: int = 4,
    compute_oracle: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Run a full experiment with multiple hands.

    Args:
        num_hands: Number of hands to play
        agent0_type: Type of agent for player 0
        agent1_type: Type of agent for player 1
        output_path: Path for JSONL output
        base_seed: Base random seed
        stacks: Starting stack sizes
        blinds: Blind amounts
        small_bet: Small bet size
        big_bet: Big bet size
        compute_oracle: Whether to compute oracle truth
        verbose: Print progress

    Returns:
        Dict with experiment summary
    """
    # Create environment
    env = PokerKitEnv(
        stacks=stacks,
        blinds=blinds,
        small_bet=small_bet,
        big_bet=big_bet,
    )

    # Create agents
    agent0 = create_agent(agent0_type, seed=base_seed)
    agent1 = create_agent(agent1_type, seed=base_seed + 1)

    # Create oracle
    oracle = WinProbOracle(num_samples=5000, seed=base_seed + 2)

    # Tracking
    total_player0_delta = 0.0
    total_player1_delta = 0.0
    hand_results = []

    with DecisionLogger(output_path) as logger:
        for i in range(num_hands):
            hand_seed = base_seed + i * 1000

            result = run_single_hand(
                env=env,
                agent0=agent0,
                agent1=agent1,
                oracle=oracle,
                logger=logger,
                seed=hand_seed,
                compute_oracle=compute_oracle,
            )

            hand_results.append(result)
            total_player0_delta += result["player0_delta"]
            total_player1_delta += result["player1_delta"]

            if verbose and (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_hands} hands")

    # Summary
    summary = {
        "num_hands": num_hands,
        "agent0": agent0_type,
        "agent1": agent1_type,
        "total_player0_delta": total_player0_delta,
        "total_player1_delta": total_player1_delta,
        "avg_player0_delta": total_player0_delta / num_hands,
        "avg_player1_delta": total_player1_delta / num_hands,
        "output_path": output_path,
    }

    if verbose:
        print("\n=== Experiment Summary ===")
        print(f"Hands played: {num_hands}")
        print(f"Agent 0 ({agent0_type}): {total_player0_delta:+.1f} chips ({total_player0_delta/num_hands:+.2f}/hand)")
        print(f"Agent 1 ({agent1_type}): {total_player1_delta:+.1f} chips ({total_player1_delta/num_hands:+.2f}/hand)")
        print(f"Output: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run poker experiments with specified agents"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=["random", "call"],
        help="Agent type for player 0 (default: random)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "call"],
        help="Agent type for player 1 (default: random)",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=100,
        help="Number of hands to play (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="logs/experiment.jsonl",
        help="Output JSONL file path (default: logs/experiment.jsonl)",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Skip oracle computation (faster)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress",
    )

    args = parser.parse_args()

    run_experiment(
        num_hands=args.hands,
        agent0_type=args.agent,
        agent1_type=args.opponent,
        output_path=args.out,
        base_seed=args.seed,
        compute_oracle=not args.no_oracle,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
