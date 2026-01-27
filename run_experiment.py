#!/usr/bin/env python3
"""
Run poker experiments with specified agents.

Usage:
    # 2 players (heads-up)
    python run_experiment.py --agent random --hands 100 --seed 42 --out logs/test.jsonl -v

    # HF agent with default 8B model
    python run_experiment.py --agent hf --opponent call --hands 10 --elicit-beliefs -v

    # HF agent with 70B research model
    python run_experiment.py --agent hf --opponent call --hf-model meta-llama/Llama-3.1-70B-Instruct --hands 10 -v

Configuration defaults are in poker_env/config.py
"""

import argparse
import random
import warnings
from pathlib import Path
from typing import Optional

from poker_env.env import PokerKitEnv
from poker_env.agents import BaseAgent, RandomAgent, CallAgent, HF_AVAILABLE
from poker_env.oracle import EquityOracle
from poker_env.logging import DecisionLogger
from poker_env.config import (
    DEFAULT_MODEL_ID,
    RESEARCH_MODEL_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_ACTION_MAX_TOKENS,
    DEFAULT_BELIEF_MAX_TOKENS,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_HISTORY_EVENTS,
    DEFAULT_BELIEF_FORMAT,
)

# Conditionally import HFAgent
if HF_AVAILABLE:
    from poker_env.agents import HFAgent


def create_agent(
    agent_type: str,
    seed: Optional[int] = None,
    name: str = "",
    hf_model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    action_max_new_tokens: int | None = None,
    belief_max_new_tokens: int | None = None,
    max_input_tokens: int | None = None,
    belief_format: str | None = None,
) -> BaseAgent:
    """
    Create an agent by type name.

    Args:
        agent_type: One of "random", "call", "hf"
        seed: Optional seed for random agent
        name: Optional name for the agent
        hf_model: HuggingFace model ID (for hf agent, defaults to config)
        temperature: Generation temperature (for hf agent, defaults to config)
        top_p: Top-p sampling parameter (for hf agent, defaults to config)
        action_max_new_tokens: Max tokens for action (for hf agent, defaults to config)
        belief_max_new_tokens: Max tokens for belief (for hf agent, defaults to config)
        max_input_tokens: Max input context length (for hf agent, defaults to config)
        belief_format: "compact" or "full" (for hf agent, defaults to config)

    Returns:
        BaseAgent instance
    """
    if agent_type == "random":
        return RandomAgent(seed=seed, name=name or "RandomAgent")
    elif agent_type == "call":
        return CallAgent(name=name or "CallAgent")
    elif agent_type == "hf":
        if not HF_AVAILABLE:
            raise ImportError(
                "HFAgent requires torch and transformers. "
                "Install with: pip install torch transformers accelerate"
            )
        return HFAgent(
            model_id=hf_model,
            temperature=temperature,
            top_p=top_p,
            action_max_new_tokens=action_max_new_tokens,
            belief_max_new_tokens=belief_max_new_tokens,
            max_input_tokens=max_input_tokens,
            belief_format=belief_format,
            name=name or "HFAgent",
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_agents(
    agent_types: list[str],
    num_players: int,
    base_seed: int,
    hf_model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    action_max_new_tokens: int | None = None,
    belief_max_new_tokens: int | None = None,
    max_input_tokens: int | None = None,
    belief_format: str | None = None,
) -> list[BaseAgent]:
    """
    Create agents for all players.

    Args:
        agent_types: List of agent types (will be cycled if shorter than num_players)
        num_players: Number of players
        base_seed: Base seed for random agents
        hf_model: HuggingFace model ID (for hf agents)
        temperature: Generation temperature (for hf agents)
        top_p: Top-p sampling parameter (for hf agents)
        action_max_new_tokens: Max tokens for action (for hf agents)
        belief_max_new_tokens: Max tokens for belief (for hf agents)
        max_input_tokens: Max input context length (for hf agents)
        belief_format: "compact" or "full" (for hf agents)

    Returns:
        List of BaseAgent instances
    """
    agents = []
    hf_agent_instance = None  # Reuse same HF agent for efficiency
    
    for i in range(num_players):
        agent_type = agent_types[i % len(agent_types)]
        seed = base_seed + i if agent_type == "random" else None
        
        if agent_type == "hf":
            # Reuse same HFAgent instance (model only needs to load once)
            if hf_agent_instance is None:
                hf_agent_instance = create_agent(
                    agent_type,
                    name=f"Player{i}_{agent_type}",
                    hf_model=hf_model,
                    temperature=temperature,
                    top_p=top_p,
                    action_max_new_tokens=action_max_new_tokens,
                    belief_max_new_tokens=belief_max_new_tokens,
                    max_input_tokens=max_input_tokens,
                    belief_format=belief_format,
                )
            agents.append(hf_agent_instance)
        else:
            agents.append(create_agent(agent_type, seed=seed, name=f"Player{i}_{agent_type}"))
    
    return agents


def get_agent_configs(agents: list[BaseAgent]) -> list[dict]:
    """Get configuration dicts for all agents."""
    configs = []
    seen_hf = False
    
    for agent in agents:
        config = {
            "name": agent.name,
            "type": type(agent).__name__,
        }
        if hasattr(agent, "seed"):
            config["seed"] = agent.seed
        # Add HF-specific config (only once since agents may be shared)
        if HF_AVAILABLE and hasattr(agent, "model_id") and not seen_hf:
            config["model_id"] = agent.model_id
            config["temperature"] = agent.temperature
            config["top_p"] = agent.top_p
            config["action_max_new_tokens"] = agent.action_max_new_tokens
            config["belief_max_new_tokens"] = agent.belief_max_new_tokens
            config["max_input_tokens"] = agent.max_input_tokens
            config["belief_format"] = agent.belief_format
            seen_hf = True
        configs.append(config)
    return configs


def run_single_hand(
    env: PokerKitEnv,
    agents: list[BaseAgent],
    oracle: EquityOracle,
    logger: Optional[DecisionLogger],
    seed: int,
    compute_oracle: bool = True,
    elicit_beliefs: bool = False,
    randomize_probe_order: bool = False,
) -> dict:
    """
    Run a single hand of poker.

    Args:
        env: Poker environment
        agents: List of agents for each player
        oracle: Equity oracle (ground truth, computed AFTER agent acts)
        logger: Optional decision logger
        seed: Random seed for this hand
        compute_oracle: Whether to compute oracle truth at each decision
        elicit_beliefs: Whether to call belief() on agents (for LLM agents)
        randomize_probe_order: Whether to randomize action/belief probe order

    Returns:
        Dict with hand results
    """
    # Reset agents
    for agent in agents:
        agent.reset()

    # Start new hand
    obs = env.reset(seed=seed)

    if logger:
        logger.start_hand(env.hand_id, seed)

    done = False
    while not done:
        player = env.current_player()
        agent = agents[player]

        # Determine probe order
        if randomize_probe_order:
            probe_order = random.choice(["action_first", "belief_first"])
        else:
            probe_order = "action_first"

        # Initialize metadata
        action_metadata = None
        belief_metadata = None
        belief = None
        prompt_template_id = None
        prompt_hash = None

        # Check if agent is HFAgent with metadata support
        is_hf_agent = HF_AVAILABLE and hasattr(agent, 'act_with_metadata')

        if probe_order == "belief_first" and elicit_beliefs:
            # Get belief first
            if is_hf_agent:
                belief, belief_meta = agent.belief_with_metadata(obs)
                belief_metadata = belief_meta.to_dict() if belief_meta else None
                prompt_hash = belief_meta.prompt_hash if belief_meta else None
                prompt_template_id = belief_meta.prompt_template_id if belief_meta else None
            else:
                belief = agent.belief(obs)

            # Then get action
            if is_hf_agent:
                action, action_meta = agent.act_with_metadata(obs)
                action_metadata = action_meta.to_dict() if action_meta else None
                if action_meta and not prompt_hash:
                    prompt_hash = action_meta.prompt_hash
                    prompt_template_id = action_meta.prompt_template_id
            else:
                action = agent.act(obs)
        else:
            # Action first (default)
            if is_hf_agent:
                action, action_meta = agent.act_with_metadata(obs)
                action_metadata = action_meta.to_dict() if action_meta else None
                prompt_hash = action_meta.prompt_hash if action_meta else None
                prompt_template_id = action_meta.prompt_template_id if action_meta else None
            else:
                action = agent.act(obs)

            # Then get belief if requested
            if elicit_beliefs:
                if is_hf_agent:
                    belief, belief_meta = agent.belief_with_metadata(obs)
                    belief_metadata = belief_meta.to_dict() if belief_meta else None
                else:
                    belief = agent.belief(obs)

        # Oracle computed AFTER agent acts - for logging only, never seen by agent
        equity_truth = None
        if compute_oracle:
            hidden = env.get_hidden_state()
            hero_hole = hidden.get(f"player{player}_hole", [])

            # Gather all opponent hole cards
            opponent_holes = []
            for i in range(env.num_players):
                if i != player:
                    opp_hole = hidden.get(f"player{i}_hole", [])
                    if opp_hole:
                        opponent_holes.append(opp_hole)

            # Each element in board_cards is a list with one card
            board = [repr(card_list[0]) for card_list in env.state.board_cards if card_list]

            if hero_hole and opponent_holes:
                equity_truth = oracle.compute(hero_hole, opponent_holes, board)

        # Log decision (oracle truth is for evaluation, agent never saw it)
        if logger:
            logger.log_decision(
                obs=obs,
                hidden=env.get_hidden_state(),
                agent_action=action,
                agent_belief=belief,
                equity_given_true_hands=equity_truth,
                action_metadata=action_metadata,
                belief_metadata=belief_metadata,
                prompt_template_id=prompt_template_id,
                prompt_hash=prompt_hash,
                probe_order=probe_order if elicit_beliefs else None,
            )

        # Apply action
        obs, reward, done, info = env.step(action)

    # Log hand completion
    if logger:
        deltas = info.get("deltas", {})
        logger.end_hand(
            final_stacks=info.get("final_stacks", []),
            deltas=deltas,
            showdown=info.get("showdown", {}),
        )

    return {
        "hand_id": env.hand_id,
        "seed": seed,
        "final_stacks": info.get("final_stacks", []),
        "deltas": info.get("deltas", {}),
        "rewards": info.get("rewards", {}),
    }


def run_experiment(
    num_hands: int,
    agent_types: list[str],
    num_players: int,
    output_path: str,
    base_seed: int = 42,
    stacks: tuple[int, ...] | None = None,
    blinds: tuple[int, int] = (1, 2),
    small_bet: int = 2,
    big_bet: int = 4,
    compute_oracle: bool = True,
    verbose: bool = False,
    hf_model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    action_max_new_tokens: int | None = None,
    belief_max_new_tokens: int | None = None,
    max_input_tokens: int | None = None,
    belief_format: str | None = None,
    elicit_beliefs: bool = False,
    randomize_probe_order: bool = False,
) -> dict:
    """
    Run a full experiment with multiple hands.

    Args:
        num_hands: Number of hands to play
        agent_types: List of agent types for each player
        num_players: Number of players (2-6, heads-up recommended)
        output_path: Path for JSONL output
        base_seed: Base random seed
        stacks: Starting stack sizes (None = 200 each)
        blinds: Blind amounts
        small_bet: Small bet size
        big_bet: Big bet size
        compute_oracle: Whether to compute oracle truth
        verbose: Print progress
        hf_model: HuggingFace model ID (for hf agents, None = use config default)
        temperature: Generation temperature (for hf agents, None = use config default)
        top_p: Top-p sampling (for hf agents, None = use config default)
        action_max_new_tokens: Max tokens for action (for hf agents, None = use config default)
        belief_max_new_tokens: Max tokens for belief (for hf agents, None = use config default)
        max_input_tokens: Max input context length (for hf agents, None = use config default)
        belief_format: "compact" or "full" (for hf agents, None = use config default)
        elicit_beliefs: Whether to call belief() on agents
        randomize_probe_order: Whether to randomize action/belief probe order

    Returns:
        Dict with experiment summary
    """
    # Suppress multi-way warning during experiment if user explicitly chose it
    with warnings.catch_warnings():
        if num_players > 2:
            warnings.filterwarnings("ignore", message="Multi-way.*")

        # Create environment
        env = PokerKitEnv(
            num_players=num_players,
            stacks=stacks,
            blinds=blinds,
            small_bet=small_bet,
            big_bet=big_bet,
        )

    # Create agents
    agents = create_agents(
        agent_types,
        num_players,
        base_seed,
        hf_model=hf_model,
        temperature=temperature,
        top_p=top_p,
        action_max_new_tokens=action_max_new_tokens,
        belief_max_new_tokens=belief_max_new_tokens,
        max_input_tokens=max_input_tokens,
        belief_format=belief_format,
    )

    # Create oracle
    oracle = EquityOracle(num_samples=5000, seed=base_seed + 100)

    # Tracking
    total_deltas = {f"player{i}": 0.0 for i in range(num_players)}
    hand_results = []

    # Create logger with config info
    with DecisionLogger(
        output_path,
        env_config_hash=env.get_config_hash(),
        agent_configs=get_agent_configs(agents),
    ) as logger:
        for i in range(num_hands):
            hand_seed = base_seed + i * 1000

            result = run_single_hand(
                env=env,
                agents=agents,
                oracle=oracle,
                logger=logger,
                seed=hand_seed,
                compute_oracle=compute_oracle,
                elicit_beliefs=elicit_beliefs,
                randomize_probe_order=randomize_probe_order,
            )

            hand_results.append(result)
            for key, delta in result.get("deltas", {}).items():
                player_key = key.replace("_delta", "")
                if player_key in total_deltas:
                    total_deltas[player_key] += delta

            if verbose and (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_hands} hands")

    # Summary
    summary = {
        "num_hands": num_hands,
        "num_players": num_players,
        "agents": [a.name for a in agents],
        "env_config_hash": env.get_config_hash(),
        "total_deltas": total_deltas,
        "avg_deltas": {k: v / num_hands for k, v in total_deltas.items()},
        "output_path": output_path,
    }

    if verbose:
        print(f"\n=== Experiment Summary ({num_players} players) ===")
        print(f"Hands played: {num_hands}")
        print(f"Config hash: {env.get_config_hash()}")
        for i, agent in enumerate(agents):
            delta = total_deltas[f"player{i}"]
            avg = delta / num_hands
            print(f"Player {i} ({agent.name}): {delta:+.1f} chips ({avg:+.2f}/hand)")
        print(f"Output: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run poker experiments with specified agents. "
                    "Configuration defaults are in poker_env/config.py"
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (2-6, default: 2). Heads-up recommended for belief research.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=["random", "call", "hf"],
        help="Default agent type for all players (default: random)",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help="Comma-separated agent types for each player (e.g., 'random,call,random,call')",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        choices=["random", "call"],
        help="Agent type for opponents (heads-up only, overrides --agent for player 1)",
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
    
    # HuggingFace agent options
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID}, research: {RESEARCH_MODEL_ID})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"Generation temperature (default: {DEFAULT_TEMPERATURE}, use 0 for deterministic)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help=f"Top-p sampling parameter (default: {DEFAULT_TOP_P})",
    )
    parser.add_argument(
        "--action-max-new-tokens",
        type=int,
        default=None,
        help=f"Max tokens for action generation (default: {DEFAULT_ACTION_MAX_TOKENS})",
    )
    parser.add_argument(
        "--belief-max-new-tokens",
        type=int,
        default=None,
        help=f"Max tokens for belief generation (default: {DEFAULT_BELIEF_MAX_TOKENS})",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help=f"Max input context length (default: {DEFAULT_MAX_INPUT_TOKENS})",
    )
    parser.add_argument(
        "--belief-format",
        type=str,
        default=None,
        choices=["compact", "full"],
        help=f"Belief output format (default: {DEFAULT_BELIEF_FORMAT})",
    )
    parser.add_argument(
        "--elicit-beliefs",
        action="store_true",
        help="Call belief() on agents at each decision (for LLM belief analysis)",
    )
    parser.add_argument(
        "--randomize-probe-order",
        action="store_true",
        help="Randomize order of action/belief probing (for robustness testing)",
    )
    parser.add_argument(
        "--i-know-what-im-doing",
        action="store_true",
        help="Allow multi-way belief experiments (not recommended for research)",
    )

    args = parser.parse_args()
    
    # Warn about multi-way belief experiments
    if args.elicit_beliefs and args.num_players > 2:
        if not args.i_know_what_im_doing:
            print("\n" + "="*70)
            print("WARNING: Multi-way belief experiments are not recommended!")
            print("="*70)
            print("""
Belief elicitation with >2 players introduces:
- Multiple opponent posteriors to track
- Exponentially complex action-likelihood modeling
- Side pot complications

For valid research results, use heads-up (2 players) for belief experiments.

To proceed anyway, add: --i-know-what-im-doing
""")
            print("="*70 + "\n")
            import sys
            sys.exit(1)

    # Determine agent types
    if args.agents:
        # Explicit list of agents
        agent_types = [a.strip() for a in args.agents.split(",")]
    elif args.opponent and args.num_players == 2:
        # Heads-up with different agents
        agent_types = [args.agent, args.opponent]
    else:
        # All same agent type
        agent_types = [args.agent]

    run_experiment(
        num_hands=args.hands,
        agent_types=agent_types,
        num_players=args.num_players,
        output_path=args.out,
        base_seed=args.seed,
        compute_oracle=not args.no_oracle,
        verbose=args.verbose,
        hf_model=args.hf_model,
        temperature=args.temperature,
        top_p=args.top_p,
        action_max_new_tokens=args.action_max_new_tokens,
        belief_max_new_tokens=args.belief_max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        belief_format=args.belief_format,
        elicit_beliefs=args.elicit_beliefs,
        randomize_probe_order=args.randomize_probe_order,
    )


if __name__ == "__main__":
    main()
