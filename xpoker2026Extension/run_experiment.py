#!/usr/bin/env python3
"""
Run poker experiments with specified agents.

Usage:
    # 2 players (heads-up)
    python run_experiment.py --agent random --hands 100 --seed 42 --out logs/test.jsonl -v

    # HF agent with default 8B model
    python run_experiment.py --agent hf --opponent call --hands 10 --elicit-beliefs -v

    # HF agent with Mistral via short name
    python run_experiment.py --agent hf --hf-model mistral-7b --opponent threshold --hands 20 -v

    # API agent (OpenAI GPT-4o)
    python run_experiment.py --agent api --api-provider openai --api-model gpt-4o --opponent threshold --hands 10 -v

    # Chain-of-Thought mode
    python run_experiment.py --agent hf --opponent threshold --hands 10 --elicit-beliefs --cot -v

    # Logit lens (open models only)
    python run_experiment.py --agent hf --opponent threshold --hands 5 --logit-lens -v

Configuration defaults are in poker_env/config.py
"""

import argparse
import random
import sys
import warnings
from pathlib import Path
from typing import Optional

from poker_env.env import PokerKitEnv
from poker_env.agents import (
    BaseAgent, RandomAgent, CallAgent, ThresholdAgent,
    HF_AVAILABLE, API_AVAILABLE,
)
from poker_env.oracle import EquityOracle
from poker_env.logging import DecisionLogger
from poker_env.config import (
    DEFAULT_MODEL_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_ACTION_MAX_TOKENS,
    DEFAULT_BELIEF_MAX_TOKENS,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_HISTORY_EVENTS,
    DEFAULT_BELIEF_FORMAT,
    MODEL_REGISTRY,
)

if HF_AVAILABLE:
    from poker_env.agents import HFAgent

if API_AVAILABLE:
    from poker_env.agents import APIAgent


def _write_sidecar(path: str, hand_id: str, decision_idx: int, data: dict) -> None:
    """Append a record to a JSONL sidecar file."""
    import json as _json
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(_json.dumps({"hand_id": hand_id, "decision_idx": decision_idx, **data}) + "\n")


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
    opponent_preset: str | None = None,
    cot_mode: bool = False,
    logit_lens: bool = False,
    capture_logprobs: bool = False,
    top_logprobs: int = 20,
    api_provider: str | None = None,
    api_model: str | None = None,
) -> BaseAgent:
    """Create an agent by type name."""
    if agent_type == "random":
        return RandomAgent(seed=seed, name=name or "RandomAgent")
    elif agent_type == "call":
        return CallAgent(name=name or "CallAgent")
    elif agent_type == "threshold":
        preset = opponent_preset or "default"
        return ThresholdAgent(preset=preset, seed=seed, name=name or f"ThresholdAgent_{preset}")
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
            cot_mode=cot_mode,
            logit_lens=logit_lens,
            capture_logprobs=capture_logprobs,
            top_logprobs=top_logprobs,
            name=name or "HFAgent",
        )
    elif agent_type == "api":
        if not API_AVAILABLE:
            raise ImportError(
                "APIAgent requires openai, anthropic, or google-genai. "
                "Install with: pip install openai anthropic google-genai"
            )
        return APIAgent(
            provider=api_provider or "openai",
            model=api_model,
            temperature=temperature,
            top_p=top_p,
            action_max_new_tokens=action_max_new_tokens,
            belief_max_new_tokens=belief_max_new_tokens,
            belief_format=belief_format,
            cot_mode=cot_mode,
            name=name or "APIAgent",
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
    opponent_preset: str | None = None,
    cot_mode: bool = False,
    logit_lens: bool = False,
    capture_logprobs: bool = False,
    top_logprobs: int = 20,
    api_provider: str | None = None,
    api_model: str | None = None,
) -> list[BaseAgent]:
    """Create agents for all players, reusing heavy instances.
    
    When multiple players share the same agent type (e.g., both HF),
    the same instance is reused to avoid duplicating GPU memory. This is
    safe because run_single_hand processes decisions sequentially, but
    callers must read metadata immediately after each act/belief call.
    """
    agents = []
    shared_instances: dict[str, BaseAgent] = {}

    for i in range(num_players):
        agent_type = agent_types[i % len(agent_types)]
        seed = base_seed + i if agent_type in ("random", "threshold") else None

        # Reuse heavy agent instances (HF model or API client)
        if agent_type in ("hf", "api"):
            if agent_type not in shared_instances:
                shared_instances[agent_type] = create_agent(
                    agent_type,
                    name=f"Player{i}_{agent_type}",
                    hf_model=hf_model,
                    temperature=temperature,
                    top_p=top_p,
                    action_max_new_tokens=action_max_new_tokens,
                    belief_max_new_tokens=belief_max_new_tokens,
                    max_input_tokens=max_input_tokens,
                    belief_format=belief_format,
                    cot_mode=cot_mode,
                    logit_lens=logit_lens,
                    capture_logprobs=capture_logprobs,
                    top_logprobs=top_logprobs,
                    api_provider=api_provider,
                    api_model=api_model,
                )
            agents.append(shared_instances[agent_type])
        else:
            agents.append(create_agent(
                agent_type,
                seed=seed,
                name=f"Player{i}_{agent_type}",
                opponent_preset=opponent_preset,
            ))

    return agents


def get_agent_configs(agents: list[BaseAgent]) -> list[dict]:
    """Get configuration dicts for all agents."""
    configs = []
    seen_ids = set()

    for agent in agents:
        # Use get_config() if available (HFAgent, APIAgent, ThresholdAgent all have it)
        if hasattr(agent, "get_config"):
            agent_id = id(agent)
            if agent_id in seen_ids:
                configs.append({"name": agent.name, "type": type(agent).__name__, "shared_instance": True})
            else:
                seen_ids.add(agent_id)
                config = agent.get_config()
                config["name"] = agent.name
                configs.append(config)
        else:
            config = {
                "name": agent.name,
                "type": type(agent).__name__,
            }
            if hasattr(agent, "seed"):
                config["seed"] = agent.seed
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
    """Run a single hand of poker."""
    for agent in agents:
        agent.reset()

    obs = env.reset(seed=seed)

    if logger:
        logger.start_hand(env.hand_id, seed)

    done = False
    while not done:
        player = env.current_player()
        agent = agents[player]

        if randomize_probe_order:
            probe_rng = random.Random(seed ^ env.current_player() ^ (logger.decision_idx if logger else 0))
            probe_order = probe_rng.choice(["action_first", "belief_first"])
        else:
            probe_order = "action_first"

        action_metadata = None
        belief_metadata = None
        belief = None
        prompt_template_id = None
        prompt_hash = None
        cot_reasoning = None
        interp_summary = None

        # Protocol-based detection: any agent with act_with_metadata is a "rich" agent
        has_metadata = hasattr(agent, 'act_with_metadata') and hasattr(agent, 'belief_with_metadata')

        if probe_order == "belief_first" and elicit_beliefs:
            if has_metadata:
                belief, belief_meta = agent.belief_with_metadata(obs)
                belief_metadata = belief_meta.to_dict() if belief_meta else None
                prompt_hash = belief_meta.prompt_hash if belief_meta else None
                prompt_template_id = belief_meta.prompt_template_id if belief_meta else None
            else:
                belief = agent.belief(obs)
            if has_metadata:
                action, action_meta = agent.act_with_metadata(obs)
                action_metadata = action_meta.to_dict() if action_meta else None
                if action_meta and not prompt_hash:
                    prompt_hash = action_meta.prompt_hash
                    prompt_template_id = action_meta.prompt_template_id
            else:
                action = agent.act(obs)
        else:
            if has_metadata:
                action, action_meta = agent.act_with_metadata(obs)
                action_metadata = action_meta.to_dict() if action_meta else None
                prompt_hash = action_meta.prompt_hash if action_meta else None
                prompt_template_id = action_meta.prompt_template_id if action_meta else None
            else:
                action = agent.act(obs)
            if elicit_beliefs:
                if has_metadata:
                    belief, belief_meta = agent.belief_with_metadata(obs)
                    belief_metadata = belief_meta.to_dict() if belief_meta else None
                else:
                    belief = agent.belief(obs)

        # Collect CoT reasoning — HF agents have split getters, API agents
        # have only get_last_cot_reasoning()
        action_cot_reasoning = None
        belief_cot_reasoning = None
        if hasattr(agent, 'get_last_action_cot'):
            action_cot_reasoning = agent.get_last_action_cot()
        if hasattr(agent, 'get_last_belief_cot'):
            belief_cot_reasoning = agent.get_last_belief_cot() if elicit_beliefs else None
        if action_cot_reasoning or belief_cot_reasoning:
            if action_cot_reasoning and belief_cot_reasoning:
                cot_reasoning = f"[ACTION] {action_cot_reasoning}\n[BELIEF] {belief_cot_reasoning}"
            else:
                cot_reasoning = action_cot_reasoning or belief_cot_reasoning
        elif hasattr(agent, 'get_last_cot_reasoning'):
            cot_reasoning = agent.get_last_cot_reasoning()

        # Collect interp data: logit lens, attention, hidden states for probing
        if hasattr(agent, 'get_last_logit_lens_data'):
            ll_data = agent.get_last_logit_lens_data()
            if ll_data and ll_data.get("num_layers", 0) > 0:
                interp_summary = {
                    "logit_lens_layers": ll_data.get("num_layers", 0),
                    "logit_lens_positions": ll_data.get("num_positions", 0),
                }
                if logger and logger.output_path:
                    from poker_env.interp.logit_lens import LogitLensExtractor
                    _out = Path(logger.output_path)
                    sidecar = str(_out.with_name(_out.stem + "_logit_lens.jsonl"))
                    LogitLensExtractor.save_sidecar(
                        ll_data, sidecar,
                        hand_id=env.hand_id,
                        decision_idx=logger.decision_idx,
                    )

        if hasattr(agent, 'get_last_attention_data'):
            attn_data = agent.get_last_attention_data()
            if attn_data and attn_data.get("num_input_tokens", 0) > 0:
                if interp_summary is None:
                    interp_summary = {}
                interp_summary["attention_categories"] = attn_data.get("category_fractions", {})
                if logger and logger.output_path:
                    _out_a = Path(logger.output_path)
                    _write_sidecar(
                        str(_out_a.with_name(_out_a.stem + "_attention.jsonl")),
                        env.hand_id, logger.decision_idx, attn_data,
                    )

        if hasattr(agent, 'get_last_hidden_states'):
            hs_data = agent.get_last_hidden_states()
            if hs_data:
                if interp_summary is None:
                    interp_summary = {}
                interp_summary["hidden_states_captured"] = True
                if logger and logger.output_path:
                    _out_h = Path(logger.output_path)
                    _write_sidecar(
                        str(_out_h.with_name(_out_h.stem + "_hiddens.jsonl")),
                        env.hand_id, logger.decision_idx, hs_data,
                    )

        # Oracle computed AFTER agent acts - for logging only
        equity_truth = None
        if compute_oracle:
            hidden = env.get_hidden_state()
            hero_hole = hidden.get(f"player{player}_hole", [])

            opponent_holes = []
            for i in range(env.num_players):
                if i != player:
                    opp_hole = hidden.get(f"player{i}_hole", [])
                    if opp_hole:
                        opponent_holes.append(opp_hole)

            board = [repr(card_list[0]) for card_list in env.state.board_cards if card_list]

            if hero_hole and opponent_holes:
                equity_truth = oracle.compute(hero_hole, opponent_holes, board)

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
                cot_reasoning=cot_reasoning,
                interp_summary=interp_summary,
            )

        obs, reward, done, info = env.step(action)

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
    opponent_preset: str | None = None,
    cot_mode: bool = False,
    logit_lens: bool = False,
    capture_logprobs: bool = False,
    top_logprobs: int = 20,
    api_provider: str | None = None,
    api_model: str | None = None,
) -> dict:
    """Run a full experiment with multiple hands."""
    with warnings.catch_warnings():
        if num_players > 2:
            warnings.filterwarnings("ignore", message="Multi-way.*")

        env = PokerKitEnv(
            num_players=num_players,
            stacks=stacks,
            blinds=blinds,
            small_bet=small_bet,
            big_bet=big_bet,
        )

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
        opponent_preset=opponent_preset,
        cot_mode=cot_mode,
        logit_lens=logit_lens,
        capture_logprobs=capture_logprobs,
        top_logprobs=top_logprobs,
        api_provider=api_provider,
        api_model=api_model,
    )

    oracle = EquityOracle(num_samples=5000, seed=base_seed + 100)

    total_deltas = {f"player{i}": 0.0 for i in range(num_players)}
    hand_results = []

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

    summary = {
        "num_hands": num_hands,
        "num_players": num_players,
        "agents": [a.name for a in agents],
        "env_config_hash": env.get_config_hash(),
        "total_deltas": total_deltas,
        "avg_deltas": {k: v / num_hands for k, v in total_deltas.items()} if num_hands > 0 else total_deltas,
        "output_path": output_path,
    }

    if verbose:
        print(f"\n=== Experiment Summary ({num_players} players) ===")
        print(f"Hands played: {num_hands}")
        print(f"Config hash: {env.get_config_hash()}")
        for i, agent in enumerate(agents):
            delta = total_deltas[f"player{i}"]
            avg = delta / num_hands if num_hands > 0 else 0.0
            print(f"Player {i} ({agent.name}): {delta:+.1f} chips ({avg:+.2f}/hand)")
        print(f"Output: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run poker experiments with specified agents. "
                    "Configuration defaults are in poker_env/config.py"
    )
    parser.add_argument(
        "--num-players", type=int, default=2,
        help="Number of players (2-6, default: 2)",
    )
    parser.add_argument(
        "--agent", type=str, default="random",
        choices=["random", "call", "threshold", "hf", "api"],
        help="Default agent type for all players (default: random)",
    )
    parser.add_argument(
        "--agents", type=str, default=None,
        help="Comma-separated agent types (e.g., 'hf,threshold')",
    )
    parser.add_argument(
        "--opponent", type=str, default=None,
        choices=["random", "call", "threshold"],
        help="Agent type for opponents (heads-up only)",
    )
    parser.add_argument(
        "--opponent-preset", type=str, default="default",
        choices=["default", "tight_passive", "tight_aggressive",
                 "loose_passive", "loose_aggressive", "informative", "informative_v2"],
        help="Preset for threshold opponent (default: default)",
    )
    parser.add_argument("--hands", type=int, default=100, help="Number of hands (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--out", type=str, default="logs/experiment.jsonl", help="Output JSONL file")
    parser.add_argument("--no-oracle", action="store_true", help="Skip oracle computation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress")

    # HuggingFace agent options
    parser.add_argument(
        "--hf-model", type=str, default=None,
        help=f"HF model ID or short name from MODEL_REGISTRY (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help=f"Generation temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--top-p", type=float, default=None,
        help=f"Top-p sampling (default: {DEFAULT_TOP_P})",
    )
    parser.add_argument("--action-max-new-tokens", type=int, default=None)
    parser.add_argument("--belief-max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--max-input-tokens", type=int, default=None,
        help=f"Max input context length (default: {DEFAULT_MAX_INPUT_TOKENS})",
    )
    parser.add_argument(
        "--belief-format", type=str, default=None, choices=["compact", "full"],
        help=f"Belief output format (default: {DEFAULT_BELIEF_FORMAT})",
    )
    parser.add_argument("--elicit-beliefs", action="store_true", help="Call belief() on agents")
    parser.add_argument("--randomize-probe-order", action="store_true")

    # API agent options
    parser.add_argument(
        "--api-provider", type=str, default=None, choices=["openai", "anthropic", "google"],
        help="API provider (default: openai)",
    )
    parser.add_argument("--api-model", type=str, default=None, help="API model name (e.g., gpt-4o)")

    # New feature flags
    parser.add_argument("--cot", action="store_true", help="Enable Chain-of-Thought prompting")
    parser.add_argument("--logit-lens", action="store_true", help="Enable logit lens capture (HF only)")
    parser.add_argument(
        "--capture-logprobs", action="store_true",
        help="Capture per-token logprobs (HF + OpenAI). API agents capture by default for OpenAI; "
             "this flag enables HF parity. Adds memory and log size.",
    )
    parser.add_argument(
        "--top-logprobs", type=int, default=20,
        help="Top-k logprobs to record per generated token (HF only; OpenAI uses 20). Default: 20",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print available model short names and exit",
    )
    parser.add_argument("--i-know-what-im-doing", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        print("Available model short names (--hf-model):")
        for short_name, cfg in MODEL_REGISTRY.items():
            sys_role = "yes" if cfg.get("supports_system_role") else "no (merged)"
            print(f"  {short_name:16s} -> {cfg['model_id']:48s}  system_role={sys_role}")
        sys.exit(0)

    if args.elicit_beliefs and args.num_players > 2:
        if not args.i_know_what_im_doing:
            print("\n" + "=" * 70)
            print("WARNING: Multi-way belief experiments are not recommended!")
            print("=" * 70)
            print(
                "\nBelief elicitation with >2 players introduces:\n"
                "- Multiple opponent posteriors to track\n"
                "- Exponentially complex action-likelihood modeling\n"
                "- Side pot complications\n\n"
                "For valid research results, use heads-up (2 players).\n"
                "To proceed anyway, add: --i-know-what-im-doing\n"
            )
            print("=" * 70 + "\n")
            sys.exit(1)

    if args.logit_lens and args.agent == "api":
        print("WARNING: --logit-lens is not supported for API agents (no layer access).")
        print("OpenAI logprobs are captured automatically in API metadata.\n")

    # Determine agent types
    if args.agents:
        agent_types = [a.strip() for a in args.agents.split(",")]
    elif args.opponent and args.num_players == 2:
        agent_types = [args.agent, args.opponent]
    else:
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
        opponent_preset=args.opponent_preset,
        cot_mode=args.cot,
        logit_lens=args.logit_lens,
        capture_logprobs=args.capture_logprobs,
        top_logprobs=args.top_logprobs,
        api_provider=args.api_provider,
        api_model=args.api_model,
    )


if __name__ == "__main__":
    main()
