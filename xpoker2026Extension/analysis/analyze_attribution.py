"""
Token attribution analysis for poker LLM agents.

Runs integrated gradients on saved prompts to determine which input
tokens most influence the model's action/belief outputs. Requires
a GPU and the captum library.

Usage:
    python -m analysis.analyze_attribution \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --data logs/experiment.jsonl \\
        --json-out results/attribution.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from poker_env.interp.attribution import AttributionAnalyzer
    from poker_env.interp.attention import AttentionExtractor
    INTERP_AVAILABLE = True
except ImportError:
    INTERP_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def load_prompts_from_experiment(jsonl_path: str) -> list[dict]:
    """Extract decision records that have prompt hashes (i.e. from LLM agents)."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") in ("run_config", "hand_summary"):
                continue
            if rec.get("prompt_hash"):
                records.append(rec)
    return records


def reconstruct_prompt(record: dict, cot: bool = False) -> str:
    """
    Reconstruct the user-facing prompt from a logged observation.

    This mirrors the prompt format used by HFAgent._build_action_prompt
    and APIAgent._build_action_prompt, including the system message prefix,
    so that attribution results are faithful to the actual model input.
    """
    obs = record.get("obs", {})
    hero_hole = obs.get("hero_hole", [])
    board = obs.get("board", [])
    pot = obs.get("pot_total", 0)
    street = obs.get("street", "PREFLOP")
    bet_to_call = obs.get("bet_to_call", 0)
    legal = record.get("legal_actions", [])
    history = obs.get("history", [])
    player_index = obs.get("player_index", 0)

    # Replicate the same truncation the agents apply before formatting
    try:
        from poker_env.config import DEFAULT_MAX_HISTORY_EVENTS
        max_events = DEFAULT_MAX_HISTORY_EVENTS
    except ImportError:
        max_events = 50
    if len(history) > max_events:
        history = history[:5] + history[-(max_events - 5):]

    if not history:
        history_str = "No actions yet"
    else:
        history_lines = []
        for event in history[-10:]:
            event_type = event.get("event", "")
            player = event.get("player")
            amount = event.get("amount")
            if event_type in ("POST_BLIND", "DEAL_HOLE", "DEAL_BOARD", "UNKNOWN"):
                continue
            player_name = "You" if player == player_index else "Opponent"
            if amount:
                history_lines.append(f"  {player_name}: {event_type} ({amount})")
            else:
                history_lines.append(f"  {player_name}: {event_type}")
        history_str = "\n".join(history_lines) if history_lines else "No betting actions yet"

    try:
        from poker_env.agents.prompts import get_action_system_message
        system_msg = get_action_system_message(cot=cot)
    except ImportError:
        system_msg = "You are a poker player."

    if cot:
        instruction = 'Provide your reasoning then your action as JSON: {"action": "YOUR_CHOICE"}'
    else:
        instruction = 'Return ONLY: {"action": "YOUR_CHOICE"}'

    return (
        f"{system_msg}\n\n"
        f"Current poker situation:\n"
        f"- Your cards: {' '.join(hero_hole)}\n"
        f"- Board: {' '.join(board) if board else 'None (preflop)'}\n"
        f"- Pot: {pot}\n"
        f"- Street: {street}\n"
        f"- Bet to call: {bet_to_call}\n"
        f"- Legal actions: {legal}\n\n"
        f"Recent actions:\n{history_str}\n\n"
        f"Choose your action. {instruction}"
    )


def run_attribution(
    model,
    tokenizer,
    prompts: list[str],
    target_tokens: list[str] | None = None,
    n_steps: int = 50,
    max_prompts: int = 50,
) -> list[dict]:
    """Run attribution analysis on a set of prompts."""
    analyzer = AttributionAnalyzer(model, tokenizer)

    results = []
    for i, prompt in enumerate(prompts[:max_prompts]):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

        multi_token_warning = False
        if target_tokens and i < len(target_tokens):
            target_ids = tokenizer.encode(target_tokens[i], add_special_tokens=False)
            if target_ids:
                target_id = target_ids[0]
                if len(target_ids) > 1:
                    multi_token_warning = True
            else:
                with torch.no_grad():
                    out = model(input_ids.to(model.device))
                    target_id = out.logits[0, -1].argmax().item()
        else:
            with torch.no_grad():
                out = model(input_ids.to(model.device))
                target_id = out.logits[0, -1].argmax().item()

        result = analyzer.attribute(input_ids, target_id, n_steps=n_steps)

        tokens = result["tokens"]
        categories = [AttentionExtractor._categorize_token(t) for t in tokens]
        cat_fracs = analyzer.attribute_to_categories(
            input_ids, target_id, categories, n_steps=n_steps,
        )

        entry = {
            "prompt_idx": i,
            "target_token": tokenizer.decode([target_id]) if isinstance(target_id, int) else tokenizer.decode(target_id[:1]),
            "top_5": result["top_5"],
            "category_fractions": cat_fracs,
        }
        if multi_token_warning:
            entry["multi_token_target"] = True
        results.append(entry)

        if (i + 1) % 10 == 0:
            print(f"  Attributed {i + 1}/{min(len(prompts), max_prompts)} prompts")

    return results


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate attribution results across prompts."""
    cat_values: dict[str, list[float]] = defaultdict(list)

    for r in results:
        for cat, frac in r.get("category_fractions", {}).items():
            cat_values[cat].append(frac)

    summary = {}
    for cat, values in cat_values.items():
        summary[cat] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": len(values),
        }

    return {
        "num_prompts": len(results),
        "category_attribution": summary,
    }


def plot_attribution(
    category_stats: dict[str, dict[str, float]],
    title: str = "Input Attribution by Token Category",
    output_path: str | None = None,
) -> None:
    if not MPL_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return

    cats = sorted(category_stats.keys())
    means = [category_stats[c]["mean"] for c in cats]
    stds = [category_stats[c]["std"] for c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cats, means, yerr=stds, capsize=5, alpha=0.8, color="coral")
    ax.set_ylabel("Fraction of Total Attribution")
    ax.set_title(title)
    ax.set_ylim(0, 1)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.1%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Token attribution analysis")
    parser.add_argument("--data", required=True, help="Experiment JSONL file")
    parser.add_argument("--model", required=True, help="HF model ID")
    parser.add_argument("--max-prompts", type=int, default=50, help="Max prompts to analyze")
    parser.add_argument("--n-steps", type=int, default=50, help="IG interpolation steps")
    parser.add_argument("--plot", type=str, default=None, help="Save plot")
    parser.add_argument("--json-out", type=str, default=None, help="Save JSON results")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("Error: torch required. pip install torch")
        return
    if not INTERP_AVAILABLE:
        print("Error: captum required. pip install captum")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    print(f"Loading experiment data from {args.data}")
    records = load_prompts_from_experiment(args.data)
    print(f"  Found {len(records)} LLM decision records")

    prompts = [reconstruct_prompt(r, cot=bool(r.get("cot_reasoning"))) for r in records]
    actions = [r.get("agent_action", "") for r in records]

    print(f"Running attribution on {min(len(prompts), args.max_prompts)} prompts...")
    results = run_attribution(
        model, tokenizer, prompts,
        target_tokens=actions,
        n_steps=args.n_steps,
        max_prompts=args.max_prompts,
    )

    summary = aggregate_results(results)

    print(f"\nAttribution by token category ({summary['num_prompts']} prompts):")
    for cat in sorted(summary["category_attribution"].keys()):
        s = summary["category_attribution"][cat]
        print(f"  {cat:20s}: {s['mean']:.3f} +/- {s['std']:.3f}")

    if args.plot:
        plot_attribution(summary["category_attribution"], output_path=args.plot)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {args.json_out}")


if __name__ == "__main__":
    main()
