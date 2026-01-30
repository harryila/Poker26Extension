#!/usr/bin/env python3
"""
Compute update coherence metrics for belief dynamics analysis.

Analyzes whether LLM beliefs update correctly in response to:
1. Public card reveals (flop/turn/river) - CARD UPDATES
2. Opponent actions (bet/raise/call/check) - ACTION UPDATES

This separates two distinct failure modes:
- "Updates on actions but not cards" → action-reactive but no combinatorial reasoning
- "Updates on cards but not actions" → has priors but ignores behavioral signal

Usage:
    python -m analysis.compute_update_coherence logs/*enriched.jsonl \
        --output results/update_coherence.csv
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Optional
from scipy.spatial.distance import jensenshannon
from scipy import stats

from analysis.buckets import BUCKET_NAMES


def belief_to_array(belief: dict, bucket_names: list[str] = BUCKET_NAMES) -> np.ndarray:
    """Convert belief dict to numpy array in consistent order."""
    return np.array([belief.get(b, 0.0) for b in bucket_names])


def normalize_belief(arr: np.ndarray) -> Optional[np.ndarray]:
    """Clip negatives and L1-normalize. Returns None if all zeros."""
    arr = np.maximum(arr, 0)
    total = arr.sum()
    if total > 0:
        return arr / total
    return None


def compute_l2_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute L2 (Euclidean) distance."""
    return float(np.sqrt(np.sum((p - q) ** 2)))


def get_update_type(prev_record: dict, curr_record: dict) -> str:
    """
    Determine what triggered the belief update between two consecutive decisions.
    
    Returns:
        'CARD_REVEAL': New board cards were dealt (street changed)
        'ACTION': Opponent took an action (same street)
        'SELF_ACTION': Hero took an action (shouldn't trigger update)
        'UNKNOWN': Can't determine
    """
    prev_street = prev_record.get("street", prev_record.get("obs", {}).get("street"))
    curr_street = curr_record.get("street", curr_record.get("obs", {}).get("street"))
    
    # If street changed, it's a card reveal
    if prev_street != curr_street:
        return "CARD_REVEAL"
    
    # Same street - check what happened in between
    prev_history_len = len(prev_record.get("obs", {}).get("history", []))
    curr_history_len = len(curr_record.get("obs", {}).get("history", []))
    
    # If history grew, an action occurred
    if curr_history_len > prev_history_len:
        # Check if the new action was by opponent (player 1)
        curr_history = curr_record.get("obs", {}).get("history", [])
        if curr_history_len > 0:
            # Find new events
            new_events = curr_history[prev_history_len:]
            for event in new_events:
                if event.get("player") == 1 and event.get("event") in ["BET", "RAISE", "CALL", "CHECK", "FOLD"]:
                    return "ACTION"
    
    return "UNKNOWN"


def load_and_group_by_hand(filepaths: list[Path]) -> dict[str, list[dict]]:
    """Load records and group by hand_id, sorted by decision_idx."""
    hands = defaultdict(list)
    
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                record = json.loads(line)
                
                # Skip non-decision records
                if record.get("type") in ("run_config", "hand_summary"):
                    continue
                
                # Only Player 0 (LLM) has beliefs
                if record.get("player_to_act") != 0:
                    continue
                
                # Must have belief and oracles
                if not record.get("agent_belief") or not record.get("oracle_strategy_aware"):
                    continue
                
                hand_id = record.get("hand_id")
                if hand_id:
                    record["_source"] = filepath.name
                    hands[hand_id].append(record)
    
    # Sort each hand by decision_idx
    for hand_id in hands:
        hands[hand_id].sort(key=lambda x: x.get("decision_idx", 0))
    
    return hands


def compute_update_metrics(
    prev_llm: np.ndarray, 
    curr_llm: np.ndarray,
    prev_oracle: np.ndarray,
    curr_oracle: np.ndarray,
) -> dict:
    """Compute metrics for a single belief update."""
    
    # Compute update magnitudes
    llm_delta = curr_llm - prev_llm
    oracle_delta = curr_oracle - prev_oracle
    
    llm_magnitude = compute_l2_distance(prev_llm, curr_llm)
    oracle_magnitude = compute_l2_distance(prev_oracle, curr_oracle)
    
    # Direction agreement: correlation between delta vectors
    if np.std(llm_delta) > 1e-10 and np.std(oracle_delta) > 1e-10:
        correlation, p_value = stats.pearsonr(llm_delta, oracle_delta)
    else:
        correlation = np.nan
        p_value = np.nan
    
    # Per-bucket direction agreement
    agreements = []
    for i in range(len(BUCKET_NAMES)):
        llm_d = llm_delta[i]
        oracle_d = oracle_delta[i]
        # Agreement if both move same direction (or both ~stay same)
        if (llm_d >= -0.001 and oracle_d >= -0.001) or (llm_d <= 0.001 and oracle_d <= 0.001):
            agreements.append(1.0)
        else:
            agreements.append(0.0)
    direction_agreement = np.mean(agreements)
    
    # Magnitude ratio (how much LLM updates vs how much it should)
    if oracle_magnitude > 0.001:
        magnitude_ratio = llm_magnitude / oracle_magnitude
    else:
        magnitude_ratio = np.nan
    
    return {
        "llm_magnitude": llm_magnitude,
        "oracle_magnitude": oracle_magnitude,
        "correlation": correlation,
        "direction_agreement": direction_agreement,
        "magnitude_ratio": magnitude_ratio,
    }


def analyze_updates(hands: dict[str, list[dict]]) -> pd.DataFrame:
    """Analyze all belief updates across hands."""
    
    rows = []
    
    for hand_id, records in hands.items():
        if len(records) < 2:
            continue
        
        for i in range(1, len(records)):
            prev_rec = records[i-1]
            curr_rec = records[i]
            
            # Normalize beliefs
            prev_llm = normalize_belief(belief_to_array(prev_rec["agent_belief"]))
            curr_llm = normalize_belief(belief_to_array(curr_rec["agent_belief"]))
            prev_oracle = normalize_belief(belief_to_array(prev_rec["oracle_strategy_aware"]))
            curr_oracle = normalize_belief(belief_to_array(curr_rec["oracle_strategy_aware"]))
            
            # Skip if any belief is degenerate
            if prev_llm is None or curr_llm is None or prev_oracle is None or curr_oracle is None:
                continue
            
            # Determine update type
            update_type = get_update_type(prev_rec, curr_rec)
            
            # Compute update metrics
            metrics = compute_update_metrics(prev_llm, curr_llm, prev_oracle, curr_oracle)
            
            rows.append({
                "hand_id": hand_id,
                "source": curr_rec.get("_source", ""),
                "prev_decision_idx": prev_rec.get("decision_idx"),
                "curr_decision_idx": curr_rec.get("decision_idx"),
                "prev_street": prev_rec.get("street", prev_rec.get("obs", {}).get("street")),
                "curr_street": curr_rec.get("street", curr_rec.get("obs", {}).get("street")),
                "update_type": update_type,
                **metrics,
            })
    
    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> dict:
    """Compute summary statistics by update type."""
    
    summary = {
        "total_updates": len(df),
        "by_type": {},
    }
    
    for update_type in ["CARD_REVEAL", "ACTION", "UNKNOWN"]:
        subset = df[df["update_type"] == update_type]
        if len(subset) == 0:
            continue
        
        summary["by_type"][update_type] = {
            "n": len(subset),
            "llm_magnitude_mean": subset["llm_magnitude"].mean(),
            "llm_magnitude_std": subset["llm_magnitude"].std(),
            "oracle_magnitude_mean": subset["oracle_magnitude"].mean(),
            "oracle_magnitude_std": subset["oracle_magnitude"].std(),
            "correlation_mean": subset["correlation"].dropna().mean(),
            "direction_agreement_mean": subset["direction_agreement"].mean(),
            "magnitude_ratio_mean": subset["magnitude_ratio"].dropna().mean(),
        }
    
    return summary


def print_report(df: pd.DataFrame, summary: dict) -> None:
    """Print comprehensive update coherence report."""
    
    print("=" * 70)
    print("UPDATE COHERENCE ANALYSIS")
    print("=" * 70)
    print(f"\nTotal belief updates analyzed: {summary['total_updates']}")
    
    print("\n" + "=" * 70)
    print("UPDATE COHERENCE BY TYPE")
    print("=" * 70)
    print("""
This separates two distinct update triggers:
- CARD_REVEAL: New board cards dealt (flop/turn/river transition)
- ACTION: Opponent took an action (same street)

Key metrics:
- Magnitude: How much the belief changed (L2 distance)
- Correlation: Do LLM and oracle updates correlate? (direction agreement)
- Magnitude Ratio: LLM update size / Oracle update size (responsiveness)
""")
    
    print("| Update Type | N | LLM Mag | Oracle Mag | Correlation | Dir Agree | Mag Ratio |")
    print("|-------------|---|---------|------------|-------------|-----------|-----------|")
    
    for update_type in ["CARD_REVEAL", "ACTION", "UNKNOWN"]:
        if update_type not in summary["by_type"]:
            continue
        s = summary["by_type"][update_type]
        print(f"| {update_type:11s} | {s['n']:>1} | {s['llm_magnitude_mean']:.4f} | {s['oracle_magnitude_mean']:.4f} | {s['correlation_mean']:.3f} | {s['direction_agreement_mean']:.3f} | {s['magnitude_ratio_mean']:.2f}x |")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    card_data = summary["by_type"].get("CARD_REVEAL", {})
    action_data = summary["by_type"].get("ACTION", {})
    
    if card_data and action_data:
        card_ratio = card_data.get("magnitude_ratio_mean", 0)
        action_ratio = action_data.get("magnitude_ratio_mean", 0)
        card_corr = card_data.get("correlation_mean", 0)
        action_corr = action_data.get("correlation_mean", 0)
        
        print(f"""
Card reveal responsiveness:  {card_ratio:.2f}x (correlation: {card_corr:.3f})
Action responsiveness:       {action_ratio:.2f}x (correlation: {action_corr:.3f})

Diagnosis:
""")
        
        if card_ratio > 0.5 and action_ratio > 0.5:
            print("- LLM updates on BOTH cards and actions (partial Bayesian)")
        elif card_ratio > 0.5 and action_ratio < 0.3:
            print("- LLM updates on CARDS but not ACTIONS (has priors, ignores behavioral signal)")
        elif card_ratio < 0.3 and action_ratio > 0.5:
            print("- LLM updates on ACTIONS but not CARDS (action-reactive, no combinatorial reasoning)")
        elif card_ratio < 0.3 and action_ratio < 0.3:
            print("- LLM updates are STATIC (largely ignores both types of information)")
        else:
            print("- LLM shows WEAK updates on both types")
        
        if card_corr > 0.5 and action_corr > 0.5:
            print("- Direction of updates is largely CORRECT for both types")
        elif card_corr < 0.3 or action_corr < 0.3:
            print("- WARNING: Update direction is often WRONG (low correlation)")


def main():
    parser = argparse.ArgumentParser(description="Compute update coherence metrics")
    parser.add_argument("files", nargs="+", type=Path, help="Enriched JSONL files")
    parser.add_argument("--output", type=Path, default=Path("results/update_coherence.csv"),
                        help="Output CSV path")
    parser.add_argument("--output-summary", type=Path, default=None,
                        help="Optional JSON summary output")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading and grouping records from {len(args.files)} files...")
    hands = load_and_group_by_hand(args.files)
    print(f"Found {len(hands)} hands with multiple LLM decisions")
    
    print("Analyzing belief updates...")
    df = analyze_updates(hands)
    print(f"Analyzed {len(df)} belief updates")
    
    # Save CSV
    df.to_csv(args.output, index=False)
    print(f"Saved per-update CSV to {args.output}")
    
    # Compute and print summary
    summary = compute_summary(df)
    print_report(df, summary)
    
    # Optionally save summary
    if args.output_summary:
        with open(args.output_summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nJSON summary saved to {args.output_summary}")


if __name__ == "__main__":
    main()
