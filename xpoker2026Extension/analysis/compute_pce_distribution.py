#!/usr/bin/env python3
"""
Compute PCE (Posterior Calibration Error) distribution for paper figures.

Outputs:
1. Per-record CSV with JS divergence sliced by street, opponent action, pot bucket
2. Aggregated summary CSV with bootstrap confidence intervals

Usage:
    python -m analysis.compute_pce_distribution logs/*enriched.jsonl \
        --output-records results/pce_distribution.csv \
        --output-summary results/pce_summary.csv \
        --bootstrap 2000
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from scipy.spatial.distance import jensenshannon

from analysis.buckets import BUCKET_NAMES


def get_opponent_last_action(history: list[dict], player_to_act: int) -> str:
    """
    Determine opponent's last action category.
    
    Returns:
        'AGGRESSIVE' for BET/RAISE
        'PASSIVE' for CALL/CHECK
        'NONE' for no prior opponent action
    """
    opponent = 1 - player_to_act
    
    # Walk backward through history to find opponent's last action
    for event in reversed(history):
        if event.get("player") == opponent:
            event_type = event.get("event", "")
            if event_type in ("BET", "RAISE"):
                return "AGGRESSIVE"
            elif event_type in ("CALL", "CHECK"):
                return "PASSIVE"
    
    return "NONE"


def get_pot_bucket(pot_total: int) -> str:
    """Categorize pot size into buckets."""
    if pot_total <= 10:
        return "small"
    elif pot_total <= 30:
        return "medium"
    elif pot_total <= 60:
        return "large"
    else:
        return "xlarge"


def belief_to_array(belief: dict, bucket_names: list[str] = BUCKET_NAMES) -> np.ndarray:
    """Convert belief dict to numpy array in consistent order."""
    return np.array([belief.get(b) or 0.0 for b in bucket_names], dtype=np.float64)


def normalize_belief(arr: np.ndarray) -> np.ndarray:
    """Clip negatives and L1-normalize."""
    arr = np.maximum(arr, 0)
    total = arr.sum()
    if total > 0:
        return arr / total
    return arr


def compute_js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute JS distance (sqrt of divergence) using scipy."""
    # Ensure valid distributions
    p = normalize_belief(p)
    q = normalize_belief(q)
    
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    
    return float(jensenshannon(p, q))


def load_records_with_beliefs(filepaths: list[Path]) -> list[dict]:
    """Load all decision records that have LLM beliefs."""
    records = []
    
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                record = json.loads(line)
                
                # Skip non-decision records
                if record.get("type") in ("run_config", "hand_summary"):
                    continue
                
                # Skip records without beliefs (non-LLM agents)
                # Don't hardcode player index; check for belief presence instead
                
                # Must have a parsed belief
                if not record.get("agent_belief"):
                    continue
                
                # Must have oracles
                if not record.get("oracle_card_only") or not record.get("oracle_strategy_aware"):
                    continue
                
                record["_source_file"] = filepath.name
                records.append(record)
    
    return records


def process_records(records: list[dict]) -> pd.DataFrame:
    """Process records into a DataFrame with all metrics."""
    rows = []
    
    for record in records:
        # Extract belief arrays
        llm_belief = belief_to_array(record["agent_belief"])
        card_only = belief_to_array(record["oracle_card_only"])
        strategy_aware = belief_to_array(record["oracle_strategy_aware"])
        
        # Skip all-zero beliefs
        if llm_belief.sum() == 0:
            continue
        
        # Normalize for JS computation
        llm_norm = normalize_belief(llm_belief)
        co_norm = normalize_belief(card_only)
        sa_norm = normalize_belief(strategy_aware)
        
        # Compute JS distances
        js_cardonly = compute_js_distance(llm_norm, co_norm)
        js_strategyaware = compute_js_distance(llm_norm, sa_norm)
        js_oracles = compute_js_distance(co_norm, sa_norm)
        
        if np.isnan(js_cardonly) or np.isnan(js_strategyaware) or np.isnan(js_oracles):
            continue
        
        # Get opponent's last action
        history = record.get("obs", {}).get("history", [])
        player_to_act = record.get("player_to_act", 0)
        opp_action = get_opponent_last_action(history, player_to_act)
        
        # Get pot bucket
        pot_total = record.get("obs", {}).get("pot_total", 0)
        pot_bucket = get_pot_bucket(pot_total)
        
        # Trash mass comparison
        trash_idx = BUCKET_NAMES.index("trash")
        llm_trash = llm_norm[trash_idx]
        oracle_trash = sa_norm[trash_idx]
        
        # Strong mass (premium_pairs + strong_pairs + premium_broadway + strong_broadway)
        strong_indices = [
            BUCKET_NAMES.index("premium_pairs"),
            BUCKET_NAMES.index("strong_pairs"),
            BUCKET_NAMES.index("premium_broadway"),
            BUCKET_NAMES.index("strong_broadway"),
        ]
        llm_strong = sum(llm_norm[i] for i in strong_indices)
        oracle_strong = sum(sa_norm[i] for i in strong_indices)
        
        # Raw belief sum (for validity analysis)
        belief_sum_raw = (record.get("belief_metadata") or {}).get("prob_sum")
        if belief_sum_raw is None:
            belief_sum_raw = llm_belief.sum()
        
        rows.append({
            "file": record.get("_source_file", ""),
            "hand_id": record.get("hand_id", ""),
            "decision_idx": record.get("decision_idx", 0),
            "street": record.get("street", record.get("obs", {}).get("street", "UNKNOWN")),
            "opp_action": opp_action,
            "pot_bucket": pot_bucket,
            "pot_total": pot_total,
            "js_cardonly": js_cardonly,
            "js_strategyaware": js_strategyaware,
            "js_oracles": js_oracles,
            "llm_trash": llm_trash,
            "oracle_trash": oracle_trash,
            "llm_strong": llm_strong,
            "oracle_strong": oracle_strong,
            "belief_sum_raw": belief_sum_raw,
        })
    
    return pd.DataFrame(rows)


def bootstrap_ci_clustered(df: pd.DataFrame, col: str, n_bootstrap: int = 2000, ci: float = 0.95) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval with clustering by hand_id.
    
    Resamples hands (not individual decisions) to account for within-hand dependence.
    """
    if len(df) == 0:
        return np.nan, np.nan, np.nan
    
    data = df[col].values
    mean = np.mean(data)
    
    # Get unique hands
    hand_ids = df["hand_id"].unique()
    n_hands = len(hand_ids)
    
    if n_hands < 2:
        return mean, np.nan, np.nan
    
    # Create mapping from hand_id to indices
    hand_to_indices = df.groupby("hand_id").indices
    
    # Bootstrap resampling at hand level
    boot_means = []
    for _ in range(n_bootstrap):
        # Sample hands with replacement
        sampled_hands = np.random.choice(hand_ids, size=n_hands, replace=True)
        
        # Gather all indices from sampled hands
        sampled_indices = []
        for h in sampled_hands:
            sampled_indices.extend(hand_to_indices[h])
        
        # Compute mean on the resampled data
        boot_means.append(np.mean(data[sampled_indices]))
    
    alpha = (1 - ci) / 2
    ci_lo = np.percentile(boot_means, alpha * 100)
    ci_hi = np.percentile(boot_means, (1 - alpha) * 100)
    
    return mean, ci_lo, ci_hi


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 2000, ci: float = 0.95) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval (unclustered, for backward compatibility)."""
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    mean = np.mean(data)
    
    if len(data) < 2:
        return mean, np.nan, np.nan
    
    # Bootstrap resampling
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    
    alpha = (1 - ci) / 2
    ci_lo = np.percentile(boot_means, alpha * 100)
    ci_hi = np.percentile(boot_means, (1 - alpha) * 100)
    
    return mean, ci_lo, ci_hi


def compute_summary(df: pd.DataFrame, n_bootstrap: int = 2000, clustered: bool = True) -> pd.DataFrame:
    """Compute aggregated summary with bootstrap CIs.
    
    Args:
        df: DataFrame with records
        n_bootstrap: Number of bootstrap samples
        clustered: If True, use hand-level clustering for CIs
    """
    
    summaries = []
    
    # Overall summary
    if clustered:
        mean_co, ci_lo_co, ci_hi_co = bootstrap_ci_clustered(df, "js_cardonly", n_bootstrap)
        mean_sa, ci_lo_sa, ci_hi_sa = bootstrap_ci_clustered(df, "js_strategyaware", n_bootstrap)
    else:
        mean_co, ci_lo_co, ci_hi_co = bootstrap_ci(df["js_cardonly"].values, n_bootstrap)
        mean_sa, ci_lo_sa, ci_hi_sa = bootstrap_ci(df["js_strategyaware"].values, n_bootstrap)
    
    summaries.append({
        "group": "OVERALL",
        "street": "ALL",
        "opp_action": "ALL",
        "n": len(df),
        "js_cardonly_mean": mean_co,
        "js_cardonly_ci_lo": ci_lo_co,
        "js_cardonly_ci_hi": ci_hi_co,
        "js_strategyaware_mean": mean_sa,
        "js_strategyaware_ci_lo": ci_lo_sa,
        "js_strategyaware_ci_hi": ci_hi_sa,
        "closer_to": "CardOnly" if mean_co < mean_sa else "StrategyAware",
        "js_difference": mean_co - mean_sa,
        "llm_trash_mean": df["llm_trash"].mean(),
        "oracle_trash_mean": df["oracle_trash"].mean(),
    })
    
    # By street
    for street in ["PREFLOP", "FLOP", "TURN", "RIVER"]:
        subset = df[df["street"] == street]
        if len(subset) == 0:
            continue
        
        if clustered:
            mean_co, ci_lo_co, ci_hi_co = bootstrap_ci_clustered(subset, "js_cardonly", n_bootstrap)
            mean_sa, ci_lo_sa, ci_hi_sa = bootstrap_ci_clustered(subset, "js_strategyaware", n_bootstrap)
        else:
            mean_co, ci_lo_co, ci_hi_co = bootstrap_ci(subset["js_cardonly"].values, n_bootstrap)
            mean_sa, ci_lo_sa, ci_hi_sa = bootstrap_ci(subset["js_strategyaware"].values, n_bootstrap)
        
        summaries.append({
            "group": f"street={street}",
            "street": street,
            "opp_action": "ALL",
            "n": len(subset),
            "js_cardonly_mean": mean_co,
            "js_cardonly_ci_lo": ci_lo_co,
            "js_cardonly_ci_hi": ci_hi_co,
            "js_strategyaware_mean": mean_sa,
            "js_strategyaware_ci_lo": ci_lo_sa,
            "js_strategyaware_ci_hi": ci_hi_sa,
            "closer_to": "CardOnly" if mean_co < mean_sa else "StrategyAware",
            "js_difference": mean_co - mean_sa,
            "llm_trash_mean": subset["llm_trash"].mean(),
            "oracle_trash_mean": subset["oracle_trash"].mean(),
        })
    
    # By opponent action
    for opp_action in ["AGGRESSIVE", "PASSIVE", "NONE"]:
        subset = df[df["opp_action"] == opp_action]
        if len(subset) == 0:
            continue
        
        if clustered:
            mean_co, ci_lo_co, ci_hi_co = bootstrap_ci_clustered(subset, "js_cardonly", n_bootstrap)
            mean_sa, ci_lo_sa, ci_hi_sa = bootstrap_ci_clustered(subset, "js_strategyaware", n_bootstrap)
        else:
            mean_co, ci_lo_co, ci_hi_co = bootstrap_ci(subset["js_cardonly"].values, n_bootstrap)
            mean_sa, ci_lo_sa, ci_hi_sa = bootstrap_ci(subset["js_strategyaware"].values, n_bootstrap)
        
        summaries.append({
            "group": f"opp_action={opp_action}",
            "street": "ALL",
            "opp_action": opp_action,
            "n": len(subset),
            "js_cardonly_mean": mean_co,
            "js_cardonly_ci_lo": ci_lo_co,
            "js_cardonly_ci_hi": ci_hi_co,
            "js_strategyaware_mean": mean_sa,
            "js_strategyaware_ci_lo": ci_lo_sa,
            "js_strategyaware_ci_hi": ci_hi_sa,
            "closer_to": "CardOnly" if mean_co < mean_sa else "StrategyAware",
            "js_difference": mean_co - mean_sa,
            "llm_trash_mean": subset["llm_trash"].mean(),
            "oracle_trash_mean": subset["oracle_trash"].mean(),
        })
    
    # By street × opponent action
    for street in ["PREFLOP", "FLOP", "TURN", "RIVER"]:
        for opp_action in ["AGGRESSIVE", "PASSIVE"]:
            subset = df[(df["street"] == street) & (df["opp_action"] == opp_action)]
            if len(subset) < 5:  # Skip tiny groups
                continue
            
            if clustered:
                mean_co, ci_lo_co, ci_hi_co = bootstrap_ci_clustered(subset, "js_cardonly", n_bootstrap)
                mean_sa, ci_lo_sa, ci_hi_sa = bootstrap_ci_clustered(subset, "js_strategyaware", n_bootstrap)
            else:
                mean_co, ci_lo_co, ci_hi_co = bootstrap_ci(subset["js_cardonly"].values, n_bootstrap)
                mean_sa, ci_lo_sa, ci_hi_sa = bootstrap_ci(subset["js_strategyaware"].values, n_bootstrap)
            
            summaries.append({
                "group": f"{street}×{opp_action}",
                "street": street,
                "opp_action": opp_action,
                "n": len(subset),
                "js_cardonly_mean": mean_co,
                "js_cardonly_ci_lo": ci_lo_co,
                "js_cardonly_ci_hi": ci_hi_co,
                "js_strategyaware_mean": mean_sa,
                "js_strategyaware_ci_lo": ci_lo_sa,
                "js_strategyaware_ci_hi": ci_hi_sa,
                "closer_to": "CardOnly" if mean_co < mean_sa else "StrategyAware",
                "js_difference": mean_co - mean_sa,
                "llm_trash_mean": subset["llm_trash"].mean(),
                "oracle_trash_mean": subset["oracle_trash"].mean(),
            })
    
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description="Compute PCE distribution for paper figures")
    parser.add_argument("files", nargs="+", type=Path, help="Enriched JSONL files")
    parser.add_argument("--output-records", type=Path, default=Path("results/pce_distribution.csv"),
                        help="Per-record CSV output path")
    parser.add_argument("--output-summary", type=Path, default=Path("results/pce_summary.csv"),
                        help="Aggregated summary CSV output path")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Number of bootstrap samples for CIs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for bootstrap")
    parser.add_argument("--clustered", action="store_true", default=True,
                        help="Use hand-level clustered bootstrap (default: True)")
    parser.add_argument("--no-clustered", dest="clustered", action="store_false",
                        help="Use unclustered bootstrap")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directories
    args.output_records.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading records from {len(args.files)} files...")
    records = load_records_with_beliefs(args.files)
    print(f"Loaded {len(records)} records with beliefs")
    
    print("Processing records...")
    df = process_records(records)
    print(f"Processed {len(df)} valid records")
    
    # Save per-record CSV
    df.to_csv(args.output_records, index=False)
    print(f"Saved per-record CSV to {args.output_records}")
    
    # Compute and save summary
    cluster_str = "clustered" if args.clustered else "unclustered"
    print(f"Computing summary with {args.bootstrap} {cluster_str} bootstrap samples...")
    summary = compute_summary(df, args.bootstrap, clustered=args.clustered)
    summary.to_csv(args.output_summary, index=False)
    print(f"Saved summary CSV to {args.output_summary}")
    
    # Print headline results
    print("\n" + "=" * 70)
    print("HEADLINE RESULTS")
    print("=" * 70)
    
    overall = summary[summary["group"] == "OVERALL"].iloc[0]
    print(f"\nOverall (N={overall['n']}):")
    print(f"  JS(LLM, CardOnly):       {overall['js_cardonly_mean']:.4f} [{overall['js_cardonly_ci_lo']:.4f}, {overall['js_cardonly_ci_hi']:.4f}]")
    print(f"  JS(LLM, StrategyAware):  {overall['js_strategyaware_mean']:.4f} [{overall['js_strategyaware_ci_lo']:.4f}, {overall['js_strategyaware_ci_hi']:.4f}]")
    print(f"  LLM closer to:           {overall['closer_to']} (by {abs(overall['js_difference']):.4f})")
    print(f"  LLM trash mass:          {overall['llm_trash_mean']:.2%}")
    print(f"  Oracle trash mass:       {overall['oracle_trash_mean']:.2%}")
    
    print("\nBy Street:")
    for _, row in summary[summary["opp_action"] == "ALL"].iterrows():
        if row["street"] == "ALL":
            continue
        print(f"  {row['street']:8s} (N={row['n']:3d}): JS_CO={row['js_cardonly_mean']:.3f}, JS_SA={row['js_strategyaware_mean']:.3f}, closer={row['closer_to']}")
    
    print("\nBy Opponent Action:")
    for _, row in summary[summary["street"] == "ALL"].iterrows():
        if row["opp_action"] == "ALL":
            continue
        print(f"  {row['opp_action']:10s} (N={row['n']:3d}): JS_CO={row['js_cardonly_mean']:.3f}, JS_SA={row['js_strategyaware_mean']:.3f}, closer={row['closer_to']}")


if __name__ == "__main__":
    main()
