"""
Comprehensive belief analysis script.

Analyzes enriched experiment logs to compute:
1. Belief validity metrics (raw outputs)
2. JS divergence to oracles (normalized)
3. Action-conditioning analysis (does LLM respond to opponent behavior?)
4. Summary table for paper

Uses the existing analysis infrastructure:
- belief_utils.py for normalization
- metrics/calibration.py for divergences
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional

from scipy.spatial.distance import jensenshannon

from analysis.buckets import BUCKET_NAMES
from analysis.belief_utils import dict_to_compact, validate_belief
from analysis.build_dataset import load_analysis_dataset


# ============================================================================
# Constants
# ============================================================================

# Bucket indices for action-conditioning analysis
STRONG_BUCKETS = ["premium_pairs", "strong_pairs", "premium_broadway", "strong_broadway"]
STRONG_INDICES = [BUCKET_NAMES.index(b) for b in STRONG_BUCKETS]
TRASH_INDEX = BUCKET_NAMES.index("trash")


# ============================================================================
# Data Loading
# ============================================================================

def load_records_from_files(filepaths: list[str]) -> list[dict]:
    """Load decision records from multiple enriched JSONL files."""
    all_records = []
    for path in filepaths:
        records = load_analysis_dataset(path)
        all_records.extend(records)
    return all_records


def filter_records_with_beliefs(records: list[dict]) -> list[dict]:
    """Filter to records that have agent beliefs and oracle posteriors."""
    valid = []
    for r in records:
        if (r.get("agent_belief") and 
            r.get("oracle_card_only") and 
            r.get("oracle_strategy_aware")):
            valid.append(r)
    return valid


# ============================================================================
# Belief Validity Audit (Raw Outputs)
# ============================================================================

def audit_belief_validity(records: list[dict]) -> dict:
    """
    Audit raw belief outputs for validity issues.
    
    This measures OUTPUT VALIDITY - whether LLM follows probability constraints.
    Computed on RAW outputs, not normalized.
    """
    stats = {
        "total_records": len(records),
        "records_with_beliefs": 0,
        "negative_entries": 0,
        "records_with_negatives": 0,
        "all_zero_records": 0,
        "valid_for_js": 0,
        "prob_sums": [],
        "min_value_seen": float('inf'),
        "max_value_seen": float('-inf'),
    }
    
    for r in records:
        belief = r.get("agent_belief")
        if not belief:
            continue
        
        stats["records_with_beliefs"] += 1
        
        # Extract probabilities in bucket order
        probs = [belief.get(b, 0.0) for b in BUCKET_NAMES]
        
        # Check for negatives
        neg_count = sum(1 for p in probs if p < 0)
        if neg_count > 0:
            stats["negative_entries"] += neg_count
            stats["records_with_negatives"] += 1
        
        # Track min/max
        stats["min_value_seen"] = min(stats["min_value_seen"], min(probs))
        stats["max_value_seen"] = max(stats["max_value_seen"], max(probs))
        
        # Track sum
        prob_sum = sum(probs)
        stats["prob_sums"].append(prob_sum)
        
        # Check for all zeros
        if prob_sum == 0:
            stats["all_zero_records"] += 1
        else:
            stats["valid_for_js"] += 1
    
    # Compute summary stats
    if stats["prob_sums"]:
        stats["prob_sum_mean"] = np.mean(stats["prob_sums"])
        stats["prob_sum_std"] = np.std(stats["prob_sums"])
        stats["prob_sum_min"] = min(stats["prob_sums"])
        stats["prob_sum_max"] = max(stats["prob_sums"])
    
    return stats


# ============================================================================
# JS Divergence Analysis (Normalized)
# ============================================================================

def normalize_belief(belief: dict[str, float]) -> Optional[np.ndarray]:
    """
    Normalize belief to valid distribution for JS computation.
    
    Process:
    1. Clip negatives to 0
    2. L1-normalize to sum to 1
    3. Return None if all zeros
    """
    probs = np.array([belief.get(b, 0.0) for b in BUCKET_NAMES])
    
    # Clip negatives
    probs = np.maximum(probs, 0)
    
    # Check for all zeros
    total = probs.sum()
    if total == 0:
        return None
    
    # Normalize
    return probs / total


def compute_js_metrics(records: list[dict]) -> dict:
    """
    Compute JS divergence metrics on normalized beliefs.
    
    This measures DISTRIBUTIONAL ACCURACY - how well LLM beliefs
    match oracle posteriors in shape (ignoring scale).
    """
    js_llm_co = []  # LLM vs CardOnly
    js_llm_sa = []  # LLM vs StrategyAware  
    js_co_sa = []   # CardOnly vs StrategyAware (oracle separation)
    
    valid_count = 0
    skipped_count = 0
    
    for r in records:
        # Normalize all distributions
        llm = normalize_belief(r["agent_belief"])
        co = normalize_belief(r["oracle_card_only"])
        sa = normalize_belief(r["oracle_strategy_aware"])
        
        # Skip if LLM belief is degenerate
        if llm is None:
            skipped_count += 1
            continue
        
        # Skip if oracles are degenerate (shouldn't happen)
        if co is None or sa is None:
            skipped_count += 1
            continue
        
        valid_count += 1
        
        # Compute JS distances using scipy (returns sqrt of divergence)
        # This is the standard "JS distance" used in ML literature
        js_llm_co.append(jensenshannon(llm, co))
        js_llm_sa.append(jensenshannon(llm, sa))
        js_co_sa.append(jensenshannon(co, sa))
    
    return {
        "valid_count": valid_count,
        "skipped_count": skipped_count,
        "js_llm_cardonly_mean": np.mean(js_llm_co) if js_llm_co else None,
        "js_llm_cardonly_std": np.std(js_llm_co) if js_llm_co else None,
        "js_llm_strataware_mean": np.mean(js_llm_sa) if js_llm_sa else None,
        "js_llm_strataware_std": np.std(js_llm_sa) if js_llm_sa else None,
        "js_cardonly_strataware_mean": np.mean(js_co_sa) if js_co_sa else None,
        "js_cardonly_strataware_std": np.std(js_co_sa) if js_co_sa else None,
        # Key comparison
        "llm_closer_to": "CardOnly" if np.mean(js_llm_co) < np.mean(js_llm_sa) else "StrategyAware",
        "js_difference": np.mean(js_llm_co) - np.mean(js_llm_sa) if js_llm_co else None,
    }


# ============================================================================
# Action-Conditioning Analysis
# ============================================================================

def get_opponent_action_category(history: list[dict]) -> str:
    """
    Categorize opponent's most recent action.
    
    Returns: "AGGRESSIVE" (bet/raise), "PASSIVE" (call/check), or "NO_ACTION"
    """
    # Filter to opponent actions (player 1)
    opp_actions = [
        h for h in history 
        if h.get("player") == 1 and h.get("event") in ["BET", "RAISE", "CHECK", "CALL", "FOLD"]
    ]
    
    if not opp_actions:
        return "NO_ACTION"
    
    last_action = opp_actions[-1]["event"]
    
    if last_action in ["BET", "RAISE"]:
        return "AGGRESSIVE"
    elif last_action in ["CHECK", "CALL"]:
        return "PASSIVE"
    else:
        return "OTHER"


def compute_action_conditioning(records: list[dict]) -> dict:
    """
    Analyze whether LLM shifts beliefs after opponent aggression.
    
    Key test: Does LLM put more mass on strong hands after opponent raises?
    Expected under StrategyAware: yes
    If LLM ignores history: no shift
    """
    # Group records by opponent action category
    by_category = defaultdict(list)
    
    for r in records:
        # Normalize beliefs
        llm = normalize_belief(r["agent_belief"])
        sa = normalize_belief(r["oracle_strategy_aware"])
        
        if llm is None or sa is None:
            continue
        
        # Get action category
        history = r.get("obs", {}).get("history", [])
        category = get_opponent_action_category(history)
        
        by_category[category].append({
            "llm": llm,
            "sa": sa,
            "llm_strong": llm[STRONG_INDICES].sum(),
            "sa_strong": sa[STRONG_INDICES].sum(),
            "llm_trash": llm[TRASH_INDEX],
            "sa_trash": sa[TRASH_INDEX],
        })
    
    # Compute per-category stats
    results = {}
    for cat in ["AGGRESSIVE", "PASSIVE", "NO_ACTION"]:
        if cat not in by_category:
            continue
        
        recs = by_category[cat]
        results[cat] = {
            "n": len(recs),
            "llm_strong_mean": np.mean([r["llm_strong"] for r in recs]),
            "sa_strong_mean": np.mean([r["sa_strong"] for r in recs]),
            "llm_trash_mean": np.mean([r["llm_trash"] for r in recs]),
            "sa_trash_mean": np.mean([r["sa_trash"] for r in recs]),
        }
    
    # Compute shifts (AGGRESSIVE vs PASSIVE)
    shift_analysis = {}
    if "AGGRESSIVE" in results and "PASSIVE" in results:
        agg = results["AGGRESSIVE"]
        pas = results["PASSIVE"]
        
        shift_analysis = {
            "oracle_strong_shift": agg["sa_strong_mean"] - pas["sa_strong_mean"],
            "llm_strong_shift": agg["llm_strong_mean"] - pas["llm_strong_mean"],
            "oracle_trash_shift": agg["sa_trash_mean"] - pas["sa_trash_mean"],
            "llm_trash_shift": agg["llm_trash_mean"] - pas["llm_trash_mean"],
        }
        
        # Compute ratio (LLM responsiveness relative to oracle)
        if abs(shift_analysis["oracle_strong_shift"]) > 0.001:
            shift_analysis["strong_shift_ratio"] = (
                shift_analysis["llm_strong_shift"] / shift_analysis["oracle_strong_shift"]
            )
        
        if abs(shift_analysis["oracle_trash_shift"]) > 0.001:
            shift_analysis["trash_shift_ratio"] = (
                shift_analysis["llm_trash_shift"] / shift_analysis["oracle_trash_shift"]
            )
    
    return {
        "by_category": results,
        "shift_analysis": shift_analysis,
    }


# ============================================================================
# Summary Report
# ============================================================================

def print_report(
    validity: dict,
    js_metrics: dict,
    action_cond: dict,
    files: list[str],
) -> None:
    """Print comprehensive analysis report."""
    
    print("=" * 70)
    print("BELIEF ANALYSIS REPORT")
    print("=" * 70)
    print(f"\nFiles analyzed: {len(files)}")
    for f in files:
        print(f"  - {f}")
    
    # Part 1: Validity Audit
    print("\n" + "=" * 70)
    print("PART 1: BELIEF VALIDITY AUDIT (Raw Outputs)")
    print("=" * 70)
    print(f"""
This measures OUTPUT VALIDITY - whether LLM follows probability constraints.
Computed on RAW model outputs (not normalized).

| Metric | Value |
|--------|-------|
| Total records | {validity['total_records']} |
| Records with beliefs | {validity['records_with_beliefs']} |
| Records with negatives | {validity['records_with_negatives']} |
| All-zero records | {validity['all_zero_records']} |
| Valid for JS analysis | {validity['valid_for_js']} |
| Prob sum (mean) | {validity.get('prob_sum_mean', 'N/A'):.4f} |
| Prob sum (min/max) | {validity.get('prob_sum_min', 'N/A'):.4f} / {validity.get('prob_sum_max', 'N/A'):.4f} |
| Min value seen | {validity['min_value_seen']:.6f} |
""")
    
    # Part 2: JS Divergence
    print("=" * 70)
    print("PART 2: JS DISTANCE ANALYSIS (Normalized)")
    print("=" * 70)
    print(f"""
This measures DISTRIBUTIONAL ACCURACY - how well LLM beliefs match oracles.
Computed on L1-normalized distributions (negatives clipped, then normalized).
All-zero records excluded (N={js_metrics['skipped_count']} dropped).

Note: Using JS DISTANCE (sqrt of JS divergence), range [0, 1], via scipy.jensenshannon.

| Comparison | Mean JS Dist | Std |
|------------|--------------|-----|
| JS(LLM, CardOnly) | {js_metrics['js_llm_cardonly_mean']:.4f} | {js_metrics['js_llm_cardonly_std']:.4f} |
| JS(LLM, StrategyAware) | {js_metrics['js_llm_strataware_mean']:.4f} | {js_metrics['js_llm_strataware_std']:.4f} |
| JS(CardOnly, StrategyAware) | {js_metrics['js_cardonly_strataware_mean']:.4f} | {js_metrics['js_cardonly_strataware_std']:.4f} |

**LLM is closer to: {js_metrics['llm_closer_to']}** (by {abs(js_metrics['js_difference']):.4f})

Note: CardOnly = BucketCountPrior (pure combinatorics). Being closer to CardOnly
means the LLM's belief *shape* resembles combo-counting more than Bayesian updating.
""")
    
    # Part 3: Action-Conditioning
    print("=" * 70)
    print("PART 3: ACTION-CONDITIONING ANALYSIS")
    print("=" * 70)
    print("""
This tests: Does LLM shift beliefs after opponent aggression?
Expected under StrategyAware: More mass on strong hands after RAISE/BET.
""")
    
    if action_cond["by_category"]:
        print("| After Opponent | N | LLM Strong | Oracle Strong | LLM Trash | Oracle Trash |")
        print("|----------------|---|------------|---------------|-----------|--------------|")
        for cat in ["AGGRESSIVE", "PASSIVE", "NO_ACTION"]:
            if cat in action_cond["by_category"]:
                c = action_cond["by_category"][cat]
                print(f"| {cat:<14} | {c['n']:>1} | {c['llm_strong_mean']:.4f} | {c['sa_strong_mean']:.4f} | {c['llm_trash_mean']:.4f} | {c['sa_trash_mean']:.4f} |")
    
    if action_cond["shift_analysis"]:
        s = action_cond["shift_analysis"]
        print(f"""
**Shift Analysis (AGGRESSIVE vs PASSIVE):**

| Metric | Oracle Shift | LLM Shift | Ratio |
|--------|--------------|-----------|-------|
| Strong-mass | {s['oracle_strong_shift']:+.4f} | {s['llm_strong_shift']:+.4f} | {s.get('strong_shift_ratio', 0):.2f}x |
| Trash-mass | {s['oracle_trash_shift']:+.4f} | {s['llm_trash_shift']:+.4f} | {s.get('trash_shift_ratio', 0):.2f}x |
""")
        
        # Interpretation
        if s.get('strong_shift_ratio', 0) > 0.5:
            print("Interpretation: LLM shows SOME directional sensitivity to opponent aggression.")
        else:
            print("Interpretation: LLM is largely INSENSITIVE to opponent aggression.")
    
    # Part 4: Summary Table for Paper
    print("\n" + "=" * 70)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 70)
    
    shift = action_cond.get("shift_analysis", {})
    oracle_shift = shift.get("oracle_strong_shift", 0)
    llm_shift = shift.get("llm_strong_shift", 0)
    shift_pct = (llm_shift / oracle_shift * 100) if oracle_shift != 0 else 0
    
    # Get average trash mass
    agg_data = action_cond.get("by_category", {}).get("AGGRESSIVE", {})
    pas_data = action_cond.get("by_category", {}).get("PASSIVE", {})
    avg_llm_trash = np.mean([
        agg_data.get("llm_trash_mean", 0),
        pas_data.get("llm_trash_mean", 0)
    ]) if agg_data and pas_data else 0
    avg_oracle_trash = np.mean([
        agg_data.get("sa_trash_mean", 0),
        pas_data.get("sa_trash_mean", 0)
    ]) if agg_data and pas_data else 0
    
    print(f"""
| Metric | Value | Interpretation |
|--------|-------|----------------|
| N (valid beliefs) | {js_metrics['valid_count']} | - |
| JS(LLM, CardOnly) | {js_metrics['js_llm_cardonly_mean']:.4f} | Distance to combo-counting |
| JS(LLM, StrategyAware) | {js_metrics['js_llm_strataware_mean']:.4f} | Distance to Bayesian posterior |
| JS(CardOnly, StrategyAware) | {js_metrics['js_cardonly_strataware_mean']:.4f} | Oracle separation (test validity) |
| LLM closer to | {js_metrics['llm_closer_to']} by {abs(js_metrics['js_difference']):.4f} | {js_metrics['llm_closer_to']} = ignores history |
| Oracle strong-shift (AGG vs PAS) | {oracle_shift:+.4f} | Aggression = stronger hands |
| LLM strong-shift (AGG vs PAS) | {llm_shift:+.4f} | LLM response ({shift_pct:.0f}% of oracle) |
| Avg LLM trash mass | {avg_llm_trash:.3f} | Should be ~0.65 |
| Avg Oracle trash mass | {avg_oracle_trash:.3f} | Baseline |
""")


def save_json_report(
    validity: dict,
    js_metrics: dict,
    action_cond: dict,
    output_path: str,
) -> None:
    """Save analysis results as JSON for programmatic access."""
    # Clean up non-serializable items
    validity_clean = {k: v for k, v in validity.items() if k != "prob_sums"}
    
    report = {
        "validity_audit": validity_clean,
        "js_divergence": js_metrics,
        "action_conditioning": action_cond,
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nJSON report saved to: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze LLM beliefs against oracle posteriors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single enriched file
  python -m analysis.analyze_beliefs logs/experiment_enriched.jsonl

  # Analyze multiple files
  python -m analysis.analyze_beliefs logs/run1_enriched.jsonl logs/run2_enriched.jsonl

  # Save JSON report
  python -m analysis.analyze_beliefs logs/experiment_enriched.jsonl --json-out results.json
"""
    )
    
    parser.add_argument(
        "files",
        nargs="+",
        help="Enriched JSONL files to analyze",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to save JSON report",
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading records...")
    records = load_records_from_files(args.files)
    records = filter_records_with_beliefs(records)
    print(f"Loaded {len(records)} records with beliefs")
    
    # Run analyses
    print("Running validity audit...")
    validity = audit_belief_validity(records)
    
    print("Computing JS divergences...")
    js_metrics = compute_js_metrics(records)
    
    print("Running action-conditioning analysis...")
    action_cond = compute_action_conditioning(records)
    
    # Print report
    print_report(validity, js_metrics, action_cond, args.files)
    
    # Optionally save JSON
    if args.json_out:
        save_json_report(validity, js_metrics, action_cond, args.json_out)


if __name__ == "__main__":
    main()
