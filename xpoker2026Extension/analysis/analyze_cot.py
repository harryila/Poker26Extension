"""
Chain-of-Thought analysis: compare CoT vs direct prompting.

Loads enriched JSONL datasets (one CoT, one direct) and compares:
- Belief calibration (JS distance to oracles)
- Reasoning quality proxies (length, keyword presence)
- Action quality (fallback rate, parse success)
- Correlation between reasoning quality and belief accuracy
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    from scipy.spatial.distance import jensenshannon
except ImportError:
    jensenshannon = None

from analysis.buckets import BUCKET_NAMES
from analysis.build_dataset import load_analysis_dataset


# ============================================================================
# Reasoning quality metrics
# ============================================================================

OPPONENT_ACTION_KEYWORDS = re.compile(
    r'raise|bet|call|check|fold|aggressive|passive|tight|loose|c-bet|continuation',
    re.IGNORECASE,
)

BOARD_TEXTURE_KEYWORDS = re.compile(
    r'flush|straight|draw|connected|suited|paired|dry|wet|texture|rainbow',
    re.IGNORECASE,
)

HAND_COMBO_KEYWORDS = re.compile(
    r'AA|KK|QQ|JJ|AK|AQ|pocket|pair|broadway|suited\s+connector|over-?pair|set|trips',
    re.IGNORECASE,
)


def score_reasoning(text: str | None) -> dict:
    """Score a CoT reasoning trace on quality proxies."""
    if not text:
        return {
            "length": 0,
            "mentions_opponent_actions": False,
            "mentions_board_texture": False,
            "mentions_hand_combos": False,
            "quality_score": 0.0,
        }

    mentions_actions = bool(OPPONENT_ACTION_KEYWORDS.search(text))
    mentions_board = bool(BOARD_TEXTURE_KEYWORDS.search(text))
    mentions_combos = bool(HAND_COMBO_KEYWORDS.search(text))

    quality = sum([mentions_actions, mentions_board, mentions_combos]) / 3.0

    return {
        "length": len(text),
        "mentions_opponent_actions": mentions_actions,
        "mentions_board_texture": mentions_board,
        "mentions_hand_combos": mentions_combos,
        "quality_score": quality,
    }


# ============================================================================
# Belief comparison
# ============================================================================

def normalize_belief(belief: dict) -> np.ndarray | None:
    """Convert belief dict to normalized probability vector."""
    if not belief:
        return None
    vec = np.array([belief.get(b) or 0.0 for b in BUCKET_NAMES], dtype=np.float64)
    vec = np.clip(vec, 0, None)
    total = vec.sum()
    if total <= 0:
        return None
    return vec / total


def js_distance(p: np.ndarray, q: np.ndarray) -> float | None:
    if jensenshannon is None:
        return None
    return float(jensenshannon(p, q))


# ============================================================================
# Main analysis
# ============================================================================

def analyze_cot_dataset(records: list[dict]) -> dict:
    """Analyze a single dataset that may contain CoT reasoning."""
    reasoning_scores = []
    js_to_card_only = []
    js_to_strategy_aware = []
    paired_quality_js = []
    parse_successes = 0
    fallbacks = 0
    total = 0

    for rec in records:
        total += 1
        cot_text = rec.get("cot_reasoning")
        score = score_reasoning(cot_text)
        reasoning_scores.append(score)

        # Action metadata
        ameta = rec.get("action_metadata", {}) or {}
        if ameta.get("parse_success"):
            parse_successes += 1
        if ameta.get("fallback_used"):
            fallbacks += 1

        # Belief comparison
        agent_belief = rec.get("agent_belief")
        oracle_co = rec.get("oracle_card_only")
        oracle_sa = rec.get("oracle_strategy_aware")

        p = normalize_belief(agent_belief)
        js_sa_val = None
        if p is not None and oracle_co:
            q = normalize_belief(oracle_co)
            if q is not None:
                val = js_distance(p, q)
                if val is not None:
                    js_to_card_only.append(val)
        if p is not None and oracle_sa:
            q = normalize_belief(oracle_sa)
            if q is not None:
                js_sa_val = js_distance(p, q)
                if js_sa_val is not None:
                    js_to_strategy_aware.append(js_sa_val)

        if score["length"] > 0 and js_sa_val is not None:
            paired_quality_js.append((score["quality_score"], js_sa_val))

    # Aggregate reasoning scores
    has_cot = [s for s in reasoning_scores if s["length"] > 0]
    avg_length = np.mean([s["length"] for s in has_cot]) if has_cot else 0
    frac_mentions_actions = np.mean([s["mentions_opponent_actions"] for s in has_cot]) if has_cot else 0
    frac_mentions_board = np.mean([s["mentions_board_texture"] for s in has_cot]) if has_cot else 0
    frac_mentions_combos = np.mean([s["mentions_hand_combos"] for s in has_cot]) if has_cot else 0
    avg_quality = np.mean([s["quality_score"] for s in has_cot]) if has_cot else 0

    # Correlation: reasoning quality vs belief JS distance (properly paired)
    quality_js_corr = None
    if len(paired_quality_js) >= 10:
        qualities = [q for q, _ in paired_quality_js]
        dists = [d for _, d in paired_quality_js]
        if np.std(qualities) > 1e-10 and np.std(dists) > 1e-10:
            quality_js_corr = float(np.corrcoef(qualities, dists)[0, 1])

    return {
        "total_decisions": total,
        "decisions_with_cot": len(has_cot),
        "parse_success_rate": parse_successes / total if total else 0,
        "fallback_rate": fallbacks / total if total else 0,
        "reasoning": {
            "avg_length": float(avg_length),
            "frac_mentions_opponent_actions": float(frac_mentions_actions),
            "frac_mentions_board_texture": float(frac_mentions_board),
            "frac_mentions_hand_combos": float(frac_mentions_combos),
            "avg_quality_score": float(avg_quality),
        },
        "belief_js_to_card_only": {
            "mean": float(np.mean(js_to_card_only)) if js_to_card_only else None,
            "std": float(np.std(js_to_card_only)) if js_to_card_only else None,
            "n": len(js_to_card_only),
        },
        "belief_js_to_strategy_aware": {
            "mean": float(np.mean(js_to_strategy_aware)) if js_to_strategy_aware else None,
            "std": float(np.std(js_to_strategy_aware)) if js_to_strategy_aware else None,
            "n": len(js_to_strategy_aware),
        },
        "quality_js_correlation": quality_js_corr,
    }


def compare_cot_vs_direct(cot_results: dict, direct_results: dict) -> dict:
    """Compare CoT vs direct prompting results."""
    comparison = {}

    for metric_key in ("belief_js_to_card_only", "belief_js_to_strategy_aware"):
        cot_mean = cot_results.get(metric_key, {}).get("mean")
        direct_mean = direct_results.get(metric_key, {}).get("mean")
        if cot_mean is not None and direct_mean is not None:
            comparison[f"{metric_key}_delta"] = cot_mean - direct_mean
            comparison[f"{metric_key}_cot_better"] = cot_mean < direct_mean

    cot_parse = cot_results.get("parse_success_rate", 0)
    direct_parse = direct_results.get("parse_success_rate", 0)
    comparison["parse_success_delta"] = cot_parse - direct_parse

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Analyze Chain-of-Thought reasoning quality")
    parser.add_argument("input", nargs="+", help="Enriched JSONL file(s)")
    parser.add_argument("--json-out", type=str, default=None, help="Save JSON report")
    args = parser.parse_args()

    all_results = {}
    for path in args.input:
        name = Path(path).stem
        records = load_analysis_dataset(path)
        result = analyze_cot_dataset(records)
        all_results[name] = result

        print(f"\n=== {name} ===")
        print(f"  Decisions: {result['total_decisions']}")
        print(f"  With CoT:  {result['decisions_with_cot']}")
        print(f"  Parse success: {result['parse_success_rate']:.1%}")
        print(f"  Fallback rate: {result['fallback_rate']:.1%}")

        r = result["reasoning"]
        print(f"  Reasoning avg length: {r['avg_length']:.0f} chars")
        print(f"  Mentions opponent actions: {r['frac_mentions_opponent_actions']:.1%}")
        print(f"  Mentions board texture:   {r['frac_mentions_board_texture']:.1%}")
        print(f"  Mentions hand combos:     {r['frac_mentions_hand_combos']:.1%}")
        print(f"  Quality score: {r['avg_quality_score']:.2f}")

        js_co = result["belief_js_to_card_only"]
        js_sa = result["belief_js_to_strategy_aware"]
        if js_co["mean"] is not None:
            print(f"  JS to CardOnly:       {js_co['mean']:.4f} +/- {js_co['std']:.4f} (n={js_co['n']})")
        if js_sa["mean"] is not None:
            print(f"  JS to StrategyAware:  {js_sa['mean']:.4f} +/- {js_sa['std']:.4f} (n={js_sa['n']})")

        if result["quality_js_correlation"] is not None:
            print(f"  Quality-JS correlation: {result['quality_js_correlation']:.3f}")

    # Compare if two files given — detect CoT by filename or reasoning stats
    if len(all_results) == 2:
        names = list(all_results.keys())
        r0 = all_results[names[0]]
        r1 = all_results[names[1]]
        name0_is_cot = "cot" in names[0].lower() or r0.get("reasoning", {}).get("avg_quality_score", 0) > r1.get("reasoning", {}).get("avg_quality_score", 0)
        if name0_is_cot:
            comparison = compare_cot_vs_direct(r0, r1)
        else:
            comparison = compare_cot_vs_direct(r1, r0)
        cot_name = names[0] if name0_is_cot else names[1]
        direct_name = names[1] if name0_is_cot else names[0]
        print(f"\n=== Comparison: {cot_name} (CoT) vs {direct_name} (Direct) ===")
        for k, v in comparison.items():
            print(f"  {k}: {v}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {args.json_out}")


if __name__ == "__main__":
    main()
