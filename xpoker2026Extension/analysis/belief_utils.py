"""
Utilities for converting between belief formats.

Provides robust conversion between:
- Compact format: {"schema": "buckets_14_v1", "probs": [0.05, 0.08, ...]}
- Full format: {"premium_pairs": 0.05, "strong_pairs": 0.08, ...}

All functions return (result, metadata) tuples for robust error handling.
No asserts - failures return (None, metadata) with error info.
"""

from typing import Any

from analysis.buckets import get_bucket_scheme, BUCKET_NAMES

# Import schema ID from config
try:
    from poker_env.config import BELIEF_SCHEMA_ID, BUCKET_ORDER
except ImportError:
    # Fallback for standalone use
    BELIEF_SCHEMA_ID = "buckets_14_v1"
    BUCKET_ORDER = BUCKET_NAMES


def compact_to_dict(
    probs: list[float], 
    scheme: str = "default"
) -> tuple[dict[str, float] | None, dict]:
    """
    Convert compact probability list to labeled dict.
    
    Args:
        probs: List of probabilities in bucket order
        scheme: Bucket scheme to use ("default", "coarse", "fine")
        
    Returns:
        (belief_dict, metadata) where metadata includes any issues found.
        Returns (None, metadata) on failure.
    """
    bucket_names = get_bucket_scheme(scheme).bucket_names
    metadata = {
        "expected_length": len(bucket_names),
        "actual_length": len(probs) if probs else 0,
        "conversion_success": True,
        "scheme": scheme,
    }
    
    # Validate input
    if probs is None:
        metadata["conversion_success"] = False
        metadata["error"] = "probs is None"
        return None, metadata
    
    if not isinstance(probs, (list, tuple)):
        metadata["conversion_success"] = False
        metadata["error"] = f"probs must be list/tuple, got {type(probs).__name__}"
        return None, metadata
    
    if len(probs) != len(bucket_names):
        metadata["conversion_success"] = False
        metadata["error"] = f"Expected {len(bucket_names)} probs, got {len(probs)}"
        return None, metadata
    
    # Try to convert all values to float
    try:
        float_probs = [float(p) for p in probs]
    except (ValueError, TypeError) as e:
        metadata["conversion_success"] = False
        metadata["error"] = f"Could not convert probs to float: {e}"
        return None, metadata
    
    return dict(zip(bucket_names, float_probs)), metadata


def dict_to_compact(
    belief: dict[str, float], 
    scheme: str = "default",
    clip_and_renormalize: bool = True,
) -> tuple[list[float], dict]:
    """
    Convert labeled dict to compact probability list.
    
    Args:
        belief: Dict mapping bucket names to probabilities
        scheme: Bucket scheme to use
        clip_and_renormalize: If True, clip negatives to 0 and renormalize
        
    Returns:
        (prob_list, metadata) with repair info in metadata.
        When clip_and_renormalize=True, always returns a valid probability
        distribution (clipped, renormalized, uniform fallback if needed).
        When clip_and_renormalize=False, returns raw values which may
        include negatives and may not sum to 1.
    """
    bucket_names = get_bucket_scheme(scheme).bucket_names
    
    metadata = {
        "scheme": scheme,
        "conversion_success": True,
    }
    
    # Handle None or empty input
    if belief is None or not isinstance(belief, dict):
        probs = [1.0 / len(bucket_names)] * len(bucket_names)
        metadata["conversion_success"] = False
        metadata["error"] = "belief is None or not a dict"
        metadata["used_uniform_fallback"] = True
        metadata["final_sum"] = 1.0
        return probs, metadata
    
    # Extract in order, fill missing with 0
    probs = []
    for name in bucket_names:
        val = belief.get(name, 0.0)
        try:
            probs.append(float(val))
        except (ValueError, TypeError):
            probs.append(0.0)
    
    metadata["missing_buckets"] = [n for n in bucket_names if n not in belief]
    metadata["extra_buckets"] = [n for n in belief if n not in bucket_names]
    metadata["original_sum"] = sum(probs)
    metadata["had_negatives"] = any(p < 0 for p in probs)
    metadata["was_repaired"] = False
    
    if clip_and_renormalize:
        # Clip negatives
        probs = [max(0.0, p) for p in probs]
        
        # Renormalize if sum > 0
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
            metadata["was_repaired"] = (
                metadata["had_negatives"] or 
                abs(metadata["original_sum"] - 1.0) > 0.01 or
                len(metadata["missing_buckets"]) > 0
            )
        else:
            # All zeros - use uniform
            probs = [1.0 / len(probs)] * len(probs)
            metadata["was_repaired"] = True
            metadata["used_uniform_fallback"] = True
    
    metadata["final_sum"] = sum(probs)
    return probs, metadata


def parse_compact_belief(
    raw_json: dict,
    scheme: str = "default",
    clip_and_renormalize: bool = True,
) -> tuple[dict[str, float] | None, dict]:
    """
    Parse a compact belief JSON and return labeled dict.
    
    Expects format: {"schema": "buckets_14_v1", "probs": [...]}
    
    Args:
        raw_json: Parsed JSON object from LLM
        scheme: Bucket scheme for validation
        clip_and_renormalize: Whether to repair invalid distributions
        
    Returns:
        (belief_dict, metadata) or (None, metadata) on failure
    """
    metadata = {
        "parse_success": False,
        "scheme": scheme,
    }
    
    if raw_json is None or not isinstance(raw_json, dict):
        metadata["error"] = "Input is None or not a dict"
        return None, metadata
    
    # Extract schema (optional validation)
    schema = raw_json.get("schema")
    metadata["schema_in_json"] = schema
    
    # Extract probs
    probs = raw_json.get("probs")
    if probs is None:
        metadata["error"] = "No 'probs' key in JSON"
        return None, metadata
    
    # Convert to dict
    belief, conv_metadata = compact_to_dict(probs, scheme)
    metadata.update(conv_metadata)
    
    if belief is None:
        return None, metadata
    
    # Optionally repair
    if clip_and_renormalize:
        compact_probs, repair_metadata = dict_to_compact(belief, scheme, clip_and_renormalize=True)
        belief = dict(zip(get_bucket_scheme(scheme).bucket_names, compact_probs))
        metadata.update(repair_metadata)
    
    metadata["parse_success"] = True
    return belief, metadata


def validate_belief(
    belief: dict[str, float],
    scheme: str = "default",
    tolerance: float = 0.01,
) -> dict:
    """
    Validate a belief distribution and return diagnostics.
    
    Args:
        belief: Dict mapping bucket names to probabilities
        scheme: Bucket scheme to validate against
        tolerance: Tolerance for sum-to-1 check
        
    Returns:
        Dict with validation results:
        - is_valid: bool
        - prob_sum: float
        - prob_min: float
        - prob_max: float
        - negative_count: int
        - missing_buckets: list
        - sum_error: float
    """
    bucket_names = get_bucket_scheme(scheme).bucket_names
    
    if belief is None or not isinstance(belief, dict):
        return {
            "is_valid": False,
            "error": "belief is None or not a dict",
        }
    
    probs = [belief.get(name, 0.0) for name in bucket_names]
    
    try:
        probs = [float(p) for p in probs]
    except (ValueError, TypeError):
        return {
            "is_valid": False,
            "error": "Could not convert all probs to float",
        }
    
    prob_sum = sum(probs)
    prob_min = min(probs)
    prob_max = max(probs)
    negative_count = sum(1 for p in probs if p < 0)
    missing_buckets = [n for n in bucket_names if n not in belief]
    sum_error = abs(prob_sum - 1.0)
    
    is_valid = (
        negative_count == 0 and
        sum_error <= tolerance and
        len(missing_buckets) == 0
    )
    
    return {
        "is_valid": is_valid,
        "prob_sum": prob_sum,
        "prob_min": prob_min,
        "prob_max": prob_max,
        "negative_count": negative_count,
        "missing_buckets": missing_buckets,
        "sum_error": sum_error,
    }


def repair_belief(
    belief: dict[str, float],
    scheme: str = "default",
) -> tuple[dict[str, float], dict]:
    """
    Repair an invalid belief distribution.
    
    Clips negatives to 0 and renormalizes to sum to 1.
    
    Args:
        belief: Dict mapping bucket names to probabilities
        scheme: Bucket scheme to use
        
    Returns:
        (repaired_belief, metadata) with repair info
    """
    # Convert to compact, repair, convert back
    compact_probs, metadata = dict_to_compact(belief, scheme, clip_and_renormalize=True)
    bucket_names = get_bucket_scheme(scheme).bucket_names
    repaired = dict(zip(bucket_names, compact_probs))
    
    return repaired, metadata
