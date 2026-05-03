"""
One-off: re-parse the existing Qwen 8B CoT pilot logs with Fix B (strip
`<think>...</think>` blocks) and report how many beliefs / actions we
recover compared to the original parse rate.

Background (see updates.md §7):
    The Qwen 8B CoT pilot ran with native thinking ON (the old coupling
    enable_thinking=cot_mode). 100% of belief responses opened a <think>
    block; 98% never closed it because the 768-token belief budget
    overflowed. The recorded parse rate was 1.3% — almost certainly an
    artifact of the budget, not Qwen's belief reasoning ability.

    Fix B (strip <think>...</think>) is a parser-side fix. It can only
    help in the ~2% of cases where the </think> tag actually closed AND
    a JSON block follows. This script quantifies exactly that recovery.

    The conclusion is expected to be: Fix B alone cannot rescue this run.
    Fix A (turn off native thinking entirely) is required.

Usage:
    python -m analysis.reparse_qwen_cot_pilot
"""

import gzip
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from poker_env.agents.json_utils import extract_json  # noqa: E402
from poker_env.config import BUCKET_ORDER  # noqa: E402


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
OPEN_THINK_RE = re.compile(r"<think>", re.IGNORECASE)
CLOSE_THINK_RE = re.compile(r"</think>", re.IGNORECASE)


def strip_think(raw: str) -> tuple[str, str]:
    """Strip closed <think>...</think> blocks. Return (cleaned, status).

    status is one of:
      - "no_think_tag": no <think> at all
      - "closed_think_stripped": at least one closed block; cleaned is the rest
      - "unterminated_think": <think> opened but never closed -> cleaned is text after the open tag
    """
    if "<think>" not in raw.lower():
        return raw, "no_think_tag"

    cleaned = THINK_BLOCK_RE.sub("", raw).strip()
    n_open = len(OPEN_THINK_RE.findall(raw))
    n_close = len(CLOSE_THINK_RE.findall(raw))

    if n_open == n_close:
        return cleaned, "closed_think_stripped"
    return cleaned, "unterminated_think"


def belief_is_valid(parsed: dict | None) -> tuple[bool, str | None]:
    """Mimic HFAgent._parse_compact_belief validity check.

    Returns (ok, reason_if_not).
    """
    if parsed is None:
        return False, "no_json_extracted"
    probs = parsed.get("probs")
    if not isinstance(probs, list) or len(probs) != len(BUCKET_ORDER):
        return False, "wrong_probs_shape"
    try:
        probs = [float(p) for p in probs]
    except (ValueError, TypeError):
        return False, "non_numeric_probs"
    if all(p == 0.0 for p in probs):
        return False, "all_zeros"
    return True, None


def open_log(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r")


def diagnose(path: str) -> dict:
    counts = {
        "n_decisions": 0,
        "n_belief_attempts": 0,
        "original_parse_ok": 0,
        "no_think_tag": 0,
        "closed_think_stripped": 0,
        "unterminated_think": 0,
        "after_fix_b_parse_ok": 0,
        "newly_recovered": 0,
        "still_broken_unterminated": 0,
        "still_broken_other_reasons": Counter(),
    }
    sample_recoveries: list[dict] = []
    with open_log(path) as f:
        for line in f:
            r = json.loads(line)
            if "agent_action" not in r:
                continue
            counts["n_decisions"] += 1
            bm = r.get("belief_metadata") or {}
            if not bm:
                continue
            counts["n_belief_attempts"] += 1
            original_ok = bool(bm.get("parse_success"))
            if original_ok:
                counts["original_parse_ok"] += 1

            raw = bm.get("raw_response") or ""
            cleaned, status = strip_think(raw)
            counts[status] += 1

            parsed = extract_json(cleaned)
            ok, reason = belief_is_valid(parsed)
            if ok:
                counts["after_fix_b_parse_ok"] += 1
                if not original_ok:
                    counts["newly_recovered"] += 1
                    if len(sample_recoveries) < 3:
                        sample_recoveries.append({
                            "status_after_fix_b": status,
                            "raw_len": len(raw),
                            "cleaned_len": len(cleaned),
                            "argmax_bucket": BUCKET_ORDER[
                                max(range(len(BUCKET_ORDER)), key=lambda i: parsed["probs"][i])
                            ],
                            "argmax_prob": float(max(parsed["probs"])),
                        })
            else:
                if status == "unterminated_think":
                    counts["still_broken_unterminated"] += 1
                else:
                    counts["still_broken_other_reasons"][reason or "unknown"] += 1

    counts["still_broken_other_reasons"] = dict(counts["still_broken_other_reasons"])
    counts["sample_recoveries"] = sample_recoveries
    return counts


def main() -> None:
    files = [
        "logs/cot_pilot_qwen8b_t0_s42_informative_v2.jsonl.gz",
        "logs/cot_pilot_qwen8b_t02_s42_informative_v2.jsonl.gz",
    ]
    rows = []
    for fp in files:
        d = diagnose(fp)
        rows.append((fp, d))

    print()
    print("=" * 90)
    print("Fix B post-hoc re-parse: Qwen 8B CoT pilot")
    print("=" * 90)
    print(
        f"{'file':<60} {'attempts':>8} {'orig_OK':>7} {'+B_OK':>7} {'recov':>6} {'untrm':>6}"
    )
    print("-" * 90)
    total_attempts = total_orig = total_after = total_recov = total_untrm = 0
    for fp, d in rows:
        attempts = d["n_belief_attempts"]
        total_attempts += attempts
        total_orig += d["original_parse_ok"]
        total_after += d["after_fix_b_parse_ok"]
        total_recov += d["newly_recovered"]
        total_untrm += d["still_broken_unterminated"]
        print(
            f"{Path(fp).name:<60} {attempts:>8d} "
            f"{d['original_parse_ok']:>7d} {d['after_fix_b_parse_ok']:>7d} "
            f"{d['newly_recovered']:>6d} {d['still_broken_unterminated']:>6d}"
        )
    print("-" * 90)
    print(
        f"{'TOTAL':<60} {total_attempts:>8d} {total_orig:>7d} "
        f"{total_after:>7d} {total_recov:>6d} {total_untrm:>6d}"
    )
    pct_orig = 100 * total_orig / total_attempts if total_attempts else 0
    pct_after = 100 * total_after / total_attempts if total_attempts else 0
    print()
    print(f"  Parse rate (original):       {pct_orig:6.2f}%  ({total_orig}/{total_attempts})")
    print(f"  Parse rate (after Fix B):    {pct_after:6.2f}%  ({total_after}/{total_attempts})")
    print(f"  Newly-recovered by Fix B:    {total_recov} decisions")
    print(f"  Unterminated <think> (lost): {total_untrm} decisions  <-- Fix A required to rescue these")

    print()
    print("Per-file <think> status breakdown:")
    for fp, d in rows:
        print(f"  {Path(fp).name}")
        print(f"    no_think_tag:           {d['no_think_tag']}")
        print(f"    closed_think_stripped:  {d['closed_think_stripped']}")
        print(f"    unterminated_think:     {d['unterminated_think']}")
        if d["still_broken_other_reasons"]:
            print(f"    still_broken_other_reasons: {d['still_broken_other_reasons']}")
        if d["sample_recoveries"]:
            print(f"    sample recoveries:")
            for s in d["sample_recoveries"]:
                print(f"      {s}")

    out_path = REPO_ROOT / "results" / "tier1a_small_cot_pilot" / "fix_b_reparse_qwen8b.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "files": [
            {
                "path": fp,
                **{k: v for k, v in d.items() if k != "sample_recoveries"},
                "sample_recoveries": d["sample_recoveries"],
            }
            for fp, d in rows
        ],
        "totals": {
            "belief_attempts": total_attempts,
            "original_parse_ok": total_orig,
            "after_fix_b_parse_ok": total_after,
            "newly_recovered_by_fix_b": total_recov,
            "still_broken_unterminated_think": total_untrm,
            "original_parse_rate_pct": pct_orig,
            "after_fix_b_parse_rate_pct": pct_after,
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"Wrote {out_path.relative_to(REPO_ROOT.parent)}")


if __name__ == "__main__":
    main()
