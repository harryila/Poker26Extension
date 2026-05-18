"""
Diagnose `PromptReconstructor` byte-identicality on opp-preset Tier 4
enriched logs.

Context
-------
`scripts/run_tier4_patching.sh` reported `baseline_top1_match_rate = 0.57–0.81`
for the three Llama Tier 4 cells that ran (vs canonical ≥0.95 on
informative_v2 baseline logs). Qwen and Ministral Tier 4 cells had
baseline 1.000. The audit (`AUDIT.md` M6, `updates.md` §22f) attributed
this to a chat-template / opponent-description serialization difference
between the opp-preset code path in `run_experiment.py` and what
`PromptReconstructor.build()` produces. This script verifies that
hypothesis and identifies the specific bytes that differ.

The Tier 4 enriched logs store `prompt_hash` (SHA-256 first-16-hex of
the user_prompt string at inference time) but NOT the user_prompt itself.
So we cannot directly diff strings; we can only:

  1. Reconstruct the user_prompt via `PromptReconstructor.build(rec)`,
     hash it the same way (SHA-256 first 16 hex chars), and report the
     hash-match rate per (model, preset) cell.
  2. For the FIRST mismatching record, print the reconstructed prompt +
     the `agent_config` from the log header, so a human can compare to
     what `run_experiment.py` would have produced at inference time.

CPU-only — no model load. Runs in seconds.

Usage::

    # Default: check all opp-preset Tier 4 logs for all three models.
    python -m experiments.diagnose_opp_preset_reconstruction

    # Or check a specific cell:
    python -m experiments.diagnose_opp_preset_reconstruction \\
        --log logs/opp_default_llama-8b_t00_s42_enriched.jsonl \\
        --n-samples 30 --print-first-mismatch

    # Cross-check against a known-good informative_v2 baseline log:
    python -m experiments.diagnose_opp_preset_reconstruction \\
        --log logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz

Exit code 0 if every checked log has hash-match >= --pass-threshold
(default 0.95). Exit code 1 otherwise.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_env.interp.forward_helpers import PromptReconstructor  # noqa: E402


DEFAULT_OPP_LOGS = [
    "logs/opp_default_llama-8b_t00_s42_enriched.jsonl",
    "logs/opp_default_ministral-8b_t00_s42_enriched.jsonl",
    "logs/opp_default_qwen-8b_t00_s42_enriched.jsonl",
    "logs/opp_informative_v2_llama-8b_t00_s42_enriched.jsonl",
    "logs/opp_informative_v2_ministral-8b_t00_s42_enriched.jsonl",
    "logs/opp_informative_v2_qwen-8b_t00_s42_enriched.jsonl",
    "logs/opp_tight_aggressive_llama-8b_t00_s42_enriched.jsonl",
    "logs/opp_tight_aggressive_ministral-8b_t00_s42_enriched.jsonl",
    "logs/opp_tight_aggressive_qwen-8b_t00_s42_enriched.jsonl",
    "logs/opp_loose_aggressive_llama-8b_t00_s42_enriched.jsonl",
    "logs/opp_loose_aggressive_ministral-8b_t00_s42_enriched.jsonl",
    "logs/opp_loose_aggressive_qwen-8b_t00_s42_enriched.jsonl",
]


def _open_log(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_decisions(path: str) -> Iterator[dict]:
    with _open_log(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_agent_config(path: str) -> dict:
    for rec in _iter_decisions(path):
        if rec.get("type") == "run_config":
            for ac in rec.get("agent_configs", []):
                if ac.get("type") == "HFAgent":
                    return ac
    raise ValueError(f"No HFAgent run_config in {path}")


def _short_hash(s: str) -> str:
    """Mirror the `prompt_hash` format used at inference time:
    SHA-256 of UTF-8 bytes, first 16 hex chars."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def diagnose_log(log_path: str, n_samples: int, print_first_mismatch: bool
                 ) -> dict:
    """Returns a dict with match counts + first-mismatch detail (if any)."""
    if not os.path.exists(log_path):
        return {"log": log_path, "status": "missing", "match": 0, "checked": 0}

    try:
        agent_config = _load_agent_config(log_path)
    except Exception as e:
        return {"log": log_path, "status": f"agent_config_error: {e}",
                "match": 0, "checked": 0}

    # PromptReconstructor needs a tokenizer for chat-template assembly in
    # build(), but for the *user_prompt* portion (which is what's hashed
    # at inference time per `_short_hash`) we only need the prompt-string
    # builder. We DON'T load a model; we just need the tokenizer for the
    # PromptReconstructor's internal book-keeping.
    #
    # Workaround: PromptReconstructor in this codebase needs a tokenizer.
    # On CPU we'd download it; instead, we instantiate with a dummy
    # tokenizer that exposes only the attributes PromptReconstructor uses
    # for the user_prompt path (pad_token / eos_token are NOT touched by
    # build() in the user_prompt-string branch).
    #
    # If that doesn't work because build() needs the tokenizer for chat-
    # template wrapping, the user can re-run this on the GPU box where
    # the tokenizer is cached.
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(agent_config["model_id"])
    except Exception as e:
        return {
            "log": log_path,
            "status": f"tokenizer_load_failed (run on GPU box where it's cached): {e}",
            "match": 0,
            "checked": 0,
        }

    recon = PromptReconstructor(tokenizer, agent_config)

    n_match = 0
    n_checked = 0
    first_mismatch = None
    for rec in _iter_decisions(log_path):
        am = rec.get("action_metadata") or {}
        if not am.get("raw_response"):
            continue
        recorded_hash = (
            am.get("prompt_hash")
            or rec.get("prompt_hash")
        )
        if not recorded_hash:
            continue

        try:
            reconstructed = recon.build(rec)
        except Exception as e:
            return {
                "log": log_path,
                "status": f"reconstruction_failed: {e}",
                "match": n_match,
                "checked": n_checked,
            }
        computed_hash = _short_hash(reconstructed)
        n_checked += 1
        if computed_hash == recorded_hash:
            n_match += 1
        elif first_mismatch is None:
            first_mismatch = {
                "hand_id": rec.get("hand_id"),
                "decision_idx": rec.get("decision_idx"),
                "recorded_hash": recorded_hash,
                "computed_hash": computed_hash,
                "reconstructed_user_prompt": reconstructed,
                "agent_config": agent_config,
            }
        if n_checked >= n_samples:
            break

    rate = (n_match / n_checked) if n_checked else 0.0
    result = {
        "log": log_path,
        "status": "OK",
        "model_id": agent_config.get("model_id"),
        "match": n_match,
        "checked": n_checked,
        "match_rate": rate,
    }
    if print_first_mismatch and first_mismatch is not None:
        result["first_mismatch"] = first_mismatch
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--log",
        action="append",
        default=None,
        help="Specific enriched log to check. Can be passed multiple times. "
             "Defaults to the standard opp-preset Tier 4 list (12 cells, "
             "excluding loose_passive which has 0 clean_LF).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="Records to check per log (default 20).",
    )
    parser.add_argument(
        "--print-first-mismatch", action="store_true",
        help="On the first hash-mismatching record per log, print the "
             "reconstructed user_prompt + the agent_config so a human "
             "can identify what differs from run_experiment.py.",
    )
    parser.add_argument(
        "--pass-threshold", type=float, default=0.95,
        help="Hash-match rate to consider the cell PASS (default 0.95).",
    )
    args = parser.parse_args()

    log_paths = args.log if args.log else DEFAULT_OPP_LOGS
    # Make paths absolute-relative to the repo root.
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    print(f"\n{'log':<70} {'model':<35} {'match/check':>12}  {'rate':>8}  {'verdict':>8}")
    print("-" * 138)
    all_results = []
    for lp in log_paths:
        r = diagnose_log(lp, args.n_samples, args.print_first_mismatch)
        all_results.append(r)
        if r["status"] != "OK":
            print(f"{lp:<70} {'?':<35} {'-':>12}  {'-':>8}  {r['status']}")
            continue
        verdict = "PASS" if r["match_rate"] >= args.pass_threshold else "FAIL"
        print(
            f"{lp:<70} {r.get('model_id', '?'):<35} "
            f"{r['match']:>4}/{r['checked']:<4}     "
            f"{r['match_rate']*100:>6.1f}%   {verdict:>8}"
        )

    # Per-model summary.
    by_model: dict[str, list[dict]] = {}
    for r in all_results:
        if r["status"] != "OK":
            continue
        by_model.setdefault(r["model_id"], []).append(r)
    print(f"\n{'model':<50} {'mean match rate':>20} {'min cell':>12}")
    print("-" * 90)
    for m, rs in by_model.items():
        mean = sum(r["match_rate"] for r in rs) / len(rs)
        mn = min(r["match_rate"] for r in rs)
        print(f"{m:<50} {mean*100:>18.1f}% {mn*100:>10.1f}%")

    # Print first-mismatch details where requested.
    if args.print_first_mismatch:
        for r in all_results:
            if r.get("first_mismatch") is not None:
                fm = r["first_mismatch"]
                print(f"\n{'='*100}")
                print(f"FIRST MISMATCH in {r['log']}")
                print(f"  hand_id={fm['hand_id']} decision_idx={fm['decision_idx']}")
                print(f"  recorded prompt_hash: {fm['recorded_hash']}")
                print(f"  computed prompt_hash: {fm['computed_hash']}")
                print(f"\n  agent_config (from log run_config):")
                for k, v in fm["agent_config"].items():
                    print(f"    {k}: {repr(v)[:200]}")
                print(f"\n  reconstructed user_prompt (first 4000 chars):")
                print("  " + "-" * 100)
                rp = fm["reconstructed_user_prompt"]
                # Indent and truncate for readability.
                for line in rp[:4000].splitlines():
                    print(f"  | {line}")
                if len(rp) > 4000:
                    print(f"  ... ({len(rp) - 4000} more chars truncated)")
                print("  " + "-" * 100)
                print(
                    "\n  Action: compare the reconstructed prompt above to what "
                    "`run_experiment.py` produces for this preset+model. The "
                    "divergence is most likely in (a) the opponent-description "
                    "sentence (different presets serialize differently), (b) the "
                    "history-bullets format, or (c) a trailing newline / whitespace."
                )
                print(
                    "  After identifying the diff, patch "
                    "`poker_env/interp/forward_helpers.PromptReconstructor` "
                    "OR `run_experiment.py`'s prompt builder to bring them into "
                    "byte-identicality."
                )

    n_fail = sum(
        1 for r in all_results
        if r["status"] == "OK" and r["match_rate"] < args.pass_threshold
    )
    if n_fail > 0:
        print(f"\n[FAIL] {n_fail} log(s) below pass threshold {args.pass_threshold}.")
        sys.exit(1)
    print(f"\n[PASS] all logs at or above pass threshold {args.pass_threshold}.")
    sys.exit(0)


if __name__ == "__main__":
    main()
