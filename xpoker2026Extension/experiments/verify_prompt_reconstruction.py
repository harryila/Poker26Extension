"""
Phase 1b verification (REQUIRES GPU model load).

For each sampled enriched-log decision:

  1. Rebuild the EXACT input string the original run used (prompt_hash check).
  2. Run a SINGLE forward pass on the model.
  3. Confirm that the recorded FIRST verb token of raw_response is one of the
     model's --tie-top-k (default 2) predictions at the LAST input position
     AND has a logit within --tie-tolerance-nats (default 0.10) of the top-1
     logit.

The top-1 strict equality used previously is a special case (gap=0). We relax
it to admit literal bf16 logit ties — at logit magnitudes of ~24 nats one
bf16 ULP is ≈ 0.19 nats, so 0.10 is sub-ULP and corresponds to "the recorded
verb is computationally indistinguishable from top-1 at bf16 precision."
This avoids spurious BLOCKs when two candidates round to the same bf16 value
and argmax tiebreak order differs between runs on the same hardware.

A real prompt-reconstruction drift would push the recorded verb out of the
top-K, or leave it in top-K but >> tolerance behind top-1; both still hard
FAIL.

If more than --max-failures samples fail (default 0 — strict), the experiment
is BLOCKED — either the prompt-builder logic has changed since the run, the
tokenizer differs, or nondeterminism has crept in. Investigate before
proceeding.

For experiments where bf16 nondeterminism on the verb position is more common
than usual (e.g. opp-preset enriched logs that may differ from the
informative_v2 baseline used to validate the prompt reconstructor), set
--tie-tolerance-nats to a higher value (e.g. 0.50) and/or --max-failures to a
small budget (e.g. 1 or 2 out of 5-10 samples). The patching driver has its
own baseline_top1_match_rate check anyway, so the upstream pre-flight only
needs to confirm the prompt-builder isn't grossly broken.

Each per-sample line prints one of three outcomes:
  [OK  ]  strict top-1 match (gap=0 by construction)
  [TIE ]  expected token in top-K and within tolerance of top-1
  [FAIL] expected token absent from top-K or beyond tolerance

Usage on a GPU box::

    # Strict (default — 0 failures allowed at 0.10 nat tolerance):
    python -m experiments.verify_prompt_reconstruction \
        --enriched-log logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
        --n-samples 5 --device cuda

    # Relaxed (for opp-preset enriched logs where 1-2 bf16-ULP flips are
    # acceptable):
    python -m experiments.verify_prompt_reconstruction \
        --enriched-log logs/opp_loose_aggressive_ministral-8b_t00_s42_enriched.jsonl \
        --n-samples 5 --device cuda \
        --tie-tolerance-nats 0.50 --max-failures 2

Prints per-record table; exits 0 if at most --max-failures samples fail under
the chosen tolerance, 1 otherwise.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_env.interp.forward_helpers import (  # noqa: E402
    PromptReconstructor,
    build_input_text_for_action_verb_position,
    run_forward_at_last_position,
)

# Pre-flight gate defaults. See module docstring for the bf16-precision
# rationale. The defaults are STRICT (zero allowed failures, 0.10 nat
# tolerance, K=2). Loosen via CLI flags for cells where bf16 ULP noise on
# the verb position is more common (e.g. opp-preset enriched logs) — the
# downstream patching driver has its own baseline_top1_match_rate check
# that catches genuine prompt-builder breakage.
DEFAULT_TIE_TOLERANCE_NATS = 0.10  # ≈ half a bf16 ULP at typical logit magnitudes
DEFAULT_TIE_TOP_K = 2              # only the runner-up to top-1 counts as a tie
DEFAULT_MAX_FAILURES = 0           # 0 = every sample must pass


def _open_log(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_decisions(path: str):
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


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1b: verify prompt reconstruction with full forward pass (GPU)."
    )
    parser.add_argument("--enriched-log", required=True)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override model_id from agent_config (defaults to recorded one).",
    )
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--tie-tolerance-nats",
        type=float,
        default=DEFAULT_TIE_TOLERANCE_NATS,
        help=(
            "Maximum logit gap (in nats) between the recorded verb and the "
            "top-1 prediction for a sample to be classified as TIE rather "
            f"than FAIL. Default {DEFAULT_TIE_TOLERANCE_NATS} (sub-bf16-ULP). "
            "Increase to e.g. 0.50 for cells where bf16 ULP noise is "
            "expected (opp-preset enriched logs)."
        ),
    )
    parser.add_argument(
        "--tie-top-k",
        type=int,
        default=DEFAULT_TIE_TOP_K,
        help=(
            "Top-K of the model prediction within which the recorded verb "
            "must appear for a TIE. Default 2 (only top-1 and runner-up "
            "qualify)."
        ),
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=DEFAULT_MAX_FAILURES,
        help=(
            "Number of FAIL samples tolerated before the script exits "
            "non-zero. Default 0 (strict, every sample must pass). For "
            "cells with known small-but-real bf16 nondeterminism, set "
            "this to a small budget (e.g. 1 or 2 out of 5-10 samples)."
        ),
    )
    args = parser.parse_args()
    tie_tolerance = float(args.tie_tolerance_nats)
    tie_top_k = int(args.tie_top_k)
    max_failures = int(args.max_failures)

    agent_config = _load_agent_config(args.enriched_log)
    model_id = args.model_id or agent_config["model_id"]
    print(f"Loading tokenizer: {model_id}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"Loading model: {model_id} (dtype={args.dtype}, device={args.device}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=args.device,
    )
    model.eval()
    print("OK\n")

    recon = PromptReconstructor(tokenizer, agent_config)

    samples = []
    for rec in _iter_decisions(args.enriched_log):
        am = rec.get("action_metadata")
        if am is None:
            continue
        if not am.get("raw_response"):
            continue
        samples.append(rec)
        if len(samples) >= args.n_samples:
            break

    print(f"Verifying on {len(samples)} samples\n")

    n_top1_match = 0
    for i, rec in enumerate(samples):
        am = rec["action_metadata"]
        raw = am["raw_response"]
        full_prompt = recon.build(rec)
        out = build_input_text_for_action_verb_position(full_prompt, raw, tokenizer)
        if out is None:
            print(f"  [FAIL] hand={rec['hand_id']} dec={rec['decision_idx']}: "
                  f"verb not found in raw_response")
            continue
        input_text, verb_resp_idx = out

        result = run_forward_at_last_position(
            model, tokenizer, input_text, device=args.device,
        )
        logits = result["logits_last_pos"]
        top1_id = int(logits.argmax().item())
        top1_token = tokenizer.decode([top1_id])

        # First verb token in raw_response, computed from offset_mapping if
        # available -- here we just take whatever character starts at the
        # found offset and the first token of the response from that offset.
        verb_text_chars = raw[len(input_text) - len(full_prompt):][:8]
        # Token that the response's tokenization places at verb_resp_idx
        # (verb_resp_idx is from find_action_verb_response_offset).
        resp_tok_ids = tokenizer(raw, add_special_tokens=False)["input_ids"]
        if 0 <= verb_resp_idx < len(resp_tok_ids):
            expected_tok_id = resp_tok_ids[verb_resp_idx]
            expected_tok = tokenizer.decode([expected_tok_id])
        else:
            expected_tok_id = -1
            expected_tok = "<unknown>"

        is_strict_match = (top1_id == expected_tok_id)
        topk = logits.topk(tie_top_k)
        topk_ids = {int(i) for i in topk.indices.tolist()}
        if 0 <= expected_tok_id < logits.shape[-1]:
            gap_to_top1 = (
                float(logits[top1_id].item())
                - float(logits[expected_tok_id].item())
            )
        else:
            gap_to_top1 = float("inf")

        if is_strict_match:
            is_match, outcome = True, "OK  "
        elif (
            expected_tok_id in topk_ids
            and gap_to_top1 < tie_tolerance
        ):
            is_match, outcome = True, "TIE "
        else:
            is_match, outcome = False, "FAIL"

        if is_match:
            n_top1_match += 1

        print(f"  [{outcome}] hand={rec['hand_id']} dec={rec['decision_idx']}")
        print(f"         expected: id={expected_tok_id} tok={expected_tok!r}")
        print(f"         model   : id={top1_id} tok={top1_token!r}")
        if outcome != "OK  ":
            top5 = logits.topk(5)
            top5_str = ", ".join(
                f"{tokenizer.decode([int(i)])!r}({float(p):.2f})"
                for i, p in zip(top5.indices, top5.values)
            )
            print(f"         top-5 alts: {top5_str}")
            if outcome == "TIE ":
                print(
                    f"         gap to top-1: {gap_to_top1:.4f} nats "
                    f"(tolerance {tie_tolerance:.2f}, K={tie_top_k})"
                )
            else:
                print(f"         verb_text first 8 chars in raw: {verb_text_chars!r}")

    n_failures = len(samples) - n_top1_match
    print()
    print("=" * 72)
    print(
        f"Summary: {n_top1_match}/{len(samples)} samples passed the pre-flight gate "
        f"(top-{tie_top_k} within {tie_tolerance:.2f} nats of top-1; "
        f"failure budget = {max_failures})."
    )
    print("=" * 72)
    if n_failures > max_failures:
        print(
            f"BLOCKED: {n_failures} sample(s) failed (> budget {max_failures}). "
            "The recorded verb is either absent from the top-K predictions or "
            "further than the tolerance behind top-1 on more samples than the "
            "configured budget allows. Likely not pure bf16 tiebreak — the "
            "prompt reconstruction may not be byte-identical to the original "
            "run, or genuine model nondeterminism is biting."
        )
        print(
            "Halt the experiment and investigate, OR re-run with a higher "
            "--tie-tolerance-nats / --max-failures budget if you've manually "
            "verified the failures are bf16 ULP flips on the verb position."
        )
        sys.exit(1)

    if n_failures > 0:
        print(
            f"WARN: {n_failures} sample(s) failed the strict gate but the "
            f"failure budget ({max_failures}) absorbs them. Proceeding."
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
