"""
Phase 1c verification (CPU-only, no GPU model load required).

Validates that for every sampled enriched-log decision:

  1. The user-prompt rebuilt from `obs` matches the recorded `prompt_hash`
     bit-for-bit (already verified for Ministral; we re-check here per cell).

  2. The action-verb char-offset inside `raw_response` is found correctly
     (locator: walk back from JSON `'"action"'` key to the value-opening
     quote, take the next char as the verb start).

  3. The full input string `prompt + response[:verb_offset]` is well-formed
     and the LAST few tokens after tokenization decode back to the JSON
     pattern `... "action": "` (the position whose lm_head output WOULD
     predict the verb token).

What this script CANNOT verify (requires a forward pass, hence GPU):
  - Whether the model's actual top-1 prediction at that position matches
    the recorded raw_response's verb. Use
    ``experiments/verify_prompt_reconstruction.py`` for that on the GPU box.

Usage::

    python -m experiments.verify_position_mapping \
        --enriched-log logs/cot_<model>_<temp>_<seed>_<opp>_logitlens_enriched.jsonl.gz \
        --n-samples 10 \
        --tokenizer mistralai/Ministral-8B-Instruct-2410

Output: prints a per-record table; exits 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

# Allow running from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_env.interp.forward_helpers import (  # noqa: E402
    PromptReconstructor,
    build_input_text_for_action_verb_position,
)


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
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield rec


def _load_agent_config(path: str) -> dict:
    """Find the first HFAgent run_config record."""
    for rec in _iter_decisions(path):
        if rec.get("type") == "run_config":
            for ac in rec.get("agent_configs", []):
                if ac.get("type") == "HFAgent":
                    return ac
    raise ValueError(f"No HFAgent run_config found in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1c: verify position mapping on a single enriched log (CPU only)."
    )
    parser.add_argument("--enriched-log", required=True)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer id (defaults to model_id from agent_config)",
    )
    args = parser.parse_args()

    agent_config = _load_agent_config(args.enriched_log)
    tokenizer_id = args.tokenizer or agent_config.get("model_id")
    print(f"Loading tokenizer: {tokenizer_id} ...")
    from transformers import AutoTokenizer  # heavy; import lazily
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("OK\n")

    recon = PromptReconstructor(tokenizer, agent_config)

    # Collect samples (with action_metadata + non-empty raw_response).
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

    print(f"Validating on {len(samples)} samples\n")

    n_hash_match = 0
    n_verb_found = 0
    n_input_well_formed = 0

    for i, rec in enumerate(samples):
        am = rec["action_metadata"]
        raw = am["raw_response"]
        log_hash = am.get("prompt_hash") or rec.get("prompt_hash")

        # (1) prompt-hash check
        user_prompt = recon.build_user_prompt(rec)
        import hashlib
        recomputed_hash = hashlib.sha256(user_prompt.encode()).hexdigest()[:16]
        hash_ok = (log_hash == recomputed_hash)
        if hash_ok:
            n_hash_match += 1

        # (2) verb-finder
        full_prompt = recon.build(rec)  # chat-template-applied
        out = build_input_text_for_action_verb_position(full_prompt, raw, tokenizer)
        verb_ok = out is not None
        verb_text = "?"
        last_tokens_decoded = "?"
        input_text = ""
        if verb_ok:
            n_verb_found += 1
            input_text, verb_resp_idx = out
            verb_text = raw[len(input_text) - len(full_prompt):][:8].replace("\n", "\\n")

            # (3) Tokenize the input_text and decode the LAST 6 tokens. The
            #     pattern should end with something like ' "action": "' or
            #     'action":' depending on tokenizer.
            tok_ids = tokenizer(input_text, add_special_tokens=False)["input_ids"]
            last6 = tokenizer.decode(tok_ids[-6:])
            last_tokens_decoded = repr(last6)
            # Heuristic: input should end with a quote-like character, since
            # the verb starts right after the open-quote of the value.
            stripped = last6.rstrip()
            input_well_formed = stripped.endswith('"') or stripped.endswith("'")
            if input_well_formed:
                n_input_well_formed += 1
        else:
            input_well_formed = False

        marker = "OK " if (hash_ok and verb_ok and input_well_formed) else "FAIL"
        print(
            f"  [{marker}] hand={rec['hand_id']} dec={rec['decision_idx']} "
            f"hash_match={hash_ok} verb_found={verb_ok} "
            f"input_ends_in_quote={input_well_formed}"
        )
        print(
            f"         verb_text={verb_text!r:<10} "
            f"last6_tokens={last_tokens_decoded}"
        )

    print()
    print("=" * 72)
    print(f"Summary: {n_hash_match}/{len(samples)} hash matches, "
          f"{n_verb_found}/{len(samples)} verbs found, "
          f"{n_input_well_formed}/{len(samples)} inputs end in quote")
    print("=" * 72)

    all_ok = (n_hash_match == len(samples)
              and n_verb_found == len(samples)
              and n_input_well_formed == len(samples))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
