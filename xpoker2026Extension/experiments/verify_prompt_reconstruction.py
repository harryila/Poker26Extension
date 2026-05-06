"""
Phase 1b verification (REQUIRES GPU model load).

For each sampled enriched-log decision:

  1. Rebuild the EXACT input string the original run used (prompt_hash check).
  2. Run a SINGLE forward pass on the model.
  3. Confirm that the top-1 predicted next-token at the LAST input position
     matches the FIRST verb token of the recorded raw_response.

If <100% of samples pass step 3, the experiment is BLOCKED — either the
prompt-builder logic has changed since the run, the tokenizer differs, or
nondeterminism has crept in. Investigate before proceeding.

Usage on a GPU box::

    python -m experiments.verify_prompt_reconstruction \
        --enriched-log logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
        --n-samples 5 \
        --device cuda

Prints per-record table; exits 0 if all checks pass, 1 otherwise.
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
    args = parser.parse_args()

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

        is_match = (top1_id == expected_tok_id)
        if is_match:
            n_top1_match += 1

        marker = "OK " if is_match else "FAIL"
        print(f"  [{marker}] hand={rec['hand_id']} dec={rec['decision_idx']}")
        print(f"         expected: id={expected_tok_id} tok={expected_tok!r}")
        print(f"         model    : id={top1_id} tok={top1_token!r}")
        if not is_match:
            top5 = logits.topk(5)
            top5_str = ", ".join(
                f"{tokenizer.decode([int(i)])!r}({float(p):.2f})"
                for i, p in zip(top5.indices, top5.values)
            )
            print(f"         top-5 alts: {top5_str}")
            print(f"         verb_text first 8 chars in raw: {verb_text_chars!r}")

    print()
    print("=" * 72)
    print(f"Summary: {n_top1_match}/{len(samples)} top-1 matches the recorded verb")
    print("=" * 72)
    if n_top1_match < len(samples):
        print("BLOCKED: Prompt reconstruction is not byte-identical to the original run.")
        print("Halt the experiment and investigate before running causal_patching.")
    sys.exit(0 if n_top1_match == len(samples) else 1)


if __name__ == "__main__":
    main()
