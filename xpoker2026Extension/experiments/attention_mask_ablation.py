"""
Attention-mask ablation: mask the legal-actions list tokens from the
attention computation and measure verb-prediction degradation.

Question answered:
    "Are the model's attention heads CAUSALLY DEPENDENT on reading
     the legal-actions list to predict the verb?"

Phase J/K showed correlation: dominant heads at L* attend to verb-token
fragments from the legal-actions list (`_OR`, `_CALL`, `OLD`, `' folding'`,
etc.) with bucket-discriminating patterns. Necessity test: if we zero
the attention TO those source positions (so heads can't read them at
all), does the model's verb prediction degrade?

Implementation: identify the character offsets of the `Legal actions:`
line in each decision's prompt, map to token positions, then mask those
positions via HF's `attention_mask` argument to model.forward(). This
masks ALL heads at ALL layers from attending to those positions
(symmetric, applied to keys). Coarser than per-head/per-layer masking
but correctness-preserving and works out-of-the-box with HF.

Trade-off: this is "necessity of the *positions*", not "necessity of
h23's attention to those positions specifically." Combined with the
per-head decomposition (h5/h23/h24 carry 73% of the per-head signal),
the inference is: positions matter, h23 reads them, therefore h23's
attention to them is plausibly load-bearing. A stronger per-head test
would require deeper hooks; this test gives most of the value at a
fraction of the implementation complexity.

Usage::

    python -m experiments.attention_mask_ablation \\
        --enriched-log <pooled> \\
        --target-bucket clean_check_or_call \\
        --n-target 50 \\
        --out-dir results/attn_mask_ablation/llama8b \\
        --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_env.interp.forward_helpers import (  # noqa: E402
    PromptReconstructor,
    build_input_text_for_action_verb_position,
)
from experiments.causal_patching import (  # noqa: E402
    BUCKET_NAMES, classify_decision, build_action_token_id_sets,
    score_logits, _iter_decisions, _load_agent_config,
)


def _find_legal_actions_token_range(input_text: str, tokenizer) -> tuple[int, int] | None:
    """Find the [start, end) token range covering the `Legal actions:`
    line in the input text. Returns None if not found.

    The prompt-builder produces a line like:
        - Legal actions: ['CHECK_OR_CALL', 'FOLD']
    We span from the `[` to the closing `]` (inclusive of both, as those
    bracket the verb tokens).
    """
    needle = "Legal actions:"
    idx = input_text.find(needle)
    if idx < 0:
        return None
    bracket_open = input_text.find("[", idx)
    bracket_close = input_text.find("]", bracket_open)
    if bracket_open < 0 or bracket_close < 0:
        return None
    char_start = bracket_open
    char_end = bracket_close + 1
    # Tokenize WITH offsets to map char range to token range.
    enc = tokenizer(input_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping")
    if offsets is None:
        return None
    tok_start = None
    tok_end = None
    for tok_idx, (s, e) in enumerate(offsets):
        if tok_start is None and s >= char_start:
            tok_start = tok_idx
        if e >= char_end:
            tok_end = tok_idx + 1
            break
    if tok_start is None or tok_end is None:
        return None
    return (tok_start, tok_end)


def main():
    parser = argparse.ArgumentParser(
        description="Attention-mask ablation on the legal-actions list."
    )
    parser.add_argument("--enriched-log", required=True, nargs="+")
    parser.add_argument("--target-bucket", default="clean_check_or_call",
                        choices=BUCKET_NAMES)
    parser.add_argument("--n-target", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enriched_logs = list(args.enriched_log)

    agent_config = _load_agent_config(enriched_logs[0])
    model_id = args.model_id or agent_config["model_id"]
    if len(enriched_logs) > 1:
        for extra in enriched_logs[1:]:
            other_cfg = _load_agent_config(extra)
            if other_cfg.get("model_id") != model_id:
                print(f"[abort] model mismatch", file=sys.stderr)
                sys.exit(2)
    print(f"[init] model_id={model_id}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import random
    rng = random.Random(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[init] loading model on {args.device} ({args.dtype}) ...")
    t0 = time.time()
    # Use eager attention so attention_mask is honored as a [batch, seq] mask
    # over keys. SDPA supports the same masking semantics but is slightly
    # less transparent under per-token zero masks; eager is safer here.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()
    print(f"[init] model loaded in {time.time() - t0:.1f}s")

    family_ids = build_action_token_id_sets(tokenizer)
    recon = PromptReconstructor(tokenizer, agent_config)

    # Bucket scan + sample.
    by_bucket: dict[str, list[dict]] = {b: [] for b in BUCKET_NAMES}
    for log_path in enriched_logs:
        for rec in _iter_decisions(log_path):
            am = rec.get("action_metadata")
            if am is None or not am.get("raw_response"):
                continue
            by_bucket[classify_decision(rec)].append(rec)
    targets_pool = by_bucket[args.target_bucket]
    if not targets_pool:
        print(f"[abort] empty target bucket", file=sys.stderr)
        sys.exit(2)
    targets = rng.sample(targets_pool, min(args.n_target * 2, len(targets_pool)))
    # We sample 2x and drop those where we can't find the legal-actions span.

    by_pair_path = out_dir / "by_target.csv"
    by_pair_f = open(by_pair_path, "w", newline="")
    writer = csv.writer(by_pair_f)
    writer.writerow([
        "target_idx", "hand_id", "decision_idx",
        "tok_start_legal", "tok_end_legal", "n_tokens_masked",
        "mode",
        "top1_id", "top1_tok", "top1_group",
        "delta_check_minus_fold",
        "GROUP_CHECK_CALL", "GROUP_FOLD", "GROUP_BET_RAISE",
    ])

    n_baseline_match = 0
    n_masked_match = 0
    n_top1_id_changed = 0
    n_top1_group_changed = 0
    sum_delta_cmf = 0.0
    n_processed = 0

    print(f"[main] running baseline + masked-attention forwards ...")
    for ti, rec in enumerate(targets):
        if n_processed >= args.n_target:
            break
        am = rec.get("action_metadata") or {}
        raw = am.get("raw_response")
        if not raw:
            continue
        full_prompt = recon.build(rec)
        out = build_input_text_for_action_verb_position(
            full_prompt, raw, tokenizer,
        )
        if out is None:
            continue
        input_text, verb_resp_idx = out
        # Locate legal-actions token range in the input text.
        rng_legal = _find_legal_actions_token_range(input_text, tokenizer)
        if rng_legal is None:
            continue
        tok_start, tok_end = rng_legal
        n_tokens_masked = tok_end - tok_start

        enc = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(args.device)
        seq_len = int(input_ids.shape[1])

        # Get the recorded verb token id for this target.
        resp_tok_ids = tokenizer(raw, add_special_tokens=False)["input_ids"]
        if not (0 <= verb_resp_idx < len(resp_tok_ids)):
            continue
        expected_verb_tok_id = int(resp_tok_ids[verb_resp_idx])

        # Baseline forward (no mask).
        with torch.no_grad():
            base_out = model(input_ids=input_ids)
        base_logits = base_out.logits[0, -1, :].detach().to("cpu")
        base_top1 = int(base_logits.argmax().item())
        base_scored = score_logits(base_logits, family_ids)
        base_group = "OTHER"
        for fam, ids in family_ids.items():
            if base_top1 in ids:
                if fam == "FOLD": base_group = "FOLD"
                elif fam in ("CHECK", "CALL"): base_group = "CHECK_CALL"
                elif fam in ("BET", "RAISE"): base_group = "BET_RAISE"
                break

        writer.writerow([
            ti, rec.get("hand_id", "?"), rec.get("decision_idx", "?"),
            tok_start, tok_end, n_tokens_masked, "baseline",
            base_top1, tokenizer.decode([base_top1]), base_group,
            f"{base_scored['delta_check_minus_fold']:.4f}",
            f"{base_scored['GROUP_CHECK_CALL']:.4f}",
            f"{base_scored['GROUP_FOLD']:.4f}",
            f"{base_scored['GROUP_BET_RAISE']:.4f}",
        ])

        # Masked forward — set attention_mask = 0 at legal-actions positions.
        # Important: DO NOT mask the last position (the verb-predecessor) or
        # any position the model needs for normal causal flow.
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=args.device)
        attention_mask[0, tok_start:tok_end] = 0
        with torch.no_grad():
            masked_out = model(input_ids=input_ids, attention_mask=attention_mask)
        masked_logits = masked_out.logits[0, -1, :].detach().to("cpu")
        masked_top1 = int(masked_logits.argmax().item())
        masked_scored = score_logits(masked_logits, family_ids)
        masked_group = "OTHER"
        for fam, ids in family_ids.items():
            if masked_top1 in ids:
                if fam == "FOLD": masked_group = "FOLD"
                elif fam in ("CHECK", "CALL"): masked_group = "CHECK_CALL"
                elif fam in ("BET", "RAISE"): masked_group = "BET_RAISE"
                break

        delta = (masked_scored["delta_check_minus_fold"]
                 - base_scored["delta_check_minus_fold"])

        writer.writerow([
            ti, rec.get("hand_id", "?"), rec.get("decision_idx", "?"),
            tok_start, tok_end, n_tokens_masked, "masked",
            masked_top1, tokenizer.decode([masked_top1]), masked_group,
            f"{delta:.4f}",
            f"{masked_scored['GROUP_CHECK_CALL']:.4f}",
            f"{masked_scored['GROUP_FOLD']:.4f}",
            f"{masked_scored['GROUP_BET_RAISE']:.4f}",
        ])

        if base_top1 == expected_verb_tok_id:
            n_baseline_match += 1
        if masked_top1 == expected_verb_tok_id:
            n_masked_match += 1
        if masked_top1 != base_top1:
            n_top1_id_changed += 1
        if masked_group != base_group:
            n_top1_group_changed += 1
        sum_delta_cmf += delta
        n_processed += 1
        if (n_processed) % 10 == 0:
            print(f"  [{n_processed}/{args.n_target}] processed")

    by_pair_f.close()

    if n_processed == 0:
        print("[abort] no valid targets processed (could not find Legal "
              "actions span). Check the prompt format.")
        sys.exit(3)

    summary = {
        "model_id": model_id,
        "target_bucket": args.target_bucket,
        "n_target_processed": n_processed,
        "frac_verb_predicted_baseline": n_baseline_match / n_processed,
        "frac_verb_predicted_masked": n_masked_match / n_processed,
        "frac_top1_id_changed": n_top1_id_changed / n_processed,
        "frac_top1_group_changed": n_top1_group_changed / n_processed,
        "mean_delta_check_minus_fold": sum_delta_cmf / n_processed,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = ["# Attention-mask ablation results", ""]
    md.append(f"- Model: `{model_id}`")
    md.append(f"- Target bucket: `{args.target_bucket}`")
    md.append(f"- n_target processed: {n_processed}")
    md.append("")
    md.append("Mask: set `attention_mask=0` at the token range covering the "
              "`Legal actions: [...]` line of the prompt. This zeros attention "
              "TO those key positions across ALL heads at ALL layers.")
    md.append("")
    md.append(f"- **Verb predicted, baseline**: {summary['frac_verb_predicted_baseline']*100:.1f}%")
    md.append(f"- **Verb predicted, masked**:   {summary['frac_verb_predicted_masked']*100:.1f}%")
    md.append(f"- **Top-1 ID changed**: {summary['frac_top1_id_changed']*100:.1f}%")
    md.append(f"- **Top-1 family changed**: {summary['frac_top1_group_changed']*100:.1f}%")
    md.append(f"- **Mean Δ(CHECK − FOLD)** under mask: {summary['mean_delta_check_minus_fold']:+.3f}")
    md.append("")
    md.append("## Reading guide")
    md.append("- **Verb-predicted-baseline ≫ Verb-predicted-masked** (e.g. "
              "100% → <50%): the model's verb prediction is causally dependent "
              "on attending to the legal-actions list. Combined with the "
              "Phase J/K finding that dominant heads at L* attend to those "
              "positions, the inference is: those heads' attention to the "
              "legal-actions tokens is load-bearing for the verb prediction.")
    md.append("- **Top-1 family changed ≥ 50%**: masking reliably shifts the "
              "model's verb-distribution. Strong necessity finding.")
    md.append("- **No degradation**: model recovers from the mask via other "
              "context. Either the heads have alternate inputs (memorization, "
              "structural cues) or the verb prediction doesn't actually need "
              "that line.")
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[done] wrote {out_dir / 'SUMMARY.md'}")
    print(f"[done] wrote {out_dir / 'summary.json'}")
    print(f"[done] wrote {by_pair_path}")


if __name__ == "__main__":
    main()
