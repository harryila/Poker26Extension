"""
Attention-pattern analysis at the dominant heads of each model's L*.

Question answered:
    "What does each dominant head ATTEND TO?"

Up to now we've established THAT certain heads carry the CHECK signal
(Llama L=14: h05/h23/h24; Ministral L=16: h22 + tail; Qwen L=23: TBD).
This script tells us WHAT those heads READ — i.e., which source positions /
tokens they place attention probability on at the verb-emission position.

For a sample of {clean_check_or_call, clean_legal_fold, illegal_fold}
decisions, run a forward with `output_attentions=True` and at the LAST
input position extract attention from the requested head:

    attentions[layer][0, head_idx, -1, :]   # shape [seq_len]

Then for each decision we record the top-K attended source tokens (by
attention weight) and aggregate across decisions.

Outputs:
  - SUMMARY.md per (layer, head): top-N most-frequently-attended token
    strings, plus mean attention entropy, broken down by bucket.
  - by_decision.csv: one row per (decision, layer, head) with the top-K
    token strings + their weights.

Compare across buckets: do the head's top-attended tokens DIFFER between
clean_check_or_call and illegal_fold decisions? Do they consistently
attend to position-related cues (e.g., "Bet to call:" line, "Legal actions:"
list, the trace's most recent action)?

Usage::

    python -m experiments.attention_patterns_at_dominant_heads \
        --enriched-log logs/cot_llama8b_t0_s42_*.jsonl.gz \
                       logs/cot_llama8b_t0_s123_*.jsonl.gz \
                       logs/cot_llama8b_t0_s456_*.jsonl.gz \
        --layer 14 \
        --heads 5 23 24 \
        --max-decisions-per-bucket 50 \
        --top-k 8 \
        --out-dir results/attention_patterns/llama8b_l14 \
        --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.recategorize_action_metadata import _reparse_one  # noqa: E402
from poker_env.interp.forward_helpers import (  # noqa: E402
    PromptReconstructor,
    build_input_text_for_action_verb_position,
)
from experiments.causal_patching import (  # noqa: E402
    BUCKET_NAMES,
    classify_decision,
    _iter_decisions,
    _load_agent_config,
)


def _get_top_k_tokens_for_head(
    attn_weights_at_layer,  # tensor [batch, num_heads, seq, seq]
    head_idx: int,
    input_ids,  # tensor [batch, seq]
    tokenizer,
    top_k: int,
):
    """Returns list[(rank, position, token_str, weight)] for the top-k
    attended source positions of the head at the LAST query position."""
    # [seq] = attention from last position to all source positions
    weights_last = attn_weights_at_layer[0, head_idx, -1, :]
    # Detach + CPU + numpy for sorting
    w = weights_last.detach().to("cpu").float().numpy()
    seq_len = w.shape[0]
    if seq_len == 0:
        return []
    # Top-k indices, descending weight.
    idx_sorted = w.argsort()[::-1][:top_k]
    out = []
    ids = input_ids[0].detach().to("cpu").tolist()
    for rank, pos in enumerate(idx_sorted):
        token_id = int(ids[pos]) if 0 <= pos < len(ids) else -1
        token_str = tokenizer.decode([token_id]) if token_id >= 0 else "<oob>"
        out.append((rank, int(pos), token_str, float(w[pos])))
    return out


def _entropy(p):
    """Shannon entropy of a non-negative vector that sums to ~1 (in nats)."""
    import numpy as np
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def main():
    parser = argparse.ArgumentParser(
        description="Attention-pattern analysis at dominant heads."
    )
    parser.add_argument("--enriched-log", required=True, nargs="+")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--heads", type=int, nargs="+", required=True,
                        help="Head indices to analyze.")
    parser.add_argument("--max-decisions-per-bucket", type=int, default=50,
                        help="How many decisions to sample per bucket. 50 is "
                             "usually enough to see clear patterns.")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Top-K most-attended source positions per "
                             "(decision, head).")
    parser.add_argument("--buckets", nargs="+",
                        default=["clean_check_or_call",
                                 "clean_legal_fold",
                                 "illegal_fold"],
                        choices=BUCKET_NAMES)
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
                print(f"[abort] model mismatch across logs.", file=sys.stderr)
                sys.exit(2)

    print(f"[init] model_id={model_id}")
    print(f"[init] target layer L={args.layer}; heads={args.heads}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[init] loading model on {args.device} ({args.dtype}) ...")
    t0 = time.time()
    # NOTE: HF defaults to attn_implementation="sdpa" in newer versions, which
    # does NOT return attention weights. We need eager attention for that.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()
    print(f"[init] model loaded in {time.time() - t0:.1f}s")

    recon = PromptReconstructor(tokenizer, agent_config)

    # ---- Bucket -----------------------------------------------------------
    by_bucket: dict[str, list[dict]] = {b: [] for b in BUCKET_NAMES}
    for log_path in enriched_logs:
        for rec in _iter_decisions(log_path):
            am = rec.get("action_metadata")
            if am is None or not am.get("raw_response"):
                continue
            rec.setdefault("_source_log", log_path)
            by_bucket[classify_decision(rec)].append(rec)
    for b in args.buckets:
        print(f"  {b:<22}: {len(by_bucket[b])}")

    # ---- Per-decision capture ---------------------------------------------
    by_pair_path = out_dir / "by_decision.csv"
    by_pair_f = open(by_pair_path, "w", newline="")
    writer = csv.writer(by_pair_f)
    writer.writerow([
        "bucket", "hand_id", "decision_idx", "head_idx",
        "rank", "src_position", "src_token", "attn_weight",
        "entropy_over_seq", "seq_len",
    ])

    # Aggregate per (bucket, head): Counter of token strings (top-k summed)
    agg: dict[tuple[str, int], Counter] = {}
    entropy_acc: dict[tuple[str, int], list[float]] = {}
    for b in args.buckets:
        for h in args.heads:
            agg[(b, h)] = Counter()
            entropy_acc[(b, h)] = []

    print(f"\n[capture] up to {args.max_decisions_per_bucket} decisions per bucket "
          f"(buckets: {args.buckets}) ...")
    for bucket in args.buckets:
        recs = by_bucket[bucket]
        if not recs:
            print(f"  [{bucket}] empty — skipping")
            continue
        n = 0
        t_b = time.time()
        for rec in recs:
            if n >= args.max_decisions_per_bucket:
                break
            am = rec.get("action_metadata") or {}
            raw = am.get("raw_response")
            if not raw:
                continue
            prompt = recon.build(rec)
            out = build_input_text_for_action_verb_position(prompt, raw, tokenizer)
            if out is None:
                continue
            input_text, _ = out
            enc = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(args.device)
            seq_len = int(input_ids.shape[1])

            with torch.no_grad():
                outputs = model(input_ids=input_ids, output_attentions=True)
            attn = outputs.attentions[args.layer]   # [1, num_heads, seq, seq]

            for h in args.heads:
                # entropy over the row at last position
                row = attn[0, h, -1, :].detach().to("cpu").float().numpy()
                ent = _entropy(row)
                entropy_acc[(bucket, h)].append(ent)
                top_items = _get_top_k_tokens_for_head(
                    attn, h, input_ids, tokenizer, args.top_k,
                )
                for rank, pos, tok, w in top_items:
                    agg[(bucket, h)][tok] += 1
                    writer.writerow([
                        bucket, rec.get("hand_id", "?"),
                        rec.get("decision_idx", "?"),
                        h, rank, pos, tok, f"{w:.4f}",
                        f"{ent:.4f}", seq_len,
                    ])
            n += 1
            if n % 10 == 0:
                rate = n / max(time.time() - t_b, 1e-3)
                print(f"  [{bucket}] {n} captured ({rate:.2f}/s)")
        print(f"  [{bucket}] DONE: {n} decisions ({time.time()-t_b:.0f}s)")

    by_pair_f.close()

    # ---- SUMMARY.md -------------------------------------------------------
    md = []
    md.append("# Attention-pattern analysis at dominant heads")
    md.append("")
    md.append(f"- Model: `{model_id}`")
    md.append(f"- Layer: **{args.layer}**")
    md.append(f"- Heads analyzed: {args.heads}")
    md.append(f"- Buckets: {args.buckets}")
    md.append(f"- Per-bucket sample size cap: {args.max_decisions_per_bucket}")
    md.append(f"- Top-K positions per (decision, head): {args.top_k}")
    md.append("")
    for h in args.heads:
        md.append(f"## Head {h}")
        md.append("")
        md.append("### Mean attention entropy at the last position (nats)")
        md.append("")
        md.append("| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |")
        md.append("|---|---:|---:|---:|---|")
        import numpy as np
        for b in args.buckets:
            ents = entropy_acc[(b, h)]
            if ents:
                md.append(f"| `{b}` | {len(ents)} | {np.mean(ents):.3f} | {np.std(ents):.3f} | |")
            else:
                md.append(f"| `{b}` | 0 | — | — | (empty bucket) |")
        md.append("")
        md.append(f"### Top-{args.top_k * 5}-most-frequently-attended token strings")
        md.append("")
        md.append("(Aggregated across decisions: count = number of times the "
                  f"token string appeared in the top-{args.top_k} attended "
                  "positions of any decision, summed.)")
        md.append("")
        md.append("| Rank | " + " | ".join(f"`{b}`" for b in args.buckets) + " |")
        md.append("|---:|" + "---|" * len(args.buckets))
        max_show = args.top_k * 5
        per_bucket_top = {}
        for b in args.buckets:
            cnt = agg[(b, h)]
            per_bucket_top[b] = cnt.most_common(max_show)
        for i in range(max_show):
            row_cells = []
            for b in args.buckets:
                items = per_bucket_top[b]
                if i < len(items):
                    tok, c = items[i]
                    # Escape pipe characters that would break the table.
                    safe_tok = repr(tok).replace("|", "\\|")
                    row_cells.append(f"{safe_tok} ×{c}")
                else:
                    row_cells.append("—")
            md.append(f"| {i+1} | " + " | ".join(row_cells) + " |")
        md.append("")
    md.append("## Interpretation guide")
    md.append("")
    md.append("- **If the top attended tokens differ markedly between "
              "`clean_check_or_call` and `illegal_fold`**: the head is reading "
              "different context for different verb decisions — strong "
              "evidence that the head is doing decision-relevant computation, "
              "not just attending to format/structural tokens.")
    md.append("- **If the top tokens are mostly format tokens (e.g., `:`, "
              "newlines, `\"`, prompt-section labels)**: the head is doing "
              "structural attention, not content-based decision-making. The "
              "decision signal would have to come from somewhere else "
              "(other heads, MLP, residual flow-through).")
    md.append("- **Mean entropy comparison**: heads with low entropy (~1-3 "
              "nats) on a long sequence (1000+ tokens) are sharply focused. "
              "If entropy differs systematically across buckets, the head's "
              "focus *itself* depends on which decision is being made.")
    md.append("- **Same top tokens across buckets but different ranks/weights**: "
              "the head looks at the same context but weighs it differently — "
              "consistent with a 'soft router' that emphasizes one feature "
              "(e.g., 'Bet to call:' line) for CHECK decisions and another "
              "(e.g., 'Stack:') for FOLD decisions.")
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md) + "\n")

    # summary.json
    summary = {
        "model_id": model_id,
        "layer": args.layer,
        "heads": args.heads,
        "buckets": args.buckets,
        "max_decisions_per_bucket": args.max_decisions_per_bucket,
        "top_k": args.top_k,
        "per_head_per_bucket": {},
    }
    for h in args.heads:
        summary["per_head_per_bucket"][str(h)] = {}
        import numpy as np
        for b in args.buckets:
            ents = entropy_acc[(b, h)]
            cnt = agg[(b, h)]
            summary["per_head_per_bucket"][str(h)][b] = {
                "n_decisions": len(ents),
                "mean_entropy": float(np.mean(ents)) if ents else None,
                "std_entropy": float(np.std(ents)) if ents else None,
                "top_attended_tokens": cnt.most_common(args.top_k * 3),
            }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[done] wrote {out_dir / 'SUMMARY.md'}")
    print(f"[done] wrote {out_dir / 'summary.json'}")
    print(f"[done] wrote {by_pair_path}")


if __name__ == "__main__":
    main()
