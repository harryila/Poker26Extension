"""
Head ablation (necessity test) for the L*=14 attention heads in Llama
(and the analog dominant heads in Ministral and Qwen).

Question answered:
    "Are the heads we identified as carrying the verb signal CAUSALLY
     NECESSARY for the model's verb prediction? If we ablate them
     (zero their contribution at L*), does the model's verb prediction
     degrade?"

Phase J/K showed SUFFICIENCY: patching a clean CHECK source residual at
L=14 in Llama flips 79% of illegal_FOLD targets to CHECK. A reviewer can
argue this is just "the heads happen to encode the signal" — sufficiency
doesn't imply necessity. The standard mech-interp counterpart is
NECESSITY: zero out the heads' contributions and measure how much the
verb prediction degrades on a clean baseline.

Implementation: leverages the existing `HiddenStatePatchAttnHeadSubset`
primitive with a zeros source residual. When the source per-head residual
is all zeros, replacing those heads' o_proj input slices with the zeros
effectively kills their contribution at the verb position.

Procedure:
  1. Load the enriched log, sample N targets from the bucket of interest.
  2. For each target:
     - Run BASELINE forward: record top-1 verb prediction.
     - Run ABLATED forward (zero specified heads at L*): record top-1.
  3. Aggregate: how many targets' verb prediction changes? Magnitude of
     logit shift on the verb? Per-head-set comparison?

Heads to test (default cells):
  Llama L=14:    {h5, h23, h24} (the triplet); {h2, h5, h23, h24} (quartet);
                  {h23} alone; control: {3 random non-dominant heads}
  Ministral L=16: {h22, h9, h15} (triplet); {h22} alone
  Qwen L=23:     {h26, h30, h28} (top 3); {h26} alone

Usage::

    python -m experiments.head_ablation \\
        --enriched-log <pooled> \\
        --layer 14 \\
        --target-bucket clean_check_or_call \\
        --head-sets "5 23 24" "2 5 23 24" "23" "0 1 2" \\
        --n-target 50 \\
        --out-dir results/head_ablation/llama8b_l14 \\
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

from analysis.recategorize_action_metadata import _reparse_one  # noqa: E402
from poker_env.interp.forward_helpers import (  # noqa: E402
    PromptReconstructor,
    build_input_text_for_action_verb_position,
    run_forward_at_last_position,
)
from poker_env.interp.patching import (  # noqa: E402
    HiddenStatePatchAttnHeadSubset, get_head_geometry,
)
from experiments.causal_patching import (  # noqa: E402
    BUCKET_NAMES, classify_decision, build_action_token_id_sets,
    score_logits, _iter_decisions, _load_agent_config,
)


def main():
    parser = argparse.ArgumentParser(description="Head ablation (necessity test).")
    parser.add_argument("--enriched-log", required=True, nargs="+")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--target-bucket", default="clean_check_or_call",
                        choices=BUCKET_NAMES)
    parser.add_argument("--head-sets", nargs="+", required=True,
                        help="Each entry is a space-separated head index "
                             "list, e.g. '5 23 24'. Quote each set!")
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
    print(f"[init] model_id={model_id}, layer={args.layer}")

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
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=args.device,
    )
    model.eval()
    print(f"[init] model loaded in {time.time() - t0:.1f}s")

    num_heads, head_dim = get_head_geometry(model)
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
    targets = rng.sample(targets_pool, min(args.n_target, len(targets_pool)))

    def _prepare(rec: dict):
        full_prompt = recon.build(rec)
        out = build_input_text_for_action_verb_position(
            full_prompt, rec["action_metadata"]["raw_response"], tokenizer,
        )
        if out is None:
            return None
        input_text, verb_resp_idx = out
        resp_tok_ids = tokenizer(rec["action_metadata"]["raw_response"],
                                 add_special_tokens=False)["input_ids"]
        if not (0 <= verb_resp_idx < len(resp_tok_ids)):
            return None
        return {
            "rec": rec,
            "input_text": input_text,
            "expected_verb_tok_id": int(resp_tok_ids[verb_resp_idx]),
        }
    targets_prep = [p for p in (_prepare(r) for r in targets) if p is not None]
    print(f"[prepare] {len(targets_prep)} targets")

    # Parse head sets.
    head_sets = []
    for s in args.head_sets:
        idxs = sorted(set(int(x) for x in s.split()))
        if not idxs:
            continue
        for h in idxs:
            if not (0 <= h < num_heads):
                print(f"[abort] head idx {h} out of range [0, {num_heads})",
                      file=sys.stderr)
                sys.exit(2)
        head_sets.append(idxs)
    print(f"[init] {len(head_sets)} head sets to ablate: "
          + ", ".join(str(s) for s in head_sets))

    # Zero per-head residual (template; we'll create one with proper dtype).
    zero_per_head = torch.zeros((num_heads, head_dim), dtype=torch.float32)

    # ---- Run baselines + ablations ----------------------------------------
    by_pair_path = out_dir / "by_target.csv"
    by_pair_f = open(by_pair_path, "w", newline="")
    writer = csv.writer(by_pair_f)
    writer.writerow([
        "target_idx", "hand_id", "decision_idx",
        "head_set", "mode",
        "top1_id", "top1_tok", "top1_group",
        "delta_check_minus_fold",
        "GROUP_CHECK_CALL", "GROUP_FOLD", "GROUP_BET_RAISE",
    ])

    # Aggregator: per (head_set, target_idx) -> baseline + ablated outcomes.
    agg = {tuple(s): {"flip_count": 0, "delta_sum": 0.0, "n": 0,
                      "top1_changed_count": 0,
                      "verb_predicted_count_ablated": 0,
                      "verb_predicted_count_baseline": 0}
           for s in head_sets}

    print(f"\n[main] running baseline + ablation for {len(targets_prep)} targets ...")
    for ti, tgt in enumerate(targets_prep):
        # Baseline forward.
        base = run_forward_at_last_position(
            model, tokenizer, tgt["input_text"], device=args.device,
        )
        base_logits = base["logits_last_pos"]
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
            ti, tgt["rec"]["hand_id"], tgt["rec"]["decision_idx"],
            "", "baseline",
            base_top1, tokenizer.decode([base_top1]), base_group,
            f"{base_scored['delta_check_minus_fold']:.4f}",
            f"{base_scored['GROUP_CHECK_CALL']:.4f}",
            f"{base_scored['GROUP_FOLD']:.4f}",
            f"{base_scored['GROUP_BET_RAISE']:.4f}",
        ])

        # Per head-set ablation.
        for hset in head_sets:
            patch = HiddenStatePatchAttnHeadSubset(
                model, args.layer, zero_per_head, hset,
            )
            with patch:
                abl = run_forward_at_last_position(
                    model, tokenizer, tgt["input_text"], device=args.device,
                )
            abl_logits = abl["logits_last_pos"]
            abl_top1 = int(abl_logits.argmax().item())
            abl_scored = score_logits(abl_logits, family_ids)
            abl_group = "OTHER"
            for fam, ids in family_ids.items():
                if abl_top1 in ids:
                    if fam == "FOLD": abl_group = "FOLD"
                    elif fam in ("CHECK", "CALL"): abl_group = "CHECK_CALL"
                    elif fam in ("BET", "RAISE"): abl_group = "BET_RAISE"
                    break
            delta = (abl_scored["delta_check_minus_fold"]
                     - base_scored["delta_check_minus_fold"])
            writer.writerow([
                ti, tgt["rec"]["hand_id"], tgt["rec"]["decision_idx"],
                "+".join(str(h) for h in hset), "ablated",
                abl_top1, tokenizer.decode([abl_top1]), abl_group,
                f"{delta:.4f}",
                f"{abl_scored['GROUP_CHECK_CALL']:.4f}",
                f"{abl_scored['GROUP_FOLD']:.4f}",
                f"{abl_scored['GROUP_BET_RAISE']:.4f}",
            ])
            d = agg[tuple(hset)]
            d["n"] += 1
            d["delta_sum"] += delta
            if abl_top1 != base_top1:
                d["top1_changed_count"] += 1
            if base_top1 == tgt["expected_verb_tok_id"]:
                d["verb_predicted_count_baseline"] += 1
            if abl_top1 == tgt["expected_verb_tok_id"]:
                d["verb_predicted_count_ablated"] += 1
            # "Flip" = top-1 family changed (not just token id).
            if abl_group != base_group:
                d["flip_count"] += 1

        if (ti + 1) % 10 == 0:
            print(f"  [{ti+1}/{len(targets_prep)}] processed")

    by_pair_f.close()

    # Summary.
    summary = {
        "model_id": model_id,
        "layer": args.layer,
        "target_bucket": args.target_bucket,
        "n_target": len(targets_prep),
        "head_sets": [list(s) for s in head_sets],
        "per_head_set": {},
    }
    for hset in head_sets:
        d = agg[tuple(hset)]
        n = max(d["n"], 1)
        summary["per_head_set"]["+".join(str(h) for h in hset)] = {
            "heads": list(hset),
            "n": d["n"],
            "mean_delta_check_minus_fold": d["delta_sum"] / n,
            "frac_top1_id_changed": d["top1_changed_count"] / n,
            "frac_top1_group_changed": d["flip_count"] / n,
            "frac_verb_predicted_baseline": d["verb_predicted_count_baseline"] / n,
            "frac_verb_predicted_ablated": d["verb_predicted_count_ablated"] / n,
        }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = ["# Head ablation (necessity test) results", ""]
    md.append(f"- Model: `{model_id}`")
    md.append(f"- Layer: **{args.layer}**")
    md.append(f"- Target bucket: `{args.target_bucket}`")
    md.append(f"- n_target: {len(targets_prep)}")
    md.append("")
    md.append("Each row is one head set ablated (zeroed at the verb position).")
    md.append("`Δ(CHECK − FOLD)` is the change vs the baseline forward (negative "
              "= more FOLD-leaning under ablation).")
    md.append("`top-1 family changed` is the fraction of targets whose top-1 "
              "family (CHECK_CALL / FOLD / BET_RAISE / OTHER) changed under "
              "ablation. Higher = more disruptive.")
    md.append("`verb predicted (ablated)` is the fraction of targets whose "
              "ablated top-1 still matches the recorded verb. Lower = bigger "
              "necessity finding (heads matter for the verb prediction).")
    md.append("")
    md.append("| Head set | n | mean Δ(CHECK − FOLD) | top-1 family changed | verb predicted (baseline → ablated) |")
    md.append("|---|---:|---:|---:|---:|")
    for hset in head_sets:
        d = summary["per_head_set"]["+".join(str(h) for h in hset)]
        md.append(
            f"| `{'+'.join(str(h) for h in hset)}` | {d['n']} "
            f"| {d['mean_delta_check_minus_fold']:+.3f} "
            f"| **{d['frac_top1_group_changed']*100:.1f}%** "
            f"| {d['frac_verb_predicted_baseline']*100:.1f}% → "
            f"**{d['frac_verb_predicted_ablated']*100:.1f}%** |"
        )
    md.append("")
    md.append("## Reading guide")
    md.append("- **Verb-predicted-baseline ≫ Verb-predicted-ablated**: the heads "
              "are necessary for the verb prediction. Strong necessity finding.")
    md.append("- **Top-1 family changed ≥ 50% AND mean Δ in expected sign**: "
              "ablation reliably shifts the model's verb-distribution toward "
              "the alternative family. Cite the specific heads.")
    md.append("- **A control set of random heads showing 0% family change AND "
              "near-zero Δ**: confirms the necessity finding is specific to "
              "the dominant heads, not generic to ablating any heads.")
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[done] wrote {out_dir / 'SUMMARY.md'}")
    print(f"[done] wrote {out_dir / 'summary.json'}")
    print(f"[done] wrote {by_pair_path}")


if __name__ == "__main__":
    main()
