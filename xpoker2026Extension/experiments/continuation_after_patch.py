"""
Continuation test after L* activation patching.

For each target decision (default: ``illegal_fold``), compares:

  1. **recorded** — the log's ``raw_response`` (no re-generation).
  2. **regenerate_baseline** — full CoT action ``generate()`` without hooks.
  3. **regenerate_ablated** — full ``generate()`` with L* head zero-ablation
     on every forward pass (behavioral necessity during decoding).
  4. **patch_verb_then_continue** — single forward at the action-verb prefix
     with each sampled CHECK source residual patched at L*, argmax-sample the
     verb, then greedy-decode the remainder **without** further patching.
     One row per (source, target) pair.

Classifies each output: ``coherent_cot_json``, ``valid_json_no_reasoning``,
``broken_json``, ``empty``.

Outputs qualitative examples in ``examples.jsonl`` + aggregate rates in
``SUMMARY.md``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_env.agents.json_utils import extract_json, parse_cot_response  # noqa: E402
from poker_env.agents.prompts import get_action_system_message  # noqa: E402
from poker_env.interp.forward_helpers import (  # noqa: E402
    PromptReconstructor,
    assemble_chat_prompt,
    build_input_text_for_action_verb_position,
    find_action_verb_response_offset,
    run_forward_at_last_position,
    attached_hooks,
)
from poker_env.interp.generation_ablation import AttnHeadZeroAblation  # noqa: E402
from poker_env.interp.patching import HiddenStateCapture, HiddenStatePatch  # noqa: E402
from experiments.causal_patching import (  # noqa: E402
    classify_decision,
    _iter_decisions,
    _load_agent_config,
)


def _classify_response(raw: str, *, cot_mode: bool) -> str:
    if not raw or not raw.strip():
        return "empty"
    # Reject prompt-template leakage misclassified as model output.
    leak_markers = (
        "You are a poker player",
        "Analyze the situation step-by-step",
        'Format: {"action":',
        "Example:",
        "First write REASONING:",
    )
    if any(m in raw for m in leak_markers):
        return "broken_json"
    if cot_mode:
        reasoning, parsed = parse_cot_response(raw)
        if parsed and "action" in parsed:
            if reasoning and len(reasoning.strip()) >= 40:
                return "coherent_cot_json"
            return "valid_json_no_reasoning"
        return "broken_json"
    parsed = extract_json(raw)
    if parsed and "action" in parsed:
        return "valid_json_no_reasoning"
    return "broken_json"


def _greedy_continue(
    model,
    tokenizer,
    prefix_ids: list[int],
    max_new_tokens: int,
    device,
) -> list[int]:
    """Greedy decode from prefix_ids (no KV cache — fine for small n)."""
    ids = list(prefix_ids)
    for _ in range(max_new_tokens):
        enc = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model(input_ids=enc)
        next_id = int(out.logits[0, -1, :].argmax().item())
        ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
    return ids[len(prefix_ids):]


def _capture_residual_at_layer(
    model,
    tokenizer,
    recon: PromptReconstructor,
    src_rec: dict,
    layer: int,
    device: str,
) -> torch.Tensor:
    src_prompt = recon.build(src_rec)
    src_inp = build_input_text_for_action_verb_position(
        src_prompt, src_rec["action_metadata"]["raw_response"], tokenizer,
    )
    if src_inp is None:
        raise ValueError("cannot locate source verb")
    src_text, _ = src_inp
    cap = HiddenStateCapture(model)
    cap.attach_hooks()
    with torch.no_grad():
        model(tokenizer(src_text, return_tensors="pt").input_ids.to(device))
    res = cap.collect()["per_layer_last_pos"][layer]
    cap.detach_hooks()
    return res


def main():
    parser = argparse.ArgumentParser(description="Continuation after L* patch.")
    parser.add_argument("--enriched-log", required=True, nargs="+")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--source-bucket", default="clean_check_or_call")
    parser.add_argument("--target-bucket", default="illegal_fold")
    parser.add_argument("--n-target", type=int, default=25)
    parser.add_argument("--n-source", type=int, default=5)
    parser.add_argument("--head-indices", nargs="+", type=int, default=None,
                        help="Heads to zero during regenerate_ablated "
                             "(default: model triplet)")
    parser.add_argument(
        "--continue-tokens", type=int, default=80,
        help="Greedy continuation length after patched verb (default 80; "
             "use 180 for paper figure)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.inference_head_ablation import DEFAULT_HEAD_SETS, _short_model_name

    agent_config = _load_agent_config(args.enriched_log[0])
    model_id = agent_config["model_id"]
    cot_mode = bool(agent_config.get("cot_mode", True))
    short = _short_model_name(model_id)
    head_idxs = args.head_indices
    if head_idxs is None:
        head_idxs = DEFAULT_HEAD_SETS.get(short, {}).get("triplet", [0])

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[init] loading {model_id} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=args.device,
    )
    model.eval()
    print(f"[init] loaded in {time.time() - t0:.1f}s")

    recon = PromptReconstructor(tokenizer, agent_config)
    rng = random.Random(args.seed)

    by_bucket: dict[str, list] = {}
    for log in args.enriched_log:
        for rec in _iter_decisions(log):
            am = rec.get("action_metadata")
            if not am or not am.get("raw_response"):
                continue
            by_bucket.setdefault(classify_decision(rec), []).append(rec)

    sources = by_bucket.get(args.source_bucket, [])
    targets = by_bucket.get(args.target_bucket, [])
    if not sources or not targets:
        print("[abort] empty source or target bucket", file=sys.stderr)
        sys.exit(2)
    sources = rng.sample(sources, min(args.n_source, len(sources)))
    targets = rng.sample(targets, min(args.n_target, len(targets)))

    # Capture ALL sampled source residuals at L*.
    source_residuals: list[torch.Tensor] = []
    for si, src_rec in enumerate(sources):
        try:
            res = _capture_residual_at_layer(
                model, tokenizer, recon, src_rec, args.layer, args.device,
            )
        except ValueError as e:
            print(f"[warn] skip source {si}: {e}")
            continue
        source_residuals.append(res)
    if not source_residuals:
        print("[abort] no valid source residuals", file=sys.stderr)
        sys.exit(2)
    print(f"[init] captured {len(source_residuals)} source residuals at L={args.layer}")

    system_message = get_action_system_message(cot=cot_mode)
    examples = []
    mode_counts: dict[str, Counter] = {
        "recorded": Counter(),
        "regenerate_baseline": Counter(),
        "regenerate_ablated": Counter(),
        "patch_verb_then_continue": Counter(),
    }

    ablation = AttnHeadZeroAblation(model, args.layer, head_idxs)

    def _full_generate(chat_prompt: str, hooks: list | None) -> str:
        enc = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(args.device)
        gen_kw = dict(
            max_new_tokens=args.continue_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        owners = hooks or []
        with attached_hooks(owners):
            with torch.no_grad():
                out = model.generate(input_ids=input_ids, **gen_kw)
        new_ids = out[0, input_ids.shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    for ti, tgt in enumerate(targets):
        recorded = tgt["action_metadata"]["raw_response"]
        user_prompt = recon.build_user_prompt(tgt)
        full_chat = assemble_chat_prompt(
            user_prompt, system_message, tokenizer=tokenizer,
            supports_system_role=recon.supports_system_role,
            has_thinking_mode=recon.has_thinking_mode,
        )
        chat_prompt = recon.build(tgt)

        regen_base = _full_generate(full_chat, None)
        regen_abl = _full_generate(full_chat, [ablation])

        rec_class = _classify_response(recorded, cot_mode=cot_mode)
        base_class = _classify_response(regen_base, cot_mode=cot_mode)
        abl_class = _classify_response(regen_abl, cot_mode=cot_mode)
        mode_counts["recorded"][rec_class] += 1
        mode_counts["regenerate_baseline"][base_class] += 1
        mode_counts["regenerate_ablated"][abl_class] += 1

        verb_found = find_action_verb_response_offset(recorded, tokenizer)
        verb_inp = build_input_text_for_action_verb_position(
            chat_prompt, recorded, tokenizer,
        )
        if verb_inp is None or verb_found is None:
            for si in range(len(source_residuals)):
                examples.append({
                    "target_idx": ti,
                    "source_idx": si,
                    "hand_id": tgt.get("hand_id"),
                    "seed": tgt.get("seed"),
                    "recorded_class": rec_class,
                    "regenerate_baseline_class": base_class,
                    "regenerate_ablated_class": abl_class,
                    "patch_continue_class": "broken_json",
                    "patch_continue_error": "verb_not_found",
                })
                mode_counts["patch_verb_then_continue"]["broken_json"] += 1
            continue

        prefix_text, _verb_idx = verb_inp
        verb_char_offset, _ = verb_found
        response_before_verb = recorded[:verb_char_offset]
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]

        for si, src_res in enumerate(source_residuals):
            with HiddenStatePatch(model, args.layer, src_res):
                fwd = run_forward_at_last_position(
                    model, tokenizer, prefix_text, device=args.device,
                )
            verb_id = int(fwd["logits_last_pos"].argmax().item())
            cont_ids = _greedy_continue(
                model, tokenizer, prefix_ids + [verb_id],
                args.continue_tokens, args.device,
            )
            # Classify assistant response only: recorded reasoning prefix +
            # model-predicted verb + greedy continuation (NOT full chat prompt).
            suffix = tokenizer.decode(
                [verb_id] + cont_ids, skip_special_tokens=True,
            )
            patch_cont = response_before_verb + suffix
            patch_class = _classify_response(patch_cont, cot_mode=cot_mode)
            mode_counts["patch_verb_then_continue"][patch_class] += 1

            examples.append({
                "target_idx": ti,
                "source_idx": si,
                "hand_id": tgt.get("hand_id"),
                "seed": tgt.get("seed"),
                "recorded_class": rec_class,
                "regenerate_baseline_class": base_class,
                "regenerate_ablated_class": abl_class,
                "patch_continue_class": patch_class,
                "recorded_snippet": recorded[:400],
                "regenerate_baseline_snippet": regen_base[:400],
                "regenerate_ablated_snippet": regen_abl[:400],
                "patch_continue_snippet": patch_cont[:400],
            })

    with open(out_dir / "examples.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    n_tgt = len(targets)
    n_patch = len(source_residuals) * n_tgt
    summary = {
        "model_id": model_id,
        "layer": args.layer,
        "source_bucket": args.source_bucket,
        "target_bucket": args.target_bucket,
        "n_target": n_tgt,
        "n_source": len(source_residuals),
        "n_patch_pairs": n_patch,
        "continue_tokens": args.continue_tokens,
        "head_indices_ablated": head_idxs,
        "class_counts": {k: dict(v) for k, v in mode_counts.items()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    def _pct(c: Counter, denom: int) -> dict[str, float]:
        if denom <= 0:
            return {}
        return {k: v / denom for k, v in c.items()}

    md = [
        "# Continuation after L* patch",
        "",
        f"- Model: `{model_id}`",
        f"- Layer: **{args.layer}**",
        f"- Source bucket: `{args.source_bucket}` → target: `{args.target_bucket}`",
        f"- n_targets: {n_tgt} | n_sources: {len(source_residuals)} | "
        f"patch pairs: {n_patch}",
        f"- Ablated heads (regenerate_ablated): `{head_idxs}`",
        f"- Continue tokens (greedy after verb): {args.continue_tokens}",
        "",
        "## Response quality",
        "",
        "| Mode | denominator | coherent CoT+JSON | valid JSON only | broken | empty |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, label, denom in [
        ("recorded", "Recorded log", n_tgt),
        ("regenerate_baseline", "Full regenerate", n_tgt),
        ("regenerate_ablated", "Full regenerate + ablation", n_tgt),
        ("patch_verb_then_continue", "Patch verb + continue (× sources)", n_patch),
    ]:
        c = mode_counts[mode]
        md.append(
            f"| {label} | {denom} | "
            f"{c.get('coherent_cot_json', 0) / denom * 100:.0f}% "
            f"| {c.get('valid_json_no_reasoning', 0) / denom * 100:.0f}% "
            f"| {c.get('broken_json', 0) / denom * 100:.0f}% "
            f"| {c.get('empty', 0) / denom * 100:.0f}% |"
        )
    md.append("")
    md.append("## Interpretation")
    md.append(
        "- **patch_verb_then_continue** aggregates over every (source × target) "
        "pair — not a single source residual."
    )
    md.append(
        "- If patch-continue is mostly `broken_json` but regenerate_ablated is "
        "`coherent_cot_json`, the one-forward patch does not sustain global "
        "coherence (circuit acts during full decoding)."
    )
    md.append("")
    md.append("See `examples.jsonl` (one row per source×target for patch mode).")
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[done] {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
