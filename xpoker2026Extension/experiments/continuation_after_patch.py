"""
Continuation test after L* activation patching.

For each target decision (default: ``illegal_fold``), compares:

  1. **recorded** — the log's ``raw_response`` (no re-generation).
  2. **regenerate_baseline** — full CoT action ``generate()`` without hooks.
  3. **regenerate_ablated** — full ``generate()`` with L* head zero-ablation
     on every forward pass (behavioral necessity during decoding).
  4. **patch_verb_then_continue** — single forward at the action-verb prefix
     with a CHECK source residual patched at L*, argmax-sample the verb
     token(s), then greedy-decode the remainder **without** further patching.

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
    """Greedy decode from prefix_ids (no KV cache — small n)."""
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
    parser.add_argument("--continue-tokens", type=int, default=180)
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

    # Capture one source residual at L*.
    src_rec = sources[0]
    src_prompt = recon.build(src_rec)
    src_inp = build_input_text_for_action_verb_position(
        src_prompt, src_rec["action_metadata"]["raw_response"], tokenizer,
    )
    if src_inp is None:
        print("[abort] cannot locate source verb", file=sys.stderr)
        sys.exit(2)
    src_text, _ = src_inp
    cap = HiddenStateCapture(model)
    cap.attach_hooks()
    with torch.no_grad():
        model(tokenizer(src_text, return_tensors="pt").input_ids.to(args.device))
    cap_states = cap.collect()["per_layer_last_pos"]
    cap.detach_hooks()
    src_res = cap_states[args.layer]

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
        chat_prompt = recon.build(tgt)
        user_prompt = recon.build_user_prompt(tgt)

        # Full chat templates for generate
        full_chat = assemble_chat_prompt(
            user_prompt, system_message, tokenizer=tokenizer,
            supports_system_role=recon.supports_system_role,
            has_thinking_mode=recon.has_thinking_mode,
        )

        regen_base = _full_generate(full_chat, None)
        regen_abl = _full_generate(full_chat, [ablation])

        # Patch verb then continue
        tgt_prompt = chat_prompt
        verb_inp = build_input_text_for_action_verb_position(
            tgt_prompt, recorded, tokenizer,
        )
        patch_cont = ""
        patch_class = "broken_json"
        if verb_inp is not None:
            prefix_text, _verb_idx = verb_inp
            prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            with HiddenStatePatch(model, args.layer, src_res):
                fwd = run_forward_at_last_position(
                    model, tokenizer, prefix_text, device=args.device,
                )
            verb_id = int(fwd["logits_last_pos"].argmax().item())
            cont_ids = _greedy_continue(
                model, tokenizer, prefix_ids + [verb_id],
                args.continue_tokens, args.device,
            )
            full_decoded = tokenizer.decode(
                prefix_ids + [verb_id] + cont_ids, skip_special_tokens=True,
            )
            patch_cont = (
                full_decoded[len(prefix_text):]
                if full_decoded.startswith(prefix_text)
                else full_decoded
            )
            patch_class = _classify_response(patch_cont, cot_mode=cot_mode)

        row = {
            "target_idx": ti,
            "hand_id": tgt.get("hand_id"),
            "seed": tgt.get("seed"),
            "recorded_class": _classify_response(recorded, cot_mode=cot_mode),
            "regenerate_baseline_class": _classify_response(regen_base, cot_mode=cot_mode),
            "regenerate_ablated_class": _classify_response(regen_abl, cot_mode=cot_mode),
            "patch_continue_class": patch_class,
            "recorded_snippet": recorded[:400],
            "regenerate_baseline_snippet": regen_base[:400],
            "regenerate_ablated_snippet": regen_abl[:400],
            "patch_continue_snippet": patch_cont[:400],
        }
        examples.append(row)
        mode_counts["recorded"][row["recorded_class"]] += 1
        mode_counts["regenerate_baseline"][row["regenerate_baseline_class"]] += 1
        mode_counts["regenerate_ablated"][row["regenerate_ablated_class"]] += 1
        mode_counts["patch_verb_then_continue"][row["patch_continue_class"]] += 1

    with open(out_dir / "examples.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    summary = {
        "model_id": model_id,
        "layer": args.layer,
        "source_bucket": args.source_bucket,
        "target_bucket": args.target_bucket,
        "n_target": len(targets),
        "head_indices_ablated": head_idxs,
        "class_counts": {k: dict(v) for k, v in mode_counts.items()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    n = len(examples) or 1
    md = [
        "# Continuation after L* patch",
        "",
        f"- Model: `{model_id}`",
        f"- Layer: **{args.layer}**",
        f"- Source bucket: `{args.source_bucket}` → target: `{args.target_bucket}`",
        f"- n_targets: {len(targets)}",
        f"- Ablated heads (regenerate_ablated): `{head_idxs}`",
        "",
        "## Response quality (fraction of targets)",
        "",
        "| Mode | coherent CoT+JSON | valid JSON only | broken | empty |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, label in [
        ("recorded", "Recorded log"),
        ("regenerate_baseline", "Full regenerate"),
        ("regenerate_ablated", "Full regenerate + head ablation"),
        ("patch_verb_then_continue", "Patch verb + greedy continue"),
    ]:
        c = mode_counts[mode]
        md.append(
            f"| {label} | {c.get('coherent_cot_json',0)/n*100:.0f}% "
            f"| {c.get('valid_json_no_reasoning',0)/n*100:.0f}% "
            f"| {c.get('broken_json',0)/n*100:.0f}% "
            f"| {c.get('empty',0)/n*100:.0f}% |"
        )
    md.append("")
    md.append("## Interpretation")
    md.append(
        "- If **patch_verb_then_continue** is mostly `broken_json` but "
        "**regenerate_ablated** is `coherent_cot_json`, the patch at the verb "
        "position alone does not sustain global coherence — the circuit acts "
        "during full decoding, not as a one-token logit hack."
    )
    md.append(
        "- If **regenerate_ablated** shifts illegal_FOLD targets toward "
        "coherent CHECK JSON vs baseline, head ablation changes the full "
        "decision process, not only the verb token."
    )
    md.append("")
    md.append("See `examples.jsonl` for side-by-side snippets.")
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[done] {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
