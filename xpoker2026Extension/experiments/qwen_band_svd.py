"""
Singular-vector view of Qwen's distributed compute band L18-20 (Tier-2 D1; GPU).

The necessity sweep found Qwen's verb decision is computed by a DISTRIBUTED attention
sub-network across L18-20 (no sparse head; largest single head ~17%). Per Ahmad et al. 2025
(arXiv 2511.20273), "distributed across heads" need not mean "high rank": the signal may live
in a few singular directions that cut across heads. This script SVDs the residual at each band
layer over a decision pool and reports (a) how low-rank the verb-relevant variance is and
(b) how the learned decision direction aligns with the top singular vectors — quantifying
"distributed across heads but low-rank in activation space."

Reuses HiddenStateCapture (known-correct last-position residual capture). The per-HEAD
singular decomposition (o_proj-input level) is a documented follow-up; this layer-level
version is the defensible first cut.

Usage (GPU):
    python -m experiments.qwen_band_svd \
        --enriched-log logs/cot_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz ... \
        --layers 18 19 20 23 --bucket clean_check_or_call --n 150 \
        --direction results/direction_probe/qwen8b_l23/raw_residuals.npz \
        --device cuda --dtype bfloat16 --out results/causal_patching/qwen_band_svd.md
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from poker_env.interp.forward_helpers import (
    PromptReconstructor, build_input_text_for_action_verb_position)
from poker_env.interp.patching import HiddenStateCapture
from experiments.causal_patching import _load_agent_config, _iter_decisions, classify_decision

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enriched-log", nargs="+", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 19, 20, 23])
    ap.add_argument("--bucket", default="clean_check_or_call")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--direction", default=None, help="npz with weight_vec (decision dir) to project")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(_DTYPES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    agent_config = _load_agent_config(args.enriched_log[0])
    model_id = args.model_id or agent_config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=_DTYPES[args.dtype], device_map=args.device)
    model.eval()
    recon = PromptReconstructor(tokenizer, agent_config)
    cap = HiddenStateCapture(model)

    texts = []
    for log in args.enriched_log:
        for rec in _iter_decisions(log):
            am = rec.get("action_metadata")
            if not isinstance(am, dict) or not am.get("raw_response"):
                continue
            if classify_decision(rec) != args.bucket:
                continue
            built = build_input_text_for_action_verb_position(recon.build(rec), am["raw_response"], tokenizer)
            if built:
                texts.append(built[0])
            if len(texts) >= args.n:
                break
        if len(texts) >= args.n:
            break

    by_layer = {L: [] for L in args.layers}
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
        cap.attach_hooks()
        try:
            with torch.no_grad():
                model(input_ids=ids)
        finally:
            states = cap.collect()
            cap.detach_hooks()
        for L in args.layers:
            r = states["per_layer_last_pos"].get(L)
            if r is not None:
                by_layer[L].append(r.float().cpu().numpy())

    wdir = None
    if args.direction and os.path.exists(args.direction):
        wd = np.load(args.direction, allow_pickle=True)
        if "weight_vec" in wd:
            wdir = wd["weight_vec"].astype(np.float64)
            wdir = wdir / (np.linalg.norm(wdir) + 1e-12)

    md = ["# Qwen band SVD — distributed but low-rank?", "",
          f"- model={model_id} bucket={args.bucket} n={len(texts)}", ""]
    md.append("| layer | n | rank for 90% var | top-1 SV energy | cos(decision dir, top-3 SVs) |")
    md.append("|---:|---:|---:|---:|---|")
    for L in args.layers:
        M = np.asarray(by_layer[L], float)
        if len(M) < 5:
            md.append(f"| {L} | {len(M)} | (too few) | | |")
            continue
        Mc = M - M.mean(0)
        U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
        energy = (S ** 2) / (S ** 2).sum()
        cum = np.cumsum(energy)
        rank90 = int(np.searchsorted(cum, 0.90) + 1)
        cos_str = "—"
        if wdir is not None:
            cos3 = [abs(float(Vt[k] @ wdir)) for k in range(min(3, len(Vt)))]
            cos_str = ", ".join(f"{c:.2f}" for c in cos3)
        md.append(f"| {L} | {len(M)} | {rank90} | {energy[0]*100:.0f}% | {cos_str} |")
    md += ["", "## Reading",
           "- Low `rank for 90% var` at L18-20 = the distributed-across-heads computation is "
           "nonetheless LOW-RANK in activation space (few directions carry it).",
           "- High cos(decision dir, top SVs) = the learned verb direction lives in that "
           "low-rank band — reconciling 'distributed heads' with 'a single usable direction'.",
           "- Follow-up: repeat at the o_proj-input (per-head) level to attribute the singular "
           "directions to head groups (Ahmad et al. 2025)."]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
