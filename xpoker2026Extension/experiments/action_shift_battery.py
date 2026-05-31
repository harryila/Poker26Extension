"""
Fold-vs-aggression battery: what does the L19 head set causally do to the action, on a FIXED,
fully-paired set of facing-bet spots? (GPU; settles the C3 tension with zero tree divergence.)

C3 (live gameplay) said L19 [31,3,21,1,0] is bet-suppression (CHECK->BET up, fold flat); the
inference-ablation (recorded fold pool) said it flips FOLD (57% to CHECK). The clean, unconfounded
test is a SINGLE-FORWARD readout at the verb position on a fixed battery of recorded facing-bet
decisions (bet_to_call>0), comparing the next-action top-1 group under:
  baseline | ablate(L19 heads) | steer(+decision direction)
No generation, no game-tree divergence -> every spot is identical across conditions, so the
top-1 transition matrix (FOLD/CHECK_CALL/BET_RAISE) is a pure causal readout.

Reuses run_forward_at_last_position + AttnHeadZeroAblation + ActivationAdditionHook + the exact
score_logits / family-id machinery (no reimplementation).

Usage (GPU):
    python -m experiments.action_shift_battery \
        --enriched-log logs/cot_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz ... \
        --ablate-layer 19 --ablate-heads 31 3 21 1 0 \
        --steer-layer 23 --steer-dir results/direction_probe/qwen8b_l23/steer_decision_direction.npz \
        --steer-alpha 0.75 --n 200 --device cuda --dtype bfloat16 \
        --out-dir results/action_shift/qwen8b_facingbet
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from poker_env.interp.forward_helpers import (
    PromptReconstructor, build_input_text_for_action_verb_position, run_forward_at_last_position)
from poker_env.interp.patching import ActivationAdditionHook
from poker_env.interp.generation_ablation import AttnHeadZeroAblation
from experiments.causal_patching import (
    _load_agent_config, _iter_decisions, classify_decision,
    build_action_token_id_sets, score_logits)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
GROUPS = ("FOLD", "CHECK_CALL", "BET_RAISE", "OTHER")


def _top1_group(logits, fam):
    t = int(logits.argmax())
    for g, ids in fam.items():
        if t in ids:
            return g
    return "OTHER"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enriched-log", nargs="+", required=True)
    ap.add_argument("--ablate-layer", type=int, default=19)
    ap.add_argument("--ablate-heads", type=int, nargs="+", default=[31, 3, 21, 1, 0])
    ap.add_argument("--steer-layer", type=int, default=23)
    ap.add_argument("--steer-dir", default=None, help="steer_decision_direction.npz (unit dir + gap norm)")
    ap.add_argument("--steer-alpha", type=float, default=0.75)
    ap.add_argument("--buckets", nargs="+", default=["clean_legal_fold", "clean_check_or_call"],
                    help="recorded buckets to include (facing-bet filter applied on top)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(_DTYPES))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ac = _load_agent_config(args.enriched_log[0])
    model_id = args.model_id or ac["model_id"]
    print(f"[load] {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=_DTYPES[args.dtype], device_map=args.device)
    model.eval()
    recon = PromptReconstructor(tok, ac)
    fam = build_action_token_id_sets(tok)

    # steering vector (optional)
    steer_vec = None; steer_norm = 1.0
    if args.steer_dir and os.path.exists(args.steer_dir):
        s = np.load(args.steer_dir, allow_pickle=True)
        steer_vec = torch.tensor(s["direction"].astype(np.float32))
        steer_norm = float(s["resid_mean_norm"]) if "resid_mean_norm" in s else 1.0

    # build the FIXED facing-bet battery
    pool = []
    for log in args.enriched_log:
        for rec in _iter_decisions(log):
            am = rec.get("action_metadata")
            ob = rec.get("obs")
            if not (isinstance(am, dict) and am.get("raw_response") and isinstance(ob, dict)):
                continue
            if (ob.get("bet_to_call") or 0) <= 0:           # facing-bet only
                continue
            if classify_decision(rec) not in args.buckets:
                continue
            built = build_input_text_for_action_verb_position(recon.build(rec), am["raw_response"], tok)
            if built:
                pool.append((built[0], classify_decision(rec)))
    rng.shuffle(pool)
    pool = pool[:args.n]
    print(f"[battery] {len(pool)} facing-bet spots")

    def measure(hook):
        rows = []
        for text, bucket in pool:
            if hook is None:
                res = run_forward_at_last_position(model, tok, text, device=args.device)
            else:
                with hook, torch.no_grad():
                    res = run_forward_at_last_position(model, tok, text, device=args.device)
            lg = res["logits_last_pos"]
            sc = score_logits(lg, fam)
            rows.append({"bucket": bucket, "top1": _top1_group(lg, fam),
                         "d_cf": sc["delta_check_minus_fold"]})
        return rows

    conds = {"baseline": None,
             "ablate": AttnHeadZeroAblation(model, args.ablate_layer, list(args.ablate_heads))}
    if steer_vec is not None:
        conds["steer"] = ActivationAdditionHook(model, args.steer_layer, steer_vec * steer_norm,
                                                alpha=float(args.steer_alpha), last_only=True)

    results = {c: measure(h) for c, h in conds.items()}
    os.makedirs(args.out_dir, exist_ok=True)
    for c, rows in results.items():
        with open(os.path.join(args.out_dir, f"{c}_rows.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    base = results["baseline"]
    md = ["# Action-shift battery (single-forward, fully paired facing-bet spots)", "",
          f"- model={model_id} n={len(pool)} ablate=L{args.ablate_layer}{list(args.ablate_heads)} "
          f"steer=L{args.steer_layer} alpha={args.steer_alpha}", ""]
    # top-1 group distribution per condition
    md.append("## Top-1 action-group distribution")
    md.append("| condition | FOLD | CHECK_CALL | BET_RAISE | mean Δ(check−fold) |")
    md.append("|---|---:|---:|---:|---:|")
    for c, rows in results.items():
        cnt = collections.Counter(r["top1"] for r in rows)
        n = len(rows)
        md.append(f"| {c} | {cnt['FOLD']/n*100:.0f}% | {cnt['CHECK_CALL']/n*100:.0f}% | "
                  f"{cnt['BET_RAISE']/n*100:.0f}% | {np.mean([r['d_cf'] for r in rows]):+.2f} |")
    md.append("")
    # transition matrices vs baseline (the causal readout)
    for c in conds:
        if c == "baseline":
            continue
        rows = results[c]
        trans = collections.Counter((base[i]["top1"], rows[i]["top1"]) for i in range(len(rows)))
        md.append(f"## Baseline → {c}  top-1 transitions (the causal effect)")
        md.append("| from\\to | FOLD | CHECK_CALL | BET_RAISE |")
        md.append("|---|---:|---:|---:|")
        for frm in GROUPS[:3]:
            md.append(f"| {frm} | " + " | ".join(str(trans.get((frm, to), 0)) for to in GROUPS[:3]) + " |")
        # headline: did FOLD spots go to CHECK or BET; did CHECK go to BET?
        fold_n = sum(trans.get(("FOLD", t), 0) for t in GROUPS)
        f2c = trans.get(("FOLD", "CHECK_CALL"), 0); f2b = trans.get(("FOLD", "BET_RAISE"), 0)
        cc_n = sum(trans.get(("CHECK_CALL", t), 0) for t in GROUPS)
        c2b = trans.get(("CHECK_CALL", "BET_RAISE"), 0)
        md.append("")
        md.append(f"- FOLD→CHECK {f2c}/{fold_n}, FOLD→BET {f2b}/{fold_n}  |  CHECK→BET {c2b}/{cc_n}")
        md.append(f"- **reads as {'FOLD-circuit (fold→check dominant)' if f2c>max(f2b,c2b) else ('aggression/bet-suppression (bet inflow dominant)' if (f2b+c2b)>f2c else 'mixed')}**")
        md.append("")
    md.append("## Reading")
    md.append("- This is single-forward and fully paired, so transitions are pure causal readouts "
              "(no game-tree divergence). FOLD→CHECK dominant under ablate ⇒ fold circuit; "
              "CHECK→BET / FOLD→BET dominant ⇒ aggression/bet-suppression (the C3 reinterpretation). "
              "The steer column tests whether the corrected decision direction moves top-1 toward CHECK.")
    open(os.path.join(args.out_dir, "SUMMARY.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[written] {args.out_dir}/SUMMARY.md")


if __name__ == "__main__":
    main()
