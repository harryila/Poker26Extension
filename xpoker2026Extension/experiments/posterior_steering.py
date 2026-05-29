"""
Posterior steering — can the calibration direction DE-BIAS the decision? (GPU; Tier-1 C2)

Tests whether ADDING the residual direction that encodes the oracle's neglected 'trash' mass
(produced by experiments/encode_vs_decode.py --save-direction) causally shifts the agent's
action away from its base-rate-neglect bias. This is the controllable-intervention payoff and
positions the work against the steering-vector literature (it is a TASK-GROUNDED, causally
validated steer, not a generic concept vector).

Method (reuses the exact forward/scoring infra; ActivationAdditionHook adds instead of patches):
  * sample target decisions (default clean_legal_fold — spots where the model folds);
  * read the baseline CHECK_CALL−FOLD logit gap at the verb position;
  * for each alpha, re-read it with the steering vector added at L* (alpha measured in units
    of the mean residual norm), and report the mean shift + top-1 flip rate.

A positive CHECK−FOLD shift = steering toward 'opponent-likely-weak' makes the agent fold less
(de-biasing the over-fold tendency). The SIGN/size are empirical — the alpha sweep maps the
dose-response and includes a NEGATIVE-control direction (random unit vector, same norm).

Usage (GPU):
    python -m experiments.posterior_steering \
        --enriched-log logs/cot_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz ... \
        --layer 23 --direction results/direction_probe/qwen8b_l23/steer_trash_direction.npz \
        --alphas 0 2 4 8 --target-bucket clean_legal_fold --n-decisions 60 \
        --device cuda --dtype bfloat16 \
        --out-dir results/posterior_steering/qwen8b_l23
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from poker_env.interp.forward_helpers import (
    PromptReconstructor,
    build_input_text_for_action_verb_position,
    run_forward_at_last_position,
)
from poker_env.interp.patching import ActivationAdditionHook
from experiments.causal_patching import (
    _load_agent_config, _iter_decisions, classify_decision,
    build_action_token_id_sets, score_logits,
)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _score(model, tokenizer, text, fam, device):
    res = run_forward_at_last_position(model, tokenizer, text, device=device)
    s = score_logits(res["logits_last_pos"], fam)
    logits = res["logits_last_pos"]
    top1 = int(logits.argmax())
    grp = "OTHER"
    for g, ids in fam.items():
        if top1 in ids:
            grp = g
    return s["delta_check_minus_fold"], grp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enriched-log", nargs="+", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--direction", required=True, help="npz with 'direction' (+ resid_mean_norm)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 2, 4, 8])
    ap.add_argument("--target-bucket", default="clean_legal_fold")
    ap.add_argument("--n-decisions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(_DTYPES))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    dirnpz = np.load(args.direction, allow_pickle=True)
    base_vec = torch.tensor(dirnpz["direction"].astype(np.float32))   # unit norm
    norm = float(dirnpz["resid_mean_norm"]) if "resid_mean_norm" in dirnpz else 1.0
    # negative control: a random unit direction (same scaling)
    g = torch.Generator().manual_seed(args.seed)
    rand_vec = torch.randn(base_vec.shape, generator=g)
    rand_vec = rand_vec / rand_vec.norm()

    agent_config = _load_agent_config(args.enriched_log[0])
    model_id = args.model_id or agent_config["model_id"]
    print(f"[load] {model_id}  layer={args.layer}  resid_norm≈{norm:.1f}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=_DTYPES[args.dtype], device_map=args.device)
    model.eval()
    recon = PromptReconstructor(tokenizer, agent_config)
    fam = build_action_token_id_sets(tokenizer)

    pool = []
    for log in args.enriched_log:
        for rec in _iter_decisions(log):
            am = rec.get("action_metadata")
            if not isinstance(am, dict) or not am.get("raw_response"):
                continue
            if classify_decision(rec) != args.target_bucket:
                continue
            built = build_input_text_for_action_verb_position(recon.build(rec), am["raw_response"], tokenizer)
            if built:
                pool.append(built[0])
    rng.shuffle(pool)
    pool = pool[:args.n_decisions]
    print(f"[pool] {len(pool)} {args.target_bucket} targets")

    def sweep(vec, label):
        rows = []
        for text in pool:
            base_delta, base_grp = _score(model, tokenizer, text, fam, args.device)
            row = {"baseline_delta": base_delta, "baseline_top1": base_grp, "by_alpha": {}}
            for a in args.alphas:
                if a == 0:
                    row["by_alpha"][str(a)] = {"delta": base_delta, "top1": base_grp}
                    continue
                hook = ActivationAdditionHook(model, args.layer, vec * norm, alpha=float(a), last_only=True)
                with hook, torch.no_grad():
                    d, grp = _score(model, tokenizer, text, fam, args.device)
                row["by_alpha"][str(a)] = {"delta": d, "top1": grp}
            rows.append(row)
        return rows

    os.makedirs(args.out_dir, exist_ok=True)
    out = {"model_id": model_id, "layer": args.layer, "target_bucket": args.target_bucket,
           "alphas": args.alphas, "n": len(pool), "norm": norm, "conditions": {}}
    for vec, label in [(base_vec, "trash_direction"), (rand_vec, "random_control")]:
        rows = sweep(vec, label)
        with open(os.path.join(args.out_dir, f"{label}_rows.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        agg = {}
        for a in args.alphas:
            ds = [r["by_alpha"][str(a)]["delta"] for r in rows]
            flips = sum(1 for r in rows if r["by_alpha"][str(a)]["top1"] == "CHECK_CALL")
            agg[str(a)] = {"mean_delta_check_minus_fold": float(np.mean(ds)),
                           "frac_top1_check_call": flips / len(rows)}
        out["conditions"][label] = agg

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    md = ["# Posterior steering — does the calibration direction de-bias the decision?", "",
          f"- model={model_id} layer={args.layer} target={args.target_bucket} n={len(pool)}",
          "- alpha in units of mean residual norm; `trash_direction` vs `random_control`.", "",
          "| alpha | trash: mean ΔCHECK−FOLD | trash: top-1→CHECK | control: mean Δ | control: top-1→CHECK |",
          "|---:|---:|---:|---:|---:|"]
    for a in args.alphas:
        t = out["conditions"]["trash_direction"][str(a)]
        c = out["conditions"]["random_control"][str(a)]
        md.append(f"| {a} | {t['mean_delta_check_minus_fold']:+.2f} | {t['frac_top1_check_call']*100:.0f}% | "
                  f"{c['mean_delta_check_minus_fold']:+.2f} | {c['frac_top1_check_call']*100:.0f}% |")
    md += ["", "## Reading",
           "- A monotone CHECK−FOLD increase with alpha for `trash_direction` that EXCEEDS the "
           "`random_control` = the calibration direction causally de-biases the over-fold "
           "tendency (controllable steering). If trash≈control, the effect is generic norm "
           "perturbation, not the direction."]
    with open(os.path.join(args.out_dir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[written] {args.out_dir}/SUMMARY.md")


if __name__ == "__main__":
    main()
