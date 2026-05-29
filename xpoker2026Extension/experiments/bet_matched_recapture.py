"""
GPU: re-capture residuals at L*, TAGGED with bet_to_call / verb / bucket, for the
bet-matched probe (the clean test of the audit crux — see CLAIMS_AND_IDENTIFICATION.md §3).

The committed `raw_residuals.npz` files store residual matrices bucketed (X_cc/X_lf/X_if)
with NO per-sample game-state metadata, so the verb↔`bet_to_call` confound cannot be broken
on CPU. This script does ONE forward pass per decision (same verb-predecessor position the
probes use), capturing the layer-L residual ALONGSIDE `bet_to_call`, the verb, and the
recorded bucket, into `raw_residuals_tagged.npz`. Then `experiments/bet_matched_probe.py`
(CPU) runs the regime-restricted and bet-balanced probes.

Reuses the exact infrastructure (no reimplementation): PromptReconstructor +
build_input_text_for_action_verb_position (verb position) + HiddenStateCapture (residual).

Usage (GPU box):
    python -m experiments.bet_matched_recapture \
        --enriched-log logs/cot_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz \
                       logs/cot_qwen8b_t0_s123_informative_v2_enriched.jsonl.gz \
                       logs/cot_qwen8b_t0_s456_informative_v2_enriched.jsonl.gz \
        --layer 23 --device cuda --dtype bfloat16 \
        --out results/direction_probe/qwen8b_l23/raw_residuals_tagged.npz
Then (CPU):
    python -m experiments.bet_matched_probe \
        --tagged results/direction_probe/qwen8b_l23/raw_residuals_tagged.npz \
        --out results/direction_probe_baselines/BET_MATCHED_qwen_l23.md
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from poker_env.interp.forward_helpers import (
    PromptReconstructor,
    build_input_text_for_action_verb_position,
)
from poker_env.interp.patching import HiddenStateCapture
from poker_env.config import BUCKET_ORDER
from experiments.causal_patching import (
    _load_agent_config,
    _iter_decisions,
    classify_decision,
)


def _belief_vec(d) -> "np.ndarray | None":
    """Convert an oracle/belief dict keyed by the 14 bucket names to a 14-vector."""
    if not isinstance(d, dict):
        return None
    return np.array([float(d.get(b) or 0.0) for b in BUCKET_ORDER], dtype=np.float32)

# verb for the bet-matched contrasts:
#   clean_check_or_call @ bet>0 -> CALL ; @ bet==0 -> CHECK
#   clean_legal_fold (bet>0) / illegal_fold (bet==0) -> FOLD
KEEP_BUCKETS = {"clean_check_or_call", "clean_legal_fold", "illegal_fold"}

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enriched-log", nargs="+", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--model-id", default=None, help="override; else taken from run_config")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(_DTYPES))
    ap.add_argument("--max-per-bucket", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    agent_config = _load_agent_config(args.enriched_log[0])
    model_id = args.model_id or agent_config["model_id"]
    print(f"[load] {model_id}  layer={args.layer}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=_DTYPES[args.dtype], device_map=args.device)
    model.eval()
    recon = PromptReconstructor(tokenizer, agent_config)
    cap = HiddenStateCapture(model)

    X, verbs, bets, buckets = [], [], [], []
    oracle_sa, oracle_co, beliefs = [], [], []   # for encode-vs-decode (may be all-NaN if absent)
    per_bucket = {b: 0 for b in KEEP_BUCKETS}
    n_seen = n_kept = 0
    _nan14 = np.full(14, np.nan, dtype=np.float32)

    for log in args.enriched_log:
        for rec in _iter_decisions(log):
            am = rec.get("action_metadata")
            if not isinstance(am, dict) or not am.get("raw_response"):
                continue
            obs = rec.get("obs")
            if not isinstance(obs, dict) or obs.get("bet_to_call") is None:
                continue
            bucket = classify_decision(rec)
            if bucket not in KEEP_BUCKETS or per_bucket[bucket] >= args.max_per_bucket:
                continue
            built = build_input_text_for_action_verb_position(
                recon.build(rec), am["raw_response"], tokenizer)
            if built is None:
                continue
            input_text, _ = built
            n_seen += 1
            ids = tokenizer(input_text, return_tensors="pt",
                            add_special_tokens=False)["input_ids"].to(args.device)
            cap.attach_hooks()
            try:
                with torch.no_grad():
                    model(input_ids=ids)
            finally:
                states = cap.collect()
                cap.detach_hooks()
            resid = states["per_layer_last_pos"].get(args.layer)
            if resid is None:
                continue
            b2c = float(obs["bet_to_call"])
            if bucket == "clean_check_or_call":
                verb = "CALL" if b2c > 0 else "CHECK"
            else:
                verb = "FOLD"
            X.append(resid.float().cpu().numpy())
            verbs.append(verb)
            bets.append(b2c)
            buckets.append(bucket)
            sa = _belief_vec(rec.get("oracle_strategy_aware"))
            co = _belief_vec(rec.get("oracle_card_only"))
            bl = _belief_vec(rec.get("agent_belief"))
            oracle_sa.append(sa if sa is not None else _nan14)
            oracle_co.append(co if co is not None else _nan14)
            beliefs.append(bl if bl is not None else _nan14)
            per_bucket[bucket] += 1
            n_kept += 1
            if n_kept % 50 == 0:
                print(f"  kept {n_kept}  ({per_bucket})")

    X = np.asarray(X, dtype=np.float32)
    print(f"[done] seen={n_seen} kept={n_kept}  per_bucket={per_bucket}  X={X.shape}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out,
             X=X,
             verb=np.array(verbs),
             bet_to_call=np.array(bets, dtype=np.float32),
             bucket=np.array(buckets),
             oracle_strategy_aware=np.asarray(oracle_sa, dtype=np.float32),
             oracle_card_only=np.asarray(oracle_co, dtype=np.float32),
             agent_belief=np.asarray(beliefs, dtype=np.float32),
             bucket_order=np.array(list(BUCKET_ORDER)),
             layer=np.int64(args.layer),
             model_id=np.array(model_id))
    print(f"[written] {args.out}")
    print("Next (CPU): python -m experiments.bet_matched_probe --tagged "
          f"{args.out} --out results/direction_probe_baselines/BET_MATCHED_<model>_l{args.layer}.md")


if __name__ == "__main__":
    main()
