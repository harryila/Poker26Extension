"""
Decision-direction probe at L* (REQUIRES GPU model load).

Question answered:
    "Is there a SINGLE LINEAR DIRECTION in residual space at L* that
     encodes the action verb? If so, are illegal_FOLD residuals aligned
     with that direction's FOLD side?"

This bridges the per-head decomposition results (which say WHICH heads
contribute) with a single-direction story (which would say WHAT direction
in residual space encodes the decision).

Procedure:
  1. Load model + tokenizer.
  2. For every clean_check_or_call decision in the pooled enriched logs,
     run a forward pass and capture the residual at L* at the LAST input
     position. Same for clean_legal_fold and (separately) illegal_fold.
  3. Train a logistic regression on
        clean_check_or_call (label +1) vs clean_legal_fold (label -1)
     residuals, with strong L2 regularization (necessary because hidden_dim
     is much larger than n_samples).
  4. Report:
        - 5-fold CV accuracy on clean vs clean
        - cosine(weight_vector, centroid_diff)
          where centroid_diff = mean(check residuals) - mean(fold residuals)
          (sanity check — these should be highly aligned)
        - mean projection of illegal_fold residuals onto the weight vector
          (positive = aligned with CHECK side; negative = aligned with FOLD)
        - distribution of those projections vs clean-bucket distributions
  5. Save: SUMMARY.md + raw_residuals.npz (residuals + labels, for any
     further offline analysis).

Cross-model comparison: run for each model at its L*. If all three models
have a clean linear direction AND illegal_FOLDs project on the FOLD side,
the cross-model story is "same direction encodes the decision; failure
mode is misalignment on that direction."

Usage::

    python -m experiments.decision_direction_probe \
        --enriched-log logs/cot_llama8b_t0_s42_*.jsonl.gz \
                       logs/cot_llama8b_t0_s123_*.jsonl.gz \
                       logs/cot_llama8b_t0_s456_*.jsonl.gz \
        --layer 14 \
        --max-decisions-per-bucket 300 \
        --out-dir results/direction_probe/llama8b_l14 \
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.recategorize_action_metadata import _reparse_one  # noqa: E402
from poker_env.interp.forward_helpers import (  # noqa: E402
    PromptReconstructor,
    build_input_text_for_action_verb_position,
)
from poker_env.interp.patching import HiddenStateCapture  # noqa: E402
from experiments.causal_patching import (  # noqa: E402
    BUCKET_NAMES,
    classify_decision,
    _iter_decisions,
    _load_agent_config,
)


def _capture_residuals_for_bucket(
    model,
    tokenizer,
    recon: PromptReconstructor,
    bucket_records: list[dict],
    layer: int,
    device: str,
    max_n: int,
    label: str,
) -> tuple[list, list]:
    """Returns (residuals_list, hand_dec_list) for up to max_n decisions
    from the given bucket. Each residual is a 1-D CPU torch tensor of
    shape [hidden_dim]."""
    import torch

    cap = HiddenStateCapture(model)
    residuals = []
    keys = []

    n = 0
    t0 = time.time()
    for rec in bucket_records:
        if n >= max_n:
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
        input_ids = enc["input_ids"].to(device)

        cap.attach_hooks()
        try:
            with torch.no_grad():
                model(input_ids=input_ids)
        finally:
            states = cap.collect()
            cap.detach_hooks()

        if layer not in states["per_layer_last_pos"]:
            continue
        residuals.append(states["per_layer_last_pos"][layer].clone())
        keys.append(f"{rec.get('hand_id', '?')}:{rec.get('decision_idx', '?')}")
        n += 1
        if n % 25 == 0:
            elapsed = time.time() - t0
            rate = n / elapsed
            print(f"    [{label}] {n} captured ({elapsed:.0f}s, {rate:.2f}/s)")
    elapsed = time.time() - t0
    print(f"    [{label}] DONE: {n} captured in {elapsed:.0f}s")
    return residuals, keys


def main():
    parser = argparse.ArgumentParser(
        description="Decision-direction linear probe at L*."
    )
    parser.add_argument("--enriched-log", required=True, nargs="+",
                        help="Pooled enriched JSONL[.gz] logs.")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--max-decisions-per-bucket", type=int, default=300)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--l2-c", type=float, default=0.01,
                        help="Inverse-regularization for sklearn LogisticRegression "
                             "(smaller = stronger regularization). Default 0.01 "
                             "is appropriate for hidden_dim >> n_samples.")
    parser.add_argument("--seed", type=int, default=42)
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
                print(f"[abort] model mismatch: {model_id} vs "
                      f"{other_cfg.get('model_id')} in {extra}", file=sys.stderr)
                sys.exit(2)
    print(f"[init] model_id={model_id}")
    print(f"[init] target layer L={args.layer}")
    print(f"[init] pooling {len(enriched_logs)} enriched logs")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

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

    recon = PromptReconstructor(tokenizer, agent_config)

    # ---- Bucket -----------------------------------------------------------
    print("[bucket] scanning ...")
    by_bucket: dict[str, list[dict]] = {b: [] for b in BUCKET_NAMES}
    for log_path in enriched_logs:
        for rec in _iter_decisions(log_path):
            am = rec.get("action_metadata")
            if am is None or not am.get("raw_response"):
                continue
            rec.setdefault("_source_log", log_path)
            by_bucket[classify_decision(rec)].append(rec)
    for b in BUCKET_NAMES:
        print(f"  {b:<22}: {len(by_bucket[b])}")

    # ---- Capture residuals -----------------------------------------------
    print(f"\n[capture] up to {args.max_decisions_per_bucket} per bucket "
          f"at L={args.layer} ...")
    cc_res, cc_keys = _capture_residuals_for_bucket(
        model, tokenizer, recon,
        by_bucket["clean_check_or_call"], args.layer, args.device,
        args.max_decisions_per_bucket, "clean_check_or_call",
    )
    lf_res, lf_keys = _capture_residuals_for_bucket(
        model, tokenizer, recon,
        by_bucket["clean_legal_fold"], args.layer, args.device,
        args.max_decisions_per_bucket, "clean_legal_fold",
    )
    if_res, if_keys = _capture_residuals_for_bucket(
        model, tokenizer, recon,
        by_bucket["illegal_fold"], args.layer, args.device,
        args.max_decisions_per_bucket, "illegal_fold",
    )

    if len(cc_res) < 10 or len(lf_res) < 10:
        print(f"[abort] need ≥10 of each clean bucket; got "
              f"clean_CC={len(cc_res)}, clean_LF={len(lf_res)}",
              file=sys.stderr)
        sys.exit(3)

    # ---- Stack into numpy arrays -----------------------------------------
    X_cc = torch.stack(cc_res, dim=0).to(torch.float32).numpy()  # [n_cc, hidden]
    X_lf = torch.stack(lf_res, dim=0).to(torch.float32).numpy()
    X_if = (torch.stack(if_res, dim=0).to(torch.float32).numpy()
            if if_res else np.zeros((0, X_cc.shape[1]), dtype=np.float32))

    print(f"\n[shape] clean_CC: {X_cc.shape}  clean_LF: {X_lf.shape}  "
          f"illegal_FOLD: {X_if.shape}")

    # ---- Train probe: clean_CC (+1) vs clean_LF (-1) ----------------------
    X_train = np.concatenate([X_cc, X_lf], axis=0)
    y_train = np.concatenate([
        np.ones(len(X_cc), dtype=np.int64),
        np.zeros(len(X_lf), dtype=np.int64),
    ])
    print(f"\n[probe] training logistic regression on {len(X_train)} samples "
          f"(C={args.l2_c}, hidden_dim={X_train.shape[1]}) ...")
    clf = LogisticRegression(
        C=args.l2_c,
        max_iter=2000,
        solver="lbfgs",
        random_state=args.seed,
    )
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                          random_state=args.seed)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="accuracy")
    print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f} "
          f"(folds: {[f'{s:.3f}' for s in cv_scores]})")
    clf.fit(X_train, y_train)
    weight_vec = clf.coef_[0]                   # [hidden]
    weight_norm = float(np.linalg.norm(weight_vec))
    train_acc = float(clf.score(X_train, y_train))
    print(f"  train accuracy: {train_acc:.3f}")
    print(f"  ||w||_2: {weight_norm:.4f}")

    # ---- Centroid-difference sanity ---------------------------------------
    centroid_diff = X_cc.mean(axis=0) - X_lf.mean(axis=0)
    cos_centroid_w = float(
        np.dot(centroid_diff, weight_vec)
        / (np.linalg.norm(centroid_diff) * weight_norm + 1e-9)
    )
    print(f"  cos(centroid_diff, w): {cos_centroid_w:.4f}  "
          f"(close to 1 = probe agrees with mean-difference direction)")

    # ---- Project illegal_FOLDs onto the direction --------------------------
    def _project(X):
        # signed projection onto the weight vector. Positive = CHECK side;
        # negative = FOLD side.
        return (X @ weight_vec) / (weight_norm + 1e-9)

    proj_cc = _project(X_cc)
    proj_lf = _project(X_lf)
    proj_if = _project(X_if) if len(X_if) > 0 else np.array([])

    print(f"\n[projection] mean ± std signed projection onto w (in nat-like units):")
    print(f"  clean_check_or_call : {proj_cc.mean():+.3f} ± {proj_cc.std():.3f}  "
          f"(n={len(proj_cc)})")
    print(f"  clean_legal_fold    : {proj_lf.mean():+.3f} ± {proj_lf.std():.3f}  "
          f"(n={len(proj_lf)})")
    if len(proj_if) > 0:
        # Bayesian-style "which side" indicator:
        midpoint = (proj_cc.mean() + proj_lf.mean()) / 2.0
        on_fold_side = (proj_if < midpoint).mean()
        print(f"  illegal_FOLD        : {proj_if.mean():+.3f} ± {proj_if.std():.3f}  "
              f"(n={len(proj_if)})")
        print(f"    fraction on FOLD side of midpoint: {on_fold_side*100:.1f}%")

    # ---- Save -------------------------------------------------------------
    summary = {
        "enriched_logs": enriched_logs,
        "model_id": model_id,
        "layer": args.layer,
        "n_clean_check_or_call": int(len(X_cc)),
        "n_clean_legal_fold": int(len(X_lf)),
        "n_illegal_fold": int(len(X_if)),
        "hidden_dim": int(X_train.shape[1]),
        "probe": {
            "C": args.l2_c,
            "cv_folds": args.cv_folds,
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "cv_accuracy_per_fold": [float(s) for s in cv_scores],
            "train_accuracy": train_acc,
            "weight_l2_norm": weight_norm,
            "cos_centroid_diff_w": cos_centroid_w,
        },
        "projection": {
            "clean_check_or_call": {
                "mean": float(proj_cc.mean()),
                "std": float(proj_cc.std()),
            },
            "clean_legal_fold": {
                "mean": float(proj_lf.mean()),
                "std": float(proj_lf.std()),
            },
            "illegal_fold": (
                {
                    "mean": float(proj_if.mean()),
                    "std": float(proj_if.std()),
                    "frac_on_fold_side_of_midpoint": float(
                        (proj_if < (proj_cc.mean() + proj_lf.mean()) / 2.0).mean()
                    ),
                } if len(proj_if) > 0 else None
            ),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save raw arrays for any later offline analysis.
    np.savez(
        out_dir / "raw_residuals.npz",
        X_cc=X_cc, X_lf=X_lf, X_if=X_if,
        weight_vec=weight_vec, centroid_diff=centroid_diff,
        proj_cc=proj_cc, proj_lf=proj_lf, proj_if=proj_if,
    )

    # SUMMARY.md
    md = []
    md.append("# Decision-direction probe results")
    md.append("")
    md.append(f"- Model: `{model_id}`")
    md.append(f"- Layer: **{args.layer}**")
    md.append(f"- Enriched logs (pooled, n={len(enriched_logs)}):")
    for p in enriched_logs:
        md.append(f"  - `{p}`")
    md.append("")
    md.append("## Probe (clean_check_or_call vs clean_legal_fold)")
    md.append(f"- Hidden dim: {X_train.shape[1]}")
    md.append(f"- Samples: {len(X_cc)} CHECK + {len(X_lf)} FOLD = {len(X_train)} total")
    md.append(f"- Regularization (sklearn LR `C`): {args.l2_c}")
    md.append(f"- **{args.cv_folds}-fold CV accuracy: "
              f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}**")
    md.append(f"- Per-fold: {[f'{s:.3f}' for s in cv_scores]}")
    md.append(f"- Train accuracy: {train_acc:.3f}")
    md.append(f"- ||w||₂: {weight_norm:.4f}")
    md.append(f"- cos(centroid_diff, w): **{cos_centroid_w:.4f}** "
              "(closer to 1 = probe weight aligns with mean-CHECK − mean-FOLD direction; "
              "this is a sanity check, not a separate finding)")
    md.append("")
    md.append("## Projection of bucket residuals onto the learned direction")
    md.append("Sign convention: positive = CHECK side; negative = FOLD side. "
              "Units are pre-normalization residual / ||w||, so they are "
              "scale-comparable across buckets within this experiment.")
    md.append("")
    md.append("| Bucket | n | mean projection | std |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| clean_check_or_call | {len(proj_cc)} | {proj_cc.mean():+.3f} | {proj_cc.std():.3f} |")
    md.append(f"| clean_legal_fold    | {len(proj_lf)} | {proj_lf.mean():+.3f} | {proj_lf.std():.3f} |")
    if len(proj_if) > 0:
        midpoint = (proj_cc.mean() + proj_lf.mean()) / 2.0
        on_fold_side = (proj_if < midpoint).mean()
        md.append(f"| **illegal_fold**        | {len(proj_if)} | **{proj_if.mean():+.3f}** | {proj_if.std():.3f} |")
        md.append("")
        md.append(f"**Fraction of illegal_FOLDs on the FOLD side of midpoint: "
                  f"{on_fold_side*100:.1f}%** "
                  "(midpoint = average of clean-bucket means)")
    md.append("")
    md.append("## Reading guide")
    md.append("- High CV accuracy (>0.85) AND high cos(centroid, w) (>0.9): "
              "a single linear direction encodes the verb decision; the "
              "circuit is direction-projectable.")
    md.append("- illegal_FOLD mean projection well below clean_LF mean: "
              "the failure mode is *more* FOLD-aligned than legal FOLDs are. "
              "Consistent with §13's 'illegal_FOLDs lock in earlier and more "
              "confidently' finding.")
    md.append("- illegal_FOLD on FOLD side fraction ≥80%: failure mode lives "
              "on the same axis as the legal decision, just past the threshold.")
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[done] wrote {out_dir / 'summary.json'}")
    print(f"[done] wrote {out_dir / 'SUMMARY.md'}")
    print(f"[done] wrote {out_dir / 'raw_residuals.npz'}")


if __name__ == "__main__":
    main()
