"""
Belief direction probe (B3) — tests how the verb-decision direction relates
to the model's belief-distribution encoding at L*.

Question answered:
    "At L*, is the verb-decision direction ORTHOGONAL to the
     belief-encoding subspace, or ALIGNED with one of its principal axes?"

Background: the original paper's central claim is *belief inertia*. We've
identified a verb-decision circuit but haven't connected it to belief. If
the verb-direction sits in the same subspace as the belief-direction(s),
the verb decision is "belief-modulated" at the residual level. If
orthogonal, verb and belief are independently represented.

Procedure:
  1. Load enriched log; for each decision capture L* residual at the
     verb position AND the oracle_strategy_aware 14-d distribution.
  2. Train a multi-output linear regression:
        Residual ∈ R^{hidden_dim} -> oracle_sa ∈ R^{14}
     This gives a weight matrix W ∈ R^{hidden_dim × 14}.
  3. Compute SVD of W to get the principal directions.
  4. Compare the cached verb direction (from decision_direction_probe)
     to each bucket's column in W (cosine), and to the principal
     direction.

Outputs: SUMMARY.md with per-bucket cosines + the principal-axis cosine,
plus interpretive guide.

Usage::

    python -m experiments.belief_direction_probe \\
        --enriched-log <pooled> \\
        --layer 14 \\
        --probe-npz results/direction_probe/llama8b_l14/raw_residuals.npz \\
        --max-decisions 300 \\
        --out-dir results/belief_direction_probe/llama8b_l14 \\
        --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_env.interp.forward_helpers import (  # noqa: E402
    PromptReconstructor,
    build_input_text_for_action_verb_position,
)
from poker_env.interp.patching import HiddenStateCapture  # noqa: E402
from experiments.causal_patching import (  # noqa: E402
    _iter_decisions, _load_agent_config,
)


# Canonical 14-bucket order (from buckets_14_v1).
BUCKET_NAMES_14 = [
    "premium_pairs", "strong_pairs", "medium_pairs", "small_pairs",
    "premium_broadway", "strong_broadway", "medium_broadway", "weak_broadway",
    "suited_connectors", "suited_aces", "suited_gappers",
    "speculative_suited", "offsuit_connectors", "trash",
]


def main():
    parser = argparse.ArgumentParser(description="Belief direction probe (B3).")
    parser.add_argument("--enriched-log", required=True, nargs="+")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--probe-npz", required=True,
                        help="Cached raw_residuals.npz from "
                             "decision_direction_probe — provides w_verb.")
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--belief-source", default="oracle_strategy_aware",
                        choices=["oracle_strategy_aware", "oracle_card_only", "agent_belief"],
                        help="Which belief distribution to regress against.")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help=(
            "If > 0, also run k-fold cross-validation and report HELD-OUT "
            "R^2 (averaged across folds) in addition to the in-sample R^2. "
            "Mandatory for high-dimensional residuals (hidden_dim ~ 4096) "
            "where in-sample R^2 can hit 0.999 due to overfitting; the "
            "held-out R^2 is the trustworthy summary statistic. "
            "Recommended: --cv-folds 5."
        ),
    )
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
    import numpy as np
    from sklearn.linear_model import Ridge

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
    cap = HiddenStateCapture(model)

    # ---- Capture residuals + belief targets ------------------------------
    print(f"\n[capture] up to {args.max_decisions} decisions ...")
    X = []   # [n, hidden]
    Y = []   # [n, 14]
    keys = []
    n_skipped_belief = 0
    n_processed = 0
    t0 = time.time()
    for log_path in enriched_logs:
        for rec in _iter_decisions(log_path):
            if n_processed >= args.max_decisions:
                break
            am = rec.get("action_metadata") or {}
            raw = am.get("raw_response")
            if not raw:
                continue
            belief = rec.get(args.belief_source)
            if belief is None or not isinstance(belief, dict):
                n_skipped_belief += 1
                continue
            # Convert belief dict to ordered 14-vector.
            try:
                y = np.array([float(belief[b]) for b in BUCKET_NAMES_14],
                             dtype=np.float32)
            except (KeyError, TypeError):
                n_skipped_belief += 1
                continue
            full_prompt = recon.build(rec)
            out = build_input_text_for_action_verb_position(
                full_prompt, raw, tokenizer,
            )
            if out is None:
                continue
            input_text, _ = out
            enc = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(args.device)

            cap.attach_hooks()
            try:
                with torch.no_grad():
                    model(input_ids=input_ids)
            finally:
                states = cap.collect()
                cap.detach_hooks()
            r = states["per_layer_last_pos"].get(args.layer)
            if r is None:
                continue
            X.append(r.to(torch.float32).cpu().numpy())
            Y.append(y)
            keys.append(f"{rec.get('hand_id', '?')}:{rec.get('decision_idx', '?')}")
            n_processed += 1
            if n_processed % 25 == 0:
                rate = n_processed / max(time.time() - t0, 1e-3)
                print(f"  [{n_processed}/{args.max_decisions}] captured "
                      f"({rate:.2f}/s)")
        if n_processed >= args.max_decisions:
            break
    print(f"[capture] DONE: n_processed={n_processed} "
          f"(skipped due to missing belief: {n_skipped_belief})")
    if n_processed < 20:
        print(f"[abort] too few captures (n_processed={n_processed})",
              file=sys.stderr)
        sys.exit(3)

    X = np.stack(X, 0)
    Y = np.stack(Y, 0)
    print(f"[shape] X={X.shape}  Y={Y.shape}")

    # ---- Train multi-output Ridge regression -----------------------------
    # Hidden_dim ≫ samples, so strong regularization is needed. Ridge with
    # alpha=1.0 is conservative and gives stable estimates.
    print(f"\n[probe] training Ridge regression (residual -> 14-d belief) ...")
    reg = Ridge(alpha=1.0, random_state=args.seed)
    reg.fit(X, Y)
    W = reg.coef_   # shape [14, hidden]
    train_score = reg.score(X, Y)   # average R^2 across outputs (in-sample)
    # Per-bucket R^2 (in-sample).
    Y_pred = reg.predict(X)
    ss_res = ((Y - Y_pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(axis=0)
    r2_per_bucket = 1 - (ss_res / np.maximum(ss_tot, 1e-12))
    print(f"  in-sample overall R^2: {train_score:.3f}")
    print(f"  in-sample per-bucket R^2: " + ", ".join(
        f"{b}={r:.2f}" for b, r in zip(BUCKET_NAMES_14, r2_per_bucket)
    ))

    # Optional k-fold CV: held-out R^2 is the trustworthy generalization
    # estimate when hidden_dim ≫ n_samples. Without this, the in-sample
    # R^2 is essentially uninterpretable for high-dimensional residuals.
    cv_overall = None
    cv_per_bucket = None
    if args.cv_folds and args.cv_folds >= 2:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        fold_overall = []
        fold_per_bucket = []
        print(f"\n[probe] running {args.cv_folds}-fold CV for held-out R^2 ...")
        for fold_i, (tr, te) in enumerate(kf.split(X)):
            reg_cv = Ridge(alpha=1.0, random_state=args.seed)
            reg_cv.fit(X[tr], Y[tr])
            Y_te_pred = reg_cv.predict(X[te])
            ss_res_te = ((Y[te] - Y_te_pred) ** 2).sum(axis=0)
            ss_tot_te = ((Y[te] - Y[tr].mean(0)) ** 2).sum(axis=0)
            r2_b = 1 - (ss_res_te / np.maximum(ss_tot_te, 1e-12))
            fold_per_bucket.append(r2_b)
            fold_overall.append(float(r2_b.mean()))
            print(f"  fold {fold_i+1}: held-out R^2 = {fold_overall[-1]:.3f}")
        cv_overall = float(np.mean(fold_overall))
        cv_overall_std = float(np.std(fold_overall))
        cv_per_bucket = np.mean(np.stack(fold_per_bucket, 0), axis=0)
        print(f"[probe] HELD-OUT overall R^2 (mean over {args.cv_folds} folds): "
              f"{cv_overall:.3f} ± {cv_overall_std:.3f}")
        print(f"[probe] HELD-OUT per-bucket R^2: " + ", ".join(
            f"{b}={r:.2f}" for b, r in zip(BUCKET_NAMES_14, cv_per_bucket)
        ))
    else:
        cv_overall_std = None

    # ---- SVD: dominant directions of W -----------------------------------
    # W ∈ R^{14 × hidden}. SVD: W = U S V^T, where columns of V^T are
    # right-singular vectors (in residual space) — these are the principal
    # belief directions.
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    principal_belief_direction = Vt[0]   # shape [hidden]
    explained = S ** 2 / (S ** 2).sum()
    print(f"\n[svd] singular values: {S[:5].tolist()}")
    print(f"[svd] explained variance ratios (first 5): "
          + ", ".join(f"{e:.3f}" for e in explained[:5]))

    # ---- Compare to cached verb direction --------------------------------
    npz = np.load(args.probe_npz)
    w_verb = npz["weight_vec"].astype(np.float32)
    if w_verb.shape[0] != X.shape[1]:
        print(f"[abort] cached verb direction has dim {w_verb.shape[0]} "
              f"but residuals have dim {X.shape[1]}", file=sys.stderr)
        sys.exit(2)

    def _cos(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        return float(a @ b / (na * nb + 1e-9))

    cos_verb_principal = _cos(w_verb, principal_belief_direction)
    cos_verb_per_bucket = [
        _cos(w_verb, W[i]) for i in range(W.shape[0])
    ]
    cos_verb_per_principal = [_cos(w_verb, Vt[i]) for i in range(min(5, Vt.shape[0]))]

    print(f"\n[result] cos(w_verb, principal_belief_direction) = {cos_verb_principal:+.4f}")
    print(f"[result] cos(w_verb, top-5 belief PCs): "
          + ", ".join(f"{c:+.3f}" for c in cos_verb_per_principal))
    print(f"[result] cos(w_verb, per-bucket weight vector):")
    for b, c in zip(BUCKET_NAMES_14, cos_verb_per_bucket):
        print(f"    {b:25s}: {c:+.4f}")

    summary = {
        "model_id": model_id,
        "layer": args.layer,
        "belief_source": args.belief_source,
        "n_decisions": int(n_processed),
        "n_belief_skipped": int(n_skipped_belief),
        "ridge_alpha": 1.0,
        "in_sample_overall_R2": float(train_score),
        "in_sample_per_bucket_R2": dict(zip(BUCKET_NAMES_14, [float(r) for r in r2_per_bucket])),
        "cv_folds": int(args.cv_folds) if args.cv_folds else 0,
        "held_out_overall_R2_mean": (float(cv_overall) if cv_overall is not None else None),
        "held_out_overall_R2_std": (float(cv_overall_std) if cv_overall_std is not None else None),
        "held_out_per_bucket_R2": (
            dict(zip(BUCKET_NAMES_14, [float(r) for r in cv_per_bucket]))
            if cv_per_bucket is not None else None
        ),
        "singular_values_top5": [float(s) for s in S[:5]],
        "explained_variance_ratios_top5": [float(e) for e in explained[:5]],
        "cos_verb_principal_belief_direction": cos_verb_principal,
        "cos_verb_top5_principals": [float(c) for c in cos_verb_per_principal],
        "cos_verb_per_bucket": dict(zip(BUCKET_NAMES_14,
                                        [float(c) for c in cos_verb_per_bucket])),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(
        out_dir / "raw.npz",
        X=X, Y=Y, W=W, w_verb=w_verb,
        principal_belief=principal_belief_direction,
    )

    md = ["# Belief direction probe (B3) results", ""]
    md.append(f"- Model: `{model_id}`")
    md.append(f"- Layer: **{args.layer}**")
    md.append(f"- Belief source: `{args.belief_source}`")
    md.append(f"- n_decisions: {n_processed} (skipped due to missing belief: {n_skipped_belief})")
    md.append("")
    md.append("## Multi-output Ridge regression: residual → 14-d belief distribution")
    md.append(f"- in-sample overall R²: {train_score:.3f}")
    if cv_overall is not None:
        md.append(
            f"- **held-out overall R² (mean over {args.cv_folds} folds): "
            f"{cv_overall:.3f} ± {cv_overall_std:.3f}** "
            "(this is the trustworthy generalization estimate)"
        )
    else:
        md.append("- held-out R²: (not computed; pass `--cv-folds 5` to enable)")
    md.append("")
    md.append("| Bucket | in-sample R² | held-out R² (CV) |")
    md.append("|---|---:|---:|")
    for i, b in enumerate(BUCKET_NAMES_14):
        in_r = r2_per_bucket[i]
        out_r = cv_per_bucket[i] if cv_per_bucket is not None else None
        out_str = f"{out_r:+.3f}" if out_r is not None else "—"
        md.append(f"| `{b}` | {in_r:+.3f} | {out_str} |")
    md.append("")
    md.append("## Principal directions of the belief subspace (SVD)")
    md.append("")
    md.append("| Component | singular value | explained variance |")
    md.append("|---|---:|---:|")
    for i in range(min(5, len(S))):
        md.append(f"| {i+1} | {S[i]:.2f} | {explained[i]*100:.1f}% |")
    md.append("")
    md.append("## Verb direction × belief subspace cosines")
    md.append("")
    md.append(f"- **cos(w_verb, principal_belief_direction): {cos_verb_principal:+.4f}**")
    md.append("- cos(w_verb, top-5 belief PCs):")
    for i, c in enumerate(cos_verb_per_principal):
        md.append(f"    {i+1}. {c:+.4f} (explained var: {explained[i]*100:.1f}%)")
    md.append("")
    md.append("- cos(w_verb, per-bucket belief direction):")
    md.append("")
    md.append("| Bucket | cosine |")
    md.append("|---|---:|")
    for b, c in zip(BUCKET_NAMES_14, cos_verb_per_bucket):
        md.append(f"| `{b}` | {c:+.4f} |")
    md.append("")
    md.append("## Reading guide")
    md.append("")
    md.append("- **|cos(w_verb, principal_belief_direction)| < 0.2**: verb and "
              "belief are encoded ORTHOGONALLY at L*. The verb-decision direction "
              "is independent of the dominant belief direction. Implies belief "
              "and verb are separately represented and the L* circuit doesn't "
              "directly use belief content for the verb choice.")
    md.append("- **|cos(w_verb, principal_belief_direction)| > 0.5**: verb and "
              "belief share a substantial axis. Implies the belief representation "
              "is RECRUITED for the verb decision at L*.")
    md.append("- **Strong cosines on specific buckets**: the verb-direction is "
              "ALIGNED with specific belief buckets (e.g. `premium_pairs` "
              "positive, `trash` negative). Suggests the model uses certain "
              "hand-strength features more than others to decide the verb.")
    md.append("- **R² high (>0.4)**: belief is well-decoded from L* residual. "
              "L* carries belief information.")
    md.append("- **R² low (<0.1)**: belief is NOT linearly decodable from L*. "
              "Either belief lives at a different layer, or it's encoded "
              "non-linearly. (Note: belief inertia is a behavioral metric; "
              "this probe asks about the residual-stream representation.)")
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[done] wrote {out_dir / 'SUMMARY.md'}")
    print(f"[done] wrote {out_dir / 'summary.json'}")
    print(f"[done] wrote {out_dir / 'raw.npz'}")


if __name__ == "__main__":
    main()
