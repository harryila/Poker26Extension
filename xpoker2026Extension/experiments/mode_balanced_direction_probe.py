"""
Mode-balanced decision-direction probe.

Companion to `decision_direction_probe.py` and `direction_cosine_compare.py`.
Trains two probes — one on CoT residuals, one on non-CoT residuals —
restricted to decisions present in BOTH modes (matched by hand_id +
decision_idx). This eliminates the data-distribution-shift confound that
inflates cosine miscalibration in the §18b cosine compare:

    Phase M §18b cosine: cos(w_CoT, w_nonCoT) = +0.27 (Llama L=14)
                                                 +0.34 (Qwen L=23)

was computed across DIFFERENT decision sets per probe (CoT used CoT-mode
clean_CC vs clean_LF; non-CoT used non-CoT-mode clean_CC vs clean_LF —
neither agreeing on which hands made it into each set).

This script's matched-hand cosine answers: "for the SAME poker situation,
how much does the verb-decision direction tilt between CoT and non-CoT?"

Procedure:
  1. Load CoT enriched log and non-CoT enriched log for the same (model,
     seed) pair.
  2. Build hand-keyed dicts: key = f"{hand_id}:{decision_idx}".
  3. Compute matched_keys = the intersection. For each matched key, both
     modes must have a decision that classifies into clean_check_or_call
     OR clean_legal_fold (one of each — same recorded action across modes
     would be ideal, but may not always agree).
  4. Capture residual at L* under CoT prompt (using CoT agent_config).
  5. Capture residual at L* under non-CoT prompt (using non-CoT agent_config).
  6. Train a probe per mode on the matched residuals, with mode-specific
     labels (each mode's recorded-action classification).
  7. Output: cos(w_CoT, w_nonCoT) on the matched data + per-mode CV
     accuracy + cross-projection signs.

Usage::

    python -m experiments.mode_balanced_direction_probe \\
        --cot-log    logs/cot_llama8b_t0_s42_*.jsonl.gz \\
        --nocot-log  logs/scaled_llama8b_t0_s42_*.jsonl \\
        --layer 14 \\
        --max-pairs 200 \\
        --out-dir results/mode_balanced_probe/llama8b_l14 \\
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
    classify_decision,
    _iter_decisions,
    _load_agent_config,
)


def _load_records_keyed(path: str) -> dict[str, dict]:
    """Load decisions keyed by '{seed}:{decision_idx}'.

    NOTE: previously used `(hand_id, decision_idx)` as the key, but
    `hand_id` is a random UUID assigned per `env.reset()` (see
    `poker_env/env.py` line 188: `self.hand_id = str(uuid.uuid4())[:8]`),
    so it never matches across runs even when the SAME seed produced the
    SAME dealt hand. The per-hand `seed` field IS deterministic
    (base_seed + i * 1000, see `run_experiment.run_multi_hand`), so
    `(seed, decision_idx)` is the correct match key for cross-run
    pairing of decisions on the same dealt hand.

    Caveat: `(seed, decision_idx)` guarantees identical hole cards / deck
    state, AND guarantees identical game state at decision_idx=1 (no prior
    actions). At later decision indices, the game state can diverge if
    CoT and non-CoT modes took different prior actions. Downstream code
    should compute a game-state-identity check (board + pot + bet_to_call)
    to report how many matched pairs are on IDENTICAL game states vs
    merely on the same dealt hand.
    """
    out = {}
    for rec in _iter_decisions(path):
        am = rec.get("action_metadata")
        if am is None or not am.get("raw_response"):
            continue
        key = f"{rec.get('seed', '?')}:{rec.get('decision_idx', '?')}"
        out[key] = rec
    return out


def _game_state_signature(rec: dict) -> tuple:
    """Compute a tuple that identifies the GAME STATE at the time of
    this decision (independent of agent stochasticity / belief
    elicitation choices). Used to verify that two matched (seed,
    decision_idx) records from different modes are on identical game
    states (not just the same dealt hand).
    """
    obs = rec.get("obs") or {}
    board = tuple(obs.get("board") or [])
    holes = obs.get("hole_cards")
    if isinstance(holes, list):
        holes = tuple(holes)
    return (
        rec.get("seed"),
        rec.get("decision_idx"),
        rec.get("player_to_act"),
        rec.get("street"),
        obs.get("pot_total"),
        obs.get("bet_to_call"),
        obs.get("position"),
        board,
        holes,
    )


def _capture_residual_at_layer(
    model, tokenizer, agent_config: dict, rec: dict,
    layer: int, device: str,
):
    """Run a single forward, capture residual at the verb position (last
    input position) at the requested layer. Returns 1-D CPU tensor or None
    on prep failure."""
    import torch
    recon = PromptReconstructor(tokenizer, agent_config)
    full_prompt = recon.build(rec)
    am = rec["action_metadata"]
    out = build_input_text_for_action_verb_position(
        full_prompt, am["raw_response"], tokenizer,
    )
    if out is None:
        return None
    input_text, _ = out
    enc = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)

    cap = HiddenStateCapture(model)
    cap.attach_hooks()
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        states = cap.collect()
        cap.detach_hooks()

    return states["per_layer_last_pos"].get(layer)


def main():
    parser = argparse.ArgumentParser(
        description="Mode-balanced direction probe (CoT vs non-CoT, hand-matched)."
    )
    parser.add_argument("--cot-log", required=True,
                        help="Path to a single CoT enriched log (jsonl[.gz]).")
    parser.add_argument("--nocot-log", required=True,
                        help="Path to a single non-CoT enriched log.")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--max-pairs", type=int, default=200,
                        help="Cap matched (seed × decision_idx) pairs to "
                             "this number, balanced across CHECK and FOLD "
                             "buckets where possible.")
    parser.add_argument(
        "--require-identical-game-state",
        action="store_true",
        help=(
            "Restrict matched pairs to those where the obs (board, pot, "
            "bet_to_call, position, hole_cards) is byte-identical across "
            "CoT and non-CoT — guaranteeing the modes saw the exact same "
            "game state, not just the same dealt hand. Recommended for "
            "publication-grade matched-cosine claims. Without this flag, "
            "later decisions in a hand can diverge across modes if CoT "
            "and non-CoT took different prior actions, in which case the "
            "'matched' residuals are on slightly different inputs."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--l2-c", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load both logs and identify matched keys.
    print(f"[init] loading CoT log: {args.cot_log}")
    cot_recs = _load_records_keyed(args.cot_log)
    print(f"[init] loading non-CoT log: {args.nocot_log}")
    nocot_recs = _load_records_keyed(args.nocot_log)
    print(f"[init] CoT decisions: {len(cot_recs)}; non-CoT: {len(nocot_recs)}")

    common_keys_loose = set(cot_recs.keys()) & set(nocot_recs.keys())
    print(f"[init] matched keys (seed × decision_idx): {len(common_keys_loose)}")

    # Diagnostic: how many of those have IDENTICAL game-state signatures
    # (board, pot, position, etc.) — these are the strict matches; the
    # rest match only on the dealt hand, not on the resulting game state
    # (e.g. CoT and non-CoT took different prior actions in this hand).
    identical_keys = set()
    for k in common_keys_loose:
        if _game_state_signature(cot_recs[k]) == _game_state_signature(nocot_recs[k]):
            identical_keys.add(k)
    print(f"[init] of those, identical game-state signature: "
          f"{len(identical_keys)} / {len(common_keys_loose)}")

    # Strict-vs-loose decision logic + fallback. If strict matching is
    # requested but produces no pairs (e.g. CoT and non-CoT took different
    # actions on every overlapping hand), fall back to loose matching with
    # a clear WARN rather than aborting. The published claim should still
    # report whether strict was used and how many strict pairs were
    # available — that goes into the SUMMARY below.
    matching_mode = "loose"
    fell_back = False
    if args.require_identical_game_state:
        if identical_keys:
            common_keys = identical_keys
            matching_mode = "strict"
            print(f"[init] --require-identical-game-state: kept "
                  f"{len(common_keys)} / {len(common_keys_loose)} (STRICT)")
        elif common_keys_loose:
            common_keys = common_keys_loose
            matching_mode = "loose-fallback"
            fell_back = True
            print(f"[WARN] --require-identical-game-state requested but "
                  f"strict matching produced 0 pairs. Falling back to "
                  f"LOOSE matching with all {len(common_keys_loose)} "
                  f"(seed, decision_idx) pairs. The matched cosine below "
                  f"is on pairs that share the dealt hand but may differ "
                  f"in game state (CoT/non-CoT took different prior "
                  f"actions). This is documented in the SUMMARY.")
        else:
            common_keys = set()
    else:
        common_keys = common_keys_loose

    if not common_keys:
        print("[abort] no matched keys; the CoT and non-CoT logs share no "
              "(seed, decision_idx) pairs.",
              file=sys.stderr)
        sys.exit(2)

    # Per-mode classification (recorded action -> bucket family).
    cot_kept = []   # list of (key, label) where label in {1=CHECK, 0=FOLD}
    nocot_kept = []
    for k in common_keys:
        cot_b = classify_decision(cot_recs[k])
        nocot_b = classify_decision(nocot_recs[k])
        # Keep only pairs where each mode's classification is one of the two
        # binary options (clean_CC or clean_LF). Note: the mode labels can
        # differ for the same hand — that's fine; each probe trains on its
        # own mode's labels, which is the correct apples-to-apples test.
        if cot_b not in ("clean_check_or_call", "clean_legal_fold"):
            continue
        if nocot_b not in ("clean_check_or_call", "clean_legal_fold"):
            continue
        cot_kept.append((k, 1 if cot_b == "clean_check_or_call" else 0))
        nocot_kept.append((k, 1 if nocot_b == "clean_check_or_call" else 0))

    if len(cot_kept) > args.max_pairs:
        # Stratified down-sampling to roughly balance CHECK and FOLD.
        import random
        rng = random.Random(args.seed)
        cot_check = [(k, l) for k, l in cot_kept if l == 1]
        cot_fold = [(k, l) for k, l in cot_kept if l == 0]
        per = args.max_pairs // 2
        rng.shuffle(cot_check); rng.shuffle(cot_fold)
        cot_check = cot_check[:per]; cot_fold = cot_fold[:per]
        keys_kept = {k for k, _ in cot_check} | {k for k, _ in cot_fold}
        cot_kept = [(k, l) for k, l in cot_kept if k in keys_kept]
        nocot_kept = [(k, l) for k, l in nocot_kept if k in keys_kept]

    # Second fallback: strict keys exist but none are CHECK/FOLD-classifiable.
    if (
        len(cot_kept) == 0
        and args.require_identical_game_state
        and matching_mode == "strict"
        and common_keys_loose
    ):
        fell_back = True
        matching_mode = "loose-fallback"
        common_keys = common_keys_loose
        cot_kept = []
        nocot_kept = []
        print(
            "[WARN] strict matching had "
            f"{len(identical_keys)} identical-state keys but 0 CHECK/FOLD "
            "classifiable pairs. Falling back to LOOSE matching on all "
            f"{len(common_keys_loose)} (seed, decision_idx) pairs."
        )
        for k in common_keys:
            cot_b = classify_decision(cot_recs[k])
            nocot_b = classify_decision(nocot_recs[k])
            if cot_b not in ("clean_check_or_call", "clean_legal_fold"):
                continue
            if nocot_b not in ("clean_check_or_call", "clean_legal_fold"):
                continue
            cot_kept.append((k, 1 if cot_b == "clean_check_or_call" else 0))
            nocot_kept.append((k, 1 if nocot_b == "clean_check_or_call" else 0))

    print(f"[init] matched-and-classified pairs: {len(cot_kept)}")
    print(f"  CoT bucket distribution:    "
          f"CHECK={sum(1 for _,l in cot_kept if l==1)} "
          f"FOLD={sum(1 for _,l in cot_kept if l==0)}")
    print(f"  non-CoT bucket distribution: "
          f"CHECK={sum(1 for _,l in nocot_kept if l==1)} "
          f"FOLD={sum(1 for _,l in nocot_kept if l==0)}")

    # Load model + tokenizer + agent configs.
    cot_cfg = _load_agent_config(args.cot_log)
    nocot_cfg = _load_agent_config(args.nocot_log)
    model_id = args.model_id or cot_cfg["model_id"]
    if cot_cfg.get("model_id") != nocot_cfg.get("model_id"):
        print(f"[abort] model mismatch between logs: "
              f"CoT={cot_cfg.get('model_id')}, non-CoT={nocot_cfg.get('model_id')}",
              file=sys.stderr)
        sys.exit(2)
    print(f"[init] model_id={model_id}, layer={args.layer}")

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

    # Capture residuals.
    print(f"\n[capture] CoT residuals at L={args.layer} ({len(cot_kept)} hands) ...")
    X_cot, y_cot = [], []
    t0 = time.time()
    for i, (key, lbl) in enumerate(cot_kept):
        r = _capture_residual_at_layer(
            model, tokenizer, cot_cfg, cot_recs[key], args.layer, args.device,
        )
        if r is None:
            continue
        X_cot.append(r); y_cot.append(lbl)
        if (i + 1) % 25 == 0:
            print(f"  [cot] {i+1}/{len(cot_kept)} ({time.time()-t0:.0f}s)")

    print(f"\n[capture] non-CoT residuals at L={args.layer} ({len(nocot_kept)} hands) ...")
    X_nocot, y_nocot = [], []
    t0 = time.time()
    for i, (key, lbl) in enumerate(nocot_kept):
        r = _capture_residual_at_layer(
            model, tokenizer, nocot_cfg, nocot_recs[key], args.layer, args.device,
        )
        if r is None:
            continue
        X_nocot.append(r); y_nocot.append(lbl)
        if (i + 1) % 25 == 0:
            print(f"  [nocot] {i+1}/{len(nocot_kept)} ({time.time()-t0:.0f}s)")

    if len(X_cot) < 10 or len(X_nocot) < 10:
        print(f"[abort] too few captures: CoT={len(X_cot)}, non-CoT={len(X_nocot)}",
              file=sys.stderr)
        sys.exit(3)

    Xc = torch.stack(X_cot, 0).to(torch.float32).numpy()
    Xn = torch.stack(X_nocot, 0).to(torch.float32).numpy()
    yc = np.array(y_cot, dtype=np.int64)
    yn = np.array(y_nocot, dtype=np.int64)
    print(f"\n[shape] X_cot={Xc.shape}  X_nocot={Xn.shape}")

    # Train per-mode probes.
    print(f"\n[probe] training CoT probe ...")
    clf_c = LogisticRegression(C=args.l2_c, max_iter=2000, solver="lbfgs",
                               random_state=args.seed)
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                          random_state=args.seed)
    cv_c = cross_val_score(clf_c, Xc, yc, cv=skf, scoring="accuracy")
    clf_c.fit(Xc, yc)
    w_c = clf_c.coef_[0]

    print(f"[probe] training non-CoT probe ...")
    clf_n = LogisticRegression(C=args.l2_c, max_iter=2000, solver="lbfgs",
                               random_state=args.seed)
    cv_n = cross_val_score(clf_n, Xn, yn, cv=skf, scoring="accuracy")
    clf_n.fit(Xn, yn)
    w_n = clf_n.coef_[0]

    # Cosine + cross-projection.
    cos_w = float(w_c @ w_n / (np.linalg.norm(w_c) * np.linalg.norm(w_n) + 1e-9))

    cd_c = Xc[yc == 1].mean(0) - Xc[yc == 0].mean(0) if (yc == 1).any() and (yc == 0).any() else None
    cd_n = Xn[yn == 1].mean(0) - Xn[yn == 0].mean(0) if (yn == 1).any() and (yn == 0).any() else None
    cos_cd = (
        float(cd_c @ cd_n / (np.linalg.norm(cd_c) * np.linalg.norm(cd_n) + 1e-9))
        if cd_c is not None and cd_n is not None else None
    )

    # Cross-projection: project nocot residuals onto CoT direction.
    proj_n_under_wc = Xn @ w_c / (np.linalg.norm(w_c) + 1e-9)
    proj_c_under_wn = Xc @ w_n / (np.linalg.norm(w_n) + 1e-9)
    diff_n_under_wc = float(proj_n_under_wc[yn == 1].mean() - proj_n_under_wc[yn == 0].mean()) if (yn == 1).any() and (yn == 0).any() else None
    diff_c_under_wn = float(proj_c_under_wn[yc == 1].mean() - proj_c_under_wn[yc == 0].mean()) if (yc == 1).any() and (yc == 0).any() else None

    print(f"\n[result] mode-balanced cosines:")
    print(f"  CoT CV accuracy:    {cv_c.mean():.3f} ± {cv_c.std():.3f}")
    print(f"  nonCoT CV accuracy: {cv_n.mean():.3f} ± {cv_n.std():.3f}")
    print(f"  cos(w_CoT, w_nonCoT):                {cos_w:+.4f}")
    if cos_cd is not None:
        print(f"  cos(centroid_CoT, centroid_nonCoT):  {cos_cd:+.4f}")
    if diff_n_under_wc is not None:
        print(f"  diff(non-CoT proj under w_CoT):      {diff_n_under_wc:+.4f}")
    if diff_c_under_wn is not None:
        print(f"  diff(CoT proj under w_nonCoT):       {diff_c_under_wn:+.4f}")

    # Save.
    summary = {
        "cot_log": args.cot_log,
        "nocot_log": args.nocot_log,
        "model_id": model_id,
        "layer": args.layer,
        "matching": {
            "key": "(seed, decision_idx)",
            "n_loose": len(common_keys_loose),
            "n_strict_identical_game_state": len(identical_keys),
            "require_identical_game_state_requested": bool(args.require_identical_game_state),
            "mode_used": matching_mode,
            "fell_back_to_loose": bool(fell_back),
        },
        "matched_pairs_attempted": len(cot_kept),
        "matched_pairs_captured": {"cot": len(X_cot), "nocot": len(X_nocot)},
        "cv_folds": args.cv_folds,
        "cv_accuracy": {
            "cot": {"mean": float(cv_c.mean()), "std": float(cv_c.std())},
            "nocot": {"mean": float(cv_n.mean()), "std": float(cv_n.std())},
        },
        "cos_w_cot_w_nocot": cos_w,
        "cos_centroid_diff": cos_cd,
        "diff_nocot_proj_under_wcot": diff_n_under_wc,
        "diff_cot_proj_under_wnocot": diff_c_under_wn,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(
        out_dir / "raw.npz",
        X_cot=Xc, y_cot=yc, X_nocot=Xn, y_nocot=yn,
        w_cot=w_c, w_nocot=w_n,
    )

    md = []
    md.append("# Mode-balanced direction probe (CoT vs non-CoT, hand-matched)")
    md.append("")
    md.append(f"- Model: `{model_id}`")
    md.append(f"- Layer: **{args.layer}**")
    md.append(f"- CoT log:    `{args.cot_log}`")
    md.append(f"- non-CoT log: `{args.nocot_log}`")
    md.append(f"- Match key: `(seed, decision_idx)` "
              f"(NOTE: not `hand_id` — hand_id is a random UUID, see `poker_env/env.py:188`)")
    md.append(f"- Matched pairs (loose, same dealt hand): "
              f"**{len(common_keys_loose)}**")
    md.append(f"- Of those, identical game-state signature (board + pot + "
              f"position + hole_cards): **{len(identical_keys)}**")
    md.append(f"- `--require-identical-game-state` requested: "
              f"`{bool(args.require_identical_game_state)}`")
    md.append(f"- **Matching mode used: `{matching_mode}`** "
              + ("(STRICT — game state byte-identical across modes)"
                 if matching_mode == "strict"
                 else ("(LOOSE-FALLBACK — strict matching produced 0 pairs; "
                       "fell back to same-dealt-hand matching. Pairs may "
                       "differ in game state because CoT and non-CoT took "
                       "different prior actions.)" if fell_back
                       else "(LOOSE — same dealt hand, may differ in game state)")))
    md.append(f"- Pairs attempted in probe (after per-mode classification): "
              f"{len(cot_kept)}")
    md.append(f"- Captured residuals: CoT {len(X_cot)}, non-CoT {len(X_nocot)}")
    md.append(f"- Hidden dim: {Xc.shape[1]}")
    md.append("")
    if fell_back:
        md.append("> ⚠️ **Note**: this run requested strict (identical-game-state) "
                  "matching but found zero strict pairs (most likely because "
                  "CoT and non-CoT took different actions on every overlapping "
                  "hand, so the game state diverged after decision_idx=1). "
                  "The probe was trained on the loose-matched pairs instead. "
                  "Treat the matched cosine below as a SAME-DEALT-HAND cosine, "
                  "not a SAME-GAME-STATE cosine.")
        md.append("")
    md.append("## Per-mode probe accuracy (5-fold CV)")
    md.append("")
    md.append(f"- CoT: **{cv_c.mean():.3f} ± {cv_c.std():.3f}**")
    md.append(f"- non-CoT: **{cv_n.mean():.3f} ± {cv_n.std():.3f}**")
    md.append("")
    md.append("## Cosines on matched data")
    md.append("")
    md.append(f"- **cos(w_CoT, w_nonCoT) = {cos_w:+.4f}**")
    if cos_cd is not None:
        md.append(f"- cos(centroid_CoT, centroid_nonCoT) = {cos_cd:+.4f}")
    md.append("")
    md.append("## Cross-projection (mean CHECK − FOLD when projecting onto the OTHER mode's direction)")
    md.append("")
    if diff_n_under_wc is not None:
        sign = "✅" if diff_n_under_wc > 0 else "❌"
        md.append(f"- non-CoT residuals projected onto w_CoT: **{diff_n_under_wc:+.3f}** {sign}")
    if diff_c_under_wn is not None:
        sign = "✅" if diff_c_under_wn > 0 else "❌"
        md.append(f"- CoT residuals projected onto w_nonCoT: **{diff_c_under_wn:+.3f}** {sign}")
    md.append("")
    md.append("## Reading guide")
    md.append("")
    md.append("Compare to the unmatched cosines from `direction_cosine_compare/` "
              "(Phase M §18b: Llama 0.27, Qwen 0.34). If the matched cosine is "
              "**substantially higher** (≥0.6), the §18b non-identity was "
              "primarily a data-distribution-shift artifact — directions are "
              "much closer than they appeared. If matched cosine is **similar** "
              "(0.2–0.4), the direction tilt is real even with hand population "
              "controlled — the model represents the verb decision along "
              "mode-specific axes regardless of which hands you sample.")
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[done] wrote {out_dir / 'SUMMARY.md'}")
    print(f"[done] wrote {out_dir / 'summary.json'}")
    print(f"[done] wrote {out_dir / 'raw.npz'}")


if __name__ == "__main__":
    main()
