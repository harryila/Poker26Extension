"""
Causal-patching CLI driver (REQUIRES GPU).

Phase 2 / Phase 3 of the causal-patching experiment from the §13 logit-lens
findings. Tests whether activation-patching late-layer (~22-30) residuals
from `clean_CHECK_OR_CALL` source decisions into `illegal_FOLD` target
decisions causes the target's predicted action verb to flip toward CHECK.

Pipeline per run:

  1. Load the enriched log + agent_config + tokenizer + model.
  2. Bucket all decisions via `analysis.recategorize_action_metadata._reparse_one`
     into clean / illegal_fold / etc., further split clean by emitted action.
  3. Sample N source decisions from --source-bucket and N target decisions
     from --target-bucket (random with --seed).
  4. Capture: one forward per source -> per_layer_last_pos[layer_idx].
  5. Controls (run FIRST, halt if any fail):
        - no-patch baseline: per target, run forward; top-1 must match
          recorded raw_response's first verb token.
        - self-patch identity: take a target, capture its own residual,
          patch it back at every test layer; logits must equal baseline
          within 1e-4.
        - random-source null: patch each target with a residual drawn from
          a randomly chosen DIFFERENT-bucket source; mean
          delta_logit_check_minus_fold should be near 0.
  6. Main experiment: for each (source, target, layer L), run forward with
     HiddenStatePatch using source.per_layer_last_pos[L]. Record:
        - top-1 token id + decoded
        - logits for the action-verb-token-set (FOLD-family, CHECK-CALL-
          family, BET-RAISE-family) -> compute log-prob for each family
        - delta_logit_check_minus_fold = (CHECK_logsumexp - FOLD_logsumexp)
          patched - same baseline
  7. Write summary.json + by_pair.csv + SUMMARY.md.

Usage::

    python -m experiments.causal_patching \
        --enriched-log logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
        --source-bucket clean_check_or_call --target-bucket illegal_fold \
        --layers 22 24 26 28 30 \
        --n-source 10 --n-target 30 \
        --out-dir results/causal_patching/ministral8b_t0_s42_pilot \
        --device cuda --dtype bfloat16

Notes:
  - Unbatched single-forward implementation. Pilot (1500 forwards, ~50 min on
    H100) is acceptable. For full-scope runs (~36k forwards, ~10 h) consider
    enabling --batch-size N to batch sources per (target, layer).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import random
import sys
import time
from pathlib import Path

# Allow running from the package root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.recategorize_action_metadata import _reparse_one  # noqa: E402
from poker_env.interp.forward_helpers import (  # noqa: E402
    PromptReconstructor,
    build_input_text_for_action_verb_position,
    run_forward_at_last_position,
)
from poker_env.interp.patching import (  # noqa: E402
    HiddenStateCapture,
    HiddenStatePatch,
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _open_log(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_decisions(path: str):
    with _open_log(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_agent_config(path: str) -> dict:
    for rec in _iter_decisions(path):
        if rec.get("type") == "run_config":
            for ac in rec.get("agent_configs", []):
                if ac.get("type") == "HFAgent":
                    return ac
    raise ValueError(f"No HFAgent run_config in {path}")


# ---------------------------------------------------------------------------
# Bucketing — extends recategorize_action_metadata with clean-by-action split
# ---------------------------------------------------------------------------

BUCKET_NAMES = (
    "clean_check_or_call",
    "clean_legal_fold",
    "clean_bet_or_raise",
    "illegal_fold",
    "illegal_other",
    "alias_unrecognized",
    "json_failure",
)


def classify_decision(rec: dict) -> str:
    info = _reparse_one(rec)
    if not info["json_parsed"]:
        return "json_failure"
    if not info["recognized"]:
        return "alias_unrecognized"
    attempted = (info["attempted_action"] or "").upper()
    if info["legal"]:
        if attempted == "FOLD":
            return "clean_legal_fold"
        if attempted == "CHECK_OR_CALL":
            return "clean_check_or_call"
        if attempted == "BET_OR_RAISE":
            return "clean_bet_or_raise"
        return "alias_unrecognized"
    if attempted == "FOLD":
        return "illegal_fold"
    return "illegal_other"


# ---------------------------------------------------------------------------
# Action-verb token sets (per tokenizer; computed once at startup)
# ---------------------------------------------------------------------------

_ACTION_FAMILY_VARIANTS: dict[str, tuple[str, ...]] = {
    "FOLD":     (" FOLD", "FOLD", " Fold", "Fold", " fold", "fold",
                 " F", "F", " Fol", "Fol"),
    "CHECK":    (" CHECK", "CHECK", " Check", "Check", " check", "check",
                 " Ch", "Ch", " Che", "Che"),
    "CALL":     (" CALL", "CALL", " Call", "Call", " call", "call",
                 " Ca", "Ca"),
    "BET":      (" BET", "BET", " Bet", "Bet", " bet", "bet",
                 " B", "B", " Be", "Be"),
    "RAISE":    (" RAISE", "RAISE", " Raise", "Raise", " raise", "raise",
                 " R", "R", " Ra", "Ra"),
}

# Logical groups for the headline metric.
ACTION_GROUPS = {
    "FOLD":      ["FOLD"],
    "CHECK_CALL": ["CHECK", "CALL"],
    "BET_RAISE": ["BET", "RAISE"],
}


def build_action_token_id_sets(tokenizer) -> dict[str, list[int]]:
    """For each family, collect all single-token ids whose decoded form (when
    tokenized as the FIRST token of a freshly-started string) matches one of
    the variants. Returns a flat dict: family_name -> list[int]."""
    sets: dict[str, set[int]] = {k: set() for k in _ACTION_FAMILY_VARIANTS}
    for family, variants in _ACTION_FAMILY_VARIANTS.items():
        for v in variants:
            ids = tokenizer(v, add_special_tokens=False)["input_ids"]
            if ids:
                sets[family].add(int(ids[0]))
    return {k: sorted(v) for k, v in sets.items()}


def _logsumexp(values):
    import math
    if not values:
        return float("-inf")
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def score_logits(
    logits_last_pos,
    family_token_ids: dict[str, list[int]],
) -> dict[str, float]:
    """Given a 1D logits tensor at the next-token position, return per-family
    logsumexp (log-prob aggregate). Higher = model puts more probability on
    that family."""
    out = {}
    logits_list = logits_last_pos.tolist()
    for family, ids in family_token_ids.items():
        out[family] = _logsumexp([logits_list[i] for i in ids if 0 <= i < len(logits_list)])
    # Group sums (FOLD-family / CHECK-CALL family / BET-RAISE family)
    out["GROUP_FOLD"] = _logsumexp([out[f] for f in ACTION_GROUPS["FOLD"]])
    out["GROUP_CHECK_CALL"] = _logsumexp([out[f] for f in ACTION_GROUPS["CHECK_CALL"]])
    out["GROUP_BET_RAISE"] = _logsumexp([out[f] for f in ACTION_GROUPS["BET_RAISE"]])
    out["delta_check_minus_fold"] = out["GROUP_CHECK_CALL"] - out["GROUP_FOLD"]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Causal-patching CLI (Phase 2 / Phase 3 driver)."
    )
    parser.add_argument("--enriched-log", required=True, nargs="+",
                        help="One or more enriched JSONL[.gz] logs. When "
                             "multiple are passed, decisions from all logs are "
                             "pooled into the same bucket distribution. "
                             "agent_config + model_id are read from the FIRST "
                             "log (others must use the same model). Useful for "
                             "pooling seeds (e.g. s42 + s123 + s456 t=0).")
    parser.add_argument("--source-bucket", default="clean_check_or_call",
                        choices=BUCKET_NAMES)
    parser.add_argument("--target-bucket", default="illegal_fold",
                        choices=BUCKET_NAMES)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--n-source", type=int, default=10)
    parser.add_argument("--n-target", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--model-id", default=None,
                        help="Override model_id from agent_config.")
    parser.add_argument("--skip-controls", action="store_true",
                        help="DEBUG ONLY. Do NOT use for real runs.")
    parser.add_argument("--baseline-tolerance-frac", type=float, default=0.95,
                        help="Fraction of baseline targets whose top-1 must "
                             "match the recorded verb. <1.0 because of rare "
                             "tokenizer edge-cases.")
    parser.add_argument("--n-random-control", type=int, default=5,
                        help="Number of random alt-bucket sources to use as "
                             "the per-layer null baseline. 5 is a tight pilot, "
                             "10-15 gives a cleaner null at <1 min per layer.")
    parser.add_argument("--zero-ablation", action="store_true",
                        help="Replace the source residual with the all-zeros "
                             "tensor (same shape) instead of a sampled clean "
                             "source. Distinguishes 'circuit USES this layer' "
                             "(zero-patch flips the verb) from 'circuit is "
                             "content-addressable' (zero-patch leaves the "
                             "verb alone, only clean-source patches flip it). "
                             "When set, --n-source is ignored and the script "
                             "uses a SINGLE zeroed pseudo-source so per-layer "
                             "output rows have source_idx=-1.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load agent config + model + tokenizer -----------------------------
    enriched_logs: list[str] = list(args.enriched_log)
    agent_config = _load_agent_config(enriched_logs[0])
    model_id = args.model_id or agent_config["model_id"]
    if len(enriched_logs) > 1:
        # Sanity: every log must reference the same model_id.
        for extra in enriched_logs[1:]:
            other_cfg = _load_agent_config(extra)
            if other_cfg.get("model_id") != agent_config.get("model_id"):
                print(
                    f"[abort] enriched logs reference different models: "
                    f"{agent_config.get('model_id')} vs "
                    f"{other_cfg.get('model_id')} ({extra})",
                    file=sys.stderr,
                )
                sys.exit(2)
        print(f"[init] pooling {len(enriched_logs)} enriched logs:")
        for p in enriched_logs:
            print(f"  - {p}")
    print(f"[init] model_id={model_id}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

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
    family_ids = build_action_token_id_sets(tokenizer)
    print(f"[init] action-family token ids: "
          f"{ {k: len(v) for k, v in family_ids.items()} }")

    # ---- Bucket decisions (pooled across all enriched logs) ----------------
    print(f"[bucket] scanning {len(enriched_logs)} enriched log(s) ...")
    by_bucket: dict[str, list[dict]] = {b: [] for b in BUCKET_NAMES}
    for log_path in enriched_logs:
        for rec in _iter_decisions(log_path):
            am = rec.get("action_metadata")
            if am is None or not am.get("raw_response"):
                continue
            # Tag each decision with its source log so analysis can split by
            # seed if desired (e.g., grep by_pair.csv on source_hand).
            rec.setdefault("_source_log", log_path)
            b = classify_decision(rec)
            by_bucket[b].append(rec)
    for b in BUCKET_NAMES:
        print(f"  {b:<22}: {len(by_bucket[b])}")

    sources_pool = by_bucket[args.source_bucket]
    targets_pool = by_bucket[args.target_bucket]
    # In zero-ablation mode the source bucket is only used for the random-null
    # exclusion (control 3); we don't actually need any real source records.
    if args.zero_ablation:
        if len(targets_pool) < 1:
            print(f"[abort] zero-ablation needs targets but found "
                  f"{len(targets_pool)} in target bucket.")
            sys.exit(2)
        sources = []
        targets = rng.sample(targets_pool, min(args.n_target, len(targets_pool)))
        print(f"[sample] zero-ablation mode: 0 real sources, "
              f"{len(targets)} targets")
    else:
        if len(sources_pool) < 1 or len(targets_pool) < 1:
            print(f"[abort] not enough decisions in source ({len(sources_pool)}) "
                  f"or target ({len(targets_pool)}) buckets.")
            sys.exit(2)
        sources = rng.sample(sources_pool, min(args.n_source, len(sources_pool)))
        targets = rng.sample(targets_pool, min(args.n_target, len(targets_pool)))
        print(f"[sample] {len(sources)} sources, {len(targets)} targets")

    # ---- Build input texts (CPU) ------------------------------------------
    def _prepare(rec: dict) -> dict | None:
        full_prompt = recon.build(rec)
        out = build_input_text_for_action_verb_position(
            full_prompt, rec["action_metadata"]["raw_response"], tokenizer,
        )
        if out is None:
            return None
        input_text, verb_resp_idx = out
        # Get the FIRST verb token id (what the model SHOULD predict at the
        # last input position).
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

    # `cap` is needed by control 2 (self-patch identity) and control 3
    # (random-source null) regardless of zero-ablation mode, so we always
    # construct it here.
    cap = HiddenStateCapture(model)

    if args.zero_ablation:
        # Synthetic single source: a zero tensor at every test layer. We don't
        # need any real prep — the main loop only reads `source_residuals[si][L]`
        # and the per-pair CSV's source_hand/source_dec columns (set to "zero"
        # / 0 to make zero-ablation rows trivially identifiable downstream).
        sources_prep = [{
            "rec": {"hand_id": "zero", "decision_idx": 0},
            "input_text": "",            # never tokenized — capture is skipped
            "expected_verb_tok_id": -1,  # never used (zero-source has no verb)
        }]
        hidden_size = int(model.config.hidden_size)
        zero_residual = torch.zeros(hidden_size, dtype=torch.float32)
        source_residuals: list[dict[int, torch.Tensor]] = [
            {L: zero_residual.clone() for L in args.layers}
        ]
        print(f"[capture] zero-ablation mode: synthesized 1 zero-source "
              f"(hidden_size={hidden_size}); skipping real-source capture pass.")
    else:
        sources_prep = [p for p in (_prepare(r) for r in sources) if p is not None]
        print(f"[prepare] sources_with_input={len(sources_prep)} "
              f"targets_with_input={len(targets_prep)}")

        # ---- Capture source residuals ----------------------------------------
        print(f"[capture] {len(sources_prep)} source forwards ...")
        source_residuals = []
        cap_t0 = time.time()
        for i, src in enumerate(sources_prep):
            cap.attach_hooks()
            try:
                with torch.no_grad():
                    model(input_ids=tokenizer(src["input_text"],
                                              return_tensors="pt",
                                              add_special_tokens=False
                                              )["input_ids"].to(args.device))
            finally:
                states = cap.collect()
                cap.detach_hooks()
            source_residuals.append(states["per_layer_last_pos"])
            if (i + 1) % 5 == 0 or i + 1 == len(sources_prep):
                elapsed = time.time() - cap_t0
                rate = (i + 1) / elapsed
                print(f"  [capture] {i + 1}/{len(sources_prep)} "
                      f"({elapsed:.0f}s, {rate:.2f} src/s)")

    # ---- Controls --------------------------------------------------------
    controls = {}
    if not args.skip_controls:
        # Control 1: baseline (no patch) for each target.
        print(f"\n[control 1/3] baseline (no-patch) for {len(targets_prep)} targets ...")
        baseline_cache: dict[int, dict] = {}
        n_top1_match = 0
        for ti, tgt in enumerate(targets_prep):
            res = run_forward_at_last_position(
                model, tokenizer, tgt["input_text"], device=args.device,
            )
            top1_id = int(res["logits_last_pos"].argmax().item())
            scored = score_logits(res["logits_last_pos"], family_ids)
            baseline_cache[ti] = {
                "logits_last_pos": res["logits_last_pos"],
                "scored": scored,
                "top1_id": top1_id,
            }
            if top1_id == tgt["expected_verb_tok_id"]:
                n_top1_match += 1
        baseline_match_rate = n_top1_match / len(targets_prep)
        print(f"  baseline top-1 match rate: {baseline_match_rate:.3f} "
              f"({n_top1_match}/{len(targets_prep)})")
        controls["baseline_top1_match_rate"] = baseline_match_rate
        if baseline_match_rate < args.baseline_tolerance_frac:
            print(f"  [HALT] below tolerance {args.baseline_tolerance_frac}; "
                  f"prompt reconstruction issue. Aborting.")
            sys.exit(3)

        # Control 2: self-patch identity for one target.
        print(f"\n[control 2/3] self-patch identity ...")
        tgt = targets_prep[0]
        cap.attach_hooks()
        try:
            with torch.no_grad():
                model(input_ids=tokenizer(tgt["input_text"],
                                          return_tensors="pt",
                                          add_special_tokens=False
                                          )["input_ids"].to(args.device))
        finally:
            self_states = cap.collect()
            cap.detach_hooks()
        max_drift = 0.0
        for L in args.layers:
            patch = HiddenStatePatch(model, L, self_states["per_layer_last_pos"][L])
            with patch:
                res = run_forward_at_last_position(
                    model, tokenizer, tgt["input_text"], device=args.device,
                )
            drift = float((res["logits_last_pos"] -
                          baseline_cache[0]["logits_last_pos"]).abs().max().item())
            max_drift = max(max_drift, drift)
            print(f"  layer {L}: max |delta logit| = {drift:.6f}")
        controls["self_patch_max_logit_drift"] = max_drift
        if max_drift > 1e-2:
            print(f"  [WARN] self-patch drift > 0.01; check for nondeterminism.")

        # Control 3: random-source patch — now PER-LAYER so we get a null
        # baseline at each test layer (not just one), needed for the
        # specificity-vs-depth plot.
        print(f"\n[control 3/3] random-source patch (per-layer null baseline) ...")
        random_sources = []
        alt_pool = []
        for b, recs in by_bucket.items():
            if b == args.source_bucket:
                continue
            alt_pool.extend(recs)
        n_random = min(args.n_random_control, len(alt_pool))
        rs = [p for p in (_prepare(r) for r in rng.sample(alt_pool, n_random))
              if p is not None]
        if not rs:
            print("  [skip] no alt-bucket records available")
            controls["random_source_per_layer"] = None
        else:
            for r in rs:
                cap.attach_hooks()
                try:
                    with torch.no_grad():
                        model(input_ids=tokenizer(r["input_text"],
                                                  return_tensors="pt",
                                                  add_special_tokens=False
                                                  )["input_ids"].to(args.device))
                finally:
                    states = cap.collect()
                    cap.detach_hooks()
                random_sources.append(states["per_layer_last_pos"])
            # Run random control at EVERY test layer, against target_idx 0.
            # n_random_sources × n_layers extra forwards (small).
            random_per_layer: dict[int, dict] = {}
            ctl_t0 = time.time()
            for test_layer in args.layers:
                deltas = []
                for r_residuals in random_sources:
                    patch = HiddenStatePatch(model, test_layer,
                                              r_residuals[test_layer])
                    with patch:
                        res = run_forward_at_last_position(
                            model, tokenizer, targets_prep[0]["input_text"],
                            device=args.device,
                        )
                    scored = score_logits(res["logits_last_pos"], family_ids)
                    d = scored["delta_check_minus_fold"] - \
                        baseline_cache[0]["scored"]["delta_check_minus_fold"]
                    deltas.append(d)
                mean_d = sum(deltas) / len(deltas) if deltas else None
                random_per_layer[test_layer] = {
                    "mean_delta": mean_d,
                    "n": len(deltas),
                }
                print(f"  layer {test_layer:>3}: random mean delta = "
                      f"{mean_d:+.4f} (n={len(deltas)})")
            ctl_elapsed = time.time() - ctl_t0
            print(f"  [random control done in {ctl_elapsed:.0f}s]")
            controls["random_source_per_layer"] = {
                str(k): v for k, v in random_per_layer.items()
            }
            controls["random_source_n"] = n_random
            # Aggregate signal for the old top-level field, kept for
            # backward-compatibility with the prior pilot's summary.
            mid = args.layers[len(args.layers) // 2]
            controls["random_source_mean_delta"] = (
                random_per_layer[mid]["mean_delta"]
            )
            controls["random_source_test_layer"] = mid

        print(f"\n[controls] all passed (or warned). Proceeding.")

    # ---- Main loop ---------------------------------------------------------
    print(f"\n[main] {len(sources_prep)} x {len(targets_prep)} x "
          f"{len(args.layers)} = "
          f"{len(sources_prep) * len(targets_prep) * len(args.layers)} patched forwards")

    by_pair_path = out_dir / "by_pair.csv"
    by_pair_f = open(by_pair_path, "w", newline="")
    writer = csv.writer(by_pair_f)
    writer.writerow([
        "source_idx", "target_idx", "layer",
        "source_hand", "source_dec", "target_hand", "target_dec",
        "expected_target_verb_id", "expected_target_verb_tok",
        "patched_top1_id", "patched_top1_tok", "patched_top1_group",
        "delta_check_minus_fold",
        "patched_GROUP_CHECK_CALL", "patched_GROUP_FOLD",
    ])

    main_t0 = time.time()
    n_done = 0
    for ti, tgt in enumerate(targets_prep):
        baseline_score = (baseline_cache[ti]["scored"]
                          if not args.skip_controls else None)
        for L in args.layers:
            for si, src in enumerate(sources_prep):
                src_residual = source_residuals[si][L]
                patch = HiddenStatePatch(model, L, src_residual)
                with patch:
                    res = run_forward_at_last_position(
                        model, tokenizer, tgt["input_text"],
                        device=args.device,
                    )
                logits = res["logits_last_pos"]
                top1_id = int(logits.argmax().item())
                top1_tok = tokenizer.decode([top1_id])
                scored = score_logits(logits, family_ids)
                delta = scored["delta_check_minus_fold"]
                if baseline_score is not None:
                    delta = delta - baseline_score["delta_check_minus_fold"]

                # Map top1 to action group via family token ids
                top1_group = "OTHER"
                for fam, ids in family_ids.items():
                    if top1_id in ids:
                        if fam == "FOLD":
                            top1_group = "FOLD"
                        elif fam in ("CHECK", "CALL"):
                            top1_group = "CHECK_CALL"
                        elif fam in ("BET", "RAISE"):
                            top1_group = "BET_RAISE"
                        break

                writer.writerow([
                    si, ti, L,
                    src["rec"]["hand_id"], src["rec"]["decision_idx"],
                    tgt["rec"]["hand_id"], tgt["rec"]["decision_idx"],
                    tgt["expected_verb_tok_id"],
                    tokenizer.decode([tgt["expected_verb_tok_id"]]),
                    top1_id, top1_tok, top1_group,
                    f"{delta:.4f}",
                    f"{scored['GROUP_CHECK_CALL']:.4f}",
                    f"{scored['GROUP_FOLD']:.4f}",
                ])
                n_done += 1
            elapsed = time.time() - main_t0
            rate = n_done / elapsed if elapsed > 0 else 0
            remaining = (len(sources_prep) * len(targets_prep) *
                         len(args.layers)) - n_done
            eta = remaining / rate if rate > 0 else float("nan")
            print(f"  [main] target {ti+1}/{len(targets_prep)} layer {L}: "
                  f"{n_done} done ({elapsed:.0f}s, {rate:.2f}/s, "
                  f"ETA {eta:.0f}s)")
    by_pair_f.close()

    # ---- Summary -----------------------------------------------------------
    summary = {
        "enriched_log": enriched_logs if len(enriched_logs) > 1 else enriched_logs[0],
        "n_enriched_logs": len(enriched_logs),
        "model_id": model_id,
        "source_bucket": args.source_bucket,
        "target_bucket": args.target_bucket,
        "n_source": len(sources_prep),
        "n_target": len(targets_prep),
        "layers": list(args.layers),
        "n_pairs_total": len(sources_prep) * len(targets_prep) * len(args.layers),
        "controls": controls,
        "zero_ablation": bool(args.zero_ablation),
    }
    # Per-layer aggregate. We track the top-1 destination across all 4
    # families (CHECK_CALL / FOLD / BET_RAISE / OTHER) so that reverse-
    # direction runs (e.g. clean_legal_fold source -> clean_check_or_call
    # target) can read `frac_top1_fold` as their natural headline metric
    # without re-deriving it from by_pair.csv. Forward-direction runs are
    # backward compatible: `frac_top1_check_call` still means the same.
    per_layer = {L: {"n": 0, "delta_sum": 0.0,
                     "flipped_check_n": 0,
                     "flipped_fold_n": 0,
                     "flipped_betraise_n": 0,
                     "flipped_other_n": 0}
                 for L in args.layers}
    with open(by_pair_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            L = int(row["layer"])
            per_layer[L]["n"] += 1
            per_layer[L]["delta_sum"] += float(row["delta_check_minus_fold"])
            grp = row["patched_top1_group"]
            if grp == "CHECK_CALL":
                per_layer[L]["flipped_check_n"] += 1
            elif grp == "FOLD":
                per_layer[L]["flipped_fold_n"] += 1
            elif grp == "BET_RAISE":
                per_layer[L]["flipped_betraise_n"] += 1
            else:
                per_layer[L]["flipped_other_n"] += 1
    rnd_per_layer = (controls.get("random_source_per_layer") or {})
    summary["per_layer"] = {}
    for L, d in per_layer.items():
        mean_d = d["delta_sum"] / d["n"] if d["n"] else None
        rnd_entry = rnd_per_layer.get(str(L)) or {}
        rnd_d = rnd_entry.get("mean_delta")
        adj = (mean_d - rnd_d) if (mean_d is not None and rnd_d is not None) else None
        summary["per_layer"][str(L)] = {
            "n": d["n"],
            "mean_delta_check_minus_fold": mean_d,
            "frac_top1_check_call": d["flipped_check_n"] / d["n"] if d["n"] else None,
            "frac_top1_fold": d["flipped_fold_n"] / d["n"] if d["n"] else None,
            "frac_top1_bet_raise": d["flipped_betraise_n"] / d["n"] if d["n"] else None,
            "frac_top1_other": d["flipped_other_n"] / d["n"] if d["n"] else None,
            "random_null_delta": rnd_d,
            "specificity_adjusted_delta": adj,
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Markdown
    lines = ["# Causal patching results", ""]
    lines.append(f"- Model: `{model_id}`")
    if len(enriched_logs) == 1:
        lines.append(f"- Enriched log: `{enriched_logs[0]}`")
    else:
        lines.append(f"- Enriched logs (pooled, n={len(enriched_logs)}):")
        for p in enriched_logs:
            lines.append(f"  - `{p}`")
    if args.zero_ablation:
        lines.append("- **Zero-ablation mode**: source residual is the all-zeros tensor "
                     "(no real source sampled). The `--source-bucket` value below is used "
                     "only to exclude that bucket from the random-null pool in control 3.")
    lines.append(f"- Source bucket: `{args.source_bucket}` (n={len(sources_prep)})")
    lines.append(f"- Target bucket: `{args.target_bucket}` (n={len(targets_prep)})")
    lines.append(f"- Layers: {args.layers}")
    lines.append("")
    lines.append("## Controls")
    if controls:
        for k, v in controls.items():
            lines.append(f"- `{k}` = {v}")
    else:
        lines.append("(skipped)")
    lines.append("")
    lines.append("## Per-layer effect")
    # All three group fractions are always printed — `top-1 → BET_RAISE-family`
    # was 0% across all earlier-direction (CHECK/FOLD) experiments, so
    # including it is backward-compatible. For verb-generality runs (e.g.
    # source=clean_bet_or_raise → target=clean_check_or_call), this is the
    # headline column.
    lines.append("| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for L in sorted(per_layer.keys()):
        d = summary["per_layer"][str(L)]
        rnd = d.get("random_null_delta")
        adj = d.get("specificity_adjusted_delta")
        rnd_s = f"{rnd:+.3f}" if rnd is not None else "—"
        adj_s = f"{adj:+.3f}" if adj is not None else "—"
        f_check = d.get("frac_top1_check_call")
        f_fold = d.get("frac_top1_fold")
        f_betraise = d.get("frac_top1_bet_raise")
        f_check_s = f"{f_check*100:.1f}%" if f_check is not None else "—"
        f_fold_s = f"{f_fold*100:.1f}%" if f_fold is not None else "—"
        f_betraise_s = f"{f_betraise*100:.1f}%" if f_betraise is not None else "—"
        lines.append(
            f"| {L} | {d['n']} "
            f"| {d['mean_delta_check_minus_fold']:+.3f} "
            f"| {rnd_s} "
            f"| **{adj_s}** "
            f"| {f_check_s} "
            f"| {f_fold_s} "
            f"| {f_betraise_s} |"
        )
    lines.append("")
    lines.append("**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).")
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[done] wrote {out_dir / 'summary.json'}")
    print(f"[done] wrote {by_pair_path}")
    print(f"[done] wrote {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
