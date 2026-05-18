"""
Component-level causal-patching CLI driver (REQUIRES GPU).

Companion to ``experiments.causal_patching``. Where the residual-stream
driver answers "AT WHICH LAYER is the action-verb decision encoded?", this
driver answers "AT THAT LAYER, WHICH SUBLAYER (attention vs MLP) — and
which attention HEADS — mediate the effect?".

Single forward pass per source captures everything needed (residual,
attn_out, mlp_out, per-head pre-o_proj at every layer). Single forward pass
per (target × component-mode) does the patched scoring. No model reload
between component modes — that's the wall-clock win over running the
residual driver with separate ``--component`` invocations.

Component modes (each row in the output table corresponds to one):
  - ``residual``: the existing experiment baseline; replaces the post-layer
    residual at the last position. Same as ``causal_patching.py``'s default.
    Run for sanity comparison against existing pooled sweep numbers.
  - ``attn``: replaces ONLY the attention sublayer's output at the last
    position. Tests "is attention carrying the signal?".
  - ``mlp``: replaces ONLY the MLP sublayer's output at the last position.
    Tests "is the MLP carrying the signal?".
  - ``head_<i>``: replaces ONLY the i-th attention head's pre-o_proj output.
    Other heads pass through unchanged. One row per head index in
    ``--head-indices`` (use ``--head-indices all`` for all heads).
  - ``head_subset`` (B1.5): replaces ALL heads listed in ``--head-indices``
    SIMULTANEOUSLY, in a single patched forward — exactly one output row,
    label ``heads_<i>_<j>_<k>...``. Tests whether a candidate sparse head
    set jointly clears the verb-flip threshold. Pair this with the per-head
    sweep (``--components head head_subset``) to measure linearity of the
    per-head contributions in one invocation.

Usage on a GPU box::

    python -m experiments.component_patching \\
        --enriched-log logs/cot_llama8b_t0_s42_*.jsonl.gz \\
                       logs/cot_llama8b_t0_s123_*.jsonl.gz \\
                       logs/cot_llama8b_t0_s456_*.jsonl.gz \\
        --source-bucket clean_check_or_call \\
        --target-bucket illegal_fold \\
        --layer 14 \\
        --components residual attn mlp head \\
        --head-indices all \\
        --n-source 10 --n-target 30 \\
        --out-dir results/causal_patching/llama8b_l14_components \\
        --device cuda --dtype bfloat16

Outputs:
  - ``by_pair_components.csv``: one row per (target, source, component_mode)
    with the scored per-family logits and top-1 destination
  - ``summary_components.json``: per-component aggregate stats
  - ``SUMMARY_components.md``: writeup-ready markdown table

The headline reading on the resulting SUMMARY_components.md is the
RATIO-TO-RESIDUAL Δ for each mode (component mode Δ / residual mode Δ at
the same layer), side-by-side across modes — NOT the same as the
specificity-adjusted Δ that `causal_patching.py` reports. This driver
does NOT compute a random-source null per component; the comparison
denominator is the residual-mode patching effect at the same layer
on the same (source, target) pairs, which serves as the cell's own
upper-bound reference.

    | Mode      | Δ (CHECK − FOLD) | ratio-to-residual | top-1 → CHECK |
    |-----------|------------------|------------------:|---------------|
    | residual  | +13.4            |             100%  | 100%          |
    | attn      | +12.8            |              96%  |  98%          |
    | mlp       | +0.7             |               5%  |   3%          |
    | head_07   | +6.3             |              47%  |  60%          |
    | head_12   | +4.8             |              36%  |  45%          |
    | head_03   | +0.2             |               1%  |   2%          |
    | ...

If attention dominates and a small set of heads carry most of the
residual-mode effect, that's the paper-banner mech-interp result.

Note: prior versions of this docstring and several wrapper scripts
described the metric as "specificity-adjusted Δ". That wording was
incorrect — the metric is `ratio_to_residual` (or its raw Δ value).
Use that terminology in writeups.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.recategorize_action_metadata import _reparse_one  # noqa: E402
from poker_env.interp.forward_helpers import (  # noqa: E402
    PromptReconstructor,
    build_input_text_for_action_verb_position,
    run_forward_at_last_position,
)
from poker_env.interp.patching import (  # noqa: E402
    HiddenStateCaptureMulti,
    HiddenStatePatch,
    HiddenStatePatchAttnOnly,
    HiddenStatePatchMLPOnly,
    HiddenStatePatchAttnHeadSubset,
    get_head_geometry,
)
# Reuse bucketing + scoring helpers from the residual driver to keep the
# bucket distribution and verb-token-id sets EXACTLY in sync.
from experiments.causal_patching import (  # noqa: E402
    BUCKET_NAMES,
    classify_decision,
    build_action_token_id_sets,
    score_logits,
    _open_log,
    _iter_decisions,
    _load_agent_config,
)


# ---------------------------------------------------------------------------
# Component mode enumeration
# ---------------------------------------------------------------------------
#
# A "component mode" is a single (mode_name, head_idx-or-None) pair that
# fully specifies what gets patched on a target forward. We enumerate them
# upfront so the main loop is component_modes × (target × source) rather
# than nested case logic.

def enumerate_component_modes(
    components: list[str],
    head_indices: list[int],
    num_heads: int,
) -> list[tuple[str, object]]:
    """Build the ordered (mode_name, head_spec) list. ``head_spec`` is:
        - ``None`` for non-head modes
        - an ``int`` for ``head`` mode (one row per head)
        - a ``tuple[int, ...]`` for ``head_subset`` mode (one row covering
          all listed heads simultaneously)
    """
    modes: list[tuple[str, object]] = []
    for c in components:
        if c == "residual":
            modes.append(("residual", None))
        elif c == "attn":
            modes.append(("attn", None))
        elif c == "mlp":
            modes.append(("mlp", None))
        elif c == "head":
            for h in head_indices:
                if not (0 <= h < num_heads):
                    raise ValueError(f"head index {h} out of range [0, {num_heads})")
                modes.append(("head", h))
        elif c == "head_subset":
            if not head_indices:
                raise ValueError("head_subset requires --head-indices to be non-empty")
            for h in head_indices:
                if not (0 <= h < num_heads):
                    raise ValueError(f"head_subset index {h} out of range [0, {num_heads})")
            # Sort + deduplicate so the label is canonical regardless of CLI order.
            modes.append(("head_subset", tuple(sorted(set(head_indices)))))
        else:
            raise ValueError(f"unknown component '{c}'")
    return modes


def mode_label(mode_name: str, head_spec) -> str:
    if mode_name == "head":
        return f"head_{head_spec:02d}"
    if mode_name == "head_subset":
        return "heads_" + "_".join(f"{h:02d}" for h in head_spec)
    return mode_name


# ---------------------------------------------------------------------------
# Patcher dispatch
# ---------------------------------------------------------------------------


def make_patcher(
    model,
    layer_idx: int,
    mode_name: str,
    head_spec,
    captured: dict,
):
    """Return a context-manager-compatible patcher for the given mode using
    the source's ``HiddenStateCaptureMulti.collect()`` payload. ``head_spec``
    is ``None`` for non-head modes, an int for single-head mode, or a tuple
    of int for head_subset."""
    if mode_name == "residual":
        src = captured["per_layer_residual"][layer_idx]
        return HiddenStatePatch(model, layer_idx, src)
    if mode_name == "attn":
        src = captured["per_layer_attn_out"][layer_idx]
        return HiddenStatePatchAttnOnly(model, layer_idx, src)
    if mode_name == "mlp":
        src = captured["per_layer_mlp_out"][layer_idx]
        return HiddenStatePatchMLPOnly(model, layer_idx, src)
    if mode_name == "head":
        if head_spec is None:
            raise ValueError("head mode requires head index")
        src_per_head = captured["per_layer_attn_per_head"][layer_idx]
        return HiddenStatePatchAttnHeadSubset(
            model, layer_idx, src_per_head, [head_spec]
        )
    if mode_name == "head_subset":
        if not head_spec:
            raise ValueError("head_subset mode requires non-empty head tuple")
        src_per_head = captured["per_layer_attn_per_head"][layer_idx]
        return HiddenStatePatchAttnHeadSubset(
            model, layer_idx, src_per_head, list(head_spec)
        )
    raise ValueError(f"unknown mode {mode_name!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Component-level causal-patching CLI (B1)."
    )
    parser.add_argument("--enriched-log", required=True, nargs="+",
                        help="One or more enriched JSONL[.gz] logs (pooled).")
    parser.add_argument("--source-bucket", default="clean_check_or_call",
                        choices=BUCKET_NAMES)
    parser.add_argument("--target-bucket", default="illegal_fold",
                        choices=BUCKET_NAMES)
    parser.add_argument("--layer", type=int, required=True,
                        help="Single layer at which to do the component sweep.")
    parser.add_argument("--components", nargs="+",
                        default=["residual", "attn", "mlp", "head"],
                        choices=["residual", "attn", "mlp", "head", "head_subset"])
    parser.add_argument("--head-indices", nargs="+", default=["all"],
                        help="Head indices to test (e.g. '0 7 12 19'). Use "
                             "'all' to test every head individually. For "
                             "`head_subset` mode, ALL listed indices are "
                             "patched together as a single combined patch.")
    parser.add_argument("--n-source", type=int, default=10)
    parser.add_argument("--n-target", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--baseline-tolerance-frac", type=float, default=0.95)
    parser.add_argument("--target-residual-top1",
                        choices=["FOLD", "CHECK_CALL", "BET_RAISE"],
                        default=None,
                        help="OPTIONAL filter: drop targets whose baseline "
                             "residual top-1 family does NOT match this. "
                             "Plumbing-identical to causal_patching.py's "
                             "flag of the same name (see that script's help "
                             "for the rationale).")
    parser.add_argument("--source-residual-top1",
                        choices=["FOLD", "CHECK_CALL", "BET_RAISE"],
                        default=None,
                        help="OPTIONAL filter: drop sources whose lm_head "
                             "top-1 (at the verb position) does NOT match "
                             "this family. Symmetric to --target-residual-"
                             "top1; same rationale.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load agent config + model + tokenizer -----------------------------
    enriched_logs: list[str] = list(args.enriched_log)
    agent_config = _load_agent_config(enriched_logs[0])
    model_id = args.model_id or agent_config["model_id"]
    if len(enriched_logs) > 1:
        for extra in enriched_logs[1:]:
            other_cfg = _load_agent_config(extra)
            if other_cfg.get("model_id") != agent_config.get("model_id"):
                print(f"[abort] model mismatch: {agent_config.get('model_id')} "
                      f"vs {other_cfg.get('model_id')} ({extra})", file=sys.stderr)
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

    num_heads, head_dim = get_head_geometry(model)
    print(f"[init] head geometry: num_heads={num_heads} head_dim={head_dim}")

    if args.head_indices == ["all"]:
        head_indices = list(range(num_heads))
    else:
        head_indices = [int(h) for h in args.head_indices]
    if "head_subset" in args.components and args.head_indices == ["all"]:
        # 'all' for head_subset would patch every head, which is identical to
        # attn-mode (modulo numerical noise) and not informative. Reject early.
        raise SystemExit(
            "[abort] --components head_subset with --head-indices all is "
            "equivalent to attn-only and not informative. Pass an explicit "
            "non-empty subset of head indices (e.g. '5 23 24')."
        )
    component_modes = enumerate_component_modes(
        args.components, head_indices, num_heads
    )
    print(f"[init] component modes ({len(component_modes)}): "
          + ", ".join(mode_label(n, h) for n, h in component_modes))

    recon = PromptReconstructor(tokenizer, agent_config)
    family_ids = build_action_token_id_sets(tokenizer)

    # ---- Bucket & sample ---------------------------------------------------
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

    sources_pool = by_bucket[args.source_bucket]
    targets_pool = by_bucket[args.target_bucket]
    if not sources_pool or not targets_pool:
        print(f"[abort] empty source ({len(sources_pool)}) or target "
              f"({len(targets_pool)}) bucket.")
        sys.exit(2)
    sources = rng.sample(sources_pool, min(args.n_source, len(sources_pool)))
    targets = rng.sample(targets_pool, min(args.n_target, len(targets_pool)))

    def _prepare(rec: dict) -> dict | None:
        full_prompt = recon.build(rec)
        out = build_input_text_for_action_verb_position(
            full_prompt, rec["action_metadata"]["raw_response"], tokenizer,
        )
        if out is None:
            return None
        input_text, verb_resp_idx = out
        resp_tok_ids = tokenizer(rec["action_metadata"]["raw_response"],
                                 add_special_tokens=False)["input_ids"]
        if not (0 <= verb_resp_idx < len(resp_tok_ids)):
            return None
        return {
            "rec": rec,
            "input_text": input_text,
            "expected_verb_tok_id": int(resp_tok_ids[verb_resp_idx]),
        }

    sources_prep = [p for p in (_prepare(r) for r in sources) if p is not None]
    targets_prep = [p for p in (_prepare(r) for r in targets) if p is not None]
    print(f"[prepare] sources_with_input={len(sources_prep)} "
          f"targets_with_input={len(targets_prep)}")

    # ---- Capture per-source MULTI-component states at the layer ----------
    print(f"[capture] {len(sources_prep)} source forwards (multi-component) ...")
    cap = HiddenStateCaptureMulti(model)
    source_captures: list[dict] = []
    source_top1_ids: list[int] = []
    filter_bookkeeping: dict = {}
    cap_t0 = time.time()
    for i, src in enumerate(sources_prep):
        cap.attach_hooks()
        try:
            with torch.no_grad():
                src_out = model(
                    input_ids=tokenizer(src["input_text"],
                                        return_tensors="pt",
                                        add_special_tokens=False
                                        )["input_ids"].to(args.device)
                )
        finally:
            states = cap.collect()
            cap.detach_hooks()
        source_captures.append(states)
        last_logits = src_out.logits[0, -1, :].detach().to("cpu")
        source_top1_ids.append(int(last_logits.argmax().item()))
        if (i + 1) % 5 == 0 or i + 1 == len(sources_prep):
            elapsed = time.time() - cap_t0
            rate = (i + 1) / elapsed
            print(f"  [capture] {i+1}/{len(sources_prep)} "
                  f"({elapsed:.0f}s, {rate:.2f} src/s)")

    # Optional source-residual-top-1 filter.
    if args.source_residual_top1 is not None:
        family = args.source_residual_top1
        if family == "CHECK_CALL":
            allowed_ids = set(family_ids["CHECK"]) | set(family_ids["CALL"])
        elif family == "BET_RAISE":
            allowed_ids = set(family_ids["BET"]) | set(family_ids["RAISE"])
        else:
            allowed_ids = set(family_ids[family])
        kept = [i for i, t in enumerate(source_top1_ids) if t in allowed_ids]
        n_before, n_after = len(sources_prep), len(kept)
        print(f"  [filter] source_residual_top1={family}: "
              f"keeping {n_after}/{n_before} sources")
        filter_bookkeeping["source_residual_top1_filter"] = {
            "family": family, "n_before": n_before, "n_after": n_after,
            "frac_kept": n_after / max(n_before, 1),
        }
        if n_after < 1:
            print(f"  [HALT] no sources remain after residual-top-1 filter.")
            sys.exit(3)
        sources_prep = [sources_prep[i] for i in kept]
        source_captures = [source_captures[i] for i in kept]
        source_top1_ids = [source_top1_ids[i] for i in kept]

    # ---- Control 1: baseline (no patch) for each target ---------------------
    print(f"\n[control 1/2] baseline (no patch) for {len(targets_prep)} targets ...")
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
    if baseline_match_rate < args.baseline_tolerance_frac:
        print(f"  [HALT] below tolerance {args.baseline_tolerance_frac}.")
        sys.exit(3)

    # Optional target-residual-top-1 filter.
    if args.target_residual_top1 is not None:
        family = args.target_residual_top1
        if family == "CHECK_CALL":
            allowed_ids = set(family_ids["CHECK"]) | set(family_ids["CALL"])
        elif family == "BET_RAISE":
            allowed_ids = set(family_ids["BET"]) | set(family_ids["RAISE"])
        else:
            allowed_ids = set(family_ids[family])
        kept = [ti for ti in range(len(targets_prep))
                if baseline_cache[ti]["top1_id"] in allowed_ids]
        n_before, n_after = len(targets_prep), len(kept)
        print(f"  [filter] target_residual_top1={family}: "
              f"keeping {n_after}/{n_before} targets")
        filter_bookkeeping["target_residual_top1_filter"] = {
            "family": family, "n_before": n_before, "n_after": n_after,
            "frac_kept": n_after / max(n_before, 1),
        }
        if n_after < 1:
            print(f"  [HALT] no targets remain after residual-top-1 filter.")
            sys.exit(3)
        targets_prep = [targets_prep[i] for i in kept]
        baseline_cache = {new_i: baseline_cache[old_i]
                          for new_i, old_i in enumerate(kept)}

    # ---- Control 2: self-patch identity at the layer (residual mode only) -
    print(f"\n[control 2/2] self-patch identity at L={args.layer} ...")
    tgt = targets_prep[0]
    cap.attach_hooks()
    try:
        with torch.no_grad():
            model(input_ids=tokenizer(tgt["input_text"], return_tensors="pt",
                                      add_special_tokens=False
                                      )["input_ids"].to(args.device))
    finally:
        self_states = cap.collect()
        cap.detach_hooks()
    self_patch = HiddenStatePatch(
        model, args.layer, self_states["per_layer_residual"][args.layer]
    )
    with self_patch:
        sp_res = run_forward_at_last_position(
            model, tokenizer, tgt["input_text"], device=args.device,
        )
    sp_drift = float((sp_res["logits_last_pos"] -
                      baseline_cache[0]["logits_last_pos"]).abs().max().item())
    print(f"  self-patch max |Δlogit| = {sp_drift:.6f}")
    if sp_drift > 1e-2:
        print("  [WARN] self-patch drift > 0.01 — investigate before trusting "
              "head-level numbers (numerical nondeterminism may swamp small "
              "per-head effects).")

    # ---- Main loop: target × source × component_mode ----------------------
    n_pairs_total = len(targets_prep) * len(sources_prep) * len(component_modes)
    print(f"\n[main] {len(targets_prep)} × {len(sources_prep)} × "
          f"{len(component_modes)} = {n_pairs_total} patched forwards "
          f"at L={args.layer}")

    by_pair_path = out_dir / "by_pair_components.csv"
    by_pair_f = open(by_pair_path, "w", newline="")
    writer = csv.writer(by_pair_f)
    writer.writerow([
        "source_idx", "target_idx", "layer", "component_mode", "head_idx",
        "source_hand", "source_dec", "target_hand", "target_dec",
        "expected_target_verb_id", "expected_target_verb_tok",
        "patched_top1_id", "patched_top1_tok", "patched_top1_group",
        "delta_check_minus_fold",
        "patched_GROUP_CHECK_CALL", "patched_GROUP_FOLD",
        "patched_GROUP_BET_RAISE",
    ])

    main_t0 = time.time()
    n_done = 0
    for ti, tgt in enumerate(targets_prep):
        baseline_score = baseline_cache[ti]["scored"]
        for mode_name, head_spec in component_modes:
            # Encode head_spec into the CSV's head_idx column. Single ints
            # stay as ints; tuples (head_subset) are dot-separated for
            # downstream CSV parsing without quoting.
            if isinstance(head_spec, int):
                head_idx_csv = str(head_spec)
            elif isinstance(head_spec, tuple):
                head_idx_csv = "+".join(str(h) for h in head_spec)
            else:
                head_idx_csv = ""
            for si, src in enumerate(sources_prep):
                patcher = make_patcher(
                    model, args.layer, mode_name, head_spec,
                    source_captures[si],
                )
                with patcher:
                    res = run_forward_at_last_position(
                        model, tokenizer, tgt["input_text"], device=args.device,
                    )
                logits = res["logits_last_pos"]
                top1_id = int(logits.argmax().item())
                top1_tok = tokenizer.decode([top1_id])
                scored = score_logits(logits, family_ids)
                delta = (scored["delta_check_minus_fold"]
                         - baseline_score["delta_check_minus_fold"])

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
                    si, ti, args.layer, mode_name,
                    head_idx_csv,
                    src["rec"]["hand_id"], src["rec"]["decision_idx"],
                    tgt["rec"]["hand_id"], tgt["rec"]["decision_idx"],
                    tgt["expected_verb_tok_id"],
                    tokenizer.decode([tgt["expected_verb_tok_id"]]),
                    top1_id, top1_tok, top1_group,
                    f"{delta:.4f}",
                    f"{scored['GROUP_CHECK_CALL']:.4f}",
                    f"{scored['GROUP_FOLD']:.4f}",
                    f"{scored['GROUP_BET_RAISE']:.4f}",
                ])
                n_done += 1
            elapsed = time.time() - main_t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_pairs_total - n_done) / rate if rate > 0 else float("nan")
            print(f"  [main] target {ti+1}/{len(targets_prep)} "
                  f"mode={mode_label(mode_name, head_spec)}: "
                  f"{n_done}/{n_pairs_total} ({elapsed:.0f}s, "
                  f"{rate:.2f}/s, ETA {eta:.0f}s)")
    by_pair_f.close()

    # ---- Aggregate per-component summary ----------------------------------
    per_mode: dict[str, dict] = {}
    for mode_name, head_spec in component_modes:
        per_mode[mode_label(mode_name, head_spec)] = {
            "mode": mode_name,
            "head_spec": (
                head_spec if not isinstance(head_spec, tuple) else list(head_spec)
            ),
            "n": 0,
            "delta_sum": 0.0,
            "flipped_check_n": 0,
            "flipped_fold_n": 0,
            "flipped_betraise_n": 0,
            "flipped_other_n": 0,
        }
    with open(by_pair_path) as f:
        for row in csv.DictReader(f):
            mode_field = row["component_mode"]
            head_field = row["head_idx"]
            if mode_field == "head" and head_field != "":
                label = f"head_{int(head_field):02d}"
            elif mode_field == "head_subset" and head_field != "":
                hs = sorted(int(x) for x in head_field.split("+"))
                label = "heads_" + "_".join(f"{h:02d}" for h in hs)
            else:
                label = mode_field
            d = per_mode[label]
            d["n"] += 1
            d["delta_sum"] += float(row["delta_check_minus_fold"])
            grp = row["patched_top1_group"]
            if grp == "CHECK_CALL":
                d["flipped_check_n"] += 1
            elif grp == "FOLD":
                d["flipped_fold_n"] += 1
            elif grp == "BET_RAISE":
                d["flipped_betraise_n"] += 1
            else:
                d["flipped_other_n"] += 1

    # Anchor: residual-mode mean delta (the existing experiment's number).
    residual_mean = (per_mode.get("residual", {}).get("delta_sum", 0)
                     / max(per_mode.get("residual", {}).get("n", 1), 1)
                     if "residual" in per_mode else None)

    summary = {
        "enriched_log": enriched_logs if len(enriched_logs) > 1 else enriched_logs[0],
        "n_enriched_logs": len(enriched_logs),
        "model_id": model_id,
        "source_bucket": args.source_bucket,
        "target_bucket": args.target_bucket,
        "n_source": len(sources_prep),
        "n_target": len(targets_prep),
        "layer": args.layer,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "components_tested": args.components,
        "head_indices_tested": head_indices if "head" in args.components else [],
        "controls": {
            "baseline_top1_match_rate": baseline_match_rate,
            "self_patch_max_logit_drift": sp_drift,
            **filter_bookkeeping,
        },
        "per_mode": {},
    }
    for label, d in per_mode.items():
        mean_d = d["delta_sum"] / d["n"] if d["n"] else None
        ratio_to_residual = (
            (mean_d / residual_mean)
            if (mean_d is not None and residual_mean and abs(residual_mean) > 1e-6)
            else None
        )
        summary["per_mode"][label] = {
            "mode": d["mode"],
            "head_spec": d["head_spec"],
            "n": d["n"],
            "mean_delta_check_minus_fold": mean_d,
            "ratio_to_residual": ratio_to_residual,
            "frac_top1_check_call": d["flipped_check_n"] / d["n"] if d["n"] else None,
            "frac_top1_fold": d["flipped_fold_n"] / d["n"] if d["n"] else None,
            "frac_top1_bet_raise": d["flipped_betraise_n"] / d["n"] if d["n"] else None,
            "frac_top1_other": d["flipped_other_n"] / d["n"] if d["n"] else None,
        }

    with open(out_dir / "summary_components.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Markdown
    lines = ["# Component-level causal patching results", ""]
    lines.append(f"- Model: `{model_id}`")
    if len(enriched_logs) == 1:
        lines.append(f"- Enriched log: `{enriched_logs[0]}`")
    else:
        lines.append(f"- Enriched logs (pooled, n={len(enriched_logs)}):")
        for p in enriched_logs:
            lines.append(f"  - `{p}`")
    if args.target_residual_top1 is not None:
        f = filter_bookkeeping.get("target_residual_top1_filter") or {}
        lines.append(
            f"- **Residual-top-1 target filter**: kept only targets whose "
            f"baseline residual top-1 was in family `{args.target_residual_top1}` "
            f"({f.get('n_after', '?')}/{f.get('n_before', '?')} = "
            f"{(f.get('frac_kept', 0)*100):.1f}% retained)."
        )
    if args.source_residual_top1 is not None:
        f = filter_bookkeeping.get("source_residual_top1_filter") or {}
        lines.append(
            f"- **Residual-top-1 source filter**: kept only sources whose "
            f"verb-position residual top-1 was in family `{args.source_residual_top1}` "
            f"({f.get('n_after', '?')}/{f.get('n_before', '?')} = "
            f"{(f.get('frac_kept', 0)*100):.1f}% retained)."
        )
    lines.append(f"- Source bucket: `{args.source_bucket}` (n={len(sources_prep)})")
    lines.append(f"- Target bucket: `{args.target_bucket}` (n={len(targets_prep)})")
    lines.append(f"- Layer: **{args.layer}**")
    lines.append(f"- Head geometry: num_heads={num_heads}, head_dim={head_dim}")
    lines.append("")
    lines.append("## Controls")
    lines.append(f"- `baseline_top1_match_rate` = {baseline_match_rate:.3f}")
    lines.append(f"- `self_patch_max_logit_drift` = {sp_drift:.6f}")
    lines.append("")
    lines.append("## Per-component / per-head effect at L=" + str(args.layer))
    lines.append("| Mode | n | mean Δ(CHECK − FOLD) | ratio to residual | top-1 → CHECK | top-1 → FOLD | top-1 → BET_RAISE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    # Sort: residual / attn / mlp first, then head_subset (next to attn for
    # readability), then individual heads in numeric order.
    def _sort_key(label):
        if label == "residual": return (0, 0)
        if label == "attn":     return (1, 0)
        if label.startswith("heads_"):
            # Sort by smallest head index in the subset, breaking ties by length
            # so a small triplet is grouped near its single-head rows.
            heads = [int(x) for x in label[len("heads_"):].split("_")]
            return (2, min(heads), len(heads))
        if label == "mlp":      return (3, 0)
        if label.startswith("head_"): return (4, int(label.split("_")[1]))
        return (9, 0)
    for label in sorted(per_mode.keys(), key=_sort_key):
        d = summary["per_mode"][label]
        mean_d = d["mean_delta_check_minus_fold"]
        ratio = d["ratio_to_residual"]
        mean_s = f"{mean_d:+.3f}" if mean_d is not None else "—"
        ratio_s = f"{ratio*100:+.0f}%" if ratio is not None else "—"
        f_check = d["frac_top1_check_call"]
        f_fold = d["frac_top1_fold"]
        f_betraise = d["frac_top1_bet_raise"]
        f_check_s = f"{f_check*100:.1f}%" if f_check is not None else "—"
        f_fold_s = f"{f_fold*100:.1f}%" if f_fold is not None else "—"
        f_betraise_s = f"{f_betraise*100:.1f}%" if f_betraise is not None else "—"
        lines.append(
            f"| `{label}` | {d['n']} | {mean_s} | {ratio_s} | "
            f"{f_check_s} | {f_fold_s} | {f_betraise_s} |"
        )
    lines.append("")
    lines.append(
        "**Interpretation guide**:\n"
        "- `residual` is the existing experiment baseline (full-residual patch); "
        "expected to match prior pooled-sweep numbers at this layer.\n"
        "- `attn` ≈ `residual` (≥80% ratio) and `mlp` ≪ `residual` (<20%): "
        "the layer's effect is mediated by attention, not the MLP.\n"
        "- A small set of `head_NN` rows each contributing >10% of the "
        "residual effect: SPARSE HEAD STORY — the strongest possible version "
        "of the result. Cite those head indices.\n"
        "- A `heads_NN_MM_KK` row near `attn`'s ratio: the listed sparse "
        "subset jointly captures the attention contribution. If its top-1 "
        "flip rate also matches `residual`, the triplet is the circuit.\n"
        "- A `heads_NN_MM_KK` row well below the linear sum of its members' "
        "individual ratios: per-head contributions are NON-additive (the "
        "heads interact through downstream MLP recomputation).\n"
        "- All `head_NN` rows roughly equal and individually small: dense "
        "attention story — the effect spreads across many heads and no "
        "single head dominates.\n"
        "- `attn` ≈ `residual` AND no head individually large: a weighted "
        "combination of heads carries the signal (not as strong a result, "
        "still publishable).\n"
        "- `attn` < `residual` AND `mlp` < `residual` AND their sum ≠ "
        "`residual`: a real interaction between sublayers exists; both "
        "are needed to flip the verb."
    )
    with open(out_dir / "SUMMARY_components.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[done] wrote {out_dir / 'summary_components.json'}")
    print(f"[done] wrote {by_pair_path}")
    print(f"[done] wrote {out_dir / 'SUMMARY_components.md'}")


if __name__ == "__main__":
    main()
