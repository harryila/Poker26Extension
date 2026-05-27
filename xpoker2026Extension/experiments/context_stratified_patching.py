"""
Context-stratified activation patching.

Holds targets fixed (``target_bucket``) and patches source residuals into
them, comparing effect size across source strata (street, facing bet, pot odds
on facing-bet spots only, etc.).

Default stratification is ``street`` because pot-odds quartiles collapse
when most ``clean_check_or_call`` decisions have ``bet_to_call == 0``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_env.interp.forward_helpers import (  # noqa: E402
    PromptReconstructor,
    build_input_text_for_action_verb_position,
    run_forward_at_last_position,
)
from poker_env.interp.patching import HiddenStateCapture, HiddenStatePatch  # noqa: E402
from experiments.causal_patching import (  # noqa: E402
    classify_decision,
    build_action_token_id_sets,
    score_logits,
    _iter_decisions,
    _load_agent_config,
)


def _pot_odds(rec: dict) -> float:
    obs = rec.get("obs") or {}
    pot = float(obs.get("pot_total") or 0)
    btc = float(obs.get("bet_to_call") or 0)
    denom = pot + btc
    if denom <= 0:
        return 0.0
    return btc / denom


def _stratum_key(rec: dict, method: str) -> str:
    obs = rec.get("obs") or {}
    if method == "street":
        return str(obs.get("street") or "UNKNOWN")
    if method == "facing_bet":
        btc = float(obs.get("bet_to_call") or 0)
        return "facing_bet" if btc > 0 else "free_check"
    if method == "pot_odds":
        return f"pot_odds_{_pot_odds(rec):.3f}"
    if method == "pot_odds_quartile":
        # Assigned later via quartile edges (only meaningful if caller filters).
        return "pending"
    if method == "pot_total_quartile":
        return "pending"
    raise ValueError(f"unknown stratify method: {method}")


def _quartile_edges(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 1.0]
    odds = sorted(values)
    n = len(odds)
    edges = [odds[0]]
    for q in (0.25, 0.5, 0.75, 1.0):
        idx = min(int(q * n), n - 1)
        edges.append(odds[idx])
    if edges[-1] <= edges[0]:
        edges = [edges[0], edges[0] + 1e-6]
    return edges


def _quartile_label(value: float, edges: list[float]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return f"Q{i + 1}"
    return f"Q{len(edges) - 1}"


def _build_strata(
    sources: list[dict],
    method: str,
) -> tuple[dict[str, list], dict]:
    meta: dict = {"method": method}
    if method == "pot_odds_quartile":
        facing = [r for r in sources if float((r.get("obs") or {}).get("bet_to_call") or 0) > 0]
        meta["n_sources_total"] = len(sources)
        meta["n_sources_facing_bet"] = len(facing)
        if len(facing) < 8:
            meta["warning"] = (
                "fewer than 8 clean_check_or_call sources with bet_to_call>0; "
                "pot_odds_quartile strata may be sparse"
            )
        pool = facing if facing else sources
        vals = [_pot_odds(r) for r in pool]
        edges = _quartile_edges(vals)
        meta["quartile_edges"] = edges
        strata: dict[str, list] = defaultdict(list)
        for rec in pool:
            strata[_quartile_label(_pot_odds(rec), edges)].append(rec)
        return dict(strata), meta

    if method == "pot_total_quartile":
        vals = [float((r.get("obs") or {}).get("pot_total") or 0) for r in sources]
        edges = _quartile_edges(vals)
        meta["quartile_edges"] = edges
        strata = defaultdict(list)
        for rec in sources:
            v = float((rec.get("obs") or {}).get("pot_total") or 0)
            strata[_quartile_label(v, edges)].append(rec)
        return dict(strata), meta

    strata = defaultdict(list)
    for rec in sources:
        strata[_stratum_key(rec, method)].append(rec)
    return dict(strata), meta


def main():
    parser = argparse.ArgumentParser(
        description="Context-stratified causal patching at L*."
    )
    parser.add_argument("--enriched-log", required=True, nargs="+")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--source-bucket", default="clean_check_or_call")
    parser.add_argument("--target-bucket", default="illegal_fold")
    parser.add_argument(
        "--stratify-by",
        default="street",
        choices=[
            "street", "facing_bet", "pot_odds_quartile", "pot_total_quartile",
        ],
        help="How to split CHECK sources (default street — pot_odds alone "
             "collapses when bet_to_call=0 for most checks).",
    )
    parser.add_argument("--n-source-per-stratum", type=int, default=5)
    parser.add_argument("--min-sources-per-stratum", type=int, default=2)
    parser.add_argument("--n-target", type=int, default=20)
    parser.add_argument("--n-random-control", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    agent_config = _load_agent_config(args.enriched_log[0])
    model_id = agent_config["model_id"]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[init] loading {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=args.device,
    )
    model.eval()

    recon = PromptReconstructor(tokenizer, agent_config)
    family_ids = build_action_token_id_sets(tokenizer)
    rng = random.Random(args.seed)

    by_bucket: dict[str, list] = {}
    for log in args.enriched_log:
        for rec in _iter_decisions(log):
            am = rec.get("action_metadata")
            if not am or not am.get("raw_response"):
                continue
            by_bucket.setdefault(classify_decision(rec), []).append(rec)

    sources_all = by_bucket.get(args.source_bucket, [])
    targets_all = by_bucket.get(args.target_bucket, [])
    if len(sources_all) < 8 or not targets_all:
        print("[abort] insufficient sources or targets", file=sys.stderr)
        sys.exit(2)

    strata, strat_meta = _build_strata(sources_all, args.stratify_by)
    print(f"[init] stratify_by={args.stratify_by} → {len(strata)} strata: "
          f"{sorted(strata.keys())}")

    targets = rng.sample(targets_all, min(args.n_target, len(targets_all)))

    def _prepare(rec):
        full = recon.build(rec)
        out = build_input_text_for_action_verb_position(
            full, rec["action_metadata"]["raw_response"], tokenizer,
        )
        if out is None:
            return None
        text, _ = out
        return {"rec": rec, "input_text": text}

    targets_prep = [p for p in (_prepare(r) for r in targets) if p]
    alt_buckets = [b for b in (
        "clean_check_or_call", "clean_legal_fold", "clean_bet_or_raise",
    ) if b != args.source_bucket]

    stratum_results = {}

    for stratum, src_pool in sorted(strata.items()):
        if len(src_pool) < args.min_sources_per_stratum:
            print(f"  [skip] {stratum}: only {len(src_pool)} sources "
                  f"(need {args.min_sources_per_stratum})")
            continue
        src_sample = rng.sample(
            src_pool, min(args.n_source_per_stratum, len(src_pool)),
        )
        src_prep = [p for p in (_prepare(r) for r in src_sample) if p]
        if not src_prep:
            continue

        src_states = []
        for sp in src_prep:
            cap = HiddenStateCapture(model)
            cap.attach_hooks()
            with torch.no_grad():
                model(tokenizer(
                    sp["input_text"], return_tensors="pt",
                ).input_ids.to(args.device))
            st = cap.collect()["per_layer_last_pos"][args.layer]
            cap.detach_hooks()
            src_states.append(st)

        deltas = []
        flips = 0
        n_pairs = 0
        for src_res in src_states:
            for tgt in targets_prep:
                base = run_forward_at_last_position(
                    model, tokenizer, tgt["input_text"], device=args.device,
                )
                base_sc = score_logits(base["logits_last_pos"], family_ids)
                with HiddenStatePatch(model, args.layer, src_res):
                    pat = run_forward_at_last_position(
                        model, tokenizer, tgt["input_text"],
                        device=args.device,
                    )
                pat_sc = score_logits(pat["logits_last_pos"], family_ids)
                d = pat_sc["delta_check_minus_fold"] - base_sc["delta_check_minus_fold"]
                deltas.append(d)
                n_pairs += 1
                if int(pat["logits_last_pos"].argmax()) != int(
                    base["logits_last_pos"].argmax()
                ):
                    flips += 1

        rand_deltas = []
        if alt_buckets and targets_prep:
            alt_pool = []
            for b in alt_buckets:
                alt_pool.extend(by_bucket.get(b, [])[:50])
            if alt_pool:
                rand_srcs = rng.sample(
                    alt_pool, min(args.n_random_control, len(alt_pool)),
                )
                for rr in rand_srcs:
                    pp = _prepare(rr)
                    if pp is None:
                        continue
                    cap = HiddenStateCapture(model)
                    cap.attach_hooks()
                    with torch.no_grad():
                        model(tokenizer(
                            pp["input_text"], return_tensors="pt",
                        ).input_ids.to(args.device))
                    rs = cap.collect()["per_layer_last_pos"][args.layer]
                    cap.detach_hooks()
                    tgt = targets_prep[0]
                    base = run_forward_at_last_position(
                        model, tokenizer, tgt["input_text"], device=args.device,
                    )
                    base_sc = score_logits(base["logits_last_pos"], family_ids)
                    with HiddenStatePatch(model, args.layer, rs):
                        pat = run_forward_at_last_position(
                            model, tokenizer, tgt["input_text"],
                            device=args.device,
                        )
                    pat_sc = score_logits(pat["logits_last_pos"], family_ids)
                    rand_deltas.append(
                        pat_sc["delta_check_minus_fold"]
                        - base_sc["delta_check_minus_fold"]
                    )

        mean_d = sum(deltas) / max(len(deltas), 1)
        mean_rand = sum(rand_deltas) / max(len(rand_deltas), 1) if rand_deltas else 0.0
        stratum_results[stratum] = {
            "n_sources": len(src_prep),
            "n_pairs": n_pairs,
            "mean_delta": mean_d,
            "spec_adj_delta": mean_d - mean_rand,
            "frac_top1_flip": flips / max(n_pairs, 1),
            "stratum_size": len(src_pool),
        }
        print(f"  [{stratum}] n_pool={len(src_pool)} mean_Δ={mean_d:+.2f} "
              f"spec_adj={mean_d - mean_rand:+.2f} flip={flips}/{n_pairs}")

    summary = {
        "model_id": model_id,
        "layer": args.layer,
        "source_bucket": args.source_bucket,
        "target_bucket": args.target_bucket,
        "stratify_by": args.stratify_by,
        "stratify_meta": strat_meta,
        "strata": stratum_results,
        "n_strata_run": len(stratum_results),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = [
        "# Context-stratified patching",
        "",
        f"- Model: `{model_id}`",
        f"- Layer: **{args.layer}**",
        f"- Source: `{args.source_bucket}` stratified by **`{args.stratify_by}`**",
        f"- Target: `{args.target_bucket}` (n={len(targets_prep)})",
        f"- Strata run: **{len(stratum_results)}** "
        f"(skipped strata with <{args.min_sources_per_stratum} sources)",
        "",
    ]
    if strat_meta.get("warning"):
        md.append(f"- ⚠️ {strat_meta['warning']}")
        md.append("")
    md.extend([
        "| Stratum | pool n | n_src | mean Δ | spec-adj Δ | top-1 flip |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for st, d in sorted(stratum_results.items()):
        md.append(
            f"| {st} | {d['stratum_size']} | {d['n_sources']} | {d['mean_delta']:+.2f} "
            f"| {d['spec_adj_delta']:+.2f} | {d['frac_top1_flip']*100:.1f}% |"
        )
    specs = [d["spec_adj_delta"] for d in stratum_results.values()]
    if len(specs) >= 2:
        spread = max(specs) - min(specs)
        md.append("")
        md.append(f"**Cross-stratum spec-adj spread: {spread:.2f} nats** "
                  f"({len(specs)} strata)")
        if spread < 1.0:
            md.append(
                "- Patch effect is **stable across strata** → verb encoding "
                "is similar across these context splits."
            )
        else:
            md.append(
                "- Patch effect **varies by stratum** → L* mediation is "
                "context-modulated."
            )
    elif len(specs) == 1:
        md.append("")
        md.append(
            "- ⚠️ Only one stratum met `min_sources_per_stratum`; "
            "cross-stratum comparison not valid. Try `--stratify-by street`."
        )
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[done] {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
