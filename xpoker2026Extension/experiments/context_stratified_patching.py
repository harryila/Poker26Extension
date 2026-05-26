"""
Context-stratified activation patching.

Holds targets fixed (same ``target_bucket``) and patches CHECK residuals from
sources stratified by game context (pot odds, board street, etc.). If
spec-adj Δ is stable across strata, the L* circuit encodes verb choice largely
downstream of equity/pot context; if it varies, context modulates the circuit.

Stratification (default): quartiles of pot_odds =
    bet_to_call / max(pot_total + bet_to_call, 1)
among ``clean_check_or_call`` sources.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

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


def _stratum_label(value: float, edges: list[float]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return f"Q{i+1}"
    return f"Q{len(edges)-1}"


def main():
    parser = argparse.ArgumentParser(
        description="Context-stratified causal patching at L*."
    )
    parser.add_argument("--enriched-log", required=True, nargs="+")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--source-bucket", default="clean_check_or_call")
    parser.add_argument("--target-bucket", default="illegal_fold")
    parser.add_argument("--n-source-per-stratum", type=int, default=5)
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
    dtype = {"bfloat16": __import__("torch").bfloat16,
             "float16": __import__("torch").float16,
             "float32": __import__("torch").float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[init] loading {model_id} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=args.device,
    )
    model.eval()
    print(f"[init] loaded in {time.time() - t0:.1f}s")

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

    # Quartile edges on pot odds
    odds = sorted(_pot_odds(r) for r in sources_all)
    n = len(odds)
    edges = [odds[0]]
    for q in (0.25, 0.5, 0.75, 1.0):
        idx = min(int(q * n), n - 1)
        edges.append(odds[idx])
    if edges[-1] <= edges[0]:
        edges = [0.0, 0.25, 0.5, 0.75, 1.01]

    strata: dict[str, list] = {}
    for rec in sources_all:
        lab = _stratum_label(_pot_odds(rec), edges)
        strata.setdefault(lab, []).append(rec)

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
        if len(src_pool) < 2:
            continue
        src_sample = rng.sample(
            src_pool, min(args.n_source_per_stratum, len(src_pool)),
        )
        src_prep = [p for p in (_prepare(r) for r in src_sample) if p]
        if not src_prep:
            continue

        # Capture sources
        src_states = []
        for sp in src_prep:
            cap = HiddenStateCapture(model)
            cap.attach_hooks()
            with __import__("torch").no_grad():
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

        # Random null (first target only, match causal_patching)
        rand_deltas = []
        if alt_buckets and targets_prep:
            alt_pool = []
            for b in alt_buckets:
                alt_pool.extend(by_bucket.get(b, [])[:50])
            if alt_pool:
                rand_srcs = rng.sample(
                    alt_pool,
                    min(args.n_random_control, len(alt_pool)),
                )
                for rr in rand_srcs:
                    pp = _prepare(rr)
                    if pp is None:
                        continue
                    cap = HiddenStateCapture(model)
                    cap.attach_hooks()
                    with __import__("torch").no_grad():
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
            "pot_odds_range": [
                min(_pot_odds(r) for r in src_sample),
                max(_pot_odds(r) for r in src_sample),
            ],
        }
        print(f"  [{stratum}] mean_Δ={mean_d:+.2f} spec_adj={mean_d - mean_rand:+.2f} "
              f"flip={flips}/{n_pairs}")

    summary = {
        "model_id": model_id,
        "layer": args.layer,
        "source_bucket": args.source_bucket,
        "target_bucket": args.target_bucket,
        "quartile_edges": edges,
        "strata": stratum_results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = [
        "# Context-stratified patching",
        "",
        f"- Model: `{model_id}`",
        f"- Layer: **{args.layer}**",
        f"- Source: `{args.source_bucket}` stratified by pot-odds quartile",
        f"- Target: `{args.target_bucket}` (n={len(targets_prep)})",
        "",
        "| Stratum | n_src | mean Δ | spec-adj Δ | top-1 flip | pot odds range |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for st, d in sorted(stratum_results.items()):
        lo, hi = d["pot_odds_range"]
        md.append(
            f"| {st} | {d['n_sources']} | {d['mean_delta']:+.2f} "
            f"| {d['spec_adj_delta']:+.2f} | {d['frac_top1_flip']*100:.1f}% "
            f"| [{lo:.2f}, {hi:.2f}] |"
        )
    specs = [d["spec_adj_delta"] for d in stratum_results.values()]
    if specs:
        spread = max(specs) - min(specs)
        md.append("")
        md.append(f"**Cross-stratum spec-adj spread: {spread:.2f} nats**")
        if spread < 1.0:
            md.append(
                "- Patch effect is **stable across pot-odds contexts** → circuit "
                "behaves like a verb encoder downstream of equity stratification."
            )
        else:
            md.append(
                "- Patch effect **varies by context** → L* mediation is "
                "modulated by pot odds / situation."
            )
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[done] {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
