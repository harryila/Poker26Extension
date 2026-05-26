"""
Behavioral head-ablation during live CoT action generation.

Zeros the L* sufficient-head set on every forward pass inside
``model.generate()`` and re-runs action selection on decision records from
an enriched log. Compares bucket rates (especially ``illegal_fold``) to an
unablated baseline on the same decisions.

This is the necessity counterpart to forward-pass head ablation: sufficiency
was shown by activation patching; this tests whether disrupting the heads
during real inference changes emitted actions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_env.agents.hf_agent import HFAgent  # noqa: E402
from poker_env.interp.forward_helpers import obs_from_dict  # noqa: E402
from poker_env.interp.generation_ablation import AttnHeadZeroAblation  # noqa: E402
from experiments.causal_patching import (  # noqa: E402
    classify_decision,
    _iter_decisions,
    _load_agent_config,
)

# Default sufficient-head sets from Phase K (updates.md §16).
DEFAULT_HEAD_SETS: dict[str, dict] = {
    "llama": {"layer": 14, "triplet": [5, 23, 24], "control": [0, 1, 2]},
    "ministral": {"layer": 16, "triplet": [22, 9, 15], "control": [0, 1, 2]},
    "qwen": {"layer": 23, "triplet": [26, 28, 30], "control": [0, 1, 2]},
}


def _short_model_name(model_id: str) -> str:
    m = model_id.lower()
    if "ministral" in m or "mistral" in m:
        return "ministral"
    if "qwen" in m:
        return "qwen"
    if "llama" in m:
        return "llama"
    return "unknown"


def _make_agent(agent_config: dict, device: str) -> HFAgent:
    return HFAgent(
        model_id=agent_config["model_id"],
        device_map=device,
        temperature=float(agent_config.get("temperature", 0.0)),
        top_p=float(agent_config.get("top_p", 1.0)),
        cot_mode=bool(agent_config.get("cot_mode", True)),
        capture_logprobs=False,
        logit_lens=False,
    )


def _run_condition(
    agent: HFAgent,
    records: list[dict],
    *,
    ablation: AttnHeadZeroAblation | None,
) -> list[dict]:
    rows = []
    agent._extra_generation_hooks = [ablation] if ablation is not None else []
    try:
        for rec in records:
            obs = obs_from_dict(rec["obs"])
            action, meta = agent.act_with_metadata(obs)
            rows.append({
                "seed": rec.get("seed"),
                "decision_idx": rec.get("decision_idx"),
                "hand_id": rec.get("hand_id"),
                "recorded_bucket": classify_decision(rec),
                "replay_bucket": classify_decision({
                    "obs": rec["obs"],
                    "legal_actions": rec.get("legal_actions"),
                    "action_metadata": meta.to_dict(),
                }),
                "recorded_action": rec.get("action"),
                "replay_action": meta.action_chosen,
                "parse_success": meta.parse_success,
                "fallback_used": meta.fallback_used,
            })
    finally:
        agent._extra_generation_hooks = []
    return rows


def _aggregate(rows: list[dict]) -> dict:
    c_recorded = Counter(r["recorded_bucket"] for r in rows)
    c_replay = Counter(r["replay_bucket"] for r in rows)
    n = len(rows) or 1
    return {
        "n": len(rows),
        "recorded_bucket_frac": {k: v / n for k, v in c_recorded.items()},
        "replay_bucket_frac": {k: v / n for k, v in c_replay.items()},
        "illegal_fold_rate": c_replay.get("illegal_fold", 0) / n,
        "clean_check_rate": c_replay.get("clean_check_or_call", 0) / n,
        "parse_success_rate": sum(r["parse_success"] for r in rows) / n,
        "fallback_rate": sum(r["fallback_used"] for r in rows) / n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Head ablation during live CoT action generation."
    )
    parser.add_argument("--enriched-log", required=True, nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-decisions", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--head-sets",
        nargs="+",
        default=None,
        help="Named sets as 'name:22 9 15' (default: triplet + control)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["baseline", "triplet", "control"],
        choices=["baseline", "triplet", "control"],
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_logs = list(args.enriched_log)
    agent_config = _load_agent_config(enriched_logs[0])
    model_id = agent_config["model_id"]
    short = _short_model_name(model_id)
    defaults = DEFAULT_HEAD_SETS.get(short)
    if defaults is None:
        print(f"[abort] unknown model family for {model_id}", file=sys.stderr)
        sys.exit(2)

    layer = args.layer if args.layer is not None else defaults["layer"]
    triplet_heads = defaults["triplet"]
    control_heads = defaults["control"]

    pool = []
    for log_path in enriched_logs:
        for rec in _iter_decisions(log_path):
            am = rec.get("action_metadata")
            if am and am.get("raw_response"):
                pool.append(rec)
    # dedupe by (seed, decision_idx) if pooling multiple logs
    seen = set()
    deduped = []
    for rec in pool:
        key = (rec.get("seed"), rec.get("decision_idx"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
    pool = deduped
    if not pool:
        print("[abort] no decisions in log", file=sys.stderr)
        sys.exit(2)

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n_decisions, len(pool)))
    print(f"[init] model={model_id} layer={layer} n={len(sample)}")

    print(f"[init] loading agent on {args.device} ...")
    t0 = time.time()
    agent = _make_agent(agent_config, args.device)
    print(f"[init] loaded in {time.time() - t0:.1f}s")

    results = {"model_id": model_id, "layer": layer, "conditions": {}}

    if "baseline" in args.conditions:
        print("[run] baseline (no ablation) ...")
        rows = _run_condition(agent, sample, ablation=None)
        results["conditions"]["baseline"] = _aggregate(rows)
        with open(out_dir / "baseline_rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    if "triplet" in args.conditions:
        print(f"[run] triplet ablation heads={triplet_heads} ...")
        abl = AttnHeadZeroAblation(agent.model, layer, triplet_heads)
        rows = _run_condition(agent, sample, ablation=abl)
        results["conditions"]["triplet"] = _aggregate(rows)
        with open(out_dir / "triplet_rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    if "control" in args.conditions:
        print(f"[run] control ablation heads={control_heads} ...")
        abl = AttnHeadZeroAblation(agent.model, layer, control_heads)
        rows = _run_condition(agent, sample, ablation=abl)
        results["conditions"]["control"] = _aggregate(rows)
        with open(out_dir / "control_rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    base = results["conditions"].get("baseline", {})
    trip = results["conditions"].get("triplet", {})
    md = [
        "# Inference-time head ablation (behavioral)",
        "",
        f"- Model: `{model_id}`",
        f"- Layer: **{layer}** (heads zeroed at last position each generate step)",
        f"- n_decisions: {len(sample)} (seed={args.seed})",
        f"- Triplet heads: `{triplet_heads}`",
        f"- Control heads: `{control_heads}`",
        "",
        "| Condition | illegal_FOLD rate | clean_CHECK rate | parse OK | fallback |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "triplet", "control"):
        if name not in results["conditions"]:
            continue
        a = results["conditions"][name]
        md.append(
            f"| **{name}** | {a['illegal_fold_rate']*100:.1f}% "
            f"| {a['clean_check_rate']*100:.1f}% "
            f"| {a['parse_success_rate']*100:.1f}% "
            f"| {a['fallback_rate']*100:.1f}% |"
        )
    if base and trip:
        delta_if = (trip["illegal_fold_rate"] - base["illegal_fold_rate"]) * 100
        md.append("")
        md.append(
            f"**Δ illegal_FOLD (triplet − baseline): {delta_if:+.1f} pp**"
        )
        if delta_if <= -5:
            md.append(
                "- Heads are **behaviorally necessary** for the illegal_FOLD "
                "failure mode: ablating them reduces rescued-FOLD rate."
            )
        elif abs(delta_if) < 2:
            md.append(
                "- No meaningful change in illegal_FOLD rate → **redundant** at "
                "generation time (consistent with forward-pass ablation)."
            )
        else:
            md.append(
                "- Mixed or increased illegal_FOLD rate; interpret with per-row "
                "JSONL and recorded vs replay bucket columns."
            )
    md.append("")
    md.append("## Reading guide")
    md.append(
        "- Compare **replay_bucket** to **recorded_bucket** in `*_rows.jsonl` "
        "to see whether re-inference matches the original log under baseline."
    )
    md.append(
        "- A large drop in `illegal_fold` under triplet ablation supports: "
        "the L* head set is necessary for the CoT-conditional failure mode."
    )
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"[done] wrote {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
