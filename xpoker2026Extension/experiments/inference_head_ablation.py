"""
Behavioral head-ablation during live CoT action generation.

Zeros the L* sufficient-head set on every forward pass inside
``model.generate()`` and re-runs action selection on decision records from
an enriched log. Compares bucket rates (especially ``illegal_fold``) to an
unablated baseline on the same decisions.

This is the necessity counterpart to forward-pass head ablation: sufficiency
was shown by activation patching; this tests whether disrupting the heads
during real inference changes emitted actions.

Two pipelines are supported:

  * ``--pipeline hfagent``   (legacy) — uses ``HFAgent.act_with_metadata``.
    Captures ``parse_success`` / ``fallback_used`` fields from the agent.
    Has a known regen-fidelity confound (see AUDIT_FINDINGS.md): on Ministral,
    only 4/80 recorded ``illegal_fold`` records reproduce a parseable JSON
    response under HFAgent re-inference at T=0.
  * ``--pipeline recon``     (default) — uses ``PromptReconstructor`` + raw
    ``model.generate(do_sample=False)``, **identical to** the
    ``regenerate_ablated`` path in ``continuation_after_patch.py``. This makes
    inference- and continuation-side necessity numbers apples-to-apples.

Use ``--filter-recorded-bucket illegal_fold`` to evaluate flip rate on the
exact pool of recorded illegal_fold targets (matches continuation).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_env.agents.hf_agent import HFAgent  # noqa: E402
from poker_env.agents.prompts import get_action_system_message  # noqa: E402
from poker_env.interp.forward_helpers import (  # noqa: E402
    PromptReconstructor,
    attached_hooks,
    obs_from_dict,
)
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


_VERB_RE = re.compile(r'"action"\s*:\s*"([A-Za-z_]+)"', re.IGNORECASE)


def _extract_verb(raw: str) -> tuple[str, bool]:
    """Returns (verb_canonical, parseable_json).

    canonical ∈ {FOLD, CHECK_OR_CALL, BET_OR_RAISE, UNK}.
    """
    if not raw:
        return ("UNK", False)
    m = _VERB_RE.search(raw)
    if not m:
        return ("UNK", False)
    v = m.group(1).upper()
    if v in {"FOLD"}:
        return ("FOLD", True)
    if v in {"CHECK_OR_CALL", "CHECK", "CALL"}:
        return ("CHECK_OR_CALL", True)
    if v in {"BET_OR_RAISE", "BET", "RAISE"}:
        return ("BET_OR_RAISE", True)
    return (v, True)


def _run_condition_hfagent(
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
            _action, meta = agent.act_with_metadata(obs)
            md = meta.to_dict()
            verb, parseable = _extract_verb(md.get("raw_response", "") or "")
            rows.append({
                "seed": rec.get("seed"),
                "decision_idx": rec.get("decision_idx"),
                "hand_id": rec.get("hand_id"),
                "recorded_bucket": classify_decision(rec),
                "replay_bucket": classify_decision({
                    "obs": rec["obs"],
                    "legal_actions": rec.get("legal_actions"),
                    "action_metadata": md,
                }),
                "recorded_action": rec.get("action"),
                "replay_action": meta.action_chosen,
                "replay_verb": verb,
                "replay_parseable_json": parseable,
                "parse_success": meta.parse_success,
                "fallback_used": meta.fallback_used,
                "raw_response": (md.get("raw_response", "") or "")[:1200],
            })
    finally:
        agent._extra_generation_hooks = []
    return rows


def _run_condition_recon(
    model,
    tokenizer,
    recon: PromptReconstructor,
    records: list[dict],
    *,
    ablation: AttnHeadZeroAblation | None,
    max_new_tokens: int,
    device: str,
) -> list[dict]:
    """Mirrors ``continuation_after_patch._full_generate``: PromptReconstructor
    + raw ``model.generate(do_sample=False)``, attaching the ablation hook for
    every forward pass."""
    rows: list[dict] = []
    hooks = [ablation] if ablation is not None else []
    for rec in records:
        chat_prompt = recon.build(rec)
        enc = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        gen_kw = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        with attached_hooks(hooks):
            with torch.no_grad():
                out = model.generate(input_ids=input_ids, **gen_kw)
        new_ids = out[0, input_ids.shape[1]:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        verb, parseable = _extract_verb(raw)

        legal = rec.get("legal_actions") or []
        # Build a synthetic action_metadata for classify_decision.
        synth_md = {
            "raw_response": raw,
            "parse_success": parseable,
            "fallback_used": not parseable,
        }
        replay_bucket = classify_decision({
            "obs": rec["obs"],
            "legal_actions": legal,
            "action_metadata": synth_md,
        })
        rows.append({
            "seed": rec.get("seed"),
            "decision_idx": rec.get("decision_idx"),
            "hand_id": rec.get("hand_id"),
            "recorded_bucket": classify_decision(rec),
            "replay_bucket": replay_bucket,
            "recorded_action": rec.get("action"),
            "replay_action": None,
            "replay_verb": verb,
            "replay_parseable_json": parseable,
            "parse_success": parseable,
            "fallback_used": not parseable,
            "raw_response": raw[:1200],
        })
    return rows


def _aggregate(rows: list[dict]) -> dict:
    c_recorded = Counter(r["recorded_bucket"] for r in rows)
    c_replay = Counter(r["replay_bucket"] for r in rows)
    c_verb = Counter(r["replay_verb"] for r in rows)
    n = len(rows) or 1
    return {
        "n": len(rows),
        "recorded_bucket_frac": {k: v / n for k, v in c_recorded.items()},
        "replay_bucket_frac": {k: v / n for k, v in c_replay.items()},
        "replay_verb_frac": {k: v / n for k, v in c_verb.items()},
        "illegal_fold_rate": c_replay.get("illegal_fold", 0) / n,
        "clean_check_rate": c_replay.get("clean_check_or_call", 0) / n,
        "parse_success_rate": sum(r["parse_success"] for r in rows) / n,
        "fallback_rate": sum(r["fallback_used"] for r in rows) / n,
        "parseable_json_rate": sum(r["replay_parseable_json"] for r in rows) / n,
    }


def _flip_rates(rows: list[dict]) -> dict:
    """For records where recorded_verb is FOLD (i.e. the recorded raw_response
    parsed FOLD), what fraction of replay verbs are non-FOLD parseable?"""
    fold_rows = [r for r in rows if (r["recorded_bucket"] in (
        "illegal_fold", "clean_legal_fold"
    ))]
    n = len(fold_rows) or 1
    flip_to_check = sum(1 for r in fold_rows
                        if r["replay_verb"] == "CHECK_OR_CALL")
    flip_to_bet = sum(1 for r in fold_rows
                      if r["replay_verb"] == "BET_OR_RAISE")
    parse_fail = sum(1 for r in fold_rows if not r["replay_parseable_json"])
    return {
        "n_recorded_fold": len(fold_rows),
        "flip_to_check_rate": flip_to_check / n,
        "flip_to_bet_rate": flip_to_bet / n,
        "any_flip_rate": (flip_to_check + flip_to_bet) / n,
        "parse_fail_rate": parse_fail / n,
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
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--head-sets", nargs="+", default=None,
        help="Named sets as 'name:22 9 15' (default: triplet + control)",
    )
    parser.add_argument(
        "--conditions", nargs="+",
        default=["baseline", "triplet", "control"],
        choices=["baseline", "triplet", "control"],
    )
    parser.add_argument(
        "--pipeline", choices=["hfagent", "recon"], default="recon",
        help="recon (default) = PromptReconstructor + raw model.generate "
             "(matches continuation_after_patch). hfagent = legacy HFAgent "
             "path (kept for backward compat; has regen-fidelity confound).",
    )
    parser.add_argument(
        "--filter-recorded-bucket", default=None,
        help="If set (e.g. 'illegal_fold'), only consider records whose "
             "recorded_bucket matches. Use this to evaluate flip rate on "
             "the exact target pool of continuation_after_patch.py.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Max new tokens for raw generate (recon pipeline).",
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
    seen = set()
    deduped = []
    for rec in pool:
        key = (rec.get("seed"), rec.get("decision_idx"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
    pool = deduped
    if args.filter_recorded_bucket:
        pool = [r for r in pool
                if classify_decision(r) == args.filter_recorded_bucket]
        print(f"[filter] recorded_bucket=={args.filter_recorded_bucket}: "
              f"{len(pool)} records")
    if not pool:
        print("[abort] no decisions in log", file=sys.stderr)
        sys.exit(2)

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n_decisions, len(pool)))
    print(f"[init] model={model_id} layer={layer} pipeline={args.pipeline} "
          f"n={len(sample)}")

    print(f"[init] loading {args.pipeline} on {args.device} ...")
    t0 = time.time()
    if args.pipeline == "hfagent":
        agent = _make_agent(agent_config, args.device)
        model = agent.model
        tokenizer = agent.tokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.dtype]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=args.device,
        )
        model.eval()
        agent = None
    recon = PromptReconstructor(tokenizer, agent_config) if args.pipeline == "recon" else None
    print(f"[init] loaded in {time.time() - t0:.1f}s")

    def _run(name: str, ablation):
        if args.pipeline == "hfagent":
            return _run_condition_hfagent(agent, sample, ablation=ablation)
        return _run_condition_recon(
            model, tokenizer, recon, sample, ablation=ablation,
            max_new_tokens=args.max_new_tokens, device=args.device,
        )

    results = {
        "model_id": model_id,
        "layer": layer,
        "pipeline": args.pipeline,
        "filter_recorded_bucket": args.filter_recorded_bucket,
        "conditions": {},
    }

    if "baseline" in args.conditions:
        print("[run] baseline (no ablation) ...")
        rows = _run("baseline", None)
        results["conditions"]["baseline"] = {
            **_aggregate(rows),
            "fold_target_flips": _flip_rates(rows),
        }
        with open(out_dir / "baseline_rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    if "triplet" in args.conditions:
        print(f"[run] triplet ablation heads={triplet_heads} ...")
        abl = AttnHeadZeroAblation(model, layer, triplet_heads)
        rows = _run("triplet", abl)
        results["conditions"]["triplet"] = {
            **_aggregate(rows),
            "fold_target_flips": _flip_rates(rows),
        }
        with open(out_dir / "triplet_rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    if "control" in args.conditions:
        print(f"[run] control ablation heads={control_heads} ...")
        abl = AttnHeadZeroAblation(model, layer, control_heads)
        rows = _run("control", abl)
        results["conditions"]["control"] = {
            **_aggregate(rows),
            "fold_target_flips": _flip_rates(rows),
        }
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
        f"- Layer: **{layer}**",
        f"- Pipeline: **`{args.pipeline}`**"
        + (" (PromptReconstructor + raw `model.generate` — matches "
           "`continuation_after_patch.regenerate_ablated`)"
           if args.pipeline == "recon" else
           " (HFAgent — legacy; has regen-fidelity confound)"),
        f"- **Ablation scope:** zeros triplet/control heads at the **last "
        f"sequence position on every forward pass during action `generate()`** "
        f"(full CoT reasoning + JSON). This is **more aggressive** than "
        f"single-position verb patching at L*.",
        f"- Filter (recorded_bucket): `{args.filter_recorded_bucket or 'none'}`",
        f"- n_decisions: {len(sample)} (seed={args.seed})",
        f"- Triplet heads: `{triplet_heads}`",
        f"- Control heads: `{control_heads}`",
        "",
        "## Aggregate replay rates",
        "",
        "| Condition | parseable JSON | illegal_FOLD | clean_CHECK | "
        "verb=FOLD | verb=CHECK | verb=BET | verb=UNK |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "triplet", "control"):
        if name not in results["conditions"]:
            continue
        a = results["conditions"][name]
        v = a.get("replay_verb_frac", {})
        md.append(
            f"| **{name}** | {a['parseable_json_rate']*100:.1f}% "
            f"| {a['illegal_fold_rate']*100:.1f}% "
            f"| {a['clean_check_rate']*100:.1f}% "
            f"| {v.get('FOLD',0)*100:.1f}% "
            f"| {v.get('CHECK_OR_CALL',0)*100:.1f}% "
            f"| {v.get('BET_OR_RAISE',0)*100:.1f}% "
            f"| {v.get('UNK',0)*100:.1f}% |"
        )
    md += [
        "",
        "## Flip rate on recorded FOLD pool",
        "",
        "(records where the **recorded** raw_response parsed as FOLD — "
        "either ``illegal_fold`` or ``clean_legal_fold``)",
        "",
        "| Condition | n | FOLD→CHECK | FOLD→BET | any flip | parse fail |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "triplet", "control"):
        cond = results["conditions"].get(name)
        if not cond:
            continue
        f_ = cond["fold_target_flips"]
        md.append(
            f"| **{name}** | {f_['n_recorded_fold']} "
            f"| {f_['flip_to_check_rate']*100:.1f}% "
            f"| {f_['flip_to_bet_rate']*100:.1f}% "
            f"| {f_['any_flip_rate']*100:.1f}% "
            f"| {f_['parse_fail_rate']*100:.1f}% |"
        )

    if base and trip:
        delta_if = (trip["illegal_fold_rate"] - base["illegal_fold_rate"]) * 100
        md.append("")
        md.append(
            f"**Δ illegal_FOLD (triplet − baseline): {delta_if:+.1f} pp** "
            f"(see flip-rate table for the cleaner direct measure)"
        )
        b_flip = base["fold_target_flips"]["any_flip_rate"]
        t_flip = trip["fold_target_flips"]["any_flip_rate"]
        net_flip_pp = (t_flip - b_flip) * 100
        md.append(
            f"**Net ablation-induced FOLD-flip: triplet {t_flip*100:.1f}% − "
            f"baseline {b_flip*100:.1f}% = {net_flip_pp:+.1f} pp**"
        )
        if net_flip_pp >= 30:
            md.append(
                "- Heads are **behaviorally necessary** for the FOLD action: "
                "ablating them flips a large fraction of recorded-FOLD "
                "decisions to CHECK/BET."
            )
        elif net_flip_pp >= 10:
            md.append(
                "- Moderate behavioral necessity — partial flip rate over "
                "baseline. Inspect parse_fail to rule out incoherence."
            )
        else:
            md.append(
                "- Heads are **behaviorally redundant** for FOLD generation "
                "(no large flip beyond baseline). Consistent with §16/Phase O "
                "redundancy framing."
            )
    md.append("")
    md.append("## Reading guide")
    md.append(
        "- `parse_fail_rate` should track between conditions; if it spikes "
        "under ablation while flip rate stays low, the heads matter for "
        "JSON formatting broadly (general damage), not FOLD-specific."
    )
    md.append(
        "- Compare to `regenerate_ablated` block in "
        "`results/continuation_after_patch/{model}/SUMMARY.md` — should agree "
        "when `--pipeline recon --filter-recorded-bucket illegal_fold` is set."
    )
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"[done] wrote {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
