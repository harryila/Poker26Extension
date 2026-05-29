"""
C2 belief-side readout: can steering the calibration direction REPAIR the miscalibrated
stated belief? (GPU) — the intervention complement to encode_vs_decode (C1).

C1 shows the correct posterior is linearly present in the residual but the STATED belief
discards it (readout failure). This tests the causal converse: if we ADD the trash/calibration
direction during belief elicitation, does the STATED belief move toward the oracle
(JS(belief, oracle) drops), beating a random-direction control — i.e. is the readout failure
fixable by intervention?

Reuses HFAgent.belief(obs) (the exact tested elicitation+parse path); the steering hook is
attached via agent._extra_generation_hooks (same mechanism as the gameplay hook). Reports
parse/None rate so over-steering (broken generation) is visible, not silently scored.

Usage (GPU):
    python -m experiments.steer_belief_readout \
        --enriched-log logs/cot_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz ... \
        --layer 19 --direction results/direction_probe/qwen8b_l23/steer_trash_direction.npz \
        --alphas 0 2 4 --n-decisions 40 --device cuda --dtype bfloat16 \
        --out-dir results/posterior_steering/qwen8b_belief_l19
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from poker_env.agents import HFAgent
from poker_env.interp.forward_helpers import obs_from_dict
from poker_env.interp.patching import ActivationAdditionHook
from poker_env.config import BUCKET_ORDER
from experiments.causal_patching import _load_agent_config, _iter_decisions, classify_decision


def _vec14(d):
    if not isinstance(d, dict):
        return None
    return np.array([float(d.get(b) or 0.0) for b in BUCKET_ORDER], dtype=np.float64)


def _js(p, q, eps=1e-9):
    p = np.clip(p, eps, None); p /= p.sum()
    q = np.clip(q, eps, None); q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enriched-log", nargs="+", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 2, 4])
    ap.add_argument("--target-bucket", default=None,
                    help="restrict to one bucket (e.g. illegal_fold); default = any decision with oracle")
    ap.add_argument("--n-decisions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    dirnpz = np.load(args.direction, allow_pickle=True)
    base_vec = torch.tensor(dirnpz["direction"].astype(np.float32))
    norm = float(dirnpz["resid_mean_norm"]) if "resid_mean_norm" in dirnpz else 1.0
    g = torch.Generator().manual_seed(args.seed)
    rand_vec = torch.randn(base_vec.shape, generator=g); rand_vec /= rand_vec.norm()

    ac = _load_agent_config(args.enriched_log[0])
    agent = HFAgent(
        model_id=ac["model_id"], device_map=args.device,
        temperature=float(ac.get("temperature", 0.2)), top_p=float(ac.get("top_p", 0.9)),
        cot_mode=bool(ac.get("cot_mode", True)), belief_format=ac.get("belief_format", "compact"),
        belief_max_new_tokens=int(ac.get("belief_max_new_tokens", 384)),
        max_input_tokens=int(ac.get("max_input_tokens", 2048)), name="steer_belief",
    )

    # build target pool: decisions with an oracle posterior (and optional bucket filter)
    pool = []
    for log in args.enriched_log:
        for rec in _iter_decisions(log):
            if not isinstance(rec.get("obs"), dict):
                continue
            if not rec.get("oracle_strategy_aware"):
                continue
            if args.target_bucket:
                am = rec.get("action_metadata")
                if not (isinstance(am, dict) and am.get("raw_response")):
                    continue
                if classify_decision(rec) != args.target_bucket:
                    continue
            pool.append(rec)
    rng.shuffle(pool)
    pool = pool[:args.n_decisions]
    print(f"[pool] {len(pool)} target decisions")

    def run(vec, label):
        rows = []
        for rec in pool:
            obs = obs_from_dict(rec["obs"])
            oracle = _vec14(rec["oracle_strategy_aware"])
            row = {"by_alpha": {}}
            for a in args.alphas:
                if a == 0:
                    agent._extra_generation_hooks = []
                else:
                    agent._extra_generation_hooks = [
                        ActivationAdditionHook(agent.model, args.layer, vec * norm,
                                               alpha=float(a), last_only=False)]
                with torch.no_grad():
                    b = agent.belief(obs)
                agent._extra_generation_hooks = []
                bv = _vec14(b)
                if bv is None or bv.sum() <= 0:
                    row["by_alpha"][str(a)] = {"js": None, "parsed": False}
                else:
                    row["by_alpha"][str(a)] = {"js": _js(bv, oracle), "parsed": True}
            rows.append(row)
        return rows

    os.makedirs(args.out_dir, exist_ok=True)
    out = {"model_id": ac["model_id"], "layer": args.layer, "target_bucket": args.target_bucket,
           "alphas": args.alphas, "n": len(pool), "conditions": {}}
    for vec, label in [(base_vec, "trash_direction"), (rand_vec, "random_control")]:
        rows = run(vec, label)
        with open(os.path.join(args.out_dir, f"{label}_rows.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        agg = {}
        for a in args.alphas:
            js = [r["by_alpha"][str(a)]["js"] for r in rows if r["by_alpha"][str(a)]["parsed"]]
            parsed = sum(1 for r in rows if r["by_alpha"][str(a)]["parsed"])
            agg[str(a)] = {"mean_js": float(np.mean(js)) if js else None,
                           "parse_rate": parsed / len(rows)}
        out["conditions"][label] = agg

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    md = ["# Steer-to-repair-belief — does steering reduce JS(stated belief, oracle)?", "",
          f"- model={ac['model_id']} layer={args.layer} target={args.target_bucket} n={len(pool)}",
          "- JS lower = belief closer to the Bayesian oracle. parse_rate guards against over-steer.",
          "", "| alpha | trash: mean JS | trash: parse | control: mean JS | control: parse |",
          "|---:|---:|---:|---:|---:|"]
    for a in args.alphas:
        t = out["conditions"]["trash_direction"][str(a)]
        c = out["conditions"]["random_control"][str(a)]
        tj = f"{t['mean_js']:.3f}" if t["mean_js"] is not None else "—"
        cj = f"{c['mean_js']:.3f}" if c["mean_js"] is not None else "—"
        md.append(f"| {a} | {tj} | {t['parse_rate']*100:.0f}% | {cj} | {c['parse_rate']*100:.0f}% |")
    md += ["", "## Reading",
           "- trash JS DROPS with alpha while parse_rate stays high, AND drops more than the random "
           "control ⇒ steering causally repairs the readout (the belief moves toward the posterior "
           "the residual already encodes — the intervention converse of encode_vs_decode).",
           "- If parse_rate collapses, alpha is too large (broken generation) — lower it."]
    with open(os.path.join(args.out_dir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[written] {args.out_dir}/SUMMARY.md")


if __name__ == "__main__":
    main()
