"""
C1 control: is the oracle decodable from the residual because the model COMPUTES the
posterior, or merely because the prompt INPUTS are linearly present? (CPU)

The oracle_strategy_aware posterior is a deterministic function of the prompt (hero cards,
board, opponent actions), which the residual encodes — so high decodability at the decision
layer L* could be trivial input-presence. Control: decode the SAME oracle from an EARLY layer
(e.g. L2). If early ≈ late, decodability is input-presence (the answer was always linearly
there); if late ≫ early, the model BUILDS the posterior across depth ("computes", not copies).

Consumes the {..}.json sidecars written by encode_vs_decode.py --emit-json, groups by model,
and compares the lowest-layer ('early') to the highest-layer ('late') trash-mass R^2.

Usage (CPU, after the early + late encode_vs_decode runs):
    python -m experiments.encode_layer_compare \
        --glob 'results/direction_probe_baselines/ENCODE_VS_DECODE_*.json' \
        --out results/direction_probe_baselines/ENCODE_LAYER_COMPARE.md
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import re

# model key from filename ENCODE_VS_DECODE_<model>_l<L>.json
_RE = re.compile(r"ENCODE_VS_DECODE_(.+?)_l(\d+)\.json$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="results/direction_probe_baselines/ENCODE_VS_DECODE_*.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_model: dict[str, list] = {}
    for f in sorted(globmod.glob(args.glob)):
        m = _RE.search(os.path.basename(f))
        if not m:
            continue
        model = m.group(1)
        rec = json.load(open(f))
        rec["_layer_from_name"] = int(m.group(2))
        by_model.setdefault(model, []).append(rec)

    md = ["# Encode-vs-decode control: early-layer vs late-layer oracle decodability", ""]
    md.append("Does the residual decode the Bayesian posterior because the model COMPUTES it "
              "(late ≫ early) or because the prompt inputs are trivially present (early ≈ late)?")
    md.append("")
    md.append("| model | early L | early trash R² | late L | late trash R² | Δ(late−early) | verdict |")
    md.append("|---|---:|---:|---:|---:|---:|---|")
    for model, recs in sorted(by_model.items()):
        recs = sorted(recs, key=lambda r: r.get("layer", r["_layer_from_name"]))
        early, late = recs[0], recs[-1]
        le = early.get("layer", early["_layer_from_name"])
        ll = late.get("layer", late["_layer_from_name"])
        re_, rl = early["trash_r2"], late["trash_r2"]
        d = rl - re_
        if ll == le:
            verdict = "(only one layer — run an early-layer recapture)"
        elif d >= 0.15:
            verdict = "COMPUTED — posterior is built up across depth (late ≫ early)"
        elif d <= 0.05 and rl > 0.2:
            verdict = "INPUT-PRESENCE — decodable from the start; 'knows' claim NOT supported"
        else:
            verdict = "partial build-up"
        md.append(f"| {model} | {le} | {re_:+.3f} | {ll} | {rl:+.3f} | {d:+.3f} | {verdict} |")
    md.append("")
    md.append("## Reading")
    md.append("- **COMPUTED** (late−early ≥ ~0.15) supports the strong claim: the model progressively "
              "builds the correct posterior; its miscalibrated *stated* belief is a readout failure.")
    md.append("- **INPUT-PRESENCE** (early ≈ late, both high) ⇒ downgrade to the safe claim: "
              "'the information sufficient to compute the posterior is linearly available; the "
              "verbalized belief discards it' — do NOT say the model 'knows'.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
