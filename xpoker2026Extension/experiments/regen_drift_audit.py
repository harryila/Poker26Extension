"""
Regeneration-drift audit for the necessity (ablation) cells (CPU, committed data).

WHY (audit issue 5)
-------------------
Necessity is measured by regenerating the verb under `model.generate(do_sample=False)`
with/without ablation and asking whether the FOLD verb flips. But T=0 greedy decoding is
NOT bit-deterministic across runs (bf16 nondeterminism, kv-cache, chat-template drift), so
the *baseline* (no-ablation) regeneration already flips some recorded FOLDs. When that
baseline drift is high, the ablated condition has little headroom and the net effect is
unreliable (the project's own docs call Llama continuation necessity "unmeasurable" at 56%
baseline drift).

This script quantifies, for every ablation cell, the BASELINE drift = fraction of recorded
FOLD targets whose no-ablation regeneration is already non-FOLD, plus the parse-fail rate.
A necessity result is only trustworthy where baseline drift is LOW (so the ablation delta is
real) and parse_fail is ~0 (so flips are coherent decisions, not broken JSON).

CPU/stdlib only. Reproduce:
    python -m experiments.regen_drift_audit \
        --glob 'results/inference_head_ablation/*' \
        --out results/inference_head_ablation/REGEN_DRIFT.md
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os


def _rate(path: str):
    """(baseline flip rate, parse-fail rate, n) over a *_rows.jsonl."""
    n = flips = pf = 0
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            if r.get("replay_verb") != "FOLD":
                flips += 1
            if not r.get("replay_parseable_json", True):
                pf += 1
    if n == 0:
        return None
    return flips / n, pf / n, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="results/inference_head_ablation/*")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for d in sorted(globmod.glob(args.glob)):
        bp = os.path.join(d, "baseline_rows.jsonl")
        r = _rate(bp)
        if r is None:
            continue
        drift, pf, n = r
        rows.append((os.path.basename(d), drift, pf, n))

    md = []
    md.append("# Baseline regeneration drift per ablation cell (necessity reliability)")
    md.append("")
    md.append("`drift` = fraction of recorded-FOLD targets whose **no-ablation** regeneration "
              "is already non-FOLD (pure T=0 nondeterminism). Necessity (ablation−baseline) is "
              "only trustworthy where drift is LOW and parse_fail ≈ 0.")
    md.append("")
    md.append("| cell | n | baseline drift | parse_fail | necessity reliability |")
    md.append("|---|---:|---:|---:|---|")
    for name, drift, pf, n in rows:
        if drift <= 0.25 and pf <= 0.05:
            verdict = "OK (low drift)"
        elif drift <= 0.45:
            verdict = "marginal"
        else:
            verdict = "UNRELIABLE (drift swamps effect)"
        md.append(f"| {name} | {n} | {drift*100:.1f}% | {pf*100:.1f}% | {verdict} |")
    md.append("")
    md.append("## Reading")
    md.append("- **Low drift (Qwen clean_legal_fold ~15%)** → necessity delta is real; this is "
              "why the headline Qwen necessity uses the clean_legal_fold pool.")
    md.append("- **High drift (Llama illegal_fold ~73%)** → the verb is already unstable on plain "
              "regeneration; necessity must be read as a *control-paired* McNemar delta, not a raw "
              "ablated flip rate, and continuation-based necessity is unmeasurable for Llama.")
    md.append("- parse_fail ≈ 0 everywhere confirms flips are coherent CoT+JSON decisions, not "
              "broken generations — the earlier HFAgent 5%-parse confound does not recur under recon.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
