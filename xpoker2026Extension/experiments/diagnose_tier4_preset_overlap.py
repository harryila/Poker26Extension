"""
Diagnose Tier 4 opponent-preset DUPLICATION (cross-preset prompt overlap).

Context
-------
Tier 4 seeds the stochastic opponent with `base_seed + player_index` (= 43)
for EVERY preset (`run_experiment.create_agents`). Presets with near-identical
policies — notably `tight_aggressive` and `loose_aggressive` (both
aggression=0.6; differ only in fold_threshold 0.4 vs 0.2 and bluff_freq) —
then draw the SAME RNG stream and can produce BYTE-IDENTICAL opponent action
sequences. When that happens the hero (the LLM under study) sees identical
game states, so the two nominal presets collapse into one distribution.

This script quantifies the collapse by computing, per model, the pairwise
overlap of the recorded `prompt_hash` set across presets. A 100% overlap means
the two presets are the same data labeled twice (NOT independent evidence for
opponent-invariance).

CPU-only — no model load, pure JSON. Runs in seconds.

Usage::

    python -m experiments.diagnose_tier4_preset_overlap
    python -m experiments.diagnose_tier4_preset_overlap \
        --models llama-8b qwen-8b ministral-8b \
        --out results/diagnostics/tier4_preset_overlap/SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

DEFAULT_PRESETS = [
    "default",
    "informative_v2",
    "tight_aggressive",
    "loose_aggressive",
    "loose_passive",
]
DEFAULT_MODELS = ["llama-8b", "qwen-8b", "ministral-8b"]


def _enriched_path(preset: str, model: str, temp_tag: str = "t00", seed: int = 42) -> str:
    return f"logs/opp_{preset}_{model}_{temp_tag}_s{seed}_enriched.jsonl"


def _decision_prompt_hashes(path: str) -> set[str]:
    """Collect prompt_hash for every decision record (skip summaries/config)."""
    hashes: set[str] = set()
    if not os.path.exists(path):
        return hashes
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") in ("hand_summary", "run_config"):
                continue
            if "action_metadata" in rec and "hand_id" in rec:
                ph = rec.get("prompt_hash")
                if ph:
                    hashes.add(ph)
    return hashes


def diagnose_model(model: str, presets: list[str]) -> dict:
    sets: dict[str, set[str]] = {}
    missing: list[str] = []
    for preset in presets:
        path = _enriched_path(preset, model)
        s = _decision_prompt_hashes(path)
        if not s:
            missing.append(preset)
        sets[preset] = s

    rows = []
    for p1, p2 in combinations(presets, 2):
        a, b = sets[p1], sets[p2]
        inter = len(a & b)
        union = len(a | b)
        jacc = (inter / union) if union else 0.0
        identical = (len(a) > 0 and a == b)
        rows.append({
            "p1": p1, "p2": p2,
            "n1": len(a), "n2": len(b),
            "intersection": inter, "union": union,
            "jaccard": jacc, "identical": identical,
        })
    return {"sizes": {p: len(s) for p, s in sets.items()}, "missing": missing, "pairs": rows}


def render(results: dict[str, dict], presets: list[str]) -> str:
    out: list[str] = []
    out.append("# Tier 4 opponent-preset duplication diagnostic")
    out.append("")
    out.append("Cross-preset overlap of recorded `prompt_hash` sets (decision records).")
    out.append("**`identical=YES` means two presets are the same data labeled twice** — ")
    out.append("the opponent's action sequence collapsed because it shares the `seed=43` ")
    out.append("RNG stream and the presets' policies are too similar to diverge. See ")
    out.append("`AUDIT_FINDINGS.md` §12.")
    out.append("")
    for model, res in results.items():
        out.append(f"## {model}")
        out.append("")
        sizes = ", ".join(f"{p}={res['sizes'][p]}" for p in presets)
        out.append(f"- distinct prompt_hashes per preset: {sizes}")
        if res["missing"]:
            out.append(f"- ⚠️ missing/empty logs: {', '.join(res['missing'])}")
        out.append("")
        out.append("| preset A | preset B | |A| | |B| | A∩B | A∪B | Jaccard | identical? |")
        out.append("|---|---|---:|---:|---:|---:|---:|:--:|")
        for r in res["pairs"]:
            flag = "**YES**" if r["identical"] else "no"
            out.append(
                f"| {r['p1']} | {r['p2']} | {r['n1']} | {r['n2']} | "
                f"{r['intersection']} | {r['union']} | {r['jaccard']:.2f} | {flag} |"
            )
        out.append("")
        # Group presets into connected components by the "identical" relation
        # (union-find) so transitive collapses count as ONE distribution.
        parent = {p: p for p in presets}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for r in res["pairs"]:
            if r["identical"]:
                union(r["p1"], r["p2"])

        groups: dict[str, list[str]] = {}
        for p in presets:
            # only count presets with a non-empty log toward distinct distributions
            if res["sizes"].get(p, 0) > 0:
                groups.setdefault(find(p), []).append(p)
        collapsed_groups = [g for g in groups.values() if len(g) > 1]
        n_distinct = len(groups)
        res["n_distinct"] = n_distinct

        if collapsed_groups:
            desc = "; ".join("≡".join(g) for g in collapsed_groups)
            out.append(
                f"- **Collapsed groups: {desc} ⇒ {n_distinct} distinct "
                f"distribution(s), NOT {len([p for p in presets if res['sizes'].get(p,0)>0])}.** "
                "Report each collapsed group as a single cell."
            )
        else:
            out.append(f"- No fully-collapsed pairs; {n_distinct} distinct distributions.")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--presets", nargs="+", default=DEFAULT_PRESETS)
    ap.add_argument("--out", default="results/diagnostics/tier4_preset_overlap/SUMMARY.md")
    args = ap.parse_args()

    results = {m: diagnose_model(m, args.presets) for m in args.models}
    md = render(results, args.presets)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print(md)
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
