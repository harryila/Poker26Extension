"""
Sufficiency reporting with the correct unit of analysis (CPU, committed data only).

WHY THIS EXISTS (audit issues 2 & 3)
------------------------------------
The headline sufficiency cells report a single pooled point estimate (e.g. Qwen L23
"top-1 -> CHECK = 100%, spec-adj +18.3 nats") over n = n_source x n_target PAIRS.
Two reporting problems:
  (2) magnitude noise: per-seed spec-adj ranges widely (Qwen L23: +25.0/+15.6/+24.9
      across seeds 42/123/456) because the random-source null varies ~5x by seed.
  (3) PSEUDOREPLICATION: the independent unit is the TARGET decision (illegal_fold is
      rare: only ~24 pooled, 4/9/11 per seed), NOT the n_source x n_target pairs. Each
      target is reused across all sources, so "n=240" overstates precision.

This script reports sufficiency the honest way:
  * categorical top-1 -> CHECK with a TARGET-CLUSTERED bootstrap CI (resample target
    decisions, not pairs);
  * mean Delta(CHECK-FOLD) with a target-clustered bootstrap CI;
  * per-seed spec-adj spread (mean delta - random-source null), so the magnitude is
    reported as a range, not a point.

It SELF-CHECKS against the committed SUMMARY by recomputing the pooled mean delta and
asserting it matches summary.json's per_layer mean within tolerance.

CPU/stdlib+numpy only. Reproduce:
    python -m experiments.sufficiency_ci \
        --pooled results/causal_patching/qwen8b_t0_pooled_layer_sweep \
        --replicates results/causal_patching/qwen8b_t0_s42_replicate \
                     results/causal_patching/qwen8b_t0_s123_replicate \
                     results/causal_patching/qwen8b_t0_s456_replicate \
        --layer 23 \
        --out results/causal_patching/qwen8b_t0_pooled_layer_sweep/SUFFICIENCY_CI.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

CHECK_GROUP = "CHECK_CALL"


def _read_pairs(cell_dir: str, layer: int):
    """Return list of dicts {target, is_check, delta} for the given layer."""
    path = os.path.join(cell_dir, "by_pair.csv")
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["layer"]) != layer:
                continue
            out.append({
                "target": f"{row['target_hand']}#{row['target_dec']}",
                "source": f"{row['source_hand']}#{row['source_dec']}",
                "is_check": 1.0 if row["patched_top1_group"] == CHECK_GROUP else 0.0,
                "delta": float(row["delta_check_minus_fold"]),
            })
    return out


def _null_at_layer(cell_dir: str, layer: int):
    sj = os.path.join(cell_dir, "summary.json")
    if not os.path.exists(sj):
        return None
    s = json.load(open(sj))
    # Preferred schema: per_layer[L].random_null_delta
    pl = s.get("per_layer") or {}
    cell = pl.get(str(layer))
    if cell and cell.get("random_null_delta") is not None:
        return float(cell["random_null_delta"])
    # Fallback schema: random_source_per_layer[L].mean_delta
    rs = s.get("random_source_per_layer") or {}
    c2 = rs.get(str(layer))
    return float(c2["mean_delta"]) if c2 else None


def _cluster_bootstrap(pairs, key, n_boot=5000, seed=0):
    """Target-clustered bootstrap of the mean of pairs[*][key].

    Resamples TARGET decisions with replacement; within a resampled target, takes all
    its pairs. Returns (point, lo, hi) for a 95% percentile CI."""
    rng = np.random.default_rng(seed)
    by_t = {}
    for p in pairs:
        by_t.setdefault(p["target"], []).append(p[key])
    targets = list(by_t)
    vals_by_t = {t: np.array(v, float) for t, v in by_t.items()}
    point = float(np.mean([v for p in pairs for v in [p[key]]]))
    boots = np.empty(n_boot)
    n_t = len(targets)
    for b in range(n_boot):
        pick = rng.choice(n_t, size=n_t, replace=True)
        acc = []
        for i in pick:
            acc.append(vals_by_t[targets[i]])
        boots[b] = np.concatenate(acc).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), n_t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooled", required=True)
    ap.add_argument("--replicates", nargs="*", default=[])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    L = args.layer
    pooled = _read_pairs(args.pooled, L)
    if not pooled:
        raise SystemExit(f"no pairs at layer {L} in {args.pooled}")

    # SELF-CHECK against committed SUMMARY
    recomputed_mean = float(np.mean([p["delta"] for p in pooled]))
    sj = json.load(open(os.path.join(args.pooled, "summary.json")))
    claimed = None
    pl = sj.get("per_layer") or {}
    if str(L) in pl:
        cell = pl[str(L)]
        claimed = cell.get("mean_delta") or cell.get("mean_delta_check_minus_fold")
    check_line = ""
    if claimed is not None:
        ok = abs(recomputed_mean - float(claimed)) < 0.05
        check_line = (f"self-check: recomputed mean Δ = {recomputed_mean:+.3f} vs "
                      f"summary.json {float(claimed):+.3f} -> {'OK' if ok else 'MISMATCH'}")

    flip_pt, flip_lo, flip_hi, n_t = _cluster_bootstrap(pooled, "is_check", args.n_boot)
    d_pt, d_lo, d_hi, _ = _cluster_bootstrap(pooled, "delta", args.n_boot)
    null = _null_at_layer(args.pooled, L)

    md = []
    md.append(f"# Sufficiency with target-clustered CIs — layer L{L}")
    md.append("")
    md.append(f"- Pooled cell: `{args.pooled}`")
    md.append(f"- Pairs at L{L}: {len(pooled)} (= n_source × n_target); "
              f"**independent units (target decisions): {n_t}**")
    if check_line:
        md.append(f"- {check_line}")
    md.append("- Bootstrap resamples TARGET decisions (the independent unit), not pairs. "
              "95% percentile CI.")
    md.append("")
    md.append("## Pooled, clustered by target")
    md.append(f"- **top-1 → CHECK: {flip_pt*100:.1f}%**  (95% CI [{flip_lo*100:.1f}, {flip_hi*100:.1f}])")
    md.append(f"- **mean Δ(CHECK−FOLD): {d_pt:+.2f} nats**  (95% CI [{d_lo:+.2f}, {d_hi:+.2f}])")
    if null is not None:
        md.append(f"- random-source null at L{L}: {null:+.2f} nats  → "
                  f"**spec-adj Δ ≈ {d_pt-null:+.2f}** (CI [{d_lo-null:+.2f}, {d_hi-null:+.2f}], null treated as fixed)")
    md.append("")

    if args.replicates:
        md.append("## Per-seed spread (magnitude is seed-sensitive — report the range)")
        md.append("| seed cell | n_target | top-1 → CHECK | mean Δ | null | spec-adj |")
        md.append("|---|---:|---:|---:|---:|---:|")
        specs = []
        for rc in args.replicates:
            pr = _read_pairs(rc, L)
            if not pr:
                md.append(f"| {os.path.basename(rc)} | (no pairs at L{L}) | | | | |")
                continue
            n_tt = len({p["target"] for p in pr})
            flip = np.mean([p["is_check"] for p in pr])
            md_ = np.mean([p["delta"] for p in pr])
            nl = _null_at_layer(rc, L)
            sa = (md_ - nl) if nl is not None else float("nan")
            specs.append(sa)
            md.append(f"| {os.path.basename(rc)} | {n_tt} | {flip*100:.1f}% | {md_:+.2f} | "
                      f"{(nl if nl is not None else float('nan')):+.2f} | {sa:+.2f} |")
        if specs:
            specs = np.array(specs, float)
            md.append("")
            md.append(f"- **spec-adj across seeds: mean {specs.mean():+.2f}, "
                      f"range [{specs.min():+.2f}, {specs.max():+.2f}]** "
                      f"(report this range, not the pooled point — the null varies ~5× by seed).")
    md.append("")
    md.append("## Reading")
    md.append("- The categorical sufficiency (top-1 → CHECK) is the robust headline; if its CI "
              "is [100,100] it means every target flips under every source (saturated).")
    md.append("- The nat-magnitude spec-adj is seed-sensitive; cite the per-seed range and the "
              "target-clustered CI, NOT the single pooled value.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
