"""
CoT vs non-CoT residual magnitude analysis (B2).

Question answered:
    "Is the §18a 'CoT attenuates the residual prior' claim mechanistically
     visible in residual magnitudes?"

§18a interpretation: in non-CoT mode the model's residual carries a
strong implicit prior (CHECK for Llama, BR for Ministral) that gates
patches' output expression. CoT apparently weakens this prior.

If the claim is true, the residual at L* should:
  - Be SMALLER in magnitude (||x||) under non-CoT than CoT (the prior is
    strong because the residual itself is small/concentrated).
  - OR be LARGER in magnitude with a strong prior axis (the prior is
    encoded as a large displacement along one direction).

This script loads cached residuals from the existing direction-probe
runs (CoT and non-CoT) and computes:
  - L2 norm of each residual
  - Per-bucket mean/std of ||x||
  - Mode-comparison: CoT vs non-CoT magnitudes per bucket
  - Projection magnitude: |residual @ w_verb / ||w_verb|||
  - Centroid distance: ||clean_CC mean - clean_LF mean|| in residual space

Pure analysis, no GPU. Inputs: results/direction_probe/<model>/raw_residuals.npz
and results/direction_probe_nocot/<model>/raw_residuals.npz.

Usage::

    python -m experiments.cot_magnitude_analysis \\
        --cot-npz   results/direction_probe/llama8b_l14/raw_residuals.npz \\
        --nocot-npz results/direction_probe_nocot/llama8b_l14/raw_residuals.npz \\
        --out-md results/cot_magnitude_analysis/llama8b_l14.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="CoT vs non-CoT residual magnitude analysis."
    )
    parser.add_argument("--cot-npz", required=True)
    parser.add_argument("--nocot-npz", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    cot = np.load(args.cot_npz)
    noc = np.load(args.nocot_npz)

    def _stats(X, name):
        norms = np.linalg.norm(X.astype(np.float32), axis=1)
        return {
            "name": name,
            "n": int(len(X)),
            "mean_norm": float(norms.mean()) if len(norms) else None,
            "std_norm":  float(norms.std())  if len(norms) else None,
            "median_norm": float(np.median(norms)) if len(norms) else None,
        }

    rows = []
    rows.append(("CoT  clean_CC", _stats(cot["X_cc"], "cot_cc")))
    rows.append(("CoT  clean_LF", _stats(cot["X_lf"], "cot_lf")))
    if "X_if" in cot.files and len(cot["X_if"]) > 0:
        rows.append(("CoT  illegal_F", _stats(cot["X_if"], "cot_if")))
    rows.append(("nonCoT clean_CC", _stats(noc["X_cc"], "noc_cc")))
    rows.append(("nonCoT clean_LF", _stats(noc["X_lf"], "noc_lf")))
    if "X_if" in noc.files and len(noc["X_if"]) > 0:
        rows.append(("nonCoT illegal_F", _stats(noc["X_if"], "noc_if")))

    # Centroid distances.
    def _centroid_dist(X1, X2):
        if len(X1) == 0 or len(X2) == 0:
            return None
        return float(np.linalg.norm(X1.astype(np.float32).mean(0)
                                    - X2.astype(np.float32).mean(0)))

    cot_centroid_dist = _centroid_dist(cot["X_cc"], cot["X_lf"])
    noc_centroid_dist = _centroid_dist(noc["X_cc"], noc["X_lf"])

    # Per-bucket projections onto each mode's own verb direction.
    def _proj(X, w):
        nw = np.linalg.norm(w) + 1e-9
        return (X.astype(np.float32) @ w) / nw

    w_cot = cot["weight_vec"].astype(np.float32)
    w_noc = noc["weight_vec"].astype(np.float32)

    proj_cot = {
        "cc": _proj(cot["X_cc"], w_cot),
        "lf": _proj(cot["X_lf"], w_cot),
    }
    proj_noc = {
        "cc": _proj(noc["X_cc"], w_noc),
        "lf": _proj(noc["X_lf"], w_noc),
    }

    summary = {
        "cot_npz": args.cot_npz,
        "nocot_npz": args.nocot_npz,
        "rows": [{"label": l, **d} for (l, d) in rows],
        "centroid_distance": {
            "cot_check_minus_fold": cot_centroid_dist,
            "nocot_check_minus_fold": noc_centroid_dist,
            "ratio_nocot_over_cot": (
                noc_centroid_dist / cot_centroid_dist
                if cot_centroid_dist and cot_centroid_dist > 1e-6 else None
            ),
        },
        "projection_magnitudes": {
            "cot_cc_mean_abs":  float(abs(proj_cot["cc"]).mean()) if len(proj_cot["cc"]) else None,
            "cot_lf_mean_abs":  float(abs(proj_cot["lf"]).mean()) if len(proj_cot["lf"]) else None,
            "noc_cc_mean_abs":  float(abs(proj_noc["cc"]).mean()) if len(proj_noc["cc"]) else None,
            "noc_lf_mean_abs":  float(abs(proj_noc["lf"]).mean()) if len(proj_noc["lf"]) else None,
        },
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)

    md = ["# CoT vs non-CoT residual magnitude analysis", ""]
    md.append(f"- CoT probe:    `{args.cot_npz}`")
    md.append(f"- non-CoT probe: `{args.nocot_npz}`")
    md.append("")
    md.append("## Per-bucket residual L2 norms at L*")
    md.append("")
    md.append("| Bucket | n | mean ||x|| | std | median |")
    md.append("|---|---:|---:|---:|---:|")
    for label, d in rows:
        if d["n"] == 0:
            md.append(f"| {label} | 0 | — | — | — |")
        else:
            md.append(f"| {label} | {d['n']} | "
                      f"{d['mean_norm']:.2f} | {d['std_norm']:.2f} | "
                      f"{d['median_norm']:.2f} |")
    md.append("")
    md.append("## Centroid distance (mean(CHECK) − mean(FOLD), L2 norm)")
    md.append("")
    md.append(f"- CoT:    {cot_centroid_dist:.2f}" if cot_centroid_dist else "- CoT: n/a")
    md.append(f"- nonCoT: {noc_centroid_dist:.2f}" if noc_centroid_dist else "- nonCoT: n/a")
    if (cot_centroid_dist and noc_centroid_dist
            and cot_centroid_dist > 1e-6):
        ratio = noc_centroid_dist / cot_centroid_dist
        md.append(f"- ratio (non-CoT / CoT): **{ratio:.2f}**")
        md.append("")
        md.append("Reading: ratio < 1 means CoT residuals are MORE separated "
                  "between CHECK and FOLD than non-CoT residuals (the verb "
                  "decision is more distinctly encoded under CoT). Ratio > 1 "
                  "means the opposite. A ratio near 1 means the geometry is "
                  "preserved — the §18a 'attenuated prior' is visible at the "
                  "*output discriminability* level (patches don't dominate) "
                  "but not at the *centroid separation* level.")
    md.append("")
    md.append("## Projection magnitudes onto each mode's own verb direction")
    md.append("")
    md.append("| Mode | bucket | mean |proj| |")
    md.append("|---|---|---:|")
    if summary["projection_magnitudes"]["cot_cc_mean_abs"] is not None:
        md.append(f"| CoT | CHECK | {summary['projection_magnitudes']['cot_cc_mean_abs']:.2f} |")
        md.append(f"| CoT | FOLD  | {summary['projection_magnitudes']['cot_lf_mean_abs']:.2f} |")
    if summary["projection_magnitudes"]["noc_cc_mean_abs"] is not None:
        md.append(f"| nonCoT | CHECK | {summary['projection_magnitudes']['noc_cc_mean_abs']:.2f} |")
        md.append(f"| nonCoT | FOLD  | {summary['projection_magnitudes']['noc_lf_mean_abs']:.2f} |")
    md.append("")
    md.append("## Reading guide")
    md.append("- **||x|| under non-CoT ≪ ||x|| under CoT**: the residual is "
              "compressed in non-CoT — small magnitude amplifies the relative "
              "weight of any prior bias.")
    md.append("- **Centroid distance smaller in non-CoT**: the verb decision "
              "is less distinctly encoded in non-CoT (patches struggle to "
              "express their content because the geometry is already biased "
              "toward the prior).")
    md.append("- **Projection magnitudes similar across modes**: the "
              "discriminating direction does similar work in both modes — "
              "the §18a finding is about the OUTPUT (action distribution "
              "softmax) not the residual representation itself.")
    with open(out_path, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
