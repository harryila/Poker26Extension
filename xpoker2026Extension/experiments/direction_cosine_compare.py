"""
Compare decision-direction probes from two different runs (e.g. CoT vs
non-CoT, or Llama-L14 vs Llama-L15).

Reads the cached weight vectors from `raw_residuals.npz` files produced
by `decision_direction_probe.py`, computes cosine similarity, and reports
whether the two directions are the same axis.

Usage::

    python -m experiments.direction_cosine_compare \\
        --probe-a results/direction_probe/llama8b_l14/raw_residuals.npz \\
        --probe-b results/direction_probe_nocot/llama8b_l14/raw_residuals.npz \\
        --label-a "Llama L=14 CoT" \\
        --label-b "Llama L=14 non-CoT" \\
        --out-md results/direction_cosine_compare/llama_cot_vs_nocot.md

Also computes:
  - Cross-projections: project A's residuals onto B's direction (and vice
    versa). If the directions are the same, A's CHECK residuals project
    POSITIVELY onto B's direction.
  - The "consistency cosine": cos(w_A, centroid_diff_B) and
    cos(w_B, centroid_diff_A). Tests whether B's CHECK-vs-FOLD
    discrimination axis is also a CHECK-vs-FOLD direction in A's data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Compare two cached direction probes by cosine similarity."
    )
    parser.add_argument("--probe-a", required=True,
                        help="Path to first raw_residuals.npz")
    parser.add_argument("--probe-b", required=True,
                        help="Path to second raw_residuals.npz")
    parser.add_argument("--label-a", default="probe A")
    parser.add_argument("--label-b", default="probe B")
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    npa = np.load(args.probe_a)
    npb = np.load(args.probe_b)

    wa = npa["weight_vec"].astype(np.float32)
    wb = npb["weight_vec"].astype(np.float32)
    if wa.shape != wb.shape:
        print(f"[abort] weight shapes differ: {wa.shape} vs {wb.shape}",
              file=sys.stderr)
        sys.exit(2)

    def _cos(x, y):
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        if nx < 1e-9 or ny < 1e-9:
            return float("nan")
        return float(x @ y / (nx * ny))

    cos_w = _cos(wa, wb)

    # Centroid-difference directions (CHECK − FOLD) inside each probe.
    cd_a = npa["centroid_diff"].astype(np.float32)
    cd_b = npb["centroid_diff"].astype(np.float32)
    cos_cd = _cos(cd_a, cd_b)
    cos_wa_cdb = _cos(wa, cd_b)
    cos_wb_cda = _cos(wb, cd_a)

    # Cross-projections: how well does each direction discriminate the
    # OTHER set's CHECK from FOLD?
    def _cross_project(X_cc, X_lf, w):
        nw = np.linalg.norm(w) + 1e-9
        return (X_cc.astype(np.float32) @ w / nw,
                X_lf.astype(np.float32) @ w / nw)

    proj_b_under_wa = _cross_project(npb["X_cc"], npb["X_lf"], wa)
    proj_a_under_wb = _cross_project(npa["X_cc"], npa["X_lf"], wb)
    cross_b_mean_diff = float(proj_b_under_wa[0].mean() - proj_b_under_wa[1].mean())
    cross_a_mean_diff = float(proj_a_under_wb[0].mean() - proj_a_under_wb[1].mean())
    # Sign-correct: positive = direction discriminates correctly (CHECK > FOLD).
    cross_b_signed_correct = cross_b_mean_diff > 0
    cross_a_signed_correct = cross_a_mean_diff > 0

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = []
    md.append(f"# Direction probe comparison: `{args.label_a}` vs `{args.label_b}`")
    md.append("")
    md.append(f"- Probe A: `{args.probe_a}` ({args.label_a})")
    md.append(f"- Probe B: `{args.probe_b}` ({args.label_b})")
    md.append(f"- Hidden dim: {wa.shape[0]}")
    md.append("")
    md.append("## Direct cosines (sign matters: positive = same axis, "
              "negative = opposite axis, near zero = orthogonal)")
    md.append("")
    md.append(f"- cos(w_A, w_B) = **{cos_w:+.4f}**  (probe weight vectors)")
    md.append(f"- cos(centroid_diff_A, centroid_diff_B) = **{cos_cd:+.4f}**  (mean-CHECK − mean-FOLD axes)")
    md.append(f"- cos(w_A, centroid_diff_B) = **{cos_wa_cdb:+.4f}**")
    md.append(f"- cos(w_B, centroid_diff_A) = **{cos_wb_cda:+.4f}**")
    md.append("")
    md.append("## Cross-projection (how well does each direction discriminate "
              "the OTHER set?)")
    md.append("")
    md.append("Mean projection difference (CHECK mean − FOLD mean) on the "
              "OTHER probe's residuals using THIS probe's weight vector. "
              "Positive = direction discriminates correctly.")
    md.append("")
    md.append(f"- B residuals projected onto w_A: **{cross_b_mean_diff:+.3f}** "
              f"({'✅ correct sign' if cross_b_signed_correct else '❌ WRONG sign'})")
    md.append(f"- A residuals projected onto w_B: **{cross_a_mean_diff:+.3f}** "
              f"({'✅ correct sign' if cross_a_signed_correct else '❌ WRONG sign'})")
    md.append("")
    md.append("## Reading guide")
    md.append("")
    md.append("- **cos(w_A, w_B) > 0.85**: probes recover the SAME direction in "
              "residual space. The verb-decision direction is shared between "
              "the two conditions (e.g. CoT vs non-CoT). This is the "
              "strongest possible 'shared circuit' evidence.")
    md.append("- **0.5 < cos(w_A, w_B) < 0.85**: directions are correlated but "
              "not identical. The decisions are encoded along similar "
              "(but not the same) axis. Could reflect overlapping circuits "
              "or shared features with condition-specific extras.")
    md.append("- **cos(w_A, w_B) < 0.5**: directions are largely independent. "
              "Each condition uses a different axis to encode the verb. "
              "This would be evidence AGAINST a shared circuit.")
    md.append("- **Cross-projection signs**: if both signs are correct (CHECK > "
              "FOLD when projected onto the other direction), the decision "
              "axes are functionally interchangeable. If signs are wrong, the "
              "directions are unrelated or anti-correlated.")
    with open(out_path, "w") as f:
        f.write("\n".join(md) + "\n")

    print(json.dumps({
        "cos_w": cos_w,
        "cos_cd": cos_cd,
        "cos_wa_cdb": cos_wa_cdb,
        "cos_wb_cda": cos_wb_cda,
        "cross_b_mean_diff": cross_b_mean_diff,
        "cross_a_mean_diff": cross_a_mean_diff,
    }, indent=2))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
