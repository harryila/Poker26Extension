"""
Paired significance test for the inference-time head/attention ablation cells.

WHY THIS EXISTS
---------------
`inference_head_ablation.py` writes a per-cell SUMMARY whose "behaviorally
necessary" verdict compares the ablated flip rate to the cell's OWN no-ablation
baseline. For a BLUNT intervention (whole-attention-block ablation) that test is
not enough: zeroing attention at *any* layer biases the model toward CHECK, so a
large flip-over-baseline can be generic disruption rather than layer-specific
necessity. The honest necessity question is whether a given layer's ablation
flips FOLD *significantly more than an early CONTROL layer's ablation*, on the
SAME decision records (a paired comparison).

This script answers that with the **McNemar exact test** (paired, exact binomial
on the discordant pairs) for every ablation cell against:
  (a) its own no-ablation baseline  — "does ablation flip beyond regen drift?"
  (b) a designated control-layer ablation — "is THIS layer more disruptive than
      removing attention at an unrelated early layer?" (the localization test)

CPU-only, stdlib-only (no numpy/torch). Reproducible from the committed
`*_rows.jsonl` files written by inference_head_ablation.py.

Usage::

    python -m experiments.necessity_significance \
        --glob 'results/inference_head_ablation/qwen8b_l*_recon_clean_legal_fold_*' \
        --control-layer 8 \
        --out results/inference_head_ablation/SIGNIFICANCE_qwen_clean_legal_fold.md
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import re
from math import comb

LAYER_RE = re.compile(r"_l(\d+)_")


def _flip_map(path: str) -> dict[tuple, bool]:
    """Map (seed, decision_idx) -> flipped (replay verb != FOLD).

    The recorded bucket for these cells is always a FOLD bucket
    (illegal_fold / clean_legal_fold), so 'flip' == the regenerated verb is no
    longer FOLD.
    """
    out: dict[tuple, bool] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("seed"), r.get("decision_idx"))
            out[key] = (r.get("replay_verb") != "FOLD")
    return out


def _parse_fail_rate(path: str) -> float:
    n = 0
    fails = 0
    if not os.path.exists(path):
        return float("nan")
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            if not r.get("replay_parseable_json", True):
                fails += 1
    return fails / n if n else float("nan")


def mcnemar_exact(a: dict[tuple, bool], b: dict[tuple, bool]) -> tuple[int, int, int, float]:
    """Paired McNemar exact two-sided test over the shared keys of a, b.

    Returns (n_paired, b_a_only, b_b_only, p_value) where b_a_only counts
    records flipped under condition a but not b, and vice versa. The exact
    p-value is the two-sided binomial tail at p=0.5 on the discordant pairs.
    """
    keys = set(a) & set(b)
    a_only = sum(1 for k in keys if a[k] and not b[k])
    b_only = sum(1 for k in keys if b[k] and not a[k])
    n = a_only + b_only
    if n == 0:
        return len(keys), a_only, b_only, 1.0
    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return len(keys), a_only, b_only, p


def _conditions(cell_dir: str) -> list[str]:
    """All non-baseline condition names present in the cell (from summary.json,
    falling back to *_rows.jsonl on disk)."""
    sj = os.path.join(cell_dir, "summary.json")
    if os.path.exists(sj):
        s = json.load(open(sj))
        conds = [c for c in s.get("conditions", {}) if c != "baseline"]
        if conds:
            return conds
    out = []
    for f in globmod.glob(os.path.join(cell_dir, "*_rows.jsonl")):
        name = os.path.basename(f)[: -len("_rows.jsonl")]
        if name != "baseline":
            out.append(name)
    return out


def _layer_of(cell_dir: str) -> int | None:
    m = LAYER_RE.search(os.path.basename(cell_dir))
    return int(m.group(1)) if m else None


def _flip_rate(fm: dict[tuple, bool]) -> float:
    return (sum(1 for v in fm.values() if v) / len(fm)) if fm else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--glob", required=True,
        help="Glob of ablation cell directories (each has baseline_rows.jsonl "
             "+ one ablation-condition rows file).",
    )
    ap.add_argument(
        "--control-layer", type=int, default=None,
        help="Layer whose whole-attention ablation cell is the cross-layer "
             "CONTROL (e.g. 8, the Qwen design). Each cell's ablation is "
             "McNemar-compared to this layer's ablation on shared records.",
    )
    ap.add_argument(
        "--within-cell-control", default="control",
        help="When no cross-layer control applies, compare each necessity "
             "condition to this within-cell condition (head-control, default "
             "'control' = heads [0,1,2]; the Llama/Ministral design).",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cells = sorted(
        (d for d in globmod.glob(args.glob) if os.path.isdir(d)),
        key=lambda d: (_layer_of(d) if _layer_of(d) is not None else 1 << 30,
                       "topk" in d),
    )
    if not cells:
        raise SystemExit(f"no cell dirs matched {args.glob!r}")

    # Locate a cross-layer control map (Qwen whole-attention design), if any.
    control_abl: dict[tuple, bool] | None = None
    control_label = None
    if args.control_layer is not None:
        for d in cells:
            if _layer_of(d) == args.control_layer:
                for cond in _conditions(d):
                    if cond.startswith("whole"):
                        control_abl = _flip_map(os.path.join(d, f"{cond}_rows.jsonl"))
                        control_label = f"L{args.control_layer} {cond}"
                        break

    # One output row per (cell, necessity condition).
    records = []
    for d in cells:
        layer = _layer_of(d)
        base = _flip_map(os.path.join(d, "baseline_rows.jsonl"))
        conds = _conditions(d)
        for cond in conds:
            # The cross-layer control cell's own whole-attn condition is the
            # control itself; the within-cell head-control row is informative
            # (control-vs-baseline) so keep it but it has no further control.
            abl_path = os.path.join(d, f"{cond}_rows.jsonl")
            abl = _flip_map(abl_path)
            if not abl:
                continue
            scope = "whole-attn" if cond.startswith("whole") else cond
            rec = {
                "dir": os.path.basename(d), "path": d,
                "layer": layer, "cond": cond,
                "scope": scope, "n": len(abl),
                "base_flip": _flip_rate(base), "abl_flip": _flip_rate(abl),
                "abl_map": abl, "base_map": base,
                "parse_fail": _parse_fail_rate(abl_path),
            }
            records.append(rec)

    # Compute McNemar vs baseline and vs the appropriate control per row.
    for rec in records:
        _, a_only, b_only, p_base = mcnemar_exact(rec["abl_map"], rec["base_map"])
        rec["vs_base"] = (a_only, b_only, p_base)
        ctrl_map = None
        ctrl_kind = None
        is_cross_ctrl = (control_abl is not None
                         and rec["layer"] == args.control_layer
                         and rec["cond"].startswith("whole"))
        if control_abl is not None and not is_cross_ctrl and rec["cond"].startswith(("whole", "topk")):
            ctrl_map, ctrl_kind = control_abl, f"xlayer:{control_label}"
        elif rec["cond"] != args.within_cell_control:
            # within-cell head control (Llama/Ministral design)
            wcp = os.path.join(rec["path"], f"{args.within_cell_control}_rows.jsonl")
            if os.path.exists(wcp):
                ctrl_map, ctrl_kind = _flip_map(wcp), f"heads:{args.within_cell_control}"
        if ctrl_map is not None:
            n, a_only, b_only, p_ctrl = mcnemar_exact(rec["abl_map"], ctrl_map)
            rec["vs_ctrl"] = (n, a_only, b_only, p_ctrl, ctrl_kind)
        else:
            rec["vs_ctrl"] = None

    # Render markdown.
    md: list[str] = []
    md.append("# Necessity ablation — paired significance (McNemar exact)")
    md.append("")
    md.append(f"- Cells: `{args.glob}`")
    md.append(f"- n_cells: {len(records)}")
    if control_label:
        md.append(f"- **Control for localization test: {control_label}**")
    md.append("- `flip` = regenerated verb is no longer FOLD (recorded bucket "
              "is a FOLD bucket). Tests are paired over shared "
              "`(seed, decision_idx)` records.")
    md.append("- `vs baseline`: does ablation flip beyond no-ablation regen "
              "drift? `vs control`: is this condition more disruptive than the "
              "control ablation (cross-layer or within-cell head control) — the "
              "localization test, direction-aware.")
    md.append("")
    md.append("| Layer | scope | n | base flip | abl flip | net pp | "
              "McNemar vs baseline | McNemar vs control | parse_fail |")
    md.append("|---|---|---:|---:|---:|---:|---|---|---:|")
    for rec in records:
        net = (rec["abl_flip"] - rec["base_flip"]) * 100
        a_only, b_only, p_base = rec["vs_base"]
        vb = f"{a_only}/{b_only} disc, p={p_base:.2e}"
        if rec["vs_ctrl"] is None:
            vc = "— (is control)"
        else:
            _, a2, b2, p_ctrl, kind = rec["vs_ctrl"]
            # Direction matters: necessity requires the CONDITION to flip MORE
            # than control (a2 > b2). If control flips more (b2 > a2), a small
            # p-value means the OPPOSITE of necessity.
            if p_ctrl >= 0.10:
                sig = "NS"
            elif a2 > b2:
                sig = "✓ sig necessary" if p_ctrl < 0.05 else "~marg"
            else:
                sig = "✗ REVERSED (control flips MORE)" if p_ctrl < 0.05 \
                    else "~marg-reversed"
            vc = f"{a2}/{b2} disc, p={p_ctrl:.2e} ({sig}) [{kind}]"
        md.append(
            f"| L{rec['layer']} | {rec['scope']} | {rec['n']} | "
            f"{rec['base_flip']*100:.1f}% | {rec['abl_flip']*100:.1f}% | "
            f"{net:+.1f} | {vb} | {vc} | {rec['parse_fail']*100:.1f}% |"
        )
    md.append("")
    md.append("## Reading")
    md.append("- A genuine, *localized* necessity result = `✓ sig necessary` "
              "(p<0.05 AND the condition flips MORE than control) with low "
              "`parse_fail`. The condition's heads/attention are necessary for "
              "FOLD over-and-above the generic disruption captured by control.")
    md.append("- `✗ REVERSED` = p<0.05 but the CONTROL flips more than the "
              "condition — the *opposite* of necessity (the named heads matter "
              "less than an arbitrary control; e.g. the Ministral §9 null).")
    md.append("- `McNemar vs baseline` significant but `vs control` NS = generic "
              "disruption, not layer/head-specific necessity.")
    md.append("- discordant counts `a/b`: `a` = records flipped by this "
              "condition but not the comparison; `b` = the reverse. The exact "
              "two-sided binomial p-value is computed on `a+b`. The comparison "
              "kind is in brackets: `xlayer:` = cross-layer control ablation; "
              "`heads:` = within-cell head control.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
