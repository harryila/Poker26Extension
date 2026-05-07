"""
Cross-model logit-lens deep analysis: per-cell × per-emitted-action layer
trajectory.

Generalizes the §13 Ministral-s42-only finding ("clean_CHECK_OR_CALL
late-revises; clean_LEGAL_FOLD locks in early; illegal_FOLD locks in even
earlier") to all 18 cells of the Tier 1A.small CoT + logit-lens grid.

What this produces
------------------
For every (model × seed × temp) cell, splits the "clean" bucket of
`analyze_logit_lens_by_failure_mode.py` further by EMITTED ACTION:

  - clean_check_or_call : legal CHECK_OR_CALL chosen
  - clean_legal_fold    : legal FOLD chosen
  - clean_bet_or_raise  : legal BET_OR_RAISE chosen
  - illegal_fold        : illegal FOLD chosen (env rescued to CHECK_OR_CALL)
  - illegal_other       : non-FOLD illegal action

For each (cell × bucket), aggregates per-layer action mix at the action-verb
position AND computes the action-group crystallization layer.

Output:

  - results/tier1a_small_cot_logitlens/CROSS_CELL_DETAILED.md
      One section per cell (18 total), with a per-bucket per-layer table.
      Plus a cross-cell summary at the top.

  - results/tier1a_small_cot_logitlens/cross_cell_detailed.json
      Machine-readable per-cell, per-bucket aggregates.

CPU only. Reuses existing logit-lens sidecars + enriched logs. Runs in
~2 min on a Mac M-series.

Usage::

    python -m analysis.cross_model_clean_split \
        --logs-dir logs/ \
        --out-dir  results/tier1a_small_cot_logitlens/

Pure analysis: no GPU forward passes.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import io
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.recategorize_action_metadata import _reparse_one  # noqa: E402
from analysis.analyze_logit_lens_by_failure_mode import (  # noqa: E402
    _find_action_position,
    _per_layer_action_at_action_position,
    _ensure_layer_buffers,
    _crystallization_layer,
    _empty_bucket,
    _add_record_to_bucket,
    _summarize_bucket,
    _iter_jsonl,
    _open_log,
)


# ---------------------------------------------------------------------------
# Bucketing — extends the 5-way classify_decision into a 5-way emitted-action
# split (FOLD into legal/illegal, CHECK_OR_CALL, BET_OR_RAISE, illegal_other,
# alias_unrecognized, json_failure)
# ---------------------------------------------------------------------------

EMITTED_BUCKETS = (
    "clean_check_or_call",
    "clean_legal_fold",
    "clean_bet_or_raise",
    "illegal_fold",
    "illegal_other",
    "alias_unrecognized",
    "json_failure",
)


def classify_emitted(rec: dict) -> str:
    """Same logic as experiments/causal_patching.py::classify_decision."""
    info = _reparse_one(rec)
    if not info["json_parsed"]:
        return "json_failure"
    if not info["recognized"]:
        return "alias_unrecognized"
    attempted = (info["attempted_action"] or "").upper()
    if info["legal"]:
        if attempted == "FOLD":
            return "clean_legal_fold"
        if attempted == "CHECK_OR_CALL":
            return "clean_check_or_call"
        if attempted == "BET_OR_RAISE":
            return "clean_bet_or_raise"
        return "alias_unrecognized"
    if attempted == "FOLD":
        return "illegal_fold"
    return "illegal_other"


# ---------------------------------------------------------------------------
# Cell discovery + processing
# ---------------------------------------------------------------------------

CELL_RE = re.compile(
    r"^cot_(?P<model>llama8b|qwen8b|ministral8b)_t(?P<temp>\d+)_s(?P<seed>\d+)_"
    r"(?P<opp>[A-Za-z0-9_]+?)_logitlens_logit_lens\.jsonl(?:\.gz)?$"
)


def discover_cells(logs_dir: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(logs_dir, "cot_*_logitlens_logit_lens.jsonl*"))):
        base = os.path.basename(p)
        m = CELL_RE.match(base)
        if not m:
            continue
        # Find matching enriched log (same prefix, _enriched.jsonl[.gz]).
        prefix = base.replace("_logit_lens.jsonl", "_enriched.jsonl").replace(".gz", "")
        enr_candidates = [
            os.path.join(logs_dir, prefix),
            os.path.join(logs_dir, prefix + ".gz"),
        ]
        enr = next((e for e in enr_candidates if os.path.exists(e)), None)
        if enr is None:
            continue
        out.append({
            "model": m.group("model"),
            "seed": int(m.group("seed")),
            "temp": "0.0" if m.group("temp") == "0" else "0.2",
            "sidecar": p,
            "enriched": enr,
            "label": f"{m.group('model')}_t{m.group('temp')}_s{m.group('seed')}",
        })
    return out


def process_cell(cell: dict) -> dict:
    """Returns {cell_meta, buckets: {bucket_name: summarized}}.

    Joins enriched-log decisions to sidecar records by (hand_id, decision_idx),
    classifies each into one of EMITTED_BUCKETS, accumulates per-bucket
    aggregates.
    """
    # Build sidecar index
    sidecar_idx: dict[tuple, dict] = {}
    for rec in _iter_jsonl(cell["sidecar"]):
        hid = rec.get("hand_id")
        didx = rec.get("decision_idx")
        if hid is None or didx is None:
            continue
        sidecar_idx[(hid, int(didx))] = rec

    buckets = {b: _empty_bucket() for b in EMITTED_BUCKETS}
    stats = {
        "enriched_decisions_seen": 0,
        "sidecar_records": len(sidecar_idx),
        "joined": 0,
        "unmatched": 0,
    }

    for rec in _iter_jsonl(cell["enriched"]):
        if rec.get("type") not in (None, "decision"):
            continue
        if "action_metadata" not in rec or rec.get("action_metadata") is None:
            continue
        stats["enriched_decisions_seen"] += 1
        bname = classify_emitted(rec)
        bucket = buckets[bname]
        bucket["n"] += 1

        key = (rec.get("hand_id"), int(rec.get("decision_idx", -1)))
        lens = sidecar_idx.get(key)
        if lens is None:
            stats["unmatched"] += 1
            continue
        stats["joined"] += 1
        _add_record_to_bucket(bucket, lens)

    return {
        "label": cell["label"],
        "model": cell["model"],
        "seed": cell["seed"],
        "temp": cell["temp"],
        "stats": stats,
        "buckets": {b: _summarize_bucket(buckets[b]) for b in EMITTED_BUCKETS},
    }


# ---------------------------------------------------------------------------
# Cross-cell aggregate (per-model summary)
# ---------------------------------------------------------------------------


def aggregate_per_model(cell_results: list[dict]) -> dict:
    """For each model, compute mean crystallization layer per bucket across
    its 6 cells (weighted by bucket n)."""
    by_model = collections.defaultdict(
        lambda: {b: {"crys_sum": 0.0, "crys_n": 0, "n_decisions": 0} for b in EMITTED_BUCKETS}
    )
    for cell in cell_results:
        m = cell["model"]
        for b, summ in cell["buckets"].items():
            cl = (summ.get("crystallization_layer") or {})
            n = summ.get("n_decisions", 0)
            mean_cl = cl.get("mean")
            if mean_cl is not None and n:
                by_model[m][b]["crys_sum"] += mean_cl * n
                by_model[m][b]["crys_n"] += n
            by_model[m][b]["n_decisions"] += n
    out = {}
    for m, bs in by_model.items():
        out[m] = {}
        for b, d in bs.items():
            out[m][b] = {
                "n_decisions": d["n_decisions"],
                "weighted_mean_crystallization": (
                    d["crys_sum"] / d["crys_n"] if d["crys_n"] else None
                ),
            }
    return out


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------


_BUCKET_DISPLAY = {
    "clean_check_or_call": "clean CHECK_OR_CALL",
    "clean_legal_fold":    "clean LEGAL FOLD",
    "clean_bet_or_raise":  "clean BET_OR_RAISE",
    "illegal_fold":        "illegal FOLD (rescued)",
    "illegal_other":       "illegal other",
    "alias_unrecognized":  "alias unrecognized",
    "json_failure":        "json failure",
}


def _fmt_layer_mix(frac: dict, keys: tuple = ("FOLD", "CHECK", "CALL", "BET", "RAISE", "OTHER")) -> str:
    parts = []
    for k in keys:
        v = frac.get(k, 0.0)
        if v >= 0.05:
            parts.append(f"{k}{v:.2f}")
    return " ".join(parts) if parts else "·"


def write_markdown(cell_results: list[dict], per_model: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lines: list[str] = []
    lines.append("# Cross-cell detailed logit-lens analysis (per emitted action)")
    lines.append("")
    lines.append(
        "Generalizes the §13 Ministral-s42-only finding to ALL 18 cells. For each\n"
        "cell × emitted action bucket, reports the action-group crystallization\n"
        "layer at the action-verb position. The per-bucket per-layer mix tables\n"
        "for each cell follow.\n"
    )

    # 1. Per-model aggregate summary
    lines.append("## Per-model aggregate (weighted-mean crystallization layer)\n")
    lines.append(
        "Weighted by per-bucket n across that model's 6 cells. Only buckets\n"
        "with non-zero records contribute. Lower = the model 'decided' earlier;\n"
        "higher = late-layer revision.\n"
    )
    lines.append("| Model | clean CHECK_OR_CALL | clean LEGAL FOLD | clean BET_OR_RAISE | illegal FOLD |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in ("llama8b", "ministral8b", "qwen8b"):
        if m not in per_model:
            continue
        row = [m]
        for b in ("clean_check_or_call", "clean_legal_fold", "clean_bet_or_raise", "illegal_fold"):
            d = per_model[m][b]
            cl = d.get("weighted_mean_crystallization")
            n = d.get("n_decisions", 0)
            cell_str = f"{cl:.1f} (n={n})" if cl is not None else f"— (n={n})"
            row.append(cell_str)
        lines.append("| `" + row[0] + "` | " + " | ".join(row[1:]) + " |")
    lines.append("")
    lines.append(
        "**Interpretation guide.** If illegal-FOLD crystallizes EARLIER than\n"
        "clean CHECK_OR_CALL by 2+ layers, that's the §13 'baseline FOLD pull\n"
        "with late-layer revision in CHECK decisions only' pattern, generalized\n"
        "across this model's 6 cells.\n"
    )

    # 2. Per-cell summary table
    lines.append("## Per-cell summary (action-group crystallization layer)\n")
    lines.append(
        "| Cell | n CHECK_OR_CALL | crys CHECK_OR_CALL | n LEGAL_FOLD | crys LEGAL_FOLD | n illegal_FOLD | crys illegal_FOLD | Δ (illegal − clean) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cell in cell_results:
        coc = cell["buckets"]["clean_check_or_call"]
        lf  = cell["buckets"]["clean_legal_fold"]
        ifd = cell["buckets"]["illegal_fold"]
        coc_cl = (coc.get("crystallization_layer") or {}).get("mean")
        lf_cl  = (lf.get("crystallization_layer") or {}).get("mean")
        ifd_cl = (ifd.get("crystallization_layer") or {}).get("mean")
        delta = (ifd_cl - coc_cl) if (coc_cl is not None and ifd_cl is not None) else None
        lines.append(
            f"| `{cell['label']}` "
            f"| {coc['n_decisions']} | {coc_cl:.1f} "
            f"| {lf['n_decisions']} | {(f'{lf_cl:.1f}' if lf_cl else '—')} "
            f"| {ifd['n_decisions']} | {(f'{ifd_cl:.1f}' if ifd_cl else '—')} "
            f"| {(f'{delta:+.1f}' if delta is not None else '—')} |"
        )
    lines.append("")

    # 3. Per-cell deep tables (compact: only key buckets, only late layers)
    lines.append("## Per-cell late-layer trajectory (final 12 layers)\n")
    lines.append(
        "Compact view: only the final 12 layers, only the buckets with n≥3.\n"
        "Format per cell: per-layer mapped action group at the verb position,\n"
        "showing fractions ≥0.05.\n"
    )
    for cell in cell_results:
        lines.append(f"### `{cell['label']}`\n")
        # Number of layers (use first non-empty bucket)
        n_layers = 0
        for b in EMITTED_BUCKETS:
            s = cell["buckets"][b]
            if s["num_layers"] > 0:
                n_layers = s["num_layers"]
                break
        if n_layers == 0:
            lines.append("_no layer data_\n")
            continue
        lines.append(f"_{n_layers} layers; showing {min(12, n_layers)} latest (L={max(0, n_layers - 12)}–{n_layers - 1})_\n")
        for bname in ("clean_check_or_call", "clean_legal_fold", "illegal_fold"):
            s = cell["buckets"][bname]
            if s["n_decisions"] < 3:
                continue
            lines.append(f"**{_BUCKET_DISPLAY[bname]}** (n={s['n_decisions']})")
            cl = (s.get("crystallization_layer") or {})
            lines.append(f"  - crystallization layer: mean={cl.get('mean','—')}, median={cl.get('median','—')}, range=[{cl.get('min','—')}, {cl.get('max','—')}]")
            # Late-layer trajectory
            lines.append("  - late-layer mix (action-group fractions at verb pos):")
            for L in range(max(0, n_layers - 12), n_layers):
                if L < len(s["per_layer_action_fraction"]):
                    mix = _fmt_layer_mix(s["per_layer_action_fraction"][L])
                    lines.append(f"    - L{L:>2}: {mix}")
            lines.append("")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model clean-bucket-split logit-lens analysis (CPU)."
    )
    parser.add_argument("--logs-dir", default="logs",
                        help="Directory containing cot_*_logitlens_*_enriched/_logit_lens.jsonl[.gz]")
    parser.add_argument("--out-dir", default="results/tier1a_small_cot_logitlens",
                        help="Output directory; writes CROSS_CELL_DETAILED.md and cross_cell_detailed.json")
    args = parser.parse_args()

    cells = discover_cells(args.logs_dir)
    print(f"[discover] {len(cells)} cells found in {args.logs_dir}")
    if not cells:
        sys.exit(1)

    cell_results = []
    for cell in cells:
        print(f"  [process] {cell['label']} ...")
        cell_results.append(process_cell(cell))

    per_model = aggregate_per_model(cell_results)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "cross_cell_detailed.json")
    with open(json_path, "w") as f:
        json.dump({
            "cells": cell_results,
            "per_model": per_model,
        }, f, indent=2, default=str)
    print(f"[wrote] {json_path}")

    md_path = os.path.join(args.out_dir, "CROSS_CELL_DETAILED.md")
    write_markdown(cell_results, per_model, md_path)
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()
