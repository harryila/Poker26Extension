"""
decompose_pot_odds_by_fallback.py
==================================

Cross-join `analysis.pot_odds_analysis` per-decision CSV output with the
matching enriched JSONL log to bucket each decision by *why* the fallback
fired (or didn't), then report mean EV-truth-regret + frac_ev_truth_optimal
per bucket.

WHY THIS EXISTS
---------------
Pot-odds aggregate stats (in `*.summary.json`) treat every decision the
same. But the recategorize_action_metadata.py dig-in (updates.md §11)
showed that a substantial fraction of CoT decisions on small-tier 8B models
are "illegal-action attempts" — the model emits {"action": "FOLD"} into a
free-check spot, and our `_fallback_action` rescues it to CHECK_OR_CALL.

The natural follow-up question is:
    Are these "illegal FOLD" decisions actually EV-better or EV-worse than
    the model's "clean" (no-fallback) decisions?

This script answers it by bucketing each decision into one of:

    clean                  : the model's chosen action was used directly
                             (no fallback)
    illegal_fold_rescued   : raw_response said FOLD but FOLD was not legal,
                             so the fallback played CHECK_OR_CALL or similar
    other_fallback         : any other fallback path (JSON parse failure,
                             unrecognized alias, etc.)

For each bucket it reports mean EV-truth-regret (chips lost vs the EV-best
action under the true equity) and the fraction of decisions that were
EV-optimal.

USAGE
-----
    python -m analysis.decompose_pot_odds_by_fallback \\
        --pot-odds-csv results/tier1a_small_cot/pot_odds/cot_ministral8b_t0_s42.csv \\
        --enriched-log logs/cot_ministral8b_t0_s42_informative_v2_enriched.jsonl.gz \\
        --label "Ministral CoT t=0 s42"

Multiple --pot-odds-csv / --enriched-log pairs can be passed; each prints a
table. Optional --json-out writes the structured result for downstream use.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import sys
from typing import Iterable


def _open_log(path: str) -> io.TextIOBase:
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _decision_key(rec: dict) -> tuple[str, int] | None:
    if "hand_id" not in rec or "decision_idx" not in rec:
        return None
    return (rec["hand_id"], int(rec["decision_idx"]))


def _classify(record: dict) -> str:
    """Bucket a decision record using the action_metadata we already have.

    Prefers the new diagnostic flags (parse_v2 fields) when present, falls
    back to the heuristic "fallback_used + raw_response contains FOLD +
    FOLD not in legal_actions" when reading old logs.
    """
    am = record.get("action_metadata") or {}
    if not am:
        return "no_action"
    if not am.get("fallback_used"):
        return "clean"

    if "action_recognized" in am and "action_legal_in_context" in am:
        # New-format log -- definitive.
        if am.get("action_recognized") and not am.get("action_legal_in_context"):
            attempted = (am.get("raw_response") or "").upper()
            if '"ACTION": "FOLD"' in attempted or '"ACTION":"FOLD"' in attempted:
                return "illegal_fold_rescued"
            return "illegal_other_rescued"
        return "other_fallback"

    raw = (am.get("raw_response") or "").upper()
    legal = record.get("legal_actions") or []
    if "FOLD" not in legal and (
        '"ACTION": "FOLD"' in raw or '"ACTION":"FOLD"' in raw
    ):
        return "illegal_fold_rescued"
    return "other_fallback"


def decompose_cell(pot_odds_csv: str, enriched_log: str) -> dict:
    """Run the cross-join for one cell. Returns a dict of bucket -> stats."""
    pot_rows: dict[tuple[str, int], dict] = {}
    with open(pot_odds_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["hand_id"], int(row["decision_idx"]))
            pot_rows[key] = row

    classifications: dict[tuple[str, int], str] = {}
    with _open_log(enriched_log) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") in ("run_config", "hand_summary"):
                continue
            key = _decision_key(rec)
            if key is None:
                continue
            classifications[key] = _classify(rec)

    buckets: dict[str, list[tuple[float, bool]]] = {}
    for key, po in pot_rows.items():
        cat = classifications.get(key, "no_action")
        regret_str = po.get("ev_truth_regret", "")
        optimal_str = po.get("ev_truth_optimal", "")
        if not regret_str:
            continue
        try:
            regret = float(regret_str)
        except ValueError:
            continue
        optimal = optimal_str.strip().lower() == "true"
        buckets.setdefault(cat, []).append((regret, optimal))

    out: dict[str, dict] = {}
    for cat in sorted(buckets):
        rows = buckets[cat]
        n = len(rows)
        out[cat] = {
            "n": n,
            "mean_ev_truth_regret": sum(r for r, _ in rows) / n,
            "frac_ev_truth_optimal": sum(1 for _, o in rows if o) / n,
        }

    out["_total"] = {
        "n_decisions_in_pot_odds_csv": len(pot_rows),
        "n_decisions_classified": sum(b["n"] for cat, b in out.items() if cat != "_total"),
    }
    return out


def _print_table(label: str, result: dict) -> None:
    print()
    print(f"=== {label}  (n={result.get('_total', {}).get('n_decisions_in_pot_odds_csv', '?')}) ===")
    print(f"{'category':30s} {'n':>5s} {'mean_ev_regret':>16s} {'frac_optimal':>14s}")
    print("-" * 75)
    for cat, stats in result.items():
        if cat == "_total":
            continue
        print(f"{cat:30s} {stats['n']:>5d} {stats['mean_ev_truth_regret']:>16.3f} "
              f"{stats['frac_ev_truth_optimal']*100:>13.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pot-odds-csv", action="append", required=True,
                    help="Path to a per-decision pot-odds CSV (repeat for multiple cells).")
    ap.add_argument("--enriched-log", action="append", required=True,
                    help="Path to the matching enriched JSONL log (.jsonl or .jsonl.gz). "
                         "Must be passed in the same order as --pot-odds-csv.")
    ap.add_argument("--label", action="append", default=None,
                    help="Display label for each cell (optional; defaults to CSV basename).")
    ap.add_argument("--json-out", default=None,
                    help="Optional path for a structured JSON dump of all results.")
    args = ap.parse_args()

    if len(args.pot_odds_csv) != len(args.enriched_log):
        sys.exit("--pot-odds-csv and --enriched-log must have the same count")

    labels = args.label or [None] * len(args.pot_odds_csv)
    if len(labels) != len(args.pot_odds_csv):
        sys.exit("--label must be passed same number of times as --pot-odds-csv (or omitted entirely)")

    out: dict[str, dict] = {}
    for csv_path, log_path, label in zip(args.pot_odds_csv, args.enriched_log, labels):
        if label is None:
            label = os.path.basename(csv_path).replace(".csv", "")
        result = decompose_cell(csv_path, log_path)
        _print_table(label, result)
        out[label] = result

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[decompose] wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
