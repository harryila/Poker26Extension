"""
recategorize_action_metadata.py
================================

Post-hoc re-attribution of action-metadata "parse failures" using the
alias-normalization fix (json_utils.ACTION_ALIASES) and a finer-grained
breakdown of *why* a decision fell into fallback.

WHY THIS EXISTS
---------------
The tier1a_small CoT grid (results/tier1a_small_cot/COMPARISON.md) reports
some cells with parse_success_rate as low as 21.7% (Ministral 8B s42, t=0).
A dig-in showed the *raw* JSON parse rate was actually ~100% for those
cells. The reported number was conflating three orthogonal failure modes:

  1. action_json_parsed=False   (no JSON object recovered)
  2. action_recognized=False    (action string not in canonical map)
  3. action_legal_in_context=False  (action was recognized but not legal)

Two real issues were hiding in (2) and (3):

  * Ministral 8B emitted {"action": "CHECK"} (a colloquial alias the
    action map didn't accept) — fixed in code by ACTION_ALIASES
    (poker_env/agents/json_utils.py).
  * Multiple 8B models under CoT attempted illegal FOLDs on s42
    (FOLD chosen when bet_to_call=0 made CHECK_OR_CALL the free option).
    This is a model-behavior finding, NOT a parser bug, and should be
    reported as such.

WHAT THIS SCRIPT DOES
---------------------
For each decision record it RE-runs the action selection logic that lives
in HFAgent.act_with_metadata, but pure-functionally on the logged
raw_response and legal_actions. It writes a per-cell breakdown:

  total_decisions
  cot_decisions               (records with action_metadata.cot_reasoning_present
                               OR a parse_cot_response that returns reasoning)
  --- ORIGINAL (as-logged) ---
  parse_success_orig
  fallback_used_orig
  --- RE-ATTRIBUTED ---
  parse_success_v2            (json_parsed AND recognized AND legal)
  recovered_by_alias_norm     (was fallback in original; now succeeds purely
                               because the alias map normalizes CHECK -> ...)
  json_parse_failures_v2      (no JSON, or no "action" key, or wrong type)
  alias_unrecognized_v2       (JSON OK; action not in canonical map even
                               after ACTION_ALIASES; unfixable in code)
  illegal_in_context_v2       (recognized action not in legal_actions;
                               model behavior — typically illegal FOLD)
  illegal_in_context_breakdown (counts by (attempted_action, missing_from_legal))

USAGE
-----
    python -m analysis.recategorize_action_metadata \\
        --logs-glob "logs/cot_*_enriched.jsonl.gz" \\
        --baseline-glob "logs/scaled_*_enriched.jsonl.gz" \\
        --json-out  results/tier1a_small_cot/comparison_v2.json \\
        --md-out    results/tier1a_small_cot/COMPARISON_v2.md

Both .jsonl and .jsonl.gz are read transparently.

If --baseline-glob is given, the markdown report includes a non-CoT vs
CoT comparison column for parse_success_v2.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import io
import json
import os
import sys
from typing import Iterable

# Make sibling poker_env importable when run from xpoker2026Extension/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from poker_env.agents.json_utils import (
    extract_json,
    normalize_action_str,
    parse_cot_response,
)


# Canonical action names (string form). We don't import the Action enum
# because we only need the string names from logs.
CANONICAL_ACTIONS = {"FOLD", "CHECK_OR_CALL", "BET_OR_RAISE"}


# ---------------------------------------------------------------------------
# Per-record re-parsing
# ---------------------------------------------------------------------------

def _reparse_one(rec: dict) -> dict:
    """Re-derive (json_parsed, recognized, legal, attempted_action) for one
    decision record using the new alias-aware logic.

    Returns a dict with keys:
        json_parsed:       bool
        recognized:        bool
        legal:             bool
        attempted_action:  str | None        # post-alias, what model meant
        was_cot:           bool              # CoT prompt template?
        legal_actions:     list[str]
        original_parse_success: bool
        original_fallback_used: bool
    """
    am = rec.get("action_metadata") or {}
    raw = am.get("raw_response") or ""
    legal_actions = list(rec.get("legal_actions") or [])

    template_id = am.get("prompt_template_id") or rec.get("prompt_template_id") or ""
    was_cot = "cot" in template_id.lower()

    # Mirror HFAgent.act_with_metadata.
    if was_cot:
        _reasoning, parsed = parse_cot_response(raw)
    else:
        parsed = extract_json(raw)

    json_parsed = bool(parsed and "action" in parsed and isinstance(parsed["action"], str))
    attempted: str | None = None
    recognized = False
    legal = False

    if json_parsed:
        attempted = normalize_action_str(parsed["action"])
        recognized = attempted in CANONICAL_ACTIONS
        legal = recognized and (attempted in legal_actions)

    return {
        "json_parsed": json_parsed,
        "recognized": recognized,
        "legal": legal,
        "attempted_action": attempted,
        "was_cot": was_cot,
        "legal_actions": legal_actions,
        "original_parse_success": bool(am.get("parse_success", False)),
        "original_fallback_used": bool(am.get("fallback_used", False)),
    }


# ---------------------------------------------------------------------------
# Per-cell aggregation
# ---------------------------------------------------------------------------

def _empty_cell() -> dict:
    return {
        "total_decisions": 0,
        "cot_decisions": 0,
        "parse_success_orig": 0,
        "fallback_used_orig": 0,
        "parse_success_v2": 0,
        "recovered_by_alias_norm": 0,
        "json_parse_failures_v2": 0,
        "alias_unrecognized_v2": 0,
        "illegal_in_context_v2": 0,
        "illegal_in_context_breakdown": collections.Counter(),
        "alias_unrecognized_examples": collections.Counter(),
    }


def _is_decision_record(rec: dict) -> bool:
    return rec.get("type") in (None, "decision") and "action_metadata" in rec


def _open_log(path: str) -> io.TextIOBase:
    """Open a log file transparently, handling .jsonl and .jsonl.gz."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _aggregate_log(path: str, max_examples: int = 5) -> dict:
    cell = _empty_cell()
    cell["log_path"] = path
    cell["log_basename"] = os.path.basename(path).replace(".jsonl.gz", ".jsonl")

    with _open_log(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _is_decision_record(rec):
                continue

            cell["total_decisions"] += 1
            r = _reparse_one(rec)
            if r["was_cot"]:
                cell["cot_decisions"] += 1
            if r["original_parse_success"]:
                cell["parse_success_orig"] += 1
            if r["original_fallback_used"]:
                cell["fallback_used_orig"] += 1

            ok_v2 = r["json_parsed"] and r["recognized"] and r["legal"]
            if ok_v2:
                cell["parse_success_v2"] += 1
                # Did re-parsing recover something the original logged as a
                # fallback? That can only happen when alias normalization
                # rescues an unrecognized action.
                if r["original_fallback_used"]:
                    cell["recovered_by_alias_norm"] += 1
            elif not r["json_parsed"]:
                cell["json_parse_failures_v2"] += 1
            elif not r["recognized"]:
                cell["alias_unrecognized_v2"] += 1
                if r["attempted_action"]:
                    cell["alias_unrecognized_examples"][r["attempted_action"]] += 1
            else:  # recognized but illegal
                cell["illegal_in_context_v2"] += 1
                missing = sorted(set(r["legal_actions"]))
                key = f'{r["attempted_action"]} not in {missing}'
                cell["illegal_in_context_breakdown"][key] += 1

    # Cap example dicts so the JSON file stays small.
    cell["illegal_in_context_breakdown"] = dict(
        cell["illegal_in_context_breakdown"].most_common(max_examples * 2)
    )
    cell["alias_unrecognized_examples"] = dict(
        cell["alias_unrecognized_examples"].most_common(max_examples)
    )
    return cell


def _add_rates(cell: dict) -> dict:
    n = max(cell["total_decisions"], 1)
    cell["parse_success_orig_pct"] = 100.0 * cell["parse_success_orig"] / n
    cell["parse_success_v2_pct"] = 100.0 * cell["parse_success_v2"] / n
    cell["json_parse_failures_v2_pct"] = 100.0 * cell["json_parse_failures_v2"] / n
    cell["alias_unrecognized_v2_pct"] = 100.0 * cell["alias_unrecognized_v2"] / n
    cell["illegal_in_context_v2_pct"] = 100.0 * cell["illegal_in_context_v2"] / n
    cell["recovered_by_alias_norm_pct"] = 100.0 * cell["recovered_by_alias_norm"] / n
    return cell


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

_MD_HEADER = """# Action-metadata re-attribution (v2)

This report re-categorizes action-metadata failures from CoT logs using
the alias-aware logic in `poker_env/agents/json_utils.py::normalize_action_str`
and a finer-grained split of "why fallback fired".

Columns:

* `n` — total decisions in the cell.
* `parse_v1%` — original `parse_success` rate (json AND recognized AND legal,
   pre-alias).
* `parse_v2%` — same, but using ACTION_ALIASES (CHECK -> CHECK_OR_CALL, etc.).
* `Δ recovered%` — points recovered purely by alias normalization
   (i.e. the in-code fix in `json_utils.py`). This is the upper bound
   of what the alias fix would have rescued in already-collected data.
* `JSON-fail%` — `parse_v2=False` because no valid JSON object was returned.
* `Alias-unk%` — `parse_v2=False` because the model emitted a string we
   still don't recognize even after normalization (NOT fixable in code;
   add another alias if it shows up frequently).
* `Illegal%` — `parse_v2=False` because the model picked a recognized
   action that wasn't in `legal_actions` (e.g. FOLD when bet_to_call=0).
   This is a *model behavior* observation, NOT a parser bug.

"""


def _write_markdown(cells: list[dict], path: str, baseline_cells: list[dict] | None = None) -> None:
    rows = sorted(cells, key=lambda c: c["log_basename"])

    def _fmt_pct(x: float) -> str:
        return f"{x:5.1f}"

    lines: list[str] = [_MD_HEADER, "## Per-cell breakdown\n"]
    lines.append("| Log | n | parse_v1% | parse_v2% | Δ recovered% | JSON-fail% | Alias-unk% | Illegal% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in rows:
        lines.append(
            f"| `{c['log_basename']}` "
            f"| {c['total_decisions']} "
            f"| {_fmt_pct(c['parse_success_orig_pct'])} "
            f"| {_fmt_pct(c['parse_success_v2_pct'])} "
            f"| {_fmt_pct(c['recovered_by_alias_norm_pct'])} "
            f"| {_fmt_pct(c['json_parse_failures_v2_pct'])} "
            f"| {_fmt_pct(c['alias_unrecognized_v2_pct'])} "
            f"| {_fmt_pct(c['illegal_in_context_v2_pct'])} |"
        )

    lines.append("\n## Failure-mode detail (top illegal-action contexts and unknown aliases)\n")
    for c in rows:
        if not c["illegal_in_context_breakdown"] and not c["alias_unrecognized_examples"]:
            continue
        lines.append(f"### `{c['log_basename']}`")
        if c["illegal_in_context_breakdown"]:
            lines.append("Top illegal-in-context attempts:")
            for k, v in c["illegal_in_context_breakdown"].items():
                lines.append(f"- `{k}` — **{v}**")
        if c["alias_unrecognized_examples"]:
            lines.append("Unknown action strings (after normalization):")
            for k, v in c["alias_unrecognized_examples"].items():
                lines.append(f"- `{k}` — **{v}**")
        lines.append("")

    if baseline_cells:
        # Match by basename minus a leading condition prefix when possible.
        # We strip whichever known prefix is present so cot_X_t0_s42 lines up
        # with scaled_X_t0_s42 / baseline_X_t0_s42 / X_t0_s42.
        _PREFIXES = ("cot_", "baseline_", "scaled_")

        def _stem(name: str) -> str:
            s = name
            for p in _PREFIXES:
                if s.startswith(p):
                    s = s[len(p):]
                    break
            return (
                s.replace("_enriched", "")
                .replace(".jsonl", "")
            )

        cot_by = {_stem(c["log_basename"]): c for c in rows}
        base_by = {_stem(c["log_basename"]): c for c in baseline_cells}
        common = sorted(set(cot_by) & set(base_by))
        if common:
            lines.append("## CoT vs baseline (parse_v2%)\n")
            lines.append("| Cell | baseline n | baseline parse_v2% | CoT n | CoT parse_v2% | Δ (CoT − baseline) |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for stem in common:
                b = base_by[stem]
                c = cot_by[stem]
                delta = c["parse_success_v2_pct"] - b["parse_success_v2_pct"]
                lines.append(
                    f"| `{stem}` | {b['total_decisions']} | {_fmt_pct(b['parse_success_v2_pct'])} "
                    f"| {c['total_decisions']} | {_fmt_pct(c['parse_success_v2_pct'])} "
                    f"| {delta:+.1f} |"
                )

    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _expand(globs: Iterable[str]) -> list[str]:
    out = []
    for g in globs:
        out.extend(sorted(glob.glob(g)))
    seen, deduped = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-glob", action="append", required=True,
                    help="Glob for CoT log files (repeat to add more). "
                         "Quote it on the shell, e.g. --logs-glob 'results/cot_*.jsonl'.")
    ap.add_argument("--baseline-glob", action="append", default=None,
                    help="Optional glob for non-CoT baseline logs to compare against.")
    ap.add_argument("--json-out", required=True, help="Path for the JSON breakdown.")
    ap.add_argument("--md-out", required=True, help="Path for the human-readable report.")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-file progress.")
    args = ap.parse_args()

    cot_paths = _expand(args.logs_glob)
    if not cot_paths:
        sys.exit(f"[recategorize] no logs matched: {args.logs_glob}")

    cells = []
    for p in cot_paths:
        if not args.quiet:
            print(f"[recategorize] {p}", file=sys.stderr)
        cells.append(_add_rates(_aggregate_log(p)))

    baseline_cells = None
    if args.baseline_glob:
        base_paths = _expand(args.baseline_glob)
        baseline_cells = []
        for p in base_paths:
            if not args.quiet:
                print(f"[recategorize] (baseline) {p}", file=sys.stderr)
            baseline_cells.append(_add_rates(_aggregate_log(p)))

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.md_out)) or ".", exist_ok=True)

    out = {
        "cot": cells,
        "baseline": baseline_cells or [],
        "schema_note": (
            "parse_success_v2 = json_parsed AND action_recognized AND "
            "action_legal_in_context (after ACTION_ALIASES)."
        ),
    }
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    _write_markdown(cells, args.md_out, baseline_cells=baseline_cells)

    if not args.quiet:
        print(f"[recategorize] wrote {args.json_out}", file=sys.stderr)
        print(f"[recategorize] wrote {args.md_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
