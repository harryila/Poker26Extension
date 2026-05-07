"""
Audit script: scans every logit-lens sidecar for tokens emitted at the
action-verb position that are NOT covered by our hardcoded action-token
families in `analyze_logit_lens_by_failure_mode.py`.

Why this exists
---------------
Our action-family classifier (`_match_action_group`) hardcodes substring
matches for FOLD/CHECK/CALL/BET/RAISE plus subword-prefix variants
(e.g. 'F'+'OLD' for Llama). If a model's tokenizer splits an action verb in
a way we haven't seen, those tokens silently map to "OTHER" and undercount
the action signal in the per-layer mix tables.

This script is FORWARD-LOOKING insurance: before we run cross-model causal
patching on Llama and Qwen sidecars, run this to confirm we're not missing
tokens. Findings get folded back into ACTION_TOKEN_GROUPS /
ACTION_SUBWORD_PREFIXES (in `analyze_logit_lens_by_failure_mode.py`).

Usage::

    python -m analysis.audit_action_verb_tokens \
        --sidecars 'logs/cot_*_logitlens_logit_lens.jsonl.gz' \
        --json-out results/audit/action_verb_tokens.json \
        --md-out   results/audit/AUDIT_ACTION_VERB_TOKENS.md

Output: per-cell histogram of tokens at the action-verb position, broken
down into "covered by alias" vs "uncovered" (= candidates for new aliases).

Pure CPU. Runs in seconds on the 18 cells we have locally.
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

from analysis.analyze_logit_lens_by_failure_mode import (  # noqa: E402
    ACTION_TOKEN_GROUPS,
    ACTION_SUBWORD_PREFIXES,
    _find_action_position,
    _match_action_group,
)


def _open_log(path: str) -> io.TextIOBase:
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_jsonl(path: str):
    with _open_log(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _cell_label_from_path(p: str) -> str:
    base = os.path.basename(p)
    stem = re.sub(r"_logit_lens\.jsonl(\.gz)?$", "", base)
    stem = re.sub(r"^cot_", "", stem)
    return stem.replace("_informative_v2_logitlens", "")


def audit_sidecar(sidecar_path: str) -> dict:
    """Returns {cell, n_records, n_with_verb, covered_counter, uncovered_counter,
                examples_uncovered (max 10)}."""
    cell = _cell_label_from_path(sidecar_path)
    covered: collections.Counter = collections.Counter()
    uncovered: collections.Counter = collections.Counter()
    examples_uncovered: list[dict] = []
    n_records = 0
    n_with_verb = 0

    for rec in _iter_jsonl(sidecar_path):
        n_records += 1
        plt = rec.get("per_layer_top_tokens", [])
        pos = _find_action_position(plt)
        if pos is None:
            continue
        n_with_verb += 1
        # Final-layer token at the action-verb position.
        final_layer = plt[-1]
        if pos >= len(final_layer):
            continue
        tok = final_layer[pos]
        # Try matching with subword fallback ON (the most lenient mode).
        match = _match_action_group(tok, allow_subword=True)
        if match is not None:
            covered[(tok, match)] += 1
        else:
            uncovered[tok] += 1
            if len(examples_uncovered) < 10:
                examples_uncovered.append({
                    "token": tok,
                    "hand_id": rec.get("hand_id"),
                    "decision_idx": rec.get("decision_idx"),
                    "context": [
                        final_layer[i] for i in range(max(0, pos-2), min(len(final_layer), pos+3))
                    ],
                })

    return {
        "cell": cell,
        "sidecar": sidecar_path,
        "n_records": n_records,
        "n_with_verb_position": n_with_verb,
        "covered_counts": [
            {"token": tok, "mapped_to": grp, "count": c}
            for (tok, grp), c in covered.most_common()
        ],
        "uncovered_counts": [
            {"token": tok, "count": c}
            for tok, c in uncovered.most_common()
        ],
        "uncovered_examples": examples_uncovered,
    }


def write_markdown(results: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Action-verb token audit\n\n")
        f.write(
            "For every logit-lens sidecar, scans the FINAL-layer top-1 token at\n"
            "the JSON action-verb position. Reports how many records map to a known\n"
            "action family (FOLD/CHECK/CALL/BET/RAISE) and how many don't.\n\n"
            "Uncovered tokens at the verb position are CANDIDATES for new\n"
            "aliases. If the count is non-trivial (>1% of records in a cell),\n"
            "consider adding the token's first piece to ACTION_TOKEN_GROUPS or\n"
            "ACTION_SUBWORD_PREFIXES in `analyze_logit_lens_by_failure_mode.py`.\n\n"
        )
        # Cross-cell summary
        f.write("## Cross-cell summary\n\n")
        f.write("| Cell | Records | With verb pos | Covered | Uncovered | Coverage % |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in results:
            cov = sum(c["count"] for c in r["covered_counts"])
            unc = sum(c["count"] for c in r["uncovered_counts"])
            tot = cov + unc
            pct = 100.0 * cov / tot if tot else 0.0
            f.write(
                f"| `{r['cell']}` | {r['n_records']} | {r['n_with_verb_position']} "
                f"| {cov} | {unc} | {pct:.1f}% |\n"
            )
        f.write("\n")
        # Aggregate uncovered tokens across all cells
        agg = collections.Counter()
        for r in results:
            for u in r["uncovered_counts"]:
                agg[u["token"]] += u["count"]
        f.write("## Aggregate uncovered tokens (across all cells)\n\n")
        if not agg:
            f.write("_No uncovered tokens found. Coverage is complete._\n\n")
        else:
            f.write("| Token | Total count | Suggested action |\n|---|---:|---|\n")
            for tok, c in agg.most_common(50):
                # Heuristic: short alphabetical token = subword candidate
                suggestion = "Add to ACTION_SUBWORD_PREFIXES" if len(tok.strip()) <= 3 else "Inspect manually"
                f.write(f"| `{tok!r}` | {c} | {suggestion} |\n")
            f.write("\n")
        # Per-cell detail (uncovered only)
        f.write("## Per-cell uncovered detail\n\n")
        for r in results:
            f.write(f"### `{r['cell']}`\n\n")
            if not r["uncovered_counts"]:
                f.write("_All verb-position tokens covered._\n\n")
                continue
            f.write("Top uncovered tokens:\n\n")
            f.write("| Token | Count |\n|---|---:|\n")
            for u in r["uncovered_counts"][:20]:
                f.write(f"| `{u['token']!r}` | {u['count']} |\n")
            f.write("\n")
            if r["uncovered_examples"]:
                f.write("Sample contexts (window of 5 tokens around the verb position):\n\n")
                for ex in r["uncovered_examples"][:5]:
                    ctx = " | ".join(repr(t) for t in ex["context"])
                    f.write(
                        f"- hand=`{ex['hand_id']}` dec={ex['decision_idx']} "
                        f"token=`{ex['token']!r}`  context: {ctx}\n"
                    )
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Audit action-verb tokens across logit-lens sidecars."
    )
    parser.add_argument(
        "--sidecars", default="logs/cot_*_logitlens_logit_lens.jsonl*",
        help="Glob for sidecar files (supports .jsonl and .jsonl.gz)",
    )
    parser.add_argument(
        "--json-out", default="results/audit/action_verb_tokens.json",
    )
    parser.add_argument(
        "--md-out", default="results/audit/AUDIT_ACTION_VERB_TOKENS.md",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.sidecars))
    if not paths:
        print(f"No sidecars matched {args.sidecars}", file=sys.stderr)
        sys.exit(1)
    print(f"[audit] scanning {len(paths)} sidecars ...")
    print(f"[audit] current ACTION_TOKEN_GROUPS keys: {list(ACTION_TOKEN_GROUPS.keys())}")
    print(f"[audit] current ACTION_SUBWORD_PREFIXES keys: {list(ACTION_SUBWORD_PREFIXES.keys())}")
    print()

    results = []
    for p in paths:
        r = audit_sidecar(p)
        results.append(r)
        cov = sum(c["count"] for c in r["covered_counts"])
        unc = sum(c["count"] for c in r["uncovered_counts"])
        tot = cov + unc
        pct = 100.0 * cov / tot if tot else 0.0
        print(
            f"  {r['cell']:<40} records={r['n_records']:>4} "
            f"verb_pos={r['n_with_verb_position']:>4} "
            f"covered={cov:>4} uncovered={unc:>4}  ({pct:.1f}%)"
        )

    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[audit] wrote {args.json_out}")
    write_markdown(results, args.md_out)
    print(f"[audit] wrote {args.md_out}")

    # Summary across cells
    agg = collections.Counter()
    for r in results:
        for u in r["uncovered_counts"]:
            agg[u["token"]] += u["count"]
    if agg:
        print("\n=== Top 20 uncovered tokens (across all cells) ===")
        for tok, c in agg.most_common(20):
            print(f"  count={c:>4}  {tok!r}")
    else:
        print("\n=== Coverage is complete: no uncovered tokens. ===")


if __name__ == "__main__":
    main()
