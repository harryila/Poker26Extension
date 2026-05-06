"""
Logit-lens × action-metadata cross-join: per-layer prediction trajectories
broken down by FAILURE MODE.

Why this exists
---------------
run_tier1a_small_cot_logitlens.sh produces TWO files per cell:

  logs/<cell>_logit_lens.jsonl        -- per-decision per-layer top-1 tokens
  logs/<cell>_enriched.jsonl[.gz]     -- per-decision action_metadata with the
                                          new diagnostic flags from updates.md
                                          §11 (action_json_parsed,
                                          action_recognized,
                                          action_legal_in_context).

`analyze_logit_lens.py` collapses the sidecar to a single per-cell summary
(mean entropy curve, mean crystallization layer). That tells us nothing about
the actual mechanistic question raised by the small-tier CoT dig-in:

  When the model emits FOLD into a free-check spot (action_recognized=True,
  action_legal_in_context=False, attempted=FOLD), do EARLIER layers ever
  predict CHECK / CALL at the action-emission position?

  - If yes  -> verbalization failure: the model "knew" the right call and
                lost it at the lm_head. Suggests fine-tuning the output head
                or constrained decoding could fix it.
  - If no   -> the entire residual stream is committed to FOLD from layer 0.
                Suggests a deeper representational issue (CoT scratchpad
                contaminated the planning circuit, etc.).

This script answers that question. It joins (hand_id, decision_idx) keys,
buckets decisions by failure mode, and per bucket reports:

  - num decisions in bucket
  - per-layer fraction whose top-1 token at the LAST generated position
    matches each candidate "answer" (FOLD / CHECK / CALL / other)
  - mean crystallization-layer position
  - mean per-layer entropy curve

Buckets:
  clean                 action_legal_in_context AND parse_success
  illegal_fold          attempted=FOLD AND not legal_in_context (the
                          interesting one — these are the rescued-to-CHECK
                          decisions in pot_odds/SUMMARY.md)
  illegal_other         not legal_in_context AND attempted != FOLD
  alias_unrecognized    json_parsed but action string not in canonical map
                          (should be empty after the alias fix; sanity
                          check)
  json_failure          model emitted no parseable JSON

Usage
-----
Per cell:
    python -m analysis.analyze_logit_lens_by_failure_mode \
        --logit-lens-sidecar logs/cot_ministral8b_t0_s42_informative_v2_logitlens_logit_lens.jsonl \
        --enriched-log       logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl \
        --label              ministral8b_cot_t0_s42 \
        --json-out           results/tier1a_small_cot_logitlens/by_failure_mode_ministral_s42_t0.json \
        --md-out             results/tier1a_small_cot_logitlens/BY_FAILURE_MODE.md

Multi-cell aggregate (use shell glob loop in calling script).

Notes
-----
- Both inputs are keyed on (hand_id, decision_idx).
- Sidecar is plain .jsonl (not gzipped at write time).
- Enriched log is read transparently from .jsonl OR .jsonl.gz.
- Token-match logic is case-folded and stripped of leading whitespace, since
  tokenizers prepend spaces (e.g. " FOLD" / "FOLD" / "fold" all match).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import io
import json
import os
import sys
from typing import Iterable

# Reuse the same alias-aware reparser used by recategorize_action_metadata.py.
# Single source of truth for failure-mode buckets.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.recategorize_action_metadata import _reparse_one  # noqa: E402


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _open_log(path: str) -> io.TextIOBase:
    """Transparent .jsonl / .jsonl.gz reader (mirrors recategorize script)."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_jsonl(path: str) -> Iterable[dict]:
    with _open_log(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Failure-mode bucketing
# ---------------------------------------------------------------------------

BUCKETS = (
    "clean",
    "illegal_fold",
    "illegal_other",
    "alias_unrecognized",
    "json_failure",
)


def _classify(rec: dict) -> str:
    """Return one of BUCKETS based on the new diagnostic flags."""
    info = _reparse_one(rec)
    if not info["json_parsed"]:
        return "json_failure"
    if not info["recognized"]:
        return "alias_unrecognized"
    if info["legal"]:
        return "clean"
    if (info["attempted_action"] or "").upper() == "FOLD":
        return "illegal_fold"
    return "illegal_other"


# ---------------------------------------------------------------------------
# Token matching at the action-emission position
# ---------------------------------------------------------------------------

# Tokens we care about for the "did any layer say CHECK/CALL while final
# layer said FOLD?" question. Matching is case-insensitive and whitespace-
# stripped — different tokenizers prepend spaces or split FOLD into pieces.
ACTION_TOKEN_GROUPS: dict[str, tuple[str, ...]] = {
    "FOLD":  ("fold", "folding"),
    "CHECK": ("check", "checking"),
    "CALL":  ("call", "calling"),
    "BET":   ("bet", "betting"),
    "RAISE": ("raise", "raising"),
}


def _normalize_token(tok: str) -> str:
    return (tok or "").strip().lower()


# Subword fragments that the tokenizer emits when an action verb gets split.
# Built from inspection of real Llama / Qwen / Ministral logit-lens sidecars.
# E.g. Llama splits FOLD into "F" + "OLD" / "old" depending on context.
ACTION_SUBWORD_PREFIXES: dict[str, tuple[str, ...]] = {
    "FOLD":  ("f", "fol"),     # 'F' + 'OLD', 'Fol' + 'd', etc.
    "CHECK": ("ch", "che"),    # rarely split but cheap to guard
    "CALL":  ("ca",),
    "BET":   ("b", "be"),
    "RAISE": ("r", "ra"),
}


def _match_action_group(tok: str, *, allow_subword: bool = False) -> str | None:
    """Map a tokenizer-decoded string to one of ACTION_TOKEN_GROUPS keys.

    `allow_subword`: if True, also match short prefixes that would only
    appear when an action verb has been split into pieces by the tokenizer
    (e.g. 'F' / 'OLD' for FOLD). Disabled by default because short prefixes
    are too generic to use for finding the action POSITION on their own; we
    only enable them once the position is anchored by a full-word match in
    the final layer."""
    n = _normalize_token(tok)
    if not n:
        return None
    for group, variants in ACTION_TOKEN_GROUPS.items():
        for v in variants:
            if v in n:
                return group
    if allow_subword:
        for group, prefixes in ACTION_SUBWORD_PREFIXES.items():
            if n in prefixes:
                return group
    return None


def _find_action_position(per_layer_top_tokens: list[list[str]]) -> int | None:
    """Find the generated-token position that emitted the canonical action
    verb (e.g. 'CHECK' for CHECK_OR_CALL, 'FOLD' for FOLD, 'B' for BET_OR_RAISE
    when the tokenizer subword-splits BET as 'B' + 'ET').

    Strategy: anchor on the FINAL layer, find the JSON close brace token
    `'"}'` near the end, walk back to the JSON open quote `' "'` (the one
    that opens the action VALUE, not the one after the action KEY), and
    return open_pos + 1. That position is always the first token of the
    action verb — works for full-word verbs ('FOLD', 'CHECK') and for
    subword-split verbs ('F'+'OLD', 'B'+'ET'+'_OR'+'_RA'+'ISE') alike.

    Why this beats substring matching: tokens like ' folding' (in reasoning),
    ' better' (contains 'bet'), or ' or' / ' raise' (in CoT exposition)
    would otherwise spuriously match the action-verb groups; this anchor-
    based search is bounded by the JSON quotes so it can only land inside
    the action value.

    Returns None if the response had no parseable JSON close (json-failure
    bucket). Those records correctly contribute 0 layer trajectories.
    """
    if not per_layer_top_tokens:
        return None
    final_layer = per_layer_top_tokens[-1]
    if not final_layer:
        return None
    n = len(final_layer)

    # 1. Find LAST '"}' close-brace (search only the last 10 tokens — JSON
    #    is always at the very end of the response).
    close_pos = None
    for pos in range(n - 1, max(-1, n - 11), -1):
        tok = (final_layer[pos] or "").strip()
        if "\"}" in tok:
            close_pos = pos
            break
    if close_pos is None:
        return None

    # 2. Walk back from close_pos looking for the value-opening quote token
    #    (looks like ' "' or '"' — whitespace + quote, NO colon, NO close).
    #    Cap the lookback at 10 tokens (BET_OR_RAISE is the longest verb at
    #    5 subword tokens; 10 is comfortable margin).
    for pos in range(close_pos - 1, max(-1, close_pos - 11), -1):
        tok = (final_layer[pos] or "").strip()
        if tok in ('"', '\\"'):
            return pos + 1
        # Some tokenizers emit the open-quote bundled with leading space, e.g. ' "'
        # (raw token starts with whitespace and ends in '"'). Detect that without
        # also matching '":' (action-key separator) or '"}' (close brace).
        raw = final_layer[pos] or ""
        if raw.endswith('"') and ':' not in raw and '}' not in raw and len(raw.strip()) <= 2:
            return pos + 1

    return None


def _per_layer_action_at_action_position(rec: dict) -> list[str | None]:
    """For one logit-lens record, return per-layer mapped action group AT THE
    ACTION-VERB POSITION (not the EOS position).

    Position is anchored by the final layer's action-verb token; subword
    matching is enabled for the per-layer scan because mid-layers often emit
    'F' / 'fol' / 'OLD' fragments that we want to count as FOLD.
    """
    per_layer = rec.get("per_layer_top_tokens", [])
    pos = _find_action_position(per_layer)
    if pos is None:
        return []
    out: list[str | None] = []
    for layer_tokens in per_layer:
        if pos >= len(layer_tokens):
            out.append(None)
            continue
        out.append(_match_action_group(layer_tokens[pos], allow_subword=True))
    return out


# ---------------------------------------------------------------------------
# Per-bucket aggregation
# ---------------------------------------------------------------------------


def _empty_bucket() -> dict:
    return {
        "n": 0,
        "n_with_lens": 0,
        "per_layer_action_counts": [],   # list of Counters, len = num_layers
        "per_layer_entropy_sum": [],     # list of floats, len = num_layers
        "per_layer_entropy_n": [],       # list of ints,   len = num_layers
        "crystallization_layers": [],    # list of ints, one per record
    }


def _ensure_layer_buffers(bucket: dict, num_layers: int) -> None:
    cur = len(bucket["per_layer_action_counts"])
    for _ in range(num_layers - cur):
        bucket["per_layer_action_counts"].append(collections.Counter())
        bucket["per_layer_entropy_sum"].append(0.0)
        bucket["per_layer_entropy_n"].append(0)


def _crystallization_layer(per_layer_actions: list[str | None]) -> int | None:
    """Earliest layer L such that for all l>=L the mapped action group is
    the same as the FINAL layer's group. Mirrors
    analysis.analyze_logit_lens.compute_crystallization_layer but works on
    the action-group axis (so noise from synonymous tokens like FOLD vs
    folding doesn't artificially inflate the crystallization layer)."""
    if not per_layer_actions:
        return None
    final = per_layer_actions[-1]
    if final is None:
        return None
    for start in range(len(per_layer_actions)):
        if all(per_layer_actions[l] == final for l in range(start, len(per_layer_actions))):
            return start
    return len(per_layer_actions) - 1


def _add_record_to_bucket(bucket: dict, lens_rec: dict) -> None:
    per_layer_actions = _per_layer_action_at_action_position(lens_rec)
    num_layers = len(per_layer_actions)
    if num_layers == 0:
        return

    _ensure_layer_buffers(bucket, num_layers)
    bucket["n_with_lens"] += 1

    for layer_idx, action_group in enumerate(per_layer_actions):
        key = action_group if action_group is not None else "OTHER"
        bucket["per_layer_action_counts"][layer_idx][key] += 1

    per_layer_entropy = lens_rec.get("per_layer_entropy", [])
    for layer_idx, ent in enumerate(per_layer_entropy):
        if layer_idx >= num_layers:
            break
        bucket["per_layer_entropy_sum"][layer_idx] += float(ent)
        bucket["per_layer_entropy_n"][layer_idx] += 1

    cl = _crystallization_layer(per_layer_actions)
    if cl is not None:
        bucket["crystallization_layers"].append(cl)


# ---------------------------------------------------------------------------
# Top-level cell processing
# ---------------------------------------------------------------------------


def _join_and_classify(enriched_path: str, sidecar_path: str) -> dict:
    """Returns {"buckets": {bucket_name: aggregated_dict}, "stats": {...}}.

    Streams the enriched log; for each decision record looks up the
    matching sidecar entry by (hand_id, decision_idx). Sidecar is loaded
    fully into memory (small — typically <500 records per cell)."""
    sidecar_index: dict[tuple[str, int], dict] = {}
    for rec in _iter_jsonl(sidecar_path):
        hid = rec.get("hand_id")
        didx = rec.get("decision_idx")
        if hid is None or didx is None:
            continue
        sidecar_index[(hid, int(didx))] = rec

    buckets = {b: _empty_bucket() for b in BUCKETS}
    stats = {
        "enriched_decisions_seen": 0,
        "enriched_decisions_with_action_metadata": 0,
        "sidecar_records_loaded": len(sidecar_index),
        "joined_records": 0,
        "unmatched_enriched": 0,
    }

    for rec in _iter_jsonl(enriched_path):
        if rec.get("type") not in (None, "decision"):
            continue
        if "action_metadata" not in rec:
            continue
        stats["enriched_decisions_seen"] += 1
        am = rec.get("action_metadata") or {}
        if am:
            stats["enriched_decisions_with_action_metadata"] += 1

        bucket_name = _classify(rec)
        bucket = buckets[bucket_name]
        bucket["n"] += 1

        hid = rec.get("hand_id")
        didx = rec.get("decision_idx")
        key = (hid, int(didx) if didx is not None else -1)
        lens_rec = sidecar_index.get(key)
        if lens_rec is None:
            stats["unmatched_enriched"] += 1
            continue
        stats["joined_records"] += 1
        _add_record_to_bucket(bucket, lens_rec)

    return {"buckets": buckets, "stats": stats}


def _summarize_bucket(bucket: dict) -> dict:
    """Reduce raw buffers to JSON-serializable summary."""
    num_layers = len(bucket["per_layer_action_counts"])
    per_layer_action_fractions = []
    for layer_idx in range(num_layers):
        counter = bucket["per_layer_action_counts"][layer_idx]
        total = sum(counter.values()) or 1
        frac = {action: round(c / total, 4) for action, c in counter.items()}
        per_layer_action_fractions.append(frac)

    per_layer_mean_entropy = []
    for s, n in zip(bucket["per_layer_entropy_sum"], bucket["per_layer_entropy_n"]):
        per_layer_mean_entropy.append(round(s / n, 4) if n else None)

    cls = bucket["crystallization_layers"]
    crystallization = None
    if cls:
        cls_sorted = sorted(cls)
        mid = len(cls_sorted) // 2
        median = (
            cls_sorted[mid]
            if len(cls_sorted) % 2 == 1
            else 0.5 * (cls_sorted[mid - 1] + cls_sorted[mid])
        )
        crystallization = {
            "n": len(cls),
            "mean": round(sum(cls) / len(cls), 2),
            "median": round(median, 2),
            "min": min(cls),
            "max": max(cls),
        }

    return {
        "n_decisions": bucket["n"],
        "n_with_logit_lens": bucket["n_with_lens"],
        "num_layers": num_layers,
        "per_layer_action_fraction": per_layer_action_fractions,
        "per_layer_mean_entropy": per_layer_mean_entropy,
        "crystallization_layer": crystallization,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _md_for_bucket(label: str, bucket_name: str, summary: dict) -> str:
    lines = [f"### {label} — bucket: `{bucket_name}`", ""]
    lines.append(f"- N decisions in bucket: **{summary['n_decisions']}**")
    lines.append(f"- N joined to logit-lens sidecar: {summary['n_with_logit_lens']}")
    lines.append(f"- Num layers: {summary['num_layers']}")
    cl = summary.get("crystallization_layer")
    if cl:
        lines.append(
            f"- Crystallization layer (action-group axis): mean={cl['mean']}, "
            f"median={cl['median']}, range=[{cl['min']}, {cl['max']}], n={cl['n']}"
        )
    else:
        lines.append("- Crystallization layer: n/a (no records joined)")
    lines.append("")

    if summary["n_with_logit_lens"] == 0:
        lines.append("_No logit-lens records joined — nothing to report._")
        lines.append("")
        return "\n".join(lines)

    fracs = summary["per_layer_action_fraction"]
    if not fracs:
        return "\n".join(lines)

    actions_seen: list[str] = []
    for frac in fracs:
        for k in frac:
            if k not in actions_seen:
                actions_seen.append(k)
    priority = ["FOLD", "CHECK", "CALL", "BET", "RAISE", "OTHER"]
    actions_seen.sort(key=lambda k: priority.index(k) if k in priority else 999)

    header = ["layer"] + actions_seen + ["entropy"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    entropies = summary["per_layer_mean_entropy"]
    for layer_idx, frac in enumerate(fracs):
        row = [str(layer_idx)]
        for action in actions_seen:
            v = frac.get(action, 0.0)
            row.append(f"{v:.2f}" if v else "·")
        ent = entropies[layer_idx] if layer_idx < len(entropies) else None
        row.append(f"{ent:.2f}" if ent is not None else "·")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    return "\n".join(lines)


def _write_md(path: str, label: str, results: dict, append: bool) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "a" if append and os.path.exists(path) else "w"
    with open(path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("# Logit-lens × failure-mode breakdown\n\n")
            f.write(
                "Per-bucket per-layer mapped action group at the FINAL generated\n"
                "position (i.e. the token where the model commits to an action in\n"
                "the JSON payload). Fractions sum across decisions in the bucket.\n\n"
                "Read like this: the FOLD column at layer L tells you what fraction\n"
                "of decisions in that bucket had top-1 token in the FOLD family at\n"
                "layer L's projection. If FOLD is 1.00 from layer 0, the model is\n"
                "FOLD-committed top-to-bottom. If FOLD only crosses 0.5 in the late\n"
                "layers while CHECK/CALL dominates early, that's the verbalization-\n"
                "failure signature.\n\n"
            )
        f.write("---\n\n")
        f.write(f"## {label}\n\n")
        stats = results.get("stats", {})
        f.write(
            f"**Stats:** enriched_decisions_seen={stats.get('enriched_decisions_seen')}, "
            f"sidecar_records_loaded={stats.get('sidecar_records_loaded')}, "
            f"joined={stats.get('joined_records')}, "
            f"unmatched={stats.get('unmatched_enriched')}\n\n"
        )
        for bucket_name in BUCKETS:
            summary = results["buckets"][bucket_name]
            f.write(_md_for_bucket(label, bucket_name, summary))
            f.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-join logit-lens sidecar with enriched action-metadata "
                    "and report per-layer prediction trajectories per failure mode."
    )
    parser.add_argument("--logit-lens-sidecar", required=True,
                        help="Path to <cell>_logit_lens.jsonl from "
                             "run_tier1a_small_cot_logitlens.sh")
    parser.add_argument("--enriched-log", required=True,
                        help="Path to <cell>_enriched.jsonl[.gz] from the same run")
    parser.add_argument("--label", required=True,
                        help="Cell label for the report (e.g. ministral8b_cot_t0_s42)")
    parser.add_argument("--json-out", default=None,
                        help="Per-cell JSON results path (overwrite)")
    parser.add_argument("--md-out", default=None,
                        help="Markdown report path (append-friendly: appends per call)")
    parser.add_argument("--md-overwrite", action="store_true",
                        help="If set, truncate --md-out instead of appending")
    args = parser.parse_args()

    raw = _join_and_classify(args.enriched_log, args.logit_lens_sidecar)
    results = {
        "label": args.label,
        "enriched_log": args.enriched_log,
        "logit_lens_sidecar": args.logit_lens_sidecar,
        "stats": raw["stats"],
        "buckets": {b: _summarize_bucket(raw["buckets"][b]) for b in BUCKETS},
    }

    print(f"[{args.label}] stats: {raw['stats']}")
    for b in BUCKETS:
        s = results["buckets"][b]
        cl = s.get("crystallization_layer") or {}
        print(
            f"  {b:>20}: n={s['n_decisions']:4d}  joined={s['n_with_logit_lens']:4d}  "
            f"crystallization_mean={cl.get('mean')}"
        )

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  -> wrote JSON to {args.json_out}")

    if args.md_out:
        _write_md(args.md_out, args.label, results, append=not args.md_overwrite)
        print(f"  -> {'overwrote' if args.md_overwrite else 'appended to'} {args.md_out}")


if __name__ == "__main__":
    main()
