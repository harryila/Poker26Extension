#!/usr/bin/env bash
# =============================================================================
# Tier 1A.small CoT pilot — does CoT rescue the 8B trash-collapse?
# =============================================================================
#
# Background: Tier 1A.small (no CoT) showed all three 8B models fail
# belief elicitation:
#   llama-8b   : 100% parse, 100% trash one-hot   (collapsed posterior)
#   qwen-8b    : 100% parse, 100% trash one-hot   (collapsed posterior)
#   ministral8b: 5.8% parse                       (cannot produce valid JSON)
#
# This pilot tests whether adding Chain-of-Thought (--cot) shifts any of
# those numbers. Cheap insurance against the unlikely-but-publishable case
# where CoT rescues an 8B model.
#
# Grid: 3 models x 2 temps (0.0, 0.2) x 1 seed (42) = 6 cells, 50 hands/cell
#   = 300 hands total
#   Wall-clock: ~30-60 min on a single A5000 / L4 (8B inference is fast)
#   CoT increases token budgets (action 64->512, belief 384->768) so this
#   is ~3-4x slower per generation than the non-CoT pilot, but still cheap.
#
# Outputs (written incrementally as each model completes):
#   logs/cot_pilot_<tag>_<temp>_s42_informative_v2.jsonl{,.gz?}
#   logs/cot_pilot_<tag>_<temp>_s42_informative_v2_enriched.jsonl
#   results/tier1a_small_cot_pilot/
#       pce_<tag>_records.csv, pce_<tag>_summary.csv
#       analyze_cot_<tag>.json                 (CoT-specific metrics:
#                                                parse rate, reasoning length,
#                                                opponent-range mentions, etc.)
#       degeneracy_<tag>.json                  (parse rate, trash-one-hot
#                                                rate vs the non-CoT baseline)
#       SUMMARY.md                             (human-readable rescue verdict)
#
# Failure isolation: re-running skips any cell whose log already exists.
#
# =============================================================================
# tmux behaviour (same pattern as run_tier1a_small.sh)
# =============================================================================
#
#   bash scripts/run_tier1a_small_cot_pilot.sh
#
# When run OUTSIDE tmux, the script creates a tmux session named
# 'poker_tier1a_small_cot_pilot' AND attaches you to it. The script does
# NOT auto-detach you — that is your call:
#   - Press Ctrl-B then D to detach (run continues in tmux)
#   - Reattach later with:  tmux attach -t poker_tier1a_small_cot_pilot
#   - Kill the run with:    tmux kill-session -t poker_tier1a_small_cot_pilot
# Set NO_TMUX=1 to force foreground execution (e.g. for CI).
# =============================================================================

set -euo pipefail

SESSION_NAME="poker_tier1a_small_cot_pilot"

# ---- Launch inside tmux (attached) if not already there ----
if [[ -z "${TMUX:-}" ]] && [[ -z "${NO_TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux is not installed. Install it (apt: sudo apt install tmux,"
        echo "       brew: brew install tmux), or rerun with NO_TMUX=1 to run in"
        echo "       the foreground (the run will die if your SSH disconnects)."
        exit 1
    fi
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists."
        echo "  Attach:        tmux attach -t $SESSION_NAME"
        echo "  Kill and rerun: tmux kill-session -t $SESSION_NAME && bash $0"
        exit 1
    fi
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    echo "Creating tmux session '$SESSION_NAME' and attaching you to it."
    echo "  When you want to leave the run going in the background:"
    echo "    Press Ctrl-B then D    (detaches you; run continues in tmux)"
    echo "  Reattach later with:   tmux attach -t $SESSION_NAME"
    echo "  Abort the run with:    tmux kill-session -t $SESSION_NAME"
    echo
    exec tmux new-session -s "$SESSION_NAME" \
        "NO_TMUX=1 bash '$SCRIPT_PATH'; echo; echo '[cot pilot finished — press any key to close window]'; read -n1 -s"
fi

# ---- Inside tmux (or NO_TMUX=1). Do the work. ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if   [[ -f "venv/bin/activate"   ]]; then source "venv/bin/activate"
    elif [[ -f "../venv/bin/activate" ]]; then source "../venv/bin/activate"
    else echo "WARNING: no venv found at ./venv or ../venv — using system python"
    fi
fi

LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results/tier1a_small_cot_pilot}"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

SEEDS=(42)
TEMPS=(0.0 0.2)
HANDS_PER_CELL="${HANDS_PER_CELL:-50}"
OPPONENT_PRESET="informative_v2"

temp_suffix() {
    case "$1" in
        0.0|0) echo "t0" ;;
        0.2)   echo "t02" ;;
        *)     echo "t${1//./}" ;;
    esac
}

# ---- Pre-flight: confirm Qwen 3 will get enable_thinking=TRUE under --cot ----
echo "=== Pre-flight: Qwen 3 thinking-mode gating under --cot ==="
python <<'PY'
from poker_env.config import MODEL_REGISTRY
cfg = MODEL_REGISTRY["qwen-8b"]
assert cfg.get("has_thinking_mode") is True, "qwen-8b missing has_thinking_mode flag"
print("  qwen-8b has_thinking_mode=True")
print("  This script DOES pass --cot, so cot_mode=True")
print("  -> enable_thinking=True (Qwen 3 native thinking ON during pilot)")
print("  This is intentional: when the experiment is 'does CoT help?', we want")
print("  ALL CoT mechanisms on (prompt-level REASONING: + Qwen native thinking).")
PY
echo

# Format: "<short_name>:<filename_tag>:<hub_dirname>"
MODELS=(
    "llama-8b:llama8b:models--meta-llama--Llama-3.1-8B-Instruct"
    "qwen-8b:qwen8b:models--Qwen--Qwen3-8B"
    "ministral-8b:ministral8b:models--mistralai--Ministral-8B-Instruct-2410"
)

# Rolling-cache mode: free each model's weights after its run/enrich/analyze
# completes. Required when HF cache lives on a tight-quota mount (e.g. /dev/shm
# tmpfs or 20 GB overlay) where the 3 x 8B model weights (~48 GB) won't fit
# simultaneously. Same pattern as run_tier1a_small.sh.
PURGE_HF_CACHE_AFTER_MODEL="${PURGE_HF_CACHE_AFTER_MODEL:-0}"

run_model() {
    local short="$1" tag="$2"
    local started_at; started_at="$(date +%s)"

    echo
    echo "############################################################"
    echo "## MODEL: $short  (tag=$tag)  [CoT pilot]"
    echo "## Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "############################################################"

    # ---- Phase 1: 2 cells (t=0, t=0.2) at seed 42 ----
    for seed in "${SEEDS[@]}"; do
        for temp in "${TEMPS[@]}"; do
            local tsuf; tsuf="$(temp_suffix "$temp")"
            local out="${LOGS_DIR}/cot_pilot_${tag}_${tsuf}_s${seed}_${OPPONENT_PRESET}.jsonl"
            if [[ -f "$out" ]] || [[ -f "${out}.gz" ]]; then
                echo "  [skip] $out already exists"
                continue
            fi
            echo "  [run]  $out  (seed=$seed temp=$temp hands=$HANDS_PER_CELL --cot)"
            python run_experiment.py \
                --agent hf \
                --hf-model "$short" \
                --opponent threshold \
                --opponent-preset "$OPPONENT_PRESET" \
                --hands "$HANDS_PER_CELL" \
                --seed "$seed" \
                --temperature "$temp" \
                --elicit-beliefs --cot --capture-logprobs \
                --out "$out" \
                -v
        done
    done

    # ---- Phase 2: enrich ----
    echo
    echo "  [enrich] $tag"
    shopt -s nullglob
    for f in "${LOGS_DIR}"/cot_pilot_${tag}_*_${OPPONENT_PRESET}.jsonl; do
        [[ "$f" == *_enriched.jsonl ]] && continue
        local out_enr="${f%.jsonl}_enriched.jsonl"
        if [[ -f "$out_enr" ]]; then
            echo "    [skip] $out_enr already exists"
            continue
        fi
        echo "    [enrich] $f -> $out_enr"
        python -m analysis.build_dataset \
            "$f" "$out_enr" \
            --opponent "$OPPONENT_PRESET"
    done

    # ---- Phase 3: standard PCE + dedicated CoT analysis ----
    echo
    echo "  [analyze] $tag"
    local files=( "${LOGS_DIR}"/cot_pilot_${tag}_*_enriched.jsonl )
    if [[ "${#files[@]}" -eq 0 ]]; then
        echo "    [skip] no enriched logs for $tag"
        return
    fi
    python -m analysis.compute_pce_distribution \
        "${files[@]}" \
        --output-records "${RESULTS_DIR}/pce_${tag}_records.csv" \
        --output-summary "${RESULTS_DIR}/pce_${tag}_summary.csv" \
        --bootstrap 1000 --seed 42 --clustered || echo "    (PCE may be meaningless if beliefs degenerate)"

    python -m analysis.analyze_cot \
        "${files[@]}" \
        --json-out "${RESULTS_DIR}/analyze_cot_${tag}.json" \
        || echo "    (analyze_cot failed — pipeline may need adjustment)"

    local elapsed=$(( $(date +%s) - started_at ))
    echo "  ## DONE $short in ${elapsed}s"
}

for entry in "${MODELS[@]}"; do
    IFS=':' read -r short tag hub_dirname <<< "$entry"
    run_model "$short" "$tag"
    if [[ "$PURGE_HF_CACHE_AFTER_MODEL" == "1" ]]; then
        hub_root="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
        target="${hub_root%/}/${hub_dirname}"
        if [[ -d "$target" ]]; then
            echo "  [purge] freeing ${target}"
            rm -rf "$target"
        fi
        xet_dir="$(dirname "$hub_root")/xet"
        if [[ -d "$xet_dir" ]]; then
            echo "  [purge] freeing ${xet_dir}"
            rm -rf "${xet_dir:?}"/*
        fi
    fi
done

# =============================================================================
# Phase 4: Side-by-side degeneracy diff (CoT pilot vs non-CoT baseline)
# =============================================================================
# This is the headline question: does CoT shift parse rate or trash-one-hot
# vs the non-CoT cells we already ran (scaled_<tag>_t{0,02}_s42_*)?
echo
echo "############################################################"
echo "## Phase 4: CoT vs no-CoT degeneracy comparison"
echo "############################################################"

python <<'PY'
"""Compute headline degeneracy diff: CoT pilot vs non-CoT tier1a_small baseline.

Both runs use the SAME seeds, temps, opponent, and hand counts (50/cell at
seed 42; the non-CoT run also has more cells, but we restrict to the matched
subset for an apples-to-apples diff).
"""
import gzip, io, json, glob, os
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results/tier1a_small_cot_pilot"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def open_log(path):
    """Open a jsonl or jsonl.gz transparently."""
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r")

def diagnose(file_globs):
    """Return parse rate, trash one-hot rate, action counter, mean reasoning chars."""
    files = []
    for g in file_globs:
        files.extend(glob.glob(g))
    decisions = belief_attempts = parse_ok = trash_one_hot = 0
    actions = Counter()
    reasoning_chars = []
    for fp in files:
        with open_log(fp) as f:
            for line in f:
                r = json.loads(line)
                if "agent_action" not in r:
                    continue
                decisions += 1
                if isinstance(r["agent_action"], str):
                    actions[r["agent_action"]] += 1
                bm = r.get("belief_metadata") or {}
                if isinstance(bm, dict) and bm:
                    belief_attempts += 1
                    if bm.get("parse_success"):
                        parse_ok += 1
                        bel = r.get("agent_belief")
                        if isinstance(bel, dict):
                            top_name, top_prob = max(bel.items(), key=lambda kv: kv[1])
                            if top_name == "trash" and top_prob >= 0.99:
                                trash_one_hot += 1
                cot = r.get("cot_reasoning")
                if isinstance(cot, str) and cot.strip():
                    reasoning_chars.append(len(cot))
    return {
        "files": len(files),
        "decisions": decisions,
        "belief_attempts": belief_attempts,
        "parse_ok": parse_ok,
        "parse_rate_pct": (parse_ok / belief_attempts * 100) if belief_attempts else 0,
        "trash_one_hot": trash_one_hot,
        "trash_one_hot_pct_of_parsed": (trash_one_hot / parse_ok * 100) if parse_ok else 0,
        "actions": dict(actions.most_common()),
        "n_reasoning_traces": len(reasoning_chars),
        "mean_reasoning_chars": (sum(reasoning_chars) / len(reasoning_chars)) if reasoning_chars else 0,
    }

# Match the non-CoT cells at seed 42 only (both temps), so the diff is apples-to-apples.
COMPARISONS = {
    "llama8b": {
        "no_cot": ["logs/scaled_llama8b_t0_s42_informative_v2_enriched.jsonl",
                   "logs/scaled_llama8b_t0_s42_informative_v2_enriched.jsonl.gz",
                   "logs/scaled_llama8b_t02_s42_informative_v2_enriched.jsonl",
                   "logs/scaled_llama8b_t02_s42_informative_v2_enriched.jsonl.gz"],
        "cot":    ["logs/cot_pilot_llama8b_t0_s42_informative_v2_enriched.jsonl",
                   "logs/cot_pilot_llama8b_t02_s42_informative_v2_enriched.jsonl"],
    },
    "qwen8b": {
        "no_cot": ["logs/scaled_qwen8b_t0_s42_informative_v2_enriched.jsonl",
                   "logs/scaled_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz",
                   "logs/scaled_qwen8b_t02_s42_informative_v2_enriched.jsonl",
                   "logs/scaled_qwen8b_t02_s42_informative_v2_enriched.jsonl.gz"],
        "cot":    ["logs/cot_pilot_qwen8b_t0_s42_informative_v2_enriched.jsonl",
                   "logs/cot_pilot_qwen8b_t02_s42_informative_v2_enriched.jsonl"],
    },
    "ministral8b": {
        "no_cot": ["logs/scaled_ministral8b_t0_s42_informative_v2_enriched.jsonl",
                   "logs/scaled_ministral8b_t0_s42_informative_v2_enriched.jsonl.gz",
                   "logs/scaled_ministral8b_t02_s42_informative_v2_enriched.jsonl",
                   "logs/scaled_ministral8b_t02_s42_informative_v2_enriched.jsonl.gz"],
        "cot":    ["logs/cot_pilot_ministral8b_t0_s42_informative_v2_enriched.jsonl",
                   "logs/cot_pilot_ministral8b_t02_s42_informative_v2_enriched.jsonl"],
    },
}

print()
print(f"{'Model':<14} | {'Mode':<7} | {'cells':>5} {'decisions':>10} {'parse_OK%':>10} {'trash1hot%':>11} | {'reasoning_chars(avg)':>22}")
print("-" * 100)
summary_rows = {}
for tag, paths in COMPARISONS.items():
    for mode in ("no_cot", "cot"):
        d = diagnose(paths[mode])
        summary_rows.setdefault(tag, {})[mode] = d
        print(f"{tag:<14} | {mode:<7} | {d['files']:>5} {d['decisions']:>10} {d['parse_rate_pct']:>9.1f}% {d['trash_one_hot_pct_of_parsed']:>10.1f}% | {d['mean_reasoning_chars']:>22.0f}")
    # Verdict
    nc = summary_rows[tag]["no_cot"]; co = summary_rows[tag]["cot"]
    parse_delta = co["parse_rate_pct"] - nc["parse_rate_pct"]
    trash_delta = co["trash_one_hot_pct_of_parsed"] - nc["trash_one_hot_pct_of_parsed"]
    print(f"{'':<14} | DELTA   | {'':>5} {'':>10} {parse_delta:+9.1f}p {trash_delta:+10.1f}p")
    if abs(trash_delta) >= 10 or abs(parse_delta) >= 10:
        print(f"{'':<14} |   ^^ NON-TRIVIAL CoT EFFECT — worth a full grid")
    else:
        print(f"{'':<14} |   ^^ no rescue — capability floor confirmed below CoT")
    print()

# Write JSON for downstream + a markdown summary
with open(RESULTS_DIR / "degeneracy_diff.json", "w") as f:
    json.dump(summary_rows, f, indent=2)

with open(RESULTS_DIR / "SUMMARY.md", "w") as f:
    f.write("# Tier 1A.small CoT pilot — summary\n\n")
    f.write("Same seed (42), same temps (0.0 and 0.2), same opponent (informative_v2), 50 hands/cell.\n")
    f.write("Only difference: --cot flag.\n\n")
    f.write("| Model | Mode | Decisions | Parse rate | Trash one-hot (of parsed) | Mean reasoning chars |\n")
    f.write("|---|---|---:|---:|---:|---:|\n")
    for tag, modes in summary_rows.items():
        for mode in ("no_cot","cot"):
            d = modes[mode]
            f.write(f"| {tag} | {mode} | {d['decisions']} | {d['parse_rate_pct']:.1f}% | {d['trash_one_hot_pct_of_parsed']:.1f}% | {d['mean_reasoning_chars']:.0f} |\n")
        nc, co = modes["no_cot"], modes["cot"]
        f.write(f"| {tag} | **delta** | | {co['parse_rate_pct']-nc['parse_rate_pct']:+.1f}p | {co['trash_one_hot_pct_of_parsed']-nc['trash_one_hot_pct_of_parsed']:+.1f}p | |\n")

print(f"\nWrote: {RESULTS_DIR/'degeneracy_diff.json'}")
print(f"Wrote: {RESULTS_DIR/'SUMMARY.md'}")
PY

echo
echo "============================================================"
echo "CoT pilot COMPLETE."
echo "  Logs:    $LOGS_DIR/cot_pilot_*"
echo "  Results: $RESULTS_DIR/"
echo "  See:     $RESULTS_DIR/SUMMARY.md  (rescue verdict per model)"
echo "============================================================"
