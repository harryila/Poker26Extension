#!/usr/bin/env bash
# =============================================================================
# Tier 1A.small CoT + logit-lens — mechanistic add-on to run_tier1a_small_cot.sh
# =============================================================================
#
# WHY THIS EXISTS
# ---------------
# The full small-tier CoT grid (run_tier1a_small_cot.sh) exposed a striking
# behavioral artifact at 8B + CoT: 537 of 9,929 decisions across the 18 cells
# were "recognized but illegal" actions, almost all FOLDs picked when the
# model could check for free. Crucially, 0 of 8,749 NON-CoT decisions had
# this pattern. See updates.md §11 for the full breakdown:
#
#   - Ministral 8B: 339 illegal FOLDs across the two seed-42 CoT cells alone
#                   (34.1% / 30.8% of decisions)
#   - Llama 8B:     consistently 16-35 illegal FOLDs per cell, all seeds
#   - Qwen 8B:      4-13 illegal FOLDs per cell, all seeds
#
# This is a verbalized-vs-internal-calibration question that logit-lens is
# uniquely suited to answer:
#
#   "When the model emits FOLD into a free-check spot, does any intermediate
#    layer actually predict CHECK_OR_CALL? Or has the model 'committed' to
#    FOLD across all layers from the start?"
#
# If early layers say CHECK and only the final layer says FOLD, that's a
# verbalization failure. If every layer says FOLD, the model is genuinely
# confused about legality. Either answer is publishable.
#
# WHAT THIS SCRIPT DOES
# ---------------------
# Re-runs ONE seed × ONE temperature for all three 8B models with the same
# --cot --capture-logprobs flags as the full grid PLUS --logit-lens. Same
# opponent, same seed, same hand count -> outputs are bit-for-bit comparable
# to the existing cot_*_t0_s42 cells, but augmented with a per-decision
# logit-lens sidecar.
#
# Per cell:
#   1. Run 100 hands with --cot --capture-logprobs --logit-lens
#   2. Enrich (oracle posteriors)
#   3. analyze_logit_lens -> per-layer entropy curve, crystallization layer
#
# Outputs (parallel to the full-grid CoT outputs, NOT replacing them):
#   logs/cot_<tag>_t0_s42_informative_v2_logitlens.jsonl              (raw)
#   logs/cot_<tag>_t0_s42_informative_v2_logitlens_enriched.jsonl     (enriched)
#   logs/cot_<tag>_t0_s42_informative_v2_logitlens_logit_lens.jsonl   (sidecar)
#   results/tier1a_small_cot_logitlens/
#       logitlens_<tag>.json        (analyze_logit_lens output)
#       entropy_<tag>.png           (per-layer entropy plot, if matplotlib OK)
#       SUMMARY.md                  (per-model crystallization-layer table)
#
# FOLLOW-UP ANALYSIS (NOT in this script)
# ---------------------------------------
# Once the sidecars exist, the actual mechanistic answer comes from a
# cross-join script that loads (hand_id, decision_idx) keys from BOTH the
# enriched logs (with our new diagnostic flags: action_json_parsed /
# action_recognized / action_legal_in_context) and the logit-lens sidecars,
# then asks per failure mode: "for the N illegal-FOLD decisions, what does
# per_layer_top_tokens look like at the action-emission position?"
#
# That analysis lives in analysis/analyze_logit_lens_by_failure_mode.py
# (TODO; see end-of-script note). It runs in seconds once the sidecars exist
# and does NOT need GPU.
#
# COST ESTIMATE
# -------------
# Default: 3 models x 1 seed x 1 temp x 100 hands = 3 cells.
# Override SEEDS / TEMPS env vars to widen the grid (see CONFIG block below).
# Logit-lens hooks add ~10-30% per generation step + ~1-2 GB CPU memory for
# hidden-state buffers, so each cell takes ~1 h on one H100/A100.
#   3 cells (default)  -> ~3  h    captures all 339 Ministral illegal FOLDs
#                                     (s42 only) but only ~16 Llama / ~4 Qwen
#   6 cells (recommended, SEEDS="42" TEMPS="0.0 0.2")
#                      -> ~6  h    same 339 Ministral + ~50 Llama + ~10 Qwen
#  18 cells (full grid, SEEDS="42 123 456" TEMPS="0.0 0.2")
#                      -> ~18 h    full mirror of run_tier1a_small_cot.sh
# Sidecar files are 50-200 MB per cell (per-layer entropy + top-1 token
# strings; full hidden states are NOT persisted).
#
# =============================================================================
# tmux behaviour (copied verbatim from run_tier1a_small_cot.sh)
# =============================================================================
#
# Recommended workflow:
#   ssh user@gpu-box
#   cd <repo>/xpoker2026Extension
#   bash scripts/run_tier1a_small_cot_logitlens.sh
#
# When run OUTSIDE tmux, the script creates a tmux session named
# 'poker_tier1a_small_cot_logitlens' AND attaches you to it.
#   - Press Ctrl-B then D to detach (run continues, exit SSH safely).
#   - Reattach later with:  tmux attach -t poker_tier1a_small_cot_logitlens
#   - Kill the run with:    tmux kill-session -t poker_tier1a_small_cot_logitlens
#
# Set NO_TMUX=1 to force foreground execution (e.g. for CI / debugging).
#
# =============================================================================
# Qwen 3 thinking-mode safety check (Fix A guard) — same as parent script
# =============================================================================

set -euo pipefail

SESSION_NAME="poker_tier1a_small_cot_logitlens"

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
    echo "  Press Ctrl-B then D    (detaches; run continues in tmux)"
    echo "  Reattach later with:   tmux attach -t $SESSION_NAME"
    echo "  Abort the run with:    tmux kill-session -t $SESSION_NAME"
    echo
    exec tmux new-session -s "$SESSION_NAME" \
        "NO_TMUX=1 bash '$SCRIPT_PATH'; echo; echo '[tier1a_small_cot_logitlens finished -- press any key to close window]'; read -n1 -s"
fi

# ---- We're now inside tmux (or NO_TMUX=1). Do the work. ----

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if   [[ -f "venv/bin/activate"   ]]; then source "venv/bin/activate"
    elif [[ -f "../venv/bin/activate" ]]; then source "../venv/bin/activate"
    else echo "WARNING: no venv found at ./venv or ../venv -- using system python"
    fi
fi

LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results/tier1a_small_cot_logitlens}"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

# Default: single seed/temp for the cheapest mechanistic add-on. Seed 42
# specifically because Ministral's seed-42 CoT cells are where the 339
# illegal FOLDs live; t=0 because deterministic.
#
# Override examples:
#   SEEDS="42" TEMPS="0.0 0.2"            -> 6 cells (recommended; full
#                                            Ministral signal, partial Llama/Qwen)
#   SEEDS="42 123 456" TEMPS="0.0 0.2"    -> 18 cells (full-grid replica)
#
# Pass as space-separated strings; the script splits them into bash arrays.
read -r -a SEEDS <<< "${SEEDS:-42}"
read -r -a TEMPS <<< "${TEMPS:-0.0}"
HANDS_PER_CELL="${HANDS_PER_CELL:-100}"
OPPONENT_PRESET="informative_v2"
RUN_TAG="logitlens"  # filename suffix to keep these logs separate from the
                    # full-grid cot_*_t0_s42 cells.

n_cells=$(( ${#SEEDS[@]} * ${#TEMPS[@]} * 3 ))  # 3 models
echo "Cell grid: SEEDS=(${SEEDS[*]}) TEMPS=(${TEMPS[*]})  ->  $n_cells cells total"
echo "  Per-cell wall-clock: ~1 h on H100 (logit-lens hooks add ~10-30%)"
echo "  Estimated total wall-clock: ~$(( n_cells )) h on one H100/A100"
echo

temp_suffix() {
    case "$1" in
        0.0|0) echo "t0" ;;
        0.2)   echo "t02" ;;
        *)     echo "t${1//./}" ;;
    esac
}

# ---- Pre-flight: Fix A guard (qwen-8b native thinking OFF under --cot) ----
# Identical to the guard in run_tier1a_small_cot.sh; if Fix A is reverted,
# Qwen 3's parse rate craters from 100% to 1.3% (see results/
# tier1a_small_cot_pilot/SUMMARY.md). We refuse to start.
echo "=== Pre-flight: Fix A guard (qwen-8b enable_thinking=False under --cot) ==="
python <<'PY'
import sys
from unittest.mock import patch, MagicMock

from poker_env.config import MODEL_REGISTRY

cfg = MODEL_REGISTRY.get("qwen-8b")
assert cfg is not None, "qwen-8b not registered in MODEL_REGISTRY"
assert cfg.get("has_thinking_mode") is True, (
    "qwen-8b is missing has_thinking_mode=True in MODEL_REGISTRY -- aborting"
)

mock_tok = MagicMock()
mock_tok.pad_token = "<pad>"; mock_tok.eos_token = "<eos>"
mock_model = MagicMock()
mock_model.eval.return_value = mock_model

with patch("poker_env.agents.hf_agent.AutoTokenizer.from_pretrained",
           return_value=mock_tok), \
     patch("poker_env.agents.hf_agent.AutoModelForCausalLM.from_pretrained",
           return_value=mock_model):
    from poker_env.agents.hf_agent import HFAgent
    agent = HFAgent(model_id="qwen-8b", cot_mode=True)

agent_cfg = agent.get_config()
print(f"  qwen-8b enable_thinking under cot_mode=True: {agent_cfg['enable_thinking']}")
assert agent_cfg["enable_thinking"] is False, (
    "FIX A REGRESSION: enable_thinking is not False under cot_mode=True. "
    "Aborting before wasting GPU time."
)
print("  -> Fix A in place. OK.")
PY
echo

# ---- Pre-flight 2: confirm logit-lens module imports cleanly ----
# A common breakage mode: missing torch / transformers, or the layers/lm_head
# accessor failing on a model architecture we haven't tested. Catch this here
# rather than 30 minutes into the first run.
echo "=== Pre-flight: logit-lens module import ==="
python <<'PY'
from poker_env.interp.logit_lens import LogitLensExtractor
print(f"  LogitLensExtractor imported OK from poker_env.interp.logit_lens")
print(f"  (architecture support: model.model.layers + model.lm_head;")
print(f"   verified for Llama / Mistral / Qwen.)")
PY
echo

# ---- Per-model run/enrich/analyze loop ----
# Same model triple as the full grid, in the same order so logs sort cleanly.
MODELS=(
    "llama-8b:llama8b:models--meta-llama--Llama-3.1-8B-Instruct"
    "qwen-8b:qwen8b:models--Qwen--Qwen3-8B"
    "ministral-8b:ministral8b:models--mistralai--Ministral-8B-Instruct-2410"
)

PURGE_HF_CACHE_AFTER_MODEL="${PURGE_HF_CACHE_AFTER_MODEL:-0}"

run_model() {
    local short="$1"
    local tag="$2"
    local started_at
    started_at="$(date +%s)"

    echo
    echo "############################################################"
    echo "## MODEL: $short  (tag=$tag)  [tier1a_small_cot_logitlens]"
    echo "## Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "############################################################"

    # ---- Phase 1: 1 cell (s42 x t=0) with --cot --capture-logprobs --logit-lens ----
    for seed in "${SEEDS[@]}"; do
        for temp in "${TEMPS[@]}"; do
            local tsuf
            tsuf="$(temp_suffix "$temp")"
            local out="${LOGS_DIR}/cot_${tag}_${tsuf}_s${seed}_${OPPONENT_PRESET}_${RUN_TAG}.jsonl"
            local sidecar="${out%.jsonl}_logit_lens.jsonl"
            if [[ -f "$out" ]] || [[ -f "${out}.gz" ]]; then
                echo "  [skip] $out already exists"
                continue
            fi
            echo "  [run]  $out  (seed=$seed temp=$temp hands=$HANDS_PER_CELL --cot --logit-lens)"
            echo "         sidecar -> $sidecar"
            python run_experiment.py \
                --agent hf \
                --hf-model "$short" \
                --opponent threshold \
                --opponent-preset "$OPPONENT_PRESET" \
                --hands "$HANDS_PER_CELL" \
                --seed "$seed" \
                --temperature "$temp" \
                --elicit-beliefs \
                --cot \
                --capture-logprobs \
                --logit-lens \
                --out "$out" \
                -v
        done
    done

    # ---- Phase 2: enrich logs with oracle posteriors ----
    echo
    echo "  [enrich] $tag"
    shopt -s nullglob
    for f in "${LOGS_DIR}"/cot_${tag}_*_${OPPONENT_PRESET}_${RUN_TAG}.jsonl; do
        [[ "$f" == *_enriched.jsonl ]] && continue
        local out_enr="${f%.jsonl}_enriched.jsonl"
        if [[ -f "$out_enr" ]] || [[ -f "${out_enr}.gz" ]]; then
            echo "    [skip] $out_enr already exists"
            continue
        fi
        echo "    [enrich] $f -> $out_enr"
        python -m analysis.build_dataset \
            "$f" "$out_enr" \
            --opponent "$OPPONENT_PRESET"
    done

    # ---- Phase 3: per-cell logit-lens analysis ----
    # analyze_logit_lens.py operates on a single sidecar at a time; we have
    # exactly one sidecar per cell here.
    echo
    echo "  [analyze_logit_lens] $tag"
    shopt -s nullglob
    for sidecar in "${LOGS_DIR}"/cot_${tag}_*_${OPPONENT_PRESET}_${RUN_TAG}_logit_lens.jsonl; do
        local stem
        stem="$(basename "${sidecar%_logit_lens.jsonl}")"
        local cell_label="${stem#cot_}"  # strip prefix for cleaner labels
        local json_out="${RESULTS_DIR}/logitlens_${cell_label}.json"
        local plot_out="${RESULTS_DIR}/entropy_${cell_label}.png"
        echo "    [analyze] $sidecar"
        # --plot is best-effort: if matplotlib isn't installed, the script
        # prints a warning and skips the plot but still writes the JSON.
        python -m analysis.analyze_logit_lens \
            "$sidecar" \
            --json-out "$json_out" \
            --plot "$plot_out" \
            || echo "    (analyze_logit_lens failed for $sidecar)"

        # ---- Phase 3b: failure-mode cross-join (the mechanistic answer) ----
        # Joins this sidecar with the matching enriched log, buckets decisions
        # by failure mode (clean / illegal_fold / illegal_other / ...), and
        # reports per-layer mapped action group at the action-emission token.
        # This is what tells us whether early layers said CHECK while the
        # final layer said FOLD on the rescued-by-fallback decisions.
        local enriched="${sidecar%_logit_lens.jsonl}_enriched.jsonl"
        if [[ ! -f "$enriched" && -f "${enriched}.gz" ]]; then
            enriched="${enriched}.gz"
        fi
        if [[ -f "$enriched" ]]; then
            local fm_json="${RESULTS_DIR}/by_failure_mode_${cell_label}.json"
            python -m analysis.analyze_logit_lens_by_failure_mode \
                --logit-lens-sidecar "$sidecar" \
                --enriched-log "$enriched" \
                --label "$cell_label" \
                --json-out "$fm_json" \
                --md-out "${RESULTS_DIR}/BY_FAILURE_MODE.md" \
                || echo "    (analyze_logit_lens_by_failure_mode failed for $sidecar)"
        else
            echo "    (no enriched log for $sidecar; skipping failure-mode join)"
        fi
    done

    local elapsed=$(( $(date +%s) - started_at ))
    echo
    echo "## DONE $short in ${elapsed}s"
    echo "## Results: ${RESULTS_DIR}/logitlens_*${tag}*.json (+ entropy_*.png)"
    echo "##          ${RESULTS_DIR}/by_failure_mode_*${tag}*.json (+ BY_FAILURE_MODE.md)"
    echo
}

# ---- Run each model in turn ----
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
# Phase 4: cross-model summary
# =============================================================================
# A small Python rollup that reads logitlens_<cell>.json for each cell and
# writes a single SUMMARY.md table comparing crystallization layers and
# entropy curves across the grid. This is the cheap descriptive view -- the
# *interesting* mechanistic question (do illegal-FOLD decisions crystallize
# earlier on the wrong token?) is answered by the per-cell BY_FAILURE_MODE.md
# produced in Phase 3b above (analyze_logit_lens_by_failure_mode.py).
echo
echo "############################################################"
echo "## Phase 4: cross-model summary"
echo "############################################################"

python <<'PY'
import json
import os
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results/tier1a_small_cot_logitlens"))

# Auto-discover all per-cell descriptive JSONs written by Phase 3.
# Filenames are logitlens_<cell_label>.json where cell_label is everything
# after the cot_ prefix (e.g. ministral8b_t0_s42_informative_v2_logitlens).
cell_paths = sorted(RESULTS_DIR.glob("logitlens_*.json"))

rows = []
for p in cell_paths:
    cell_label = p.stem[len("logitlens_"):]
    with open(p) as f:
        d = json.load(f)
    cl = d.get("crystallization_layer") or {}
    rows.append((
        cell_label,
        d.get("num_records", 0),
        d.get("num_layers"),
        cl.get("mean"),
        cl.get("median"),
    ))

print()
print(f"{'Cell':<48} | {'records':>8} | {'layers':>6} | {'crys (mean)':>11} | {'(median)':>9}")
print("-" * 100)
for cell, n, layers, cmean, cmed in rows:
    layers_s = str(layers) if layers is not None else "-"
    cmean_s = f"{cmean:.1f}" if cmean is not None else "-"
    cmed_s  = f"{cmed:.1f}"  if cmed  is not None else "-"
    print(f"{cell:<48} | {str(n):>8} | {layers_s:>6} | {cmean_s:>11} | {cmed_s:>9}")

with open(RESULTS_DIR / "SUMMARY.md", "w") as f:
    f.write("# Tier 1A.small CoT + logit-lens — descriptive summary\n\n")
    f.write("Per-cell logit-lens descriptive stats (from `analyze_logit_lens.py`). ")
    f.write("Grid size depends on SEEDS / TEMPS env vars — default is 1 seed × 1 temp ")
    f.write("× 3 models = 3 cells. See script header for grid options.\n\n")
    f.write("| Cell | n records | Layers | Crystallization (mean) | (median) |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for cell, n, layers, cmean, cmed in rows:
        cmean_s = f"{cmean:.1f}" if cmean is not None else "—"
        cmed_s  = f"{cmed:.1f}"  if cmed  is not None else "—"
        f.write(f"| `{cell}` | {n} | {layers if layers is not None else '—'} "
                f"| {cmean_s} | {cmed_s} |\n")
    f.write("\n*Crystallization layer* = earliest layer from which the top-1 ")
    f.write("token never changes through to the final layer. Lower = the model ")
    f.write("'decided' earlier. Higher = the model is still revising late.\n\n")
    f.write("## Companion artifacts\n\n")
    f.write("- `entropy_<cell>.png` — per-layer mean entropy plot for each cell.\n")
    f.write("- `logitlens_<cell>.json` — full per-cell stats from `analyze_logit_lens.py`.\n")
    f.write("- `by_failure_mode_<cell>.json` + `BY_FAILURE_MODE.md` —\n")
    f.write("  the mechanistic answer: per-bucket (clean / illegal_fold / ...)\n")
    f.write("  per-layer mapped action group at the action-emission token.\n")
    f.write("- `logs/cot_<cell>_logit_lens.jsonl` — raw sidecar (per-decision\n")
    f.write("  per-layer top-1 tokens + entropies).\n\n")
    f.write("## Mechanistic question this run targets\n\n")
    f.write("> Do hidden states encode `CHECK_OR_CALL` even when the model verbalizes\n")
    f.write("> `FOLD` into a free-check spot?\n\n")
    f.write("That question is answered by Phase 3b's `BY_FAILURE_MODE.md` — see the\n")
    f.write("`illegal_fold` bucket per cell. If early layers favour CHECK while only\n")
    f.write("the final layers cross to FOLD, that's a verbalization-stage failure;\n")
    f.write("if FOLD dominates from layer 0, the model is FOLD-committed top to bottom.\n")
print()
print(f"Wrote: {RESULTS_DIR/'SUMMARY.md'}")
PY

echo
echo "============================================================"
echo "Tier 1A.small CoT + logit-lens COMPLETE."
echo "  Logs:    $LOGS_DIR/cot_{llama8b,qwen8b,ministral8b}_*_${OPPONENT_PRESET}_${RUN_TAG}*"
echo "  Sidecars:$LOGS_DIR/cot_{...}_${RUN_TAG}_logit_lens.jsonl"
echo "  Results: $RESULTS_DIR/"
echo "  Read:    $RESULTS_DIR/SUMMARY.md            (cross-cell descriptive)"
echo "  Read:    $RESULTS_DIR/BY_FAILURE_MODE.md    (mechanistic answer)"
echo "============================================================"
