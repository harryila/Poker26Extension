#!/usr/bin/env bash
# =============================================================================
# Tier 1A.large CoT + logit-lens — mechanistic add-on for 70B models
# =============================================================================
#
# Mirrors run_tier1a_small_cot_logitlens.sh but for 70B-class models.
# Runs --cot --capture-logprobs --logit-lens with the model loaded once per
# model, then explicitly unloaded before the next model loads.
#
# Default grid: 1 seed (42) x 1 temp (0.0) for cost control.
# Override via env vars:
#   SEEDS="42 123 456" TEMPS="0.0 0.2" bash scripts/run_tier1a_large_cot_logitlens.sh
#
# Models (~70B class):
#   - llama-70b      (meta-llama/Llama-3.1-70B-Instruct)
#   - llama-3.3-70b  (meta-llama/Llama-3.3-70B-Instruct)
#   - qwen-72b       (Qwen/Qwen2.5-72B-Instruct)
#
# Hand budget:
#   llama-70b      : 500 hands/cell  (anchor)
#   llama-3.3-70b  : 350 hands/cell
#   qwen-72b       : 350 hands/cell
#
# Cost estimate (default 3 cells):
#   ~3-6 h on one H100 80GB. Logit-lens hooks add ~10-30% per generation.
#   Sidecar files are 50-300 MB per cell.
#
# Outputs:
#   logs/cot_<tag>_<temp>_s<seed>_informative_v2_logitlens.jsonl              (raw)
#   logs/cot_<tag>_<temp>_s<seed>_informative_v2_logitlens_enriched.jsonl     (enriched)
#   logs/cot_<tag>_<temp>_s<seed>_informative_v2_logitlens_logit_lens.jsonl   (sidecar)
#   results/tier1a_large_cot_logitlens/
#       logitlens_<cell>.json
#       entropy_<cell>.png
#       by_failure_mode_<cell>.json
#       BY_FAILURE_MODE.md
#       SUMMARY.md
#
# Failure isolation: models 1..N-1 results survive if model N crashes.
# Re-running skips cells whose output files already exist.
#
# =============================================================================
# tmux behaviour
# =============================================================================
#
#   bash scripts/run_tier1a_large_cot_logitlens.sh
#
# Auto-creates tmux session 'poker_tier1a_large_cot_logitlens':
#   - Ctrl-B then D to detach
#   - tmux attach -t poker_tier1a_large_cot_logitlens to reattach
#   - tmux kill-session -t poker_tier1a_large_cot_logitlens to abort
#
# Set NO_TMUX=1 for foreground execution.
#
# =============================================================================
# Note: 70B inference needs >= 80 GB total VRAM. Models load one at a time.
# =============================================================================

set -euo pipefail

SESSION_NAME="poker_tier1a_large_cot_logitlens"

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
        "NO_TMUX=1 bash '$SCRIPT_PATH'; echo; echo '[tier1a_large_cot_logitlens finished -- press any key to close window]'; read -n1 -s"
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
RESULTS_DIR="${RESULTS_DIR:-results/tier1a_large_cot_logitlens}"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

read -r -a SEEDS <<< "${SEEDS:-42}"
read -r -a TEMPS <<< "${TEMPS:-0.0}"
OPPONENT_PRESET="informative_v2"
RUN_TAG="logitlens"

LLAMA_70B_HANDS="${LLAMA_70B_HANDS:-500}"
CROSS_FAMILY_HANDS="${CROSS_FAMILY_HANDS:-350}"

n_cells=$(( ${#SEEDS[@]} * ${#TEMPS[@]} * 3 ))
echo "Cell grid: SEEDS=(${SEEDS[*]}) TEMPS=(${TEMPS[*]})  ->  $n_cells cells total"
echo "  Estimated wall-clock: ~$(( n_cells * 2 )) h on one H100/A100"
echo

# ---- Pre-flight: GPU / VRAM check ----
echo "=== Pre-flight: GPU / VRAM check ==="
python <<'PY'
try:
    import torch
    if not torch.cuda.is_available():
        print("WARNING: torch.cuda.is_available() is False. 70B inference needs CUDA.")
    else:
        n = torch.cuda.device_count()
        total_gb = 0.0
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            gb = props.total_memory / (1024**3)
            total_gb += gb
            print(f"  GPU {i}: {props.name}  {gb:.1f} GB")
        print(f"  TOTAL VRAM: {total_gb:.1f} GB across {n} device(s)")
        if total_gb < 80:
            print(f"  WARNING: 70B Instruct typically needs >= 80 GB VRAM.")
except ImportError:
    print("WARNING: torch is not installed; skipping VRAM check.")
PY
echo

# ---- Pre-flight: logit-lens module import ----
echo "=== Pre-flight: logit-lens module import ==="
python <<'PY'
from poker_env.interp.logit_lens import LogitLensExtractor
print("  LogitLensExtractor imported OK from poker_env.interp.logit_lens")
print("  (architecture support: model.model.layers + model.lm_head;")
print("   verified for Llama 3.x / Qwen 2.x.)")
PY
echo

# ---- Per-model run/enrich/analyze loop ----
# Format: "<short_name>:<filename_tag>:<hands>"
MODELS=(
    "llama-70b:llama70b:${LLAMA_70B_HANDS}"
    "llama-3.3-70b:llama33-70b:${CROSS_FAMILY_HANDS}"
    "qwen-72b:qwen72b:${CROSS_FAMILY_HANDS}"
)

temp_suffix() {
    case "$1" in
        0.0|0) echo "t0" ;;
        0.2)   echo "t02" ;;
        *)     echo "t${1//./}" ;;
    esac
}

run_model() {
    local short="$1"
    local tag="$2"
    local hands="$3"
    local started_at
    started_at="$(date +%s)"

    echo
    echo "############################################################"
    echo "## MODEL: $short  (tag=$tag, hands=$hands)  [tier1a_large_cot_logitlens]"
    echo "## Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "############################################################"

    # ---- Phase 1: run cells (single process, model loaded once) ----
    local seeds_csv
    seeds_csv="$(IFS=,; echo "${SEEDS[*]}")"
    local temps_csv
    temps_csv="$(IFS=,; echo "${TEMPS[*]}")"
    echo "  [run]  multi-cell: seeds=${seeds_csv} temps=${temps_csv} hands=$hands --cot --logit-lens"
    python run_experiment.py \
        --agent hf \
        --hf-model "$short" \
        --opponent threshold \
        --opponent-preset "$OPPONENT_PRESET" \
        --hands "$hands" \
        --seeds "$seeds_csv" \
        --temps "$temps_csv" \
        --out-dir "$LOGS_DIR" \
        --out-prefix "cot_${tag}" \
        --elicit-beliefs \
        --cot \
        --capture-logprobs \
        --logit-lens \
        -v

    # ---- Phase 2: enrich logs with oracle posteriors ----
    echo
    echo "  [enrich] $tag"
    shopt -s nullglob
    for f in "${LOGS_DIR}"/cot_${tag}_*_${OPPONENT_PRESET}.jsonl; do
        [[ "$f" == *_enriched.jsonl ]] && continue
        [[ "$f" == *_logit_lens.jsonl ]] && continue
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

    # ---- Phase 3: per-cell logit-lens analysis ----
    echo
    echo "  [analyze_logit_lens] $tag"
    shopt -s nullglob
    for sidecar in "${LOGS_DIR}"/cot_${tag}_*_${OPPONENT_PRESET}_logit_lens.jsonl; do
        local stem
        stem="$(basename "${sidecar%_logit_lens.jsonl}")"
        local cell_label="${stem#cot_}"
        local json_out="${RESULTS_DIR}/logitlens_${cell_label}.json"
        local plot_out="${RESULTS_DIR}/entropy_${cell_label}.png"
        echo "    [analyze] $sidecar"
        python -m analysis.analyze_logit_lens \
            "$sidecar" \
            --json-out "$json_out" \
            --plot "$plot_out" \
            || echo "    (analyze_logit_lens failed for $sidecar)"

        # Failure-mode cross-join
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
    echo
}

# ---- Run each model in turn ----
for entry in "${MODELS[@]}"; do
    short="${entry%%:*}"
    rest="${entry#*:}"
    tag="${rest%%:*}"
    hands="${rest#*:}"
    run_model "$short" "$tag" "$hands"
done

# ---- Cross-model summary ----
echo
echo "############################################################"
echo "## Cross-model summary"
echo "############################################################"

python <<'PY'
import json, os
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results/tier1a_large_cot_logitlens"))

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
    f.write("# Tier 1A.large CoT + logit-lens -- descriptive summary\n\n")
    f.write("| Cell | n records | Layers | Crystallization (mean) | (median) |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for cell, n, layers, cmean, cmed in rows:
        cmean_s = f"{cmean:.1f}" if cmean is not None else "---"
        cmed_s  = f"{cmed:.1f}"  if cmed  is not None else "---"
        f.write(f"| `{cell}` | {n} | {layers if layers is not None else '---'} "
                f"| {cmean_s} | {cmed_s} |\n")

print()
print(f"Wrote: {RESULTS_DIR/'SUMMARY.md'}")
PY

echo
echo "============================================================"
echo "Tier 1A.large CoT + logit-lens COMPLETE."
echo "  Logs:     $LOGS_DIR/cot_{llama70b,llama33-70b,qwen72b}_*"
echo "  Sidecars: $LOGS_DIR/cot_*_logit_lens.jsonl"
echo "  Results:  $RESULTS_DIR/"
echo "  Read:     $RESULTS_DIR/SUMMARY.md"
echo "  Read:     $RESULTS_DIR/BY_FAILURE_MODE.md"
echo "============================================================"
