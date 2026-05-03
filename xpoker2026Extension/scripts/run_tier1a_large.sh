#!/usr/bin/env bash
# =============================================================================
# Tier 1A.large — Scaled non-CoT baseline at 70B class
# =============================================================================
#
# What this script does (in order, per model):
#   1. Run all 6 cells: 3 seeds (42, 123, 456) x 2 temps (0.0, 0.2)
#   2. Enrich each cell's log with oracle posteriors (StrategyAware + CardOnly)
#   3. Run PCE distribution + update coherence on that model's pooled logs
#   4. Move on to the next model
#
# Models (~70B class, parameter-matched within ~3%):
#   - llama-70b      (meta-llama/Llama-3.1-70B-Instruct)   <-- paper anchor
#   - llama-3.3-70b  (meta-llama/Llama-3.3-70B-Instruct)
#   - qwen-72b       (Qwen/Qwen2.5-72B-Instruct)
#
# Hand budget:
#   llama-70b      : 500 hands/cell  (anchor, ~3x paper sample for tighter CIs)
#   llama-3.3-70b  : 350 hands/cell  (paper-quality)
#   qwen-72b       : 350 hands/cell  (paper-quality)
#   Total: 500*6 + 350*6 + 350*6 = 7,200 hands of 70B HF compute.
#   Wall-clock: roughly overnight on one H100 80GB (or multi-GPU H100/A100).
#
# Outputs (written incrementally as each model completes):
#   logs/scaled_<tag>_<temp>_s<seed>_informative_v2.jsonl            (raw)
#   logs/scaled_<tag>_<temp>_s<seed>_informative_v2_enriched.jsonl   (enriched)
#   results/tier1a_large/pce_<tag>_records.csv                       (per-decision)
#   results/tier1a_large/pce_<tag>_summary.csv                       (clustered bootstrap)
#   results/tier1a_large/uc_<tag>.csv + uc_<tag>_summary.json
#   results/tier1a_large/pce_pool_summary.csv                        (cross-model pool)
#
# Failure isolation: if model N crashes, models 1..N-1 results are already on
# disk. Re-running this script will skip any cell whose log already exists.
#
# =============================================================================
# tmux behaviour
# =============================================================================
#
# Recommended workflow:
#   ssh user@gpu-box
#   cd <repo>/xpoker2026Extension
#   bash scripts/run_tier1a_large.sh
#
# When run OUTSIDE tmux, the script creates a tmux session named
# 'poker_tier1a_large' AND attaches you to it, so you see the run starting.
# The script does NOT auto-detach you — that is your call:
#   - Press Ctrl-B then D to detach (run continues, you exit tmux back to
#     your shell, and you can safely close the SSH connection).
#   - Reattach later with:  tmux attach -t poker_tier1a_large
#   - Kill the run with:    tmux kill-session -t poker_tier1a_large
#
# To force foreground execution without tmux (e.g. for CI / debugging), set
# NO_TMUX=1 before invoking.
#
# =============================================================================
# Note: 70B inference needs >= 80 GB total VRAM (one H100 80GB, or 2x A100 40GB
# with device_map="auto"). Models load one at a time (sequential) so peak VRAM
# is one 70B model loaded, not three.
# =============================================================================

set -euo pipefail

SESSION_NAME="poker_tier1a_large"

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
        "NO_TMUX=1 bash '$SCRIPT_PATH'; echo; echo '[tier1a_large finished — press any key to close window]'; read -n1 -s"
fi

# ---- We're now inside tmux (or NO_TMUX=1 was set). Do the work. ----

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source venv/bin/activate
    elif [[ -f "../venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source ../venv/bin/activate
    else
        echo "WARNING: no venv found at ./venv or ../venv — using system python"
    fi
fi

LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results/tier1a_large}"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

SEEDS=(42 123 456)
TEMPS=(0.0 0.2)
OPPONENT_PRESET="informative_v2"

# Per-model hand budgets (anchor gets a larger budget for tighter CIs).
LLAMA_70B_HANDS="${LLAMA_70B_HANDS:-500}"
CROSS_FAMILY_HANDS="${CROSS_FAMILY_HANDS:-350}"

temp_suffix() {
    case "$1" in
        0.0|0) echo "t0" ;;
        0.2)   echo "t02" ;;
        *)     echo "t${1//./}" ;;
    esac
}

# ---- Pre-flight: report VRAM / GPU info to make GPU-fit issues obvious early ----
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
            print(f"           You may need 4-bit quantization or multiple GPUs.")
except ImportError:
    print("WARNING: torch is not installed; skipping VRAM check.")
PY
echo

# ---- Per-model run/enrich/analyze loop ----
# Format: "<short_name>:<filename_tag>:<hands>"
MODELS=(
    "llama-70b:llama70b:${LLAMA_70B_HANDS}"
    "llama-3.3-70b:llama33-70b:${CROSS_FAMILY_HANDS}"
    "qwen-72b:qwen72b:${CROSS_FAMILY_HANDS}"
)

run_model() {
    local short="$1"
    local tag="$2"
    local hands="$3"
    local started_at
    started_at="$(date +%s)"

    echo
    echo "############################################################"
    echo "## MODEL: $short  (tag=$tag, hands=$hands)"
    echo "## Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "############################################################"

    # ---- Phase 1: run all 6 cells for this model ----
    for seed in "${SEEDS[@]}"; do
        for temp in "${TEMPS[@]}"; do
            local tsuf
            tsuf="$(temp_suffix "$temp")"
            local out="${LOGS_DIR}/scaled_${tag}_${tsuf}_s${seed}_${OPPONENT_PRESET}.jsonl"
            if [[ -f "$out" ]]; then
                echo "  [skip] $out already exists"
                continue
            fi
            echo "  [run]  $out  (seed=$seed temp=$temp hands=$hands)"
            python run_experiment.py \
                --agent hf \
                --hf-model "$short" \
                --opponent threshold \
                --opponent-preset "$OPPONENT_PRESET" \
                --hands "$hands" \
                --seed "$seed" \
                --temperature "$temp" \
                --elicit-beliefs \
                --capture-logprobs \
                --out "$out" \
                -v
        done
    done

    # ---- Phase 2: enrich logs with oracle posteriors ----
    echo
    echo "  [enrich] $tag"
    shopt -s nullglob
    for f in "${LOGS_DIR}"/scaled_${tag}_*_${OPPONENT_PRESET}.jsonl; do
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

    # ---- Phase 3: per-model analysis ----
    echo
    echo "  [analyze] $tag"
    local files=( "${LOGS_DIR}"/scaled_${tag}_*_enriched.jsonl )
    if [[ "${#files[@]}" -eq 0 ]]; then
        echo "    [skip] no enriched logs for $tag"
        return
    fi
    python -m analysis.compute_pce_distribution \
        "${files[@]}" \
        --output-records "${RESULTS_DIR}/pce_${tag}_records.csv" \
        --output-summary "${RESULTS_DIR}/pce_${tag}_summary.csv" \
        --bootstrap 1000 --seed 42 --clustered

    python -m analysis.compute_update_coherence \
        "${files[@]}" \
        --output "${RESULTS_DIR}/uc_${tag}.csv" \
        --output-summary "${RESULTS_DIR}/uc_${tag}_summary.json"

    local elapsed=$(( $(date +%s) - started_at ))
    echo
    echo "## DONE $short in ${elapsed}s"
    echo "## Results: ${RESULTS_DIR}/pce_${tag}_*.csv, uc_${tag}.*"
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

# ---- Final cross-model pool ----
echo
echo "############################################################"
echo "## POOLED ANALYSIS — all 70B-class models"
echo "############################################################"
shopt -s nullglob
all_files=(
    "${LOGS_DIR}"/scaled_llama70b_*_enriched.jsonl
    "${LOGS_DIR}"/scaled_llama33-70b_*_enriched.jsonl
    "${LOGS_DIR}"/scaled_qwen72b_*_enriched.jsonl
)
if [[ "${#all_files[@]}" -gt 0 ]]; then
    python -m analysis.compute_pce_distribution \
        "${all_files[@]}" \
        --output-records "${RESULTS_DIR}/pce_pool_records.csv" \
        --output-summary "${RESULTS_DIR}/pce_pool_summary.csv" \
        --bootstrap 1000 --seed 42 --clustered
    echo
    echo "Pooled summary: ${RESULTS_DIR}/pce_pool_summary.csv"
fi

echo
echo "============================================================"
echo "Tier 1A.large COMPLETE."
echo "  Logs:    $LOGS_DIR/scaled_{llama70b,llama33-70b,qwen72b}_*"
echo "  Results: $RESULTS_DIR/"
echo "  Compare each model's mean_js_to_sa to the paper anchor 0.014."
echo "============================================================"
