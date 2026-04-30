#!/usr/bin/env bash
# Scaled non-CoT baseline runs (Tier 1.5 — before CoT).
#
# Produces 18 experiment logs spanning 3 models x 3 seeds x 2 temperatures,
# all against the same threshold opponent (informative_v2) used in the paper.
#
# Total: 7,200 hands of HF compute. Parallelize across GPUs by model.
#
# Usage:
#   bash scripts/run_scaled_baseline.sh                   # all 18 jobs sequentially
#   bash scripts/run_scaled_baseline.sh llama-70b         # only the anchor
#   bash scripts/run_scaled_baseline.sh qwen-72b          # only Qwen
#   bash scripts/run_scaled_baseline.sh llama-3.3-70b     # only Llama 3.3

set -euo pipefail

# Activate the venv if not already active. Adjust path if running on a cluster.
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

LOGS_DIR="${LOGS_DIR:-logs}"
mkdir -p "$LOGS_DIR"

# Paper anchor: Llama 3.1 70B at 500 hands/cell (target ~2,000 valid beliefs, 3x paper).
LLAMA_70B_HANDS="${LLAMA_70B_HANDS:-500}"

# Cross-family comparators at paper-quality 350 hands/cell.
CROSS_FAMILY_HANDS="${CROSS_FAMILY_HANDS:-350}"

SEEDS=(42 123 456)
TEMPS=(0.0 0.2)
OPPONENT_PRESET="informative_v2"

# Map temperature to a filename-safe suffix (0.0 -> t0, 0.2 -> t02).
temp_suffix() {
    local t="$1"
    case "$t" in
        0.0|0) echo "t0" ;;
        0.2)   echo "t02" ;;
        *)     echo "t${t//./}" ;;
    esac
}

# Run all 6 cells for one model.
# $1 = short model name (e.g. llama-70b)
# $2 = hand count per cell
# $3 = filename-safe model tag (e.g. 70b, qwen72b, llama33-70b)
run_model_grid() {
    local short_name="$1"
    local hands="$2"
    local tag="$3"
    echo "=== Running grid for $short_name ($hands hands/cell, 6 cells) ==="
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
                --hf-model "$short_name" \
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
}

target="${1:-all}"

case "$target" in
    llama-70b|all)
        run_model_grid "llama-70b"     "$LLAMA_70B_HANDS"   "llama70b"
        ;;
esac
case "$target" in
    qwen-72b|all)
        run_model_grid "qwen-72b"      "$CROSS_FAMILY_HANDS" "qwen72b"
        ;;
esac
case "$target" in
    llama-3.3-70b|all)
        run_model_grid "llama-3.3-70b" "$CROSS_FAMILY_HANDS" "llama33-70b"
        ;;
esac

echo ""
echo "Done. Next steps:"
echo "  bash scripts/enrich_scaled_baseline.sh"
echo "  bash scripts/analyze_scaled_baseline.sh"
