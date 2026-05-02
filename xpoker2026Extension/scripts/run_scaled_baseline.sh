#!/usr/bin/env bash
# Scaled non-CoT baseline runs (Tier 1A — before CoT).
#
# Two sub-tiers, both using the paper's env / opponent / belief schema so that
# every run is directly comparable to poker26's published numbers:
#
#   large (~70B class):  llama-70b, llama-3.3-70b, qwen-72b
#   small (8B class):    llama-8b, qwen-8b (Qwen3-8B), ministral-8b
#
# Grid per sub-tier: 3 models x 3 seeds (42, 123, 456) x 2 temps (0.0, 0.2)
#   = 18 cells per sub-tier (36 cells total when running both)
#
# Hand budgets:
#   llama-70b (anchor): 500 hands/cell  (target ~3x paper sample for tighter CIs)
#   other 70B-class:    350 hands/cell  (paper-quality)
#   8B-class:           100 hands/cell  (8B inference is fast; original 8B sanity
#                                        was 50 hands so 100 already doubles it)
#
# IMPORTANT: Qwen3-8B has a built-in "thinking mode" that the chat template
# enables by default. The HFAgent gates this on --cot, so non-CoT runs of
# qwen-8b correctly disable thinking. See poker_env/agents/hf_agent.py.
#
# Total compute estimate:
#   large: 500*6 + 350*12 = 7,200 hands of 70B HF compute (overnight on 1xH100)
#   small: 100*18         = 1,800 hands of 8B HF compute  (a few hours)
#
# Usage:
#   bash scripts/run_scaled_baseline.sh                  # all 36 cells (both tiers)
#   bash scripts/run_scaled_baseline.sh large            # only 70B tier
#   bash scripts/run_scaled_baseline.sh small            # only 8B tier
#   bash scripts/run_scaled_baseline.sh llama-70b        # one model only
#   bash scripts/run_scaled_baseline.sh qwen-72b
#   bash scripts/run_scaled_baseline.sh llama-3.3-70b
#   bash scripts/run_scaled_baseline.sh llama-8b
#   bash scripts/run_scaled_baseline.sh qwen-8b
#   bash scripts/run_scaled_baseline.sh ministral-8b
#
# RECOMMENDED for SSH workflows: use the per-tier wrappers instead. They
# auto-launch in tmux and run analysis after each model so you get incremental
# results even if the run is interrupted:
#   bash scripts/run_tier1a_small.sh   # 8B tier with tmux + per-model analysis
#   (run_tier1a_large.sh — coming next)

set -euo pipefail

# Activate the venv if not already active. Adjust path if running on a cluster.
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

LOGS_DIR="${LOGS_DIR:-logs}"
mkdir -p "$LOGS_DIR"

# Paper anchor: Llama 3.1 70B at 500 hands/cell (target ~2,000 valid beliefs, 3x paper).
LLAMA_70B_HANDS="${LLAMA_70B_HANDS:-500}"

# Cross-family 70B comparators at paper-quality 350 hands/cell.
CROSS_FAMILY_HANDS="${CROSS_FAMILY_HANDS:-350}"

# Small-tier (8B) hand budget. Cheap to run; 100 hands gives ~250-400 parsed
# beliefs per cell which is enough for clustered bootstrap CIs.
SMALL_TIER_HANDS="${SMALL_TIER_HANDS:-100}"

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
# $3 = filename-safe model tag (e.g. llama70b, qwen72b, llama33-70b, llama8b)
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

# ---- Tier 1A.large (~70B class) ----
case "$target" in
    llama-70b|large|all)
        run_model_grid "llama-70b"     "$LLAMA_70B_HANDS"   "llama70b"
        ;;
esac
case "$target" in
    qwen-72b|large|all)
        run_model_grid "qwen-72b"      "$CROSS_FAMILY_HANDS" "qwen72b"
        ;;
esac
case "$target" in
    llama-3.3-70b|large|all)
        run_model_grid "llama-3.3-70b" "$CROSS_FAMILY_HANDS" "llama33-70b"
        ;;
esac

# ---- Tier 1A.small (8B class, parameter-matched) ----
case "$target" in
    llama-8b|small|all)
        run_model_grid "llama-8b"      "$SMALL_TIER_HANDS"   "llama8b"
        ;;
esac
case "$target" in
    qwen-8b|small|all)
        run_model_grid "qwen-8b"       "$SMALL_TIER_HANDS"   "qwen8b"
        ;;
esac
case "$target" in
    ministral-8b|small|all)
        run_model_grid "ministral-8b"  "$SMALL_TIER_HANDS"   "ministral8b"
        ;;
esac

echo ""
echo "Done. Next steps:"
echo "  bash scripts/enrich_scaled_baseline.sh"
echo "  bash scripts/analyze_scaled_baseline.sh"
