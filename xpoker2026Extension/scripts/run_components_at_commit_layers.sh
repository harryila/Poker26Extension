#!/usr/bin/env bash
# =============================================================================
# Component decomposition at the COMMITMENT layer (L*+1) for each model.
# =============================================================================
# We already have Llama L=15 (Phase J §15c). This adds the analogous cells
# for Ministral and Qwen so the cross-model commitment-layer story is
# complete: at the commit layer, residual flips fully but neither sublayer
# nor any single head dominates locally — the signal arrives via residual
# flow-through.
#
# Cells (commit layer = L* + 1, except Qwen which already saturated by L=23
# so we test L=24 for the commit-layer profile):
#   Ministral L=17 (after L=16 saturation)
#   Qwen L=24      (after L=23 saturation)
#
# Wall-clock: ~50 min per cell × 2 = ~100 min on H100.
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
SEED="${SEED:-42}"

run_one() {
    local short="$1"
    local layer="$2"
    shift 2
    local logs="$@"
    local out_dir="results/causal_patching/${short}8b_l${layer}_components"
    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY_components.md" ]]; then
        echo "[skip] $out_dir exists"
        return 0
    fi
    local first_log
    first_log=$(echo $logs | awk '{print $1}')
    if [[ ! -f "$first_log" ]]; then
        echo "[skip] $short: missing $first_log"
        return 0
    fi
    mkdir -p "$out_dir"
    echo
    echo "############################################################"
    echo "## COMMIT-LAYER COMPONENTS: $short L=$layer"
    echo "##   out: $out_dir"
    echo "############################################################"
    python -m experiments.component_patching \
        --enriched-log $logs \
        --source-bucket clean_check_or_call \
        --target-bucket illegal_fold \
        --layer "$layer" \
        --components residual attn mlp head \
        --head-indices all \
        --n-source "$N_SOURCE" \
        --n-target "$N_TARGET" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE"
}

# Ministral L=17 (commit layer, saturation at L=16).
run_one ministral 17 \
    logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

# Qwen L=24 (one layer past L=23 saturation).
run_one qwen 24 \
    logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

echo
echo "============================================================"
echo "Commit-layer components COMPLETE."
echo "Compare ratio_to_residual (residual=100% always; attn and mlp"
echo "should be small fractions; no individual head should dominate)."
echo "If true: same compute-then-commit pattern as Llama."
echo "============================================================"
