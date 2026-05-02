#!/usr/bin/env bash
# Run PCE distribution + update coherence + pot-odds analyses on the
# enriched scaled baseline logs (Tier 1A: large + small sub-tiers).
#
# Produces:
#   results/scaled_baseline/
#     pce_<model>_records.csv          per-decision PCE for each model
#     pce_<model>_summary.csv          clustered-bootstrap summary per model
#     uc_<model>.csv                   per-decision update coherence
#     uc_<model>_summary.json          update coherence summary
#     pot_odds_<model>_t0_s42.csv      EV / pot-odds analysis on one cell
#     pce_pooled_large_*.csv           pooled across all 70B-class models
#     pce_pooled_small_*.csv           pooled across all 7-8B-class models
#     pce_pooled_all_*.csv             pooled across everything
#
# Compare each model's gap (LLM closer to CardOnly vs StrategyAware) against
# the paper's anchor of 0.014.

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results/scaled_baseline}"
mkdir -p "$RESULTS_DIR"

# Each entry: tag : tier : glob
# tier is used only to drive tier-pooled comparisons at the end.
MODELS=(
    "llama70b:large:${LOGS_DIR}/scaled_llama70b_*_enriched.jsonl"
    "qwen72b:large:${LOGS_DIR}/scaled_qwen72b_*_enriched.jsonl"
    "llama33-70b:large:${LOGS_DIR}/scaled_llama33-70b_*_enriched.jsonl"
    "llama8b:small:${LOGS_DIR}/scaled_llama8b_*_enriched.jsonl"
    "qwen8b:small:${LOGS_DIR}/scaled_qwen8b_*_enriched.jsonl"
    "ministral8b:small:${LOGS_DIR}/scaled_ministral8b_*_enriched.jsonl"
)

shopt -s nullglob

run_model_analysis() {
    local tag="$1"
    local glob="$2"
    # shellcheck disable=SC2206
    local files=( $glob )
    if [[ "${#files[@]}" -eq 0 ]]; then
        echo "[skip $tag] no enriched logs match $glob"
        return
    fi
    echo "=== $tag (${#files[@]} files) ==="

    python -m analysis.compute_pce_distribution \
        "${files[@]}" \
        --output-records "${RESULTS_DIR}/pce_${tag}_records.csv" \
        --output-summary "${RESULTS_DIR}/pce_${tag}_summary.csv" \
        --bootstrap 1000 --seed 42 --clustered

    python -m analysis.compute_update_coherence \
        "${files[@]}" \
        --output "${RESULTS_DIR}/uc_${tag}.csv" \
        --output-summary "${RESULTS_DIR}/uc_${tag}_summary.json"

    # Pot-odds / EV analysis on a single representative cell to keep this fast;
    # rerun by hand on more cells for tighter EV CIs.
    rep="${LOGS_DIR}/scaled_${tag}_t0_s42_informative_v2_enriched.jsonl"
    if [[ -f "$rep" ]]; then
        python -m analysis.pot_odds_analysis \
            --input "$rep" \
            --output "${RESULTS_DIR}/pot_odds_${tag}_t0_s42.csv" \
            --num-rollouts 100 --samples-per-bucket 5 \
            --opponent-preset informative_v2 \
            --skip-belief-ev || echo "  (pot-odds skipped for $tag)"
    fi
}

# Per-model analyses
for entry in "${MODELS[@]}"; do
    tag="${entry%%:*}"
    rest="${entry#*:}"
    glob="${rest#*:}"
    run_model_analysis "$tag" "$glob"
done

# ---- Tier-pooled comparisons ----
# Compare against the paper anchor 0.014 with a single, larger sample.
run_tier_pool() {
    local tier="$1"
    shift
    local globs=("$@")
    local files=()
    for g in "${globs[@]}"; do
        # shellcheck disable=SC2206
        local m=( $g )
        files+=("${m[@]}")
    done
    if [[ "${#files[@]}" -eq 0 ]]; then
        echo "[skip pool $tier] no enriched logs"
        return
    fi
    echo "=== pooled $tier (${#files[@]} files) ==="
    python -m analysis.compute_pce_distribution \
        "${files[@]}" \
        --output-records "${RESULTS_DIR}/pce_pooled_${tier}_records.csv" \
        --output-summary "${RESULTS_DIR}/pce_pooled_${tier}_summary.csv" \
        --bootstrap 1000 --seed 42 --clustered
}

run_tier_pool "large" \
    "${LOGS_DIR}/scaled_llama70b_*_enriched.jsonl" \
    "${LOGS_DIR}/scaled_qwen72b_*_enriched.jsonl" \
    "${LOGS_DIR}/scaled_llama33-70b_*_enriched.jsonl"

run_tier_pool "small" \
    "${LOGS_DIR}/scaled_llama8b_*_enriched.jsonl" \
    "${LOGS_DIR}/scaled_qwen8b_*_enriched.jsonl" \
    "${LOGS_DIR}/scaled_ministral8b_*_enriched.jsonl"

# Cross-tier full pool (everything together)
all_files=( ${LOGS_DIR}/scaled_*_enriched.jsonl )
if [[ "${#all_files[@]}" -gt 0 ]]; then
    echo "=== pooled all (${#all_files[@]} files) ==="
    python -m analysis.compute_pce_distribution \
        "${all_files[@]}" \
        --output-records "${RESULTS_DIR}/pce_pooled_all_records.csv" \
        --output-summary "${RESULTS_DIR}/pce_pooled_all_summary.csv" \
        --bootstrap 1000 --seed 42 --clustered
fi

echo ""
echo "Outputs in: $RESULTS_DIR"
echo ""
echo "Key files to inspect:"
echo "  $RESULTS_DIR/pce_pooled_large_summary.csv   <- 70B-class headline"
echo "  $RESULTS_DIR/pce_pooled_small_summary.csv   <- 8B-class headline"
echo "  $RESULTS_DIR/pce_<model>_summary.csv        <- per-model breakdowns"
echo "  $RESULTS_DIR/uc_<model>_summary.json        <- update coherence per model"
