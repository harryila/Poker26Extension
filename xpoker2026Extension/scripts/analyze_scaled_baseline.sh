#!/usr/bin/env bash
# Run PCE distribution + update coherence + pot-odds analyses on the
# enriched scaled baseline logs, grouped by model.
#
# Produces per-model summaries plus a pooled cross-model report so we can
# compare each model's gap (LLM closer to CardOnly vs StrategyAware) against
# the paper's anchor of 0.014.

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results/scaled_baseline}"
mkdir -p "$RESULTS_DIR"

# Each entry: tag : glob
MODELS=(
    "llama70b:${LOGS_DIR}/scaled_llama70b_*_enriched.jsonl"
    "qwen72b:${LOGS_DIR}/scaled_qwen72b_*_enriched.jsonl"
    "llama33-70b:${LOGS_DIR}/scaled_llama33-70b_*_enriched.jsonl"
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

for entry in "${MODELS[@]}"; do
    tag="${entry%%:*}"
    glob="${entry#*:}"
    run_model_analysis "$tag" "$glob"
done

# Cross-model comparison: pool every enriched log, run PCE once.
all_files=( ${LOGS_DIR}/scaled_*_enriched.jsonl )
if [[ "${#all_files[@]}" -gt 0 ]]; then
    echo "=== pooled cross-model (${#all_files[@]} files) ==="
    python -m analysis.compute_pce_distribution \
        "${all_files[@]}" \
        --output-records "${RESULTS_DIR}/pce_pooled_records.csv" \
        --output-summary "${RESULTS_DIR}/pce_pooled_summary.csv" \
        --bootstrap 1000 --seed 42 --clustered
fi

echo ""
echo "Outputs in: $RESULTS_DIR"
