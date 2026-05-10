#!/usr/bin/env bash
# =============================================================================
# Position-sweep direction projection (D1) — wrapper for 3 models.
# =============================================================================
#
# For each model, capture residuals at L* across the FULL input + response,
# project onto the cached verb direction, and report per-bucket mean
# projection at each relative position. Tests "where in the response does
# the decision crystallize?" without requiring position-flexible patching.
#
# Reuses the cached weight vectors from results/direction_probe/<model>8b_l*/raw_residuals.npz
# (produced in Phase L). Will fail cleanly if those files don't exist.
#
# Wall-clock: ~5-10 min per model after model load, ~30-40 min total.
#
# Outputs
# -------
#   results/position_sweep/{llama,ministral,qwen}8b_l*/SUMMARY.md
#
# Env knobs
# ---------
#   N_PER_BUCKET=50    decisions per bucket (default 50)
#   DEVICE / DTYPE     default cuda / bfloat16
# =============================================================================

set -uo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_PER_BUCKET="${N_PER_BUCKET:-50}"

run_one() {
    local short="$1"
    local layer="$2"
    shift 2
    local logs="$@"

    local probe_npz="results/direction_probe/${short}8b_l${layer}/raw_residuals.npz"
    local out_dir="results/position_sweep/${short}8b_l${layer}"

    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already populated"
        return 0
    fi
    if [[ ! -f "$probe_npz" ]]; then
        echo "[skip] $short: missing direction probe ($probe_npz)"
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
    echo "## POSITION SWEEP: $short L=$layer"
    echo "##   probe: $probe_npz"
    echo "##   logs: $logs"
    echo "##   n_per_bucket=$N_PER_BUCKET"
    echo "##   out: $out_dir"
    echo "############################################################"
    python -m experiments.position_sweep_direction_projection \
        --enriched-log $logs \
        --layer "$layer" \
        --probe-npz "$probe_npz" \
        --max-decisions-per-bucket "$N_PER_BUCKET" \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE"
}

run_one llama 14 \
    logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

run_one ministral 16 \
    logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

run_one qwen 23 \
    logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

echo
echo "============================================================"
echo "Position-sweep COMPLETE."
echo
echo "Read each SUMMARY.md and look at the projection trajectory:"
echo "  - rel_pos << 0 (early in response): projections should be near zero"
echo "    if the decision isn't yet committed."
echo "  - rel_pos = 0 (verb position): bucket means should diverge."
echo "  - rel_pos > 0 (after verb): does the residual stay committed?"
echo
echo "Compute-then-commit prediction: a sharp transition from undecided"
echo "(near 0) to committed (large +/-) within a few tokens of the verb"
echo "position. Cross-model: Llama (sharp 1-layer transition) vs Qwen"
echo "(distributed across multiple layers) — does this also show up at"
echo "the position level within the response?"
echo "============================================================"
