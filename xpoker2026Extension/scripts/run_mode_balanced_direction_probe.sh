#!/usr/bin/env bash
# =============================================================================
# Mode-balanced direction probe — wrapper for Llama L=14 and Qwen L=23.
# =============================================================================
#
# Trains hand-matched CoT vs non-CoT direction probes for each (model, L*)
# pair we have non-CoT data for: Llama (L=14, s42) and Qwen (L=23, s42).
# The matched cosine answers: "does the verb-decision direction tilt
# between CoT and non-CoT for the SAME hand?" — eliminating the
# data-distribution-shift confound that drove the §18b cosine into 0.27/0.34.
#
# Wall-clock: ~30 min total (~10-15 min per model).
#
# Outputs
# -------
#   results/mode_balanced_probe/llama8b_l14/SUMMARY.md
#   results/mode_balanced_probe/qwen8b_l23/SUMMARY.md
#
# Env knobs
# ---------
#   MAX_PAIRS=200    cap matched (hand × decision) pairs (default 200)
#   DEVICE / DTYPE   default cuda / bfloat16
# =============================================================================

set -uo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MAX_PAIRS="${MAX_PAIRS:-200}"

run_one() {
    local short="$1"
    local layer="$2"
    local cot_log="$3"
    local nocot_log="$4"

    local out_dir="results/mode_balanced_probe/${short}8b_l${layer}"
    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already populated"
        return 0
    fi
    if [[ ! -f "$cot_log" ]]; then
        if [[ -f "${cot_log}.gz" ]]; then
            cot_log="${cot_log}.gz"
        else
            echo "[skip] $short: missing $cot_log"
            return 0
        fi
    fi
    if [[ ! -f "$nocot_log" ]]; then
        if [[ -f "${nocot_log}.gz" ]]; then
            nocot_log="${nocot_log}.gz"
        else
            echo "[skip] $short: missing $nocot_log"
            return 0
        fi
    fi
    mkdir -p "$out_dir"

    echo
    echo "############################################################"
    echo "## Mode-balanced probe: $short L=$layer"
    echo "##   CoT log:    $cot_log"
    echo "##   non-CoT log: $nocot_log"
    echo "##   max_pairs=$MAX_PAIRS"
    echo "##   out: $out_dir"
    echo "############################################################"

    python -m experiments.mode_balanced_direction_probe \
        --cot-log "$cot_log" \
        --nocot-log "$nocot_log" \
        --layer "$layer" \
        --max-pairs "$MAX_PAIRS" \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE"
}

run_one llama 14 \
    "logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz" \
    "logs/scaled_llama8b_t0_s42_informative_v2_enriched.jsonl"

run_one qwen 23 \
    "logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz" \
    "logs/scaled_qwen8b_t0_s42_informative_v2_enriched.jsonl"

echo
echo "============================================================"
echo "Mode-balanced probe COMPLETE."
echo
echo "Compare to Phase M §18b unmatched cosines:"
echo "  Llama L=14 unmatched: 0.27;   matched: read SUMMARY.md"
echo "  Qwen  L=23 unmatched: 0.34;   matched: read SUMMARY.md"
echo
echo "Matched cosine ≥0.6: §18b non-identity was data-distribution artifact."
echo "Matched cosine still in 0.2-0.4 range: direction tilt is real even"
echo "with hand population controlled — the verb encoding is genuinely"
echo "mode-specific within a shared subspace."
echo "============================================================"
