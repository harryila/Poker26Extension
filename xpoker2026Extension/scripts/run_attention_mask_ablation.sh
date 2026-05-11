#!/usr/bin/env bash
# =============================================================================
# Attention-mask ablation wrapper — Llama, Ministral, Qwen.
# =============================================================================
# Masks attention TO the legal-actions list tokens (across all heads/layers)
# and measures verb-prediction degradation. Tests whether the model's verb
# prediction is causally dependent on attending to that line.
#
# Wall-clock: ~10 min per model on H100 = ~30 min total.
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_TARGET="${N_TARGET:-50}"

run_one() {
    local short="$1"
    shift
    local logs="$@"

    local out_dir="results/attn_mask_ablation/${short}8b"
    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already populated"
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
    echo "## ATTN-MASK ABLATION: $short (Legal actions list)"
    echo "##   out: $out_dir"
    echo "############################################################"
    python -m experiments.attention_mask_ablation \
        --enriched-log $logs \
        --target-bucket clean_check_or_call \
        --n-target "$N_TARGET" \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE"
}

run_one llama \
    logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

run_one ministral \
    logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

run_one qwen \
    logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

echo
echo "============================================================"
echo "Attention-mask ablation COMPLETE."
echo "Read each SUMMARY.md and look for 'Verb predicted, baseline → masked'"
echo "drop. Big drops = legal-actions attention is causally needed."
echo "============================================================"
