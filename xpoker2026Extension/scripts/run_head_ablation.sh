#!/usr/bin/env bash
# =============================================================================
# Head ablation (necessity test) wrapper — Llama, Ministral, Qwen at L*.
# =============================================================================
# For each model: ablate the dominant heads we identified (as a triplet, the
# extended quartet/sextet, the single dominant head, and a CONTROL set of
# random non-dominant heads) and measure verb-prediction degradation.
#
# Wall-clock: ~10-15 min per model on H100 = ~30-45 min total.
# =============================================================================

set -uo pipefail

cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_TARGET="${N_TARGET:-50}"
SEED="${SEED:-42}"

run_one() {
    local short="$1"
    local layer="$2"
    local target_bucket="$3"
    shift 3
    local logs="$@"

    local out_dir="results/head_ablation/${short}8b_l${layer}"
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

    case "$short" in
        llama)
            HEAD_SETS=("5 23 24" "2 5 23 24" "23" "0 1 7 9")
            ;;
        ministral)
            HEAD_SETS=("9 15 22" "9 15 21 22 24 30 31" "22" "0 1 5 13")
            ;;
        qwen)
            HEAD_SETS=("26 28 30" "26 30" "26" "0 1 5 9")
            ;;
        *)
            echo "[abort] unknown model $short"
            return 1
            ;;
    esac

    echo
    echo "############################################################"
    echo "## HEAD ABLATION: $short L=$layer (necessity test)"
    echo "##   target_bucket: $target_bucket"
    echo "##   head sets: ${HEAD_SETS[@]}"
    echo "##   out: $out_dir"
    echo "############################################################"

    python -m experiments.head_ablation \
        --enriched-log $logs \
        --layer "$layer" \
        --target-bucket "$target_bucket" \
        --head-sets "${HEAD_SETS[@]}" \
        --n-target "$N_TARGET" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE"
}

run_one llama 14 clean_check_or_call \
    logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

run_one ministral 16 clean_check_or_call \
    logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

run_one qwen 23 clean_check_or_call \
    logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

echo
echo "============================================================"
echo "Head ablation COMPLETE."
echo "Read each SUMMARY.md and look for 'verb predicted (baseline → ablated)'"
echo "and 'top-1 family changed' columns. Big drops = necessity confirmed."
echo "Compare dominant-head sets to the random control set (last row)."
echo "============================================================"
