#!/usr/bin/env bash
# =============================================================================
# A1 head ablation on `illegal_fold` (near-threshold) targets — Phase P.
# =============================================================================
# Per updates.md §19j.5: the original A1 head ablation only tested
# `clean_check_or_call` targets, where the model is CHECK-saturated and
# has plenty of headroom before threshold-crossing. Ablating CHECK-encoding
# heads from a CHECK-confident baseline is an "easy" test — the model has
# many redundant paths, so we saw 0% top-1 family change.
#
# A more sensitive necessity test is on `illegal_fold` targets (the
# near-threshold failure mode). On these, the model's residual is
# presumably mid-strength FOLD-leaning and ablating dominant heads
# COULD push it back into CHECK territory. If even THIS test shows
# no top-1 family change, the redundancy claim is strong; if there IS
# change, we have a contingent necessity result.
#
# Wall-clock: ~10-15 min per model on H100 = ~30-45 min total.
#
# Outputs:
#   results/head_ablation/{llama,ministral,qwen}8b_l*_illegal_fold/SUMMARY.md
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
    shift 2
    local logs="$@"

    local out_dir="results/head_ablation/${short}8b_l${layer}_illegal_fold"
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
    echo "## A1 illegal_fold ablation: $short L=$layer"
    echo "##   target_bucket: illegal_fold (near-threshold FOLD-leaning)"
    echo "##   head sets: ${HEAD_SETS[@]}"
    echo "##   out: $out_dir"
    echo "############################################################"

    python -m experiments.head_ablation \
        --enriched-log $logs \
        --layer "$layer" \
        --target-bucket illegal_fold \
        --head-sets "${HEAD_SETS[@]}" \
        --n-target "$N_TARGET" \
        --seed "$SEED" \
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
echo "A1 illegal_fold head ablation COMPLETE."
echo
echo "Compare each <model>_illegal_fold/SUMMARY.md to the original"
echo "<model>/SUMMARY.md (which used clean_check_or_call targets):"
echo "  - If illegal_fold ablation ALSO shows ~0% top-1 flip:"
echo "    necessity is null even at near-threshold — strong"
echo "    redundancy claim."
echo "  - If illegal_fold ablation shows >10% top-1 flip:"
echo "    contingent necessity at near-threshold; the heads"
echo "    matter when the model is uncertain even if not when"
echo "    it is CHECK-saturated."
echo "============================================================"
