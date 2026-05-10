#!/usr/bin/env bash
# =============================================================================
# Extended attention-pattern analysis — 200 decisions per bucket (vs 50).
# =============================================================================
#
# Why this exists
# ---------------
# The Phase L attention-pattern run sampled 50 decisions per bucket — enough
# to see clear trends but tighter top-token frequency statistics need 200+
# per bucket. This script re-runs the same protocol with a larger sample
# cap, into separate output directories so the prior 50-sample results
# remain intact for comparison.
#
# Also extends to a 4th condition: per-bucket attention pattern at non-CoT
# decisions (using the scaled_*_enriched.jsonl logs). This lets us compare
# what the dominant head reads in CoT vs non-CoT.
#
# Wall-clock: ~30-40 min total.
#
# Outputs
# -------
# results/attention_patterns/{model}8b_l*_extended/   (200/bucket CoT)
# results/attention_patterns/{model}8b_l*_nocot/      (200/bucket non-CoT, where logs exist)
#
# Env knobs
# ---------
#   N_PER_BUCKET     default 200
#   DEVICE / DTYPE   default cuda / bfloat16
# =============================================================================

set -uo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_PER_BUCKET="${N_PER_BUCKET:-200}"

run_one() {
    local short="$1"
    local layer="$2"
    local heads="$3"
    local out_subdir="$4"
    shift 4
    local logs="$@"

    local out_dir="results/attention_patterns/${short}8b_l${layer}_${out_subdir}"
    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already populated"
        return 0
    fi
    local first_log
    first_log=$(echo $logs | awk '{print $1}')
    if [[ ! -f "$first_log" ]]; then
        if [[ -f "${first_log}.gz" ]]; then
            : # try .gz
        else
            echo "[skip] $short ($out_subdir): missing $first_log"
            return 0
        fi
    fi
    mkdir -p "$out_dir"
    echo
    echo "############################################################"
    echo "## $short L=$layer heads=$heads ($out_subdir)"
    echo "##   logs: $logs"
    echo "##   n_per_bucket=$N_PER_BUCKET"
    echo "##   out: $out_dir"
    echo "############################################################"
    python -m experiments.attention_patterns_at_dominant_heads \
        --enriched-log $logs \
        --layer "$layer" \
        --heads $heads \
        --max-decisions-per-bucket "$N_PER_BUCKET" \
        --top-k 8 \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE"
}

# Llama L=14 — extended sample on CoT.
run_one llama 14 "5 23 24" extended \
    logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

# Ministral L=16 — extended sample on CoT.
run_one ministral 16 "22 9 15" extended \
    logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

# Qwen L=23 — extended sample on CoT.
run_one qwen 23 "26 30 28" extended \
    logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

# NON-CoT attention patterns — only the buckets that actually exist (no
# illegal_fold in non-CoT). Use --buckets to override the default which
# includes illegal_fold.
run_one_nocot() {
    local short="$1"
    local layer="$2"
    local heads="$3"
    local enriched="$4"

    local out_dir="results/attention_patterns/${short}8b_l${layer}_nocot"
    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already populated"
        return 0
    fi
    if [[ ! -f "$enriched" ]]; then
        if [[ -f "${enriched}.gz" ]]; then
            enriched="${enriched}.gz"
        else
            echo "[skip] $short non-CoT: missing $enriched"
            return 0
        fi
    fi
    mkdir -p "$out_dir"
    echo
    echo "############################################################"
    echo "## $short L=$layer non-CoT heads=$heads"
    echo "##   enriched: $enriched"
    echo "##   buckets: clean_check_or_call, clean_legal_fold, clean_bet_or_raise"
    echo "##   n_per_bucket=$N_PER_BUCKET"
    echo "##   out: $out_dir"
    echo "############################################################"
    python -m experiments.attention_patterns_at_dominant_heads \
        --enriched-log "$enriched" \
        --layer "$layer" \
        --heads $heads \
        --buckets clean_check_or_call clean_legal_fold clean_bet_or_raise \
        --max-decisions-per-bucket "$N_PER_BUCKET" \
        --top-k 8 \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE"
}

run_one_nocot llama 14 "5 23 24" \
    logs/scaled_llama8b_t0_s42_informative_v2_enriched.jsonl
run_one_nocot qwen 23 "26 30 28" \
    logs/scaled_qwen8b_t0_s42_informative_v2_enriched.jsonl

echo
echo "============================================================"
echo "Extended attention patterns COMPLETE."
echo
echo "Compare CoT (extended) vs non-CoT directly per model:"
echo "  results/attention_patterns/llama8b_l14_extended/SUMMARY.md"
echo "  results/attention_patterns/llama8b_l14_nocot/SUMMARY.md"
echo "  results/attention_patterns/qwen8b_l23_extended/SUMMARY.md"
echo "  results/attention_patterns/qwen8b_l23_nocot/SUMMARY.md"
echo
echo "Same dominant tokens across CoT/non-CoT? Same entropy? Different?"
echo "============================================================"
