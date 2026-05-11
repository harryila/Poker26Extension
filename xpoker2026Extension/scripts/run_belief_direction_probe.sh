#!/usr/bin/env bash
# Belief direction probe (B3) wrapper — Llama L=14, Ministral L=16, Qwen L=23.
set -uo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N="${N:-300}"

run_one() {
    local short="$1"
    local layer="$2"
    shift 2
    local logs="$@"
    local out_dir="results/belief_direction_probe/${short}8b_l${layer}"
    local probe_npz="results/direction_probe/${short}8b_l${layer}/raw_residuals.npz"
    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir exists"; return 0
    fi
    if [[ ! -f "$probe_npz" ]]; then
        echo "[skip] $short: missing $probe_npz"; return 0
    fi
    local first
    first=$(echo $logs | awk '{print $1}')
    if [[ ! -f "$first" ]]; then
        echo "[skip] $short: missing $first"; return 0
    fi
    mkdir -p "$out_dir"
    python -m experiments.belief_direction_probe \
        --enriched-log $logs \
        --layer "$layer" \
        --probe-npz "$probe_npz" \
        --max-decisions "$N" \
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

echo "[done] read results/belief_direction_probe/*/SUMMARY.md"
