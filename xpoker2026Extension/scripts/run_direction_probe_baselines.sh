#!/usr/bin/env bash
# =============================================================================
# Direction-probe baselines wrapper — run for all 3 models.
# =============================================================================
# Pure analysis on cached residuals — no GPU needed. ~5 min total.
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")/.."

run_one() {
    local short="$1"
    local layer="$2"
    shift 2
    local logs="$@"

    local probe_npz="results/direction_probe/${short}8b_l${layer}/raw_residuals.npz"
    local out_md="results/direction_probe_baselines/${short}8b_l${layer}.md"

    if [[ -f "$out_md" ]]; then
        echo "[skip] $out_md exists"
        return 0
    fi
    if [[ ! -f "$probe_npz" ]]; then
        echo "[skip] $short: missing $probe_npz"
        return 0
    fi
    mkdir -p "$(dirname "$out_md")"
    echo
    echo "## Direction-probe baselines: $short L=$layer"
    python -m experiments.direction_probe_baselines \
        --probe-npz "$probe_npz" \
        --enriched-log $logs \
        --out-md "$out_md"
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
echo "Direction-probe baselines COMPLETE."
echo "Read each .md and look for: learned ≫ random + permuted-label ~50%"
echo "============================================================"
