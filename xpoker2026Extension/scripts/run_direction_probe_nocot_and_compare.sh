#!/usr/bin/env bash
# =============================================================================
# Direction probe in non-CoT mode + CoT-vs-nonCoT cosine comparison.
# =============================================================================
#
# Two-step:
#   1. Run the same direction probe on non-CoT enriched logs (clean_CC vs
#      clean_LF residuals at L*). Output: results/direction_probe_nocot/...
#   2. Compare each model's CoT direction (cached from Phase L) to its
#      non-CoT direction. Output: results/direction_cosine_compare/...
#
# Wall-clock: ~30-40 min total (probe steps are ~10 min each; cosine
# compare is seconds).
#
# Outputs
# -------
#   results/direction_probe_nocot/{llama,qwen}8b_l*/SUMMARY.md
#   results/direction_cosine_compare/{llama,qwen}_cot_vs_nocot.md
#
# Env knobs
# ---------
#   PROBE_MAX_DECISIONS    default 300 (per bucket)
#   DEVICE / DTYPE         default cuda / bfloat16
# =============================================================================

set -uo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
PROBE_MAX_DECISIONS="${PROBE_MAX_DECISIONS:-300}"

run_probe_nocot() {
    local short="$1"
    local layer="$2"
    local enriched="$3"

    local out_dir="results/direction_probe_nocot/${short}8b_l${layer}"
    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already populated"
        return 0
    fi
    if [[ ! -f "$enriched" ]]; then
        if [[ -f "${enriched}.gz" ]]; then
            enriched="${enriched}.gz"
        else
            echo "[skip] $short non-CoT probe: missing $enriched"
            return 0
        fi
    fi
    mkdir -p "$out_dir"
    echo
    echo "############################################################"
    echo "## $short non-CoT direction probe at L=$layer"
    echo "##   enriched: $enriched"
    echo "##   out: $out_dir"
    echo "############################################################"
    python -m experiments.decision_direction_probe \
        --enriched-log "$enriched" \
        --layer "$layer" \
        --max-decisions-per-bucket "$PROBE_MAX_DECISIONS" \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE"
}

# Step 1: probes on non-CoT.
run_probe_nocot llama 14 \
    logs/scaled_llama8b_t0_s42_informative_v2_enriched.jsonl
run_probe_nocot qwen 23 \
    logs/scaled_qwen8b_t0_s42_informative_v2_enriched.jsonl

# Step 2: cosine compare for each model that has both probes.
compare_one() {
    local short="$1"
    local layer="$2"
    local cot_npz="results/direction_probe/${short}8b_l${layer}/raw_residuals.npz"
    local nocot_npz="results/direction_probe_nocot/${short}8b_l${layer}/raw_residuals.npz"
    local out_md="results/direction_cosine_compare/${short}_cot_vs_nocot_l${layer}.md"

    if [[ -f "$out_md" ]]; then
        echo "[skip] $out_md exists"
        return 0
    fi
    if [[ ! -f "$cot_npz" ]]; then
        echo "[skip] $short: missing CoT probe ($cot_npz)"
        return 0
    fi
    if [[ ! -f "$nocot_npz" ]]; then
        echo "[skip] $short: missing non-CoT probe ($nocot_npz)"
        return 0
    fi
    mkdir -p "$(dirname "$out_md")"
    echo
    echo "## CoT vs non-CoT direction comparison ($short L=$layer)"
    python -m experiments.direction_cosine_compare \
        --probe-a "$cot_npz" \
        --probe-b "$nocot_npz" \
        --label-a "${short} L=${layer} CoT" \
        --label-b "${short} L=${layer} non-CoT" \
        --out-md "$out_md"
}

compare_one llama 14
compare_one qwen 23

echo
echo "============================================================"
echo "Direction-probe-nocot + CoT-vs-nonCoT cosine COMPLETE."
echo
echo "Read each comparison .md:"
echo "  results/direction_cosine_compare/llama_cot_vs_nocot_l14.md"
echo "  results/direction_cosine_compare/qwen_cot_vs_nocot_l23.md"
echo
echo "If cos(w_CoT, w_nonCoT) > 0.85: SAME direction encodes the verb in"
echo "both CoT and non-CoT modes. Strongest possible 'circuit is intrinsic'"
echo "evidence — supports the §17c reframe."
echo "============================================================"
