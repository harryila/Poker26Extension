#!/usr/bin/env bash
# =============================================================================
# Ministral 8B causal-patching sweep REPLICATION across seeds and temps.
# =============================================================================
#
# The original sweep (run_causal_patching_layer_sweep.sh) only ran on Ministral
# s42 t=0. To confirm the L=14-16 boundary is robust to seed and temperature,
# this script runs 3 replication cells:
#
#   - s42  t=0.2  (same seed, different temp)
#   - s123 t=0.0  (different seed, same temp)
#   - s456 t=0.0  (different seed, same temp)
#
# Each is 36 layers x 10 sources x N_target patched forwards. The s42 t=0.2
# run uses N_target=30 (160 illegal_FOLDs available); s123 and s456 cells use
# N_target = whatever's available (1 and 3 respectively — basically a coverage
# check, not statistical replication).
#
# Total: ~2 + 0.5 + 0.5 = ~3 hours on H100.
#
# Outputs:
#   results/causal_patching/ministral8b_t02_s42_replicate/
#   results/causal_patching/ministral8b_t0_s123_replicate/
#   results/causal_patching/ministral8b_t0_s456_replicate/
# =============================================================================

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LAYERS="${LAYERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35}"
N_SOURCE="${N_SOURCE:-10}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
SEED="${SEED:-42}"

cd "$(dirname "$0")/.."  # repo root

run_one() {
    local cell="$1"
    local n_target="$2"
    local out_subdir="$3"

    local enriched="logs/cot_${cell}_informative_v2_logitlens_enriched.jsonl.gz"
    local out_dir="results/causal_patching/${out_subdir}"
    mkdir -p "$out_dir"

    echo "============================================================"
    echo "Ministral replication cell: $cell  (n_target=$n_target)"
    echo "============================================================"
    echo "  Enriched: $enriched"
    echo "  Out dir:  $out_dir"
    echo

    python -m experiments.verify_position_mapping \
        --enriched-log "$enriched" --n-samples 10 \
        || { echo "ERROR: position-mapping failed for $cell"; return 1; }
    echo

    python -m experiments.verify_prompt_reconstruction \
        --enriched-log "$enriched" --n-samples 5 \
        --device "$DEVICE" --dtype "$DTYPE" \
        || { echo "ERROR: prompt-reconstruction failed for $cell"; return 1; }
    echo

    python -m experiments.causal_patching \
        --enriched-log "$enriched" \
        --source-bucket clean_check_or_call \
        --target-bucket illegal_fold \
        --layers $LAYERS \
        --n-source "$N_SOURCE" \
        --n-target "$n_target" \
        --n-random-control "$N_RANDOM_CONTROL" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" \
        --dtype "$DTYPE"

    echo
    echo "Cell $cell done. Read $out_dir/SUMMARY.md"
    echo
}

# s42 t=0.2: 161 illegal_FOLDs → use 30 (matched to s42 t=0 sweep scope)
run_one ministral8b_t02_s42 30 ministral8b_t02_s42_replicate

# s123 t=0: 1 illegal_FOLD → use the 1 (smoke test for the boundary)
run_one ministral8b_t0_s123 1 ministral8b_t0_s123_replicate

# s456 t=0: 3 illegal_FOLDs → use all 3
run_one ministral8b_t0_s456 3 ministral8b_t0_s456_replicate

echo "============================================================"
echo "All 3 replication cells complete."
echo "Compare boundary L* across:"
echo "  - results/causal_patching/ministral8b_t0_s42_layer_sweep/SUMMARY.md  (original)"
echo "  - results/causal_patching/ministral8b_t02_s42_replicate/SUMMARY.md"
echo "  - results/causal_patching/ministral8b_t0_s123_replicate/SUMMARY.md"
echo "  - results/causal_patching/ministral8b_t0_s456_replicate/SUMMARY.md"
echo "Hypothesis: all 4 cells show transition at L=14-16."
echo "============================================================"
