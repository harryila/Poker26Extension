#!/usr/bin/env bash
# =============================================================================
# Causal patching FULL (Phase 3 of plan).
# =============================================================================
#
# Phase 3 follow-up to scripts/run_causal_patching_pilot.sh. Runs the full
# scope on Ministral s42 t=0 (where the §13 illegal-FOLD signal is strongest):
# all 29 clean CHECK_OR_CALL sources × all 179 illegal_FOLD targets × 7 layers.
#
# Cost on one H100 (unbatched): ~10 hours.
# Use SCOPE=subsampled to drop target_n to 30 -> ~30 min, recommended unless
# pilot showed weak signal.
#
# Phase 4 (multi-model) extension via SCOPE=multi: also runs Llama 8B s42 t=0
# under the same protocol. Add ~4-10h.
#
# Pre-flight: scripts/run_causal_patching_pilot.sh must have completed cleanly
# AND the layer-specificity gate must have passed (see plan §C).
#
# Usage:
#   SCOPE=subsampled bash scripts/run_causal_patching_full.sh   # default
#   SCOPE=full       bash scripts/run_causal_patching_full.sh
#   SCOPE=multi      bash scripts/run_causal_patching_full.sh
# =============================================================================

set -euo pipefail

SCOPE="${SCOPE:-subsampled}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LAYERS="${LAYERS:-22 23 25 27 28 30 32}"
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
    echo "Cell: $cell  |  n_target=$n_target  |  scope=$SCOPE"
    echo "============================================================"

    # Pre-flight: tokenizer-level position mapping (cheap CPU check)
    python -m experiments.verify_position_mapping \
        --enriched-log "$enriched" --n-samples 10 \
        || { echo "ERROR: position-mapping failed for $cell"; return 1; }

    # Pre-flight: GPU prompt reconstruction
    python -m experiments.verify_prompt_reconstruction \
        --enriched-log "$enriched" --n-samples 5 \
        --device "$DEVICE" --dtype "$DTYPE" \
        || { echo "ERROR: prompt-reconstruction failed for $cell"; return 1; }

    python -m experiments.causal_patching \
        --enriched-log "$enriched" \
        --source-bucket clean_check_or_call \
        --target-bucket illegal_fold \
        --layers $LAYERS \
        --n-source 29 \
        --n-target "$n_target" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" \
        --dtype "$DTYPE"

    echo "Cell $cell done. Read $out_dir/SUMMARY.md"
    echo
}

case "$SCOPE" in
    subsampled)
        # 29 src x 30 tgt x 7 layers = 6,090 patched forwards ~ 30 min on H100.
        run_one ministral8b_t0_s42 30 ministral8b_t0_s42_subsampled
        ;;
    full)
        # 29 src x 179 tgt x 7 layers = 36,337 patched forwards ~ 10 h on H100.
        run_one ministral8b_t0_s42 179 ministral8b_t0_s42_full
        ;;
    multi)
        # Both Ministral and Llama at full scope. ~14-20 h.
        run_one ministral8b_t0_s42 179 ministral8b_t0_s42_full
        run_one llama8b_t0_s42      16  llama8b_t0_s42_full   # only 16 illegal_FOLDs
        ;;
    *)
        echo "ERROR: unknown SCOPE='$SCOPE' (use subsampled|full|multi)"
        exit 1
        ;;
esac

echo "============================================================"
echo "All cells in scope=$SCOPE complete."
echo "Aggregate per-layer table: results/causal_patching/*/SUMMARY.md"
echo "============================================================"
