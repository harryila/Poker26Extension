#!/usr/bin/env bash
# =============================================================================
# Causal patching PILOT (Phase 2 of plan).
# =============================================================================
#
# Tests whether activation-patching late-layer (~22-30) residuals from CLEAN
# CHECK_OR_CALL decisions into ILLEGAL_FOLD decisions causes the target's
# predicted action verb to flip toward CHECK.
#
# Scope (default, overrideable via env vars):
#   - Cell:     Ministral 8B Instruct, seed 42, t=0  (where signal is strongest)
#   - Sources:  10 clean CHECK_OR_CALL decisions
#   - Targets:  30 illegal-FOLD decisions
#   - Layers:   22 24 26 28 30
#   = 1500 patched forwards + 10 source captures + 30 baseline + ~30 control
#   ~= 50 min on one H100 (unbatched)
#
# Outputs:
#   results/causal_patching/ministral8b_t0_s42_pilot/
#     summary.json
#     by_pair.csv
#     SUMMARY.md
#
# Pre-flight (REQUIRED, all on the GPU box):
#   1. python -m experiments.verify_position_mapping --enriched-log $LOG \
#          --n-samples 10
#      (CPU-only; verifies prompt_hash + verb-finder; should pass 10/10)
#
#   2. python -m experiments.verify_prompt_reconstruction --enriched-log $LOG \
#          --n-samples 5 --device cuda
#      (GPU; verifies forward-pass top-1 matches recorded verb)
#      EXPERIMENT BLOCKED if this isn't 5/5.
#
# Usage:
#   bash scripts/run_causal_patching_pilot.sh
# =============================================================================

set -euo pipefail

ENRICHED_LOG="${ENRICHED_LOG:-logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz}"
SOURCE_BUCKET="${SOURCE_BUCKET:-clean_check_or_call}"
TARGET_BUCKET="${TARGET_BUCKET:-illegal_fold}"
LAYERS="${LAYERS:-22 24 26 28 30}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
OUT_DIR="${OUT_DIR:-results/causal_patching/ministral8b_t0_s42_pilot}"

cd "$(dirname "$0")/.."  # repo root
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Causal patching PILOT"
echo "============================================================"
echo "  Enriched log: $ENRICHED_LOG"
echo "  Source bucket: $SOURCE_BUCKET (n=$N_SOURCE)"
echo "  Target bucket: $TARGET_BUCKET (n=$N_TARGET)"
echo "  Layers: $LAYERS"
echo "  Out dir: $OUT_DIR"
echo "  Device: $DEVICE  Dtype: $DTYPE"
echo "============================================================"
echo

# Pre-flight 1: position mapping (CPU)
echo "[pre-flight 1/2] verify position mapping (CPU) ..."
python -m experiments.verify_position_mapping \
    --enriched-log "$ENRICHED_LOG" \
    --n-samples 10 \
    || { echo "ERROR: position-mapping verification failed."; exit 1; }
echo

# Pre-flight 2: prompt reconstruction (GPU)
echo "[pre-flight 2/2] verify prompt reconstruction (GPU) ..."
python -m experiments.verify_prompt_reconstruction \
    --enriched-log "$ENRICHED_LOG" \
    --n-samples 5 \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    || { echo "ERROR: prompt-reconstruction verification failed. EXPERIMENT BLOCKED."; exit 1; }
echo

# Main pilot run
echo "[pilot] starting main run ..."
python -m experiments.causal_patching \
    --enriched-log "$ENRICHED_LOG" \
    --source-bucket "$SOURCE_BUCKET" \
    --target-bucket "$TARGET_BUCKET" \
    --layers $LAYERS \
    --n-source "$N_SOURCE" \
    --n-target "$N_TARGET" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR" \
    --device "$DEVICE" \
    --dtype "$DTYPE"

echo
echo "============================================================"
echo "Pilot COMPLETE."
echo "  Read: $OUT_DIR/SUMMARY.md      (per-layer Δlogit table)"
echo "  Read: $OUT_DIR/summary.json    (machine-readable)"
echo "  Read: $OUT_DIR/by_pair.csv     (per-pair raw rows)"
echo "============================================================"
echo
echo "Decision gate (from plan §C):"
echo "  Strong success: mean_delta @ L=28 >= 3x value at L=22 AND L=28 abs > 1.0 nat -> Phase 3"
echo "  Weak success:   layer-specificity holds but small absolute -> Phase 3 with caveat"
echo "  Null:           no layer specificity OR random-source control failed -> publish negative"
