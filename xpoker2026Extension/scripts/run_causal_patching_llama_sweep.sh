#!/usr/bin/env bash
# =============================================================================
# Causal patching layer sweep for LLAMA 8B (8B-family generalization, Phase H).
# =============================================================================
#
# Companion to run_causal_patching_layer_sweep.sh (Ministral). Runs the same
# 36-layer sweep on Llama 8B to test whether the L=14-16 deliberation circuit
# discovered for Ministral is a general 8B-class property.
#
# Llama 8B has 32 layers (vs Ministral's 36). If the circuit is at "the same
# proportional depth" (~44%) we'd predict Llama L=14 (close to Ministral's
# L=14-16). If "the same absolute depth" we'd also predict L=14-16.
#
# SCOPE options
# -------------
# - SCOPE=s42_only (default for parity with Ministral pilot):
#     Single seed (s42) × t=0 × 100 hands = 16 illegal_FOLDs available.
#     n_target=16 (use them all). ~36 layers x 10 src x 16 tgt = 5,760 forwards.
#     Wall-clock ~3 h on H100.
#
# - SCOPE=pooled (recommended for cross-model claim):
#     Pool s42 + s123 + s456 t=0 → 68 illegal_FOLDs available.
#     n_target=30 (random sample). ~36 layers x 10 src x 30 tgt = 10,800 forwards.
#     Wall-clock ~5-6 h on H100. Better statistics but the boundary is for
#     "Llama 8B under CoT" rather than "Llama 8B s42 under CoT".
#
# Outputs
# -------
# - SCOPE=s42_only:
#     results/causal_patching/llama8b_t0_s42_layer_sweep/
# - SCOPE=pooled:
#     results/causal_patching/llama8b_t0_pooled_layer_sweep/
# Each contains SUMMARY.md (per-layer table) + summary.json + by_pair.csv.
#
# Pre-flights baked in (verify_position_mapping + verify_prompt_reconstruction).
# =============================================================================

set -euo pipefail

SCOPE="${SCOPE:-s42_only}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
# Llama 8B has 32 layers; sweep all of them.
LAYERS="${LAYERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31}"
N_SOURCE="${N_SOURCE:-10}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
SEED="${SEED:-42}"

cd "$(dirname "$0")/.."  # repo root

case "$SCOPE" in
    s42_only)
        ENRICHED_LOGS="logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz"
        N_TARGET="${N_TARGET:-16}"  # all illegal_FOLDs in s42 t0
        OUT_DIR="${OUT_DIR:-results/causal_patching/llama8b_t0_s42_layer_sweep}"
        ;;
    pooled)
        ENRICHED_LOGS="logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
                        logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
                        logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz"
        N_TARGET="${N_TARGET:-30}"
        OUT_DIR="${OUT_DIR:-results/causal_patching/llama8b_t0_pooled_layer_sweep}"
        ;;
    *)
        echo "ERROR: unknown SCOPE='$SCOPE' (use s42_only|pooled)"
        exit 1
        ;;
esac

mkdir -p "$OUT_DIR"

n_layers=$(echo "$LAYERS" | wc -w | tr -d ' ')
echo "============================================================"
echo "Causal patching LAYER SWEEP — LLAMA 8B"
echo "============================================================"
echo "  Scope:           $SCOPE"
echo "  Enriched logs:   $(echo $ENRICHED_LOGS | tr ' ' '\n' | wc -l | tr -d ' ') log(s)"
echo "  Layers:          $n_layers (0..$(( n_layers - 1 )))"
echo "  Sources:         $N_SOURCE clean CHECK_OR_CALL"
echo "  Targets:         $N_TARGET illegal_FOLD"
echo "  Random ctl:      $N_RANDOM_CONTROL alt-bucket sources per layer"
echo "  Out dir:         $OUT_DIR"
echo "============================================================"
echo

# Pre-flight 1: position mapping (CPU)
echo "[pre-flight 1/2] verify position mapping (uses first log) ..."
first_log=$(echo $ENRICHED_LOGS | awk '{print $1}')
python -m experiments.verify_position_mapping \
    --enriched-log "$first_log" \
    --n-samples 10 \
    || { echo "ERROR: position-mapping verification failed."; exit 1; }
echo

# Pre-flight 2: prompt reconstruction (GPU)
echo "[pre-flight 2/2] verify prompt reconstruction (GPU) ..."
python -m experiments.verify_prompt_reconstruction \
    --enriched-log "$first_log" \
    --n-samples 5 \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    || { echo "ERROR: prompt-reconstruction failed. EXPERIMENT BLOCKED."; exit 1; }
echo

echo "[sweep] starting main run ..."
python -m experiments.causal_patching \
    --enriched-log $ENRICHED_LOGS \
    --source-bucket clean_check_or_call \
    --target-bucket illegal_fold \
    --layers $LAYERS \
    --n-source "$N_SOURCE" \
    --n-target "$N_TARGET" \
    --n-random-control "$N_RANDOM_CONTROL" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR" \
    --device "$DEVICE" \
    --dtype "$DTYPE"

echo
echo "============================================================"
echo "Llama sweep COMPLETE."
echo "  Read: $OUT_DIR/SUMMARY.md"
echo "  Compare to: results/causal_patching/ministral8b_t0_s42_layer_sweep/SUMMARY.md"
echo "  Look for: where does specificity-adjusted Δ first cross +1 nat?"
echo "  Hypothesis: Llama transition at L~14 (matching Ministral L=14-16)."
echo "============================================================"
