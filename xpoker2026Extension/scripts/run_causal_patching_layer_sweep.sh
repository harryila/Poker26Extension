#!/usr/bin/env bash
# =============================================================================
# Causal patching LAYER SWEEP — fine-grained scan to find where the
# deliberation circuit lives.
# =============================================================================
#
# Why this script exists
# ----------------------
# The first pilot (scripts/run_causal_patching_pilot.sh) tested layers
# 22 24 26 28 30 and found 100% of targets flipped to CHECK with mean
# Δlogit(CHECK − FOLD) = +10 to +11 nats AT EVERY LAYER. The effect was
# saturated -- meaning by L=22 the residual at the action-verb position
# already encodes the verb prediction, and patching it wholesale flips the
# output regardless of which specific layer in the late stack.
#
# To pin the deliberation to a specific layer/circuit, we need to scan
# EARLIER layers (where the effect should be near zero) and find where the
# transition happens. This script sweeps every layer of the model.
#
# Default scope (Ministral 8B = 36 layers, indices 0..35)
# -------------------------------------------------------
# - Cell:       Ministral 8B Instruct, seed 42, t=0
# - Sources:    10 clean CHECK_OR_CALL  (sample with --seed 42)
# - Targets:    15 illegal_FOLD          (halved from pilot to keep cost down;
#                                         stdev of mean ~0.28 is still tight)
# - Layers:     0 1 2 3 ... 35           (every single layer, dense scan)
# - Random ctl: 5 alt-bucket sources patched at EVERY layer (per-layer null)
#
# Cost (unbatched on H100):
#   - 10 source captures        :    10 forwards (~20s)
#   - 15 baseline forwards      :    15 forwards (~30s)
#   - 36 layers x 5 random ctl  :   180 forwards (~6 min)
#   - 36 layers x 36 self-patch :    36 forwards (~1 min)
#   - 36 layers x 10 src x 15 t :  5400 forwards (~3 h)
#   ----------------------------------------------------------
#   Total: ~5641 forwards    Wall-clock: ~3 hours on one H100
#
# What you get back (results/causal_patching/<out_dir>/)
# ------------------------------------------------------
# - SUMMARY.md          : per-layer table with main effect, random null,
#                         AND specificity-adjusted Δ (the writeup-ready signal)
# - summary.json        : machine-readable (controls.random_source_per_layer
#                         is the new per-layer null distribution)
# - by_pair.csv         : 5400 rows; one per (src, tgt, layer) for plotting
#
# Read the SUMMARY.md table top-to-bottom: at the layer where
# specificity-adjusted Δ first crosses ~+1 nat from ~0 nat is where the
# deliberation circuit causally locks in the verb.
#
# Pre-flights (reused from the original pilot script)
# ---------------------------------------------------
# 1. python -m experiments.verify_position_mapping ...      (CPU, 30s)
# 2. python -m experiments.verify_prompt_reconstruction ... (GPU, 30s)
# Halts immediately if either fails.
#
# Override examples
# -----------------
#   # Coarser sweep (every other layer; ~1.5h instead of 3h):
#   LAYERS="0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 35" \
#       bash scripts/run_causal_patching_layer_sweep.sh
#
#   # Even tighter on the suspected boundary (after seeing first pass):
#   LAYERS="14 15 16 17 18 19 20 21 22" N_TARGET=30 \
#       OUT_DIR=results/causal_patching/ministral8b_t0_s42_boundary \
#       bash scripts/run_causal_patching_layer_sweep.sh
#
#   # Different model:
#   ENRICHED_LOG=logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
#       LAYERS="$(seq 0 31 | tr '\n' ' ')" \
#       OUT_DIR=results/causal_patching/llama8b_t0_s42_layer_sweep \
#       bash scripts/run_causal_patching_layer_sweep.sh
# =============================================================================

set -euo pipefail

ENRICHED_LOG="${ENRICHED_LOG:-logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz}"
SOURCE_BUCKET="${SOURCE_BUCKET:-clean_check_or_call}"
TARGET_BUCKET="${TARGET_BUCKET:-illegal_fold}"
# Default: every layer of a 36-layer model.
LAYERS="${LAYERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-15}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
OUT_DIR="${OUT_DIR:-results/causal_patching/ministral8b_t0_s42_layer_sweep}"

cd "$(dirname "$0")/.."  # repo root
mkdir -p "$OUT_DIR"

# Count the layers for the wall-clock estimate.
n_layers=$(echo "$LAYERS" | wc -w | tr -d ' ')
n_main_forwards=$(( n_layers * N_SOURCE * N_TARGET ))
n_total_forwards=$(( N_SOURCE + N_TARGET + n_layers * N_RANDOM_CONTROL + n_layers + n_main_forwards ))

echo "============================================================"
echo "Causal patching LAYER SWEEP"
echo "============================================================"
echo "  Enriched log:    $ENRICHED_LOG"
echo "  Source bucket:   $SOURCE_BUCKET (n=$N_SOURCE)"
echo "  Target bucket:   $TARGET_BUCKET (n=$N_TARGET)"
echo "  Layers:          $n_layers layers ($LAYERS)"
echo "  Random control:  $N_RANDOM_CONTROL alt-bucket sources per layer"
echo "  Out dir:         $OUT_DIR"
echo "  Device/dtype:    $DEVICE / $DTYPE"
echo "  Estimated total: $n_total_forwards forwards"
echo "                   = $n_main_forwards main + overhead"
echo "                   ~ $(( n_total_forwards * 2 / 3600 ))-$(( n_total_forwards * 3 / 3600 )) h on H100 (unbatched)"
echo "============================================================"
echo

# Pre-flight 1: position mapping (CPU, fast)
echo "[pre-flight 1/2] verify position mapping ..."
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

echo "[sweep] starting main run ..."
python -m experiments.causal_patching \
    --enriched-log "$ENRICHED_LOG" \
    --source-bucket "$SOURCE_BUCKET" \
    --target-bucket "$TARGET_BUCKET" \
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
echo "Layer sweep COMPLETE."
echo "  Read: $OUT_DIR/SUMMARY.md      (per-layer table w/ specificity-adjusted Δ)"
echo "  Read: $OUT_DIR/summary.json    (machine-readable, per-layer null inside)"
echo "  Read: $OUT_DIR/by_pair.csv     ($n_main_forwards rows for plotting)"
echo "============================================================"
echo
echo "What to look for in SUMMARY.md:"
echo "  - 'specificity-adjusted Δ' column = the writeup-ready signal."
echo "  - Read top-to-bottom; the layer where it FIRST crosses ~+1 nat"
echo "    from baseline ~0 is the deliberation-circuit boundary."
echo "  - Plot Δ vs layer for the BlackboxNLP figure."
