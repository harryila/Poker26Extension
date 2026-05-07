#!/usr/bin/env bash
# =============================================================================
# Causal patching layer sweep for QWEN 8B (8B-family generalization, Phase H).
# =============================================================================
#
# Companion to run_causal_patching_layer_sweep.sh (Ministral) and
# run_causal_patching_llama_sweep.sh (Llama). Tests Qwen 8B because:
#
#   - The §13 logit-lens AND the new CROSS_CELL_DETAILED.md analysis show
#     Qwen has Δ ≈ 0 (no detectable late-layer revision pattern).
#   - The §12 EV analysis shows Qwen doesn't benefit from CoT (slightly
#     worse, in fact).
#   - These two observations are MUTUALLY CONSISTENT: Qwen lacks both the
#     behavioral pathology AND the mechanistic circuit.
#
# Causal patching on Qwen is the strongest test of this consistency. Predicted
# outcome:
#
#   - If Qwen also shows a sharp L=14-16 transition like Llama/Ministral,
#     the absence of late-layer revision in §13 was just measurement noise
#     and the circuit is universal across the 8B family.
#
#   - If Qwen shows NO sharp transition (effect is null or gradual across all
#     layers), then the circuit is REAL and Qwen lacks it. That would be a
#     stronger story: "the deliberation circuit is what makes CoT 'work' for
#     small models; Qwen lacks it, hence Qwen doesn't benefit from CoT."
#
# Either result is publishable for BlackboxNLP.
#
# Qwen 8B has 36 layers (same as Ministral). Expected analog: L=14-16 if
# circuit is universal at the same absolute depth.
#
# SCOPE options
# -------------
# - SCOPE=s42_only:
#     Only 4 illegal_FOLDs in s42 t=0. Statistically thin.
#     Wall-clock ~1.5 h on H100 but interpretable for "Qwen on s42".
#
# - SCOPE=pooled (recommended):
#     Pool s42 + s123 + s456 t=0 → 24 illegal_FOLDs. n_target=24 (all of them).
#     ~36 layers x 10 src x 24 tgt = 8,640 forwards. Wall-clock ~5 h.
#
# Outputs:
#   results/causal_patching/qwen8b_t0_<scope>_layer_sweep/
# =============================================================================

set -euo pipefail

SCOPE="${SCOPE:-pooled}"  # pooled is the recommended default for Qwen
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LAYERS="${LAYERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35}"
N_SOURCE="${N_SOURCE:-10}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
SEED="${SEED:-42}"

cd "$(dirname "$0")/.."  # repo root

case "$SCOPE" in
    s42_only)
        ENRICHED_LOGS="logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz"
        N_TARGET="${N_TARGET:-4}"
        OUT_DIR="${OUT_DIR:-results/causal_patching/qwen8b_t0_s42_layer_sweep}"
        ;;
    pooled)
        ENRICHED_LOGS="logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
                        logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
                        logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz"
        N_TARGET="${N_TARGET:-24}"
        OUT_DIR="${OUT_DIR:-results/causal_patching/qwen8b_t0_pooled_layer_sweep}"
        ;;
    *)
        echo "ERROR: unknown SCOPE='$SCOPE' (use s42_only|pooled)"
        exit 1
        ;;
esac

mkdir -p "$OUT_DIR"

n_layers=$(echo "$LAYERS" | wc -w | tr -d ' ')
echo "============================================================"
echo "Causal patching LAYER SWEEP — QWEN 8B"
echo "============================================================"
echo "  Scope:           $SCOPE"
echo "  Enriched logs:   $(echo $ENRICHED_LOGS | tr ' ' '\n' | wc -l | tr -d ' ') log(s)"
echo "  Layers:          $n_layers (0..$(( n_layers - 1 )))"
echo "  Sources:         $N_SOURCE clean CHECK_OR_CALL"
echo "  Targets:         $N_TARGET illegal_FOLD"
echo "  Out dir:         $OUT_DIR"
echo "============================================================"
echo
echo "Hypothesis to test:"
echo "  H1 (universal circuit): sharp L=14-16 transition like Llama/Ministral"
echo "  H2 (Qwen lacks circuit): null effect or gradual non-localized rise"
echo "  Either result is publishable for BlackboxNLP."
echo

first_log=$(echo $ENRICHED_LOGS | awk '{print $1}')
echo "[pre-flight 1/2] verify position mapping ..."
python -m experiments.verify_position_mapping \
    --enriched-log "$first_log" --n-samples 10 \
    || { echo "ERROR: position-mapping failed."; exit 1; }
echo

echo "[pre-flight 2/2] verify prompt reconstruction (GPU) ..."
python -m experiments.verify_prompt_reconstruction \
    --enriched-log "$first_log" --n-samples 5 \
    --device "$DEVICE" --dtype "$DTYPE" \
    || { echo "ERROR: prompt-reconstruction failed."; exit 1; }
echo

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
echo "Qwen sweep COMPLETE. Read: $OUT_DIR/SUMMARY.md"
echo "  Compare against Ministral L=14-16 sharp transition."
echo "============================================================"
