#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Qwen behavioral NECESSITY test at the localized compute band.
#
# WHY THIS EXISTS
# ---------------
# The component sweep (results/causal_patching/qwen8b_l{18..21}_components/)
# showed Qwen's verb signal is INJECTED by ATTENTION across L18-L20 and is
# DISTRIBUTED (no sparse head: top heads at L19 are only 11-17%; L20 has a
# +30%/-19% cancelling pair). By L21-L23 the signal is residual flow-through
# (attn ratio ~0-8%). Sufficiency is already shown by activation patching.
#
# Necessity for a DISTRIBUTED circuit cannot be tested with a sparse head
# triplet (there isn't one). So we test:
#   (A) whole-attention-block ablation at each compute layer 18/19/20, plus
#       the saturation layer 23 and an early CONTROL layer 8; and
#   (B) a concentrated top-positive-head set at L19 and L20.
#
# CRITICAL READING: whole-attention ablation on every forward pass of a long
# CoT is BLUNT and may break generation generically. The SUMMARY reports
# FOLD-flip rate AND parse_fail_rate side by side. A genuine necessity result
# is a HIGH flip-to-CHECK with LOW parse_fail. A high parse_fail with low flip
# means we measured general damage, not FOLD-specific necessity -- report it
# honestly as such.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

export DEVICE="${DEVICE:-cuda}"
export DTYPE="${DTYPE:-bfloat16}"
export SEED="${SEED:-42}"
export MODEL=qwen
export PIPELINE=recon
export FILTER_RECORDED_BUCKET="${FILTER_RECORDED_BUCKET:-illegal_fold}"
export N_DECISIONS="${N_DECISIONS:-150}"      # capped at illegal_fold pool size
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export CONDITIONS="baseline"                  # custom sets added via HEAD_SETS
export FORCE_RERUN="${FORCE_RERUN:-0}"

# Compute band + saturation + control. Whole-attention block ablation.
WHOLE_LAYERS="${WHOLE_LAYERS:-18 19 20 23 8}"

echo "==================================================================="
echo " Qwen necessity: whole-attention ablation @ L={$WHOLE_LAYERS}"
echo "==================================================================="
for L in $WHOLE_LAYERS; do
  LAYER="$L" OUT_SUFFIX="_wholeattn" HEAD_SETS="whole_attn:all" \
    bash scripts/run_inference_head_ablation.sh
done

# Concentrated top-positive-head sets (from the L19/L20 component sweep).
echo "==================================================================="
echo " Qwen necessity: concentrated top-head sets @ L19, L20"
echo "==================================================================="
LAYER=19 OUT_SUFFIX="_topk" HEAD_SETS="topk_l19:31 3 21 1 0" \
  bash scripts/run_inference_head_ablation.sh
LAYER=20 OUT_SUFFIX="_topk" HEAD_SETS="topk_l20:29 15 16" \
  bash scripts/run_inference_head_ablation.sh

echo "==================================================================="
echo " DONE (pool=${FILTER_RECORDED_BUCKET}, n<=${N_DECISIONS}). Inspect:"
echo "   results/inference_head_ablation/qwen8b_l*_recon_${FILTER_RECORDED_BUCKET}_wholeattn/SUMMARY.md"
echo "   results/inference_head_ablation/qwen8b_l{19,20}_recon_${FILTER_RECORDED_BUCKET}_topk/SUMMARY.md"
echo " Compare each layer's net any-flip to the L8 CONTROL: a real localization"
echo " shows the compute band (L18-20, esp L19) well above L8 with low parse_fail."
echo " A flat profile = distributed/non-localized necessity (the n=24 read)."
echo "==================================================================="
