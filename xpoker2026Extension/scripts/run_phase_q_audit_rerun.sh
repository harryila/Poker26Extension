#!/usr/bin/env bash
# Phase Q audit rerun — fixes for the Ministral-vs-continuation discrepancy
# and other audit findings (see AUDIT_FINDINGS.md).
#
# Re-runs:
#   1. Inference head ablation with `--pipeline recon
#      --filter-recorded-bucket illegal_fold` (apples-to-apples with
#      continuation_after_patch's regenerate_ablated). All 3 models.
#   2. Continuation after patch with verb counts + flip-rate breakdown
#      (already produced by the new code; 1 GPU-hour per model).
#   3. Context-stratified by `pot_odds_quartile` on facing-bet decisions
#      only (`OUT_SUFFIX=_pot_odds`) to answer the original
#      equity-stratification question.
#
# Skips:
#   - Reverse FOLD→CHECK and BET→illegal_FOLD patching (already valid in
#     `c5eb2f2`/`6fe23d4`).
#   - Mode-balanced probe (Ministral n=16 broken; Qwen clean already).
#   - Tier 4 (target_bucket=clean_legal_fold; documented separately).
#
# Usage (GPU box):
#   cd xpoker2026Extension
#   export HF_HOME=/workspace/huggingface HF_TOKEN=...
#   FORCE_RERUN=1 bash scripts/run_phase_q_audit_rerun.sh
#
# Optional env:
#   MODELS="ministral llama qwen"     (default)
#   CONTINUE_TOKENS=180             (paper figure; default 80)
#   N_DECISIONS=80                   (per-model inference ablation pool size)
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-ministral llama qwen}"
CONTINUE_TOKENS="${CONTINUE_TOKENS:-80}"
N_DECISIONS="${N_DECISIONS:-80}"
export DEVICE="${DEVICE:-cuda}"
export DTYPE="${DTYPE:-bfloat16}"
export FORCE_RERUN="${FORCE_RERUN:-1}"

echo "=== Phase Q audit rerun $(date -u +%FT%TZ) ==="
echo "MODELS=$MODELS CONTINUE_TOKENS=$CONTINUE_TOKENS FORCE_RERUN=$FORCE_RERUN"

echo ""
echo "--- (1) inference head ablation: recon pipeline, illegal_fold filter ---"
for m in $MODELS; do
  echo "  [$m] inference ablation (recon, n=$N_DECISIONS) ..."
  MODEL="$m" PIPELINE=recon \
    FILTER_RECORDED_BUCKET=illegal_fold \
    N_DECISIONS="$N_DECISIONS" \
    bash scripts/run_inference_head_ablation.sh
done

echo ""
echo "--- (2) continuation after patch (verb-flip breakdown) ---"
for m in $MODELS; do
  MODEL="$m" CONTINUE_TOKENS="$CONTINUE_TOKENS" \
    bash scripts/run_continuation_after_patch.sh
done

echo ""
echo "--- (3) context-stratified by pot_odds_quartile (facing-bet pool) ---"
for m in $MODELS; do
  MODEL="$m" STRATIFY_BY=pot_odds_quartile OUT_SUFFIX=_pot_odds \
    bash scripts/run_context_stratified_patching.sh
done

echo ""
echo "=== audit rerun complete $(date -u +%FT%TZ) ==="
echo "Outputs:"
echo "  results/inference_head_ablation/{ministral,llama,qwen}8b_l*_recon_illegal_fold/"
echo "  results/continuation_after_patch/{ministral,llama,qwen}8b_l*/SUMMARY.md (overwritten)"
echo "  results/context_stratified_patching/{ministral,llama,qwen}8b_l*_pot_odds/"
