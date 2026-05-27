#!/usr/bin/env bash
# GPU follow-up after disk increase: re-run fixed cells + missing Llama/Qwen.
#
# Usage (on GPU box):
#   cd xpoker2026Extension
#   export HF_HOME=/workspace/huggingface HF_TOKEN=...
#   FORCE_RERUN=1 bash scripts/run_phase_q_gpu_followup.sh
#
# Optional:
#   MODELS="ministral llama qwen"   (default: all three)
#   CONTINUE_TOKENS=180           (paper figure; default 80)
#   STRATIFY_BY=street            (default; or facing_bet, pot_odds_quartile)
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-ministral llama qwen}"
CONTINUE_TOKENS="${CONTINUE_TOKENS:-80}"
STRATIFY_BY="${STRATIFY_BY:-street}"
export DEVICE="${DEVICE:-cuda}"
export DTYPE="${DTYPE:-bfloat16}"
export FORCE_RERUN="${FORCE_RERUN:-1}"
export STRATIFY_BY

echo "=== Phase Q GPU follow-up $(date -u +%FT%TZ) ==="
echo "MODELS=$MODELS CONTINUE_TOKENS=$CONTINUE_TOKENS STRATIFY_BY=$STRATIFY_BY FORCE_RERUN=$FORCE_RERUN"

echo "--- reverse FOLD→CHECK (llama + qwen; ministral skipped if present) ---"
bash scripts/run_causal_patching_reverse_full.sh

echo "--- BET→illegal_FOLD ---"
bash scripts/run_causal_patching_bet_to_illegal_fold.sh

for m in $MODELS; do
  echo "--- context-stratified ($m) ---"
  MODEL="$m" bash scripts/run_context_stratified_patching.sh
done

for m in $MODELS; do
  echo "--- continuation after patch ($m) ---"
  MODEL="$m" CONTINUE_TOKENS="$CONTINUE_TOKENS" bash scripts/run_continuation_after_patch.sh
done

echo "=== follow-up complete $(date -u +%FT%TZ) ==="
