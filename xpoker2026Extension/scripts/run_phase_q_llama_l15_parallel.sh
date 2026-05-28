#!/usr/bin/env bash
# Llama L=15 parallel sweep — patching cells only.
#
# Why: pooled layer sweep (commit 6fe23d4) shows Llama L=14 is the *transition*
# layer (top-1 → CHECK 79%, spec-adj +6.5) and L=15 is *saturation* (top-1
# → CHECK 100%, spec-adj +10.2). Per the project's own §17g component
# decomposition (`results/causal_patching/llama8b_l15_components/`), no sparse
# head story exists at L=15 — heads at L=14 produce the verb signal, L=15+
# residual carries it. So:
#
#   - Patching cells (no head set required): can run at BOTH L=14 and L=15;
#     this script adds the L=15 parallel set.
#   - Head-ablation cells (require sparse head set): KEEP at L=14; L=15 has
#     no sparse head story, so ablation there isn't a meaningful necessity
#     test.
#
# Cells run here (all FORCE_RERUN-aware):
#   1. Reverse FOLD→CHECK at L=15
#   2. BET→illegal_FOLD at L=15
#   3. Context-stratified by street at L=15
#   4. (optional) Tier 4 at L=15 — set RUN_TIER4=1
#
# Usage:
#   cd xpoker2026Extension
#   FORCE_RERUN=1 bash scripts/run_phase_q_llama_l15_parallel.sh
set -uo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
SEED="${SEED:-42}"
LLAMA_LAYER=15

LOGS=(
  logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
  logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
  logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
)

run_patching() {
  local out="$1"; shift
  local source_bucket="$1"; shift
  local target_bucket="$1"; shift
  if [[ -f "$out/SUMMARY.md" ]] && [[ "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "[skip] $out"
    return 0
  fi
  mkdir -p "$out"
  python -m experiments.causal_patching \
    --enriched-log "${LOGS[@]}" \
    --source-bucket "$source_bucket" \
    --target-bucket "$target_bucket" \
    --layers "$LLAMA_LAYER" \
    --n-source "$N_SOURCE" --n-target "$N_TARGET" \
    --n-random-control 5 --n-random-target 10 \
    --seed "$SEED" \
    --out-dir "$out" \
    --device "$DEVICE" --dtype "$DTYPE"
}

echo "=== Llama L=15 parallel patching $(date -u +%FT%TZ) ==="

echo ""
echo "--- (1) reverse FOLD→CHECK at L=15 ---"
run_patching \
  "results/causal_patching/llama8b_reverse_fold_to_check_l15" \
  clean_legal_fold clean_check_or_call

echo ""
echo "--- (2) BET→illegal_FOLD at L=15 ---"
run_patching \
  "results/causal_patching/llama8b_bet_to_illegal_fold_l15" \
  clean_bet_or_raise illegal_fold

echo ""
echo "--- (3) context-stratified by street at L=15 ---"
MODEL=llama LAYER=15 OUT_SUFFIX=_street \
  bash scripts/run_context_stratified_patching.sh

if [[ "${RUN_TIER4:-0}" == "1" ]]; then
  echo ""
  echo "--- (4) Tier 4 at L=15 (opp-invariance) ---"
  echo "(Tier 4 driver auto-selects layer per model; need to override Llama=15)"
  echo "Run manually: MODELS=llama LLAMA_LAYER=15 bash scripts/run_tier4_patching.sh"
fi

echo ""
echo "[analyze] BET→illegal_FOLD top-1 rollup (across layers + models)"
python -m experiments.analyze_patching_top1_groups \
  --results-dir results/causal_patching \
  --glob '*bet_to_illegal_fold*' \
  --out results/causal_patching/SUMMARY_bet_to_illegal_fold_top1.md

echo ""
echo "=== Llama L=15 parallel done $(date -u +%FT%TZ) ==="
echo "New dirs:"
echo "  results/causal_patching/llama8b_reverse_fold_to_check_l15/"
echo "  results/causal_patching/llama8b_bet_to_illegal_fold_l15/"
echo "  results/context_stratified_patching/llama8b_l15_street/"
