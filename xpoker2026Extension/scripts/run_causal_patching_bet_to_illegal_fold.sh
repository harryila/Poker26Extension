#!/usr/bin/env bash
# Verb-generality: clean_bet_or_raise source → illegal_fold targets at L*.
set -uo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
SEED="${SEED:-42}"

run_cell() {
  local short="$1"
  local layer="$2"
  shift 2
  local logs=("$@")
  local out="results/causal_patching/${short}8b_bet_to_illegal_fold_l${layer}"
  if [[ -f "$out/SUMMARY.md" ]] && [[ "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "[skip] $out"
    return 0
  fi
  mkdir -p "$out"
  python -m experiments.causal_patching \
    --enriched-log "${logs[@]}" \
    --source-bucket clean_bet_or_raise \
    --target-bucket illegal_fold \
    --layers "$layer" \
    --n-source "$N_SOURCE" --n-target "$N_TARGET" \
    --n-random-control 5 --n-random-target 10 \
    --seed "$SEED" \
    --out-dir "$out" \
    --device "$DEVICE" --dtype "$DTYPE"
}

LLAMA_LOGS=(
  logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
  logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
  logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
)
MINI_LOGS=(
  logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
  logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
  logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
)
QWEN_LOGS=(
  logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
  logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
  logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
)

run_cell llama 14 "${LLAMA_LOGS[@]}"
run_cell ministral 16 "${MINI_LOGS[@]}"
run_cell qwen 23 "${QWEN_LOGS[@]}"

echo "[analyze] patched top-1 families (BET-generality readout) ..."
python -m experiments.analyze_patching_top1_groups \
  --results-dir results/causal_patching \
  --glob '*bet_to_illegal_fold*' \
  --out results/causal_patching/SUMMARY_bet_to_illegal_fold_top1.md

echo "[done] bet→illegal_fold cells + SUMMARY_bet_to_illegal_fold_top1.md"
