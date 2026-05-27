#!/usr/bin/env bash
# Full reverse-direction patching: clean_legal_fold → clean_check_or_call at L*.
# (Forward CHECK→illegal_fold already exists in layer sweeps.)
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
  local out="results/causal_patching/${short}8b_reverse_fold_to_check_l${layer}"
  if [[ -f "$out/SUMMARY.md" ]] && [[ "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "[skip] $out"
    return 0
  fi
  mkdir -p "$out"
  python -m experiments.causal_patching \
    --enriched-log "${logs[@]}" \
    --source-bucket clean_legal_fold \
    --target-bucket clean_check_or_call \
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

_want_model() {
  local m="$1"
  local list="${PATCHING_MODELS:-ministral llama qwen}"
  [[ " $list " == *" $m "* ]]
}

if _want_model llama; then
  run_cell llama 14 "${LLAMA_LOGS[@]}"
fi
if _want_model ministral; then
  run_cell ministral 16 "${MINI_LOGS[@]}"
fi
if _want_model qwen; then
  run_cell qwen 23 "${QWEN_LOGS[@]}"
fi

echo "[done] reverse fold→check cells (PATCHING_MODELS=${PATCHING_MODELS:-ministral llama qwen})"
