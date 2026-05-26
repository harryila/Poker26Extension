#!/usr/bin/env bash
# Context-stratified patching (pot-odds quartiles on CHECK sources).
set -uo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MODEL="${MODEL:-ministral}"

case "$MODEL" in
  ministral)
    LAYER=16
    LOGS=(
      logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
    )
    OUT="results/context_stratified_patching/ministral8b_l16"
    ;;
  llama)
    LAYER=14
    LOGS=(
      logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
    )
    OUT="results/context_stratified_patching/llama8b_l14"
    ;;
  qwen)
    LAYER=23
    LOGS=(
      logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
    )
    OUT="results/context_stratified_patching/qwen8b_l23"
    ;;
  *)
    echo "Unknown MODEL=$MODEL" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT"
python -m experiments.context_stratified_patching \
  --enriched-log "${LOGS[@]}" \
  --layer "$LAYER" \
  --source-bucket clean_check_or_call \
  --target-bucket illegal_fold \
  --n-source-per-stratum 5 \
  --n-target 20 \
  --out-dir "$OUT" \
  --device "$DEVICE" --dtype "$DTYPE"

echo "[done] $OUT/SUMMARY.md"
