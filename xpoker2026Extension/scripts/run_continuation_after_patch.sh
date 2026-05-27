#!/usr/bin/env bash
# Continuation / full-regenerate comparison after L* patch.
set -uo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
# Default 80 for iteration; paper figure: CONTINUE_TOKENS=180
CONTINUE_TOKENS="${CONTINUE_TOKENS:-80}"
MODEL="${MODEL:-ministral}"

case "$MODEL" in
  ministral)
    LAYER=16
    LOGS=(
      logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
    )
    OUT="results/continuation_after_patch/ministral8b_l16"
    ;;
  llama)
    LAYER=14
    LOGS=(
      logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
    )
    OUT="results/continuation_after_patch/llama8b_l14"
    ;;
  qwen)
    LAYER=23
    LOGS=(
      logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
    )
    OUT="results/continuation_after_patch/qwen8b_l23"
    ;;
  *)
    echo "Unknown MODEL=$MODEL" >&2
    exit 1
    ;;
esac

if [[ -f "$OUT/SUMMARY.md" ]] && [[ "${FORCE_RERUN:-0}" != "1" ]]; then
  echo "[skip] $OUT (set FORCE_RERUN=1 to override)"
  exit 0
fi

mkdir -p "$OUT"
python -m experiments.continuation_after_patch \
  --enriched-log "${LOGS[@]}" \
  --layer "$LAYER" \
  --source-bucket clean_check_or_call \
  --target-bucket illegal_fold \
  --n-target 25 \
  --n-source 5 \
  --continue-tokens "$CONTINUE_TOKENS" \
  --out-dir "$OUT" \
  --device "$DEVICE" --dtype "$DTYPE"

echo "[done] $OUT/SUMMARY.md"
