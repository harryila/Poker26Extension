#!/usr/bin/env bash
# Context-stratified patching (pot-odds quartiles on CHECK sources).
set -uo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MODEL="${MODEL:-ministral}"
STRATIFY_BY="${STRATIFY_BY:-street}"        # street | facing_bet | pot_odds_quartile | pot_total_quartile
OUT_SUFFIX="${OUT_SUFFIX:-}"                # e.g. "_pot_odds" to keep multiple stratifications

case "$MODEL" in
  ministral)
    LAYER="${LAYER:-16}"
    LOGS=(
      logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
    )
    OUT="results/context_stratified_patching/ministral8b_l${LAYER}${OUT_SUFFIX}"
    ;;
  llama)
    LAYER="${LAYER:-14}"
    LOGS=(
      logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
    )
    OUT="results/context_stratified_patching/llama8b_l${LAYER}${OUT_SUFFIX}"
    ;;
  qwen)
    LAYER="${LAYER:-23}"
    LOGS=(
      logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
    )
    OUT="results/context_stratified_patching/qwen8b_l${LAYER}${OUT_SUFFIX}"
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
python -m experiments.context_stratified_patching \
  --enriched-log "${LOGS[@]}" \
  --layer "$LAYER" \
  --stratify-by "$STRATIFY_BY" \
  --source-bucket clean_check_or_call \
  --target-bucket illegal_fold \
  --n-source-per-stratum 5 \
  --n-target 20 \
  --out-dir "$OUT" \
  --device "$DEVICE" --dtype "$DTYPE"

echo "[done] $OUT/SUMMARY.md"
