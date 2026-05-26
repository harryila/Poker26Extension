#!/usr/bin/env bash
# Behavioral head ablation during live CoT generation (Ministral default).
set -uo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_DECISIONS="${N_DECISIONS:-150}"
SEED="${SEED:-42}"

# Pooled CoT logitlens logs (3 seeds) — use s42 only for speed unless POOLED=1
if [[ "${POOLED:-0}" == "1" ]]; then
  LOGS=(
    logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
    logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
    logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
  )
  LOG_ARG=(--enriched-log "${LOGS[@]}")
else
  LOG="logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz"
  if [[ ! -f "$LOG" ]]; then
    LOG="logs/cot_ministral8b_t0_s42_informative_v2_enriched.jsonl.gz"
  fi
  LOG_ARG=(--enriched-log "$LOG")
fi

OUT="results/inference_head_ablation/ministral8b_l16_cot"
mkdir -p "$OUT"

python -m experiments.inference_head_ablation \
  "${LOG_ARG[@]}" \
  --out-dir "$OUT" \
  --n-decisions "$N_DECISIONS" \
  --seed "$SEED" \
  --layer 16 \
  --device "$DEVICE" \
  --conditions baseline triplet control

echo "[done] $OUT/SUMMARY.md"
