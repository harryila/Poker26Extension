#!/usr/bin/env bash
# Behavioral head ablation during live CoT generation (Ministral default).
set -uo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_DECISIONS="${N_DECISIONS:-200}"
SEED="${SEED:-42}"

# Paper figure: POOLED=1 (default). Quick iteration: POOLED=0 N_DECISIONS=50
if [[ "${POOLED:-1}" == "1" ]]; then
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
