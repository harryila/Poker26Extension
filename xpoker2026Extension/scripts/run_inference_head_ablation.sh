#!/usr/bin/env bash
# Behavioral head ablation during live CoT generation (per-model).
# Default pipeline: PromptReconstructor + raw model.generate (matches
# continuation_after_patch's regenerate_ablated). Filters to recorded
# illegal_fold targets to be apples-to-apples with continuation.
set -uo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"
MODEL="${MODEL:-ministral}"
PIPELINE="${PIPELINE:-recon}"               # recon (default) | hfagent
FILTER="${FILTER_RECORDED_BUCKET:-illegal_fold}"
N_DECISIONS="${N_DECISIONS:-150}"           # capped at pool size
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

case "$MODEL" in
  ministral)
    LAYER="${LAYER:-16}"
    LOGS=(
      logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
    )
    SHORT="ministral8b"
    ;;
  llama)
    LAYER="${LAYER:-14}"
    LOGS=(
      logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
    )
    SHORT="llama8b"
    ;;
  qwen)
    LAYER="${LAYER:-23}"
    LOGS=(
      logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
      logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
    )
    SHORT="qwen8b"
    ;;
  *)
    echo "Unknown MODEL=$MODEL" >&2
    exit 1
    ;;
esac

# Space-separated condition list: e.g. CONDITIONS="baseline triplet extended control"
CONDITIONS="${CONDITIONS:-baseline triplet control}"
# Optional suffix on out-dir to keep multiple variants (e.g. "_sextet", "_negctrl").
OUT_SUFFIX="${OUT_SUFFIX:-}"

OUT_BASENAME="${SHORT}_l${LAYER}_${PIPELINE}"
[[ -n "$FILTER" ]] && OUT_BASENAME="${OUT_BASENAME}_${FILTER}"
OUT_BASENAME="${OUT_BASENAME}${OUT_SUFFIX}"
OUT="results/inference_head_ablation/${OUT_BASENAME}"

if [[ -f "$OUT/SUMMARY.md" ]] && [[ "${FORCE_RERUN:-0}" != "1" ]]; then
  echo "[skip] $OUT (set FORCE_RERUN=1 to override)"
  exit 0
fi
mkdir -p "$OUT"

ARGS=(
  --enriched-log "${LOGS[@]}"
  --out-dir "$OUT"
  --n-decisions "$N_DECISIONS"
  --seed "$SEED"
  --layer "$LAYER"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --pipeline "$PIPELINE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --conditions $CONDITIONS
)
if [[ -n "$FILTER" ]]; then
  ARGS+=( --filter-recorded-bucket "$FILTER" )
fi

python -m experiments.inference_head_ablation "${ARGS[@]}"
echo "[done] $OUT/SUMMARY.md"
