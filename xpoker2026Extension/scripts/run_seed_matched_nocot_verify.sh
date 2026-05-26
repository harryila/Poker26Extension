#!/usr/bin/env bash
# =============================================================================
# Seed-matched non-CoT verification (no 10h re-inference unless logs missing).
# =============================================================================
#
# Tier 1A.small non-CoT runs use the SAME seeds (42,123,456) and opponent
# (informative_v2) as CoT logs. Cross-mode pairing uses (seed, decision_idx)
# per updates.md §22j-bis — NOT hand_id (random UUID per reset).
#
# This script:
#   1. Verifies scaled_* non-CoT enriched logs exist for each model.
#   2. Optionally runs tier1a_small for a missing model (RUN_MISSING=1).
#   3. Runs mode-balanced probe (all three models).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

check_log() {
  local path="$1"
  if [[ -f "$path" ]] || [[ -f "${path}.gz" ]]; then
    echo "[ok] $path"
    return 0
  fi
  echo "[MISSING] $path"
  return 1
}

MISSING=0
for m in llama8b qwen8b ministral8b; do
  check_log "logs/scaled_${m}_t0_s42_informative_v2_enriched.jsonl" || MISSING=1
done

if [[ "$MISSING" == "1" ]] && [[ "${RUN_MISSING:-0}" == "1" ]]; then
  echo "[run] launching tier1a_small non-CoT baseline (all models) ..."
  bash scripts/run_tier1a_small.sh
else
  if [[ "$MISSING" == "1" ]]; then
    echo "[warn] Some scaled logs missing. Set RUN_MISSING=1 to generate, or"
    echo "       copy logs from the GPU box before mode-balanced probe."
  fi
fi

echo "[run] mode-balanced direction probe (llama + qwen + ministral) ..."
bash scripts/run_mode_balanced_direction_probe.sh

echo "[done] seed-matched verify + mode-balanced probe"
