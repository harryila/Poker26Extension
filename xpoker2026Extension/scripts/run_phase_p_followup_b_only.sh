#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${HF_TOKEN:-}" ]]; then
  _tok="$(ps aux 2>/dev/null | grep -oP 'HF_TOKEN=\Khf_\S+' | head -1 || true)"
  [[ -n "$_tok" ]] && export HF_TOKEN="$_tok" HUGGING_FACE_HUB_TOKEN="$_tok"
  unset _tok
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "[error] HF_TOKEN required"; exit 1; }
source venv/bin/activate
export PYTHONUNBUFFERED=1 HF_HOME=/dev/shm/.hf_cache HF_HUB_CACHE=/dev/shm/.hf_cache/hub
export HF_HUB_DISABLE_XET=1 DEVICE=cuda DTYPE=bfloat16
mkdir -p "$HF_HUB_CACHE" logs
TS="$(date -u +%Y%m%d_%H%M%SZ)"
echo "=== (b) mode-balanced FORCE_RERUN start $(date -u +%FT%TZ) ===" | tee -a logs/phase_p_followup_wrapper.log
FORCE_RERUN=1 bash scripts/run_mode_balanced_direction_probe.sh 2>&1 | tee "logs/mode_balanced_followup_${TS}.log" | tee -a logs/phase_p_followup_wrapper.log
echo "=== (b) done exit=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a logs/phase_p_followup_wrapper.log
