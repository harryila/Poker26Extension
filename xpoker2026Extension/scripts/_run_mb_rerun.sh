#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
export HF_TOKEN=$(ps aux 2>/dev/null | grep -oP 'HF_TOKEN=\Khf_\S+' | head -1)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=/dev/shm/.hf_cache HF_HUB_CACHE=/dev/shm/.hf_cache/hub HF_HUB_DISABLE_XET=1
export DEVICE=cuda DTYPE=bfloat16 PYTHONUNBUFFERED=1
mkdir -p "$HF_HUB_CACHE" logs
TS=$(date -u +%Y%m%d_%H%M%SZ)
FORCE_RERUN=1 bash scripts/run_mode_balanced_direction_probe.sh 2>&1 | tee "logs/mode_balanced_followup_${TS}.log"
echo "=== done exit=${PIPESTATUS[0]} $(date -u +%FT%TZ) ==="
