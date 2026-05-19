#!/usr/bin/env bash
# Phase P follow-up: (a) opp-preset recon diagnostic, then (b) mode-balanced probe.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ -f /root/.hf_token ]]; then
        HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"
        export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    else
        _tok="$(ps aux 2>/dev/null | grep -oP 'HF_TOKEN=\Khf_\S+' | head -1 || true)"
        if [[ -n "$_tok" ]]; then
            export HF_TOKEN="$_tok" HUGGING_FACE_HUB_TOKEN="$_tok"
        fi
        unset _tok
    fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[error] HF_TOKEN not set. export HF_TOKEN=... or write token to /root/.hf_token"
    exit 1
fi

source venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/dev/shm/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET=1
export DEVICE="${DEVICE:-cuda}"
export DTYPE="${DTYPE:-bfloat16}"
mkdir -p "$HF_HUB_CACHE" logs

TS="$(date -u +%Y%m%d_%H%M%SZ)"
LOG_A="logs/opp_reconstruction_diagnostic_${TS}.log"
LOG_B="logs/mode_balanced_followup_${TS}.log"

echo "=== (a) opp-preset reconstruction diagnostic start $(date -u +%FT%TZ) ==="
set +e
python -m experiments.diagnose_opp_preset_reconstruction \
    --print-first-mismatch \
    2>&1 | tee "$LOG_A"
_a_exit=${PIPESTATUS[0]}
set -e
echo "=== (a) done exit=${_a_exit} log=$LOG_A ==="

echo
echo "=== (b) mode-balanced probe (FORCE_RERUN=1) start $(date -u +%FT%TZ) ==="
FORCE_RERUN=1 bash scripts/run_mode_balanced_direction_probe.sh 2>&1 | tee "$LOG_B"
echo "=== (b) done exit=${PIPESTATUS[0]} log=$LOG_B ==="
