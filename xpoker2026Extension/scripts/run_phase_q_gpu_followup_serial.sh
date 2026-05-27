#!/usr/bin/env bash
# Phase Q GPU follow-up — one HF model on disk at a time (tight /workspace quota).
#
# Re-runs fixed context + continuation (all models). Runs missing llama/qwen patching.
# Skips re-running ministral reverse/BET (already valid). Does NOT run head ablation.
#
# Usage:
#   cd xpoker2026Extension
#   export HF_HOME=/workspace/huggingface HF_TOKEN=...
#   bash scripts/run_phase_q_gpu_followup_serial.sh
#
# Env:
#   CONTINUE_TOKENS=180   (default 180)
#   STRATIFY_BY=street     (default)
#   HF_HUB_DISABLE_XET=1   (default 1 — smaller cache writes on MooseFS)
#   NO_TMUX=1              skip tmux wrapper
set -uo pipefail

SESSION_NAME="poker_phase_q_followup"

if [[ -z "${TMUX:-}" ]] && [[ -z "${NO_TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux not installed, or set NO_TMUX=1"
        exit 1
    fi
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists."
        echo "  Attach: tmux attach -t $SESSION_NAME"
        exit 1
    fi
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    LOG="$REPO_ROOT/logs/phase_q_followup_serial_$(date -u +%Y%m%d_%H%M%SZ).log"
    mkdir -p "$REPO_ROOT/logs"
    echo "Creating detached tmux session '$SESSION_NAME'"
    echo "  Attach:  tmux attach -t $SESSION_NAME"
    echo "  Log:     $LOG"
    tmux new-session -d -s "$SESSION_NAME" \
        "cd '$REPO_ROOT' && NO_TMUX=1 bash '$SCRIPT_PATH' 2>&1 | tee '$LOG'; echo; echo '[follow-up finished]'; sleep 999999"
    exit 0
fi

cd "$(dirname "$0")/.."
mkdir -p logs

if [[ -f venv/bin/activate ]]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
fi

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f /root/.hf_token ]]; then
    export HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "[error] HF_TOKEN required"; exit 1; }

export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export DEVICE="${DEVICE:-cuda}"
export DTYPE="${DTYPE:-bfloat16}"
export FORCE_RERUN="${FORCE_RERUN:-1}"
export CONTINUE_TOKENS="${CONTINUE_TOKENS:-180}"
export STRATIFY_BY="${STRATIFY_BY:-street}"

purge_hf_hub_dir() {
    local hub_dirname="$1"
    local target="${HF_HUB_CACHE%/}/${hub_dirname}"
    if [[ -d "$target" ]]; then
        echo "  [purge] rm -rf $target"
        rm -rf "$target"
    fi
    local xet_dir
    xet_dir="$(dirname "$HF_HUB_CACHE")/xet"
    if [[ -d "$xet_dir" ]]; then
        echo "  [purge] xet temp under $xet_dir"
        rm -rf "${xet_dir:?}"/*
    fi
    df -h "$HF_HOME" 2>/dev/null | tail -1 || true
}

run_context_and_continuation() {
    local model="$1"
    echo ""
    echo "======== context-stratified ($model) ========"
    MODEL="$model" bash scripts/run_context_stratified_patching.sh
    echo ""
    echo "======== continuation after patch ($model) ========"
    MODEL="$model" CONTINUE_TOKENS="$CONTINUE_TOKENS" bash scripts/run_continuation_after_patch.sh
}

echo "=== Phase Q serial follow-up $(date -u +%FT%TZ) ==="
echo "HF_HOME=$HF_HOME CONTINUE_TOKENS=$CONTINUE_TOKENS STRATIFY_BY=$STRATIFY_BY"

# --- 1) Ministral: fixed metrics only (keep existing reverse/BET) ---
echo ""
echo "########## MODEL 1/3: ministral (context + continuation only) ##########"
run_context_and_continuation ministral
purge_hf_hub_dir "models--mistralai--Ministral-8B-Instruct-2410"

# --- 2) Llama: missing patching + fixed context/continuation ---
echo ""
echo "########## MODEL 2/3: llama ##########"
export PATCHING_MODELS=llama
export SKIP_TOP1_ANALYZE=1
bash scripts/run_causal_patching_reverse_full.sh
bash scripts/run_causal_patching_bet_to_illegal_fold.sh
unset SKIP_TOP1_ANALYZE
run_context_and_continuation llama
purge_hf_hub_dir "models--meta-llama--Llama-3.1-8B-Instruct"

# --- 3) Qwen ---
echo ""
echo "########## MODEL 3/3: qwen ##########"
export PATCHING_MODELS=qwen
export SKIP_TOP1_ANALYZE=1
bash scripts/run_causal_patching_reverse_full.sh
bash scripts/run_causal_patching_bet_to_illegal_fold.sh
unset SKIP_TOP1_ANALYZE PATCHING_MODELS
run_context_and_continuation qwen
purge_hf_hub_dir "models--Qwen--Qwen3-8B"

echo ""
echo "[analyze] BET→illegal_fold top-1 rollup (all models) ..."
python -m experiments.analyze_patching_top1_groups \
  --results-dir results/causal_patching \
  --glob '*bet_to_illegal_fold*' \
  --out results/causal_patching/SUMMARY_bet_to_illegal_fold_top1.md

echo ""
echo "=== Phase Q serial follow-up COMPLETE $(date -u +%FT%TZ) ==="
echo "Expected new/updated dirs:"
echo "  results/causal_patching/llama8b_* qwen8b_* (reverse + bet)"
echo "  results/context_stratified_patching/*/"
echo "  results/continuation_after_patch/*/"
