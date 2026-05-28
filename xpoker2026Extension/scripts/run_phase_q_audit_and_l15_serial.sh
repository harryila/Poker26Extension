#!/usr/bin/env bash
# =============================================================================
# Phase Q — full audit rerun + Llama L=15 parallel + Tier 4 L=15 (all-in-one).
# =============================================================================
#
# Runs every Phase Q audit follow-up the project has, serialized so only one
# HF model lives on /workspace at a time (tight RunPod quota).
#
# Order:
#   For each model in {ministral, llama, qwen}:
#     a) inference_head_ablation  (pipeline=recon, filter=illegal_fold)
#     b) continuation_after_patch (CONTINUE_TOKENS=180)
#     c) context_stratified_patching (STRATIFY_BY=pot_odds_quartile, OUT_SUFFIX=_pot_odds)
#     -> purge HF cache for that model
#   Then Llama-only L=15:
#     d) reverse FOLD->CHECK at L=15
#     e) BET->illegal_FOLD at L=15
#     f) context_stratified by street at L=15
#     g) Tier 4 at L=15 for llama-8b (uses LLAMA_LAYER override)
#     -> purge llama
#   Final BET->illegal_FOLD top-1 rollup across layers/models.
#
# Usage (GPU box):
#   cd xpoker2026Extension
#   export HF_HOME=/workspace/huggingface HF_TOKEN=...
#   bash scripts/run_phase_q_audit_and_l15_serial.sh
#
# Env knobs:
#   MODELS="ministral llama qwen"
#   CONTINUE_TOKENS=180
#   N_DECISIONS=80    (per-model inference ablation pool size)
#   FORCE_RERUN=1     (default)
#   NO_TMUX=1         (skip tmux wrapper)
# =============================================================================
set -uo pipefail

SESSION_NAME="poker_phase_q_audit_l15"

if [[ -z "${TMUX:-}" ]] && [[ -z "${NO_TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux not installed (apt install tmux), or set NO_TMUX=1"
        exit 1
    fi
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists."
        echo "  Attach: tmux attach -t $SESSION_NAME"
        echo "  Kill:   tmux kill-session -t $SESSION_NAME"
        exit 1
    fi
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    LOG="$REPO_ROOT/logs/phase_q_audit_l15_$(date -u +%Y%m%d_%H%M%SZ).log"
    mkdir -p "$REPO_ROOT/logs"
    echo "Creating detached tmux session '$SESSION_NAME'"
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Log:    $LOG"
    tmux new-session -d -s "$SESSION_NAME" \
        "cd '$REPO_ROOT' && NO_TMUX=1 bash '$SCRIPT_PATH' 2>&1 | tee '$LOG'; echo; echo '[audit+L15 finished]'; sleep 999999"
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

MODELS="${MODELS:-ministral llama qwen}"
CONTINUE_TOKENS="${CONTINUE_TOKENS:-180}"
N_DECISIONS="${N_DECISIONS:-80}"

hub_dir_for_model() {
    case "$1" in
        ministral) echo "models--mistralai--Ministral-8B-Instruct-2410" ;;
        llama)     echo "models--meta-llama--Llama-3.1-8B-Instruct" ;;
        qwen)      echo "models--Qwen--Qwen3-8B" ;;
    esac
}

purge_hf_for_model() {
    local m="$1"
    local hub_dirname
    hub_dirname="$(hub_dir_for_model "$m")"
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

run_audit_for_model() {
    local m="$1"
    echo ""
    echo "######### MODEL $m: audit rerun (3 sections) #########"

    echo ""
    echo "------ ($m) inference head ablation (recon, illegal_fold) ------"
    MODEL="$m" PIPELINE=recon \
        FILTER_RECORDED_BUCKET=illegal_fold \
        N_DECISIONS="$N_DECISIONS" \
        bash scripts/run_inference_head_ablation.sh

    echo ""
    echo "------ ($m) continuation after patch (CONTINUE_TOKENS=$CONTINUE_TOKENS) ------"
    MODEL="$m" CONTINUE_TOKENS="$CONTINUE_TOKENS" \
        bash scripts/run_continuation_after_patch.sh

    echo ""
    echo "------ ($m) context-stratified by pot_odds_quartile ------"
    MODEL="$m" STRATIFY_BY=pot_odds_quartile OUT_SUFFIX=_pot_odds \
        bash scripts/run_context_stratified_patching.sh
}

echo "=== Phase Q audit + L=15 serial run $(date -u +%FT%TZ) ==="
echo "MODELS=$MODELS  CONTINUE_TOKENS=$CONTINUE_TOKENS  N_DECISIONS=$N_DECISIONS"
echo "HF_HOME=$HF_HOME  FORCE_RERUN=$FORCE_RERUN"

# Per-model audit rerun, purging HF cache after each.
for m in $MODELS; do
    run_audit_for_model "$m"
    purge_hf_for_model "$m"
done

# Llama L=15 parallel cells (re-downloads llama).
echo ""
echo "######### Llama L=15 PARALLEL #########"
bash scripts/run_phase_q_llama_l15_parallel.sh

# Tier 4 at L=15 for llama (uses LLAMA_LAYER override from our patched script).
echo ""
echo "######### Tier 4 at L=15 (llama-only) #########"
MODELS=llama-8b LLAMA_LAYER=15 bash scripts/run_tier4_patching.sh

purge_hf_for_model llama

# Final BET top-1 rollup across all layers/models.
echo ""
echo "[analyze] BET->illegal_FOLD top-1 rollup (final) ..."
python -m experiments.analyze_patching_top1_groups \
  --results-dir results/causal_patching \
  --glob '*bet_to_illegal_fold*' \
  --out results/causal_patching/SUMMARY_bet_to_illegal_fold_top1.md

echo ""
echo "=== Phase Q audit + L=15 serial COMPLETE $(date -u +%FT%TZ) ==="
echo "Outputs:"
echo "  results/inference_head_ablation/*_recon_illegal_fold/"
echo "  results/continuation_after_patch/{ministral,llama,qwen}8b_l*/"
echo "  results/context_stratified_patching/*_pot_odds/"
echo "  results/causal_patching/llama8b_{reverse_fold_to_check,bet_to_illegal_fold}_l15/"
echo "  results/context_stratified_patching/llama8b_l15_street/"
echo "  results/causal_patching/tier4_*_llama_l15/"
echo "  results/causal_patching/SUMMARY_bet_to_illegal_fold_top1.md (refreshed)"
