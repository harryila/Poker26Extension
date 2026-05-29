#!/usr/bin/env bash
# =============================================================================
# Phase Q — FINAL remaining GPU work (everything still pending, nothing optional).
# =============================================================================
#
# Runs the two remaining GPU tasks serially, one HF model on disk at a time:
#
#   TASK 1 — Qwen compute-layer localization (component sweep L=18..21).
#            Fills the gap below the existing L=22/23/24 decompositions so we
#            can name WHERE Qwen's attention/MLP injects the verb signal
#            (symmetric to Llama L=14). AUDIT_FINDINGS.md §11.
#
#   TASK 2 — Tier 4 distinct-seed regeneration (all 3 models, 5 presets).
#            Decorrelates the opponent RNG per preset so the collapsed presets
#            (Qwen 3 distinct, Llama 4, Ministral 4 — AUDIT §12) become 5
#            genuinely-distinct, independently-sampled opponent distributions,
#            then re-patches at each model's L*. AUDIT_FINDINGS.md §12.
#
#   TASK 3 — Qwen necessity ablation at the compute band located by Task 1
#            (whole-attention ablation @ L18/19/20 + saturation L23 + control
#            L8, plus concentrated top-head sets). AUDIT_FINDINGS.md §11.
#            On a re-run this consumes the already-written Task 1 sweep.
#
# Usage (GPU box):
#   cd xpoker2026Extension
#   export HF_HOME=/workspace/huggingface HF_TOKEN=...
#   bash scripts/run_phase_q_final_gpu.sh
#
# Env knobs:
#   FORCE_RERUN=1   (default)
#   NO_TMUX=1       skip tmux wrapper
#   QWEN_LAYERS="18 19 20 21"
#   TIER4_MODELS="llama-8b qwen-8b ministral-8b"
# =============================================================================
set -uo pipefail

SESSION_NAME="poker_phase_q_final"

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
    LOG="$REPO_ROOT/logs/phase_q_final_$(date -u +%Y%m%d_%H%M%SZ).log"
    mkdir -p "$REPO_ROOT/logs"
    echo "Creating detached tmux session '$SESSION_NAME'"
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Log:    $LOG"
    tmux new-session -d -s "$SESSION_NAME" \
        "cd '$REPO_ROOT' && NO_TMUX=1 bash '$SCRIPT_PATH' 2>&1 | tee '$LOG'; echo; echo '[phase_q_final finished]'; sleep 999999"
    exit 0
fi

cd "$(dirname "$0")/.."
mkdir -p logs
export FORCE_RERUN="${FORCE_RERUN:-1}"

# HF cache env + token so per-model purges target the right dir on tight quota.
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f /root/.hf_token ]]; then
    export HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

hub_dir_for_tier4_model() {
    case "$1" in
        llama-8b)     echo "models--meta-llama--Llama-3.1-8B-Instruct" ;;
        qwen-8b)      echo "models--Qwen--Qwen3-8B" ;;
        ministral-8b) echo "models--mistralai--Ministral-8B-Instruct-2410" ;;
    esac
}
purge_hf_hub() {
    local hub_dirname="$1"
    local target="${HF_HUB_CACHE%/}/${hub_dirname}"
    [[ -d "$target" ]] && { echo "  [purge] rm -rf $target"; rm -rf "$target"; }
    local xet_dir; xet_dir="$(dirname "$HF_HUB_CACHE")/xet"
    [[ -d "$xet_dir" ]] && { echo "  [purge] xet temp"; rm -rf "${xet_dir:?}"/*; }
    df -h "$HF_HOME" 2>/dev/null | tail -1 || true
}

echo "=== Phase Q FINAL GPU run $(date -u +%FT%TZ) ==="

# -------------------------------------------------------------------------
# TASK 1 — Qwen compute-layer component sweep (purges Qwen weights at end).
# -------------------------------------------------------------------------
echo ""
echo "######### TASK 1/2: Qwen compute-layer sweep (L=${QWEN_LAYERS:-18 19 20 21}) #########"
QWEN_LAYERS="${QWEN_LAYERS:-18 19 20 21}" PURGE=1 \
    bash scripts/run_qwen_compute_layer_sweep.sh

# -------------------------------------------------------------------------
# TASK 2 — Tier 4 distinct-seed regeneration (all 3 models).
# Run one model at a time and purge its HF weights afterward so the tight
# /workspace quota (~20 GB) never has to hold two 8B models at once.
# -------------------------------------------------------------------------
echo ""
echo "######### TASK 2/2: Tier 4 distinct-seed regeneration (all 3 models, serial) #########"
for _m in ${TIER4_MODELS:-llama-8b qwen-8b ministral-8b}; do
    echo ""
    echo "===== Tier 4 distinct-seed: $_m ====="
    MODELS="$_m" bash scripts/run_tier4_regen_distinct_presets.sh
    purge_hf_hub "$(hub_dir_for_tier4_model "$_m")"
done

# -------------------------------------------------------------------------
# Refresh the cross-preset overlap diagnostic on the NEW distinct-seed logs.
# (CPU-only; safe to run here.)
# -------------------------------------------------------------------------
echo ""
echo "[verify] distinct-seed preset overlap (targets *_distinctseed logs) ..."
python -m experiments.diagnose_tier4_preset_overlap \
    --models llama-8b qwen-8b ministral-8b \
    --log-suffix _distinctseed \
    --out results/diagnostics/tier4_preset_overlap_distinctseed/SUMMARY.md

# -------------------------------------------------------------------------
# TASK 3 — Qwen necessity ablation at the localized compute band.
# Unblocked by Task 1: the sweep placed Qwen's verb compute in ATTENTION
# across L18-L20 (distributed; no sparse head), residual flow-through by
# L21-23. Since there is no sparse triplet, necessity is tested with a
# whole-attention-block ablation at the compute layers (+ saturation L23 and
# control L8) and a concentrated top-head set. AUDIT_FINDINGS.md §11.
# (Needs only the Qwen weights — kept after the Task 2 Qwen purge re-pulls.)
# -------------------------------------------------------------------------
echo ""
echo "######### TASK 3: Qwen necessity ablation (compute band L18-20) #########"
bash scripts/run_qwen_necessity_ablation.sh

echo ""
echo "=== Phase Q FINAL COMPLETE $(date -u +%FT%TZ) ==="
echo "Outputs:"
echo "  results/causal_patching/qwen8b_l{18,19,20,21}_components/SUMMARY_components.md"
echo "  results/causal_patching/tier4_*_{llama,qwen,ministral}_l*_distinctseed/SUMMARY.md"
echo "  results/inference_head_ablation/qwen8b_l*_recon_illegal_fold_wholeattn/SUMMARY.md"
echo "  results/inference_head_ablation/qwen8b_l1{9,9}_recon_illegal_fold_topk/SUMMARY.md"
echo "  logs/opp_*_*_t00_s42_distinctseed_enriched.jsonl"
