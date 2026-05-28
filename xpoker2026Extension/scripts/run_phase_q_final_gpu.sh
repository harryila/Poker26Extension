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
# A follow-up that DEPENDS on Task 1's result (Qwen necessity ablation at the
# identified compute layer) is described at the end — it needs the sweep to
# name the layer first, so it is intentionally not pre-scripted here.
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

echo "=== Phase Q FINAL GPU run $(date -u +%FT%TZ) ==="

# -------------------------------------------------------------------------
# TASK 1 — Qwen compute-layer component sweep (purges Qwen weights at end).
# -------------------------------------------------------------------------
echo ""
echo "######### TASK 1/2: Qwen compute-layer sweep (L=${QWEN_LAYERS:-18 19 20 21}) #########"
QWEN_LAYERS="${QWEN_LAYERS:-18 19 20 21}" PURGE=1 \
    bash scripts/run_qwen_compute_layer_sweep.sh

# -------------------------------------------------------------------------
# TASK 2 — Tier 4 distinct-seed regeneration (all 3 models, purges each).
# -------------------------------------------------------------------------
echo ""
echo "######### TASK 2/2: Tier 4 distinct-seed regeneration (all 3 models) #########"
MODELS="${TIER4_MODELS:-llama-8b qwen-8b ministral-8b}" \
    bash scripts/run_tier4_regen_distinct_presets.sh

# -------------------------------------------------------------------------
# Refresh the cross-preset overlap diagnostic on the NEW distinct-seed logs.
# (CPU-only; safe to run here.)
# -------------------------------------------------------------------------
echo ""
echo "[verify] distinct-seed preset overlap (should now show 5 distinct each) ..."
python -m experiments.diagnose_tier4_preset_overlap \
    --models llama-8b qwen-8b ministral-8b \
    --out results/diagnostics/tier4_preset_overlap_distinctseed/SUMMARY.md \
    || echo "[warn] diagnostic expects *_distinctseed_enriched.jsonl naming; adjust if needed"

echo ""
echo "=== Phase Q FINAL COMPLETE $(date -u +%FT%TZ) ==="
echo "Outputs:"
echo "  results/causal_patching/qwen8b_l{18,19,20,21}_components/SUMMARY_components.md"
echo "  results/causal_patching/tier4_*_{llama,qwen,ministral}_l*_distinctseed/SUMMARY.md"
echo "  logs/opp_*_*_t00_s42_distinctseed_enriched.jsonl"
echo ""
echo "FOLLOW-UP (needs Task 1 result): once the sweep names Qwen's compute"
echo "layer L_c (earliest layer with attn ratio >= ~80%% of residual, or a"
echo "head > 10%%), run a necessity ablation of that layer's attention during"
echo "generation to complete the Qwen necessity story. This needs a small"
echo "addition to inference_head_ablation.py to accept an arbitrary head set"
echo "(e.g. all heads at L_c) — flagged in AUDIT_FINDINGS.md."
