#!/usr/bin/env bash
# =============================================================================
# Phase Q — remaining four tasks from the missing-work list.
# =============================================================================
#
# Tasks (run serially; one HF model on disk at a time):
#   P0.2  Ministral sextet [9,15,22,24,30,31] inference ablation
#         (run_inference_head_ablation.sh with CONDITIONS including 'extended'
#         and OUT_SUFFIX=_sextet; uses defaults["ministral"]["extended"]).
#   P1.2  Qwen L=22 component sweep (component_patching across residual,
#         attn, mlp, head — pooled over 3 seeds).
#   P1.3  Llama L=15 inference ablation NEGATIVE CONTROL: same triplet
#         [5,23,24] applied at L=15 (where the head story does NOT exist).
#         Per the project notes (REDO_PLAN §P1.3), illegal_FOLD rate should
#         NOT move at L=15 — that's what makes it a negative control.
#   Minor  Tier 4 Llama L=15 opp-preset regen diagnostic:
#         (a) Re-run the 3 cells that failed at L=15 with a permissive
#             BASELINE_TOLERANCE_FRAC=0.20 so the patching effect is captured
#             even though baseline_top1_match is degraded — this lets us see
#             whether the issue is regen pathology specific to opp-preset
#             prompts or a genuine no-effect at L=15.
#         (b) Run verify_prompt_reconstruction at higher n on the failed
#             enriched logs to quantify the regen pathology vs informative_v2.
#
# Order to minimize HF cache churn:
#   1) MINISTRAL: P0.2
#   2) QWEN:      P1.2
#   3) LLAMA:     P1.3 + Minor (re-uses cached Llama weights)
#
# Usage (GPU box):
#   cd xpoker2026Extension
#   bash scripts/run_phase_q_missing_four.sh
#
# Env knobs:
#   N_DECISIONS=80               (Ministral sextet & Llama negctrl pool size)
#   CONTINUE_TOKENS_*            (unused here)
#   FORCE_RERUN=1                (default)
#   NO_TMUX=1                    skip tmux wrapper
# =============================================================================
set -uo pipefail

SESSION_NAME="poker_phase_q_missing_four"

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
    LOG="$REPO_ROOT/logs/phase_q_missing_four_$(date -u +%Y%m%d_%H%M%SZ).log"
    mkdir -p "$REPO_ROOT/logs"
    echo "Creating detached tmux session '$SESSION_NAME'"
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Log:    $LOG"
    tmux new-session -d -s "$SESSION_NAME" \
        "cd '$REPO_ROOT' && NO_TMUX=1 bash '$SCRIPT_PATH' 2>&1 | tee '$LOG'; echo; echo '[missing-four finished]'; sleep 999999"
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

N_DECISIONS="${N_DECISIONS:-80}"

purge_hf_for_model() {
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

echo "=== Phase Q missing-four run $(date -u +%FT%TZ) ==="
echo "HF_HOME=$HF_HOME  N_DECISIONS=$N_DECISIONS  FORCE_RERUN=$FORCE_RERUN"

# -------------------------------------------------------------------------
# 1) P0.2 — MINISTRAL sextet ablation
# -------------------------------------------------------------------------
echo ""
echo "######### TASK 1/4 (P0.2): Ministral sextet [9,15,22,24,30,31] #########"
MODEL=ministral \
  PIPELINE=recon \
  FILTER_RECORDED_BUCKET=illegal_fold \
  CONDITIONS="baseline triplet extended control" \
  OUT_SUFFIX="_sextet" \
  N_DECISIONS="$N_DECISIONS" \
  bash scripts/run_inference_head_ablation.sh
purge_hf_for_model "models--mistralai--Ministral-8B-Instruct-2410"

# -------------------------------------------------------------------------
# 2) P1.2 — QWEN L=22 component sweep
# -------------------------------------------------------------------------
echo ""
echo "######### TASK 2/4 (P1.2): Qwen L=22 component sweep #########"
QWEN_OUT="results/causal_patching/qwen8b_l22_components"
if [[ -f "$QWEN_OUT/SUMMARY_components.md" ]] && [[ "$FORCE_RERUN" != "1" ]]; then
    echo "[skip] $QWEN_OUT"
else
    mkdir -p "$QWEN_OUT"
    python -m experiments.component_patching \
        --enriched-log \
            logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz \
        --source-bucket clean_check_or_call \
        --target-bucket illegal_fold \
        --layer 22 \
        --components residual attn mlp head \
        --head-indices all \
        --n-source 10 \
        --n-target 30 \
        --seed 42 \
        --out-dir "$QWEN_OUT" \
        --device "$DEVICE" --dtype "$DTYPE"
    echo "[done] $QWEN_OUT/SUMMARY_components.md"
fi
purge_hf_for_model "models--Qwen--Qwen3-8B"

# -------------------------------------------------------------------------
# 3) P1.3 — LLAMA L=15 negative-control ablation
# -------------------------------------------------------------------------
echo ""
echo "######### TASK 3/4 (P1.3): Llama L=15 inference ablation (neg ctrl) #########"
MODEL=llama \
  LAYER=15 \
  PIPELINE=recon \
  FILTER_RECORDED_BUCKET=illegal_fold \
  CONDITIONS="baseline triplet control" \
  OUT_SUFFIX="_negctrl" \
  N_DECISIONS="$N_DECISIONS" \
  bash scripts/run_inference_head_ablation.sh

# -------------------------------------------------------------------------
# 4) Minor — Tier 4 Llama L=15 opp-preset regen diagnostic
#    (Llama weights are still cached from task 3.)
# -------------------------------------------------------------------------
echo ""
echo "######### TASK 4/4 (Minor): Tier 4 Llama L=15 regen pathology + rerun #########"

# 4a) Diagnostic: prompt-reconstruction quality on each opp-preset enriched log
# Compare to informative_v2 (which succeeded with baseline=0.667).
echo ""
echo "------ 4a) prompt-reconstruction diagnostic across opp-presets ------"
mkdir -p results/diagnostics/tier4_llama_l15_regen
for preset in default informative_v2 tight_aggressive loose_aggressive loose_passive; do
    enriched="logs/opp_${preset}_llama-8b_t00_s42_enriched.jsonl"
    if [[ ! -f "$enriched" ]]; then
        echo "[skip] $preset: missing $enriched"
        continue
    fi
    diag_log="results/diagnostics/tier4_llama_l15_regen/${preset}_recon.txt"
    echo "  -> $preset"
    python -m experiments.verify_prompt_reconstruction \
        --enriched-log "$enriched" \
        --n-samples 30 \
        --tie-tolerance-nats 0.50 \
        --max-failures 30 \
        --device "$DEVICE" --dtype "$DTYPE" \
        > "$diag_log" 2>&1 || echo "    [exit=$?]"
done
echo "[done] diagnostics in results/diagnostics/tier4_llama_l15_regen/"

# 4b) Re-run the 3 failed Tier 4 cells with permissive baseline tolerance so
# the patching effect is captured even though baseline top-1 match is below
# 0.5 (likely 0.2–0.3 based on prior log). The patching driver's no-patch
# baseline gate is the only thing that aborted these; with the relaxed gate
# the patching numbers themselves are reported and we can decide whether to
# trust them based on the diagnostic above.
echo ""
echo "------ 4b) Tier 4 Llama L=15 rerun (BASELINE_TOLERANCE_FRAC=0.20) ------"
PRESETS="informative_v2 tight_aggressive loose_aggressive" \
  MODELS=llama-8b \
  LLAMA_LAYER=15 \
  BASELINE_TOLERANCE_FRAC=0.20 \
  FORCE_RERUN=1 \
  bash scripts/run_tier4_patching.sh \
  || echo "[warn] some tier4 cells may still fail; see log"

purge_hf_for_model "models--meta-llama--Llama-3.1-8B-Instruct"

echo ""
echo "=== Phase Q missing-four COMPLETE $(date -u +%FT%TZ) ==="
echo "Outputs:"
echo "  P0.2  results/inference_head_ablation/ministral8b_l16_recon_illegal_fold_sextet/"
echo "  P1.2  results/causal_patching/qwen8b_l22_components/"
echo "  P1.3  results/inference_head_ablation/llama8b_l15_recon_illegal_fold_negctrl/"
echo "  Minor results/diagnostics/tier4_llama_l15_regen/"
echo "        results/causal_patching/tier4_{informative_v2,tight_aggressive,loose_aggressive}_llama_l15/"
