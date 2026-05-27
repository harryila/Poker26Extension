#!/usr/bin/env bash
# =============================================================================
# Phase Q — all seven follow-up experiments (GPU-heavy).
# =============================================================================
#
# Order (fast checks first, then heavy):
#   1. Seed-matched verify + mode-balanced (CPU/light GPU)
#   2. Reverse full + BET→illegal_fold patching (~2h)
#   3. Tier 4 patching refresh (Qwen/Ministral; Llama optional)
#   4. Context-stratified patching (~1h/model)
#   5. Continuation after patch (~1h/model)
#   6. Inference head ablation (~2-4h, loads model once per condition)
#
# Env knobs:
#   MODELS="ministral llama qwen"   subset for per-model scripts
#   SKIP_TIER4=1                    skip tier 4
#   RUN_LLAMA_TIER4=0               default skip Llama tier4 (template drift)
#   FORCE_RERUN=1                   re-run even if SUMMARY exists
#
# tmux (recommended on GPU box):
#   bash scripts/run_phase_q_all.sh
#   Ctrl-B then D to detach;  tmux attach -t poker_phase_q
#   NO_TMUX=1 to run in foreground/nohup without tmux.
# =============================================================================
set -uo pipefail

SESSION_NAME="poker_phase_q"

if [[ -z "${TMUX:-}" ]] && [[ -z "${NO_TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux not installed (apt install tmux), or set NO_TMUX=1"
        exit 1
    fi
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists."
        echo "  Attach:  tmux attach -t $SESSION_NAME"
        echo "  Kill:    tmux kill-session -t $SESSION_NAME"
        exit 1
    fi
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    echo "Creating tmux session '$SESSION_NAME' (detach: Ctrl-B D)"
    exec tmux new-session -s "$SESSION_NAME" \
        "NO_TMUX=1 bash '$SCRIPT_PATH'; echo; echo '[phase_q finished — press any key]'; read -n1 -s"
fi

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
export DEVICE
export DTYPE="${DTYPE:-bfloat16}"
export FORCE_RERUN="${FORCE_RERUN:-0}"

log_section() {
  echo ""
  echo "============================================================"
  echo "## $1"
  echo "============================================================"
}

log_section "1/7 — Seed-matched verify + mode-balanced probe"
bash scripts/run_seed_matched_nocot_verify.sh || true

log_section "2/7 — Reverse-direction patching (FOLD → CHECK)"
bash scripts/run_causal_patching_reverse_full.sh

log_section "3/7 — BET_RAISE → illegal_fold patching"
bash scripts/run_causal_patching_bet_to_illegal_fold.sh

if [[ "${SKIP_TIER4:-0}" != "1" ]]; then
  log_section "4/7 — Tier 4 opponent-preset patching"
  if [[ "${RUN_LLAMA_TIER4:-0}" == "1" ]]; then
    MODELS="llama-8b qwen-8b ministral-8b" bash scripts/run_tier4_patching.sh
  else
    echo "[info] Skipping Llama Tier4 (RUN_LLAMA_TIER4=0); Qwen+Ministral only"
    MODELS="qwen-8b ministral-8b" bash scripts/run_tier4_patching.sh
    # Attempt missing informative_v2 llama cell only if explicitly requested
  fi
else
  echo "[skip] Tier 4 (SKIP_TIER4=1)"
fi

log_section "5/7 — Context-stratified patching"
for m in ${MODELS:-ministral llama qwen}; do
  MODEL="$m" bash scripts/run_context_stratified_patching.sh || true
done

log_section "6/7 — Continuation after patch"
for m in ${MODELS:-ministral llama qwen}; do
  MODEL="$m" bash scripts/run_continuation_after_patch.sh || true
done

log_section "7/7 — Inference-time head ablation (behavioral)"
bash scripts/run_inference_head_ablation.sh

echo ""
echo "============================================================"
echo "Phase Q COMPLETE. Key outputs:"
echo "  results/inference_head_ablation/ministral8b_l16_cot/"
echo "  results/continuation_after_patch/*/"
echo "  results/context_stratified_patching/*/"
echo "  results/causal_patching/*_reverse_fold_to_check_l*/"
echo "  results/causal_patching/*_bet_to_illegal_fold_l*/"
echo "  results/mode_balanced_probe/*/"
echo "  results/causal_patching/tier4_*/"
echo "============================================================"
