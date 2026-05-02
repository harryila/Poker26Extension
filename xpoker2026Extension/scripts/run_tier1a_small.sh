#!/usr/bin/env bash
# =============================================================================
# Tier 1A.small — Scaled non-CoT baseline at 8B class
# =============================================================================
#
# What this script does (in order, per model):
#   1. Run all 6 cells: 3 seeds (42, 123, 456) x 2 temps (0.0, 0.2)
#   2. Enrich each cell's log with oracle posteriors (StrategyAware + CardOnly)
#   3. Run PCE distribution + update coherence on that model's pooled logs
#   4. Move on to the next model
#
# Models (8B class, parameter-matched):
#   - llama-8b      (meta-llama/Llama-3.1-8B-Instruct)
#   - qwen-8b       (Qwen/Qwen3-8B)            <-- thinking mode AUTO-DISABLED
#                                                  for non-CoT runs (see note)
#   - ministral-8b  (mistralai/Ministral-8B-Instruct-2410)
#
# Hand budget: 100 hands/cell x 6 cells x 3 models = 1,800 hands total.
#   Wall-clock: ~3-6 hours on one H100/A100 (8B inference is fast).
#
# Outputs (written incrementally as each model completes):
#   logs/scaled_<tag>_<temp>_s<seed>_informative_v2.jsonl            (raw)
#   logs/scaled_<tag>_<temp>_s<seed>_informative_v2_enriched.jsonl   (enriched)
#   results/tier1a_small/pce_<tag>_records.csv                       (per-decision)
#   results/tier1a_small/pce_<tag>_summary.csv                       (clustered bootstrap)
#   results/tier1a_small/uc_<tag>.csv + uc_<tag>_summary.json
#   results/tier1a_small/pce_pool_summary.csv                        (cross-model pool)
#
# Failure isolation: if model N crashes, models 1..N-1 results are already on
# disk. Re-running this script will skip any cell whose log already exists.
#
# =============================================================================
# tmux behaviour
# =============================================================================
#
# Recommended workflow:
#   ssh user@gpu-box
#   cd <repo>/xpoker2026Extension
#   bash scripts/run_tier1a_small.sh
#
# When run OUTSIDE tmux, the script creates a tmux session named
# 'poker_tier1a_small' AND attaches you to it, so you see the run starting.
# The script does NOT auto-detach you — that is your call:
#   - Press Ctrl-B then D to detach (run continues, you exit tmux back to
#     your shell, and you can safely close the SSH connection).
#   - Reattach later with:  tmux attach -t poker_tier1a_small
#   - Kill the run with:    tmux kill-session -t poker_tier1a_small
#
# To force foreground execution without tmux (e.g. for CI / debugging), set
# NO_TMUX=1 before invoking.
#
# =============================================================================
# Qwen 3 thinking-mode note
# =============================================================================
#
# Qwen3-8B's chat template enables "thinking mode" by default, which would
# silently inject internal CoT reasoning and invalidate this baseline.
# poker_env/agents/hf_agent.py passes enable_thinking=self.cot_mode to the
# tokenizer ONLY for models flagged has_thinking_mode=True in the registry.
# Since this script does NOT pass --cot, qwen-8b runs with thinking OFF.
# The agent config logged in every JSONL will show enable_thinking: false.
# A pre-flight check below verifies this and aborts if it's not the case.
# =============================================================================

set -euo pipefail

SESSION_NAME="poker_tier1a_small"

# ---- Launch inside tmux (attached) if not already there ----
if [[ -z "${TMUX:-}" ]] && [[ -z "${NO_TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux is not installed. Install it (apt: sudo apt install tmux,"
        echo "       brew: brew install tmux), or rerun with NO_TMUX=1 to run in"
        echo "       the foreground (the run will die if your SSH disconnects)."
        exit 1
    fi
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists."
        echo "  Attach:        tmux attach -t $SESSION_NAME"
        echo "  Kill and rerun: tmux kill-session -t $SESSION_NAME && bash $0"
        exit 1
    fi
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    echo "Creating tmux session '$SESSION_NAME' and attaching you to it."
    echo "  When you want to leave the run going in the background:"
    echo "    Press Ctrl-B then D    (detaches you; run continues in tmux)"
    echo "  Reattach later with:   tmux attach -t $SESSION_NAME"
    echo "  Abort the run with:    tmux kill-session -t $SESSION_NAME"
    echo
    # NO -d flag: tmux creates the session AND attaches us to it. The user
    # decides when (or whether) to detach.
    # NO_TMUX=1 inside the session short-circuits the relaunch guard above.
    exec tmux new-session -s "$SESSION_NAME" \
        "NO_TMUX=1 bash '$SCRIPT_PATH'; echo; echo '[tier1a_small finished — press any key to close window]'; read -n1 -s"
fi

# ---- We're now inside tmux (or NO_TMUX=1 was set). Do the work. ----

# Resolve repo root from script location, regardless of where it was invoked.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXT_DIR"

# Activate venv (typical layout: <ext>/venv).
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source venv/bin/activate
    elif [[ -f "../venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source ../venv/bin/activate
    else
        echo "WARNING: no venv found at ./venv or ../venv — using system python"
    fi
fi

LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results/tier1a_small}"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

SEEDS=(42 123 456)
TEMPS=(0.0 0.2)
HANDS_PER_CELL="${HANDS_PER_CELL:-100}"
OPPONENT_PRESET="informative_v2"

# Map "0.0" -> "t0", "0.2" -> "t02".
temp_suffix() {
    case "$1" in
        0.0|0) echo "t0" ;;
        0.2)   echo "t02" ;;
        *)     echo "t${1//./}" ;;
    esac
}

# ---- Pre-flight: verify Qwen 3 thinking mode will be OFF ----
echo "=== Pre-flight: confirming qwen-8b non-CoT thinking mode is OFF ==="
python <<'PY'
from poker_env.config import MODEL_REGISTRY
cfg = MODEL_REGISTRY.get("qwen-8b")
assert cfg is not None, "qwen-8b not registered"
assert cfg.get("has_thinking_mode") is True, (
    "qwen-8b is missing has_thinking_mode=True in the registry. "
    "Without this, HFAgent will not pass enable_thinking=False to the chat "
    "template and Qwen 3 will silently perform internal CoT, invalidating "
    "the non-CoT baseline. Aborting."
)
print(f"  qwen-8b registered: {cfg['model_id']}")
print(f"  has_thinking_mode=True  -> HFAgent will pass enable_thinking=cot_mode")
print(f"  This script does NOT pass --cot, so cot_mode=False")
print(f"  -> enable_thinking=False (verified in agent config of every log)")
PY
echo

# ---- Per-model run/enrich/analyze loop ----
# Format: "<short_name>:<filename_tag>"
MODELS=(
    "llama-8b:llama8b"
    "qwen-8b:qwen8b"
    "ministral-8b:ministral8b"
)

run_model() {
    local short="$1"
    local tag="$2"
    local started_at
    started_at="$(date +%s)"

    echo
    echo "############################################################"
    echo "## MODEL: $short  (tag=$tag)"
    echo "## Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "############################################################"

    # ---- Phase 1: run all 6 cells for this model ----
    for seed in "${SEEDS[@]}"; do
        for temp in "${TEMPS[@]}"; do
            local tsuf
            tsuf="$(temp_suffix "$temp")"
            local out="${LOGS_DIR}/scaled_${tag}_${tsuf}_s${seed}_${OPPONENT_PRESET}.jsonl"
            if [[ -f "$out" ]]; then
                echo "  [skip] $out already exists"
                continue
            fi
            echo "  [run]  $out  (seed=$seed temp=$temp hands=$HANDS_PER_CELL)"
            python run_experiment.py \
                --agent hf \
                --hf-model "$short" \
                --opponent threshold \
                --opponent-preset "$OPPONENT_PRESET" \
                --hands "$HANDS_PER_CELL" \
                --seed "$seed" \
                --temperature "$temp" \
                --elicit-beliefs \
                --capture-logprobs \
                --out "$out" \
                -v
        done
    done

    # ---- Phase 2: enrich logs with oracle posteriors ----
    echo
    echo "  [enrich] $tag"
    shopt -s nullglob
    for f in "${LOGS_DIR}"/scaled_${tag}_*_${OPPONENT_PRESET}.jsonl; do
        [[ "$f" == *_enriched.jsonl ]] && continue
        local out_enr="${f%.jsonl}_enriched.jsonl"
        if [[ -f "$out_enr" ]]; then
            echo "    [skip] $out_enr already exists"
            continue
        fi
        echo "    [enrich] $f -> $out_enr"
        python -m analysis.build_dataset \
            --input "$f" \
            --output "$out_enr" \
            --opponent-preset "$OPPONENT_PRESET"
    done

    # ---- Phase 3: per-model analysis ----
    echo
    echo "  [analyze] $tag"
    local files=( "${LOGS_DIR}"/scaled_${tag}_*_enriched.jsonl )
    if [[ "${#files[@]}" -eq 0 ]]; then
        echo "    [skip] no enriched logs for $tag"
        return
    fi
    python -m analysis.compute_pce_distribution \
        "${files[@]}" \
        --output-records "${RESULTS_DIR}/pce_${tag}_records.csv" \
        --output-summary "${RESULTS_DIR}/pce_${tag}_summary.csv" \
        --bootstrap 1000 --seed 42 --clustered

    python -m analysis.compute_update_coherence \
        "${files[@]}" \
        --output "${RESULTS_DIR}/uc_${tag}.csv" \
        --output-summary "${RESULTS_DIR}/uc_${tag}_summary.json"

    local elapsed=$(( $(date +%s) - started_at ))
    echo
    echo "## DONE $short in ${elapsed}s"
    echo "## Results: ${RESULTS_DIR}/pce_${tag}_*.csv, uc_${tag}.*"
    echo
}

# ---- Run each model in turn ----
for entry in "${MODELS[@]}"; do
    short="${entry%%:*}"
    tag="${entry#*:}"
    run_model "$short" "$tag"
done

# ---- Final cross-model pool ----
echo
echo "############################################################"
echo "## POOLED ANALYSIS — all 8B models"
echo "############################################################"
shopt -s nullglob
all_files=(
    "${LOGS_DIR}"/scaled_llama8b_*_enriched.jsonl
    "${LOGS_DIR}"/scaled_qwen8b_*_enriched.jsonl
    "${LOGS_DIR}"/scaled_ministral8b_*_enriched.jsonl
)
if [[ "${#all_files[@]}" -gt 0 ]]; then
    python -m analysis.compute_pce_distribution \
        "${all_files[@]}" \
        --output-records "${RESULTS_DIR}/pce_pool_records.csv" \
        --output-summary "${RESULTS_DIR}/pce_pool_summary.csv" \
        --bootstrap 1000 --seed 42 --clustered
    echo
    echo "Pooled summary: ${RESULTS_DIR}/pce_pool_summary.csv"
fi

echo
echo "============================================================"
echo "Tier 1A.small COMPLETE."
echo "  Logs:    $LOGS_DIR/scaled_{llama8b,qwen8b,ministral8b}_*"
echo "  Results: $RESULTS_DIR/"
echo "  Compare each model's mean_js_to_sa to the paper anchor 0.014."
echo "============================================================"
