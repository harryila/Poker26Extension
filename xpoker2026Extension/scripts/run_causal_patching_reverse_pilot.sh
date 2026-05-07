#!/usr/bin/env bash
# =============================================================================
# Reverse-direction causal-patching pilot — FOLD source -> CHECK target.
# =============================================================================
#
# Mirrors the forward experiment to tighten the causality claim.
#
#   Forward (already done, in results/causal_patching/*_layer_sweep/):
#     source = clean_check_or_call    (a clean, well-formed CHECK/CALL)
#     target = illegal_fold           (a FOLD-committed broken decision)
#     Hypothesis: patching CHECK content into a FOLD-committed target flips
#                 it toward CHECK at saturated layers.
#     Result:    100% top-1 flip + ~+11 to +28 nat shift in CHECK direction
#                in all 3 models.
#
#   Reverse (this script):
#     source = clean_legal_fold       (a clean, well-formed FOLD)
#     target = clean_check_or_call    (a clean, well-formed CHECK/CALL)
#     Hypothesis: patching FOLD content into a CHECK-committed target flips
#                 it toward FOLD at saturated layers.
#     Expected: 100% top-1 flip to FOLD + LARGE NEGATIVE delta_check_minus_fold
#               (~-9 to -28 nat) in Llama and Ministral.
#               Possibly weaker / non-localized in Qwen (consistent with its
#               gradual non-localized forward profile).
#
# Why the reverse matters: the forward result alone could be explained by
# "patching ANY clean residual into a broken target rescues it" (a generic
# coherence-restoration story). The reverse result rules that out by showing
# the patch direction is content-specific, not just clean-vs-broken.
#
# A clean reversal (= 100% flip in the OPPOSITE direction with similar
# saturated magnitude) is the strongest possible single-paragraph evidence
# that the deliberation circuit is (a) genuinely encoding a verb decision
# and (b) doing so in a content-addressable, swappable way.
#
# Compute budget
# --------------
# Sparse layer set in each model's saturated regime + 1 below the boundary
# for a per-model "is the boundary symmetric?" check. Roughly 4 layers x
# 10 sources x 20-30 targets = 800-1200 patched forwards per model.
# Wall-clock ~30-50 min per model on H100, so ~1.5-2.5 h total for all 3.
#
# Outputs
# -------
#   results/causal_patching/ministral8b_t0_pooled_reverse_pilot/
#   results/causal_patching/llama8b_t0_pooled_reverse_pilot/
#   results/causal_patching/qwen8b_t0_pooled_reverse_pilot/
# Each contains SUMMARY.md (per-layer table — read top-1 → FOLD-family
# column for the headline), summary.json, by_pair.csv.
#
# tmux behaviour matches scripts/run_tier1a_small_cot_pilot.sh.
#
# Env knobs
# ---------
#   MODELS=ministral,llama,qwen   (subset / order of models to run)
#   N_SOURCE                      (default 10)
#   N_TARGET                      (default 30)
#   N_RANDOM_CONTROL              (default 5)
#   SEED                          (default 42)
#   DEVICE / DTYPE                (default cuda / bfloat16)
#   NO_TMUX=1                     (run foreground, no tmux session)
# =============================================================================

set -euo pipefail

SESSION_NAME="poker_causal_patching_reverse_pilot"

if [[ -z "${TMUX:-}" ]] && [[ -z "${NO_TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux not installed. Install or set NO_TMUX=1." >&2
        exit 1
    fi
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists."
        echo "  Attach:        tmux attach -t $SESSION_NAME"
        echo "  Kill and rerun: tmux kill-session -t $SESSION_NAME && bash $0"
        exit 1
    fi
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    echo "Creating tmux session '$SESSION_NAME' and attaching."
    echo "  Detach:   Ctrl-B then D"
    echo "  Reattach: tmux attach -t $SESSION_NAME"
    echo "  Kill:     tmux kill-session -t $SESSION_NAME"
    exec tmux new-session -s "$SESSION_NAME" \
        "NO_TMUX=1 bash '$SCRIPT_PATH'; echo; echo '[reverse pilot finished — press any key]'; read -n1 -s"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if   [[ -f "venv/bin/activate"   ]]; then source "venv/bin/activate"
    elif [[ -f "../venv/bin/activate" ]]; then source "../venv/bin/activate"
    else echo "WARNING: no venv found at ./venv or ../venv — using system python"
    fi
fi

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
SEED="${SEED:-42}"
MODELS_ENV="${MODELS:-ministral,llama,qwen}"

# Pooled enriched logs per model (matches what the forward sweeps used).
# Layer choices: 1 below the forward L*, then 3 layers stepping into the
# saturated region of each model. The "below" sample is so we can confirm
# the reverse boundary is at the same depth as the forward boundary.
MINISTRAL_LOGS="logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
                logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
                logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz"
MINISTRAL_LAYERS="${MINISTRAL_LAYERS:-12 16 20 26 30}"   # forward L*=14, saturate by 18

LLAMA_LOGS="logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz"
LLAMA_LAYERS="${LLAMA_LAYERS:-10 14 18 24 30}"            # forward L*=12, saturate by 16

QWEN_LOGS="logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
           logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
           logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz"
QWEN_LAYERS="${QWEN_LAYERS:-16 20 24 30 34}"              # forward "ramp" 16-25, saturate ~23

run_one() {
    local short="$1"            # ministral|llama|qwen
    local enriched_logs_var="$2"
    local layers_var="$3"
    local out_dir="$4"

    local enriched_logs="${!enriched_logs_var}"
    local layers="${!layers_var}"

    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already has SUMMARY.md"
        return 0
    fi
    mkdir -p "$out_dir"

    local first_log
    first_log=$(echo $enriched_logs | awk '{print $1}')
    local n_layers
    n_layers=$(echo $layers | wc -w | tr -d ' ')

    echo
    echo "############################################################"
    echo "## REVERSE pilot — $short"
    echo "##   source: clean_legal_fold   target: clean_check_or_call"
    echo "##   layers ($n_layers): $layers"
    echo "##   logs (pooled, n=$(echo $enriched_logs | wc -w | tr -d ' ')):"
    for p in $enriched_logs; do
        echo "##     $p"
    done
    echo "##   out:    $out_dir"
    echo "############################################################"

    echo
    echo "[pre-flight 1/2] verify position mapping ..."
    python -m experiments.verify_position_mapping \
        --enriched-log "$first_log" --n-samples 10 \
        || { echo "ERROR: position-mapping failed for $short"; return 1; }

    echo
    echo "[pre-flight 2/2] verify prompt reconstruction (GPU) ..."
    python -m experiments.verify_prompt_reconstruction \
        --enriched-log "$first_log" --n-samples 5 \
        --device "$DEVICE" --dtype "$DTYPE" \
        || { echo "ERROR: prompt-reconstruction failed for $short"; return 1; }

    echo
    echo "[main] reverse-direction patching ..."
    python -m experiments.causal_patching \
        --enriched-log $enriched_logs \
        --source-bucket clean_legal_fold \
        --target-bucket clean_check_or_call \
        --layers $layers \
        --n-source "$N_SOURCE" \
        --n-target "$N_TARGET" \
        --n-random-control "$N_RANDOM_CONTROL" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" \
        --dtype "$DTYPE"

    echo
    echo "[done] $short reverse pilot wrote $out_dir/SUMMARY.md"
    echo "      Read the 'top-1 → FOLD-family' column. At saturated layers we"
    echo "      expect %FOLD to climb toward 100% AND specificity-adjusted Δ"
    echo "      to go strongly NEGATIVE (-9 to -28 nat for Llama/Ministral)."
}

IFS=',' read -r -a MODELS_ARR <<< "$MODELS_ENV"
for short in "${MODELS_ARR[@]}"; do
    case "$short" in
        ministral)
            run_one ministral MINISTRAL_LOGS MINISTRAL_LAYERS \
                results/causal_patching/ministral8b_t0_pooled_reverse_pilot
            ;;
        llama)
            run_one llama LLAMA_LOGS LLAMA_LAYERS \
                results/causal_patching/llama8b_t0_pooled_reverse_pilot
            ;;
        qwen)
            run_one qwen QWEN_LOGS QWEN_LAYERS \
                results/causal_patching/qwen8b_t0_pooled_reverse_pilot
            ;;
        *)
            echo "WARNING: unknown model '$short' — skipping"
            ;;
    esac
done

echo
echo "============================================================"
echo "Reverse pilot COMPLETE."
echo
echo "Read all three SUMMARY.md files and look for:"
echo "  1. specificity-adjusted Δ at saturated layers should go strongly"
echo "     NEGATIVE (mirror of the forward result's positive saturation)."
echo "  2. 'top-1 → FOLD-family' should climb to ~100% in Llama and"
echo "     Ministral at L >= forward-L* + ~2."
echo "  3. Qwen — if reversal also shows the same gradual non-localized"
echo "     pattern, the cross-model story is symmetric: localized circuit"
echo "     in Llama/Ministral, distributed in Qwen, in BOTH directions."
echo "============================================================"
