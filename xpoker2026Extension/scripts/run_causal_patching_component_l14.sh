#!/usr/bin/env bash
# =============================================================================
# Component-level causal patching at L*=14 (Llama first; Ministral optional).
# =============================================================================
#
# Why this exists
# ---------------
# The forward causal-patching result says "patching the residual stream at
# L=14 in Llama 8B flips the verb to CHECK in 100% of pairs with a +14 nat
# delta." That is the layer-as-a-whole answer.
#
# The next-resolution question is which SUBLAYER and which HEADS at L=14
# carry the signal. Standard mech-interp drill-down:
#
#     residual = attn_contribution + mlp_contribution + everything_else
#
# We re-run the patching experiment under three additional modes:
#   - attn-only: replace the attention sublayer's output at the last
#     position; everything else (MLP, future layers' residual additions)
#     proceeds normally.
#   - mlp-only: replace the MLP sublayer's output at the last position.
#   - per-head: for each of the 32 attention heads, replace ONLY that
#     head's pre-o_proj contribution; other heads pass through.
#
# Compares each mode's RATIO-TO-RESIDUAL (mode Δ / residual-mode Δ on the
# same (source, target) pairs). NOTE: this is NOT a specificity-adjusted Δ
# in the sense of `causal_patching.py` — the component driver does not
# compute a random-source null per component; the comparison denominator is
# the residual-mode Δ. The strongest possible version of the result:
#   "attention contribution alone explains ~95% of the residual-patch
#    effect at L=14, and three heads (h7, h12, h19) each carry >25% of
#    the attention effect."
#
# Anything weaker is still informative:
#   - attn ≈ residual and heads dense → "attention-mediated, distributed"
#   - mlp ≈ residual                  → "MLP-mediated" (surprising; flag)
#   - both small, sum ≈ residual      → "interaction-mediated" (still
#     publishable; suggests a non-linear coupling)
#
# Single-load efficiency
# ----------------------
# We use experiments/component_patching.py which captures EVERY component
# state in a single source forward pass and patches all modes against the
# same source-target pairs. This is ~30x faster than running
# `causal_patching.py --component {residual,attn,mlp,head_<i>}` separately
# for each of {residual, attn, mlp} + 32 heads.
#
# Per-cell budget at L=14, all modes:
#   captures:    n_source forwards = 10
#   baselines:   n_target forwards = 30
#   self-patch:  1 forward
#   patched:     n_target × n_source × n_modes = 30 × 10 × 35 = 10500 forwards
# Total ≈ 10550 forwards × ~0.3 sec = ~50 min wall-clock per cell.
#
# Cells run
# ---------
#   Llama 8B, L=14, pooled across 3 seeds, all 32 heads + attn + mlp + residual
#   (Ministral 8B, L=14, same protocol — gated on RUN_MINISTRAL=1)
#   (Qwen 8B is intentionally NOT here: Qwen's distributed L=19-23 ramp is
#    qualitatively different and a single-layer component sweep there
#    answers a less crisp question. If Qwen B1 ever runs, do it as a
#    separate experiment over multiple layers.)
#
# Outputs
# -------
#   results/causal_patching/llama8b_l14_components/
#     SUMMARY_components.md      <- read this for the headline
#     summary_components.json
#     by_pair_components.csv
#   results/causal_patching/ministral8b_l14_components/    (if RUN_MINISTRAL=1)
#
# Env knobs
# ---------
#   RUN_MINISTRAL=1   also run Ministral 8B at L=14 (~50 min extra)
#   LAYER             override layer (default 14 — set 12 or 16 to test
#                     boundary edges)
#   N_SOURCE          default 10
#   N_TARGET          default 30
#   SEED              default 42
#   DEVICE / DTYPE    default cuda / bfloat16
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
SEED="${SEED:-42}"
LAYER="${LAYER:-14}"
RUN_MINISTRAL="${RUN_MINISTRAL:-0}"

LLAMA_LOGS="logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz"
MINISTRAL_LOGS="logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
                logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
                logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz"

run_one() {
    local short="$1"
    local logs_var="$2"
    local out_dir="$3"

    local enriched_logs="${!logs_var}"
    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY_components.md" ]]; then
        echo "[skip] $out_dir already has SUMMARY_components.md"
        return 0
    fi

    local first_log
    first_log=$(echo $enriched_logs | awk '{print $1}')
    if [[ ! -f "$first_log" ]]; then
        echo "[skip] $short: missing $first_log"
        return 0
    fi

    mkdir -p "$out_dir"

    echo
    echo "############################################################"
    echo "## COMPONENT-LEVEL: $short at L=$LAYER"
    echo "##   modes: residual, attn, mlp, head (all)"
    echo "##   logs (pooled): $(echo $enriched_logs | wc -w | tr -d ' ')"
    echo "##   out: $out_dir"
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
    echo "[main] component-level patching at L=$LAYER ..."
    python -m experiments.component_patching \
        --enriched-log $enriched_logs \
        --source-bucket clean_check_or_call \
        --target-bucket illegal_fold \
        --layer "$LAYER" \
        --components residual attn mlp head \
        --head-indices all \
        --n-source "$N_SOURCE" \
        --n-target "$N_TARGET" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" \
        --dtype "$DTYPE"

    echo
    echo "[done] $short wrote $out_dir/SUMMARY_components.md"
}

run_one llama LLAMA_LOGS \
    "results/causal_patching/llama8b_l${LAYER}_components"

if [[ "$RUN_MINISTRAL" == "1" ]]; then
    run_one ministral MINISTRAL_LOGS \
        "results/causal_patching/ministral8b_l${LAYER}_components"
fi

echo
echo "============================================================"
echo "Component-level patching COMPLETE."
echo
echo "Read the per-mode table in:"
echo "  results/causal_patching/llama8b_l${LAYER}_components/SUMMARY_components.md"
if [[ "$RUN_MINISTRAL" == "1" ]]; then
    echo "  results/causal_patching/ministral8b_l${LAYER}_components/SUMMARY_components.md"
fi
echo
echo "Look for:"
echo "  1. residual row: should ≈ existing pooled-sweep number at L=$LAYER"
echo "     (sanity check that the new driver reproduces the old result)"
echo "  2. attn row:    if Δ ≥ 80% of residual, attention dominates"
echo "     mlp row:     if Δ <  20% of residual, MLP is irrelevant"
echo "  3. head_NN rows: any heads with Δ >25% of residual mean: SPARSE"
echo "     HEAD STORY. Cite those heads in the writeup."
echo "============================================================"
