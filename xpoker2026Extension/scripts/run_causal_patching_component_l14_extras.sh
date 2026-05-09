#!/usr/bin/env bash
# =============================================================================
# Component-level patching follow-ups: Ministral B1 + Llama B1.5 head-triplet.
# =============================================================================
#
# Why this exists
# ---------------
# The first component-level run (`run_causal_patching_component_l14.sh`) gave
# us Llama L=14 only. Two follow-ups are queued now:
#
# Cell 1 — Ministral L=14 component sweep (B1 cross-model)
# --------------------------------------------------------
# Identical protocol to the Llama L=14 run, but on Ministral 8B.
# Question: does the same kind of sparse-triplet head profile show up in
# Ministral, or is the per-head story Llama-specific? If Ministral also has a
# sparse triplet — even at different head indices — the head-localization
# story is architecturally meaningful. If Ministral is dense, the head
# story is Llama-specific and we say so.
#
# Cell 2 — Llama L=14 head-triplet patch (B1.5)
# ----------------------------------------------
# Llama's per-head sweep (existing
# results/causal_patching/llama8b_l14_components/SUMMARY_components.md) found
# three heads dominating: head_05 (18%), head_23 (35%), head_24 (20%).
# Linear sum: 73% of residual magnitude.
#
# B1.5 patches all three heads SIMULTANEOUSLY in a single forward and
# compares the joint patch against:
#   - the linear sum of individual head contributions (= 73% if linear)
#   - the full-residual patch (= 100% by definition)
#   - the attn-only patch (= 49% from the existing run)
#
# Three possible outcomes, each scientifically valuable:
#   (a) heads_05_23_24 ≈ 73% (linearity holds): per-head contributions are
#       additive; the triplet is mechanistically the circuit; clean
#       3-head story.
#   (b) heads_05_23_24 well above linear sum (≈ residual or attn): the
#       three heads INTERACT through downstream MLP recomputation; the
#       triplet jointly clears the verb-flip threshold even though no
#       single head does.
#   (c) heads_05_23_24 well below linear sum: the heads interfere
#       destructively (unlikely; would be surprising and worth its own
#       investigation).
#
# Output co-located with cell 1's component sweep so the table reads as a
# single comparison: residual / attn / mlp / heads_05_23_24 / head_05 /
# head_23 / head_24 in one SUMMARY.
#
# Cell 3 — Ministral B1.5 (CONDITIONAL on cell 1 results)
# -------------------------------------------------------
# This script does NOT run cell 3 directly. After cell 1 completes, READ
# `results/causal_patching/ministral8b_l14_components/SUMMARY_components.md`
# and identify Ministral's top-3 heads by ratio-to-residual. Then run:
#
#     bash scripts/run_causal_patching_component_l14_extras.sh \\
#         --ministral-triplet "<h_a> <h_b> <h_c>"
#
# (or set MINISTRAL_TRIPLET="<h_a> <h_b> <h_c>" before invoking).
# An empty MINISTRAL_TRIPLET skips cell 3.
#
# Wall-clock
# ----------
#   Cell 1 (Ministral B1):       ~50-60 min
#   Cell 2 (Llama B1.5):         ~5-10 min  (only 4 modes: residual + attn + mlp + 1 subset)
#   Cell 3 (Ministral B1.5):     ~5-10 min  (same; only if MINISTRAL_TRIPLET set)
# Total: ~60-90 min on H100.
#
# Outputs
# -------
#   Cell 1: results/causal_patching/ministral8b_l14_components/
#           (SUMMARY_components.md + by_pair_components.csv + summary_components.json)
#   Cell 2: results/causal_patching/llama8b_l14_head_triplet/
#           (same three files; the triplet row is heads_05_23_24)
#   Cell 3: results/causal_patching/ministral8b_l14_head_triplet/
#           (only created if MINISTRAL_TRIPLET is set)
#
# Env knobs
# ---------
#   LLAMA_TRIPLET="5 23 24"           override Llama triplet (default from
#                                      cell-1 SUMMARY headline above)
#   MINISTRAL_TRIPLET=""               required for cell 3 to fire
#   LAYER                              default 14
#   N_SOURCE                           default 10
#   N_TARGET                           default 30
#   SEED                               default 42
#   DEVICE / DTYPE                     default cuda / bfloat16
#   SKIP_MINISTRAL_B1=1                skip cell 1 (e.g. if it already ran)
#   SKIP_LLAMA_B1_5=1                  skip cell 2 (e.g. if it already ran)
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
SEED="${SEED:-42}"
LAYER="${LAYER:-14}"
LLAMA_TRIPLET="${LLAMA_TRIPLET:-5 23 24}"
MINISTRAL_TRIPLET="${MINISTRAL_TRIPLET:-}"

# Shared pooled-log paths.
LLAMA_LOGS="logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz"
MINISTRAL_LOGS="logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
                logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
                logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz"

# -------- Cell 1: Ministral B1 (full component sweep at L=14) ----------------

if [[ "${SKIP_MINISTRAL_B1:-0}" != "1" ]]; then
    OUT_DIR="results/causal_patching/ministral8b_l${LAYER}_components"
    if [[ -d "$OUT_DIR" ]] && [[ -f "$OUT_DIR/SUMMARY_components.md" ]]; then
        echo "[skip] cell 1: $OUT_DIR already has SUMMARY_components.md"
    else
        FIRST_LOG=$(echo $MINISTRAL_LOGS | awk '{print $1}')
        if [[ ! -f "$FIRST_LOG" ]]; then
            echo "[skip] cell 1: missing $FIRST_LOG"
        else
            mkdir -p "$OUT_DIR"
            echo
            echo "############################################################"
            echo "## CELL 1 — Ministral B1 (component sweep at L=$LAYER)"
            echo "##   modes: residual, attn, mlp, head (all 32)"
            echo "##   logs (pooled): 3"
            echo "##   out: $OUT_DIR"
            echo "############################################################"

            echo
            echo "[pre-flight 1/2] verify position mapping ..."
            python -m experiments.verify_position_mapping \
                --enriched-log "$FIRST_LOG" --n-samples 10 \
                || { echo "ERROR: position-mapping failed"; exit 1; }

            echo
            echo "[pre-flight 2/2] verify prompt reconstruction (GPU) ..."
            python -m experiments.verify_prompt_reconstruction \
                --enriched-log "$FIRST_LOG" --n-samples 5 \
                --device "$DEVICE" --dtype "$DTYPE" \
                || { echo "ERROR: prompt-reconstruction failed"; exit 1; }

            echo
            echo "[main] component-level patching at L=$LAYER ..."
            python -m experiments.component_patching \
                --enriched-log $MINISTRAL_LOGS \
                --source-bucket clean_check_or_call \
                --target-bucket illegal_fold \
                --layer "$LAYER" \
                --components residual attn mlp head \
                --head-indices all \
                --n-source "$N_SOURCE" \
                --n-target "$N_TARGET" \
                --seed "$SEED" \
                --out-dir "$OUT_DIR" \
                --device "$DEVICE" \
                --dtype "$DTYPE"

            echo
            echo "[done] cell 1 wrote $OUT_DIR/SUMMARY_components.md"
            echo "      Read the per-head ratio column. Identify the top-3"
            echo "      heads (by ratio_to_residual) — those are the candidates"
            echo "      for cell 3 (Ministral B1.5)."
        fi
    fi
else
    echo "[skip] cell 1: SKIP_MINISTRAL_B1=1"
fi

# -------- Cell 2: Llama B1.5 (head triplet patch + comparison) ---------------

if [[ "${SKIP_LLAMA_B1_5:-0}" != "1" ]]; then
    OUT_DIR="results/causal_patching/llama8b_l${LAYER}_head_triplet"
    if [[ -d "$OUT_DIR" ]] && [[ -f "$OUT_DIR/SUMMARY_components.md" ]]; then
        echo "[skip] cell 2: $OUT_DIR already has SUMMARY_components.md"
    else
        FIRST_LOG=$(echo $LLAMA_LOGS | awk '{print $1}')
        if [[ ! -f "$FIRST_LOG" ]]; then
            echo "[skip] cell 2: missing $FIRST_LOG"
        else
            mkdir -p "$OUT_DIR"
            echo
            echo "############################################################"
            echo "## CELL 2 — Llama B1.5 (head-triplet patch at L=$LAYER)"
            echo "##   modes: residual, attn, mlp, head_subset, head"
            echo "##   triplet: $LLAMA_TRIPLET"
            echo "##   logs (pooled): 3"
            echo "##   out: $OUT_DIR"
            echo "############################################################"

            echo
            echo "[pre-flight 1/2] verify position mapping ..."
            python -m experiments.verify_position_mapping \
                --enriched-log "$FIRST_LOG" --n-samples 10 \
                || { echo "ERROR: position-mapping failed"; exit 1; }

            echo
            echo "[pre-flight 2/2] verify prompt reconstruction (GPU) ..."
            python -m experiments.verify_prompt_reconstruction \
                --enriched-log "$FIRST_LOG" --n-samples 5 \
                --device "$DEVICE" --dtype "$DTYPE" \
                || { echo "ERROR: prompt-reconstruction failed"; exit 1; }

            # Modes: residual + attn + mlp + the triplet AS A SUBSET + the
            # three individual heads. Five-row table; comparison is direct.
            echo
            echo "[main] head-triplet patching at L=$LAYER ..."
            python -m experiments.component_patching \
                --enriched-log $LLAMA_LOGS \
                --source-bucket clean_check_or_call \
                --target-bucket illegal_fold \
                --layer "$LAYER" \
                --components residual attn mlp head_subset head \
                --head-indices $LLAMA_TRIPLET \
                --n-source "$N_SOURCE" \
                --n-target "$N_TARGET" \
                --seed "$SEED" \
                --out-dir "$OUT_DIR" \
                --device "$DEVICE" \
                --dtype "$DTYPE"

            echo
            echo "[done] cell 2 wrote $OUT_DIR/SUMMARY_components.md"
            echo "      Read the heads_05_23_24 row vs:"
            echo "        - residual (full): expected ratio 100% by construction"
            echo "        - attn (sublayer): expected ratio ~49% (existing result)"
            echo "        - linear sum of head_05 + head_23 + head_24: ~73%"
            echo "      If heads_05_23_24 ratio ≈ 73%: linearity holds, triplet IS the circuit."
            echo "      If ≈ attn (49%): the triplet captures attention's contribution."
            echo "      If ≈ residual (100%) AND top-1 → CHECK ≈ 79%: the triplet IS the circuit AND clears threshold."
        fi
    fi
else
    echo "[skip] cell 2: SKIP_LLAMA_B1_5=1"
fi

# -------- Cell 3: Ministral B1.5 (only if MINISTRAL_TRIPLET set) -------------

if [[ -n "$MINISTRAL_TRIPLET" ]]; then
    OUT_DIR="results/causal_patching/ministral8b_l${LAYER}_head_triplet"
    if [[ -d "$OUT_DIR" ]] && [[ -f "$OUT_DIR/SUMMARY_components.md" ]]; then
        echo "[skip] cell 3: $OUT_DIR already has SUMMARY_components.md"
    else
        FIRST_LOG=$(echo $MINISTRAL_LOGS | awk '{print $1}')
        if [[ ! -f "$FIRST_LOG" ]]; then
            echo "[skip] cell 3: missing $FIRST_LOG"
        else
            mkdir -p "$OUT_DIR"
            echo
            echo "############################################################"
            echo "## CELL 3 — Ministral B1.5 (head-triplet patch at L=$LAYER)"
            echo "##   modes: residual, attn, mlp, head_subset, head"
            echo "##   triplet: $MINISTRAL_TRIPLET"
            echo "##   logs (pooled): 3"
            echo "##   out: $OUT_DIR"
            echo "############################################################"

            python -m experiments.component_patching \
                --enriched-log $MINISTRAL_LOGS \
                --source-bucket clean_check_or_call \
                --target-bucket illegal_fold \
                --layer "$LAYER" \
                --components residual attn mlp head_subset head \
                --head-indices $MINISTRAL_TRIPLET \
                --n-source "$N_SOURCE" \
                --n-target "$N_TARGET" \
                --seed "$SEED" \
                --out-dir "$OUT_DIR" \
                --device "$DEVICE" \
                --dtype "$DTYPE"

            echo
            echo "[done] cell 3 wrote $OUT_DIR/SUMMARY_components.md"
        fi
    fi
else
    echo
    echo "[note] MINISTRAL_TRIPLET is empty — cell 3 (Ministral B1.5) skipped."
    echo "       After cell 1 completes, identify Ministral's top-3 heads"
    echo "       in ministral8b_l${LAYER}_components/SUMMARY_components.md"
    echo "       and re-run with MINISTRAL_TRIPLET=\"<a> <b> <c>\"."
fi

echo
echo "============================================================"
echo "Component follow-ups COMPLETE."
echo
echo "Read:"
echo "  Cell 1: results/causal_patching/ministral8b_l${LAYER}_components/SUMMARY_components.md"
echo "  Cell 2: results/causal_patching/llama8b_l${LAYER}_head_triplet/SUMMARY_components.md"
if [[ -n "$MINISTRAL_TRIPLET" ]]; then
    echo "  Cell 3: results/causal_patching/ministral8b_l${LAYER}_head_triplet/SUMMARY_components.md"
fi
echo
echo "Side-by-side comparison to write up:"
echo
echo "  Llama  L=14  attn-only       =  49% of residual"
echo "  Llama  L=14  heads_05_23_24  =  ___% of residual    <- new"
echo "  Llama  L=14  linear sum      =  73% of residual"
echo
echo "  Ministral L=14  attn-only        =  ___% of residual    <- new"
echo "  Ministral L=14  top-3 heads     =  see cell 1 SUMMARY"
echo "  Ministral L=14  triplet patch   =  ___% (cell 3)"
echo "============================================================"
