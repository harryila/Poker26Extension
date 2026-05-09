#!/usr/bin/env bash
# =============================================================================
# Verb-generality patching: BET_RAISE source ↔ CHECK_CALL target.
# =============================================================================
#
# Why this exists
# ---------------
# The existing patching results test ONE binary distinction:
#     CHECK source → illegal_FOLD target   (forward)
#     legal_FOLD source → CHECK target     (reverse)
# Both flip the verb at L*=14 in Llama and L*=14-16 in Ministral. That's
# consistent with two interpretations:
#
#   (a) L* is a FOLD-OR-NOT specific circuit. The model has a binary
#       "raise the white flag?" decision encoded at this layer; CHECK and
#       FOLD are two outcomes of that one circuit, and other action verbs
#       (BET, RAISE) live in a different circuit entirely.
#
#   (b) L* is a GENERAL DECISION CIRCUIT. It represents whichever verb
#       the model is about to commit, regardless of which verb. CHECK ↔
#       FOLD is just the most frequent binary in our data.
#
# Test: patch a clean BET_RAISE source residual into a clean CHECK_CALL
# target at L*. If the patched verb flips to BET / RAISE at the same L*,
# the circuit is general (b). If the verb does NOT flip at L=L* but DOES
# flip at some other layer L', the circuit is verb-conditional and we
# discover a SECOND boundary (informative either way). If nothing flips,
# the circuit is fold-or-not specific (a).
#
# This is the verb-generality experiment that strengthens the existing
# binary result into a "decision circuit" claim.
#
# Cells run
# ---------
# 3 models × 1 (pooled-3-seeds) × layers tightly around the CoT-derived L*:
#   Llama:     {12, 13, 14, 15, 18}
#   Ministral: {12, 14, 15, 16, 20}
#   Qwen:      {18, 20, 22, 24, 30}
#
# Why so few layers: this is a targeted question (does the EXISTING L* show
# verb-generality?), not a layer-discovery sweep. We can extend the layer
# range later if the headline reading suggests a different L' for BET_RAISE.
#
# Inputs / outputs
# ----------------
# Inputs:  pooled enriched logs from {s42, s123, s456} per model
# Outputs:
#   results/causal_patching/llama8b_verb_generality_raise_to_check/
#   results/causal_patching/ministral8b_verb_generality_raise_to_check/
#   results/causal_patching/qwen8b_verb_generality_raise_to_check/
#
# Each contains SUMMARY.md (per-layer table — the headline column for THIS
# experiment is `top-1 → BET_RAISE-family`) + summary.json + by_pair.csv.
#
# Wall-clock: ~30-50 min per cell, ~1.5-2.5 h total on H100.
#
# Caveat — bucket sizes
# ---------------------
# `clean_bet_or_raise` may be smaller than `clean_check_or_call` in some
# cells (raises/bets are rarer than checks/calls in this preset). The driver
# auto-caps n_source to whatever's available. If a cell has < 3
# clean_bet_or_raise records, it's skipped with a clear note. To force a
# run anyway, set N_SOURCE=N (1 ≤ N ≤ available).
#
# Env knobs
# ---------
#   MODELS=llama,ministral,qwen   subset / order
#   N_SOURCE                      default 10
#   N_TARGET                      default 30
#   N_RANDOM_CONTROL              default 5
#   SEED                          default 42
#   DEVICE / DTYPE                default cuda / bfloat16
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
SEED="${SEED:-42}"
MODELS_ENV="${MODELS:-llama,ministral,qwen}"

LLAMA_LOGS="logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz"
LLAMA_LAYERS="${LLAMA_LAYERS:-12 13 14 15 18}"

MINISTRAL_LOGS="logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
                logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
                logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz"
MINISTRAL_LAYERS="${MINISTRAL_LAYERS:-12 14 15 16 20}"

QWEN_LOGS="logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
           logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
           logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz"
QWEN_LAYERS="${QWEN_LAYERS:-18 20 22 24 30}"

count_in_bucket() {
    # Count records in a specific bucket across one or more enriched logs.
    local bucket="$1"
    shift
    local logs="$@"
    python - <<PY
import sys
sys.path.insert(0, ".")
from experiments.causal_patching import _iter_decisions, classify_decision
n = 0
for path in """$logs""".split():
    for rec in _iter_decisions(path):
        if rec.get("action_metadata") and rec["action_metadata"].get("raw_response"):
            if classify_decision(rec) == "$bucket":
                n += 1
print(n)
PY
}

run_one() {
    local short="$1"
    local logs_var="$2"
    local layers_var="$3"
    local out_dir="$4"

    local enriched_logs="${!logs_var}"
    local layers="${!layers_var}"

    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already has SUMMARY.md"
        return 0
    fi

    local first_log
    first_log=$(echo $enriched_logs | awk '{print $1}')
    if [[ ! -f "$first_log" ]]; then
        echo "[skip] $short: missing $first_log"
        return 0
    fi

    local n_raise
    n_raise=$(count_in_bucket clean_bet_or_raise $enriched_logs)
    local n_check
    n_check=$(count_in_bucket clean_check_or_call $enriched_logs)
    if [[ "$n_raise" -lt 3 ]]; then
        echo "[skip] $short: only $n_raise clean_bet_or_raise records "
        echo "       in pooled logs (< 3). Verb-generality requires real "
        echo "       BET/RAISE sources. Reduce N_SOURCE or pool more seeds."
        return 0
    fi

    mkdir -p "$out_dir"
    local n_layers
    n_layers=$(echo $layers | wc -w | tr -d ' ')

    echo
    echo "############################################################"
    echo "## VERB-GENERALITY: $short"
    echo "##   source=clean_bet_or_raise (n_avail=$n_raise)"
    echo "##   target=clean_check_or_call (n_avail=$n_check)"
    echo "##   layers ($n_layers): $layers"
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
    echo "[main] verb-generality patching (RAISE → CHECK) ..."
    python -m experiments.causal_patching \
        --enriched-log $enriched_logs \
        --source-bucket clean_bet_or_raise \
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
    echo "[done] $short wrote $out_dir/SUMMARY.md"
    echo "      Read 'top-1 → BET_RAISE-family' as the headline column."
    echo "      A high % at L=L* (matching the CHECK/FOLD experiment's L*)"
    echo "      means L* is a GENERAL DECISION circuit, not just fold-or-not."
}

IFS=',' read -r -a MODELS_ARR <<< "$MODELS_ENV"
for short in "${MODELS_ARR[@]}"; do
    case "$short" in
        llama)
            run_one llama LLAMA_LOGS LLAMA_LAYERS \
                results/causal_patching/llama8b_verb_generality_raise_to_check
            ;;
        ministral)
            run_one ministral MINISTRAL_LOGS MINISTRAL_LAYERS \
                results/causal_patching/ministral8b_verb_generality_raise_to_check
            ;;
        qwen)
            run_one qwen QWEN_LOGS QWEN_LAYERS \
                results/causal_patching/qwen8b_verb_generality_raise_to_check
            ;;
        *)
            echo "WARNING: unknown model '$short' — skipping"
            ;;
    esac
done

echo
echo "============================================================"
echo "Verb-generality COMPLETE."
echo
echo "Read each SUMMARY.md and look for:"
echo
echo "  HEADLINE col: top-1 → BET_RAISE-family"
echo "  - Llama L=14:     should jump to >70% if L* is verb-general"
echo "  - Ministral L=14: should jump to >70% if L* is verb-general"
echo "  - Qwen L=22-24:   should jump to >70% if L* is verb-general"
echo
echo "  SANITY col: mean Δlogit(CHECK − FOLD)"
echo "  - Should be NEAR ZERO (the patch isn't pushing toward CHECK or"
echo "    FOLD; it's pushing toward BET_RAISE). Non-zero means the"
echo "    BET_RAISE source residual co-encodes a CHECK/FOLD signal."
echo
echo "If a model shows a BET_RAISE flip at the SAME L* as the original"
echo "experiment, that's the strong general-decision-circuit story."
echo "If it flips at a DIFFERENT L', that's a layered circuit story."
echo "If it does not flip at all, that's the fold-or-not specific story."
echo "============================================================"
