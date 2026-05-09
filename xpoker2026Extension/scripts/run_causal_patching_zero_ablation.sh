#!/usr/bin/env bash
# =============================================================================
# Zero-ablation control at the saturated layer (L*) for each model.
# =============================================================================
#
# Why this exists
# ---------------
# The existing forward causal-patching results say:
#     "Patching a clean CHECK source residual into an illegal_FOLD target at
#      L=14 flips the verb to CHECK in 100% of pairs."
# That result is consistent with TWO interpretations:
#   (a) The layer-L state is causally load-bearing: the model needs SOME
#       coherent signal at L* to commit to a verb, and any sufficiently
#       on-distribution signal works.
#   (b) The circuit at L* is content-addressable: the specific CHECK-content
#       residual matters, and a generic "well-formed" residual would NOT
#       flip the verb.
#
# Zero-ablation distinguishes them. We patch the all-zeros tensor at L* and
# read the result:
#   - If the verb flips toward CHECK (or any other family at large
#     magnitude): the layer is load-bearing in interpretation (a).
#   - If the verb does NOT flip but the clean-source patch did: the circuit
#     is content-addressable in interpretation (b).
#   - If the verb flips to a third destination (e.g. BET or "OTHER"): the
#     layer is load-bearing AND the zero residual breaks coherence in a
#     direction-specific way — also informative.
#
# Either way this experiment sharpens the existing claim. It is also nearly
# free (~30 min total wall-clock) because it bypasses source sampling
# entirely and runs only n_target × n_layer patched forwards per cell.
#
# Cells run
# ---------
#   - Llama 8B, pooled across 3 seeds, layers {12, 13, 14, 15, 18}
#   - Ministral 8B, pooled across 3 seeds, layers {12, 14, 15, 16, 20}
#   - Qwen 8B, pooled across 3 seeds, layers {18, 20, 22, 24, 30}
# Layers chosen as: {L*-2, L*-1, L*, L*+1, L*+saturation_offset} per model,
# matching the boundary identified in the existing pooled forward sweeps.
#
# Per-cell budget
# ---------------
# 5 layers × 30 targets × 1 (zero) source = 150 patched forwards per cell,
# plus controls (~50 forwards) = ~200 forwards/cell × 3 cells = ~600 total.
# Wall-clock: ~5-10 min/cell on H100, ~15-30 min total.
#
# Outputs
# -------
#   results/causal_patching/llama8b_zero_ablation/
#   results/causal_patching/ministral8b_zero_ablation/
#   results/causal_patching/qwen8b_zero_ablation/
#
# Read the SUMMARY.md `mean Δlogit(CHECK − FOLD)` column and compare to the
# corresponding row in the matching layer-sweep SUMMARY.md. The PAPER-READY
# headline is the side-by-side at L=L*:
#
#     | model     | L* | Δ (clean source patch) | Δ (zero patch) | ratio |
#     | Llama     | 14 | +14.3                  | <to-fill>      | <to-fill> |
#     | Ministral | ?? | ??                     | <to-fill>      | <to-fill> |
#     | Qwen      | ?? | ??                     | <to-fill>      | <to-fill> |
#
# A small ratio (< 0.2) is the "content-addressable circuit" reading.
# A large ratio (> 0.7) is the "load-bearing layer" reading.
#
# Env knobs
# ---------
#   MODELS=llama,ministral,qwen   subset / order
#   N_TARGET                       default 30
#   N_RANDOM_CONTROL               default 5
#   SEED                           default 42
#   DEVICE / DTYPE                 default cuda / bfloat16
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_TARGET="${N_TARGET:-30}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
SEED="${SEED:-42}"
MODELS_ENV="${MODELS:-llama,ministral,qwen}"

# Pooled enriched logs per model (matching the forward sweep inputs).
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

    # Skip if no enriched log is actually present.
    local first_log
    first_log=$(echo $enriched_logs | awk '{print $1}')
    if [[ ! -f "$first_log" ]]; then
        echo "[skip] $short: missing $first_log"
        return 0
    fi

    mkdir -p "$out_dir"
    local n_layers
    n_layers=$(echo $layers | wc -w | tr -d ' ')

    echo
    echo "############################################################"
    echo "## ZERO-ABLATION: $short"
    echo "##   layers ($n_layers): $layers"
    echo "##   logs (pooled): $(echo $enriched_logs | wc -w | tr -d ' ')"
    echo "##   out: $out_dir"
    echo "############################################################"

    echo
    echo "[main] zero-ablation patching ..."
    # NOTE: we DO NOT run the prompt-reconstruction pre-flight here because
    # zero-ablation does not depend on the source residual being byte-faithful
    # to the original run — the source IS just zeros. The baseline (control 1)
    # inside the driver still verifies prompt-reconstruction faithfulness for
    # the TARGETS, which is what matters for this experiment.
    python -m experiments.causal_patching \
        --enriched-log $enriched_logs \
        --source-bucket clean_check_or_call \
        --target-bucket illegal_fold \
        --layers $layers \
        --n-target "$N_TARGET" \
        --n-random-control "$N_RANDOM_CONTROL" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --zero-ablation

    echo
    echo "[done] $short wrote $out_dir/SUMMARY.md"
}

IFS=',' read -r -a MODELS_ARR <<< "$MODELS_ENV"
for short in "${MODELS_ARR[@]}"; do
    case "$short" in
        llama)
            run_one llama LLAMA_LOGS LLAMA_LAYERS \
                results/causal_patching/llama8b_zero_ablation
            ;;
        ministral)
            run_one ministral MINISTRAL_LOGS MINISTRAL_LAYERS \
                results/causal_patching/ministral8b_zero_ablation
            ;;
        qwen)
            run_one qwen QWEN_LOGS QWEN_LAYERS \
                results/causal_patching/qwen8b_zero_ablation
            ;;
        *)
            echo "WARNING: unknown model '$short' — skipping"
            ;;
    esac
done

echo
echo "============================================================"
echo "Zero-ablation COMPLETE."
echo
echo "To produce the side-by-side table for the paper, compare the L*"
echo "row in each zero-ablation SUMMARY.md against the corresponding"
echo "row in the matching forward-sweep SUMMARY.md:"
echo "  Llama L=14:"
echo "    zero:  results/causal_patching/llama8b_zero_ablation/SUMMARY.md"
echo "    clean: results/causal_patching/llama8b_t0_pooled_layer_sweep/SUMMARY.md"
echo "  Ministral L=14-16:"
echo "    zero:  results/causal_patching/ministral8b_zero_ablation/SUMMARY.md"
echo "    clean: results/causal_patching/ministral8b_t0_s42_layer_sweep/SUMMARY.md"
echo "  Qwen L=22-24:"
echo "    zero:  results/causal_patching/qwen8b_zero_ablation/SUMMARY.md"
echo "    clean: results/causal_patching/qwen8b_t0_pooled_layer_sweep/SUMMARY.md"
echo
echo "If clean / zero ratio < 0.2 at L*: circuit is content-addressable."
echo "If clean / zero ratio > 0.7 at L*: layer is load-bearing."
echo "Anything in between: nuanced (and worth flagging to interpret)."
echo "============================================================"
