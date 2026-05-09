#!/usr/bin/env bash
# =============================================================================
# Non-CoT causal-patching parity check across the 8B family.
# =============================================================================
#
# Why this exists
# ---------------
# Every existing causal-patching result in this repo runs against CoT
# enriched logs (`logs/cot_<model>8b_*_logitlens_enriched.jsonl.gz`). All of
# them show a saturated-layer flip — Llama L*=14, Ministral L*=14-16, Qwen
# distributed L=19-23. That leaves a paper-relevant question open:
#
#     Is L* the model's NATURAL decision circuit, or is it specifically the
#     CoT-induced deliberation circuit?
#
# Two outcomes, each scientifically valuable:
#   (A) L*=14 boundary persists in non-CoT runs.
#       => the circuit is the model's, not the prompt's. STRONGER VERSION
#       of the existing paper claim. We can then say "the same residual-
#       stream layer mediates the action decision regardless of whether
#       the model is asked to deliberate explicitly."
#
#   (B) L* shifts or vanishes without CoT.
#       => CoT is reshaping internals, not just outputs. This is its own
#       paper-worthy story about what CoT *actually does* mechanistically.
#       We would write a separate paragraph: "the deliberation circuit at
#       L=14 is induced (or reorganized) by the CoT prompt — without it,
#       the model's verb-decision is mediated at L=N != 14".
#
# Either is a publishable result. There is no "failure mode" — the controls
# inside the patching driver still gate prompt-reconstruction faithfulness,
# so the experiment cannot silently produce nonsense.
#
# Inputs
# ------
# Non-CoT enriched logs from the Tier-1A.small baseline runs:
#   logs/scaled_<model>8b_t0_s<seed>_informative_v2_enriched.jsonl
# (Note the absence of `_logitlens_` and `.gz` — these are the regular
#  enriched logs from `analysis.build_dataset`, NOT the logit-lens sidecar
#  format. The patching driver does not require the logit-lens sidecar.)
#
# Caveat — illegal_FOLD availability
# ----------------------------------
# Non-CoT runs may have far fewer `illegal_fold` decisions than CoT runs,
# because the model commits the verb in one step rather than after several
# generated lines of reasoning. The script auto-detects available targets;
# if a cell has < 3 illegal_FOLDs we skip it with a clear note. If ALL
# cells for a model have <3, that model is silently dropped from the run
# (printed, but no error).
#
# Cells run
# ---------
# 3 models × 1 seed (s42) × 1 temp (t=0) = 3 cells, layers chosen as a
# tight sweep around the CoT-derived L*:
#   Llama:     layers 10-18 (CoT L*=14)
#   Ministral: layers 12-20 (CoT L*=14-16)
#   Qwen:      layers 16-24 (CoT distributed L=19-23)
#
# Per-cell budget: ~9 layers × 10 sources × N_target patched forwards
# (~30 if N_target available). Roughly 30-60 min per cell on H100,
# ~1.5-3 h total.
#
# Outputs
# -------
#   results/causal_patching/llama8b_nocot_parity/
#   results/causal_patching/ministral8b_nocot_parity/
#   results/causal_patching/qwen8b_nocot_parity/
#
# Headline reading
# ----------------
# For each non-CoT SUMMARY.md, read the per-layer table and find the layer
# at which top-1 → CHECK first crosses 50%. Compare to the corresponding
# CoT cell's L*:
#
#     model     | CoT L* | non-CoT L* | interpretation
#     ----------|-------:|-----------:|----------------
#     Llama     | 14     | (read)     | match? circuit is intrinsic.
#     Ministral | 14-16  | (read)     | match? circuit is intrinsic.
#     Qwen      | 19-23  | (read)     | match? distributed circuit is intrinsic.
#
# Env knobs
# ---------
#   MODELS=llama,ministral,qwen   subset / order
#   SEED_TAG=s42                  which seed's non-CoT log to use (s42, s123, s456)
#   N_SOURCE                      default 10
#   N_TARGET_OVERRIDE             force a specific target count
#   N_RANDOM_CONTROL              default 5
#   SEED                          default 42 (RNG for sampling)
#   DEVICE / DTYPE                default cuda / bfloat16
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_SOURCE="${N_SOURCE:-10}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
SEED="${SEED:-42}"
SEED_TAG="${SEED_TAG:-s42}"
N_TARGET_OVERRIDE="${N_TARGET_OVERRIDE:-}"
MODELS_ENV="${MODELS:-llama,ministral,qwen}"

# Layer windows tightly around the CoT-derived L* per model.
LLAMA_LAYERS="${LLAMA_LAYERS:-10 11 12 13 14 15 16 17 18}"
MINISTRAL_LAYERS="${MINISTRAL_LAYERS:-12 13 14 15 16 17 18 19 20}"
QWEN_LAYERS="${QWEN_LAYERS:-16 17 18 19 20 21 22 23 24}"

# Helper: count illegal_FOLDs in a single non-CoT enriched log.
count_illegal_folds() {
    local enriched="$1"
    if [[ ! -f "$enriched" ]]; then
        echo "0"; return
    fi
    python - <<PY
import sys
sys.path.insert(0, ".")
from experiments.causal_patching import _iter_decisions, classify_decision
n = 0
for rec in _iter_decisions("$enriched"):
    if rec.get("action_metadata") and rec["action_metadata"].get("raw_response"):
        if classify_decision(rec) == "illegal_fold":
            n += 1
print(n)
PY
}

run_one() {
    local short="$1"
    local model_tag="$2"          # llama8b | ministral8b | qwen8b
    local layers_var="$3"
    local out_dir="$4"

    local enriched="logs/scaled_${model_tag}_t0_${SEED_TAG}_informative_v2_enriched.jsonl"
    local layers="${!layers_var}"

    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already has SUMMARY.md"
        return 0
    fi
    if [[ ! -f "$enriched" ]]; then
        # Try a .gz fallback in case the local convention differs.
        if [[ -f "${enriched}.gz" ]]; then
            enriched="${enriched}.gz"
        else
            echo "[skip] $short: missing $enriched (and no .gz fallback)"
            return 0
        fi
    fi

    local n_avail
    n_avail=$(count_illegal_folds "$enriched")
    local n_target
    if [[ -n "$N_TARGET_OVERRIDE" ]]; then
        n_target="$N_TARGET_OVERRIDE"
    else
        n_target="$n_avail"
    fi
    if [[ "$n_target" -lt 3 ]]; then
        echo "[skip] $short: only $n_avail illegal_FOLDs in non-CoT log "
        echo "       (< 3 minimum). Non-CoT runs typically have far fewer "
        echo "       illegal_FOLDs because the verb is committed in one step."
        echo "       To force a run anyway, set N_TARGET_OVERRIDE=N (N>=1)."
        return 0
    fi

    mkdir -p "$out_dir"
    local n_layers
    n_layers=$(echo $layers | wc -w | tr -d ' ')

    echo
    echo "############################################################"
    echo "## NON-CoT parity: $short ($SEED_TAG, t=0)"
    echo "##   illegal_FOLDs available : $n_avail"
    echo "##   n_target (this run)     : $n_target"
    echo "##   layers ($n_layers)         : $layers"
    echo "##   enriched                : $enriched"
    echo "##   out_dir                 : $out_dir"
    echo "############################################################"

    echo
    echo "[pre-flight 1/2] verify position mapping ..."
    python -m experiments.verify_position_mapping \
        --enriched-log "$enriched" --n-samples 10 \
        || { echo "ERROR: position-mapping failed for $short"; return 1; }

    echo
    echo "[pre-flight 2/2] verify prompt reconstruction (GPU) ..."
    python -m experiments.verify_prompt_reconstruction \
        --enriched-log "$enriched" --n-samples 5 \
        --device "$DEVICE" --dtype "$DTYPE" \
        || { echo "ERROR: prompt-reconstruction failed for $short"; return 1; }

    echo
    echo "[main] forward-direction layer sweep (NON-CoT input) ..."
    python -m experiments.causal_patching \
        --enriched-log "$enriched" \
        --source-bucket clean_check_or_call \
        --target-bucket illegal_fold \
        --layers $layers \
        --n-source "$N_SOURCE" \
        --n-target "$n_target" \
        --n-random-control "$N_RANDOM_CONTROL" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" \
        --dtype "$DTYPE"

    echo
    echo "[done] $short wrote $out_dir/SUMMARY.md"
}

IFS=',' read -r -a MODELS_ARR <<< "$MODELS_ENV"
for short in "${MODELS_ARR[@]}"; do
    case "$short" in
        llama)
            run_one llama llama8b LLAMA_LAYERS \
                results/causal_patching/llama8b_nocot_parity
            ;;
        ministral)
            run_one ministral ministral8b MINISTRAL_LAYERS \
                results/causal_patching/ministral8b_nocot_parity
            ;;
        qwen)
            run_one qwen qwen8b QWEN_LAYERS \
                results/causal_patching/qwen8b_nocot_parity
            ;;
        *)
            echo "WARNING: unknown model '$short' — skipping"
            ;;
    esac
done

echo
echo "============================================================"
echo "Non-CoT parity COMPLETE."
echo
echo "Compare boundary L* between non-CoT and CoT for each model:"
echo
echo "  Llama:"
echo "    non-CoT: results/causal_patching/llama8b_nocot_parity/SUMMARY.md"
echo "    CoT:     results/causal_patching/llama8b_t0_pooled_layer_sweep/SUMMARY.md"
echo
echo "  Ministral:"
echo "    non-CoT: results/causal_patching/ministral8b_nocot_parity/SUMMARY.md"
echo "    CoT:     results/causal_patching/ministral8b_t0_s42_layer_sweep/SUMMARY.md"
echo
echo "  Qwen:"
echo "    non-CoT: results/causal_patching/qwen8b_nocot_parity/SUMMARY.md"
echo "    CoT:     results/causal_patching/qwen8b_t0_pooled_layer_sweep/SUMMARY.md"
echo
echo "Hypothesis A (likely): L* matches between CoT and non-CoT in each"
echo "model. Circuit is INTRINSIC to the model, not induced by CoT."
echo "Hypothesis B (interesting if true): L* shifts. CoT is reshaping"
echo "internals, not just outputs."
echo "============================================================"
