#!/usr/bin/env bash
# =============================================================================
# Non-CoT circuit-vs-failure-mode test:
#   "Does the L* circuit exist in non-CoT, or only the FAILURE MODE that
#    uses it?"
# =============================================================================
#
# Why this exists
# ---------------
# A3 (`nocot_parity_a3/`) showed that non-CoT mode produces ZERO illegal_fold
# decisions across 18 conditions. We've been describing this as "the L*
# circuit is CoT-induced." But that conflates two distinct claims:
#
#   (i)  The illegal_fold FAILURE MODE is CoT-conditional. ← demonstrated
#   (ii) The L* CIRCUIT itself is CoT-conditional.        ← NOT demonstrated
#
# These are different. The circuit could exist in non-CoT mode and just
# never get exercised pathologically. A reviewer will catch this slippage.
#
# This script does the experiment that distinguishes (i) from (ii):
#
#   Source:  clean_check_or_call (in non-CoT mode)
#   Target:  clean_legal_fold    (in non-CoT mode)
#   Patch at the model's L*:
#     Llama L=14 / Ministral L=16 / Qwen L=23
#
#   If the verb flips toward CHECK in the patched target → the L* circuit
#   ENCODES CHECK as content in non-CoT mode too. The "failure mode is
#   CoT-conditional" framing is correct; "the circuit is CoT-conditional"
#   is wrong. Strong rephrasing for the paper:
#       "L* is the model's intrinsic action-decision circuit; CoT exposes
#        a pathway through it (the FOLD-pull-then-stuck failure) that
#        one-shot decoding never traverses."
#
#   If the verb does NOT flip → the L* circuit really IS CoT-induced.
#   The deliberation-circuit framing is correct as written.
#
# Either outcome is paper-grade. There is no "negative result" here.
#
# Cells run
# ---------
#   Llama 8B non-CoT s42, L=14 (most decisions per cell, cleanest data)
#   Qwen 8B non-CoT s42, L=23 (saturation layer per pooled-CoT result)
#   Ministral 8B non-CoT s42, L=16 — SKIPPED by default because Ministral's
#     non-CoT cells have <30 clean decisions each (insufficient n).
#     Override with FORCE_MINISTRAL=1 if you want to try anyway.
#
# Layer choices: same L* identified in the CoT pooled sweeps. We are NOT
# doing a layer sweep here — the question is "does the circuit exist at
# the same L* in non-CoT?" not "is there a different L* in non-CoT?"
# (If the answer to the first is "no", a layer sweep would be the
# follow-up. Cheap to add later.)
#
# Wall-clock: ~30-40 min per cell (no layer sweep), ~60-80 min for both.
#
# Outputs
# -------
#   results/causal_patching/llama8b_nocot_clean_to_clean_l14/
#   results/causal_patching/qwen8b_nocot_clean_to_clean_l23/
#   (Ministral cell only if FORCE_MINISTRAL=1 — likely n_target < 3 anyway)
#
# Each contains SUMMARY.md (per-layer table; one layer here = one row) +
# summary.json + by_pair.csv.
#
# Headline reading
# ----------------
# Read the `top-1 → CHECK-family` column at the patched layer. Pass criteria:
#
#   ≥50% top-1 → CHECK-family AND specificity-adjusted Δ > +5 nats:
#       circuit IS intrinsic. Failure mode is the only CoT-conditional thing.
#
#   <10% top-1 → CHECK-family OR spec-adj Δ < +2 nats:
#       circuit IS CoT-induced. Original framing stands.
#
#   Anything in between: nuanced; the patching has weakened effect in
#       non-CoT mode but isn't gone. Worth reporting both numbers.
#
# Compare directly to the corresponding CoT pooled-sweep number at the
# same L* (e.g. Llama CoT L=14 spec-adj Δ ≈ +6.48, top-1 → CHECK ≈ 79%).
#
# Env knobs
# ---------
#   MODELS=llama,qwen[,ministral]   subset / order
#   SEED_TAG=s42                    which non-CoT cell's log to use
#   N_SOURCE                        default 10
#   N_TARGET                        default 30 (auto-capped to available)
#   N_RANDOM_CONTROL                default 5
#   SEED                            default 42 (RNG)
#   DEVICE / DTYPE                  default cuda / bfloat16
#   FORCE_MINISTRAL=1               try Ministral despite low n
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
SEED="${SEED:-42}"
SEED_TAG="${SEED_TAG:-s42}"
MODELS_ENV="${MODELS:-llama,qwen}"   # ministral excluded by default
FORCE_MINISTRAL="${FORCE_MINISTRAL:-0}"
BASELINE_TOLERANCE_FRAC="${BASELINE_TOLERANCE_FRAC:-0.95}"

# Per-model L* (from CoT pooled sweeps).
LLAMA_LAYER=14
MINISTRAL_LAYER=16
QWEN_LAYER=23

count_in_bucket() {
    local enriched="$1"
    local bucket="$2"
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
        if classify_decision(rec) == "$bucket":
            n += 1
print(n)
PY
}

run_one() {
    local short="$1"          # llama | ministral | qwen
    local model_tag="$2"      # llama8b | ministral8b | qwen8b
    local layer="$3"
    local out_dir="$4"

    local enriched="logs/scaled_${model_tag}_t0_${SEED_TAG}_informative_v2_enriched.jsonl"

    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already has SUMMARY.md"
        return 0
    fi
    if [[ ! -f "$enriched" ]]; then
        if [[ -f "${enriched}.gz" ]]; then
            enriched="${enriched}.gz"
        else
            echo "[skip] $short: missing $enriched (and no .gz fallback)"
            return 0
        fi
    fi

    local n_source_avail
    n_source_avail=$(count_in_bucket "$enriched" clean_check_or_call)
    local n_target_avail
    n_target_avail=$(count_in_bucket "$enriched" clean_legal_fold)

    if [[ "$n_source_avail" -lt 3 ]] || [[ "$n_target_avail" -lt 3 ]]; then
        echo
        echo "[skip] $short non-CoT $SEED_TAG has insufficient buckets:"
        echo "       clean_check_or_call available: $n_source_avail"
        echo "       clean_legal_fold    available: $n_target_avail"
        echo "       (need >= 3 of each — Ministral non-CoT typically fails this)"
        return 0
    fi

    local n_target_to_use="$N_TARGET"
    if [[ "$n_target_avail" -lt "$N_TARGET" ]]; then
        n_target_to_use="$n_target_avail"
    fi
    local n_source_to_use="$N_SOURCE"
    if [[ "$n_source_avail" -lt "$N_SOURCE" ]]; then
        n_source_to_use="$n_source_avail"
    fi

    mkdir -p "$out_dir"

    echo
    echo "############################################################"
    echo "## CIRCUIT-vs-FAILURE non-CoT test: $short at L=$layer"
    echo "##   source=clean_check_or_call (n_avail=$n_source_avail, using=$n_source_to_use)"
    echo "##   target=clean_legal_fold    (n_avail=$n_target_avail, using=$n_target_to_use)"
    echo "##   non-CoT log: $enriched"
    echo "##   out: $out_dir"
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
    echo "[main] non-CoT clean-to-clean patching at L=$layer ..."
    python -m experiments.causal_patching \
        --enriched-log "$enriched" \
        --source-bucket clean_check_or_call \
        --target-bucket clean_legal_fold \
        --layers "$layer" \
        --n-source "$n_source_to_use" \
        --n-target "$n_target_to_use" \
        --n-random-control "$N_RANDOM_CONTROL" \
        --seed "$SEED" \
        --baseline-tolerance-frac "$BASELINE_TOLERANCE_FRAC" \
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
            run_one llama llama8b "$LLAMA_LAYER" \
                "results/causal_patching/llama8b_nocot_clean_to_clean_l${LLAMA_LAYER}"
            ;;
        ministral)
            if [[ "$FORCE_MINISTRAL" != "1" ]]; then
                echo "[skip] Ministral non-CoT (insufficient n; set FORCE_MINISTRAL=1 to try)"
                continue
            fi
            run_one ministral ministral8b "$MINISTRAL_LAYER" \
                "results/causal_patching/ministral8b_nocot_clean_to_clean_l${MINISTRAL_LAYER}"
            ;;
        qwen)
            run_one qwen qwen8b "$QWEN_LAYER" \
                "results/causal_patching/qwen8b_nocot_clean_to_clean_l${QWEN_LAYER}"
            ;;
        *)
            echo "WARNING: unknown model '$short' — skipping"
            ;;
    esac
done

echo
echo "============================================================"
echo "Non-CoT circuit-vs-failure-mode test COMPLETE."
echo
echo "Read each SUMMARY.md and look at the top-1 → CHECK-family column"
echo "AND the specificity-adjusted Δ at the patched layer."
echo
echo "Compare against the CoT pooled-sweep numbers at the same L*:"
echo "  Llama CoT L=14:    spec-adj Δ ≈ +6.48, top-1 → CHECK ≈ 79%"
echo "  Qwen CoT L=23:     spec-adj Δ ≈ +18.3, top-1 → CHECK ≈ 100%"
echo "  Ministral CoT L=16: spec-adj Δ ≈ +7.81, top-1 → CHECK ≈ 100% (skipped here)"
echo
echo "PASS (circuit intrinsic):   non-CoT spec-adj Δ > +5 nats AND flip ≥ 50%"
echo "FAIL (circuit CoT-induced): non-CoT spec-adj Δ < +2 nats AND flip < 10%"
echo "MIXED:                      report both numbers; weakened-but-present circuit"
echo
echo "Either outcome is paper-grade — there is no 'experiment failed' result."
echo "============================================================"
