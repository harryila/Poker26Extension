#!/usr/bin/env bash
# =============================================================================
# Non-CoT circuit hunt — match the CoT-level mechanistic testing on non-CoT.
# =============================================================================
#
# Why this exists
# ---------------
# Phase L (§17) demonstrated the L*=23 circuit is INTRINSIC in Qwen non-CoT
# (88.7% top-1 → CHECK on a clean baseline). To make the cross-model
# claim symmetric, we need to do the analogous testing for Llama and (where
# possible) Ministral non-CoT. This script runs the full mechanistic stack
# against the non-CoT baseline logs:
#
#   Section A — layer sweep (find non-CoT L*)
#   Section B — per-seed concordance at non-CoT L*
#   Section C — component decomposition at non-CoT L* (residual / attn / mlp / per-head)
#   Section D — verb-pair generality matrix in non-CoT (4 of 6 directions)
#
# Together with the existing direction probe + attention-pattern scripts (run
# separately), this produces the same level of mechanistic evidence for non-CoT
# as we have for CoT.
#
# Critical design constraint: Llama non-CoT clean_legal_fold has only 20%
# residual-top-1 = FOLD (the §17c muddle). All Llama cells in this script
# pass `--target-residual-top1 FOLD` to filter out the 80% of mislabeled
# targets. Qwen non-CoT does not need the filter (baseline_top1 = 1.00).
#
# Ministral non-CoT is mostly empty across seeds (only s42 t=0 has 30
# clean_check_or_call records; s123/s456 have 0). We attempt only the s42
# layer sweep for Ministral; per-seed concordance is impossible.
#
# Wall-clock estimate
# -------------------
# Section A: ~30-40 min × 2 models (Llama, Qwen) = ~70-80 min
# Section B: ~30-40 min × 6 cells (3 seeds × 2 models) = ~3-4 h
# Section C: ~50 min × 2 models = ~100 min
# Section D: ~30 min × 4 verb-pair × 2 models = ~4 h
# Total: ~9-11 h. Acceptable for an overnight run.
#
# Outputs
# -------
# results/causal_patching/<model>8b_nocot_layer_sweep_s42/        (Section A)
# results/causal_patching/<model>8b_nocot_perseed_<seed>_l<L>/    (Section B)
# results/causal_patching/<model>8b_nocot_l<L>_components/        (Section C)
# results/causal_patching/<model>8b_nocot_verbpair_*/             (Section D)
#
# Env knobs
# ---------
#   SECTIONS=A,B,C,D       subset / order
#   SEEDS_TO_RUN=s42 s123 s456   (Section B)
#   LLAMA_LAYER=14         L* used for Section B/C/D
#   QWEN_LAYER=23          (same)
#   N_SOURCE / N_TARGET    defaults 10 / 30
#   N_RANDOM_CONTROL       default 5
#   SEED                   default 42 (RNG)
#   DEVICE / DTYPE         default cuda / bfloat16
# =============================================================================

set -uo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
SEED="${SEED:-42}"
LLAMA_LAYER="${LLAMA_LAYER:-14}"
QWEN_LAYER="${QWEN_LAYER:-23}"
SECTIONS="${SECTIONS:-A,B,C,D}"
SEEDS_TO_RUN="${SEEDS_TO_RUN:-s42 s123 s456}"

# Layer sweep range per model — wider than CoT to allow non-CoT L* to drift.
LLAMA_LAYERS_SWEEP="${LLAMA_LAYERS_SWEEP:-8 10 12 14 16 18 20}"
QWEN_LAYERS_SWEEP="${QWEN_LAYERS_SWEEP:-15 18 21 23 25 28 31}"

# Filter: Llama clean_legal_fold needs the residual-top1 fix; Qwen doesn't.
# Set to "" to disable filter.
LLAMA_FILTER="--target-residual-top1 FOLD --baseline-tolerance-frac 0.0"
QWEN_FILTER=""

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

# Helper: run one cell. Auto-skip if SUMMARY.md exists or if input missing.
run_cell() {
    local label="$1"
    local enriched="$2"
    local source_bucket="$3"
    local target_bucket="$4"
    local layers="$5"
    local out_dir="$6"
    local extra_args="$7"

    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already has SUMMARY.md"
        return 0
    fi
    if [[ ! -f "$enriched" ]]; then
        if [[ -f "${enriched}.gz" ]]; then
            enriched="${enriched}.gz"
        else
            echo "[skip] $label: missing $enriched"
            return 0
        fi
    fi

    # Pre-count source and target buckets.
    local n_src
    n_src=$(count_in_bucket "$enriched" "$source_bucket")
    local n_tgt
    n_tgt=$(count_in_bucket "$enriched" "$target_bucket")
    if [[ "$n_src" -lt 3 ]] || [[ "$n_tgt" -lt 3 ]]; then
        echo "[skip] $label: insufficient buckets ($source_bucket=$n_src, $target_bucket=$n_tgt)"
        return 0
    fi

    local n_target_to_use="$N_TARGET"
    if [[ "$n_tgt" -lt "$N_TARGET" ]]; then
        n_target_to_use="$n_tgt"
    fi
    local n_source_to_use="$N_SOURCE"
    if [[ "$n_src" -lt "$N_SOURCE" ]]; then
        n_source_to_use="$n_src"
    fi

    mkdir -p "$out_dir"

    echo
    echo "############################################################"
    echo "## $label"
    echo "##   enriched: $enriched"
    echo "##   source=$source_bucket (n_avail=$n_src, using=$n_source_to_use)"
    echo "##   target=$target_bucket (n_avail=$n_tgt, using=$n_target_to_use)"
    echo "##   layers: $layers"
    echo "##   filter: $extra_args"
    echo "##   out: $out_dir"
    echo "############################################################"

    python -m experiments.causal_patching \
        --enriched-log "$enriched" \
        --source-bucket "$source_bucket" \
        --target-bucket "$target_bucket" \
        --layers $layers \
        --n-source "$n_source_to_use" \
        --n-target "$n_target_to_use" \
        --n-random-control "$N_RANDOM_CONTROL" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        $extra_args
    echo "[done] $label wrote $out_dir/SUMMARY.md"
}

run_components_cell() {
    local label="$1"
    local enriched="$2"
    local source_bucket="$3"
    local target_bucket="$4"
    local layer="$5"
    local out_dir="$6"

    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY_components.md" ]]; then
        echo "[skip] $out_dir already has SUMMARY_components.md"
        return 0
    fi
    if [[ ! -f "$enriched" ]]; then
        echo "[skip] $label: missing $enriched"
        return 0
    fi
    # Components driver doesn't yet support --target-residual-top1; for Llama
    # we pre-filter target hand-ids by running the residual driver first and
    # extracting the kept set. Simpler approach: trust that the component
    # decomposition at L* on residual-FOLD targets only matters once the
    # residual-driver test confirms a flip. For now, skip components on Llama
    # if we don't have residual-FOLD-only targets readily available.
    # SO: the component decomposition will run on the recorded-action label
    # but with a permissive baseline tolerance. The flip-rate column may show
    # the "2/30 flip" pattern that's an artifact; the per-head ratio_to_residual
    # column is still informative.

    mkdir -p "$out_dir"
    echo
    echo "############################################################"
    echo "## $label  (NB: component decomposition uses recorded-action labels)"
    echo "##   enriched: $enriched"
    echo "##   layer: $layer"
    echo "##   out: $out_dir"
    echo "############################################################"

    python -m experiments.component_patching \
        --enriched-log "$enriched" \
        --source-bucket "$source_bucket" \
        --target-bucket "$target_bucket" \
        --layer "$layer" \
        --components residual attn mlp head \
        --head-indices all \
        --n-source "$N_SOURCE" \
        --n-target "$N_TARGET" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" \
        --dtype "$DTYPE"
    echo "[done] $label wrote $out_dir/SUMMARY_components.md"
}

# -----------------------------------------------------------------------------
# SECTION A — Non-CoT layer sweep (Llama, Qwen)
# -----------------------------------------------------------------------------
if [[ ",$SECTIONS," == *,A,* ]]; then
    echo
    echo "================================================================"
    echo "SECTION A — non-CoT layer sweep (forward, clean→clean)"
    echo "================================================================"

    # Llama: baseline pathology — use --target-residual-top1 FOLD.
    run_cell "Llama non-CoT s42 layer sweep (filtered)" \
        "logs/scaled_llama8b_t0_s42_informative_v2_enriched.jsonl" \
        clean_check_or_call clean_legal_fold \
        "$LLAMA_LAYERS_SWEEP" \
        "results/causal_patching/llama8b_nocot_layer_sweep_s42_filtered" \
        "$LLAMA_FILTER"

    # Qwen: baseline is clean.
    run_cell "Qwen non-CoT s42 layer sweep" \
        "logs/scaled_qwen8b_t0_s42_informative_v2_enriched.jsonl" \
        clean_check_or_call clean_legal_fold \
        "$QWEN_LAYERS_SWEEP" \
        "results/causal_patching/qwen8b_nocot_layer_sweep_s42" \
        "$QWEN_FILTER"

    # Ministral: try s42, will likely auto-skip due to bucket emptiness.
    run_cell "Ministral non-CoT s42 layer sweep" \
        "logs/scaled_ministral8b_t0_s42_informative_v2_enriched.jsonl" \
        clean_check_or_call clean_legal_fold \
        "$LLAMA_LAYERS_SWEEP" \
        "results/causal_patching/ministral8b_nocot_layer_sweep_s42" \
        ""
fi

# -----------------------------------------------------------------------------
# SECTION B — Per-seed concordance at non-CoT L*
# -----------------------------------------------------------------------------
if [[ ",$SECTIONS," == *,B,* ]]; then
    echo
    echo "================================================================"
    echo "SECTION B — non-CoT per-seed concordance at L*"
    echo "================================================================"

    for seed in $SEEDS_TO_RUN; do
        # Llama.
        run_cell "Llama non-CoT $seed at L=$LLAMA_LAYER" \
            "logs/scaled_llama8b_t0_${seed}_informative_v2_enriched.jsonl" \
            clean_check_or_call clean_legal_fold \
            "$LLAMA_LAYER" \
            "results/causal_patching/llama8b_nocot_perseed_${seed}_l${LLAMA_LAYER}" \
            "$LLAMA_FILTER"

        # Qwen.
        run_cell "Qwen non-CoT $seed at L=$QWEN_LAYER" \
            "logs/scaled_qwen8b_t0_${seed}_informative_v2_enriched.jsonl" \
            clean_check_or_call clean_legal_fold \
            "$QWEN_LAYER" \
            "results/causal_patching/qwen8b_nocot_perseed_${seed}_l${QWEN_LAYER}" \
            "$QWEN_FILTER"
    done
fi

# -----------------------------------------------------------------------------
# SECTION C — Non-CoT component decomposition at L*
# -----------------------------------------------------------------------------
if [[ ",$SECTIONS," == *,C,* ]]; then
    echo
    echo "================================================================"
    echo "SECTION C — non-CoT component decomposition at L*"
    echo "================================================================"

    # Llama at L=14.
    run_components_cell "Llama non-CoT s42 components at L=$LLAMA_LAYER" \
        "logs/scaled_llama8b_t0_s42_informative_v2_enriched.jsonl" \
        clean_check_or_call clean_legal_fold \
        "$LLAMA_LAYER" \
        "results/causal_patching/llama8b_nocot_l${LLAMA_LAYER}_components"

    # Qwen at L=23.
    run_components_cell "Qwen non-CoT s42 components at L=$QWEN_LAYER" \
        "logs/scaled_qwen8b_t0_s42_informative_v2_enriched.jsonl" \
        clean_check_or_call clean_legal_fold \
        "$QWEN_LAYER" \
        "results/causal_patching/qwen8b_nocot_l${QWEN_LAYER}_components"
fi

# -----------------------------------------------------------------------------
# SECTION D — Non-CoT verb-pair matrix
# -----------------------------------------------------------------------------
if [[ ",$SECTIONS," == *,D,* ]]; then
    echo
    echo "================================================================"
    echo "SECTION D — non-CoT verb-pair matrix"
    echo "================================================================"
    # Four directions of interest in non-CoT:
    #   1. CHECK → BET_RAISE  (forward)
    #   2. BET_RAISE → CHECK  (reverse — analog of C1)
    #   3. BET_RAISE → FOLD
    #   4. FOLD → BET_RAISE
    # Each at the model's non-CoT L*. Skip Ministral (insufficient data).

    for src_tgt in "clean_check_or_call:clean_bet_or_raise:check_to_bet" \
                   "clean_bet_or_raise:clean_check_or_call:bet_to_check" \
                   "clean_bet_or_raise:clean_legal_fold:bet_to_fold" \
                   "clean_legal_fold:clean_bet_or_raise:fold_to_bet"; do
        IFS=':' read -r src tgt name <<< "$src_tgt"

        # For Llama, the residual-top-1 filter only makes sense when
        # target=clean_legal_fold (the bucket with the pathology). For
        # other targets, baseline tolerance is fine at default.
        if [[ "$tgt" == "clean_legal_fold" ]]; then
            llama_filter_args="$LLAMA_FILTER"
        else
            llama_filter_args=""
        fi

        run_cell "Llama non-CoT s42 verbpair ${name} at L=$LLAMA_LAYER" \
            "logs/scaled_llama8b_t0_s42_informative_v2_enriched.jsonl" \
            "$src" "$tgt" \
            "$LLAMA_LAYER" \
            "results/causal_patching/llama8b_nocot_verbpair_${name}_s42_l${LLAMA_LAYER}" \
            "$llama_filter_args"

        run_cell "Qwen non-CoT s42 verbpair ${name} at L=$QWEN_LAYER" \
            "logs/scaled_qwen8b_t0_s42_informative_v2_enriched.jsonl" \
            "$src" "$tgt" \
            "$QWEN_LAYER" \
            "results/causal_patching/qwen8b_nocot_verbpair_${name}_s42_l${QWEN_LAYER}" \
            ""
    done
fi

echo
echo "============================================================"
echo "Non-CoT circuit hunt COMPLETE."
echo
echo "Read in priority order:"
echo "  1. Section A: layer sweeps to find non-CoT L*."
echo "     (Compare to CoT L* — same? shifted? gone?)"
echo "  2. Section B: per-seed concordance at L*."
echo "  3. Section C: component decomposition at L*."
echo "     (Are the same heads dominating in non-CoT as CoT?)"
echo "  4. Section D: verb-pair matrix in non-CoT."
echo "============================================================"
