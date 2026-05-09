#!/usr/bin/env bash
# =============================================================================
# Qwen 8B causal-patching sweep REPLICATION across seeds (cross-seed check).
# =============================================================================
#
# Mirror of scripts/run_causal_patching_llama_seeds_replicate.sh and
# scripts/run_causal_patching_ministral_replicate.sh, but for Qwen.
#
# Why this exists
# ---------------
# The original Qwen sweep (pooled across s42+s123+s456 in
# run_causal_patching_qwen_sweep.sh) confirmed the gradual L=19→23 ramp at
# the pooled level. Llama and Ministral both have 4 cells of per-seed
# evidence (s42 standalone + s123 + s456 + pooled); Qwen has only the pooled
# cell. This script produces the per-seed cells so we can claim cross-seed
# replication for Qwen specifically and write the cross-model story
# symmetrically.
#
# Expected result (paper-banner if confirmed)
# -------------------------------------------
# All three per-seed cells should show:
#   - top-1 → CHECK climbs from <2% at L≤19 to ≥75% at L=22 to 100% at L=23
#   - specificity-adjusted Δ rises monotonically across the 5-layer ramp
# Qualitatively distinct from Llama/Ministral's 2-layer flip at L=14, with
# the boundary at ~55% of model depth (vs ~40% in Llama/Ministral).
#
# A consistent gradual shape across seeds is what lets us write:
#   "localized circuit at depth ~40% in Llama and Ministral; gradual,
#    distributed circuit at depth ~55% in Qwen, in both directions, in all
#    seeds."
# That is the methodologically symmetric cross-model story.
#
# Cells run
# ---------
# - qwen8b s42  t=0.0   (NEW: there is no standalone Qwen s42 sweep yet)
# - qwen8b s123 t=0.0
# - qwen8b s456 t=0.0
#
# Per-cell budget: 36 layers x 10 sources x N_target patched forwards.
# Qwen 8B has 36 layers (matches Ministral, +4 vs Llama). N_target is the
# number of illegal_FOLD decisions actually present in each cell
# (auto-detected, falls back to 1).
#
# Roughly: 30-50 min per cell, ~1.5-2.5 h total on H100.
#
# Outputs
# -------
#   results/causal_patching/qwen8b_t0_s42_replicate/
#   results/causal_patching/qwen8b_t0_s123_replicate/
#   results/causal_patching/qwen8b_t0_s456_replicate/
# Each contains SUMMARY.md (per-layer table + frac_top1_check_call) +
# summary.json + by_pair.csv.
#
# Env knobs
# ---------
#   SEEDS_TO_RUN="s42 s123 s456"  (subset / order of seeds)
#   LAYERS                         (default: all 36 Qwen layers)
#   N_SOURCE                       (default 10)
#   N_TARGET_OVERRIDE              (force a specific target count instead of
#                                   using all available illegal_FOLDs)
#   N_RANDOM_CONTROL               (default 5)
#   SEED                           (default 42 — RNG seed for sampling, NOT
#                                   the cell's data seed)
#   DEVICE / DTYPE                 (default cuda / bfloat16)
# =============================================================================

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LAYERS="${LAYERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35}"
N_SOURCE="${N_SOURCE:-10}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
SEED="${SEED:-42}"
SEEDS_TO_RUN="${SEEDS_TO_RUN:-s42 s123 s456}"
N_TARGET_OVERRIDE="${N_TARGET_OVERRIDE:-}"

cd "$(dirname "$0")/.."  # repo root

# -----------------------------------------------------------------------------
# Helper: count illegal_FOLD decisions in a single enriched log.
#
# We use the same classifier the patching driver uses, so the count matches
# what the driver will see at run time. If the file is absent OR the count
# is zero, the cell is skipped with a clear note.
# -----------------------------------------------------------------------------
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
    local seed_tag="$1"          # s42, s123, or s456
    local enriched="logs/cot_qwen8b_t0_${seed_tag}_informative_v2_logitlens_enriched.jsonl.gz"
    local out_dir="results/causal_patching/qwen8b_t0_${seed_tag}_replicate"

    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already has SUMMARY.md"
        return 0
    fi
    if [[ ! -f "$enriched" ]]; then
        echo "[skip] enriched log missing: $enriched"
        return 0
    fi

    local n_avail
    n_avail=$(count_illegal_folds "$enriched")
    local n_target
    if [[ -n "$N_TARGET_OVERRIDE" ]]; then
        n_target="$N_TARGET_OVERRIDE"
    else
        n_target="$n_avail"
    fi
    if [[ "$n_target" -lt 1 ]]; then
        echo "[skip] $seed_tag has 0 illegal_FOLDs available — nothing to patch into"
        return 0
    fi

    mkdir -p "$out_dir"
    local n_layers
    n_layers=$(echo $LAYERS | wc -w | tr -d ' ')

    echo
    echo "############################################################"
    echo "## Qwen replication cell: t=0 ${seed_tag}"
    echo "##   illegal_FOLDs available : $n_avail"
    echo "##   n_target (this run)     : $n_target"
    echo "##   layers ($n_layers)            : $LAYERS"
    echo "##   enriched                : $enriched"
    echo "##   out_dir                 : $out_dir"
    echo "############################################################"

    echo
    echo "[pre-flight 1/2] verify position mapping ..."
    python -m experiments.verify_position_mapping \
        --enriched-log "$enriched" --n-samples 10 \
        || { echo "ERROR: position-mapping failed for $seed_tag"; return 1; }

    echo
    echo "[pre-flight 2/2] verify prompt reconstruction (GPU) ..."
    python -m experiments.verify_prompt_reconstruction \
        --enriched-log "$enriched" --n-samples 5 \
        --device "$DEVICE" --dtype "$DTYPE" \
        || { echo "ERROR: prompt-reconstruction failed for $seed_tag"; return 1; }

    echo
    echo "[main] forward-direction layer sweep ..."
    python -m experiments.causal_patching \
        --enriched-log "$enriched" \
        --source-bucket clean_check_or_call \
        --target-bucket illegal_fold \
        --layers $LAYERS \
        --n-source "$N_SOURCE" \
        --n-target "$n_target" \
        --n-random-control "$N_RANDOM_CONTROL" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" \
        --dtype "$DTYPE"

    echo
    echo "[done] $seed_tag wrote $out_dir/SUMMARY.md"
}

for seed_tag in $SEEDS_TO_RUN; do
    run_one "$seed_tag"
done

echo
echo "============================================================"
echo "Qwen seed replication COMPLETE."
echo
echo "Compare boundary across:"
echo "  - results/causal_patching/qwen8b_t0_s42_replicate/SUMMARY.md   (NEW)"
echo "  - results/causal_patching/qwen8b_t0_s123_replicate/SUMMARY.md  (NEW)"
echo "  - results/causal_patching/qwen8b_t0_s456_replicate/SUMMARY.md  (NEW)"
echo "  - results/causal_patching/qwen8b_t0_pooled_layer_sweep/SUMMARY.md (existing pooled)"
echo
echo "Hypothesis: all three per-seed cells show a gradual ramp L=19→23,"
echo "with top-1 → CHECK climbing from <2% to 100%, in roughly the same"
echo "shape across all seeds. This confirms the gradual non-localized"
echo "pattern is a Qwen architectural/training signature (not a seed"
echo "or pooling artifact) and lets us write the cross-model story"
echo "symmetrically: localized at L=14 in Llama+Ministral, distributed"
echo "across L=19-23 in Qwen, in every seed."
echo "============================================================"
