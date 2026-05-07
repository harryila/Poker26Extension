#!/usr/bin/env bash
# =============================================================================
# Llama 8B causal-patching sweep REPLICATION across seeds (cross-seed check).
# =============================================================================
#
# Mirror of scripts/run_causal_patching_ministral_replicate.sh, but for Llama.
#
# The original Llama sweeps (s42_only and pooled in
# run_causal_patching_llama_sweep.sh) confirmed the L=12 boundary at the
# pooled-and-s42-mixed level. To match the Ministral evidence — where 4
# different seed/temp cells all show the same L=14-16 boundary — this script
# runs the per-seed Llama cells separately so we can claim cross-seed
# replication for Llama specifically.
#
# Cells run:
#   - llama8b s123 t=0.0   (different seed, same temp as the original sweep)
#   - llama8b s456 t=0.0   (different seed, same temp as the original sweep)
#
# (s42 t=0 is already done as the standalone llama8b_t0_s42_layer_sweep.)
#
# Per-cell budget: 32 layers x 10 sources x N_target patched forwards.
# Llama 8B has 32 layers (vs Ministral's 36), so each cell is ~10% cheaper
# than the Ministral analog. n_target is set to the number of illegal_FOLD
# decisions actually present in that cell (auto-detected, falls back to 1).
#
# Roughly: 30-50 min per cell, ~1-1.5 h total on H100.
#
# Outputs:
#   results/causal_patching/llama8b_t0_s123_replicate/
#   results/causal_patching/llama8b_t0_s456_replicate/
# Each contains SUMMARY.md (per-layer table + frac_top1_check_call) +
# summary.json + by_pair.csv.
#
# Env knobs:
#   SEEDS_TO_RUN="s123 s456"      (subset / order of seeds)
#   LAYERS                        (default: all 32 Llama layers)
#   N_SOURCE                      (default 10)
#   N_TARGET_OVERRIDE             (force a specific target count instead of
#                                  using all available illegal_FOLDs)
#   N_RANDOM_CONTROL              (default 5)
#   SEED                          (default 42 — RNG seed for sampling, NOT
#                                   the cell's data seed)
#   DEVICE / DTYPE                (default cuda / bfloat16)
# =============================================================================

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LAYERS="${LAYERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31}"
N_SOURCE="${N_SOURCE:-10}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
SEED="${SEED:-42}"
SEEDS_TO_RUN="${SEEDS_TO_RUN:-s123 s456}"
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
    local seed_tag="$1"          # s123 or s456
    local enriched="logs/cot_llama8b_t0_${seed_tag}_informative_v2_logitlens_enriched.jsonl.gz"
    local out_dir="results/causal_patching/llama8b_t0_${seed_tag}_replicate"

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
    echo "## Llama replication cell: t=0 ${seed_tag}"
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
echo "Llama seed replication COMPLETE."
echo
echo "Compare boundary L* across:"
echo "  - results/causal_patching/llama8b_t0_s42_layer_sweep/SUMMARY.md   (original)"
echo "  - results/causal_patching/llama8b_t0_s123_replicate/SUMMARY.md    (this run)"
echo "  - results/causal_patching/llama8b_t0_s456_replicate/SUMMARY.md    (this run)"
echo "  - results/causal_patching/llama8b_t0_pooled_layer_sweep/SUMMARY.md (3-seed pool)"
echo
echo "Hypothesis: all per-seed cells show transition at L=12-14 (matching"
echo "the Ministral cross-seed concordance at L=14-16 / ~38% of model depth)."
echo "============================================================"
