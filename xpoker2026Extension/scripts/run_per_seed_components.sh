#!/usr/bin/env bash
# =============================================================================
# Per-seed component decomposition at Llama L=14.
# =============================================================================
#
# Why this exists
# ---------------
# The Llama L=14 head story (h5/h23/h24) is currently established only at
# the POOLED level (s42 + s123 + s456). A defensive reviewer will ask
# "are those heads consistent within each seed, or did pooling smear the
# finding?" This script answers by running the per-head sweep on EACH SEED
# SEPARATELY and we then compare the top-3 sets across seeds.
#
# Pass criterion: the same {h05, h23, h24} appear as top-3 (within ±1
# head identity) in all three per-seed runs. If so, the pooled finding is
# robust at the seed level. If different heads dominate per seed, that's
# its own story and we report it honestly.
#
# Wall-clock: ~50 min × 3 seeds = ~150 min on H100.
#
# Outputs
# -------
#   results/causal_patching/llama8b_l14_components_s42/
#   results/causal_patching/llama8b_l14_components_s123/
#   results/causal_patching/llama8b_l14_components_s456/
# (Each contains SUMMARY_components.md + by_pair_components.csv +
#  summary_components.json — same format as the pooled run.)
#
# Env knobs
# ---------
#   SEEDS_TO_RUN="s42 s123 s456"   subset / order
#   LAYER                           default 14
#   N_SOURCE                        default 10
#   N_TARGET                        default 30 (auto-capped to available)
#   SEED                            default 42 (RNG)
#   DEVICE / DTYPE                  default cuda / bfloat16
# =============================================================================

set -uo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
SEED="${SEED:-42}"
LAYER="${LAYER:-14}"
SEEDS_TO_RUN="${SEEDS_TO_RUN:-s42 s123 s456}"

count_in_bucket() {
    local bucket="$1"
    local enriched="$2"
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
    local seed_tag="$1"
    local enriched="logs/cot_llama8b_t0_${seed_tag}_informative_v2_logitlens_enriched.jsonl.gz"
    local out_dir="results/causal_patching/llama8b_l${LAYER}_components_${seed_tag}"

    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY_components.md" ]]; then
        echo "[skip] $out_dir already has SUMMARY_components.md"
        return 0
    fi
    if [[ ! -f "$enriched" ]]; then
        echo "[skip] $seed_tag: missing $enriched"
        return 0
    fi

    local n_avail
    n_avail=$(count_in_bucket illegal_fold "$enriched")
    local n_target
    if [[ "$n_avail" -ge "$N_TARGET" ]]; then
        n_target="$N_TARGET"
    else
        n_target="$n_avail"
    fi
    if [[ "$n_target" -lt 3 ]]; then
        echo "[skip] $seed_tag: only $n_avail illegal_FOLDs"
        return 0
    fi

    mkdir -p "$out_dir"

    echo
    echo "############################################################"
    echo "## PER-SEED Llama L=$LAYER components: $seed_tag"
    echo "##   illegal_FOLD n_avail=$n_avail  using=$n_target"
    echo "##   out: $out_dir"
    echo "############################################################"

    # Skip pre-flights here for speed — these logs already passed the
    # gate during phase H. (If a reviewer wants we can re-enable.)

    python -m experiments.component_patching \
        --enriched-log "$enriched" \
        --source-bucket clean_check_or_call \
        --target-bucket illegal_fold \
        --layer "$LAYER" \
        --components residual attn mlp head \
        --head-indices all \
        --n-source "$N_SOURCE" \
        --n-target "$n_target" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" \
        --dtype "$DTYPE"

    echo "[done] $seed_tag wrote $out_dir/SUMMARY_components.md"
}

for s in $SEEDS_TO_RUN; do
    run_one "$s"
done

echo
echo "============================================================"
echo "Per-seed Llama L=$LAYER component sweeps COMPLETE."
echo
echo "Compare top-3 heads (by ratio_to_residual, positive only) across:"
for s in $SEEDS_TO_RUN; do
    echo "  results/causal_patching/llama8b_l${LAYER}_components_${s}/SUMMARY_components.md"
done
echo
echo "Pass: same {5, 23, 24} (or near-equivalent) across all per-seed runs."
echo "Pooled-vs-per-seed story holds if yes; revise the writeup if no."
echo "============================================================"
