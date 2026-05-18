#!/usr/bin/env bash
# =============================================================================
# A3 baseline-cleanup wrapper (Phase P).
# =============================================================================
# Re-runs `direction_probe_baselines` for all 3 models with two fixes:
#
#   1. --cross-task-feature position
#      Replaces the legacy `bet_to_call > 0` cross-task label (which was
#      highly correlated with verb because all FOLD decisions have
#      bet_to_call > 0) with `position == BB`, which is essentially a
#      coin flip and uncorrelated with verb. Cross-task accuracy near
#      0.50 with this feature is the genuine "verb-direction is
#      task-specific" control we wanted in updates.md §19j.3.
#
#   2. --balance-classes
#      Upsamples the minority class to match the majority before CV.
#      Fixes the §19j.4 issue where Ministral's 90/10 CHECK/FOLD class
#      split made permuted-label and random-direction baselines achieve
#      ~0.90 by predicting the majority class, rendering them
#      uninformative. With balanced classes both baselines collapse to
#      ~0.50 (true chance) which is the comparison we want.
#
#   3. --n-permutation-trials 20
#      Replaces the original 1-shuffle permuted-label null with a 20-trial
#      average + std. Tightens the null distribution; addresses audit
#      item M7.
#
#   4. --also-fixed-threshold-random
#      Adds a SECOND random-direction row: instead of picking the
#      accuracy-maximizing threshold per random unit vector (which is
#      an UPPER BOUND on what random projections can achieve), threshold
#      at the median of each projection. This is the conservative null
#      a reviewer will accept. Addresses audit item M8.
#
# Pure analysis on cached `raw_residuals.npz` — no GPU needed. ~5-10 min total.
#
# Outputs (separate dirs from the legacy A3 run so we don't clobber):
#   results/direction_probe_baselines/llama8b_l14_phaseP.md
#   results/direction_probe_baselines/ministral8b_l16_phaseP.md
#   results/direction_probe_baselines/qwen8b_l23_phaseP.md
#
# Auto-skips if the per-model `_phaseP.md` already exists.
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")/.."

# Set FORCE_RERUN=1 to ignore existing *_phaseP.md outputs and re-run.
# Useful after we change the baselines (e.g. multi-permutation null or
# fixed-threshold random-direction control) and want to regenerate the
# SUMMARYs without manually deleting them.
FORCE_RERUN="${FORCE_RERUN:-0}"

run_one() {
    local short="$1"
    local layer="$2"
    shift 2
    local logs="$@"

    local probe_npz="results/direction_probe/${short}8b_l${layer}/raw_residuals.npz"
    local out_md="results/direction_probe_baselines/${short}8b_l${layer}_phaseP.md"

    if [[ -f "$out_md" ]] && [[ "$FORCE_RERUN" != "1" ]]; then
        echo "[skip] $out_md exists (set FORCE_RERUN=1 to override)"
        return 0
    fi
    if [[ ! -f "$probe_npz" ]]; then
        echo "[skip] $short: missing $probe_npz"
        return 0
    fi
    mkdir -p "$(dirname "$out_md")"

    echo
    echo "## A3 cleanup: $short L=$layer (cross-task=position, balanced, multi-perm + fixed-threshold rand)"
    python -m experiments.direction_probe_baselines \
        --probe-npz "$probe_npz" \
        --enriched-log $logs \
        --cross-task-feature position \
        --balance-classes \
        --n-permutation-trials 20 \
        --also-fixed-threshold-random \
        --out-md "$out_md"
}

run_one llama 14 \
    logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

run_one ministral 16 \
    logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

run_one qwen 23 \
    logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

echo
echo "============================================================"
echo "A3 cleanup COMPLETE. Compare to the legacy *.md results in"
echo "results/direction_probe_baselines/{llama,ministral,qwen}8b_l*.md"
echo
echo "Expected: cross-task accuracy DROPS from 0.99 (verb-correlated"
echo "bet_to_call) to ~0.50 (verb-orthogonal position), and Ministral's"
echo "permuted-label drops from 0.90 to ~0.50 with class balancing."
echo "============================================================"
