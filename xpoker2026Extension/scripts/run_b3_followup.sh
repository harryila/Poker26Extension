#!/usr/bin/env bash
# =============================================================================
# B3 follow-up wrapper (Phase P).
# =============================================================================
# Two follow-ups to the original belief × verb orthogonality result
# (updates.md §19a):
#
#   FOLLOWUP-1: HELD-OUT R^2 with 5-fold CV.
#       Original B3 reported in-sample R^2 = 0.999 for Qwen, which is
#       suspiciously high given hidden_dim=4096 ≫ n_samples=300. Need
#       to confirm via held-out CV before publishing the "belief is
#       highly decodable" claim. (§19a caveat #1.)
#
#   FOLLOWUP-2: --belief-source agent_belief.
#       Original B3 used `oracle_strategy_aware`. The reviewer-natural
#       follow-up is "is the model's OWN stated belief (parsed from CoT)
#       also orthogonal to verb?" — closes the circle on the belief-
#       inertia mechanism. (§19a caveat #2.)
#
# Each run produces a separate dir from the original B3 run.
#
# Wall-clock: ~10 min/model × 3 models × 2 variants ≈ 60 min total.
#
# Outputs:
#   results/belief_direction_probe/<model>8b_l*_heldout/SUMMARY.md
#   results/belief_direction_probe/<model>8b_l*_agent_belief/SUMMARY.md
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N="${N:-300}"
CV_FOLDS="${CV_FOLDS:-5}"

run_one() {
    local short="$1"
    local layer="$2"
    local belief_source="$3"
    local tag="$4"
    shift 4
    local logs="$@"

    local out_dir="results/belief_direction_probe/${short}8b_l${layer}_${tag}"
    local probe_npz="results/direction_probe/${short}8b_l${layer}/raw_residuals.npz"

    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir exists"
        return 0
    fi
    if [[ ! -f "$probe_npz" ]]; then
        echo "[skip] $short: missing $probe_npz"
        return 0
    fi
    local first
    first=$(echo $logs | awk '{print $1}')
    if [[ ! -f "$first" ]]; then
        echo "[skip] $short: missing $first"
        return 0
    fi
    mkdir -p "$out_dir"

    echo
    echo "## B3 ${tag}: $short L=$layer (belief_source=$belief_source, cv_folds=$CV_FOLDS)"
    python -m experiments.belief_direction_probe \
        --enriched-log $logs \
        --layer "$layer" \
        --probe-npz "$probe_npz" \
        --max-decisions "$N" \
        --belief-source "$belief_source" \
        --cv-folds "$CV_FOLDS" \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE"
}

# ---- Followup 1: held-out R^2 against oracle_strategy_aware --------------
echo "============================================================"
echo "B3 FOLLOWUP-1: held-out R^2 (oracle_strategy_aware)"
echo "============================================================"
run_one llama 14 oracle_strategy_aware heldout \
    logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

run_one ministral 16 oracle_strategy_aware heldout \
    logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

run_one qwen 23 oracle_strategy_aware heldout \
    logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

# ---- Followup 2: agent_belief (model-stated) ----------------------------
echo
echo "============================================================"
echo "B3 FOLLOWUP-2: agent_belief (model-stated, with held-out R^2)"
echo "============================================================"
run_one llama 14 agent_belief agent_belief \
    logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

run_one ministral 16 agent_belief agent_belief \
    logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

run_one qwen 23 agent_belief agent_belief \
    logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
    logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz

echo
echo "============================================================"
echo "B3 follow-up COMPLETE."
echo
echo "Read each <model>8b_l*_heldout/SUMMARY.md and look at the"
echo "'held-out overall R^2' line vs the original 'in-sample R^2'."
echo "If held-out R^2 << in-sample R^2 (e.g. 0.10 vs 0.99), the"
echo "original 0.999 was overfit and we soften the §19a writeup."
echo
echo "Then read <model>8b_l*_agent_belief/SUMMARY.md and check"
echo "cos(w_verb, principal_belief). If still ~0 with agent_belief,"
echo "the orthogonality finding generalizes from oracle to model-"
echo "stated belief — a much stronger paper claim."
echo "============================================================"
