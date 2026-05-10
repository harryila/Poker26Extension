#!/usr/bin/env bash
# =============================================================================
# Additional verb-pair causal-patching sweeps (extends C1).
# =============================================================================
#
# Why this exists
# ---------------
# C1 (already done) tested clean_BET_OR_RAISE → clean_CHECK_OR_CALL,
# answering "does L*-equivalent layer mediate RAISE→CHECK?" and finding
# the BET_RAISE flip is at L*+1 to L*+2.
#
# This script extends to the OTHER verb pairs we haven't tested:
#   1. clean_BET_OR_RAISE  → clean_legal_FOLD     (does RAISE patch flip
#       FOLD targets to BET_RAISE? i.e. is the bet/raise circuit also
#       a "general decision circuit" or specifically a "BET-vs-CHECK"
#       circuit?)
#   2. clean_legal_FOLD     → clean_BET_OR_RAISE  (reverse: can a FOLD
#       source flip a BET target to FOLD?)
#
# Combined with the existing CHECK↔FOLD and RAISE→CHECK results, these
# fill in the remaining cells of the verb-direction matrix:
#
#               source ↓
#   target →  | CHECK   | FOLD    | RAISE
#   ----------+---------+---------+--------
#   CHECK     | (self)  | reverse | C1 ✅
#   FOLD      | forward | (self)  | NEW (this script, cell 1)
#   RAISE     | ?       | NEW     | (self)
#                  cell 2
#
# Each combination is a separate experiment. We pick the most informative
# additions (FOLD-target with RAISE source, RAISE-target with FOLD source);
# the "CHECK-target-from-RAISE-source" is C1 which we already have.
#
# Wall-clock: ~50-60 min per cell × 3 models × 2 cells = ~5-6 h total.
# Auto-skip per output dir keeps re-runs cheap.
#
# Outputs
# -------
#   results/causal_patching/<model>8b_verbpair_raise_to_fold/   (cell 1)
#   results/causal_patching/<model>8b_verbpair_fold_to_raise/   (cell 2)
#
# Env knobs
# ---------
#   MODELS=llama,ministral,qwen   subset / order
#   N_SOURCE                      default 10
#   N_TARGET                      default 30 (auto-capped to available)
#   N_RANDOM_CONTROL              default 5
#   SEED                          default 42 (RNG)
#   DEVICE / DTYPE                default cuda / bfloat16
# =============================================================================

set -uo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
SEED="${SEED:-42}"
MODELS_ENV="${MODELS:-llama,ministral,qwen}"

# Per-model layer set: tight sweep around L* (matching C1's choices).
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
    local source_bucket="$4"
    local target_bucket="$5"
    local out_dir="$6"

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

    local n_src
    n_src=$(count_in_bucket "$source_bucket" $enriched_logs)
    local n_tgt
    n_tgt=$(count_in_bucket "$target_bucket" $enriched_logs)
    if [[ "$n_src" -lt 3 ]] || [[ "$n_tgt" -lt 3 ]]; then
        echo "[skip] $short ($source_bucket → $target_bucket): "
        echo "  source n=$n_src, target n=$n_tgt (need >= 3 of each)"
        return 0
    fi

    mkdir -p "$out_dir"
    local n_layers
    n_layers=$(echo $layers | wc -w | tr -d ' ')

    echo
    echo "############################################################"
    echo "## VERB-PAIR: $short  $source_bucket → $target_bucket"
    echo "##   source n_avail=$n_src  target n_avail=$n_tgt"
    echo "##   layers ($n_layers): $layers"
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
    echo "[main] $source_bucket → $target_bucket patching ..."
    python -m experiments.causal_patching \
        --enriched-log $enriched_logs \
        --source-bucket "$source_bucket" \
        --target-bucket "$target_bucket" \
        --layers $layers \
        --n-source "$N_SOURCE" \
        --n-target "$N_TARGET" \
        --n-random-control "$N_RANDOM_CONTROL" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" \
        --dtype "$DTYPE"

    echo
    echo "[done] $short ($source_bucket → $target_bucket) wrote $out_dir/SUMMARY.md"
}

IFS=',' read -r -a MODELS_ARR <<< "$MODELS_ENV"
for short in "${MODELS_ARR[@]}"; do
    case "$short" in
        llama)     LV=LLAMA_LOGS;     LY=LLAMA_LAYERS ;;
        ministral) LV=MINISTRAL_LOGS; LY=MINISTRAL_LAYERS ;;
        qwen)     LV=QWEN_LOGS;      LY=QWEN_LAYERS ;;
        *)        echo "WARNING: unknown model '$short' — skipping"; continue ;;
    esac

    # Cell 1: BET_OR_RAISE → legal_FOLD
    run_one "$short" "$LV" "$LY" \
        clean_bet_or_raise clean_legal_fold \
        "results/causal_patching/${short}8b_verbpair_raise_to_fold"

    # Cell 2: legal_FOLD → BET_OR_RAISE
    run_one "$short" "$LV" "$LY" \
        clean_legal_fold clean_bet_or_raise \
        "results/causal_patching/${short}8b_verbpair_fold_to_raise"
done

echo
echo "============================================================"
echo "Verb-pair sweeps COMPLETE."
echo
echo "Read each SUMMARY.md and check the per-layer table at L*."
echo
echo "Cell 1 (RAISE → FOLD): the question is 'does a RAISE source flip a"
echo "  legal-FOLD target's verb to BET_RAISE at L*?' Look at the"
echo "  top-1 → BET_RAISE-family column at L*."
echo
echo "Cell 2 (FOLD → RAISE): 'does a FOLD source flip a BET_RAISE target's"
echo "  verb to FOLD at L*?' Look at the top-1 → FOLD-family column at L*."
echo
echo "Combined with C1 (BET_RAISE → CHECK) and the original forward+reverse"
echo "(CHECK ↔ FOLD), we now have all six pairwise (source, target)"
echo "combinations among the three legal-action families. If L* mediates"
echo "all of them, it is comprehensively a 'general decision circuit'."
echo "============================================================"
