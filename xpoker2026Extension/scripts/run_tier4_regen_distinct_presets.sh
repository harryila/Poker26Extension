#!/usr/bin/env bash
# =============================================================================
# Tier 4 — OPTIONAL regeneration with DECORRELATED opponent RNG per preset.
# =============================================================================
#
# WHY THIS EXISTS
# ---------------
# The original Tier 4 behavioral logs seed the stochastic opponent with
# `base_seed + player_index` (= 43 for every preset). Because some presets
# have near-identical policies — notably tight_aggressive and loose_aggressive
# (both aggression=0.6; they differ only in fold_threshold 0.4 vs 0.2 and
# bluff_freq 0.08 vs 0.15, which rarely bind) — sharing the RNG stream made
# the opponent's action sequence BYTE-IDENTICAL across the two presets for
# Llama and Qwen. The hero therefore saw the same game states under both
# labels:
#
#   prompt_hash overlap (tight_aggressive ∩ loose_aggressive):
#     llama-8b  371/371  (100% identical)
#     qwen-8b   143/143  (100% identical)
#     ministral  94/108  (~87%; genuinely distinct)
#
# So Llama/Qwen effectively have 4 distinct opponent distributions, not 5.
# See AUDIT_FINDINGS.md "Tier 4 preset duplication".
#
# RECOMMENDATION
# --------------
# For the writeup, the SIMPLEST honest fix is to report tight/loose_aggressive
# as ONE "aggressive" cell for Llama/Qwen (no GPU needed — already documented).
# Regeneration is OPTIONAL: it only manufactures an RNG-level distinction
# between two policies that are near-identical BY DESIGN. Run this ONLY if you
# specifically want a genuinely-distinct 5th opponent distribution for
# Llama/Qwen with matching patching numbers.
#
# WHAT IT DOES
# ------------
# For each (preset, model): regenerate the behavioral run with a DISTINCT,
# preset-derived `--opponent-seed`, re-enrich, then re-run the Tier 4 L*
# patching cell on the fresh enriched log. New outputs are suffixed
# `_distinctseed` so the originals are preserved for comparison.
#
# Usage (GPU box):
#   cd xpoker2026Extension
#   export HF_HOME=/workspace/huggingface HF_TOKEN=...
#   bash scripts/run_tier4_regen_distinct_presets.sh
#
# Env knobs:
#   MODELS="llama-8b qwen-8b ministral-8b"   (all 3; each has a collapse — §12)
#   PRESETS="tight_aggressive loose_aggressive default informative_v2 loose_passive"
#   HANDS=50  SEED=42  TEMP=0.0
#   LLAMA_LAYER=15  QWEN_LAYER=23  MINISTRAL_LAYER=16   (L* for patching)
#   FORCE_RERUN=1
#
# NOTE (§12): all three models collapse some presets under the shared seed=43
# opponent RNG (Qwen 3 distinct, Llama 4, Ministral 4). Distinct per-preset
# seeds below give every preset an INDEPENDENT opponent sample, restoring a
# genuine 5-population opponent-invariance test.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

if [[ -f venv/bin/activate ]]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
fi
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f /root/.hf_token ]]; then
    export HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MODELS="${MODELS:-llama-8b qwen-8b ministral-8b}"
PRESETS="${PRESETS:-tight_aggressive loose_aggressive default informative_v2 loose_passive}"
HANDS="${HANDS:-50}"
SEED="${SEED:-42}"
TEMP="${TEMP:-0.0}"
TEMP_TAG="${TEMP/./}"
FORCE_RERUN="${FORCE_RERUN:-0}"
N_SOURCE="${N_SOURCE:-10}"
N_RANDOM_CONTROL="${N_RANDOM_CONTROL:-5}"
N_RANDOM_TARGET="${N_RANDOM_TARGET:-10}"
BASELINE_TOLERANCE_FRAC="${BASELINE_TOLERANCE_FRAC:-0.50}"

# Deterministic, DISTINCT opponent seed per preset (decorrelates RNG streams).
# Chosen as 43 + a fixed prime offset so no two presets share a stream and
# none collides with the canonical 43 used by the original logs.
opponent_seed_for_preset() {
    case "$1" in
        default)          echo 1043 ;;
        informative_v2)   echo 2043 ;;
        tight_aggressive) echo 3043 ;;
        loose_aggressive) echo 4043 ;;
        loose_passive)    echo 5043 ;;
        *)                echo 9043 ;;
    esac
}

layer_for_model() {
    case "$1" in
        llama-8b)     echo "${LLAMA_LAYER:-15}" ;;
        ministral-8b) echo "${MINISTRAL_LAYER:-16}" ;;
        qwen-8b)      echo "${QWEN_LAYER:-23}" ;;
        *) echo "0" ;;
    esac
}
short_for_model() {
    case "$1" in
        llama-8b) echo "llama" ;; ministral-8b) echo "ministral" ;;
        qwen-8b) echo "qwen" ;; *) echo "$1" ;;
    esac
}

count_clean_lf() {
    local enriched="$1"
    [[ ! -f "$enriched" ]] && { echo "0"; return; }
    python - "$enriched" <<'PY'
import sys
sys.path.insert(0, ".")
from experiments.causal_patching import _iter_decisions, classify_decision
n = 0
for rec in _iter_decisions(sys.argv[1]):
    if rec.get("action_metadata") and rec["action_metadata"].get("raw_response"):
        if classify_decision(rec) == "clean_legal_fold":
            n += 1
print(n)
PY
}

run_cell() {
    local preset="$1" model="$2"
    local oseed; oseed=$(opponent_seed_for_preset "$preset")
    local layer; layer=$(layer_for_model "$model")
    local short; short=$(short_for_model "$model")
    local raw="logs/opp_${preset}_${model}_t${TEMP_TAG}_s${SEED}_distinctseed.jsonl"
    local enriched="${raw%.jsonl}_enriched.jsonl"
    local out_dir="results/causal_patching/tier4_${preset}_${short}_l${layer}_distinctseed"

    echo
    echo "############################################################"
    echo "## TIER 4 REGEN (distinct seed): $model vs $preset  (L=$layer, opp_seed=$oseed)"
    echo "############################################################"

    if [[ -f "$out_dir/SUMMARY.md" ]] && [[ "$FORCE_RERUN" != "1" ]]; then
        echo "[skip] $out_dir already populated (FORCE_RERUN=1 to override)"
        return 0
    fi

    if [[ ! -f "$enriched" ]]; then
        if [[ ! -f "$raw" ]]; then
            echo "[run] generating $HANDS hands with --opponent-seed $oseed ..."
            python run_experiment.py \
                --agent hf --hf-model "$model" \
                --opponent threshold --opponent-preset "$preset" \
                --opponent-seed "$oseed" \
                --hands "$HANDS" --seed "$SEED" --temperature "$TEMP" \
                --elicit-beliefs --capture-logprobs \
                --out "$raw" -v \
                || { echo "[fail] raw run $preset/$model"; return 1; }
        fi
        echo "[enrich] build_dataset ..."
        python -m analysis.build_dataset "$raw" "$enriched" --opponent "$preset" \
            || { echo "[fail] enrich $preset/$model"; return 1; }
    fi

    local n_lf; n_lf=$(count_clean_lf "$enriched")
    if [[ "$n_lf" -lt 5 ]]; then
        echo "[skip] $preset/$model: only $n_lf clean_legal_fold targets (need >=5)"
        return 0
    fi
    local n_target="$n_lf"; [[ "$n_target" -gt 30 ]] && n_target=30

    mkdir -p "$out_dir"
    echo "[main] L=$layer patching: clean_check_or_call -> clean_legal_fold (n_target=$n_target)"
    python -m experiments.causal_patching \
        --enriched-log "$enriched" \
        --source-bucket clean_check_or_call \
        --target-bucket clean_legal_fold \
        --layers "$layer" \
        --n-source "$N_SOURCE" \
        --n-target "$n_target" \
        --n-random-control "$N_RANDOM_CONTROL" \
        --n-random-target "$N_RANDOM_TARGET" \
        --seed "$SEED" \
        --baseline-tolerance-frac "$BASELINE_TOLERANCE_FRAC" \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE" \
        || { echo "[fail] patching $preset/$model"; return 1; }
    echo "[done] $out_dir/SUMMARY.md"
}

echo "=== Tier 4 distinct-seed regeneration $(date -u +%FT%TZ) ==="
echo "MODELS=$MODELS  PRESETS=$PRESETS"
for model in $MODELS; do
    for preset in $PRESETS; do
        run_cell "$preset" "$model"
    done
done

echo
echo "=== DONE. Verify the collapse is gone (should NOT be identical now): ==="
echo "  python - <<'PY'"
echo "  import json"
echo "  def ph(p):"
echo "      s=set()"
echo "      for line in open(p):"
echo "          r=json.loads(line)"
echo "          if r.get('type') in ('hand_summary','run_config'): continue"
echo "          if 'action_metadata' in r and 'hand_id' in r: s.add(r['prompt_hash'])"
echo "      return s"
echo "  a=ph('logs/opp_tight_aggressive_llama-8b_t00_s42_distinctseed_enriched.jsonl')"
echo "  b=ph('logs/opp_loose_aggressive_llama-8b_t00_s42_distinctseed_enriched.jsonl')"
echo "  print('tight∩loose =', len(a&b), 'of', len(a|b))"
echo "  PY"
