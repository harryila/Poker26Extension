#!/usr/bin/env bash
# =============================================================================
# Tier 4 — opponent robustness (8B behavioral runs).
# =============================================================================
# Vary opponent preset (informative_v2 / tight_aggressive / loose_aggressive /
# loose_passive / default) holding model + seed fixed. Tests whether the
# JS-to-StrategyAware gap and the L*=14 verb-decision behavior are
# opponent-specific or stable across opponents.
#
# This is the 8B variant of the EXPERIMENTS.md Tier 4 plan (the original
# was specified for 70B which we're skipping).
#
# Cells: 5 presets × 3 models = 15 cells × 50 hands each.
# Wall-clock per cell: ~10-20 min on 8B with H100. Total: ~3-5 hours.
#
# This script ONLY runs the BEHAVIORAL cells. Patching at L* on the
# resulting non-informative_v2 enriched logs is queued as a separate
# follow-up after this completes — it requires illegal_FOLD targets that
# may not be present in all opponent×preset combinations.
#
# Outputs
# -------
#   logs/opp_<preset>_<model>_t0_s42.jsonl              — raw run
#   logs/opp_<preset>_<model>_t0_s42_enriched.jsonl     — after build_dataset
#   results/tier4_opponent/<preset>_<model>/SUMMARY.md  — per-cell behavioral metrics
#
# Env knobs
# ---------
#   PRESETS="informative_v2 tight_aggressive loose_aggressive loose_passive default"
#   MODELS="llama-8b qwen-8b ministral-8b"
#   SEED=42
#   HANDS=50
#   TEMP=0.0
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")/.."

PRESETS="${PRESETS:-informative_v2 tight_aggressive loose_aggressive loose_passive default}"
MODELS="${MODELS:-llama-8b qwen-8b ministral-8b}"
SEED="${SEED:-42}"
HANDS="${HANDS:-50}"
TEMP="${TEMP:-0.0}"
TEMP_TAG="${TEMP/./}"

# IMPORTANT: there are TWO run_experiment.py files in this workspace —
# the older one at the repo root and the newer (correct) one inside
# xpoker2026Extension/. The newer one has the model-name registry that maps
# short names like 'llama-8b' → 'meta-llama/Llama-3.1-8B-Instruct'.
# We use the EXT-local one, matching the convention in run_tier1a_small.sh.
RUN_EXP="run_experiment.py"
if [[ ! -f "$RUN_EXP" ]]; then
    echo "[abort] run_experiment.py not found in $(pwd) (the EXT-local one)"
    exit 2
fi

run_cell() {
    local preset="$1"
    local model="$2"
    local raw="logs/opp_${preset}_${model}_t${TEMP_TAG}_s${SEED}.jsonl"
    local enriched="${raw%.jsonl}_enriched.jsonl"
    local out_dir="results/tier4_opponent/${preset}_${model}"

    if [[ -f "$enriched" ]] && [[ -d "$out_dir" ]]; then
        echo "[skip] $preset / $model already done"
        return 0
    fi

    mkdir -p "$out_dir"
    echo
    echo "############################################################"
    echo "## TIER 4 cell: $model vs $preset"
    echo "##   raw: $raw"
    echo "##   out: $out_dir"
    echo "############################################################"

    if [[ ! -f "$raw" ]]; then
        echo "[run] generating $HANDS hands ..."
        python run_experiment.py \
            --agent hf --hf-model "$model" \
            --opponent threshold --opponent-preset "$preset" \
            --hands "$HANDS" --seed "$SEED" --temperature "$TEMP" \
            --elicit-beliefs --capture-logprobs \
            --out "$raw" -v \
            || { echo "[fail] $preset / $model raw run (see error above)"; return 1; }
    else
        echo "[skip-raw] $raw already exists"
    fi

    if [[ ! -f "$enriched" ]]; then
        echo "[enrich] building enriched dataset ..."
        python -m analysis.build_dataset \
            --input "$raw" \
            --output "$enriched" \
            --opponent-preset "$preset" \
            || { echo "[fail] enrich"; return 1; }
    else
        echo "[skip-enrich] $enriched already exists"
    fi

    # Per-cell summary: action distribution + parsed-belief rate.
    python - <<PY > "$out_dir/SUMMARY.md"
import json, gzip, statistics
import os
path = "$enriched"
opener = gzip.open if path.endswith(".gz") else open
total = 0
parse_ok = 0
actions = {}
with opener(path, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: rec = json.loads(line)
        except: continue
        if rec.get("type") == "run_config": continue
        am = rec.get("action_metadata") or {}
        if not am.get("raw_response"): continue
        total += 1
        if am.get("parse_success"): parse_ok += 1
        a = am.get("action_chosen") or "?"
        actions[a] = actions.get(a, 0) + 1
print("# Tier 4 cell: $model vs $preset")
print()
print(f"- Raw: \`$raw\`")
print(f"- Enriched: \`$enriched\`")
print(f"- Total decisions: {total}")
print(f"- Parse-OK: {parse_ok} ({100*parse_ok/max(total,1):.1f}%)")
print()
print("## Action distribution")
print()
print("| Action | n | % |")
print("|---|---:|---:|")
for a, n in sorted(actions.items(), key=lambda x: -x[1]):
    print(f"| {a} | {n} | {100*n/max(total,1):.1f}% |")
PY
    echo "[done] $preset / $model"
}

for preset in $PRESETS; do
    for model in $MODELS; do
        run_cell "$preset" "$model"
    done
done

echo
echo "============================================================"
echo "Tier 4 (8B) opponent robustness BEHAVIORAL runs COMPLETE."
echo
echo "Per-cell summaries: results/tier4_opponent/*_*/SUMMARY.md"
echo
echo "Next step (separate follow-up): re-run the L* patching protocol on"
echo "each preset's enriched log to test whether the L*=14 verb-decision"
echo "circuit is opponent-stable. Requires checking illegal_FOLD counts"
echo "per cell first (some presets may have very few)."
echo "============================================================"
