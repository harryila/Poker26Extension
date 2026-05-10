#!/usr/bin/env bash
# =============================================================================
# Cross-temperature Ministral component decomposition (t=0.2).
# =============================================================================
#
# Why this exists
# ---------------
# All current component decompositions are at t=0.0. Are the same dominant
# heads (Ministral L=16 head_22 + tail) active at t=0.2? Tests robustness
# of the head-circuit finding to decoding temperature. Ministral's existing
# t=0.2 enriched log (cot_ministral8b_t02_s42_*) is on disk.
#
# Cells
# -----
# 1. Ministral t=0.2 s42 component sweep at L=16
# 2. (optional, if logs exist) Llama t=0.2 component sweep at L=14 — but we
#    don't have cot_llama8b_t02_*_logitlens enriched logs. So we skip.
#
# Wall-clock: ~50 min on H100.
#
# Outputs
# -------
#   results/causal_patching/ministral8b_t02_s42_l16_components/
#
# Env knobs
# ---------
#   LAYER          default 16
#   N_SOURCE       default 10
#   N_TARGET       default 30
#   SEED           default 42 (RNG)
#   DEVICE / DTYPE default cuda / bfloat16
# =============================================================================

set -uo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
SEED="${SEED:-42}"
LAYER="${LAYER:-16}"

ENRICHED="logs/cot_ministral8b_t02_s42_informative_v2_logitlens_enriched.jsonl.gz"
OUT_DIR="results/causal_patching/ministral8b_t02_s42_l${LAYER}_components"

if [[ -d "$OUT_DIR" ]] && [[ -f "$OUT_DIR/SUMMARY_components.md" ]]; then
    echo "[skip] $OUT_DIR already has SUMMARY_components.md"
    exit 0
fi

if [[ ! -f "$ENRICHED" ]]; then
    echo "[skip] missing enriched log: $ENRICHED"
    echo "(Ministral t=0.2 logit-lens log was produced in phase F; if it"
    echo " isn't on this GPU box, this cell is just skipped.)"
    exit 0
fi

# Count illegal_FOLDs.
n_avail=$(python - <<PY
import sys
sys.path.insert(0, ".")
from experiments.causal_patching import _iter_decisions, classify_decision
n = 0
for rec in _iter_decisions("$ENRICHED"):
    if rec.get("action_metadata") and rec["action_metadata"].get("raw_response"):
        if classify_decision(rec) == "illegal_fold":
            n += 1
print(n)
PY
)
if [[ "$n_avail" -lt 3 ]]; then
    echo "[skip] only $n_avail illegal_FOLDs at t=0.2 — too few"
    exit 0
fi
n_target="$N_TARGET"
if [[ "$n_avail" -lt "$N_TARGET" ]]; then
    n_target="$n_avail"
fi

mkdir -p "$OUT_DIR"

echo
echo "############################################################"
echo "## CROSS-TEMP Ministral t=0.2 s42 components at L=$LAYER"
echo "##   illegal_FOLD n_avail=$n_avail  using=$n_target"
echo "##   out: $OUT_DIR"
echo "############################################################"

python -m experiments.component_patching \
    --enriched-log "$ENRICHED" \
    --source-bucket clean_check_or_call \
    --target-bucket illegal_fold \
    --layer "$LAYER" \
    --components residual attn mlp head \
    --head-indices all \
    --n-source "$N_SOURCE" \
    --n-target "$n_target" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR" \
    --device "$DEVICE" \
    --dtype "$DTYPE"

echo
echo "[done] wrote $OUT_DIR/SUMMARY_components.md"
echo
echo "Compare top-3 heads against pooled t=0 result:"
echo "  pooled t=0:   results/causal_patching/ministral8b_l${LAYER}_components/SUMMARY_components.md"
echo "  s42 t=0.2:    $OUT_DIR/SUMMARY_components.md"
echo
echo "Pass: head_22 still dominates at t=0.2; same long-tail set roughly."
echo "Robust-across-temperature head story confirmed."
