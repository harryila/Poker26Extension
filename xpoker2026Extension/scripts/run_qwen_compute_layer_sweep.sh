#!/usr/bin/env bash
# =============================================================================
# Qwen 8B — component-level patching sweep to LOCALIZE the compute layer.
# =============================================================================
#
# WHY
# ---
# The L=22 component decomposition (results/causal_patching/qwen8b_l22_components)
# showed L=22 is RESIDUAL FLOW-THROUGH: full-residual patch = +19.4 nats /
# 76% top-1 -> CHECK, but attn = +1.6 nats (8%), mlp = -0.7 nats (-4%), and no
# single head exceeds 24% of the residual effect. So L=22's own sublayers do
# almost nothing; the verb signal ARRIVES at L=22 via the residual stream from
# earlier layers. L=23 (saturation) is also flow-through.
#
# This sweep tests L=18..21 the same way to find WHERE Qwen's attention/MLP
# actually injects the verb signal — giving Qwen a "compute layer" coordinate
# symmetric to Llama L=14 (sparse triplet) for the consolidation-gradient story.
#
# Decision rule per layer (read SUMMARY_components.md):
#   - attn ratio >= ~80% of residual  => this layer's ATTENTION computes it.
#   - a few head_NN each > 10% of residual => SPARSE HEAD STORY (cite indices).
#   - all heads small & attn small     => still flow-through; go earlier.
#
# Usage (GPU box):
#   cd xpoker2026Extension
#   export HF_HOME=/workspace/huggingface HF_TOKEN=...
#   bash scripts/run_qwen_compute_layer_sweep.sh
#
# Env knobs:
#   LAYERS="18 19 20 21"   (override to widen/narrow)
#   N_SOURCE=10  N_TARGET=30  SEED=42
#   FORCE_RERUN=1
#   PURGE=1                (rm -rf the Qwen HF cache dir when finished)
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
LAYERS="${LAYERS:-18 19 20 21}"
N_SOURCE="${N_SOURCE:-10}"
N_TARGET="${N_TARGET:-30}"
SEED="${SEED:-42}"
FORCE_RERUN="${FORCE_RERUN:-0}"
PURGE="${PURGE:-0}"

QWEN_LOGS=(
    logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz
    logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz
    logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
)

echo "=== Qwen compute-layer component sweep $(date -u +%FT%TZ) ==="
echo "LAYERS=$LAYERS  n_source=$N_SOURCE n_target=$N_TARGET"

for layer in $LAYERS; do
    out_dir="results/causal_patching/qwen8b_l${layer}_components"
    if [[ -f "$out_dir/SUMMARY_components.md" ]] && [[ "$FORCE_RERUN" != "1" ]]; then
        echo "[skip] $out_dir (FORCE_RERUN=1 to override)"
        continue
    fi
    mkdir -p "$out_dir"
    echo
    echo "######### Qwen L=$layer component patching #########"
    python -m experiments.component_patching \
        --enriched-log "${QWEN_LOGS[@]}" \
        --source-bucket clean_check_or_call \
        --target-bucket illegal_fold \
        --layer "$layer" \
        --components residual attn mlp head \
        --head-indices all \
        --n-source "$N_SOURCE" \
        --n-target "$N_TARGET" \
        --seed "$SEED" \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE" \
        || { echo "[fail] L=$layer"; continue; }
    echo "[done] $out_dir/SUMMARY_components.md"
done

if [[ "$PURGE" == "1" ]]; then
    target="${HF_HUB_CACHE%/}/models--Qwen--Qwen3-8B"
    [[ -d "$target" ]] && { echo "[purge] rm -rf $target"; rm -rf "$target"; }
fi

echo
echo "=== DONE. Compare attn/mlp/head ratios across L=18..23: ==="
echo "  for L in 18 19 20 21 22 23; do"
echo "    echo \"L=\$L:\"; sed -n '20,24p' results/causal_patching/qwen8b_l\${L}_components/SUMMARY_components.md 2>/dev/null;"
echo "  done"
echo
echo "Look for the EARLIEST layer where attn ratio jumps toward residual"
echo "(>=80%) and/or individual heads exceed 10% — that is Qwen's compute layer."
