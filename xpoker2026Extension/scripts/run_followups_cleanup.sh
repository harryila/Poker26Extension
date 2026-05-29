#!/usr/bin/env bash
# =============================================================================
# run_followups_cleanup.sh — finish the follow-ups that the single-pass
# orchestrator (run_followups_gpu.sh) could not complete because the ~20 GB
# /workspace HF-cache quota only holds ONE 8B model at a time, plus the C2
# posterior-steering re-run after the alpha-key bug fix.
#
# Order (one model resident at a time; purge between):
#   1. C2 re-run            (Qwen — still cached from the orchestrator)
#   2. purge Qwen
#   3. Llama  : B1 recapture+probe, C1 encode_vs_decode, B2 facing/nobet (L14,L15)
#   4. purge Llama
#   5. Ministral: B1 recapture+probe, C1 encode_vs_decode, B2 facing/nobet (L16),
#                 B3 suff + necc + sig + drift
#
# Mirrors the exact commands/paths in run_followups_gpu.sh. Continue-on-fail.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

: "${DEVICE:=cuda}"
: "${DTYPE:=bfloat16}"
: "${SEEDS:=42 123 456}"

export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f /root/.hf_token ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

LOG="results/followups_cleanup_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p results
echo "=== cleanup start=$(date -u) device=$DEVICE dtype=$DTYPE seeds='$SEEDS' ===" | tee "$LOG"

logs_for () { local m=$1 out=""; for s in $SEEDS; do out="$out logs/cot_${m}8b_t0_s${s}_informative_v2_enriched.jsonl.gz"; done; echo "$out"; }

hub_dir () { case $1 in
  llama)     echo "models--meta-llama--Llama-3.1-8B-Instruct";;
  ministral) echo "models--mistralai--Ministral-8B-Instruct-2410";;
  qwen)      echo "models--Qwen--Qwen3-8B";; esac; }

purge () {  # purge HF weights for a model short-name to free the quota
  local d="${HF_HUB_CACHE%/}/$(hub_dir "$1")"
  [[ -d "$d" ]] && { echo "  [purge] rm -rf $d" | tee -a "$LOG"; rm -rf "$d"; }
  local xet="$(dirname "$HF_HUB_CACHE")/xet"; [[ -d "$xet" ]] && rm -rf "${xet:?}"/* 2>/dev/null
  du -sh "$HF_HOME" 2>/dev/null | tee -a "$LOG"
}

step () { local name=$1; shift
  echo "" | tee -a "$LOG"; echo "----- [$name] start $(date -u +%H:%M:%S) -----" | tee -a "$LOG"
  echo "+ $*" | tee -a "$LOG"; local t0=$SECONDS
  if "$@" >>"$LOG" 2>&1; then echo "[$name] OK ($((SECONDS-t0))s)" | tee -a "$LOG"
  else echo "[$name] !! FAILED rc=$? — continuing" | tee -a "$LOG"; fi
}

STEER=results/direction_probe/qwen8b_l23/steer_trash_direction.npz

# --- 1. C2 re-run (Qwen still cached) ----------------------------------------
if [[ -f "$STEER" ]]; then
  step "C2.steer.qwen" python -m experiments.posterior_steering \
    --enriched-log $(logs_for qwen) --layer 23 --direction "$STEER" \
    --alphas 0 2 4 8 --target-bucket clean_legal_fold --n-decisions 60 \
    --device "$DEVICE" --dtype "$DTYPE" --out-dir results/posterior_steering/qwen8b_l23
else
  echo "[C2] SKIP — missing $STEER" | tee -a "$LOG"
fi
purge qwen

# --- 3. Llama pass -----------------------------------------------------------
LTAG=results/direction_probe/llama8b_l14/raw_residuals_tagged.npz
step "B1.recapture.llama.l14" python -m experiments.bet_matched_recapture \
  --enriched-log $(logs_for llama) --layer 14 --device "$DEVICE" --dtype "$DTYPE" --out "$LTAG"
[[ -f "$LTAG" ]] && step "B1.probe.llama.l14" python -m experiments.bet_matched_probe \
  --tagged "$LTAG" --out results/direction_probe_baselines/BET_MATCHED_llama_l14.md
[[ -f "$LTAG" ]] && step "C1.encode_vs_decode.llama" python -m experiments.encode_vs_decode \
  --tagged "$LTAG" --out results/direction_probe_baselines/ENCODE_VS_DECODE_llama_l14.md
for L in 14 15; do
  step "B2.facing.llama.l$L" python -m experiments.causal_patching \
    --enriched-log $(logs_for llama) \
    --source-bucket clean_check_or_call --source-bet-filter facing \
    --target-bucket clean_legal_fold --target-bet-filter facing \
    --layers "$L" --n-source 10 --n-target 30 --seed 42 --device "$DEVICE" --dtype "$DTYPE" \
    --out-dir results/causal_patching/llama8b_betmatched_facing_l${L}
  step "B2.nobet.llama.l$L" python -m experiments.causal_patching \
    --enriched-log $(logs_for llama) \
    --source-bucket clean_check_or_call --source-bet-filter nobet \
    --target-bucket illegal_fold --target-bet-filter nobet \
    --layers "$L" --n-source 10 --n-target 30 --seed 42 --device "$DEVICE" --dtype "$DTYPE" \
    --out-dir results/causal_patching/llama8b_betmatched_nobet_l${L}
done
purge llama

# --- 5. Ministral pass -------------------------------------------------------
MTAG=results/direction_probe/ministral8b_l16/raw_residuals_tagged.npz
step "B1.recapture.ministral.l16" python -m experiments.bet_matched_recapture \
  --enriched-log $(logs_for ministral) --layer 16 --device "$DEVICE" --dtype "$DTYPE" --out "$MTAG"
[[ -f "$MTAG" ]] && step "B1.probe.ministral.l16" python -m experiments.bet_matched_probe \
  --tagged "$MTAG" --out results/direction_probe_baselines/BET_MATCHED_ministral_l16.md
[[ -f "$MTAG" ]] && step "C1.encode_vs_decode.ministral" python -m experiments.encode_vs_decode \
  --tagged "$MTAG" --out results/direction_probe_baselines/ENCODE_VS_DECODE_ministral_l16.md
step "B2.facing.ministral.l16" python -m experiments.causal_patching \
  --enriched-log $(logs_for ministral) \
  --source-bucket clean_check_or_call --source-bet-filter facing \
  --target-bucket clean_legal_fold --target-bet-filter facing \
  --layers 16 --n-source 10 --n-target 30 --seed 42 --device "$DEVICE" --dtype "$DTYPE" \
  --out-dir results/causal_patching/ministral8b_betmatched_facing_l16
step "B2.nobet.ministral.l16" python -m experiments.causal_patching \
  --enriched-log $(logs_for ministral) \
  --source-bucket clean_check_or_call --source-bet-filter nobet \
  --target-bucket illegal_fold --target-bet-filter nobet \
  --layers 16 --n-source 10 --n-target 30 --seed 42 --device "$DEVICE" --dtype "$DTYPE" \
  --out-dir results/causal_patching/ministral8b_betmatched_nobet_l16
step "B3.suff.ministral" python -m experiments.causal_patching \
  --enriched-log $(logs_for ministral) \
  --source-bucket clean_check_or_call --target-bucket illegal_fold \
  --layers 16 --n-source 10 --n-target 30 --seed 42 --device "$DEVICE" --dtype "$DTYPE" \
  --out-dir results/causal_patching/ministral8b_t0_pooled_layer_sweep
step "B3.necc.ministral" python -m experiments.inference_head_ablation \
  --enriched-log $(logs_for ministral) \
  --layer 16 --pipeline recon --filter-recorded-bucket illegal_fold \
  --conditions baseline triplet control extended --device "$DEVICE" --dtype "$DTYPE" \
  --out-dir results/inference_head_ablation/ministral8b_l16_recon_illegal_fold_3seed
step "B3.sig.ministral" python -m experiments.necessity_significance \
  --glob 'results/inference_head_ablation/ministral8b_l16_recon_illegal_fold_3seed' \
  --out results/inference_head_ablation/SIGNIFICANCE_ministral_l16_3seed.md
step "B3.drift.ministral" python -m experiments.regen_drift_audit \
  --glob 'results/inference_head_ablation/ministral8b_l16_recon_illegal_fold_3seed' \
  --out results/inference_head_ablation/REGEN_DRIFT_ministral_3seed.md

echo "" | tee -a "$LOG"
echo "=== cleanup DONE $(date -u) ===" | tee -a "$LOG"
grep -E "\] (OK|!! FAILED|SKIP)" "$LOG" | tee -a "$LOG"
