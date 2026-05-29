#!/usr/bin/env bash
# =============================================================================
# run_followups_gpu.sh — all GPU follow-ups in DEPENDENCY ORDER, one after another.
#
# CPU analyses are already done & committed (CONFOUND_PROJECTION / SUFFICIENCY_CI /
# REGEN_DRIFT / CLAIMS_AND_IDENTIFICATION). This runs Phases B (rejection-proofing),
# C (novelty), D (depth) from RUNBOOK_followups.md.
#
# Dependency order enforced here:
#   B1 recapture (3 models) ──► C1 encode_vs_decode (reads tagged npz, SAVES steering dir)
#                                      │
#   B2 bet-matched patch (3 models)    ├──► C2 posterior steering  (needs steering dir)
#   B3 Ministral 3-seed                └──► C3 steer-at-scale       (needs steering dir)
#   B4 Qwen L19 same-depth control          C3 baseline + ablate-at-scale (no dep)
#   D1 Qwen band SVD (no dep)
#
# NOT `set -e`: independent steps continue if one fails; every DEPENDENT step is
# guarded by a file-existence check so nothing runs on missing inputs.
#
# Usage (GPU box):
#   cd xpoker2026Extension && git pull
#   export HF_HOME=/workspace/huggingface HF_TOKEN=...
#   bash scripts/run_followups_gpu.sh            # everything
#   PHASES="B" bash scripts/run_followups_gpu.sh # just rejection-proofing
# Overridable: DEVICE DTYPE SEEDS HANDS PHASES
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

: "${DEVICE:=cuda}"
: "${DTYPE:=bfloat16}"
: "${SEEDS:=42 123 456}"
: "${HANDS:=300}"
: "${PHASES:=B C D}"          # which phases to run

LOG="results/followups_run_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p results
echo "=== followups run  start=$(date -u)  device=$DEVICE dtype=$DTYPE seeds='$SEEDS' phases='$PHASES' ===" | tee "$LOG"

want () { [[ " $PHASES " == *" $1 "* ]]; }     # is phase $1 requested?

logs_for () {  # model short (llama|ministral|qwen) -> space-separated enriched logs
  local m=$1 out=""
  for s in $SEEDS; do out="$out logs/cot_${m}8b_t0_s${s}_informative_v2_enriched.jsonl.gz"; done
  echo "$out"
}
suff_layers () { case $1 in llama) echo "14 15";; ministral) echo 16;; qwen) echo 23;; esac; }
probe_layer () { case $1 in llama) echo 14;; ministral) echo 16;; qwen) echo 23;; esac; }

step () {  # step NAME cmd...   — log, time, continue-on-fail, record status
  local name=$1; shift
  echo "" | tee -a "$LOG"
  echo "----- [$name] start $(date -u +%H:%M:%S) -----" | tee -a "$LOG"
  echo "+ $*" | tee -a "$LOG"
  local t0=$SECONDS
  if "$@" >>"$LOG" 2>&1; then
    echo "[$name] OK (${SECONDS}-${t0}s)" | tee -a "$LOG"
  else
    echo "[$name] !! FAILED rc=$? — continuing" | tee -a "$LOG"
  fi
}

STEER=results/direction_probe/qwen8b_l23/steer_trash_direction.npz

# =============================================================================
# PHASE B1 — bet-matched recapture (+CPU probe). All 3 models. Feeds C1.
# =============================================================================
if want B; then
  for m in llama ministral qwen; do
    L=$(probe_layer $m)
    TAG=results/direction_probe/${m}8b_l${L}/raw_residuals_tagged.npz
    step "B1.recapture.$m.l$L" python -m experiments.bet_matched_recapture \
      --enriched-log $(logs_for $m) --layer "$L" --device "$DEVICE" --dtype "$DTYPE" --out "$TAG"
    if [[ -f "$TAG" ]]; then
      step "B1.probe.$m.l$L" python -m experiments.bet_matched_probe \
        --tagged "$TAG" --out results/direction_probe_baselines/BET_MATCHED_${m}_l${L}.md
    else
      echo "[B1.probe.$m] SKIP — recapture produced no $TAG" | tee -a "$LOG"
    fi
  done
fi

# =============================================================================
# PHASE C1 — encode-vs-decode. Reuses B1 recaptures (oracle+belief tagged).
#            Qwen run SAVES the steering direction used by C2 / C3-steer.
# =============================================================================
if want C; then
  QTAG=results/direction_probe/qwen8b_l23/raw_residuals_tagged.npz
  if [[ -f "$QTAG" ]]; then
    step "C1.encode_vs_decode.qwen" python -m experiments.encode_vs_decode \
      --tagged "$QTAG" --out results/direction_probe_baselines/ENCODE_VS_DECODE_qwen_l23.md \
      --save-direction "$STEER"
  else
    echo "[C1.qwen] SKIP — missing $QTAG (run Phase B first)" | tee -a "$LOG"
  fi
  for m in llama ministral; do
    L=$(probe_layer $m); T=results/direction_probe/${m}8b_l${L}/raw_residuals_tagged.npz
    [[ -f "$T" ]] && step "C1.encode_vs_decode.$m" python -m experiments.encode_vs_decode \
      --tagged "$T" --out results/direction_probe_baselines/ENCODE_VS_DECODE_${m}_l${L}.md
  done
fi

# =============================================================================
# PHASE B2 — bet-matched patching (facing & no-bet regimes). All 3 models.
# =============================================================================
if want B; then
  for m in llama ministral qwen; do
    for L in $(suff_layers $m); do
      step "B2.facing.$m.l$L" python -m experiments.causal_patching \
        --enriched-log $(logs_for $m) \
        --source-bucket clean_check_or_call --source-bet-filter facing \
        --target-bucket clean_legal_fold   --target-bet-filter facing \
        --layers "$L" --n-source 10 --n-target 30 --seed 42 \
        --device "$DEVICE" --dtype "$DTYPE" \
        --out-dir results/causal_patching/${m}8b_betmatched_facing_l${L}
      step "B2.nobet.$m.l$L" python -m experiments.causal_patching \
        --enriched-log $(logs_for $m) \
        --source-bucket clean_check_or_call --source-bet-filter nobet \
        --target-bucket illegal_fold        --target-bet-filter nobet \
        --layers "$L" --n-source 10 --n-target 30 --seed 42 \
        --device "$DEVICE" --dtype "$DTYPE" \
        --out-dir results/causal_patching/${m}8b_betmatched_nobet_l${L}
    done
  done
fi

# =============================================================================
# PHASE B3 — Ministral 3-seed sufficiency + necessity + significance.
# =============================================================================
if want B; then
  step "B3.suff.ministral" python -m experiments.causal_patching \
    --enriched-log $(logs_for ministral) \
    --source-bucket clean_check_or_call --target-bucket illegal_fold \
    --layers 16 --n-source 10 --n-target 30 --seed 42 \
    --device "$DEVICE" --dtype "$DTYPE" \
    --out-dir results/causal_patching/ministral8b_t0_pooled_layer_sweep
  step "B3.necc.ministral" python -m experiments.inference_head_ablation \
    --enriched-log $(logs_for ministral) \
    --layer 16 --pipeline recon --filter-recorded-bucket illegal_fold \
    --conditions baseline triplet control extended \
    --device "$DEVICE" --dtype "$DTYPE" \
    --out-dir results/inference_head_ablation/ministral8b_l16_recon_illegal_fold_3seed
  step "B3.sig.ministral" python -m experiments.necessity_significance \
    --glob 'results/inference_head_ablation/ministral8b_l16_recon_illegal_fold_3seed' \
    --out results/inference_head_ablation/SIGNIFICANCE_ministral_l16_3seed.md
  step "B3.drift.ministral" python -m experiments.regen_drift_audit \
    --glob 'results/inference_head_ablation/ministral8b_l16_recon_illegal_fold_3seed' \
    --out results/inference_head_ablation/REGEN_DRIFT_ministral_3seed.md
fi

# =============================================================================
# PHASE B4 — Qwen L19 same-depth random-head control + significance.
# =============================================================================
if want B; then
  step "B4.qwen.l19ctrl" python -m experiments.inference_head_ablation \
    --enriched-log $(logs_for qwen) \
    --layer 19 --pipeline recon --filter-recorded-bucket clean_legal_fold \
    --conditions baseline \
    --head-sets 'top5:31 3 21 1 0' 'rand5a:rand:5:101' 'rand5b:rand:5:202' 'rand5c:rand:5:303' \
    --device "$DEVICE" --dtype "$DTYPE" \
    --out-dir results/inference_head_ablation/qwen8b_l19_samedepth_control
  step "B4.sig" python -m experiments.necessity_significance \
    --glob 'results/inference_head_ablation/qwen8b_l19_samedepth_control' \
    --within-cell-control rand5a \
    --out results/inference_head_ablation/SIGNIFICANCE_qwen_l19_samedepth.md
fi

# =============================================================================
# PHASE C2 — posterior steering (needs C1 steering dir).
# =============================================================================
if want C; then
  if [[ -f "$STEER" ]]; then
    step "C2.steer.qwen" python -m experiments.posterior_steering \
      --enriched-log $(logs_for qwen) --layer 23 --direction "$STEER" \
      --alphas 0 2 4 8 --target-bucket clean_legal_fold --n-decisions 60 \
      --device "$DEVICE" --dtype "$DTYPE" \
      --out-dir results/posterior_steering/qwen8b_l23
  else
    echo "[C2] SKIP — missing $STEER (C1 did not produce it)" | tee -a "$LOG"
  fi
fi

# =============================================================================
# PHASE C3 — behavior at scale: baseline / ablate / steer (env-var hook injection).
# =============================================================================
if want C; then
  step "C3.baseline" python run_experiment.py --agent hf --hf-model qwen-8b \
    --opponent threshold --opponent-preset informative_v2 --hands "$HANDS" --seed 42 \
    --elicit-beliefs --out logs/scale_qwen_baseline.jsonl -v
  step "C3.ablate" env CIRCUIT_ABLATE_LAYER=19 CIRCUIT_ABLATE_HEADS="31 3 21 1 0" \
    python run_experiment.py --agent hf --hf-model qwen-8b \
    --opponent threshold --opponent-preset informative_v2 --hands "$HANDS" --seed 42 \
    --elicit-beliefs --out logs/scale_qwen_ablate_l19.jsonl -v
  if [[ -f "$STEER" ]]; then
    step "C3.steer" env CIRCUIT_STEER_LAYER=23 CIRCUIT_STEER_NPZ="$STEER" CIRCUIT_STEER_ALPHA=4 \
      python run_experiment.py --agent hf --hf-model qwen-8b \
      --opponent threshold --opponent-preset informative_v2 --hands "$HANDS" --seed 42 \
      --elicit-beliefs --out logs/scale_qwen_steer.jsonl -v
  else
    echo "[C3.steer] SKIP — missing $STEER" | tee -a "$LOG"
  fi
fi

# =============================================================================
# PHASE D1 — Qwen distributed-band SVD.
# =============================================================================
if want D; then
  step "D1.svd.qwen" python -m experiments.qwen_band_svd \
    --enriched-log $(logs_for qwen) --layers 18 19 20 23 --bucket clean_check_or_call --n 150 \
    --direction results/direction_probe/qwen8b_l23/raw_residuals.npz \
    --device "$DEVICE" --dtype "$DTYPE" --out results/causal_patching/qwen_band_svd.md
fi

echo "" | tee -a "$LOG"
echo "=== followups run DONE $(date -u) ===" | tee -a "$LOG"
echo "Status lines:" | tee -a "$LOG"
grep -E "\] (OK|!! FAILED|SKIP)" "$LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Pull back: results/  logs/scale_qwen_*.jsonl   (then post-process with the assistant)" | tee -a "$LOG"
