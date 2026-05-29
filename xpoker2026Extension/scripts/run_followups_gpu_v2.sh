#!/usr/bin/env bash
# =============================================================================
# run_followups_gpu_v2.sh — second GPU batch: the controls + reworks the first
# batch (run_followups_gpu.sh) showed were needed. Dependency-ordered, continue-
# on-fail, guarded. CPU steps run inline.
#
#   C1-CTRL  early-layer control for encode-vs-decode (the make-or-break for the
#            "knows-but-mis-states" headline): recapture at L2, decode oracle,
#            compare L2 vs L*. If late >> early -> COMPUTED; if early ~ late ->
#            input-presence (downgrade the claim).
#   C2-DEC   steering DECISION readout, reworked: at the L19 COMPUTE layer (not
#            just L23 commit), BOTH directions (±alpha), on WRONG-fold targets
#            (illegal_fold) and legal folds, vs the built-in random control.
#   C2-BEL   steering BELIEF readout: does steering REPAIR JS(belief,oracle)?
#            (intervention converse of C1). Reuses HFAgent.belief under the hook.
#   C3       behavior-at-scale RERUN with --cot (the v1 run was non-CoT ->
#            degenerate constant-loss agent) + safe steering (last_only, small a).
#
# Usage:
#   cd xpoker2026Extension && git pull
#   export HF_HOME=/workspace/huggingface HF_TOKEN=...
#   bash scripts/run_followups_gpu_v2.sh                # all
#   PHASES="C1" bash scripts/run_followups_gpu_v2.sh    # just the C1 control
# Overridable: DEVICE DTYPE SEEDS HANDS PHASES
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

: "${DEVICE:=cuda}"
: "${DTYPE:=bfloat16}"
: "${SEEDS:=42 123 456}"
: "${HANDS:=300}"
: "${PHASES:=C1 C2 C3}"

LOG="results/followups_v2_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p results
echo "=== followups v2  start=$(date -u)  device=$DEVICE dtype=$DTYPE phases='$PHASES' ===" | tee "$LOG"

want () { [[ " $PHASES " == *" $1 "* ]]; }
logs_for () { local m=$1 out=""; for s in $SEEDS; do out="$out logs/cot_${m}8b_t0_s${s}_informative_v2_enriched.jsonl.gz"; done; echo "$out"; }
probe_layer () { case $1 in llama) echo 14;; ministral) echo 16;; qwen) echo 23;; esac; }
step () {
  local name=$1; shift
  echo "" | tee -a "$LOG"; echo "----- [$name] $(date -u +%H:%M:%S) -----" | tee -a "$LOG"
  echo "+ $*" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then echo "[$name] OK" | tee -a "$LOG"
  else echo "[$name] !! FAILED rc=$? — continuing" | tee -a "$LOG"; fi
}

STEER=results/direction_probe/qwen8b_l23/steer_trash_direction.npz

# =============================================================================
# C1-CTRL — early-layer control for encode-vs-decode (all 3 models).
#   (1) re-emit the L* encode WITH a json sidecar (reuses existing tagged npz; CPU)
#   (2) recapture residuals at L2 (GPU)  (3) encode at L2 (CPU)  (4) compare (CPU)
# =============================================================================
if want C1; then
  for m in llama ministral qwen; do
    L=$(probe_layer $m)
    LATE=results/direction_probe/${m}8b_l${L}/raw_residuals_tagged.npz
    EARLY=results/direction_probe/${m}8b_l2/raw_residuals_tagged.npz
    # (1) CPU: add json sidecar to the existing L* encode
    [[ -f "$LATE" ]] && step "C1.reEmitLate.$m" python -m experiments.encode_vs_decode \
      --tagged "$LATE" --out results/direction_probe_baselines/ENCODE_VS_DECODE_${m}_l${L}.md --emit-json
    # (2) GPU: recapture at L2
    step "C1.recaptureEarly.$m" python -m experiments.bet_matched_recapture \
      --enriched-log $(logs_for $m) --layer 2 --device "$DEVICE" --dtype "$DTYPE" --out "$EARLY"
    # (3) CPU: encode at L2
    [[ -f "$EARLY" ]] && step "C1.encodeEarly.$m" python -m experiments.encode_vs_decode \
      --tagged "$EARLY" --out results/direction_probe_baselines/ENCODE_VS_DECODE_${m}_l2.md --emit-json
  done
  # (3b) CPU: (re)generate the Qwen steering direction so C2 is self-contained
  QLATE=results/direction_probe/qwen8b_l23/raw_residuals_tagged.npz
  [[ -f "$QLATE" ]] && step "C1.saveSteerDir.qwen" python -m experiments.encode_vs_decode \
    --tagged "$QLATE" --out results/direction_probe_baselines/ENCODE_VS_DECODE_qwen_l23.md \
    --emit-json --save-direction "$STEER"
  # (4) CPU: compare early vs late across all models
  step "C1.compare" python -m experiments.encode_layer_compare \
    --glob 'results/direction_probe_baselines/ENCODE_VS_DECODE_*.json' \
    --out results/direction_probe_baselines/ENCODE_LAYER_COMPARE.md
fi

# =============================================================================
# C2-DEC — steering decision readout, reworked (Qwen). L19 compute + L23 commit,
#          ±alpha, wrong-folds (illegal_fold) + legal folds, vs random control.
# =============================================================================
if want C2; then
  if [[ -f "$STEER" ]]; then
    for L in 19 23; do
      for tgt in illegal_fold clean_legal_fold; do
        step "C2dec.qwen.l${L}.${tgt}" python -m experiments.posterior_steering \
          --enriched-log $(logs_for qwen) --layer "$L" --direction "$STEER" \
          --alphas -8 -4 0 4 8 --target-bucket "$tgt" --n-decisions 60 \
          --device "$DEVICE" --dtype "$DTYPE" \
          --out-dir results/posterior_steering/qwen8b_dec_l${L}_${tgt}
      done
    done
  else
    echo "[C2dec] SKIP — missing $STEER (run Phase C of v1, or C1 here re-emits it via encode_vs_decode --save-direction)" | tee -a "$LOG"
  fi

# =============================================================================
# C2-BEL — steering BELIEF readout (Qwen): does steering reduce JS(belief,oracle)?
# =============================================================================
  if [[ -f "$STEER" ]]; then
    for tgt in illegal_fold ALL; do
      TGTARG=""; [[ "$tgt" != "ALL" ]] && TGTARG="--target-bucket $tgt"
      step "C2bel.qwen.l19.${tgt}" python -m experiments.steer_belief_readout \
        --enriched-log $(logs_for qwen) --layer 19 --direction "$STEER" \
        --alphas 0 2 4 6 --n-decisions 40 $TGTARG \
        --device "$DEVICE" --dtype "$DTYPE" \
        --out-dir results/posterior_steering/qwen8b_belief_l19_${tgt}
    done
  fi
fi

# =============================================================================
# C3 — behavior-at-scale RERUN with --cot (env is correct; v1 was non-CoT=degenerate).
#      steer with last_only + small alpha to avoid the over-steer fallbacks v1 hit.
# =============================================================================
if want C3; then
  step "C3.baseline.cot" python run_experiment.py --agent hf --hf-model qwen-8b --cot \
    --opponent threshold --opponent-preset informative_v2 --hands "$HANDS" --seed 42 \
    --elicit-beliefs --out logs/scale_qwen_cot_baseline.jsonl -v
  step "C3.ablate.cot" env CIRCUIT_ABLATE_LAYER=19 CIRCUIT_ABLATE_HEADS="31 3 21 1 0" \
    python run_experiment.py --agent hf --hf-model qwen-8b --cot \
    --opponent threshold --opponent-preset informative_v2 --hands "$HANDS" --seed 42 \
    --elicit-beliefs --out logs/scale_qwen_cot_ablate_l19.jsonl -v
  if [[ -f "$STEER" ]]; then
    step "C3.steer.cot" env CIRCUIT_STEER_LAYER=19 CIRCUIT_STEER_NPZ="$STEER" \
      CIRCUIT_STEER_ALPHA=2 CIRCUIT_STEER_LASTONLY=1 \
      python run_experiment.py --agent hf --hf-model qwen-8b --cot \
      --opponent threshold --opponent-preset informative_v2 --hands "$HANDS" --seed 42 \
      --elicit-beliefs --out logs/scale_qwen_cot_steer_l19.jsonl -v
  fi
fi

echo "" | tee -a "$LOG"
echo "=== followups v2 DONE $(date -u) ===" | tee -a "$LOG"
grep -E "\] (OK|!! FAILED|SKIP)" "$LOG" | tee -a "$LOG"
echo "Pull back: results/direction_probe_baselines/ENCODE_*  results/posterior_steering/*  logs/scale_qwen_cot_*.jsonl" | tee -a "$LOG"
