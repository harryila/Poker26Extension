#!/usr/bin/env bash
# =============================================================================
# run_followups_gpu_v3.sh — third GPU batch. Fixes the v2 design errors and
# converts the v2 artifacts into either positive results or DEFENSIBLE nulls.
# Dependency-ordered, continue-on-fail, guarded; CPU steps run inline.
#
# v2 verdicts (CLAIMS_AND_IDENTIFICATION.md "RESULTS LANDED v2-batch"):
#   - steering null = ARTIFACT (wrong vector orthogonal to causal axis + alpha 6-25x too big)
#   - encode "knows" = INPUT-PRESENCE (oracle = fn of prompt token counts) -> redirect to equity
#   - C3 = L19 reads as bet-suppression in free play; on recorded folds it flips 57% CHECK -> POOL-DEPENDENT
#
# v3:
#   P0-STEER   re-run steering with the CAUSAL decision direction (centroid_diff, cos 0.999) at
#              SANE alpha (gap units {0..1.5}); single-forward readout. Built-in random control.
#   P0-NECC    same-depth permutation null: top5 vs 50 random L19 5-head draws -> calibrated p,
#              with flip-destination split (CHECK vs BET) = the fold-vs-aggression test on the pool.
#   P1-BATTERY action-shift battery: single-forward, fully-paired facing-bet spots, baseline vs
#              ablate(L19) vs steer(L23 decision dir) -> clean FOLD/CHECK/BET transition matrix.
#   P1-COMPUTE recapture L2 + L* WITH equity+input-feats, then computed_quantity_probe: does the
#              residual ESTIMATE equity (hidden-card win prob) beyond the prompt, late>>early?
#
# Usage:
#   cd xpoker2026Extension && git pull
#   export HF_HOME=/workspace/huggingface HF_TOKEN=...
#   bash scripts/run_followups_gpu_v3.sh                 # all (~4-7 GPU-h; P0-NECC dominates)
#   PHASES="P0-STEER" bash scripts/run_followups_gpu_v3.sh
# Overridable: DEVICE DTYPE SEEDS NDRAWS PHASES MODELS_COMPUTE
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

: "${DEVICE:=cuda}"
: "${DTYPE:=bfloat16}"
: "${SEEDS:=42 123 456}"
: "${NDRAWS:=50}"                       # random head draws for the permutation null
: "${PHASES:=P0-STEER P0-NECC P1-BATTERY P1-COMPUTE}"
: "${MODELS_COMPUTE:=qwen}"             # models for the computed-quantity probe (qwen primary)

LOG="results/followups_v3_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p results
echo "=== followups v3 start=$(date -u) device=$DEVICE dtype=$DTYPE ndraws=$NDRAWS phases='$PHASES' ===" | tee "$LOG"

want () { [[ " $PHASES " == *" $1 "* ]]; }
qlogs () { local out=""; for s in $SEEDS; do out="$out logs/cot_qwen8b_t0_s${s}_informative_v2_enriched.jsonl.gz"; done; echo "$out"; }
mlogs () { local m=$1 out=""; for s in $SEEDS; do out="$out logs/cot_${m}8b_t0_s${s}_informative_v2_enriched.jsonl.gz"; done; echo "$out"; }
probe_layer () { case $1 in llama) echo 14;; ministral) echo 16;; qwen) echo 23;; esac; }
step () {
  local name=$1; shift
  echo "" | tee -a "$LOG"; echo "----- [$name] $(date -u +%H:%M:%S) -----" | tee -a "$LOG"; echo "+ $*" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then echo "[$name] OK" | tee -a "$LOG"; else echo "[$name] !! FAILED rc=$? — continuing" | tee -a "$LOG"; fi
}

L23_STEER=results/direction_probe/qwen8b_l23/steer_decision_direction.npz
L19_STEER=results/direction_probe/qwen8b_l19/steer_decision_direction.npz
L19_TAG=results/direction_probe/qwen8b_l19/raw_residuals_tagged.npz

# =============================================================================
# P0-STEER — corrected steering with the CAUSAL decision direction, sane alpha.
# =============================================================================
if want P0-STEER; then
  # L23 decision dir from the existing probe residuals (CPU)
  step "P0steer.mkdir.l23" python -m experiments.make_decision_steer_dir \
    --in results/direction_probe/qwen8b_l23/raw_residuals.npz --layer 23 --out "$L23_STEER"
  # L19 decision dir needs an L19 recapture first (GPU) then build (CPU)
  step "P0steer.recap.l19" python -m experiments.bet_matched_recapture \
    --enriched-log $(qlogs) --layer 19 --device "$DEVICE" --dtype "$DTYPE" --out "$L19_TAG"
  [[ -f "$L19_TAG" ]] && step "P0steer.mkdir.l19" python -m experiments.make_decision_steer_dir \
    --in "$L19_TAG" --layer 19 --out "$L19_STEER"
  # single-forward steering readout at both layers x both fold buckets, sane alphas (gap units)
  for L in 23 19; do
    DIR=$([[ $L == 23 ]] && echo "$L23_STEER" || echo "$L19_STEER")
    [[ -f "$DIR" ]] || { echo "[P0steer.l$L] SKIP — missing $DIR" | tee -a "$LOG"; continue; }
    for tgt in illegal_fold clean_legal_fold; do
      step "P0steer.run.l${L}.${tgt}" python -m experiments.posterior_steering \
        --enriched-log $(qlogs) --layer "$L" --direction "$DIR" \
        --alphas 0 0.25 0.5 0.75 1.0 1.5 --target-bucket "$tgt" --n-decisions 60 \
        --device "$DEVICE" --dtype "$DTYPE" \
        --out-dir results/posterior_steering/qwen8b_decdir_l${L}_${tgt}
    done
  done
fi

# =============================================================================
# P0-NECC — same-depth permutation null at Qwen L19 (top5 vs NDRAWS random 5-head sets).
# =============================================================================
if want P0-NECC; then
  RAND_SETS=""
  for i in $(seq 0 $((NDRAWS-1))); do RAND_SETS="$RAND_SETS $(printf 'r%03d:rand:5:%d' $i $i)"; done
  step "P0necc.ablate.l19" python -m experiments.inference_head_ablation \
    --enriched-log $(qlogs) \
    --layer 19 --pipeline recon --filter-recorded-bucket clean_legal_fold \
    --conditions baseline \
    --head-sets 'top5:31 3 21 1 0' $RAND_SETS \
    --device "$DEVICE" --dtype "$DTYPE" \
    --out-dir results/inference_head_ablation/qwen8b_l19_permnull_clean_legal_fold
  step "P0necc.pvalue" python -m experiments.permutation_null_pvalue \
    --cell results/inference_head_ablation/qwen8b_l19_permnull_clean_legal_fold \
    --named top5 --rand-prefix r --min-draws 20 \
    --out results/inference_head_ablation/PERMNULL_qwen_l19.md
fi

# =============================================================================
# P1-BATTERY — fold-vs-aggression action-shift battery (single-forward, fully paired).
# =============================================================================
if want P1-BATTERY; then
  STEERARG=""; [[ -f "$L23_STEER" ]] && STEERARG="--steer-dir $L23_STEER --steer-layer 23 --steer-alpha 0.75"
  step "P1battery.qwen" python -m experiments.action_shift_battery \
    --enriched-log $(qlogs) --ablate-layer 19 --ablate-heads 31 3 21 1 0 \
    $STEERARG --buckets clean_legal_fold clean_check_or_call --n 200 \
    --device "$DEVICE" --dtype "$DTYPE" \
    --out-dir results/action_shift/qwen8b_facingbet
fi

# =============================================================================
# P1-COMPUTE — recapture L2 + L* WITH equity/input-feats, then computed-quantity probe.
# =============================================================================
if want P1-COMPUTE; then
  for m in $MODELS_COMPUTE; do
    L=$(probe_layer $m)
    E=results/direction_probe/${m}8b_l2/raw_residuals_tagged.npz
    Lf=results/direction_probe/${m}8b_l${L}/raw_residuals_tagged.npz
    # re-recapture (the v2 npz lack equity/input_feats; the script now saves them)
    step "P1compute.recap.${m}.l2" python -m experiments.bet_matched_recapture \
      --enriched-log $(mlogs $m) --layer 2 --device "$DEVICE" --dtype "$DTYPE" --out "$E"
    step "P1compute.recap.${m}.l${L}" python -m experiments.bet_matched_recapture \
      --enriched-log $(mlogs $m) --layer "$L" --device "$DEVICE" --dtype "$DTYPE" --out "$Lf"
    if [[ -f "$E" && -f "$Lf" ]]; then
      step "P1compute.probe.${m}" python -m experiments.computed_quantity_probe \
        --early "$E" --late "$Lf" --target equity \
        --out results/direction_probe_baselines/COMPUTED_QUANTITY_${m}.md
    fi
  done
fi

echo "" | tee -a "$LOG"
echo "=== followups v3 DONE $(date -u) ===" | tee -a "$LOG"
grep -E "\] (OK|!! FAILED|SKIP)" "$LOG" | tee -a "$LOG"
echo "Pull back: results/posterior_steering/qwen8b_decdir_*  results/inference_head_ablation/PERMNULL_qwen_l19.md  results/action_shift/*  results/direction_probe_baselines/COMPUTED_QUANTITY_*" | tee -a "$LOG"
