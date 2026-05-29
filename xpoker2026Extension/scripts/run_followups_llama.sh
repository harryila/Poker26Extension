#!/usr/bin/env bash
# Llama-only follow-up pass (fresh download). Purges other 8B caches first so
# the ~20 GB quota is free, then runs B1 recapture+probe, C1 encode_vs_decode,
# B2 facing/nobet (L14,L15) — mirrors run_followups_gpu.sh exactly.
set -uo pipefail
cd "$(dirname "$0")/.."

: "${DEVICE:=cuda}"; : "${DTYPE:=bfloat16}"; : "${SEEDS:=42 123 456}"
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f /root/.hf_token ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

LOG="results/followups_llama_$(date -u +%Y%m%dT%H%M%SZ).log"; mkdir -p results
echo "=== llama pass start=$(date -u) ===" | tee "$LOG"
logs_for () { local out=""; for s in $SEEDS; do out="$out logs/cot_llama8b_t0_s${s}_informative_v2_enriched.jsonl.gz"; done; echo "$out"; }
step () { local name=$1; shift
  echo "" | tee -a "$LOG"; echo "----- [$name] start $(date -u +%H:%M:%S) -----" | tee -a "$LOG"; echo "+ $*" | tee -a "$LOG"; local t0=$SECONDS
  if "$@" >>"$LOG" 2>&1; then echo "[$name] OK ($((SECONDS-t0))s)" | tee -a "$LOG"; else echo "[$name] !! FAILED rc=$? — continuing" | tee -a "$LOG"; fi; }

# Free the quota: drop any other cached 8B weights + any corrupt llama partial.
for d in models--Qwen--Qwen3-8B models--mistralai--Ministral-8B-Instruct-2410 models--meta-llama--Llama-3.1-8B-Instruct; do
  t="${HF_HUB_CACHE%/}/$d"; [[ -d "$t" ]] && { echo "[purge] $t" | tee -a "$LOG"; rm -rf "$t"; }
done
xet="$(dirname "$HF_HUB_CACHE")/xet"; [[ -d "$xet" ]] && rm -rf "${xet:?}"/* 2>/dev/null
du -sh "$HF_HOME" 2>/dev/null | tee -a "$LOG"

LTAG=results/direction_probe/llama8b_l14/raw_residuals_tagged.npz
step "B1.recapture.llama.l14" python -m experiments.bet_matched_recapture \
  --enriched-log $(logs_for) --layer 14 --device "$DEVICE" --dtype "$DTYPE" --out "$LTAG"
[[ -f "$LTAG" ]] && step "B1.probe.llama.l14" python -m experiments.bet_matched_probe \
  --tagged "$LTAG" --out results/direction_probe_baselines/BET_MATCHED_llama_l14.md
[[ -f "$LTAG" ]] && step "C1.encode_vs_decode.llama" python -m experiments.encode_vs_decode \
  --tagged "$LTAG" --out results/direction_probe_baselines/ENCODE_VS_DECODE_llama_l14.md
for L in 14 15; do
  step "B2.facing.llama.l$L" python -m experiments.causal_patching \
    --enriched-log $(logs_for) \
    --source-bucket clean_check_or_call --source-bet-filter facing \
    --target-bucket clean_legal_fold --target-bet-filter facing \
    --layers "$L" --n-source 10 --n-target 30 --seed 42 --device "$DEVICE" --dtype "$DTYPE" \
    --out-dir results/causal_patching/llama8b_betmatched_facing_l${L}
  step "B2.nobet.llama.l$L" python -m experiments.causal_patching \
    --enriched-log $(logs_for) \
    --source-bucket clean_check_or_call --source-bet-filter nobet \
    --target-bucket illegal_fold --target-bet-filter nobet \
    --layers "$L" --n-source 10 --n-target 30 --seed 42 --device "$DEVICE" --dtype "$DTYPE" \
    --out-dir results/causal_patching/llama8b_betmatched_nobet_l${L}
done

echo "" | tee -a "$LOG"; echo "=== llama pass DONE $(date -u) ===" | tee -a "$LOG"
grep -E "\] (OK|!! FAILED|SKIP)" "$LOG" | tee -a "$LOG"
