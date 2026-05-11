#!/usr/bin/env bash
# =============================================================================
# Overnight master orchestrator v4 — full Phase O cleanup + extensions.
# =============================================================================
#
# 11 sections covering: D1 Tier 0 smoke test (paper foundation), C3
# commit-layer components, A3+B2 cheap analyses on cached residuals,
# A1+A2 GPU ablation experiments, B3 belief direction probe, C2
# residual-top1-labeled probe, mode-balanced direction probe (item 4 from
# previous batch — re-runs if needed), and D3 Tier 4 8B behavioral runs.
#
# Total wall-clock estimate (worst case, no auto-skips):
#   §1 D1 Tier 0 smoke test:                        ~5-10 min  (CPU)
#   §2 A3 + B2 (cached-residual analyses):          ~5-10 min  (CPU)
#   §3 A1 head ablation (3 models):                 ~30-45 min
#   §4 A2 attention-mask ablation (3 models):       ~30 min
#   §5 B3 belief direction probe (3 models):        ~60-90 min
#   §6 C3 commit-layer components (Ministral+Qwen): ~100 min
#   §7 C2 residual-top1-labeled probe (3 models):   ~60-90 min
#   §8 mode-balanced direction probe (Llama+Qwen):  ~30 min
#   §9 D3 Tier 4 opponent behavioral (5×3 cells):   ~3-5 h
#   §10 (deferred): Tier 4 patching at L* on opponent logs — fired by
#       separate follow-up if illegal_fold counts permit.
# Grand total: ~7-10 h. Comfortably overnight.
#
# Auto-skip per cell on existing outputs; failures don't abort.
#
# Env knobs
# ---------
#   SKIP_<N>=1       skip section N (1..9)
#   DEVICE / DTYPE   default cuda / bfloat16
#
# Usage
# -----
#   tmux new -s overnight4 'bash scripts/run_overnight_master_v4.sh; \
#       echo; echo "[overnight v4 done]"; read -n1 -s'
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

TS="$(date -u +%Y%m%d_%H%M%SZ)"
LOG_DIR="logs/overnight_v4_${TS}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"
declare -a SECTION_STATUS=()

log_master() { echo "$@" | tee -a "$MASTER_LOG"; }

run_section() {
    local n="$1"; local name="$2"; local desc="$3"; shift 3
    local skip_var="SKIP_${n}"
    if [[ "${!skip_var:-0}" == "1" ]]; then
        log_master ""
        log_master "============================================================"
        log_master "SECTION $n / SKIPPED via $skip_var=1: $desc"
        log_master "============================================================"
        SECTION_STATUS+=("$n|$name|SKIPPED|0")
        return 0
    fi
    log_master ""
    log_master "============================================================"
    log_master "SECTION $n: $desc"
    log_master "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    log_master "============================================================"
    local cell_log="$LOG_DIR/${n}_${name}.log"
    local start=$SECONDS
    if "$@" 2>&1 | tee "$cell_log" >> "$MASTER_LOG"; then
        local elapsed=$((SECONDS - start))
        log_master "[section $n DONE] ${elapsed}s — $desc"
        SECTION_STATUS+=("$n|$name|PASSED|$elapsed")
    else
        local elapsed=$((SECONDS - start))
        log_master "[section $n FAILED but continuing] ${elapsed}s — $desc"
        SECTION_STATUS+=("$n|$name|FAILED|$elapsed")
    fi
}

log_master "============================================================"
log_master "OVERNIGHT MASTER ORCHESTRATOR v4 — Phase O"
log_master "Run started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_master "Device: $DEVICE  Dtype: $DTYPE"
log_master "Cell logs: $LOG_DIR/<n>_<name>.log"
log_master "============================================================"

# §1 D1 Tier 0 smoke test (must pass before everything else).
run_section 1 "tier0_smoke" \
    "D1 Tier 0 replication smoke test (70B paper anchor)" \
    bash scripts/run_tier0_smoke_test.sh

# §2 A3 baselines + B2 magnitude — cheap analysis on cached residuals.
run_section 2 "probe_baselines" \
    "A3 direction-probe baselines (random + cross-task) — analysis only" \
    bash scripts/run_direction_probe_baselines.sh

run_section 3 "magnitude_analysis" \
    "B2 CoT vs non-CoT residual magnitude analysis — analysis only" \
    bash scripts/run_cot_magnitude_analysis.sh

# §4 A1 head ablation — necessity test (~45 min).
run_section 4 "head_ablation" \
    "A1 head ablation (necessity) — Llama+Ministral+Qwen at L*" \
    bash scripts/run_head_ablation.sh

# §5 A2 attention-mask ablation — legal-actions necessity (~30 min).
run_section 5 "attn_mask_ablation" \
    "A2 attention-mask ablation (legal_actions list)" \
    bash scripts/run_attention_mask_ablation.sh

# §6 B3 belief direction probe — connect verb to belief subspace.
run_section 6 "belief_probe" \
    "B3 belief direction probe (oracle_strategy_aware)" \
    bash scripts/run_belief_direction_probe.sh

# §7 C3 commit-layer components — Ministral L=17, Qwen L=24.
run_section 7 "commit_components" \
    "C3 commit-layer components — Ministral L=17, Qwen L=24" \
    bash scripts/run_components_at_commit_layers.sh

# §8 C2 residual-top1-labeled probes — re-run with new label flag.
SECTION_8_OUT_LLAMA="results/direction_probe_residual_top1/llama8b_l14"
SECTION_8_OUT_MINISTRAL="results/direction_probe_residual_top1/ministral8b_l16"
SECTION_8_OUT_QWEN="results/direction_probe_residual_top1/qwen8b_l23"

run_one_residual_top1() {
    local short="$1"; local layer="$2"; shift 2
    local logs="$@"
    local out_dir="results/direction_probe_residual_top1/${short}8b_l${layer}"
    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir exists"
        return 0
    fi
    local first_log
    first_log=$(echo $logs | awk '{print $1}')
    if [[ ! -f "$first_log" ]]; then
        echo "[skip] $short: missing $first_log"
        return 0
    fi
    mkdir -p "$out_dir"
    python -m experiments.decision_direction_probe \
        --enriched-log $logs \
        --layer "$layer" \
        --max-decisions-per-bucket 300 \
        --label-source residual_top1 \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE"
}

run_section 8 "residual_top1_probe" \
    "C2 residual-top1-labeled probes (3 models)" \
    bash -c '
        set -uo pipefail
        cd '"$(pwd)"'
        '"$(declare -f run_one_residual_top1)"'
        export DEVICE='"$DEVICE"' DTYPE='"$DTYPE"'
        run_one_residual_top1 llama 14 \
            logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
        run_one_residual_top1 ministral 16 \
            logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
        run_one_residual_top1 qwen 23 \
            logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
    '

# §9 D3 Tier 4 opponent robustness (8B behavioral) — ~3-5 h, biggest cell.
run_section 9 "tier4_opponent_8b" \
    "D3 Tier 4 opponent robustness (5 presets × 3 models, 8B behavioral)" \
    bash scripts/run_tier4_opponent_behavioral.sh

# Final tally.
log_master ""
log_master "============================================================"
log_master "OVERNIGHT MASTER v4 — FINAL TALLY"
log_master "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_master "============================================================"
log_master "Section results:"
log_master "  $(printf '%-3s %-32s %-8s %s' "#" "name" "status" "elapsed (s)")"
for entry in "${SECTION_STATUS[@]}"; do
    IFS='|' read -r n name status elapsed <<< "$entry"
    log_master "  $(printf '%-3s %-32s %-8s %s' "$n" "$name" "$status" "$elapsed")"
done

log_master ""
log_master "Output directories to inspect (priority order):"
log_master "  1. results/tier0_smoke_test/SUMMARY.md                 (paper foundation pass/fail)"
log_master "  2. results/head_ablation/{llama,ministral,qwen}8b_l*/SUMMARY.md  (necessity)"
log_master "  3. results/attn_mask_ablation/{llama,ministral,qwen}8b/SUMMARY.md (attn necessity)"
log_master "  4. results/direction_probe_baselines/*.md             (probe credibility)"
log_master "  5. results/cot_magnitude_analysis/*.md                (CoT prior mechanism)"
log_master "  6. results/belief_direction_probe/*/SUMMARY.md        (belief × verb subspaces)"
log_master "  7. results/causal_patching/{ministral8b_l17,qwen8b_l24}_components/SUMMARY_components.md  (commit layer)"
log_master "  8. results/direction_probe_residual_top1/*/SUMMARY.md (residual-vs-recorded labels)"
log_master "  9. results/tier4_opponent/*_*/SUMMARY.md              (opponent robustness behavioral)"

for entry in "${SECTION_STATUS[@]}"; do
    if [[ "$entry" == *"|FAILED|"* ]]; then
        log_master ""
        log_master "[overnight v4] one or more sections FAILED; check $LOG_DIR/<n>_<name>.log"
        exit 1
    fi
done
log_master ""
log_master "[overnight v4] all sections completed (some may have skipped)"
exit 0
