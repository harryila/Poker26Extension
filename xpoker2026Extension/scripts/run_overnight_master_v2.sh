#!/usr/bin/env bash
# =============================================================================
# Overnight master orchestrator v2 — non-CoT circuit hunt + position sweep
# + extended attention patterns + CoT-vs-nonCoT direction comparison.
# =============================================================================
#
# Designed to be fired once and forgotten. All sections auto-skip if their
# primary outputs already exist; failures don't abort the orchestrator.
#
# Total wall-clock estimate (worst case, no auto-skips):
#   Section 1 — Non-CoT circuit hunt master:    ~9-11 h
#   Section 2 — Direction probe (non-CoT) + cos: ~30-40 min
#   Section 3 — Position sweep (D1):              ~30-40 min
#   Section 4 — Extended attention patterns:      ~30-40 min
# Grand total: ~11-13 h. Fits an overnight slot.
#
# Order rationale:
#   - Section 1 first because it's the substantive non-CoT testing that
#     parallels the CoT mechanistic stack. If it fails or has issues, the
#     other sections still produce useful results from CoT-side data.
#   - Sections 2-4 are smaller, mostly read cached residuals, and are
#     ordered roughly by dependency — direction probe non-CoT (Section 2)
#     produces the cached direction; position sweep (Section 3) reads
#     existing CoT direction; extended attn (Section 4) is independent.
#
# Env knobs
# ---------
#   SKIP_<N>=1                skip section N (1..4)
#   SECTIONS=A,B,C,D          (Section 1) which non-CoT-circuit-hunt subsections to run
#   N_PER_BUCKET_ATTN         default 200 (Section 4)
#   N_PER_BUCKET_POSSWEEP     default 50 (Section 3)
#   PROBE_MAX_DECISIONS       default 300 (Section 2)
#   DEVICE / DTYPE            default cuda / bfloat16 (passed to all)
#
# Usage
# -----
#   tmux new -s overnight2 'bash scripts/run_overnight_master_v2.sh; \
#       echo; echo "[overnight v2 done]"; read -n1 -s'
# Detach with Ctrl-B D; reattach with `tmux attach -t overnight2`.
# =============================================================================

set -uo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

TS="$(date -u +%Y%m%d_%H%M%SZ)"
LOG_DIR="logs/overnight_v2_${TS}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

declare -a SECTION_STATUS=()

log_master() {
    echo "$@" | tee -a "$MASTER_LOG"
}

run_section() {
    local n="$1"
    local name="$2"
    local desc="$3"
    shift 3

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
log_master "OVERNIGHT MASTER ORCHESTRATOR v2"
log_master "Run started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_master "Device: $DEVICE  Dtype: $DTYPE"
log_master "Cell logs: $LOG_DIR/<n>_<name>.log"
log_master "Master log: $MASTER_LOG"
log_master "============================================================"

# -----------------------------------------------------------------------------
# SECTION 1 — Non-CoT circuit hunt (Sections A/B/C/D within the script)
# -----------------------------------------------------------------------------
run_section 1 "nocot_circuit_hunt" \
    "Non-CoT circuit hunt: layer sweep + per-seed + components + verb pairs" \
    bash scripts/run_nocot_circuit_hunt.sh

# -----------------------------------------------------------------------------
# SECTION 2 — Direction probe in non-CoT + CoT-vs-nonCoT cosine compare
# -----------------------------------------------------------------------------
run_section 2 "direction_probe_nocot_and_compare" \
    "Direction probe in non-CoT mode + CoT-vs-nonCoT cosine compare" \
    bash scripts/run_direction_probe_nocot_and_compare.sh

# -----------------------------------------------------------------------------
# SECTION 3 — Position-sweep direction projection (D1)
# -----------------------------------------------------------------------------
run_section 3 "position_sweep" \
    "Position-sweep direction projection (D1) across 3 models" \
    bash scripts/run_position_sweep.sh

# -----------------------------------------------------------------------------
# SECTION 4 — Extended attention patterns (200/bucket) + non-CoT attn
# -----------------------------------------------------------------------------
run_section 4 "extended_attention_patterns" \
    "Attention patterns at 200/bucket (CoT) + non-CoT attention patterns" \
    bash scripts/run_extended_attention_patterns.sh

# -----------------------------------------------------------------------------
# Final tally
# -----------------------------------------------------------------------------
log_master ""
log_master "============================================================"
log_master "OVERNIGHT MASTER v2 — FINAL TALLY"
log_master "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_master "============================================================"
log_master ""
log_master "Section results:"
log_master "  $(printf '%-3s %-40s %-8s %s' "#" "name" "status" "elapsed (s)")"
for entry in "${SECTION_STATUS[@]}"; do
    IFS='|' read -r n name status elapsed <<< "$entry"
    log_master "  $(printf '%-3s %-40s %-8s %s' "$n" "$name" "$status" "$elapsed")"
done

log_master ""
log_master "Output directories to inspect (in priority order):"
log_master "  1. results/causal_patching/llama8b_nocot_layer_sweep_s42_filtered/SUMMARY.md   <- non-CoT L* in Llama"
log_master "  2. results/causal_patching/qwen8b_nocot_layer_sweep_s42/SUMMARY.md             <- non-CoT L* in Qwen"
log_master "  3. results/direction_cosine_compare/{llama,qwen}_cot_vs_nocot_l*.md            <- shared direction?"
log_master "  4. results/position_sweep/{llama,ministral,qwen}8b_l*/SUMMARY.md               <- decision crystallization"
log_master "  5. results/causal_patching/{llama,qwen}8b_nocot_perseed_*/SUMMARY.md           <- per-seed non-CoT"
log_master "  6. results/causal_patching/{llama,qwen}8b_nocot_l*_components/SUMMARY_components.md   <- non-CoT heads"
log_master "  7. results/causal_patching/{llama,qwen}8b_nocot_verbpair_*/SUMMARY.md          <- non-CoT verb pairs"
log_master "  8. results/attention_patterns/{model}8b_l*_extended/SUMMARY.md                 <- 200/bucket"
log_master "  9. results/attention_patterns/{model}8b_l*_nocot/SUMMARY.md                    <- non-CoT attn"
log_master ""
log_master "Master log: $MASTER_LOG"

for entry in "${SECTION_STATUS[@]}"; do
    if [[ "$entry" == *"|FAILED|"* ]]; then
        log_master ""
        log_master "[overnight v2] one or more sections FAILED; check $LOG_DIR/<n>_<name>.log"
        exit 1
    fi
done
log_master ""
log_master "[overnight v2] all sections completed (some may have skipped — check tally)"
exit 0
