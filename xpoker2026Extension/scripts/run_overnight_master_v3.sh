#!/usr/bin/env bash
# =============================================================================
# Overnight master orchestrator v3 — Phase N cleanup + mode-balanced probe.
# =============================================================================
#
# This batch closes the §18g/h known gaps from Phase M:
#   - Re-runs Section C (non-CoT components) with --target-residual-top1 +
#     --source-residual-top1 filters (now supported in component_patching.py).
#   - Re-runs Section D (non-CoT verb pairs) with the filter applied
#     uniformly to ALL Llama cells (the §18g halt fix).
#   - Adds a mode-balanced direction probe: same-hand CoT vs non-CoT
#     residuals at L*, then cosine compare. Tells us whether the §18b
#     0.27/0.34 cosine was a data-distribution artifact or a real
#     direction tilt.
#
# Pre-existing cells (e.g. earlier Section A/B/D bet_to_fold) are PRESERVED.
# The new filtered cells write to `*_filtered` directories.
#
# Wall-clock estimate
# -------------------
# Section 1 — non-CoT circuit hunt rerun (Sections C + D filtered):  ~2-3 h
# Section 2 — mode-balanced direction probe:                         ~30 min
# Total: ~3 h. Comfortably overnight.
#
# Env knobs
# ---------
#   SKIP_<N>=1       skip section N (1..2)
#   SECTIONS=C,D     which non-CoT-circuit-hunt sub-sections to run
#   MAX_PAIRS=200    mode-balanced probe pair cap
#   DEVICE / DTYPE   default cuda / bfloat16
#
# Usage
# -----
#   tmux new -s overnight3 'bash scripts/run_overnight_master_v3.sh; \
#       echo; echo "[overnight v3 done]"; read -n1 -s'
# =============================================================================

set -uo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

TS="$(date -u +%Y%m%d_%H%M%SZ)"
LOG_DIR="logs/overnight_v3_${TS}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

declare -a SECTION_STATUS=()

log_master() { echo "$@" | tee -a "$MASTER_LOG"; }

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
log_master "OVERNIGHT MASTER ORCHESTRATOR v3 — Phase N cleanup"
log_master "Run started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_master "Device: $DEVICE  Dtype: $DTYPE"
log_master "Cell logs: $LOG_DIR/<n>_<name>.log"
log_master "Master log: $MASTER_LOG"
log_master "============================================================"

# -----------------------------------------------------------------------------
# SECTION 1 — Non-CoT circuit hunt rerun (filtered components + verb pairs)
# Re-runs Sections C and D of run_nocot_circuit_hunt.sh which now use the
# new --target-residual-top1 + --source-residual-top1 + --baseline-tolerance-frac
# 0.0 flags. Sections A and B are skipped because they already produced
# clean outputs in Phase M; the cells will auto-skip on existing SUMMARY.md
# files anyway.
# -----------------------------------------------------------------------------
SECTIONS_FOR_NOCOT="${SECTIONS:-C,D}"
run_section 1 "nocot_circuit_hunt_rerun" \
    "Non-CoT circuit hunt rerun: Sections $SECTIONS_FOR_NOCOT (filtered)" \
    env "SECTIONS=$SECTIONS_FOR_NOCOT" \
    bash scripts/run_nocot_circuit_hunt.sh

# -----------------------------------------------------------------------------
# SECTION 2 — Mode-balanced direction probe
# -----------------------------------------------------------------------------
run_section 2 "mode_balanced_probe" \
    "Mode-balanced direction probe (hand-matched CoT vs non-CoT)" \
    bash scripts/run_mode_balanced_direction_probe.sh

# -----------------------------------------------------------------------------
# Final tally
# -----------------------------------------------------------------------------
log_master ""
log_master "============================================================"
log_master "OVERNIGHT MASTER v3 — FINAL TALLY"
log_master "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_master "============================================================"
log_master ""
log_master "Section results:"
log_master "  $(printf '%-3s %-32s %-8s %s' "#" "name" "status" "elapsed (s)")"
for entry in "${SECTION_STATUS[@]}"; do
    IFS='|' read -r n name status elapsed <<< "$entry"
    log_master "  $(printf '%-3s %-32s %-8s %s' "$n" "$name" "$status" "$elapsed")"
done

log_master ""
log_master "Output directories to inspect:"
log_master "  1. results/causal_patching/{llama,qwen}8b_nocot_l*_components_filtered/SUMMARY_components.md"
log_master "     (NEW: non-CoT component decomposition with proper filtering)"
log_master "  2. results/causal_patching/{llama,qwen}8b_nocot_verbpair_*_s42_l*_filtered/SUMMARY.md"
log_master "     (NEW: completes the verb-pair matrix in non-CoT for Llama)"
log_master "  3. results/mode_balanced_probe/{llama,qwen}8b_l*/SUMMARY.md"
log_master "     (NEW: hand-matched CoT vs non-CoT direction cosine)"
log_master ""
log_master "Master log: $MASTER_LOG"

for entry in "${SECTION_STATUS[@]}"; do
    if [[ "$entry" == *"|FAILED|"* ]]; then
        log_master ""
        log_master "[overnight v3] one or more sections FAILED; check $LOG_DIR/<n>_<name>.log"
        exit 1
    fi
done
log_master ""
log_master "[overnight v3] all sections completed (some may have skipped)"
exit 0
