#!/usr/bin/env bash
# =============================================================================
# Overnight master orchestrator v5 — Phase P polish + mechanistic Tier 4.
# =============================================================================
#
# Closes every §19j follow-up from updates.md and adds the natural
# mechanistic next step on the Tier 4 opponent-robustness data:
#
#   §1 P1 A3 cleanup     — direction-probe baselines with `position`
#                           cross-task feature + class balancing.
#                           Closes §19j.3 + §19j.4.   Pure CPU. ~5 min.
#
#   §2 P2 A1 illegal_fold — head ablation on near-threshold targets.
#                           Closes §19j.5 (sufficiency-without-necessity
#                           caveat). 3 models × ~12 min ≈ 35 min.
#
#   §3 P3 B3 follow-up    — held-out R^2 (§19a caveat 1) AND
#                           agent_belief orthogonality (§19a caveat 2)
#                           combined into one wrapper. 3 models × 2
#                           variants × ~10 min ≈ 60 min.
#
#   §4 P4 mode-balanced   — Llama L=14 + Qwen L=23 hand-matched CoT-vs-
#                           non-CoT cosine. Re-runs from earlier batch
#                           (results were not committed). ~30 min.
#
#   §5 P5 Tier 4 patching — 12 cells (5 presets × 3 models, minus the
#                           3 loose_passive cells with 0 clean_LF).
#                           clean_CC -> clean_LF at each model's L*.
#                           Tests opponent-stability of the verb circuit.
#                           ~10-15 min/cell × 12 ≈ 2-3 h.
#
# Total wall-clock estimate: ~3.5-4.5 h, comfortably overnight.
#
# Auto-skip per cell on existing outputs; failures don't abort.
#
# Env knobs
# ---------
#   SKIP_<N>=1       skip section N (1..5)
#   DEVICE / DTYPE   default cuda / bfloat16
#
# Usage
# -----
#   tmux new -s overnight5 'bash scripts/run_overnight_master_v5.sh; \
#       echo; echo "[overnight v5 done]"; read -n1 -s'
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
export DEVICE DTYPE

TS="$(date -u +%Y%m%d_%H%M%SZ)"
LOG_DIR="logs/overnight_v5_${TS}"
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
log_master "OVERNIGHT MASTER ORCHESTRATOR v5 — Phase P"
log_master "Run started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_master "Device: $DEVICE  Dtype: $DTYPE"
log_master "Cell logs: $LOG_DIR/<n>_<name>.log"
log_master "============================================================"

# §1 P1 A3 cleanup — analysis only, no GPU.
run_section 1 "a3_cleanup" \
    "P1 A3 baseline cleanup (position cross-task + class balance)" \
    bash scripts/run_a3_cleanup.sh

# §2 P2 A1 head ablation on illegal_fold (near-threshold).
run_section 2 "a1_illegal_fold" \
    "P2 A1 head ablation on illegal_fold targets (near-threshold)" \
    bash scripts/run_a1_illegal_fold_ablation.sh

# §3 B3 held-out R² + agent_belief — combined into one wrapper script.
run_section 3 "b3_followup" \
    "P3 B3 follow-up (held-out R^2 + agent_belief variant)" \
    bash scripts/run_b3_followup.sh

# §4 mode-balanced direction probe (Llama+Qwen).
run_section 4 "mode_balanced" \
    "P4 mode-balanced direction probe (CoT vs non-CoT, hand-matched)" \
    bash scripts/run_mode_balanced_direction_probe.sh

# §5 Tier 4 L* patching — 12 cells (skip 3 loose_passive on missing LF).
run_section 5 "tier4_patching" \
    "P5 Tier 4 L* patching (5 presets x 3 models, minus loose_passive)" \
    bash scripts/run_tier4_patching.sh

# Final tally.
log_master ""
log_master "============================================================"
log_master "OVERNIGHT MASTER v5 — FINAL TALLY"
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
log_master "  1. results/direction_probe_baselines/*_phaseP.md         (probe credibility cleanup)"
log_master "  2. results/head_ablation/*_illegal_fold/SUMMARY.md       (near-threshold necessity)"
log_master "  3. results/belief_direction_probe/*_heldout/SUMMARY.md   (held-out R^2)"
log_master "  4. results/belief_direction_probe/*_agent_belief/SUMMARY.md (agent-belief orthogonality)"
log_master "  5. results/mode_balanced_probe/*/SUMMARY.md              (matched cross-mode cosine)"
log_master "  6. results/causal_patching/tier4_*_l*/SUMMARY.md         (cross-preset patching)"

for entry in "${SECTION_STATUS[@]}"; do
    if [[ "$entry" == *"|FAILED|"* ]]; then
        log_master ""
        log_master "[overnight v5] one or more sections FAILED; check $LOG_DIR/<n>_<name>.log"
        exit 1
    fi
done
log_master ""
log_master "[overnight v5] all sections completed (some may have skipped)"
exit 0
