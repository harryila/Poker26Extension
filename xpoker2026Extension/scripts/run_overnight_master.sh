#!/usr/bin/env bash
# =============================================================================
# Overnight master orchestrator — runs every queued + candidate experiment
# sequentially, with continue-on-error and structured logging.
# =============================================================================
#
# Designed to be fired once and forgotten. Each cell:
#   - Auto-skips if its primary output already exists.
#   - Logs to logs/overnight_<timestamp>/<cell>.log
#   - Failures don't abort the orchestrator (set +e per-cell).
#   - A final tally at the end shows which cells PASSED, FAILED, or SKIPPED.
#
# Total wall-clock estimate (worst case, no auto-skips firing):
#   - Section 1 critical (non-CoT clean→clean):    ~60-80 min
#   - Section 2 Qwen B1 at L=23:                   ~50 min
#   - Section 3 reverse component decomposition:    ~50 min
#   - Section 4 per-seed Llama L=14 components:     ~150 min
#   - Section 5 verb-pair sweeps (RAISE↔FOLD):      ~5-6 h
#   - Section 6 cross-temperature Ministral t=0.2:  ~50 min
#   - Section 7 direction probe (3 models):         ~30 min
#   - Section 8 attention pattern analysis:         ~30 min
# Grand total worst case: ~10-12 h. Fits an overnight slot.
#
# Order rationale:
#   - Section 1 first because it's THE highest-value experiment (closes
#     the circuit-vs-failure-mode analytical hole).
#   - Sections 2-3 are quick one-shots that close cross-model parity.
#   - Section 4 is the bigger time sink but valuable for reviewer-defense.
#   - Sections 5-6 fill out the verb-pair matrix and cross-temperature
#     robustness.
#   - Sections 7-8 are the new probe-style experiments — they use a fresh
#     model load each, so running them last after all the patching cells
#     is fine (no shared state to clobber).
#
# Env knobs
# ---------
#   SKIP_<N>=1       skip section N (1..8). Useful to re-run only certain
#                    cells without editing this file.
#   DEVICE / DTYPE   default cuda / bfloat16 (passed to all sub-scripts)
#   ATTN_PROBE_MAX_DECISIONS   default 50 (per-bucket cap for attention pattern)
#   PROBE_MAX_DECISIONS        default 300 (per-bucket cap for direction probe)
#
# Usage
# -----
#   tmux new -s overnight 'bash scripts/run_overnight_master.sh; \
#       echo; echo "[overnight done]"; read -n1 -s'
# Detach with Ctrl-B D; reattach with `tmux attach -t overnight`.
# =============================================================================

# Note: NOT using set -e — we want continue-on-error.
set -uo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_PROBE_MAX_DECISIONS="${ATTN_PROBE_MAX_DECISIONS:-50}"
PROBE_MAX_DECISIONS="${PROBE_MAX_DECISIONS:-300}"

TS="$(date -u +%Y%m%d_%H%M%SZ)"
LOG_DIR="logs/overnight_${TS}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

# Track per-section status for the final tally.
declare -a SECTION_STATUS=()

log_master() {
    echo "$@" | tee -a "$MASTER_LOG"
}

run_section() {
    local n="$1"          # section number 1..8
    local name="$2"       # section short name (used in filename)
    local desc="$3"       # human-readable description
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
    # Run the command, tee to the cell log, mirror to master log.
    # `set +e` style: capture exit code without aborting the orchestrator.
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

# Header.
log_master "============================================================"
log_master "OVERNIGHT MASTER ORCHESTRATOR"
log_master "Run started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_master "Device: $DEVICE  Dtype: $DTYPE"
log_master "Cell logs: $LOG_DIR/<n>_<name>.log"
log_master "Master log: $MASTER_LOG"
log_master "============================================================"

# -----------------------------------------------------------------------------
# SECTION 1 — Non-CoT clean→clean (the critical analytical hole)
# -----------------------------------------------------------------------------
run_section 1 "nocot_clean_to_clean" \
    "Non-CoT clean→clean patching (circuit vs failure-mode test)" \
    bash scripts/run_causal_patching_nocot_clean_to_clean.sh

# -----------------------------------------------------------------------------
# SECTION 2 — Qwen B1 at L=23 (closes 3-model component table)
# -----------------------------------------------------------------------------
SECTION_2_OUT="results/causal_patching/qwen8b_l23_components"
if [[ -d "$SECTION_2_OUT" ]] && [[ -f "$SECTION_2_OUT/SUMMARY_components.md" ]]; then
    log_master ""
    log_master "============================================================"
    log_master "SECTION 2 / SKIPPED: $SECTION_2_OUT already populated"
    log_master "============================================================"
    SECTION_STATUS+=("2|qwen_b1_l23|SKIPPED|0")
else
    run_section 2 "qwen_b1_l23" \
        "Qwen B1 component sweep at L=23 (saturation layer)" \
        python -m experiments.component_patching \
            --enriched-log logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
                           logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
                           logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz \
            --source-bucket clean_check_or_call \
            --target-bucket illegal_fold \
            --layer 23 \
            --components residual attn mlp head \
            --head-indices all \
            --n-source 10 --n-target 30 \
            --seed 42 \
            --out-dir "$SECTION_2_OUT" \
            --device "$DEVICE" --dtype "$DTYPE"
fi

# -----------------------------------------------------------------------------
# SECTION 3 — Reverse-direction component decomposition at Llama L=14
# -----------------------------------------------------------------------------
SECTION_3_OUT="results/causal_patching/llama8b_l14_components_reverse"
if [[ -d "$SECTION_3_OUT" ]] && [[ -f "$SECTION_3_OUT/SUMMARY_components.md" ]]; then
    log_master ""
    log_master "============================================================"
    log_master "SECTION 3 / SKIPPED: $SECTION_3_OUT already populated"
    log_master "============================================================"
    SECTION_STATUS+=("3|llama_reverse_components|SKIPPED|0")
else
    run_section 3 "llama_reverse_components" \
        "Reverse-direction component decomposition at Llama L=14" \
        python -m experiments.component_patching \
            --enriched-log logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
                           logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
                           logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz \
            --source-bucket clean_legal_fold \
            --target-bucket clean_check_or_call \
            --layer 14 \
            --components residual attn mlp head \
            --head-indices all \
            --n-source 10 --n-target 30 \
            --seed 42 \
            --out-dir "$SECTION_3_OUT" \
            --device "$DEVICE" --dtype "$DTYPE"
fi

# -----------------------------------------------------------------------------
# SECTION 4 — Per-seed Llama L=14 components (defends pooled finding)
# -----------------------------------------------------------------------------
run_section 4 "per_seed_llama_l14" \
    "Per-seed Llama L=14 component sweeps (s42 / s123 / s456)" \
    bash scripts/run_per_seed_components.sh

# -----------------------------------------------------------------------------
# SECTION 5 — Additional verb-pair sweeps (fills the 6-pair matrix)
# -----------------------------------------------------------------------------
run_section 5 "more_verb_pairs" \
    "Verb-pair sweeps: RAISE→FOLD and FOLD→RAISE across all 3 models" \
    bash scripts/run_more_verb_pairs.sh

# -----------------------------------------------------------------------------
# SECTION 6 — Cross-temperature Ministral t=0.2 components
# -----------------------------------------------------------------------------
run_section 6 "cross_temp_ministral_t02" \
    "Cross-temperature Ministral t=0.2 s42 component sweep at L=16" \
    bash scripts/run_cross_temp_components.sh

# -----------------------------------------------------------------------------
# SECTION 7 — Direction probe at L* (3 models)
# -----------------------------------------------------------------------------
run_probe_one_model() {
    local short="$1"
    local layer="$2"
    shift 2
    local logs="$@"
    local out_dir="results/direction_probe/${short}8b_l${layer}"
    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already has SUMMARY.md"
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
        --max-decisions-per-bucket "$PROBE_MAX_DECISIONS" \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE"
}

run_section 7 "direction_probe" \
    "Decision-direction linear probe at L* in all 3 models" \
    bash -c '
        set -uo pipefail
        cd '"$(pwd)"'
        echo "--- Llama L=14 ---"
        '"$(declare -f run_probe_one_model)"'
        run_probe_one_model llama 14 \
            logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
        echo "--- Ministral L=16 ---"
        run_probe_one_model ministral 16 \
            logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
        echo "--- Qwen L=23 ---"
        run_probe_one_model qwen 23 \
            logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
        echo "[direction_probe section done]"
    '

# -----------------------------------------------------------------------------
# SECTION 8 — Attention pattern analysis at dominant heads
# -----------------------------------------------------------------------------
run_attn_one_model() {
    local short="$1"
    local layer="$2"
    local heads="$3"
    shift 3
    local logs="$@"
    local out_dir="results/attention_patterns/${short}8b_l${layer}"
    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/SUMMARY.md" ]]; then
        echo "[skip] $out_dir already has SUMMARY.md"
        return 0
    fi
    local first_log
    first_log=$(echo $logs | awk '{print $1}')
    if [[ ! -f "$first_log" ]]; then
        echo "[skip] $short: missing $first_log"
        return 0
    fi
    mkdir -p "$out_dir"
    python -m experiments.attention_patterns_at_dominant_heads \
        --enriched-log $logs \
        --layer "$layer" \
        --heads $heads \
        --max-decisions-per-bucket "$ATTN_PROBE_MAX_DECISIONS" \
        --top-k 8 \
        --out-dir "$out_dir" \
        --device "$DEVICE" --dtype "$DTYPE"
}

run_section 8 "attention_patterns" \
    "Attention-pattern analysis at dominant heads in 2 models (3 if Qwen B1 result available)" \
    bash -c '
        set -uo pipefail
        cd '"$(pwd)"'
        '"$(declare -f run_attn_one_model)"'
        echo "--- Llama L=14, heads 5/23/24 ---"
        run_attn_one_model llama 14 "5 23 24" \
            logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
        echo "--- Ministral L=16, heads 22/9/15 ---"
        run_attn_one_model ministral 16 "22 9 15" \
            logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
            logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
        # Qwen analysis: if section 2 produced a SUMMARY, pull the top-3
        # positive heads from it. Otherwise default to a guess.
        if [ -f "results/causal_patching/qwen8b_l23_components/SUMMARY_components.md" ]; then
            qwen_heads=$(python - <<PY
import re, json
try:
    with open("results/causal_patching/qwen8b_l23_components/summary_components.json") as f:
        d = json.load(f)
    rows = []
    for label, v in d["per_mode"].items():
        if v.get("mode") == "head" and v.get("mean_delta_check_minus_fold") is not None:
            rows.append((label, v["mean_delta_check_minus_fold"], v.get("head_spec")))
    rows.sort(key=lambda r: -(r[1] or 0))
    top3 = [str(r[2]) for r in rows[:3] if r[2] is not None]
    print(" ".join(top3) if top3 else "0 1 2")
except Exception:
    print("0 1 2")
PY
)
            echo "--- Qwen L=23, heads $qwen_heads (top-3 from cell-2 SUMMARY) ---"
            run_attn_one_model qwen 23 "$qwen_heads" \
                logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
                logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
                logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz
        else
            echo "[note] Qwen attention pattern skipped — qwen8b_l23_components/ not yet present."
        fi
        echo "[attention_patterns section done]"
    '

# -----------------------------------------------------------------------------
# Final tally
# -----------------------------------------------------------------------------
log_master ""
log_master "============================================================"
log_master "OVERNIGHT MASTER ORCHESTRATOR — FINAL TALLY"
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
log_master "Output directories to inspect (in priority order):"
log_master "  1. results/causal_patching/llama8b_nocot_clean_to_clean_l14/SUMMARY.md   <- THE big experiment"
log_master "  2. results/causal_patching/qwen8b_nocot_clean_to_clean_l23/SUMMARY.md"
log_master "  3. results/causal_patching/qwen8b_l23_components/SUMMARY_components.md"
log_master "  4. results/causal_patching/llama8b_l14_components_reverse/SUMMARY_components.md"
log_master "  5. results/causal_patching/llama8b_l14_components_{s42,s123,s456}/SUMMARY_components.md"
log_master "  6. results/causal_patching/{model}8b_verbpair_{raise_to_fold,fold_to_raise}/SUMMARY.md"
log_master "  7. results/causal_patching/ministral8b_t02_s42_l16_components/SUMMARY_components.md"
log_master "  8. results/direction_probe/{model}8b_l*/SUMMARY.md"
log_master "  9. results/attention_patterns/{model}8b_l*/SUMMARY.md"
log_master ""
log_master "Master log: $MASTER_LOG"

# Exit code: nonzero if any section failed.
for entry in "${SECTION_STATUS[@]}"; do
    if [[ "$entry" == *"|FAILED|"* ]]; then
        log_master ""
        log_master "[overnight] one or more sections FAILED; check $LOG_DIR/<n>_<name>.log"
        exit 1
    fi
done
log_master ""
log_master "[overnight] all sections completed (some may have skipped — check tally)"
exit 0
