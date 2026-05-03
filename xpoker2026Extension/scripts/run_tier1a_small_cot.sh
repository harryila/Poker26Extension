#!/usr/bin/env bash
# =============================================================================
# Tier 1A.small.cot — Full small-tier CoT grid at 8B class
# =============================================================================
#
# This is the CoT counterpart to `run_tier1a_small.sh`. Same models, same
# seeds, same temperatures, same opponent, same hand counts — only difference
# is `--cot` is passed to run_experiment.py for every cell.
#
# What this script does (in order, per model):
#   1. Run all 6 cells: 3 seeds (42, 123, 456) x 2 temps (0.0, 0.2) WITH --cot
#   2. Enrich each cell's log with oracle posteriors (StrategyAware + CardOnly)
#   3. Run PCE distribution + update coherence on that model's pooled logs
#   4. Run analyze_cot on that model's enriched logs (CoT-specific metrics)
#   5. Move on to the next model
#   6. Final: cross-condition COMPARISON.md (CoT vs non-CoT degeneracy diff,
#            mirrors the pilot script's diff but on the full 18-cell grid)
#
# Models (8B class, parameter-matched):
#   - llama-8b      (meta-llama/Llama-3.1-8B-Instruct)
#   - qwen-8b       (Qwen/Qwen3-8B)
#   - ministral-8b  (mistralai/Ministral-8B-Instruct-2410)
#
# Hand budget: 100 hands/cell x 6 cells x 3 models = 1,800 hands total.
#   Wall-clock: CoT generation uses ~2-3x more output tokens than direct
#   prompting (action 64->512, belief 384->768), so estimate ~6-12 hours on
#   one H100/A100 for the full grid. Comparable to the non-CoT run with the
#   ~2-3x CoT multiplier.
#
# Outputs (written incrementally as each model completes):
#   logs/cot_full_<tag>_<temp>_s<seed>_informative_v2.jsonl            (raw)
#   logs/cot_full_<tag>_<temp>_s<seed>_informative_v2_enriched.jsonl   (enriched)
#   results/tier1a_small_cot/pce_<tag>_records.csv                     (per-decision)
#   results/tier1a_small_cot/pce_<tag>_summary.csv                     (clustered bootstrap)
#   results/tier1a_small_cot/uc_<tag>.csv + uc_<tag>_summary.json
#   results/tier1a_small_cot/analyze_cot_<tag>.json                    (CoT-specific)
#   results/tier1a_small_cot/pce_pool_summary.csv                      (cross-model pool)
#   results/tier1a_small_cot/COMPARISON.md                             (vs non-CoT baseline)
#
# Failure isolation: re-running this script will skip any cell whose log
# already exists (checks both .jsonl and .jsonl.gz so post-hoc gzipping
# does not trigger re-runs).
#
# =============================================================================
# IMPORTANT: This script depends on the post-bugfix code path (Fix A)
# =============================================================================
#
# The earlier CoT pilot (`run_tier1a_small_cot_pilot.sh`) ran with Qwen 3's
# native thinking mode ON (the old `enable_thinking=cot_mode` coupling),
# which caused the 768-token belief budget to overflow during <think>
# generation and produced a misleading 1.3% parse rate for qwen-8b.
#
# Post-bugfix policy (2026-05-03, see updates.md §7):
# poker_env/agents/hf_agent.py now ALWAYS passes enable_thinking=False for
# thinking-mode models, regardless of cot_mode. Native thinking is
# suppressed in all conditions; only prompt-level CoT (--cot) drives
# reasoning, uniformly across model families.
#
# A pre-flight check below verifies Fix A is in place by inspecting the
# get_config() output of an HFAgent instance (mocked tokenizer/model so
# this works without GPU).
# =============================================================================
#
# tmux behaviour
# =============================================================================
#
# Recommended workflow:
#   ssh user@gpu-box
#   cd <repo>/xpoker2026Extension
#   bash scripts/run_tier1a_small_cot.sh
#
# When run OUTSIDE tmux, the script creates a tmux session named
# 'poker_tier1a_small_cot' AND attaches you to it. The script does NOT
# auto-detach you — that is your call:
#   - Press Ctrl-B then D to detach (run continues in tmux)
#   - Reattach later with:  tmux attach -t poker_tier1a_small_cot
#   - Kill the run with:    tmux kill-session -t poker_tier1a_small_cot
#
# To force foreground execution (e.g. CI), set NO_TMUX=1.
# =============================================================================

set -euo pipefail

SESSION_NAME="poker_tier1a_small_cot"

# ---- Launch inside tmux (attached) if not already there ----
if [[ -z "${TMUX:-}" ]] && [[ -z "${NO_TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux is not installed. Install it (apt: sudo apt install tmux,"
        echo "       brew: brew install tmux), or rerun with NO_TMUX=1 to run in"
        echo "       the foreground (the run will die if your SSH disconnects)."
        exit 1
    fi
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists."
        echo "  Attach:        tmux attach -t $SESSION_NAME"
        echo "  Kill and rerun: tmux kill-session -t $SESSION_NAME && bash $0"
        exit 1
    fi
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    echo "Creating tmux session '$SESSION_NAME' and attaching you to it."
    echo "  When you want to leave the run going in the background:"
    echo "    Press Ctrl-B then D    (detaches you; run continues in tmux)"
    echo "  Reattach later with:   tmux attach -t $SESSION_NAME"
    echo "  Abort the run with:    tmux kill-session -t $SESSION_NAME"
    echo
    exec tmux new-session -s "$SESSION_NAME" \
        "NO_TMUX=1 bash '$SCRIPT_PATH'; echo; echo '[tier1a_small_cot finished — press any key to close window]'; read -n1 -s"
fi

# ---- We're now inside tmux (or NO_TMUX=1 was set). Do the work. ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if   [[ -f "venv/bin/activate"   ]]; then source "venv/bin/activate"
    elif [[ -f "../venv/bin/activate" ]]; then source "../venv/bin/activate"
    else echo "WARNING: no venv found at ./venv or ../venv — using system python"
    fi
fi

LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results/tier1a_small_cot}"
NON_COT_LOGS_PREFIX="${NON_COT_LOGS_PREFIX:-${LOGS_DIR}/scaled}"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

SEEDS=(42 123 456)
TEMPS=(0.0 0.2)
HANDS_PER_CELL="${HANDS_PER_CELL:-100}"
OPPONENT_PRESET="informative_v2"

temp_suffix() {
    case "$1" in
        0.0|0) echo "t0" ;;
        0.2)   echo "t02" ;;
        *)     echo "t${1//./}" ;;
    esac
}

# ---- Pre-flight: verify Fix A is in code, not just in policy notes ----
echo "=== Pre-flight: verifying Fix A (enable_thinking always False under --cot) ==="
python <<'PY'
import sys, unittest.mock as mock
sys.path.insert(0, ".")

import torch
class FakeTok:
    pad_token = None; eos_token = "</s>"; eos_token_id = 1; pad_token_id = 1
    @classmethod
    def from_pretrained(cls, *_, **__): return cls()
    def encode(self, s): return [0]
    def apply_chat_template(self, *_, **__): return "x"
    def decode(self, *_, **__): return ""
    def __call__(self, *_, **__): return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
class FakeMdl:
    device = "cpu"
    @classmethod
    def from_pretrained(cls, *_, **__): return cls()
    def eval(self): return self

with mock.patch("transformers.AutoTokenizer", FakeTok), \
     mock.patch("transformers.AutoModelForCausalLM", FakeMdl):
    from poker_env.agents.hf_agent import HFAgent
    qcfg = HFAgent(model_id="qwen-8b", cot_mode=True).get_config()
    assert qcfg["has_thinking_mode"] is True, "qwen-8b registry flag missing"
    assert qcfg["enable_thinking"] is False, (
        "Fix A NOT applied: HFAgent.get_config() reports "
        f"enable_thinking={qcfg['enable_thinking']} for cot_mode=True. "
        "Update poker_env/agents/hf_agent.py per updates.md §7 before re-running."
    )
    print("  qwen-8b cot_mode=True -> enable_thinking=False  [OK]")
    lcfg = HFAgent(model_id="llama-8b", cot_mode=True).get_config()
    assert lcfg["has_thinking_mode"] is False
    assert lcfg["enable_thinking"] is None
    print("  llama-8b cot_mode=True -> enable_thinking=None  [OK]")
print("  Fix A verified — full small-tier CoT grid is safe to launch.")
PY
echo

# ---- Per-model run/enrich/analyze loop ----
# Format: "<short_name>:<filename_tag>:<hub_dirname>"
MODELS=(
    "llama-8b:llama8b:models--meta-llama--Llama-3.1-8B-Instruct"
    "qwen-8b:qwen8b:models--Qwen--Qwen3-8B"
    "ministral-8b:ministral8b:models--mistralai--Ministral-8B-Instruct-2410"
)

# Rolling-cache mode: free each model's weights after its run/enrich/analyze
# completes. Required when HF cache lives on a tight-quota mount.
PURGE_HF_CACHE_AFTER_MODEL="${PURGE_HF_CACHE_AFTER_MODEL:-0}"

run_model() {
    local short="$1" tag="$2"
    local started_at; started_at="$(date +%s)"

    echo
    echo "############################################################"
    echo "## MODEL: $short  (tag=$tag)  [CoT, full grid]"
    echo "## Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "############################################################"

    # ---- Phase 1: run all 6 cells for this model with --cot ----
    for seed in "${SEEDS[@]}"; do
        for temp in "${TEMPS[@]}"; do
            local tsuf; tsuf="$(temp_suffix "$temp")"
            local out="${LOGS_DIR}/cot_full_${tag}_${tsuf}_s${seed}_${OPPONENT_PRESET}.jsonl"
            if [[ -f "$out" ]] || [[ -f "${out}.gz" ]]; then
                echo "  [skip] $out already exists"
                continue
            fi
            echo "  [run]  $out  (seed=$seed temp=$temp hands=$HANDS_PER_CELL --cot)"
            python run_experiment.py \
                --agent hf \
                --hf-model "$short" \
                --opponent threshold \
                --opponent-preset "$OPPONENT_PRESET" \
                --hands "$HANDS_PER_CELL" \
                --seed "$seed" \
                --temperature "$temp" \
                --elicit-beliefs --cot --capture-logprobs \
                --out "$out" \
                -v
        done
    done

    # ---- Phase 2: enrich logs with oracle posteriors ----
    echo
    echo "  [enrich] $tag"
    shopt -s nullglob
    for f in "${LOGS_DIR}"/cot_full_${tag}_*_${OPPONENT_PRESET}.jsonl; do
        [[ "$f" == *_enriched.jsonl ]] && continue
        local out_enr="${f%.jsonl}_enriched.jsonl"
        if [[ -f "$out_enr" ]] || [[ -f "${out_enr}.gz" ]]; then
            echo "    [skip] $out_enr already exists"
            continue
        fi
        echo "    [enrich] $f -> $out_enr"
        python -m analysis.build_dataset \
            "$f" "$out_enr" \
            --opponent "$OPPONENT_PRESET"
    done

    # ---- Phase 3: per-model analysis (PCE + UC + analyze_cot) ----
    echo
    echo "  [analyze] $tag"
    local files=( "${LOGS_DIR}"/cot_full_${tag}_*_enriched.jsonl )
    if [[ "${#files[@]}" -eq 0 ]]; then
        echo "    [skip] no enriched logs for $tag"
        return
    fi

    python -m analysis.compute_pce_distribution \
        "${files[@]}" \
        --output-records "${RESULTS_DIR}/pce_${tag}_records.csv" \
        --output-summary "${RESULTS_DIR}/pce_${tag}_summary.csv" \
        --bootstrap 1000 --seed 42 --clustered \
        || echo "    (PCE may be meaningless if beliefs degenerate)"

    python -m analysis.compute_update_coherence \
        "${files[@]}" \
        --output "${RESULTS_DIR}/uc_${tag}.csv" \
        --output-summary "${RESULTS_DIR}/uc_${tag}_summary.json" \
        || echo "    (UC may fail if too few belief updates)"

    python -m analysis.analyze_cot \
        "${files[@]}" \
        --json-out "${RESULTS_DIR}/analyze_cot_${tag}.json" \
        || echo "    (analyze_cot failed — pipeline may need adjustment)"

    local elapsed=$(( $(date +%s) - started_at ))
    echo
    echo "## DONE $short in ${elapsed}s"
    echo "## Results: ${RESULTS_DIR}/{pce,uc,analyze_cot}_${tag}.*"
    echo
}

# ---- Run each model in turn ----
for entry in "${MODELS[@]}"; do
    IFS=':' read -r short tag hub_dirname <<< "$entry"
    run_model "$short" "$tag"
    if [[ "$PURGE_HF_CACHE_AFTER_MODEL" == "1" ]]; then
        hub_root="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
        target="${hub_root%/}/${hub_dirname}"
        if [[ -d "$target" ]]; then
            echo "  [purge] freeing ${target}"
            rm -rf "$target"
        fi
        xet_dir="$(dirname "$hub_root")/xet"
        if [[ -d "$xet_dir" ]]; then
            echo "  [purge] freeing ${xet_dir}"
            rm -rf "${xet_dir:?}"/*
        fi
    fi
done

# ---- Final cross-model pool (CoT only) ----
echo
echo "############################################################"
echo "## POOLED ANALYSIS — all 8B models under CoT"
echo "############################################################"
shopt -s nullglob
all_files=(
    "${LOGS_DIR}"/cot_full_llama8b_*_enriched.jsonl
    "${LOGS_DIR}"/cot_full_qwen8b_*_enriched.jsonl
    "${LOGS_DIR}"/cot_full_ministral8b_*_enriched.jsonl
)
if [[ "${#all_files[@]}" -gt 0 ]]; then
    python -m analysis.compute_pce_distribution \
        "${all_files[@]}" \
        --output-records "${RESULTS_DIR}/pce_pool_records.csv" \
        --output-summary "${RESULTS_DIR}/pce_pool_summary.csv" \
        --bootstrap 1000 --seed 42 --clustered \
        || echo "(pooled PCE may be meaningless if beliefs degenerate)"
    echo
    echo "Pooled summary: ${RESULTS_DIR}/pce_pool_summary.csv"
fi

# =============================================================================
# Final phase: cross-condition (CoT vs non-CoT) degeneracy comparison
# =============================================================================
# Mirrors run_tier1a_small_cot_pilot.sh's diff step but on the full 18-cell
# grid against the 18-cell non-CoT baseline already on disk. Writes a
# COMPARISON.md (markdown table) and degeneracy_diff_full.json.
echo
echo "############################################################"
echo "## CROSS-CONDITION COMPARISON: CoT vs non-CoT (full grid)"
echo "############################################################"

NON_COT_LOGS_PREFIX="$NON_COT_LOGS_PREFIX" \
RESULTS_DIR="$RESULTS_DIR" \
LOGS_DIR="$LOGS_DIR" \
python <<'PY'
"""Headline degeneracy diff: full CoT grid vs non-CoT baseline.

Apples-to-apples: same opponent, same seeds, same temps, same hand counts.
Only difference is the --cot flag. Reads enriched logs (handles .jsonl
and .jsonl.gz transparently).
"""
import gzip, io, json, glob, os
from collections import Counter
from pathlib import Path

LOGS_DIR = os.environ.get("LOGS_DIR", "logs")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results/tier1a_small_cot"))
NON_COT_PREFIX = os.environ.get("NON_COT_LOGS_PREFIX", f"{LOGS_DIR}/scaled")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def open_log(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r")

def diagnose(file_globs):
    files = []
    for g in file_globs:
        files.extend(glob.glob(g))
    files = sorted(set(files))
    decisions = belief_attempts = parse_ok = trash_one_hot = 0
    actions = Counter()
    reasoning_chars = []
    for fp in files:
        with open_log(fp) as f:
            for line in f:
                r = json.loads(line)
                if "agent_action" not in r:
                    continue
                decisions += 1
                if isinstance(r["agent_action"], str):
                    actions[r["agent_action"]] += 1
                bm = r.get("belief_metadata") or {}
                if isinstance(bm, dict) and bm:
                    belief_attempts += 1
                    if bm.get("parse_success"):
                        parse_ok += 1
                        bel = r.get("agent_belief")
                        if isinstance(bel, dict):
                            top_name, top_prob = max(bel.items(), key=lambda kv: kv[1])
                            if top_name == "trash" and top_prob >= 0.99:
                                trash_one_hot += 1
                cot = r.get("cot_reasoning")
                if isinstance(cot, str) and cot.strip():
                    reasoning_chars.append(len(cot))
    return {
        "files": len(files),
        "decisions": decisions,
        "belief_attempts": belief_attempts,
        "parse_ok": parse_ok,
        "parse_rate_pct": (parse_ok / belief_attempts * 100) if belief_attempts else 0,
        "trash_one_hot": trash_one_hot,
        "trash_one_hot_pct_of_parsed": (trash_one_hot / parse_ok * 100) if parse_ok else 0,
        "actions": dict(actions.most_common()),
        "n_reasoning_traces": len(reasoning_chars),
        "mean_reasoning_chars": (sum(reasoning_chars) / len(reasoning_chars)) if reasoning_chars else 0,
    }

# Match all seeds and temps (full grid).
COMPARISONS = {
    "llama8b": {
        "no_cot": [f"{NON_COT_PREFIX}_llama8b_*_informative_v2_enriched.jsonl",
                   f"{NON_COT_PREFIX}_llama8b_*_informative_v2_enriched.jsonl.gz"],
        "cot":    [f"{LOGS_DIR}/cot_full_llama8b_*_informative_v2_enriched.jsonl",
                   f"{LOGS_DIR}/cot_full_llama8b_*_informative_v2_enriched.jsonl.gz"],
    },
    "qwen8b": {
        "no_cot": [f"{NON_COT_PREFIX}_qwen8b_*_informative_v2_enriched.jsonl",
                   f"{NON_COT_PREFIX}_qwen8b_*_informative_v2_enriched.jsonl.gz"],
        "cot":    [f"{LOGS_DIR}/cot_full_qwen8b_*_informative_v2_enriched.jsonl",
                   f"{LOGS_DIR}/cot_full_qwen8b_*_informative_v2_enriched.jsonl.gz"],
    },
    "ministral8b": {
        "no_cot": [f"{NON_COT_PREFIX}_ministral8b_*_informative_v2_enriched.jsonl",
                   f"{NON_COT_PREFIX}_ministral8b_*_informative_v2_enriched.jsonl.gz"],
        "cot":    [f"{LOGS_DIR}/cot_full_ministral8b_*_informative_v2_enriched.jsonl",
                   f"{LOGS_DIR}/cot_full_ministral8b_*_informative_v2_enriched.jsonl.gz"],
    },
}

print()
print(f"{'Model':<14} | {'Mode':<7} | {'cells':>5} {'decisions':>10} {'parse_OK%':>10} {'trash1hot%':>11} | {'reasoning_chars(avg)':>22}")
print("-" * 100)
summary_rows = {}
for tag, paths in COMPARISONS.items():
    for mode in ("no_cot", "cot"):
        d = diagnose(paths[mode])
        summary_rows.setdefault(tag, {})[mode] = d
        print(f"{tag:<14} | {mode:<7} | {d['files']:>5} {d['decisions']:>10} {d['parse_rate_pct']:>9.1f}% {d['trash_one_hot_pct_of_parsed']:>10.1f}% | {d['mean_reasoning_chars']:>22.0f}")
    nc = summary_rows[tag]["no_cot"]; co = summary_rows[tag]["cot"]
    parse_delta = co["parse_rate_pct"] - nc["parse_rate_pct"]
    trash_delta = co["trash_one_hot_pct_of_parsed"] - nc["trash_one_hot_pct_of_parsed"]
    print(f"{'':<14} | DELTA   | {'':>5} {'':>10} {parse_delta:+9.1f}p {trash_delta:+10.1f}p")
    if abs(trash_delta) >= 10 or abs(parse_delta) >= 10:
        print(f"{'':<14} |   ^^ NON-TRIVIAL CoT EFFECT — keep this model in the writeup")
    else:
        print(f"{'':<14} |   ^^ no rescue — capability floor below CoT for this model")
    print()

with open(RESULTS_DIR / "degeneracy_diff_full.json", "w") as f:
    json.dump(summary_rows, f, indent=2)

with open(RESULTS_DIR / "COMPARISON.md", "w") as f:
    f.write("# Tier 1A.small CoT — full-grid CoT vs non-CoT comparison\n\n")
    f.write("Same opponent (informative_v2), same seeds {42,123,456}, same temps {0.0, 0.2}, ")
    f.write(f"100 hands/cell. Only difference: `--cot` flag.\n\n")
    f.write("| Model | Mode | Decisions | Parse rate | Trash one-hot (of parsed) | Mean reasoning chars |\n")
    f.write("|---|---|---:|---:|---:|---:|\n")
    for tag, modes in summary_rows.items():
        for mode in ("no_cot","cot"):
            d = modes[mode]
            f.write(f"| {tag} | {mode} | {d['decisions']} | {d['parse_rate_pct']:.1f}% | {d['trash_one_hot_pct_of_parsed']:.1f}% | {d['mean_reasoning_chars']:.0f} |\n")
        nc, co = modes["no_cot"], modes["cot"]
        f.write(f"| {tag} | **delta** | | {co['parse_rate_pct']-nc['parse_rate_pct']:+.1f}p | {co['trash_one_hot_pct_of_parsed']-nc['trash_one_hot_pct_of_parsed']:+.1f}p | |\n")
    f.write("\n_Action distributions and JS distances are in the per-model `analyze_cot_<tag>.json` and `pce_<tag>_summary.csv` files._\n")

print(f"\nWrote: {RESULTS_DIR/'degeneracy_diff_full.json'}")
print(f"Wrote: {RESULTS_DIR/'COMPARISON.md'}")
PY

echo
echo "============================================================"
echo "Tier 1A.small.cot COMPLETE."
echo "  Logs:    $LOGS_DIR/cot_full_{llama8b,qwen8b,ministral8b}_*"
echo "  Results: $RESULTS_DIR/"
echo "  See:     $RESULTS_DIR/COMPARISON.md  (CoT vs non-CoT verdict per model)"
echo "  Compare each model's CoT mean_js_to_sa to the paper anchor 0.014."
echo "============================================================"
