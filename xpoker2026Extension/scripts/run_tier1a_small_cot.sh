#!/usr/bin/env bash
# =============================================================================
# Tier 1A.small CoT — full-strength CoT counterpart to run_tier1a_small.sh
# =============================================================================
#
# What this script does (in order, per model):
#   1. Run 6 cells: 3 seeds (42, 123, 456) x 2 temps (0.0, 0.2)
#                   x 100 hands/cell  --- ALL with --cot --capture-logprobs
#   2. Enrich each cell's log with oracle posteriors (StrategyAware + CardOnly)
#   3. PCE distribution + update coherence + analyze_cot on that model's logs
#   4. (Optional) Purge that model's weights to free the HF cache
#
# After all three models:
#   5. Pooled PCE across all 3 models under CoT
#   6. Cross-condition diff vs the existing non-CoT scaled_* baseline ->
#      results/tier1a_small_cot/COMPARISON.md
#      (apples-to-apples 18-cell vs 18-cell: same seeds, temps, opponent,
#      hands/cell -- ONLY difference is --cot)
#
# Models (8B class, parameter-matched):
#   - llama-8b      (meta-llama/Llama-3.1-8B-Instruct)
#   - qwen-8b       (Qwen/Qwen3-8B)            <-- native thinking ALWAYS off
#                                                  (Fix A in hf_agent.py;
#                                                  pre-flight asserts this)
#   - ministral-8b  (mistralai/Ministral-8B-Instruct-2410)
#
# Hand budget: 100 hands/cell x 6 cells x 3 models = 1,800 hands total.
#   Wall-clock estimate: ~6-12 hours on one H100/A100.
#   CoT generation uses larger token budgets (action 64->512, belief 384->768),
#   roughly 2-3x slower per generation than non-CoT.
#
# Outputs (written incrementally as each model completes):
#   logs/cot_<tag>_<temp>_s<seed>_informative_v2.jsonl              (raw)
#   logs/cot_<tag>_<temp>_s<seed>_informative_v2_enriched.jsonl     (enriched)
#   results/tier1a_small_cot/
#       pce_<tag>_records.csv, pce_<tag>_summary.csv
#       uc_<tag>.csv, uc_<tag>_summary.json
#       analyze_cot_<tag>.json
#       pce_pool_records.csv, pce_pool_summary.csv      (cross-model CoT pool)
#       comparison.json                                 (CoT vs no-CoT diff)
#       COMPARISON.md                                   (human-readable verdict)
#
# Failure isolation: if model N crashes, models 1..N-1 results are already on
# disk. Re-running this script will skip any cell whose log already exists.
#
# =============================================================================
# tmux behaviour
# =============================================================================
#
# Recommended workflow:
#   ssh user@gpu-box
#   cd <repo>/xpoker2026Extension
#   bash scripts/run_tier1a_small_cot.sh
#
# When run OUTSIDE tmux, the script creates a tmux session named
# 'poker_tier1a_small_cot' AND attaches you to it.
#   - Press Ctrl-B then D to detach (run continues, exit SSH safely).
#   - Reattach later with:  tmux attach -t poker_tier1a_small_cot
#   - Kill the run with:    tmux kill-session -t poker_tier1a_small_cot
#
# Set NO_TMUX=1 to force foreground execution (e.g. for CI / debugging).
#
# =============================================================================
# Tight-quota disk note (RunPod /workspace MooseFS, /dev/shm tmpfs, etc.)
# =============================================================================
#
# Set PURGE_HF_CACHE_AFTER_MODEL=1 to free each model's weights (~16 GB) after
# its run/enrich/analyze finishes. Required when the HF cache filesystem
# can't fit 3 x 8B model weights (~48 GB) simultaneously.
#
#   PURGE_HF_CACHE_AFTER_MODEL=1 \
#     HF_HOME=/dev/shm/.hf_cache HF_HUB_CACHE=/dev/shm/.hf_cache/hub \
#     HF_HUB_DISABLE_XET=1 \
#     bash scripts/run_tier1a_small_cot.sh
#
# =============================================================================
# Qwen 3 thinking-mode safety check (Fix A guard)
# =============================================================================
#
# Qwen 3's chat template has enable_thinking=True by default. The pilot
# (run_tier1a_small_cot_pilot.sh) deliberately left it on under --cot, which
# turned out to catastrophically break structured belief output (parse rate
# 100% -> 1.3%, see results/tier1a_small_cot_pilot/SUMMARY.md). Fix A in
# poker_env/agents/hf_agent.py decouples native thinking from prompt-level CoT:
#   - If has_thinking_mode is True for the model, enable_thinking is ALWAYS
#     passed as False to the chat template, regardless of cot_mode.
#   - cot_mode controls only the prompt-level REASONING:/JSON: scaffolding.
#
# Before any model loads, this script constructs an HFAgent("qwen-8b",
# cot_mode=True) with a mocked tokenizer/model and asserts that the agent
# config reports enable_thinking=False. If anyone reverts Fix A, this check
# fails and the script aborts BEFORE a single hand is played.
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
    echo "  Press Ctrl-B then D    (detaches; run continues in tmux)"
    echo "  Reattach later with:   tmux attach -t $SESSION_NAME"
    echo "  Abort the run with:    tmux kill-session -t $SESSION_NAME"
    echo
    exec tmux new-session -s "$SESSION_NAME" \
        "NO_TMUX=1 bash '$SCRIPT_PATH'; echo; echo '[tier1a_small_cot finished -- press any key to close window]'; read -n1 -s"
fi

# ---- We're now inside tmux (or NO_TMUX=1). Do the work. ----

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if   [[ -f "venv/bin/activate"   ]]; then source "venv/bin/activate"
    elif [[ -f "../venv/bin/activate" ]]; then source "../venv/bin/activate"
    else echo "WARNING: no venv found at ./venv or ../venv -- using system python"
    fi
fi

LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results/tier1a_small_cot}"
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

# ---- Pre-flight: Fix A guard (qwen-8b native thinking OFF under --cot) ----
echo "=== Pre-flight: Fix A guard (qwen-8b enable_thinking=False under --cot) ==="
python <<'PY'
"""Construct HFAgent("qwen-8b", cot_mode=True) with a mocked tokenizer/model
and assert that get_config() reports enable_thinking=False. If anyone reverts
Fix A in poker_env/agents/hf_agent.py (decoupling native thinking from
cot_mode), this assertion fails and the script aborts before any model loads.
"""
import sys
from unittest.mock import patch, MagicMock

from poker_env.config import MODEL_REGISTRY

cfg = MODEL_REGISTRY.get("qwen-8b")
assert cfg is not None, "qwen-8b not registered in MODEL_REGISTRY"
assert cfg.get("has_thinking_mode") is True, (
    "qwen-8b is missing has_thinking_mode=True in MODEL_REGISTRY. "
    "Without this, HFAgent will NOT pass enable_thinking=False to the chat "
    "template and Qwen 3 will silently perform internal CoT, doubling the "
    "reasoning budget and crushing parse rate (see pilot SUMMARY.md). "
    "Aborting."
)

# Mock tokenizer and model so HFAgent.__init__ doesn't try to download anything.
mock_tok = MagicMock()
mock_tok.pad_token = "<pad>"
mock_tok.eos_token = "<eos>"
mock_model = MagicMock()
mock_model.eval.return_value = mock_model

with patch("poker_env.agents.hf_agent.AutoTokenizer.from_pretrained",
           return_value=mock_tok), \
     patch("poker_env.agents.hf_agent.AutoModelForCausalLM.from_pretrained",
           return_value=mock_model):
    from poker_env.agents.hf_agent import HFAgent
    agent = HFAgent(model_id="qwen-8b", cot_mode=True)

agent_cfg = agent.get_config()
print(f"  qwen-8b registered:   {cfg['model_id']}")
print(f"  cot_mode:             {agent_cfg['cot_mode']}")
print(f"  has_thinking_mode:    {agent_cfg['has_thinking_mode']}")
print(f"  enable_thinking:      {agent_cfg['enable_thinking']}")

assert agent_cfg["cot_mode"] is True, "HFAgent did not register cot_mode=True"
assert agent_cfg["has_thinking_mode"] is True, "agent has_thinking_mode flag mismatch"
assert agent_cfg["enable_thinking"] is False, (
    "FIX A REGRESSION: enable_thinking is not False under cot_mode=True. "
    "Someone has re-coupled native thinking to prompt-level CoT in "
    "poker_env/agents/hf_agent.py. The pilot showed this drops Qwen 3 "
    "parse rate from 100% to 1.3% (see results/tier1a_small_cot_pilot/"
    "SUMMARY.md). Aborting before wasting GPU time."
)
print("  -> Fix A in place; native Qwen thinking decoupled from --cot. OK.")
PY
echo

# ---- Per-model run/enrich/analyze loop ----
# Format: "<short_name>:<filename_tag>:<hub_dirname>"
MODELS=(
    "llama-8b:llama8b:models--meta-llama--Llama-3.1-8B-Instruct"
    "qwen-8b:qwen8b:models--Qwen--Qwen3-8B"
    "ministral-8b:ministral8b:models--mistralai--Ministral-8B-Instruct-2410"
)

# When PURGE_HF_CACHE_AFTER_MODEL=1, delete each model's weights after its
# run/enrich/analyze finishes. Required when the HF cache filesystem has a
# tight quota (e.g. RunPod /workspace MooseFS, /dev/shm tmpfs) and the
# 3 x 8B model weights (~48 GB) won't fit simultaneously.
PURGE_HF_CACHE_AFTER_MODEL="${PURGE_HF_CACHE_AFTER_MODEL:-0}"

run_model() {
    local short="$1"
    local tag="$2"
    local started_at
    started_at="$(date +%s)"

    echo
    echo "############################################################"
    echo "## MODEL: $short  (tag=$tag)  [tier1a_small_cot]"
    echo "## Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "############################################################"

    # ---- Phase 1: run all 6 cells for this model (with --cot) ----
    for seed in "${SEEDS[@]}"; do
        for temp in "${TEMPS[@]}"; do
            local tsuf
            tsuf="$(temp_suffix "$temp")"
            local out="${LOGS_DIR}/cot_${tag}_${tsuf}_s${seed}_${OPPONENT_PRESET}.jsonl"
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
                --elicit-beliefs \
                --cot \
                --capture-logprobs \
                --out "$out" \
                -v
        done
    done

    # ---- Phase 2: enrich logs with oracle posteriors ----
    echo
    echo "  [enrich] $tag"
    shopt -s nullglob
    for f in "${LOGS_DIR}"/cot_${tag}_*_${OPPONENT_PRESET}.jsonl; do
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
    local files=( "${LOGS_DIR}"/cot_${tag}_*_enriched.jsonl )
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
        || echo "    (compute_update_coherence failed for $tag)"

    python -m analysis.analyze_cot \
        "${files[@]}" \
        --json-out "${RESULTS_DIR}/analyze_cot_${tag}.json" \
        || echo "    (analyze_cot failed for $tag)"

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
        # Also purge xet temp/incomplete data so next download starts clean.
        xet_dir="$(dirname "$hub_root")/xet"
        if [[ -d "$xet_dir" ]]; then
            echo "  [purge] freeing ${xet_dir}"
            rm -rf "${xet_dir:?}"/*
        fi
    fi
done

# ---- Phase 5: Pooled PCE across all 3 CoT models ----
echo
echo "############################################################"
echo "## Phase 5: POOLED PCE -- all 8B models under --cot"
echo "############################################################"
shopt -s nullglob
all_files=(
    "${LOGS_DIR}"/cot_llama8b_*_enriched.jsonl
    "${LOGS_DIR}"/cot_qwen8b_*_enriched.jsonl
    "${LOGS_DIR}"/cot_ministral8b_*_enriched.jsonl
)
if [[ "${#all_files[@]}" -gt 0 ]]; then
    python -m analysis.compute_pce_distribution \
        "${all_files[@]}" \
        --output-records "${RESULTS_DIR}/pce_pool_records.csv" \
        --output-summary "${RESULTS_DIR}/pce_pool_summary.csv" \
        --bootstrap 1000 --seed 42 --clustered \
        || echo "  (pooled PCE failed)"
    echo
    echo "Pooled summary: ${RESULTS_DIR}/pce_pool_summary.csv"
fi

# =============================================================================
# Phase 6: Cross-condition diff -- CoT (this run) vs non-CoT scaled_* baseline
# =============================================================================
# Apples-to-apples 18-cell vs 18-cell (same seeds 42/123/456, same temps
# 0.0/0.2, same opponent informative_v2, same 100 hands/cell). Only difference
# is --cot. Reads both .jsonl and .jsonl.gz transparently so it works with
# the gzipped baseline logs from the prior tier1a_small commit.
echo
echo "############################################################"
echo "## Phase 6: CoT vs no-CoT cross-condition comparison"
echo "############################################################"

python <<'PY'
"""Compute per-model CoT vs non-CoT degeneracy diffs across the full 18-cell
grid (3 seeds x 2 temps x 100 hands per cell, per model).

Reads both .jsonl and .jsonl.gz transparently so it pairs cleanly with the
gzipped scaled_* baseline logs from the prior tier1a_small commit.

Writes:
  results/tier1a_small_cot/comparison.json   (machine-readable)
  results/tier1a_small_cot/COMPARISON.md     (human-readable verdict)
"""
import gzip, io, json, glob, os
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results/tier1a_small_cot"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def open_log(path):
    """Open a jsonl or jsonl.gz transparently."""
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r")


def diagnose(file_globs):
    """Return parse rate, trash one-hot rate, action counter, mean reasoning chars."""
    files = []
    for g in file_globs:
        files.extend(glob.glob(g))
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
                        if isinstance(bel, dict) and bel:
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


# Full 18-cell-vs-18-cell pairing: ALL seeds/temps for each model.
COMPARISONS = {
    "llama8b": {
        "no_cot": [
            "logs/scaled_llama8b_*_informative_v2_enriched.jsonl",
            "logs/scaled_llama8b_*_informative_v2_enriched.jsonl.gz",
        ],
        "cot": [
            "logs/cot_llama8b_*_informative_v2_enriched.jsonl",
            "logs/cot_llama8b_*_informative_v2_enriched.jsonl.gz",
        ],
    },
    "qwen8b": {
        "no_cot": [
            "logs/scaled_qwen8b_*_informative_v2_enriched.jsonl",
            "logs/scaled_qwen8b_*_informative_v2_enriched.jsonl.gz",
        ],
        "cot": [
            "logs/cot_qwen8b_*_informative_v2_enriched.jsonl",
            "logs/cot_qwen8b_*_informative_v2_enriched.jsonl.gz",
        ],
    },
    "ministral8b": {
        "no_cot": [
            "logs/scaled_ministral8b_*_informative_v2_enriched.jsonl",
            "logs/scaled_ministral8b_*_informative_v2_enriched.jsonl.gz",
        ],
        "cot": [
            "logs/cot_ministral8b_*_informative_v2_enriched.jsonl",
            "logs/cot_ministral8b_*_informative_v2_enriched.jsonl.gz",
        ],
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
        print(f"{'':<14} |   ^^ NON-TRIVIAL CoT EFFECT")
    else:
        print(f"{'':<14} |   ^^ no rescue -- capability floor confirmed below CoT")
    print()

with open(RESULTS_DIR / "comparison.json", "w") as f:
    json.dump(summary_rows, f, indent=2)

with open(RESULTS_DIR / "COMPARISON.md", "w") as f:
    f.write("# Tier 1A.small CoT vs no-CoT -- 18-cell cross-condition diff\n\n")
    f.write("Same 3 seeds (42, 123, 456), same 2 temps (0.0, 0.2), ")
    f.write("same opponent (informative_v2), same 100 hands/cell.\n")
    f.write("Only difference: --cot flag.\n\n")
    f.write("| Model | Mode | Cells | Decisions | Parse rate | Trash one-hot (of parsed) | Mean reasoning chars |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|\n")
    for tag, modes in summary_rows.items():
        for mode in ("no_cot", "cot"):
            d = modes[mode]
            f.write(f"| {tag} | {mode} | {d['files']} | {d['decisions']} | "
                    f"{d['parse_rate_pct']:.1f}% | {d['trash_one_hot_pct_of_parsed']:.1f}% | "
                    f"{d['mean_reasoning_chars']:.0f} |\n")
        nc, co = modes["no_cot"], modes["cot"]
        parse_delta = co["parse_rate_pct"] - nc["parse_rate_pct"]
        trash_delta = co["trash_one_hot_pct_of_parsed"] - nc["trash_one_hot_pct_of_parsed"]
        f.write(f"| {tag} | **delta** | | | {parse_delta:+.1f}p | {trash_delta:+.1f}p | |\n")

    f.write("\n## Companion artifacts\n\n")
    f.write("- `pce_<tag>_summary.csv` -- clustered-bootstrap CIs on JS distances\n")
    f.write("  to CardOnly and StrategyAware oracles per model.\n")
    f.write("- `uc_<tag>_summary.json` -- update coherence metrics per model.\n")
    f.write("- `analyze_cot_<tag>.json` -- reasoning quality scores +\n")
    f.write("  JS-to-quality correlation per model.\n")
    f.write("- `pce_pool_summary.csv` -- pooled CoT PCE across all 3 models.\n")

print(f"\nWrote: {RESULTS_DIR/'comparison.json'}")
print(f"Wrote: {RESULTS_DIR/'COMPARISON.md'}")
PY

echo
echo "============================================================"
echo "Tier 1A.small CoT COMPLETE."
echo "  Logs:    $LOGS_DIR/cot_{llama8b,qwen8b,ministral8b}_*"
echo "  Results: $RESULTS_DIR/"
echo "  See:     $RESULTS_DIR/COMPARISON.md  (CoT vs no-CoT verdict)"
echo "============================================================"
