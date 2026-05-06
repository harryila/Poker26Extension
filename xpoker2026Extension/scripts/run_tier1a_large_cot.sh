#!/usr/bin/env bash
# =============================================================================
# Tier 1A.large CoT — CoT counterpart to run_tier1a_large.sh (70B class)
# =============================================================================
#
# What this script does (in order, per model):
#   1. Load the model ONCE, run all 6 cells: 3 seeds (42, 123, 456) x 2 temps
#      (0.0, 0.2) with --cot --capture-logprobs, then unload.
#   2. Enrich each cell's log with oracle posteriors (StrategyAware + CardOnly)
#   3. PCE distribution + update coherence + analyze_cot on that model's logs
#   4. Move on to the next model.
#
# After all three models:
#   5. Pooled PCE across all 3 models under CoT
#   6. Cross-condition diff vs existing non-CoT scaled_* baseline
#
# Models (~70B class, parameter-matched within ~3%):
#   - llama-70b      (meta-llama/Llama-3.1-70B-Instruct)   <-- paper anchor
#   - llama-3.3-70b  (meta-llama/Llama-3.3-70B-Instruct)
#   - qwen-72b       (Qwen/Qwen2.5-72B-Instruct)
#
# Hand budget:
#   llama-70b      : 500 hands/cell  (anchor)
#   llama-3.3-70b  : 350 hands/cell
#   qwen-72b       : 350 hands/cell
#   Total: 500*6 + 350*6 + 350*6 = 7,200 hands of 70B CoT compute.
#
# Wall-clock estimate: ~2x non-CoT (CoT uses larger token budgets).
#
# Outputs:
#   logs/cot_<tag>_<temp>_s<seed>_informative_v2.jsonl             (raw)
#   logs/cot_<tag>_<temp>_s<seed>_informative_v2_enriched.jsonl    (enriched)
#   results/tier1a_large_cot/
#       pce_<tag>_records.csv, pce_<tag>_summary.csv
#       uc_<tag>.csv, uc_<tag>_summary.json
#       analyze_cot_<tag>.json
#       pce_pool_records.csv, pce_pool_summary.csv
#       comparison.json, COMPARISON.md
#
# Failure isolation: models 1..N-1 results survive if model N crashes.
# Re-running skips cells whose output files already exist.
#
# =============================================================================
# tmux behaviour
# =============================================================================
#
# Recommended workflow:
#   ssh user@gpu-box
#   cd <repo>/xpoker2026Extension
#   bash scripts/run_tier1a_large_cot.sh
#
# When run OUTSIDE tmux, creates a session 'poker_tier1a_large_cot' and
# attaches you:
#   - Ctrl-B then D to detach (run continues)
#   - tmux attach -t poker_tier1a_large_cot to reattach
#   - tmux kill-session -t poker_tier1a_large_cot to abort
#
# Set NO_TMUX=1 to force foreground execution.
#
# =============================================================================
# Note: 70B inference needs >= 80 GB total VRAM. Models load one at a time
# (sequential) so peak VRAM is one 70B model loaded, not three.
# =============================================================================

set -euo pipefail

SESSION_NAME="poker_tier1a_large_cot"

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
        "NO_TMUX=1 bash '$SCRIPT_PATH'; echo; echo '[tier1a_large_cot finished -- press any key to close window]'; read -n1 -s"
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
RESULTS_DIR="${RESULTS_DIR:-results/tier1a_large_cot}"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

SEEDS=(42 123 456)
TEMPS=(0.0 0.2)
OPPONENT_PRESET="informative_v2"

LLAMA_70B_HANDS="${LLAMA_70B_HANDS:-500}"
CROSS_FAMILY_HANDS="${CROSS_FAMILY_HANDS:-350}"

# ---- Pre-flight: GPU / VRAM check ----
echo "=== Pre-flight: GPU / VRAM check ==="
python <<'PY'
try:
    import torch
    if not torch.cuda.is_available():
        print("WARNING: torch.cuda.is_available() is False. 70B inference needs CUDA.")
    else:
        n = torch.cuda.device_count()
        total_gb = 0.0
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            gb = props.total_memory / (1024**3)
            total_gb += gb
            print(f"  GPU {i}: {props.name}  {gb:.1f} GB")
        print(f"  TOTAL VRAM: {total_gb:.1f} GB across {n} device(s)")
        if total_gb < 80:
            print(f"  WARNING: 70B Instruct typically needs >= 80 GB VRAM.")
except ImportError:
    print("WARNING: torch is not installed; skipping VRAM check.")
PY
echo

# ---- Per-model run/enrich/analyze loop ----
# Format: "<short_name>:<filename_tag>:<hands>"
MODELS=(
    "llama-70b:llama70b:${LLAMA_70B_HANDS}"
    "llama-3.3-70b:llama33-70b:${CROSS_FAMILY_HANDS}"
    "qwen-72b:qwen72b:${CROSS_FAMILY_HANDS}"
)

run_model() {
    local short="$1"
    local tag="$2"
    local hands="$3"
    local started_at
    started_at="$(date +%s)"

    echo
    echo "############################################################"
    echo "## MODEL: $short  (tag=$tag, hands=$hands)  [tier1a_large_cot]"
    echo "## Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "############################################################"

    # ---- Phase 1: run all 6 cells (single process, model loaded once) ----
    local seeds_csv
    seeds_csv="$(IFS=,; echo "${SEEDS[*]}")"
    local temps_csv
    temps_csv="$(IFS=,; echo "${TEMPS[*]}")"
    echo "  [run]  multi-cell: seeds=${seeds_csv} temps=${temps_csv} hands=$hands --cot"
    python run_experiment.py \
        --agent hf \
        --hf-model "$short" \
        --opponent threshold \
        --opponent-preset "$OPPONENT_PRESET" \
        --hands "$hands" \
        --seeds "$seeds_csv" \
        --temps "$temps_csv" \
        --out-dir "$LOGS_DIR" \
        --out-prefix "cot_${tag}" \
        --elicit-beliefs \
        --cot \
        --capture-logprobs \
        -v

    # ---- Phase 2: enrich logs with oracle posteriors ----
    echo
    echo "  [enrich] $tag"
    shopt -s nullglob
    for f in "${LOGS_DIR}"/cot_${tag}_*_${OPPONENT_PRESET}.jsonl; do
        [[ "$f" == *_enriched.jsonl ]] && continue
        local out_enr="${f%.jsonl}_enriched.jsonl"
        if [[ -f "$out_enr" ]]; then
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
        || echo "    (PCE failed for $tag)"

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
    short="${entry%%:*}"
    rest="${entry#*:}"
    tag="${rest%%:*}"
    hands="${rest#*:}"
    run_model "$short" "$tag" "$hands"
done

# ---- Pooled PCE across all 3 CoT models ----
echo
echo "############################################################"
echo "## POOLED PCE -- all 70B models under --cot"
echo "############################################################"
shopt -s nullglob
all_files=(
    "${LOGS_DIR}"/cot_llama70b_*_enriched.jsonl
    "${LOGS_DIR}"/cot_llama33-70b_*_enriched.jsonl
    "${LOGS_DIR}"/cot_qwen72b_*_enriched.jsonl
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

# ---- Cross-condition diff: CoT vs non-CoT ----
echo
echo "############################################################"
echo "## CoT vs no-CoT cross-condition comparison (70B)"
echo "############################################################"

python <<'PY'
import gzip, io, json, glob, os
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results/tier1a_large_cot"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def open_log(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r")


def diagnose(file_globs):
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


COMPARISONS = {
    "llama70b": {
        "no_cot": [
            "logs/scaled_llama70b_*_informative_v2_enriched.jsonl",
            "logs/scaled_llama70b_*_informative_v2_enriched.jsonl.gz",
        ],
        "cot": [
            "logs/cot_llama70b_*_informative_v2_enriched.jsonl",
            "logs/cot_llama70b_*_informative_v2_enriched.jsonl.gz",
        ],
    },
    "llama33-70b": {
        "no_cot": [
            "logs/scaled_llama33-70b_*_informative_v2_enriched.jsonl",
            "logs/scaled_llama33-70b_*_informative_v2_enriched.jsonl.gz",
        ],
        "cot": [
            "logs/cot_llama33-70b_*_informative_v2_enriched.jsonl",
            "logs/cot_llama33-70b_*_informative_v2_enriched.jsonl.gz",
        ],
    },
    "qwen72b": {
        "no_cot": [
            "logs/scaled_qwen72b_*_informative_v2_enriched.jsonl",
            "logs/scaled_qwen72b_*_informative_v2_enriched.jsonl.gz",
        ],
        "cot": [
            "logs/cot_qwen72b_*_informative_v2_enriched.jsonl",
            "logs/cot_qwen72b_*_informative_v2_enriched.jsonl.gz",
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
    print()

with open(RESULTS_DIR / "comparison.json", "w") as f:
    json.dump(summary_rows, f, indent=2)

with open(RESULTS_DIR / "COMPARISON.md", "w") as f:
    f.write("# Tier 1A.large CoT vs no-CoT -- cross-condition diff\n\n")
    f.write("Same 3 seeds (42, 123, 456), same 2 temps (0.0, 0.2), ")
    f.write("same opponent (informative_v2). Only difference: --cot flag.\n\n")
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

print(f"\nWrote: {RESULTS_DIR/'comparison.json'}")
print(f"Wrote: {RESULTS_DIR/'COMPARISON.md'}")
PY

echo
echo "============================================================"
echo "Tier 1A.large CoT COMPLETE."
echo "  Logs:    $LOGS_DIR/cot_{llama70b,llama33-70b,qwen72b}_*"
echo "  Results: $RESULTS_DIR/"
echo "  See:     $RESULTS_DIR/COMPARISON.md  (CoT vs no-CoT verdict)"
echo "============================================================"
