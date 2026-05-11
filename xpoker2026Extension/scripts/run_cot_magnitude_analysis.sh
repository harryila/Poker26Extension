#!/usr/bin/env bash
# CoT vs non-CoT residual magnitude analysis (B2). Pure analysis, ~1 min.
set -uo pipefail
cd "$(dirname "$0")/.."

run_one() {
    local short="$1"
    local layer="$2"
    local cot_npz="results/direction_probe/${short}8b_l${layer}/raw_residuals.npz"
    local nocot_npz="results/direction_probe_nocot/${short}8b_l${layer}/raw_residuals.npz"
    local out_md="results/cot_magnitude_analysis/${short}8b_l${layer}.md"
    if [[ -f "$out_md" ]]; then
        echo "[skip] $out_md exists"; return 0
    fi
    if [[ ! -f "$cot_npz" ]]; then echo "[skip] missing $cot_npz"; return 0; fi
    if [[ ! -f "$nocot_npz" ]]; then echo "[skip] missing $nocot_npz"; return 0; fi
    mkdir -p "$(dirname "$out_md")"
    python -m experiments.cot_magnitude_analysis \
        --cot-npz "$cot_npz" --nocot-npz "$nocot_npz" --out-md "$out_md"
}
run_one llama 14
run_one qwen 23
echo "[done] read results/cot_magnitude_analysis/*.md"
