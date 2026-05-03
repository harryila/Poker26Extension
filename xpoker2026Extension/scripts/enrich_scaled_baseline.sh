#!/usr/bin/env bash
# Build enriched datasets (oracle posteriors) from the scaled baseline runs.
#
# Reads:  logs/scaled_*.jsonl
# Writes: logs/scaled_*_enriched.jsonl
#
# The opponent preset MUST match the gameplay preset to get valid
# StrategyAware oracle comparisons. We use informative_v2 (the paper's anchor).

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

LOGS_DIR="${LOGS_DIR:-logs}"
OPPONENT_PRESET="informative_v2"

shopt -s nullglob
inputs=("${LOGS_DIR}"/scaled_*.jsonl)
# Drop already-enriched files from the input list.
filtered=()
for f in "${inputs[@]}"; do
    [[ "$f" == *_enriched.jsonl ]] && continue
    filtered+=("$f")
done

if [[ "${#filtered[@]}" -eq 0 ]]; then
    echo "No scaled baseline logs found in ${LOGS_DIR}/. Run run_scaled_baseline.sh first."
    exit 1
fi

for f in "${filtered[@]}"; do
    out="${f%.jsonl}_enriched.jsonl"
    if [[ -f "$out" ]]; then
        echo "[skip] $out already exists"
        continue
    fi
    echo "[enrich] $f -> $out"
    python -m analysis.build_dataset \
        "$f" "$out" \
        --opponent "$OPPONENT_PRESET"
done

echo ""
echo "Done. Next: bash scripts/analyze_scaled_baseline.sh"
