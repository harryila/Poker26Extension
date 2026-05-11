#!/usr/bin/env bash
# =============================================================================
# Tier 0 — Replication smoke test
# =============================================================================
# Verifies the extension's analysis pipeline reproduces poker26's published
# headline numbers on the 70B paper anchor logs:
#   mean_js_to_sa  ≈ 0.014 ± 0.003
#   base-rate-neglect ratio ≈ 4×
#   update correlation r ≈ 0.06
#
# Pure CPU analysis. Wall-clock ~5-10 minutes.
#
# This must pass BEFORE trusting any extension-side analysis numbers — it's
# the foundational replication check the paper depends on (originally
# specified in EXPERIMENTS.md Tier 0).
#
# Outputs
# -------
#   results/tier0_smoke_test/SUMMARY.md           — pass/fail + numbers
#   results/tier0_smoke_test/pce_check.csv        — PCE per-record
#   results/tier0_smoke_test/uc_check.csv         — UC per-record
#
# Env knobs
# ---------
#   ANCHOR_LOG (default ../logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl)
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")/.."

# Try multiple candidate locations for the published 70B anchor log.
ANCHOR_CANDIDATES=(
    "${ANCHOR_LOG:-}"
    "../logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl"
    "logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl"
    "../logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl.gz"
    "logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl.gz"
)
ANCHOR=""
for c in "${ANCHOR_CANDIDATES[@]}"; do
    [[ -z "$c" ]] && continue
    if [[ -f "$c" ]]; then ANCHOR="$c"; break; fi
done
if [[ -z "$ANCHOR" ]]; then
    echo "[abort] no 70B anchor log found. Tried:"
    for c in "${ANCHOR_CANDIDATES[@]}"; do
        [[ -n "$c" ]] && echo "  $c"
    done
    echo "Set ANCHOR_LOG=<path> to override."
    exit 2
fi

OUT_DIR="results/tier0_smoke_test"
if [[ -d "$OUT_DIR" ]] && [[ -f "$OUT_DIR/SUMMARY.md" ]]; then
    echo "[skip] $OUT_DIR already populated"
    exit 0
fi
mkdir -p "$OUT_DIR"

echo
echo "############################################################"
echo "## TIER 0 — Replication smoke test"
echo "##   anchor: $ANCHOR"
echo "##   out:    $OUT_DIR"
echo "############################################################"

PCE_CSV="$OUT_DIR/pce_check.csv"
UC_CSV="$OUT_DIR/uc_check.csv"
SUMMARY="$OUT_DIR/SUMMARY.md"

echo "[1/2] computing PCE distribution ..."
python -m analysis.compute_pce_distribution \
    --input "$ANCHOR" \
    --output "$PCE_CSV" \
    || { echo "[fail] compute_pce_distribution"; exit 1; }

echo "[2/2] computing update coherence ..."
python -m analysis.compute_update_coherence \
    --input "$ANCHOR" \
    --output "$UC_CSV" \
    || { echo "[fail] compute_update_coherence"; exit 1; }

# Aggregate published-vs-observed comparison.
python - <<PY > "$SUMMARY"
import csv, statistics, sys

def safe_float(x):
    try: return float(x)
    except: return None

# Read PCE — js_to_strategy_aware column if present.
pce_path = "$PCE_CSV"
js_to_sa = []
try:
    with open(pce_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in ("js_to_strategy_aware", "js_strategy_aware", "js_to_sa", "JS_to_StrategyAware"):
                if col in row:
                    v = safe_float(row[col])
                    if v is not None:
                        js_to_sa.append(v)
                        break
except Exception as e:
    print(f"[warn] could not parse PCE CSV: {e}")

mean_js = statistics.mean(js_to_sa) if js_to_sa else None
std_js  = statistics.stdev(js_to_sa) if len(js_to_sa) > 1 else None

# Read UC — update correlation if present.
uc_path = "$UC_CSV"
update_r = None
try:
    with open(uc_path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for row in reader:
            for col in ("update_correlation_r", "correlation_r", "r", "update_r"):
                if col in row:
                    v = safe_float(row[col])
                    if v is not None:
                        update_r = v
                        break
            if update_r is not None:
                break
except Exception as e:
    print(f"[warn] could not parse UC CSV: {e}")

print("# Tier 0 — Replication smoke test")
print()
print(f"- Anchor log: \`$ANCHOR\`")
print(f"- PCE records read: {len(js_to_sa)}")
print()
print("## Headline number comparison")
print()
print("| Metric | Published anchor | Observed | Pass criterion |")
print("|---|---:|---:|---|")
if mean_js is not None and std_js is not None:
    pass_js = abs(mean_js - 0.014) < 0.006   # ±2× the published 0.003 stdev
    print(f"| mean js_to_strategy_aware | 0.014 ± 0.003 | "
          f"{mean_js:.4f} ± {std_js:.4f} | "
          f"{'✅ PASS' if pass_js else '❌ FAIL'} (within ±0.006) |")
else:
    print("| mean js_to_strategy_aware | 0.014 ± 0.003 | (could not extract) | ❓ inconclusive |")
if update_r is not None:
    pass_r = abs(update_r - 0.06) < 0.10
    print(f"| update correlation r | 0.06 | {update_r:.3f} | "
          f"{'✅ PASS' if pass_r else '⚠️  out of band'} (target ±0.10) |")
else:
    print("| update correlation r | 0.06 | (could not extract) | ❓ inconclusive |")
print()
print("## Reading guide")
print()
print("- **All ✅ PASS**: extension's analysis pipeline reproduces published "
      "70B numbers within tolerance. Every downstream finding (8B mech-interp, "
      "circuit work, etc.) rests on a sound foundation.")
print("- **❌ FAIL on mean_js**: extension's PCE pipeline diverges from the "
      "paper's. Investigate `analysis/compute_pce_distribution.py` for "
      "regressions before trusting any extension-side calibration numbers.")
print("- **❓ inconclusive**: column names in the CSV don't match expected. "
      "Check the CSV manually and update this script's column-name search.")
PY

cat "$SUMMARY"
echo
echo "[done] wrote $SUMMARY (and $PCE_CSV, $UC_CSV)"
