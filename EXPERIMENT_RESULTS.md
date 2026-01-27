# Experiment Results: Sanity Check Grid

This document records the findings from the first sanity check experiment following the [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md) protocol.

## Experiment Overview

**Date:** January 27, 2026  
**Phase:** 1A - Mini Sanity Grid (partial)  
**Objective:** Determine if Llama 3.1 70B uses betting history to form beliefs, or relies only on card information

---

## Experiment Configuration

### Run 1: Llama 3.1 70B, Temperature 0.0, Seed 42

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.1-70B-Instruct` |
| Temperature | 0.0 (deterministic) |
| Seed | 42 |
| Hands | 50 |
| Opponent | `call` (always calls) |
| Belief Format | `compact` |
| Belief Max Tokens | 384 |

**Command executed:**
```bash
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent call \
    --hands 50 \
    --seed 42 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/sanity_70b_t0_s42.jsonl \
    -v
```

**Runtime:** ~45 minutes (70B model inference is slow)

---

## Output Files

| File | Description | Lines |
|------|-------------|-------|
| `logs/sanity_70b_t0_s42.jsonl` | Raw experiment log with decisions and beliefs | 489 |
| `logs/sanity_70b_t0_s42_enriched.jsonl` | Enriched with oracle posteriors | 489 |

---

## Summary Statistics

### Experiment Metrics

| Metric | Value |
|--------|-------|
| Total hands | 50 |
| Total decisions | 488 |
| Decisions with beliefs (Player 0 only) | 36 |
| Belief parse rate | **100%** |

### Belief Quality Metrics

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Avg `prob_sum` | 1.148 | 1.0 | ⚠️ Violation |
| Min `prob_sum` | 0.0 | 1.0 | ❌ Critical |
| Max `prob_sum` | 1.2 | 1.0 | ⚠️ Violation |
| Avg `repair_distance_l1` | 0.204 | ~0.0 | ⚠️ High |
| Avg `repair_distance_l2` | 0.059 | ~0.0 | ⚠️ Moderate |

**Finding:** Despite explicit instructions to sum to 1.0, the LLM outputs probabilities that average 1.148. This indicates a constraint-following failure.

---

## Sanity Check Results: CardOnly vs StrategyAware

The core sanity check compares:
- **JS(LLM, CardOnly)**: How close is LLM to ignoring betting history?
- **JS(LLM, StrategyAware)**: How close is LLM to using betting history?

### Enrichment Command

```bash
python -m analysis.build_dataset \
    logs/sanity_70b_t0_s42.jsonl \
    logs/sanity_70b_t0_s42_enriched.jsonl \
    --opponent default
```

### Results

| Comparison | JS Divergence |
|------------|---------------|
| JS(LLM, CardOnly) | **0.4056** |
| JS(LLM, StrategyAware) | **0.4133** |
| JS(CardOnly, StrategyAware) | **0.0147** |

### Interpretation

1. **LLM is equidistant from both oracles** (difference = -0.0076, negligible)
2. **CardOnly ≈ StrategyAware** (JS = 0.0147, nearly identical)
3. **LLM is far from both** (JS ~0.41, substantial divergence)

**Critical insight:** The `call` opponent provides almost no information via betting history because they call with everything. This makes CardOnly and StrategyAware nearly identical, rendering the sanity check inconclusive for detecting "does LLM use action history."

---

## Sample Belief Analysis

### Preflop Decision Sample

| Bucket | LLM Belief | CardOnly Oracle | StrategyAware Oracle |
|--------|------------|-----------------|---------------------|
| premium_pairs | 0.035 | 0.0147 | 0.0133 |
| strong_pairs | 0.075 | 0.0073 | 0.0067 |
| medium_pairs | 0.100 | 0.0171 | 0.0166 |
| small_pairs | 0.150 | 0.0196 | 0.0212 |
| premium_broadway | 0.075 | 0.0163 | 0.0148 |
| strong_broadway | 0.075 | 0.0188 | 0.0171 |
| medium_broadway | 0.100 | 0.0335 | 0.0314 |
| suited_aces | 0.050 | 0.0253 | 0.0237 |
| suited_connectors | 0.050 | 0.0171 | 0.0157 |
| suited_gappers | 0.050 | 0.0204 | 0.0187 |
| offsuit_connectors | 0.075 | 0.0424 | 0.0416 |
| weak_broadway | 0.075 | 0.0490 | 0.0476 |
| speculative_suited | 0.100 | 0.0547 | 0.0540 |
| **trash** | **0.190** | **0.6637** | **0.6776** |

### Key Observations

1. **Massive underestimation of trash hands**
   - LLM: 19%
   - Oracle: 66-68%
   - This is a 3.5x underestimation

2. **Overestimation of premium hands**
   - LLM: 3.5% for premium_pairs
   - Oracle: 1.4%
   - This is a 2.5x overestimation

3. **"Flat heuristic" pattern**
   - LLM distributes probability more uniformly across buckets
   - Ignores combinatorial reality (there are many more trash combos than premium combos)

---

## Conclusions

### Finding 1: LLM uses a "plausibility heuristic" not Bayesian reasoning

The 70B model outputs beliefs that look reasonable at a glance but are statistically naive. It overweights "interesting" hands (pairs, broadway) and severely underweights the combinatorially dominant trash category.

### Finding 2: Constraint-following is imperfect

Despite explicit instructions ("sum must equal 1.0 exactly"), the model outputs probabilities averaging 1.148. The compact format with schema helps parsing but doesn't guarantee validity.

### Finding 3: Sanity check is inconclusive due to opponent choice

The `call` opponent doesn't provide discriminative betting information. CardOnly and StrategyAware posteriors are nearly identical (JS = 0.0147), so we cannot determine if the LLM uses action history from this experiment.

---

## Recommendations for Next Steps

### Option A: Re-run with informative opponent

Run against `random` opponent where betting actions (fold/raise) carry signal about hand strength:

```bash
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent random \
    --hands 50 \
    --seed 42 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/sanity_70b_t0_s42_random.jsonl \
    -v
```

### Option B: Accept current finding as a result

The finding "LLM produces plausible-looking but statistically naive beliefs" is itself a valid research contribution. The model:
- Parses correctly (100%)
- Fails to follow sum constraint (avg 1.148)
- Uses flat heuristics instead of proper priors
- Shows JS divergence ~0.41 from Bayesian oracles

### Option C: Improve prompting

Add explicit combinatorial guidance to the belief prompt:
- "Note: ~66% of hands are 'trash' due to combinatorics"
- "Premium pairs (AA, KK, QQ) are only ~1.4% of hands"

---

## Remaining Sanity Grid (Not Yet Run)

| Condition | Status |
|-----------|--------|
| 70B, temp 0.0, seed 42 | ✅ Complete |
| 70B, temp 0.0, seed 123 | ⏳ Pending |
| 70B, temp 0.2, seed 42 | ⏳ Pending |
| 70B, temp 0.2, seed 123 | ⏳ Pending |

---

## Appendix: Log File Schema

Each decision record in the JSONL contains:

```json
{
  "hand_id": "04f62ac0",
  "street": "PREFLOP",
  "player_to_act": 0,
  "agent_action": "BET_OR_RAISE",
  "agent_belief": {"premium_pairs": 0.035, ...},
  "belief_metadata": {
    "parse_success": true,
    "prob_sum": 1.2,
    "repair_distance_l1": 0.15,
    "repair_distance_l2": 0.04,
    "belief_format": "compact",
    "belief_schema_id": "buckets_14_v1"
  },
  "oracle_card_only": {"premium_pairs": 0.0147, ...},
  "oracle_strategy_aware": {"premium_pairs": 0.0133, ...}
}
```
