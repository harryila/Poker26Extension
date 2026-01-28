# Experiment Results: Sanity Check Grid

This document records the findings from the sanity check experiments following the [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md) protocol.

## Experiment Overview

**Date:** January 27, 2026  
**Phase:** 1A - Mini Sanity Grid  
**Objective:** Determine if Llama 3.1 models use betting history to form beliefs, or rely only on card information

### Models Tested

| Model | Status | Finding |
|-------|--------|---------|
| Llama 3.1 70B Instruct | ✅ Complete | Distributed beliefs, flat heuristic |
| Llama 3.1 8B Instruct | ✅ Complete | **Degenerate** - always outputs 100% trash |

---

# Part 1: Llama 3.1 70B Analysis

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

---

# Part 2: Llama 3.1 8B Analysis

## Experiment Configuration

### Run 2: Llama 3.1 8B, Temperature 0.0, Seed 42

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
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
    --hf-model meta-llama/Llama-3.1-8B-Instruct \
    --opponent call \
    --hands 50 \
    --seed 42 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/sanity_8b_t0_s42.jsonl \
    -v
```

---

## Output Files

| File | Description | Lines |
|------|-------------|-------|
| `logs/sanity_8b_t0_s42.jsonl` | Raw experiment log with decisions and beliefs | 501 |
| `logs/sanity_8b_t0_s42_enriched.jsonl` | Enriched with oracle posteriors | 501 |

---

## Summary Statistics

### Experiment Metrics

| Metric | Value |
|--------|-------|
| Total hands | 50 |
| Total decisions | 450 |
| Decisions with beliefs (Player 0 only) | 200 |
| Belief parse rate | **100%** |

### Belief Quality Metrics

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Avg `prob_sum` | 1.0000 | 1.0 | ✅ Perfect |
| Min `prob_sum` | 1.0000 | 1.0 | ✅ Perfect |
| Max `prob_sum` | 1.0000 | 1.0 | ✅ Perfect |
| Avg `repair_distance_l1` | 0.0000 | ~0.0 | ✅ Perfect |
| Avg `repair_distance_l2` | 0.0000 | ~0.0 | ✅ Perfect |

**Finding:** The 8B model outputs perfectly coherent probabilities (sum = 1.0 exactly). However, this is because the model outputs a **trivial degenerate distribution** (see below).

---

## Sanity Check Results: CardOnly vs StrategyAware

### Enrichment Command

```bash
python -m analysis.build_dataset \
    logs/sanity_8b_t0_s42.jsonl \
    logs/sanity_8b_t0_s42_enriched.jsonl \
    --opponent default
```

### Results

| Comparison | JS Divergence |
|------------|---------------|
| JS(LLM, CardOnly) | **0.3731** |
| JS(LLM, StrategyAware) | **0.3510** |
| JS(CardOnly, StrategyAware) | **0.0344** |

### Interpretation

1. **LLM appears closer to StrategyAware** (difference = 0.0221)
2. **But this is a statistical artifact** (see below)
3. **LLM is far from both oracles** (JS ~0.35-0.37)

---

## Critical Finding: Degenerate Output

### Belief Pattern Analysis

| Pattern | Count | Percentage |
|---------|-------|------------|
| ALL-IN on "trash" bucket | 200 | **100.0%** |
| Distributed across buckets | 0 | 0.0% |

**The 8B model outputs exactly the same belief for every single decision:**

```json
{"schema": "buckets_14_v1", "probs": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
```

This means: **100% probability on "trash", 0% on all other buckets.**

### Belief Distribution by Street

| Street | Avg Trash Mass | Sample Size |
|--------|----------------|-------------|
| PREFLOP | 1.000 | 50 |
| FLOP | 1.000 | 50 |
| TURN | 1.000 | 50 |
| RIVER | 1.000 | 50 |

**The model outputs identical beliefs regardless of:**
- Street (PREFLOP, FLOP, TURN, RIVER)
- Board cards
- Opponent's betting actions
- Hero's hole cards
- Any game context

### Sample Belief Comparison

| Bucket | LLM Belief | CardOnly Oracle | StrategyAware Oracle |
|--------|------------|-----------------|---------------------|
| premium_pairs | 0.000 | 0.0147 | 0.0133 |
| strong_pairs | 0.000 | 0.0098 | 0.0089 |
| medium_pairs | 0.000 | 0.0171 | 0.0164 |
| small_pairs | 0.000 | 0.0171 | 0.0187 |
| premium_broadway | 0.000 | 0.0163 | 0.0148 |
| strong_broadway | 0.000 | 0.0196 | 0.0179 |
| medium_broadway | 0.000 | 0.0359 | 0.0337 |
| suited_aces | 0.000 | 0.0245 | 0.0230 |
| suited_connectors | 0.000 | 0.0171 | 0.0157 |
| suited_gappers | 0.000 | 0.0204 | 0.0187 |
| offsuit_connectors | 0.000 | 0.0441 | 0.0431 |
| weak_broadway | 0.000 | 0.0653 | 0.0635 |
| speculative_suited | 0.000 | 0.0522 | 0.0517 |
| **trash** | **1.000** | **0.6457** | **0.6605** |

### Why 8B Appears "Closer to StrategyAware"

The JS divergence to StrategyAware (0.3510) is lower than to CardOnly (0.3731) purely because:

1. The true distribution has ~65% trash hands
2. The 8B model says 100% trash
3. StrategyAware has slightly MORE trash mass (66.05%) than CardOnly (64.57%)
4. So 100% trash is mathematically closer to StrategyAware

**This is NOT evidence the model uses action history.** It's an accidental statistical artifact of outputting a fixed degenerate distribution.

---

## 8B Model Conclusions

### Finding 1: Model outputs are completely degenerate

The 8B model does not perform any poker reasoning. It outputs a fixed response (`[0,0,0,0,0,0,0,0,0,0,0,0,0,1]`) for every belief query, regardless of game state.

### Finding 2: Perfect coherence is meaningless

While `prob_sum = 1.0` and `repair_distance = 0.0` suggest perfect constraint-following, this is trivially achieved by outputting a one-hot vector. The model is not "following instructions well" - it's simply not reasoning at all.

### Finding 3: 8B is unsuitable for belief modeling research

The model's output contains no information about:
- Opponent hand ranges
- Bayesian reasoning
- Use of betting history
- Understanding of poker combinatorics

**Research value: None.** The 8B model cannot be used for belief modeling studies.

---

# Part 3: Model Comparison

## Side-by-Side Comparison

| Metric | Llama 3.1 8B | Llama 3.1 70B |
|--------|--------------|---------------|
| **Beliefs analyzed** | 200 | 36 |
| **Avg prob_sum** | 1.0000 ✅ | 1.1480 ⚠️ |
| **Min prob_sum** | 1.0000 | 0.0000 |
| **Max prob_sum** | 1.0000 | 1.2000 |
| **JS(LLM, CardOnly)** | 0.3731 | 0.4056 |
| **JS(LLM, StrategyAware)** | 0.3510 | 0.4133 |
| **Belief pattern** | 100% degenerate | 97% distributed |

## Belief Pattern Comparison

| Pattern | 8B Model | 70B Model |
|---------|----------|-----------|
| ALL-IN one bucket | 200 (100%) | 1 (2.8%) |
| DISTRIBUTED | 0 (0%) | 35 (97.2%) |

## Qualitative Comparison

| Aspect | 8B Model | 70B Model |
|--------|----------|-----------|
| **Reasoning** | ❌ None | ⚠️ Naive heuristics |
| **Constraint following** | ✅ Perfect (trivial) | ⚠️ Imperfect (avg 1.15) |
| **Belief variation** | ❌ Fixed output | ✅ Context-dependent |
| **Combinatorial awareness** | ❌ None | ⚠️ Poor (underweights trash) |
| **Research value** | ❌ Unusable | ✅ Worth studying |

## Key Insight

The comparison reveals a **quality threshold** between model sizes:

- **8B**: Cannot perform belief elicitation at all. Outputs are degenerate.
- **70B**: Performs some reasoning, though naive. Shows distributed beliefs that vary with context.

This suggests belief modeling requires models above a certain capability threshold. The 8B model may lack the capacity for structured probabilistic reasoning over poker hand ranges.

---

## Remaining Sanity Grid

| Condition | Status |
|-----------|--------|
| 70B, temp 0.0, seed 42 | ✅ Complete |
| 70B, temp 0.0, seed 123 | ⏳ Pending |
| 70B, temp 0.2, seed 42 | ⏳ Pending |
| 70B, temp 0.2, seed 123 | ⏳ Pending |
| 8B, temp 0.0, seed 42 | ✅ Complete |
| 8B, temp 0.0, seed 123 | ⏳ Pending (likely not useful) |
| 8B, temp 0.2, seed 42 | ⏳ Pending (likely not useful) |
| 8B, temp 0.2, seed 123 | ⏳ Pending (likely not useful) |

**Recommendation:** Focus remaining experiments on the 70B model. The 8B model's degenerate outputs make further analysis unproductive.

---

## Appendix: Opponent Informativeness Validation

**Note:** This section documents infrastructure validation tests, not LLM belief experiments. These tests verify that the experimental setup produces meaningful data. See [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md#critical-why-opponent-choice-matters) for the full methodology.

### The Problem

The original sanity check used `--opponent call` (always calls). This opponent's actions carry **no information** about their hand strength, making `CardOnlyPosterior` and `StrategyAwarePosterior` nearly identical:

```
JS(CardOnly, StrategyAware) = 0.0147  (with call opponent)
```

With such low separation, we cannot test whether the LLM uses betting history - there's no history signal to use!

### The Solution: ThresholdAgent

We created `ThresholdAgent` (`poker_env/agents/threshold_agent.py`) that plays based on hand strength:
- **Strong hands:** Mostly raise (aggression parameter)
- **Weak hands:** Mostly fold (fold_threshold parameter)
- **Medium hands:** Mix of call/raise

The `informative` preset maximizes action-hand correlation:
- `aggression = 0.85` (raise often with playable hands)
- `fold_threshold = 0.55` (fold most weak hands)
- `bluff_freq = 0.02` (very low bluff rate)

### Validation Results

**Test:** 50 hands, RandomAgent vs ThresholdAgent, various presets

| Opponent | Preset | JS(CardOnly, StrategyAware) | Decisions |
|----------|--------|------------------------------|-----------|
| `call` | - | 0.0147 | ~450 |
| `threshold` | `default` | 0.0167 | 416 |
| `threshold` | `informative` v1 | 0.0369 | 383 |
| `threshold` | `informative` v2 | **0.0622** | 311 |

**Target:** JS > 0.05 means actions carry sufficient information.

### Distribution Analysis (informative v2)

| JS Threshold | Count | Percentage |
|--------------|-------|------------|
| JS > 0.01 | 261/311 | 83.9% |
| JS > 0.05 | 163/311 | 52.4% |
| JS > 0.10 | 64/311 | 20.6% |
| JS > 0.15 | 17/311 | 5.5% |
| JS > 0.20 | 7/311 | 2.3% |

**Conclusion:** The `informative` preset achieves 4.2x improvement over `call` opponent. Over half of decisions now have JS > 0.05, providing a meaningful signal for testing whether LLMs use action history.

### Files Created

| File | Description |
|------|-------------|
| `poker_env/agents/threshold_agent.py` | ThresholdAgent implementation |
| Updated `analysis/opponent_model.py` | Added `informative` preset to ParametricOpponent |
| Updated `run_experiment.py` | Added `--opponent threshold` and `--opponent-preset` flags |
| Updated `analysis/build_dataset.py` | Added `informative` to opponent choices |

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
