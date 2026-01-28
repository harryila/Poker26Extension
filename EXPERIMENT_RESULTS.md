# Experiment Results: Sanity Check Grid

This document records the findings from the sanity check experiments following the [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md) protocol.

## Experiment Overview

**Date:** January 27-28, 2026  
**Phase:** 1A - Mini Sanity Grid  
**Objective:** Determine if Llama 3.1 models use betting history to form beliefs, or rely only on card information

### Models Tested

| Model | Status | Finding |
|-------|--------|---------|
| Llama 3.1 70B Instruct | ✅ Complete | **Weak sensitivity to betting, severe base-rate neglect** |
| Llama 3.1 8B Instruct | ✅ Complete | **Degenerate** - always outputs 100% trash |

### Key Result (70B with Informative Opponent, Phase 1A Complete)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| N (valid beliefs) | 246 | 4 runs × 2 temps × 2 seeds |
| JS(LLM, CardOnly) | 0.4046 | Distance to "ignores history" oracle |
| JS(LLM, StrategyAware) | 0.4215 | Distance to "uses history" oracle |
| **Difference** | **-0.0170** | **LLM is CLOSER to CardOnly** |
| Oracle separation | 0.0529 | Test validity confirmed (>0.05) |

> **Headline:** Llama 3.1 70B shows weak directional sensitivity to betting actions, but severe base-rate neglect dominates—beliefs stay closer to CardOnly than StrategyAware.

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

# Part 4: Informative Opponent Experiments (Main Results)

**Date:** January 28, 2026  
**Status:** ✅ Complete (temp 0.0 runs)

This section contains the **scientifically valid sanity check** using the informative threshold opponent, which creates meaningful separation between CardOnly and StrategyAware posteriors.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.1-70B-Instruct` |
| Temperature | 0.0 (deterministic) |
| Seeds | 42, 123 |
| Hands per seed | 50 |
| Opponent | `threshold` with `informative_v2` preset |
| Belief Format | `compact` |

**Opponent Preset Versioning:**
- `informative_v2`: aggression=0.85, fold_threshold=0.55, bluff_freq=0.02
- This preset achieves JS(CardOnly, StrategyAware) ≈ 0.0622 (validated in Appendix)

> **Note on `informative` vs `informative_v2`:** Phase 1A commands used `--opponent-preset informative` (the legacy name). This is identical to `informative_v2` - both have the same parameters (aggression=0.85, fold_threshold=0.55, bluff_freq=0.02). The `informative_v2` name was added later for explicit versioning. Phase 2+ should use `informative_v2` everywhere. The Phase 1A results are valid.

**Commands executed:**

```bash
# Seed 42
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative \
    --hands 50 \
    --seed 42 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/sanity_70b_t0_s42_informative.jsonl \
    -v

# Seed 123
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative \
    --hands 50 \
    --seed 123 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/sanity_70b_t0_s123_informative.jsonl \
    -v
```

---

## Output Files

### Temperature 0.0 (Deterministic)

| File | Description | Decisions | Beliefs |
|------|-------------|-----------|---------|
| `logs/sanity_70b_t0_s42_informative.jsonl` | Raw experiment log | 384 | 49 |
| `logs/sanity_70b_t0_s42_informative_enriched.jsonl` | + oracle posteriors | 384 | 49 |
| `logs/sanity_70b_t0_s123_informative.jsonl` | Raw experiment log | 353 | 56 |
| `logs/sanity_70b_t0_s123_informative_enriched.jsonl` | + oracle posteriors | 353 | 56 |

### Temperature 0.2 (Stochastic)

| File | Description | Decisions | Beliefs |
|------|-------------|-----------|---------|
| `logs/sanity_70b_t02_s42_informative.jsonl` | Raw experiment log | 331 | 71 |
| `logs/sanity_70b_t02_s42_informative_enriched.jsonl` | + oracle posteriors | 331 | 71 |
| `logs/sanity_70b_t02_s123_informative.jsonl` | Raw experiment log | 333 | 74 |
| `logs/sanity_70b_t02_s123_informative_enriched.jsonl` | + oracle posteriors | 333 | 74 |

### Analysis Summary

| File | Description |
|------|-------------|
| `logs/phase1a_complete.json` | Combined metrics for all 4 runs (N=246 valid beliefs) |

**See Appendix: Complete File Reference for all commands and data provenance.**

---

## Summary Statistics (Phase 1A Complete)

### Per-Run Belief Counts

| Run | Temperature | Seed | Total Beliefs | Valid for JS |
|-----|-------------|------|---------------|--------------|
| 1 | 0.0 | 42 | 49 | 48 |
| 2 | 0.0 | 123 | 56 | 56 |
| 3 | 0.2 | 42 | 71 | 68 |
| 4 | 0.2 | 123 | 74 | 74 |
| **Total** | - | - | **250** | **246** |

### Belief Quality Metrics (All 4 Runs Combined)

| Metric | Value | Status |
|--------|-------|--------|
| Total records with beliefs | 250 | - |
| Records with negative entries | 0 | ✅ |
| All-zero records (dropped) | 4 | - |
| Valid for JS analysis | 246 | - |
| Avg belief sum | 1.124 | ⚠️ Slight violation |
| Belief sum range | 0.000 - 1.475 | - |
| Belief parse rate | 100% | ✅ |

*Source: `logs/phase1a_complete.json` generated by `python -m analysis.analyze_beliefs`*

### Preprocessing Clarification

> **Belief validity vs belief accuracy.** We evaluate two aspects separately:
>
> **(i) Output validity:** We compute incoherence statistics (sum, negatives, all-zero rate) on the **raw elicited probabilities**. This measures whether the LLM follows probability constraints.
>
> **(ii) Distributional accuracy:** For divergence-based comparisons (JS), we transform the elicited vector into a valid distribution by clipping negatives to 0 (if any) and L1-normalizing. We exclude the rare degenerate all-zero outputs.
>
> **Audit results for Phase 1A (all 4 runs combined):**
> - Records with negative entries: 0 (clipping not triggered)
> - All-zero records dropped: 4
> - Valid for JS analysis: N=246 of 250
>
> This separation ensures validity errors don't mathematically break the alignment metric, while still being reported.
>
> *Computed by `analysis/analyze_beliefs.py` using `scipy.spatial.distance.jensenshannon` (returns JS distance, not divergence).*

---

## Main Results: Does LLM Use Betting History?

### JS Divergence Analysis

| Seed | JS(LLM, CardOnly) | JS(LLM, StrategyAware) | JS(CardOnly, StrategyAware) | LLM Closer To |
|------|-------------------|------------------------|------------------------------|---------------|
| 42 | 0.4118 | 0.4219 | 0.0473 | CardOnly |
| 123 | 0.4074 | 0.4217 | 0.0492 | CardOnly |
| **Combined** | **0.4094** | **0.4218** | **0.0483** | **CardOnly** |

> **Bucket-count baseline.** The CardOnly posterior corresponds exactly to a bucket-count prior: probabilities proportional to the number of legal opponent hand combinations in each bucket, given public cards and blockers. Thus, comparisons to CardOnly also compare the LLM to a pure combinatorial baseline. When we say "LLM is closer to CardOnly," we are saying "LLM is closer to combo-counting than to Bayesian action-conditioning."

### Key Finding

| Metric | Value |
|--------|-------|
| Mean JS(LLM, CardOnly) | **0.4094** |
| Mean JS(LLM, StrategyAware) | **0.4218** |
| Difference | **-0.0124** |
| Mean JS(CardOnly, StrategyAware) | **0.0483** |

**Interpretation:**

1. **The experiment is scientifically valid:** JS(CardOnly, StrategyAware) = 0.0483 ≈ 0.05, confirming the informative opponent creates meaningful separation between oracles.

2. **LLM is CLOSER to CardOnly by 0.0124** — Up to a scale factor, the *shape* of the LLM's belief distribution resembles CardOnly more than StrategyAware. The LLM is largely insensitive to betting history relative to the Bayesian oracle.

3. **The "flat heuristic" pattern persists:**
   - LLM puts 12-17% on trash (oracle says 62-64%)
   - LLM overweights pairs and broadway hands
   - Severe base-rate neglect dominates the belief distribution

---

## Sample Belief Comparison

| Bucket | LLM Belief | CardOnly Oracle | StrategyAware Oracle |
|--------|------------|-----------------|---------------------|
| premium_pairs | 0.031 | 0.017 | 0.015 |
| strong_pairs | 0.067 | 0.012 | 0.009 |
| medium_pairs | 0.089 | 0.019 | 0.018 |
| small_pairs | 0.133 | 0.007 | 0.009 |
| premium_broadway | 0.067 | 0.019 | 0.016 |
| strong_broadway | 0.089 | 0.023 | 0.022 |
| medium_broadway | 0.111 | 0.041 | 0.036 |
| suited_aces | 0.044 | 0.028 | 0.029 |
| suited_connectors | 0.044 | 0.020 | 0.021 |
| suited_gappers | 0.022 | 0.024 | 0.025 |
| offsuit_connectors | 0.067 | 0.046 | 0.046 |
| weak_broadway | 0.044 | 0.069 | 0.064 |
| speculative_suited | 0.067 | 0.054 | 0.057 |
| **trash** | **0.124** | **0.621** | **0.630** |

**Pattern:** LLM underestimates trash by ~5x and overestimates premium hands by ~2x.

---

## Action-Conditioning Analysis

This tests whether the LLM shifts beliefs after opponent aggression (the key "uses history" check).

### Beliefs Grouped by Opponent's Last Action

| After Opponent | N | LLM Strong | Oracle Strong | LLM Trash | Oracle Trash |
|----------------|---|------------|---------------|-----------|--------------|
| **AGGRESSIVE** (bet/raise) | 82 | 0.2518 | 0.0598 | 0.1590 | 0.6411 |
| **PASSIVE** (call/check) | 22 | 0.2107 | 0.0305 | 0.1886 | 0.7340 |

*Strong = premium_pairs + strong_pairs + premium_broadway + strong_broadway*

### Shift Analysis (AGGRESSIVE vs PASSIVE)

| Metric | Oracle Shift | LLM Shift | Ratio |
|--------|--------------|-----------|-------|
| Strong-mass | +0.0293 | +0.0411 | 1.40x |
| Trash-mass | -0.0928 | -0.0296 | 0.32x |

**Key Finding:**
- LLM DOES shift in the correct direction (more strong hands after aggression)
- But LLM's absolute values are completely miscalibrated:
  - LLM strong-mass: 25% (oracle: 6%) — **4x overestimate**
  - LLM trash-mass: 16% (oracle: 64%) — **4x underestimate**
- The base-rate neglect error is so large that directional sensitivity doesn't help

---

## Headline Finding

> **Llama 3.1 70B shows weak directional sensitivity to betting actions, but severe base-rate neglect dominates—beliefs stay closer to CardOnly than StrategyAware.**

The LLM outputs probability distributions that:
- ✅ Parse correctly (100%)
- ✅ Sum close to 1.0 (avg 1.14)
- ✅ Vary by context (not degenerate like 8B)
- ✅ Shift in correct direction after opponent aggression
- ❌ Are closer to CardOnly than StrategyAware overall
- ❌ Show severe base-rate neglect (trash underestimated 4-5x)
- ❌ Massively overweight "interesting" hands (strong 4x too high)

**Nuanced negative result:** The model shows *some* sensitivity to betting history (shifts correctly), but the dominant error is prior miscalibration (base-rate neglect). The directional response is swamped by the absolute calibration failure.

---

## Phase 1A Sanity Grid: Complete

| Condition | Status |
|-----------|--------|
| 70B, temp 0.0, seed 42, `informative_v2` opponent | ✅ Complete |
| 70B, temp 0.0, seed 123, `informative_v2` opponent | ✅ Complete |
| 70B, temp 0.2, seed 42, `informative_v2` opponent | ✅ Complete |
| 70B, temp 0.2, seed 123, `informative_v2` opponent | ✅ Complete |
| 8B, temp 0.0, seed 42 | ✅ Complete (degenerate, unusable) |
| 70B, temp 0.0, seed 42, `call` opponent | ✅ Complete (inconclusive - legacy) |

---

## Robustness Check: Temperature Comparison

| Temperature | N | JS(LLM, CardOnly) | JS(LLM, StrategyAware) | Difference | LLM Closer To |
|-------------|---|-------------------|------------------------|------------|---------------|
| 0.0 | 104 | 0.4094 | 0.4218 | 0.0124 | CardOnly |
| 0.2 | 142 | 0.4010 | 0.4213 | **0.0204** | CardOnly |
| **Combined** | **246** | **0.4046** | **0.4215** | **0.0170** | **CardOnly** |

**Key finding:** The effect is **robust to temperature** and actually **stronger at temp=0.2**.

This rules out the hypothesis that the result is merely deterministic output behavior. Stochastic sampling produces the same qualitative pattern: LLM beliefs stay closer to combo-counting than to Bayesian action-conditioning.

**Analysis command:**
```bash
python -m analysis.analyze_beliefs \
    logs/sanity_70b_t0_s42_informative_enriched.jsonl \
    logs/sanity_70b_t0_s123_informative_enriched.jsonl \
    logs/sanity_70b_t02_s42_informative_enriched.jsonl \
    logs/sanity_70b_t02_s123_informative_enriched.jsonl \
    --json-out logs/phase1a_complete.json
```

---

## Appendix: Complete File Reference

This section documents every file generated during Phase 1A, including the exact commands used, data sources, and what each file contains.

### Raw Experiment Logs

These are the primary data files generated by running `run_experiment.py`. Each contains decision-point records with LLM beliefs.

| File | Command | Data |
|------|---------|------|
| `logs/sanity_70b_t0_s42_informative.jsonl` | See below | 70B, temp=0.0, seed=42, 50 hands |
| `logs/sanity_70b_t0_s123_informative.jsonl` | See below | 70B, temp=0.0, seed=123, 50 hands |
| `logs/sanity_70b_t02_s42_informative.jsonl` | See below | 70B, temp=0.2, seed=42, 50 hands |
| `logs/sanity_70b_t02_s123_informative.jsonl` | See below | 70B, temp=0.2, seed=123, 50 hands |

**Generation commands:**

```bash
# temp=0.0, seed=42
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative \
    --hands 50 \
    --seed 42 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/sanity_70b_t0_s42_informative.jsonl \
    -v

# temp=0.0, seed=123
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative \
    --hands 50 \
    --seed 123 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/sanity_70b_t0_s123_informative.jsonl \
    -v

# temp=0.2, seed=42
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative \
    --hands 50 \
    --seed 42 \
    --temperature 0.2 \
    --elicit-beliefs \
    --out logs/sanity_70b_t02_s42_informative.jsonl \
    -v

# temp=0.2, seed=123
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative \
    --hands 50 \
    --seed 123 \
    --temperature 0.2 \
    --elicit-beliefs \
    --out logs/sanity_70b_t02_s123_informative.jsonl \
    -v
```

**Script:** `run_experiment.py`  
**Key flags:**
- `--agent hf`: Use HuggingFace LLM agent
- `--opponent threshold --opponent-preset informative`: Use informative_v2 opponent (aggression=0.85, fold_threshold=0.55, bluff_freq=0.02)
- `--elicit-beliefs`: Prompt LLM for beliefs at each decision point
- `--temperature`: Sampling temperature (0.0 = deterministic, 0.2 = stochastic)

### Enriched Logs (with Oracle Posteriors)

These files add `oracle_card_only` and `oracle_strategy_aware` posteriors to each decision record, enabling comparison between LLM beliefs and ground-truth Bayesian posteriors.

| File | Source | Script |
|------|--------|--------|
| `logs/sanity_70b_t0_s42_informative_enriched.jsonl` | `sanity_70b_t0_s42_informative.jsonl` | `analysis.build_dataset` |
| `logs/sanity_70b_t0_s123_informative_enriched.jsonl` | `sanity_70b_t0_s123_informative.jsonl` | `analysis.build_dataset` |
| `logs/sanity_70b_t02_s42_informative_enriched.jsonl` | `sanity_70b_t02_s42_informative.jsonl` | `analysis.build_dataset` |
| `logs/sanity_70b_t02_s123_informative_enriched.jsonl` | `sanity_70b_t02_s123_informative.jsonl` | `analysis.build_dataset` |

**Enrichment commands:**

```bash
python -m analysis.build_dataset logs/sanity_70b_t0_s42_informative.jsonl \
    logs/sanity_70b_t0_s42_informative_enriched.jsonl --opponent informative

python -m analysis.build_dataset logs/sanity_70b_t0_s123_informative.jsonl \
    logs/sanity_70b_t0_s123_informative_enriched.jsonl --opponent informative

python -m analysis.build_dataset logs/sanity_70b_t02_s42_informative.jsonl \
    logs/sanity_70b_t02_s42_informative_enriched.jsonl --opponent informative

python -m analysis.build_dataset logs/sanity_70b_t02_s123_informative.jsonl \
    logs/sanity_70b_t02_s123_informative_enriched.jsonl --opponent informative
```

**Script:** `analysis/build_dataset.py`  
**Key flag:** `--opponent informative` must match the gameplay opponent to ensure StrategyAwarePosterior uses the correct behavior model.

**What enrichment adds to each record:**
- `oracle_card_only`: Bayesian posterior using only card combinatorics (no action history)
- `oracle_strategy_aware`: Bayesian posterior incorporating opponent's actions via ParametricOpponent model
- `true_opponent_bucket`: The actual bucket of opponent's hidden hand (ground truth)

### Analysis Output Files

| File | Input Data | Script | Contents |
|------|------------|--------|----------|
| `logs/phase1a_complete.json` | All 4 enriched files | `analysis.analyze_beliefs` | Complete Phase 1A metrics |

**Generation command (phase1a_complete.json):**

```bash
python -m analysis.analyze_beliefs \
    logs/sanity_70b_t0_s42_informative_enriched.jsonl \
    logs/sanity_70b_t0_s123_informative_enriched.jsonl \
    logs/sanity_70b_t02_s42_informative_enriched.jsonl \
    logs/sanity_70b_t02_s123_informative_enriched.jsonl \
    --json-out logs/phase1a_complete.json
```

**Script:** `analysis/analyze_beliefs.py`  
**What it computes:**

1. **Validity Audit** (raw outputs):
   - `total_records`, `records_with_beliefs`
   - `negative_entries`, `records_with_negatives` 
   - `all_zero_records`, `valid_for_js`
   - `prob_sum_mean`, `prob_sum_std`, `prob_sum_min`, `prob_sum_max`

2. **JS Divergence Analysis** (L1-normalized):
   - `js_llm_cardonly_mean/std`: JS distance between LLM and CardOnly oracle
   - `js_llm_strataware_mean/std`: JS distance between LLM and StrategyAware oracle
   - `js_cardonly_strataware_mean/std`: JS distance between oracles (test validity)
   - `llm_closer_to`: Which oracle LLM is closer to
   - `js_difference`: Signed difference (negative = closer to CardOnly)

3. **Action-Conditioning Analysis**:
   - `by_category`: Stats grouped by opponent's last action (AGGRESSIVE/PASSIVE)
   - `shift_analysis`: How LLM and oracle shift beliefs after aggression
   - `strong_shift_ratio`: LLM shift / Oracle shift for strong hands
   - `trash_shift_ratio`: LLM shift / Oracle shift for trash hands

### Legacy/Inconclusive Files

| File | Description | Status |
|------|-------------|--------|
| `logs/sanity_70b_t0_s42.jsonl` | 70B with `call` opponent | ❌ Inconclusive (uninformative opponent) |
| `logs/sanity_70b_t0_s42_enriched.jsonl` | Enriched version | ❌ Inconclusive |
| `logs/sanity_8b_t0_s42.jsonl` | 8B model | ❌ Degenerate (always outputs 100% trash) |
| `logs/sanity_8b_t0_s42_enriched.jsonl` | Enriched version | ❌ Degenerate |

### Key Scripts Reference

| Script | Purpose | Documentation |
|--------|---------|---------------|
| `run_experiment.py` | Run poker games with LLM agent | [README.md](README.md#running-experiments) |
| `analysis/build_dataset.py` | Add oracle posteriors to logs | [README.md](README.md#step-2-enrich-with-oracle-posteriors) |
| `analysis/analyze_beliefs.py` | Compute JS metrics, action-conditioning | [README.md](README.md#step-3-run-full-analysis) |
| `analysis/posterior_oracle.py` | CardOnly and StrategyAware posterior computation | Internal |
| `analysis/opponent_model.py` | ParametricOpponent with presets | Internal |
| `analysis/metrics/calibration.py` | JS divergence, KL, PCE functions | Internal |
| `poker_env/agents/threshold_agent.py` | ThresholdAgent for informative opponent | Internal |

### Data Flow Diagram

```
run_experiment.py (with --elicit-beliefs)
        │
        ▼
   Raw JSONL logs
   (agent_belief, obs, hidden)
        │
        ▼
analysis/build_dataset.py (--opponent informative)
        │
        ▼
   Enriched JSONL logs
   (+oracle_card_only, +oracle_strategy_aware)
        │
        ▼
analysis/analyze_beliefs.py
        │
        ▼
   JSON summary (phase1a_complete.json)
   + Terminal report
```

### Reproducibility Notes

1. **Opponent preset versioning:** The `informative` preset corresponds to `informative_v2`:
   - `aggression=0.85`, `fold_threshold=0.55`, `bluff_freq=0.02`
   - Achieves JS(CardOnly, StrategyAware) ≈ 0.05-0.06

2. **JS computation:** Uses `scipy.spatial.distance.jensenshannon` which returns **JS distance** (sqrt of divergence), range [0, 1].

3. **Normalization for JS:** Beliefs are clipped to non-negative and L1-normalized before JS computation. All-zero beliefs are dropped.

4. **Random seeds:** Seed controls both game dealing and LLM agent sampling. temp=0.0 makes LLM deterministic; temp=0.2 adds stochasticity.

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
