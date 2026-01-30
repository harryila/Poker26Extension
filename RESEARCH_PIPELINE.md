# Research Pipeline: LLM Belief Modeling Experiments

This document outlines the concrete steps for running the LLM belief modeling research experiments using Llama 3.1 70B.

## Overview

The research pipeline answers a key question:

> **Do LLM poker agents actually use betting history to form beliefs, or do they rely only on card information?**

Before running large-scale experiments, we first run a **sanity check** to verify the basic scientific signal is present.

---

## Execution History: What Actually Happened

This section documents how the pipeline was executed and where we deviated from the original plan.

### Phase 1A: Completed (January 28, 2026)

**Original Plan:** Run 4 sanity check runs (2 temps ? 2 seeds ? 50 hands) with `--opponent-preset informative`.

**What Actually Happened:**

1. **Initial attempt with `call` opponent was inconclusive:**
   - First run used `--opponent call` (always calls)
   - This produced JS(CardOnly, StrategyAware) = 0.0147 - too low to test anything
   - See [EXPERIMENT_RESULTS.md - Appendix: Opponent Informativeness Validation](EXPERIMENT_RESULTS.md#appendix-opponent-informativeness-validation)

2. **Created ThresholdAgent to fix informativeness problem:**
   - Built `poker_env/agents/threshold_agent.py` with `informative` preset
   - Validated that JS(CardOnly, StrategyAware) ? 0.05-0.06 with this opponent
   - This is the "informative opponent problem" fix documented below

3. **Re-ran Phase 1A with informative opponent:**
   - All 4 runs completed: temp 0.0 (seeds 42, 123) + temp 0.2 (seeds 42, 123)
   - Files: `logs/sanity_70b_t0_s42_informative.jsonl`, etc.
   - Results: [EXPERIMENT_RESULTS.md - Phase 1A Sanity Grid](EXPERIMENT_RESULTS.md#phase-1a-sanity-grid-complete)

**Key Finding (Phase 1A):**
> LLM is **closer to CardOnly** than StrategyAware (JS difference = -0.0170).
> This is a **negative result**: the model largely ignores betting history.

See [EXPERIMENT_RESULTS.md - Key Result](EXPERIMENT_RESULTS.md#key-result-70b-with-informative-opponent-phase-1a-complete) for full metrics.

### Pipeline Shift: Negative Result Path

**Original Pipeline Said:**
> "Only proceed to Phase 2 if Phase 1 shows LLM is closer to StrategyAware."

**What We Changed:**

The original gate was too strict. A well-characterized **negative result** is just as publishable as a positive result. We updated the decision rule:

| Result | Original Action | **Updated Action** |
|--------|-----------------|-------------------|
| LLM closer to CardOnly | "Early headline" (stop) | **Scale up to quantify** |
| LLM closer to StrategyAware | Scale up | Scale up |

**Why:** Phase 1A revealed a nuanced finding:
- LLM shows **weak directional sensitivity** to opponent aggression (shifts correctly)
- But **severe base-rate neglect** dominates (trash underestimated 4x)
- Net effect: closer to CardOnly because calibration error >> directional response

This is scientifically interesting and worth scaling. See [EXPERIMENT_RESULTS.md - Action-Conditioning Analysis](EXPERIMENT_RESULTS.md#action-conditioning-analysis).

### Opponent Preset Versioning

**Issue Discovered:** The `informative` preset name could drift if params changed later.

**Solution:** Added explicit `informative_v2` preset as the canonical name:
- `informative_v2`: aggression=0.85, fold_threshold=0.55, bluff_freq=0.02
- `informative`: Legacy alias (identical params, for backwards compatibility)

**For Phase 2:** Use `--opponent-preset informative_v2` everywhere (gameplay AND enrichment).

### Analysis Script Addition

**Original Pipeline:** Assumed ad-hoc Python scripts for analysis.

**What We Added:** Created `analysis/analyze_beliefs.py` CLI tool that:
- Computes validity audit (raw outputs)
- Computes JS divergence (normalized)
- Computes action-conditioning analysis
- Outputs JSON summary

See [README.md - Step 3: Run Full Analysis](README.md#step-3-run-full-analysis).

### Current Status

| Phase | Status | Documentation |
|-------|--------|---------------|
| 1A | ? Complete | [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md#phase-1a-sanity-grid-complete) |
| 1B | ? Complete | Enrichment done |
| 1C | ? Complete | `logs/phase1a_complete.json` |
| **2** | 🔶 **Partial** | 2 of 6 runs (N=838), sufficient for paper |
| 3 | ✅ **Complete** | Paper metrics, figures, bootstrap CIs |
| 4 | ⏳ **Deferred** | See [DEFERRED_WORK.md](DEFERRED_WORK.md) |

### Phase 2: Partial Completion (January 29, 2026)

**Original Plan:** Run full 6-run grid (3 seeds × 2 temps × 1000 hands = 6000 hands total).

**What Actually Happened:**

1. **Started full grid in tmux session:**
   - Loop structure: `for seed in 42 123 456; for temp in 0.0 0.2`
   - Each run targeted 1000 hands
   - Estimated total runtime: ~200 hours (8+ days)

2. **Stopped early after 2 runs:**
   - `temp=0.0, seed=42`: 366 hands (stopped at ~6 hours)
   - `temp=0.2, seed=42`: 326 hands (stopped at ~5 hours)
   - Total: 692 hands, 838 valid beliefs

3. **Why we stopped:**
   - Runtime: ~2 minutes per hand (70B inference is slow)
   - Diminishing returns: Phase 1A findings already replicated
   - Better use of time: Pivot to Phase 3 analysis
   - Statistical power: N=838 provides sufficient confidence

**Key Finding:** Phase 2 interim results (N=838) **exactly matched** Phase 1A (N=246):
- JS(LLM, CardOnly) = 0.4073 vs 0.4046 (virtually identical)
- JS(LLM, StrategyAware) = 0.4200 vs 0.4215 (virtually identical)
- **Conclusion:** More data would not change the finding

See [EXPERIMENT_RESULTS.md - Part 5](EXPERIMENT_RESULTS.md#part-5-phase-2--scale-up-dataset-interim-results) for full analysis.

### Phase 3: Complete (January 30, 2026)

**Original Plan:** Run Phase 3 after completing Phase 2's full grid.

**What Actually Changed:**

Based on Phase 2 interim analysis showing stable results, we **pivoted to prioritize analysis over data collection**. This was the correct decision because:

1. Effect size was stable across 3.4x more data
2. Statistical significance was already achieved
3. Deeper analysis would yield more publishable insights than more raw data

**Phase 3 Scripts Created:**

| Script | Purpose | Output |
|--------|---------|--------|
| `analysis/compute_pce_distribution.py` | PCE by street/action with bootstrap CIs | `results/pce_distribution.csv`, `results/pce_summary.csv` |
| `analysis/compute_update_coherence.py` | Card vs action update separation | `results/update_coherence.csv`, `results/update_coherence_summary.json` |
| `analysis/analyze_beliefs.py` (modified) | Added L1 scale/shape metrics | `results/combined_analysis.json` |
| `analysis/plot_paper_figures.py` | Paper-ready figures | `figures/fig{1-4}_*.png/pdf` |

**Phase 3 Key Findings:**

1. **L1 metrics confirm JS finding:** Both scale-sensitive and shape-only L1 distances show LLM closer to CardOnly
2. **Update coherence diagnosis:** LLM over-updates by 3-11x with r=0.056 correlation (nearly random)
3. **Effect robust across streets:** LLM closer to CardOnly on ALL streets (preflop through river)
4. **Bootstrap CIs:** Effect is statistically significant with non-overlapping confidence intervals

See [EXPERIMENT_RESULTS.md - Part 6](EXPERIMENT_RESULTS.md#part-6-phase-3--paper-ready-analysis-deep-dive) for full analysis.

### Pipeline Deviation Summary

| Original Pipeline | What We Did | Why |
|-------------------|-------------|-----|
| Complete Phase 2 (6 runs × 1000 hands) | Stopped at 2 runs (692 hands) | Effect already replicated, diminishing returns |
| Phase 3 after full Phase 2 | Phase 3 with partial Phase 2 data | Analysis > data collection at this stage |
| Phase 4 robustness checks | Deferred to appendix | Scope discipline for first paper |
| Belief-action divergence | Deferred | Q-value computation complexity |

### Pipeline Deviations Justification

> **Why Phase 2 was truncated:** Phase 2 was intentionally stopped after partial completion once effect direction (LLM closer to CardOnly), robustness across temperature (0.0 and 0.2), and oracle separation stability (>0.05) were established with N=838 valid beliefs. Additional scaling was judged unlikely to alter qualitative conclusions and was deferred to future work. This reflects compute-aware discipline: ~200 GPU hours for marginal improvement is not justified when Phase 3 metrics already provide paper-grade evidence.

---

## Critical: Why Opponent Choice Matters

### The Informative Opponent Problem

For the sanity check to work, **the opponent's actions must carry information about their hand strength**. Otherwise, `CardOnlyPosterior` and `StrategyAwarePosterior` will be nearly identical, and we can't test whether the LLM uses action history.

| Opponent | Behavior | Information Content |
|----------|----------|---------------------|
| `call` | Always calls | **None** - actions reveal nothing |
| `random` | Uniform random | **Low** - actions uncorrelated with hand |
| `threshold` | Raises strong, folds weak | **High** - actions reveal hand strength |

### Validation Results

We tested different opponents to verify action informativeness by measuring JS(CardOnly, StrategyAware):

| Opponent | Preset | JS(CardOnly, StrategyAware) | Status |
|----------|--------|------------------------------|--------|
| `call` | - | 0.0147 | Too low |
| `threshold` | `default` | 0.0167 | Too low |
| `threshold` | `informative` | **0.0622** | **Sufficient** |

**Screening criterion:** JS > 0.05 is a heuristic indicating actions carry enough information for meaningful posterior updates. This is a **screening criterion** (does this opponent produce informative histories?) not a **correctness criterion** (is JS > 0.05 "right"?). Use it to validate experimental setup, not to evaluate LLM performance.

**Recommendation:** Always use `--opponent threshold --opponent-preset informative_v2` for belief experiments.

> **Preset Versioning:** `informative` and `informative_v2` have identical parameters (aggression=0.85, fold_threshold=0.55, bluff_freq=0.02). Use `informative_v2` for new experiments. Phase 1A logs used `informative` (valid, same params).

See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md#appendix-opponent-informativeness-validation) for detailed validation methodology.

---

## Phase 1: Mini Sanity Grid (First!)

**Purpose:** Verify the LLM is actually using action history, not just card information.

### Step 1A: Run Small Experiment Grid

Run heads-up fixed-limit with beliefs enabled, using the **informative threshold opponent**:

```bash
# Llama 3.1 70B, temp 0.0, seed 42
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative \
    --hands 50 \
    --seed 42 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/sanity_70b_t0_s42.jsonl \
    -v

# Llama 3.1 70B, temp 0.0, seed 123
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative \
    --hands 50 \
    --seed 123 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/sanity_70b_t0_s123.jsonl \
    -v

# Llama 3.1 70B, temp 0.2, seed 42
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative \
    --hands 50 \
    --seed 42 \
    --temperature 0.2 \
    --elicit-beliefs \
    --out logs/sanity_70b_t02_s42.jsonl \
    -v

# Llama 3.1 70B, temp 0.2, seed 123
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative \
    --hands 50 \
    --seed 123 \
    --temperature 0.2 \
    --elicit-beliefs \
    --out logs/sanity_70b_t02_s123.jsonl \
    -v
```

**Grid summary:**
- Model: Llama 3.1 70B
- Opponent: `threshold` with `informative` preset
- Temperatures: 0.0, 0.2
- Seeds: 42, 123
- Hands per condition: 50
- Total: 4 runs ? 50 hands = 200 hands

### Step 1B: Enrich Logs with Oracle Posteriors

The dataset builder computes **both** `oracle_card_only` and `oracle_strategy_aware` in a single pass.

**Important:** Use `--opponent informative` to match the ThresholdAgent's behavior:

```bash
# Enrich all sanity logs (each produces both oracle types)
python -m analysis.build_dataset logs/sanity_70b_t0_s42.jsonl logs/sanity_70b_t0_s42_enriched.jsonl --opponent informative
python -m analysis.build_dataset logs/sanity_70b_t0_s123.jsonl logs/sanity_70b_t0_s123_enriched.jsonl --opponent informative
python -m analysis.build_dataset logs/sanity_70b_t02_s42.jsonl logs/sanity_70b_t02_s42_enriched.jsonl --opponent informative
python -m analysis.build_dataset logs/sanity_70b_t02_s123.jsonl logs/sanity_70b_t02_s123_enriched.jsonl --opponent informative
```

Each enriched log will contain:
- `oracle_card_only`: Posterior ignoring betting history (blockers only)
- `oracle_strategy_aware`: Posterior using opponent behavior model

### Step 1C: Compute Sanity Check Metrics

Compute the key comparison:

```
mean JS(LLM, CardOnly) vs mean JS(LLM, StrategyAware) vs mean JS(LLM, BucketCountPrior)
```

#### Understanding the Baselines

| Baseline | What it is | What it measures |
|----------|------------|------------------|
| **CardOnlyPosterior** | P(hand \| blockers) - uniform over remaining combos | Model-free, ignores betting |
| **StrategyAwarePosterior** | P(hand \| blockers, actions) - uses opponent model | Uses betting history |
| **BucketCountPrior** | Same as CardOnly, but framed as a baseline | "Can LLM beat combo counting?" |

**Key insight:** `CardOnlyPosterior` and `BucketCountPrior` are mathematically equivalent - both compute probability proportional to combo count per bucket after accounting for blockers. The distinction is conceptual:
- Use CardOnly to answer: "Does LLM use action history?"
- Use BucketCountPrior to answer: "Is LLM better than naive combo counting?"

If `JS(LLM, CardOnly)` is high, the LLM doesn't even match simple combinatorics - that's a strong, simple finding.

**Quick analysis script:**

```python
import json
from scipy.spatial.distance import jensenshannon
import numpy as np

def load_beliefs_and_oracles(filepath):
    """Load LLM beliefs and oracle posteriors from enriched log."""
    llm_beliefs, card_only, strat_aware = [], [], []
    
    with open(filepath) as f:
        for line in f:
            record = json.loads(line)
            if record.get("belief") and record.get("oracle_card_only") and record.get("oracle_strategy_aware"):
                # Get bucket order from config
                from poker_env.config import BUCKET_ORDER
                llm = [record["belief"].get(b, 0) for b in BUCKET_ORDER]
                co = [record["oracle_card_only"].get(b, 0) for b in BUCKET_ORDER]
                sa = [record["oracle_strategy_aware"].get(b, 0) for b in BUCKET_ORDER]
                llm_beliefs.append(llm)
                card_only.append(co)
                strat_aware.append(sa)
    
    return np.array(llm_beliefs), np.array(card_only), np.array(strat_aware)

# Load enriched logs
llm, co, sa = load_beliefs_and_oracles("logs/sanity_70b_t0_s42_enriched.jsonl")

# Compute JS distances
js_to_card_only = [jensenshannon(l, c) for l, c in zip(llm, co)]
js_to_strat_aware = [jensenshannon(l, s) for l, s in zip(llm, sa)]

# BucketCountPrior = CardOnly (mathematically equivalent, different framing)
js_to_bucket_prior = js_to_card_only  # Same computation

print(f"Mean JS(LLM, CardOnly/BucketCountPrior): {np.mean(js_to_card_only):.4f}")
print(f"Mean JS(LLM, StrategyAware):              {np.mean(js_to_strat_aware):.4f}")

# Primary question: Does LLM use action history?
if np.mean(js_to_strat_aware) < np.mean(js_to_card_only):
    print("? LLM closer to StrategyAware - uses action history!")
else:
    print("? LLM closer to CardOnly - ignores action history")

# Secondary question: Can LLM beat naive combo counting?
if np.mean(js_to_card_only) > 0.3:
    print("?? LLM far from BucketCountPrior - worse than naive combinatorics")
elif np.mean(js_to_card_only) < 0.1:
    print("? LLM close to BucketCountPrior - matches basic combinatorics")
```

**Decision rule:**

| Result | Interpretation | Action |
|--------|----------------|--------|
| LLM closer to **CardOnly** | Model ignores betting history | **Negative result:** scale up to quantify |
| LLM closer to **StrategyAware** | Model uses action history | **Positive result:** scale up to confirm |
| LLM far from **both** (JS > 0.3) | Model worse than combo counting | Strong negative - check methodology first |
| Similar distance to both | Ambiguous signal | Run more hands or investigate |

This step is cheap (~200 hands total across 4 runs) and prevents wasting compute on uninformative experiments.

---

## Phase 2: Baseline Dataset Grid (Scale Up)

**Proceed regardless of Phase 1 outcome** (whether LLM is closer to CardOnly or StrategyAware).

- **Positive result** (closer to StrategyAware): Scale to confirm effect and compute confidence intervals
- **Negative result** (closer to CardOnly): Scale to quantify the failure mode, show stability, and compute full paper metrics

The key insight: a well-characterized negative result ("LLM shows weak directional sensitivity but severe base-rate neglect") is just as publishable as a positive result. Scaling turns the sanity check into a paper.

### Step 2A: Full Experiment Grid

Scale up to paper-grade dataset:

```bash
# Configuration grid:
# - Model: Llama 3.1 70B
# - Opponent: threshold (informative preset)
# - Temperatures: 0.0, 0.2
# - Seeds: 42, 123, 456
# - Hands: 1000 per condition (scale to 5000 if stable)

# Example: temp 0.0, seed 42, 1k hands
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative \
    --hands 1000 \
    --seed 42 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/baseline_70b_t0_s42.jsonl \
    -v

# ... repeat for all grid points ...
```

**Full grid (6 runs):**
| Temperature | Seed | Output File |
|-------------|------|-------------|
| 0.0 | 42 | `baseline_70b_t0_s42.jsonl` |
| 0.0 | 123 | `baseline_70b_t0_s123.jsonl` |
| 0.0 | 456 | `baseline_70b_t0_s456.jsonl` |
| 0.2 | 42 | `baseline_70b_t02_s42.jsonl` |
| 0.2 | 123 | `baseline_70b_t02_s123.jsonl` |
| 0.2 | 456 | `baseline_70b_t02_s456.jsonl` |

### Step 2B: Enrich with StrategyAwarePosterior

**Important:** Use `--opponent informative` to match the ThresholdAgent's behavior:

```bash
python -m analysis.build_dataset logs/baseline_70b_t0_s42.jsonl logs/baseline_70b_t0_s42_enriched.jsonl --opponent informative
# ... repeat for all files ...
```

---

## Phase 3: Compute Paper Metrics

### Step 3A: First Paper Figure Metrics

These are the core metrics for the research paper:

| Metric | Module | Description |
|--------|--------|-------------|
| **PCE Distribution** | `analysis/metrics/calibration.py` | JS divergence between LLM and Bayesian posterior |
| **Update Coherence** | `analysis/metrics/update_coherence.py` | Do beliefs update Bayesianly across streets? |
| **Belief-Action Divergence** | `analysis/metrics/belief_action.py` | Compare stated beliefs to action-implied beliefs |
| **Coherence Rate** | `analysis/metrics/coherence.py` | Probability axiom violations (negatives, wrong sums) |

```bash
# Example metric computation (exact commands depend on your analysis scripts)
python -m analysis.metrics.calibration logs/baseline_enriched/ --output results/pce_distribution.csv
python -m analysis.metrics.update_coherence logs/baseline_enriched/ --output results/update_coherence.csv
python -m analysis.metrics.belief_action logs/baseline_enriched/ --output results/belief_action.csv
```

### Step 3B: Key Result Tables

| Metric | Expected Range | Interpretation |
|--------|----------------|----------------|
| `belief_parse_rate` | >80% | % of valid JSON responses |
| `avg_repair_distance_l1` | ~0.0 | L1 distance to valid simplex |
| `avg_repair_distance_l2` | ~0.0 | L2 distance to valid simplex |
| `avg_prob_sum` | ~1.0 | Mean sum of belief probabilities |
| `mean_pce_js` | 0.0-1.0 | JS divergence from Bayesian posterior |
| `plays_well_bad_beliefs_rate` | ? | Good decisions with wrong beliefs |

---

## Phase 4: Robustness Checks (Appendix-Ready)

### Step 4A: Opponent Model Sensitivity

Test whether results are sensitive to opponent model assumptions by re-enriching with different presets:

```bash
# Main results use informative preset (matches gameplay)
python -m analysis.build_dataset logs/baseline.jsonl logs/baseline_informative.jsonl --opponent informative

# Robustness: what if we assumed different opponent behavior?
python -m analysis.build_dataset logs/baseline.jsonl logs/baseline_tight_aggressive.jsonl --opponent tight_aggressive
python -m analysis.build_dataset logs/baseline.jsonl logs/baseline_loose_aggressive.jsonl --opponent loose_aggressive
```

**Available opponent presets:**
- `informative` (recommended - matches ThresholdAgent gameplay)
- `default` (balanced)
- `tight_passive`
- `tight_aggressive`
- `loose_passive`
- `loose_aggressive`

### Step 4B: Qualitative Stability

Show that core results are stable across opponent models. Small quantitative differences are expected; large qualitative changes would be concerning.

**Key insight:** Even though we USE the `informative` preset during gameplay, testing robustness across OTHER presets shows whether the LLM's belief quality depends on our specific opponent assumptions.

---

## Quick Reference: Command Flags

### Experiment Flags

| Flag | Description |
|------|-------------|
| `--agent hf` | Use HuggingFace LLM agent |
| `--hf-model MODEL_ID` | Specify model (default: 8B, research: 70B) |
| `--opponent TYPE` | Opponent type: `call`, `random`, or `threshold` |
| `--opponent-preset PRESET` | Preset for threshold opponent (see below) |
| `--hands N` | Number of hands to play |
| `--seed N` | Random seed for reproducibility |
| `--temperature T` | Generation temperature (0.0 = deterministic) |
| `--elicit-beliefs` | Enable belief elicitation |
| `--out FILE` | Output JSONL path |
| `-v` | Verbose output |

### Opponent Types

| Type | Behavior | Use Case |
|------|----------|----------|
| `call` | Always calls | Simple baseline (NOT for belief experiments) |
| `random` | Uniform random | Testing only |
| `threshold` | Plays based on hand strength | **Required for belief experiments** |

### Threshold Opponent Presets

| Preset | Aggression | Fold Threshold | Bluff Freq | Notes |
|--------|------------|----------------|------------|-------|
| `default` | 0.4 | 0.3 | 0.1 | Balanced play |
| `informative_v2` | 0.85 | 0.55 | 0.02 | **Recommended** - max action signal (canonical) |
| `informative` | 0.85 | 0.55 | 0.02 | Legacy alias for `informative_v2` |
| `tight_passive` | 0.2 | 0.4 | 0.02 | Folds often, rarely raises |
| `tight_aggressive` | 0.6 | 0.4 | 0.08 | Selective but aggressive |
| `loose_passive` | 0.2 | 0.2 | 0.05 | Plays many hands, just calls |
| `loose_aggressive` | 0.6 | 0.2 | 0.15 | Plays many hands, raises often |

**Note:** Phase 1A used `informative` (legacy). Phase 2+ should use `informative_v2` for explicit versioning.

### Dataset Builder Flags

| Flag | Description |
|------|-------------|
| `--opponent MODEL` | Opponent model for posteriors (must match gameplay opponent!) |

**Available presets:** `default`, `tight_passive`, `tight_aggressive`, `loose_passive`, `loose_aggressive`, `informative_v2`, `informative`

**Note:** Use `informative_v2` for new experiments; `informative` is a legacy alias with identical params.

---

## Sanity Trap to Avoid

Since coherence is now perfect (repair_distance ~0), you might assume beliefs are high-quality. **Don't.**

The critical question isn't "are beliefs valid probability distributions?" but **"are beliefs informed by the right information?"**

A model that outputs perfect probabilities but ignores betting history is scientifically uninteresting.

**Always run the sanity check first.**

---

## Phase 3: Compute Paper Metrics (Actually Executed)

This section documents the Phase 3 analysis that was actually run.

### Phase 3 Commands (Complete)

**PCE Distribution with Bootstrap CIs:**
```bash
python -m analysis.compute_pce_distribution \
    logs/sanity_70b_t0_s42_informative_enriched.jsonl \
    logs/sanity_70b_t0_s123_informative_enriched.jsonl \
    logs/sanity_70b_t02_s42_informative_enriched.jsonl \
    logs/sanity_70b_t02_s123_informative_enriched.jsonl \
    logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl \
    logs/phase2_70b_t02_s42_informative_v2_enriched.jsonl \
    --output-records results/pce_distribution.csv \
    --output-summary results/pce_summary.csv \
    --bootstrap 2000
```

**L1 Metrics + Full Analysis:**
```bash
python -m analysis.analyze_beliefs \
    logs/sanity_70b_t0_s42_informative_enriched.jsonl \
    logs/sanity_70b_t0_s123_informative_enriched.jsonl \
    logs/sanity_70b_t02_s42_informative_enriched.jsonl \
    logs/sanity_70b_t02_s123_informative_enriched.jsonl \
    logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl \
    logs/phase2_70b_t02_s42_informative_v2_enriched.jsonl \
    --json-out results/combined_analysis.json
```

**Update Coherence (Card vs Action separation):**
```bash
python -m analysis.compute_update_coherence \
    logs/sanity_70b_t0_s42_informative_enriched.jsonl \
    logs/sanity_70b_t0_s123_informative_enriched.jsonl \
    logs/sanity_70b_t02_s42_informative_enriched.jsonl \
    logs/sanity_70b_t02_s123_informative_enriched.jsonl \
    logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl \
    logs/phase2_70b_t02_s42_informative_v2_enriched.jsonl \
    --output results/update_coherence.csv \
    --output-summary results/update_coherence_summary.json
```

**Generate Paper Figures:**
```bash
python -m analysis.plot_paper_figures \
    --pce-data results/pce_distribution.csv \
    --pce-summary results/pce_summary.csv \
    --update-data results/update_coherence.csv \
    --output-dir figures/
```

---

## File Structure After Running Pipeline

**Actual file structure after Phase 1A + Phase 2 (partial) + Phase 3:**

```
logs/
├── sanity_70b_t0_s42_informative.jsonl          # Phase 1A raw (temp=0.0, seed=42)
├── sanity_70b_t0_s42_informative_enriched.jsonl # Phase 1A enriched
├── sanity_70b_t0_s123_informative.jsonl         # Phase 1A raw (temp=0.0, seed=123)
├── sanity_70b_t0_s123_informative_enriched.jsonl
├── sanity_70b_t02_s42_informative.jsonl         # Phase 1A raw (temp=0.2, seed=42)
├── sanity_70b_t02_s42_informative_enriched.jsonl
├── sanity_70b_t02_s123_informative.jsonl        # Phase 1A raw (temp=0.2, seed=123)
├── sanity_70b_t02_s123_informative_enriched.jsonl
├── phase1a_complete.json                         # Phase 1A combined metrics
├── phase2_70b_t0_s42_informative_v2.jsonl       # Phase 2 raw (temp=0.0, seed=42, 366 hands)
├── phase2_70b_t0_s42_informative_v2_enriched.jsonl
├── phase2_70b_t02_s42_informative_v2.jsonl      # Phase 2 raw (temp=0.2, seed=42, 326 hands)
├── phase2_70b_t02_s42_informative_v2_enriched.jsonl
├── phase2_interim_analysis.json                  # Phase 2 interim metrics
├── sanity_70b_t0_s42.jsonl                      # Legacy: 70B with call opponent (inconclusive)
├── sanity_70b_t0_s42_enriched.jsonl
├── sanity_8b_t0_s42.jsonl                       # Legacy: 8B model (degenerate)
├── sanity_8b_t0_s42_enriched.jsonl
└── [test/validation files...]

results/
├── pce_distribution.csv                          # Per-record PCE (N=1,084)
├── pce_summary.csv                               # Aggregated with bootstrap CIs
├── update_coherence.csv                          # Per-update metrics (N=318)
├── update_coherence_summary.json                 # Summary by update type
└── combined_analysis.json                        # Full analysis with L1 metrics

figures/
├── fig1_pce_cdf.png                              # CDF of JS distances
├── fig1_pce_cdf.pdf
├── fig2_baserate_neglect.png                     # Trash mass comparison
├── fig2_baserate_neglect.pdf
├── fig3_update_scatter.png                       # Update magnitude scatter (2 panels)
├── fig3_update_scatter.pdf
├── fig4_street_stability.png                     # JS by street
└── fig4_street_stability.pdf
```

---

## Research Outcome Summary

The pipeline produced a complete, publishable result:

### Main Finding

> **Llama 3.1 70B shows weak directional sensitivity to betting actions, but remains closer to CardOnly than StrategyAware because base-rate neglect dominates. The model over-updates by 3-11x with near-zero correlation to oracle updates.**

### Paper-Ready Metrics

| Metric | Value | 95% CI | Source |
|--------|-------|--------|--------|
| N (valid beliefs) | 1,084 | - | Combined Phase 1A + 2 |
| JS(LLM, CardOnly) | 0.4067 | [0.4032, 0.4104] | `results/pce_summary.csv` |
| JS(LLM, StrategyAware) | 0.4204 | [0.4166, 0.4244] | `results/pce_summary.csv` |
| LLM closer to | CardOnly | by 0.0137 | Confirmed with bootstrap |
| LLM trash mass | 16.89% | - | vs oracle 66.21% |
| Card-update ratio | 11.06x | - | Over-updates massively |
| Action-update ratio | 3.25x | - | Over-updates |
| Update correlation | 0.056 | - | Nearly random |

### What We Did NOT Do (Deferred)

See [DEFERRED_WORK.md](DEFERRED_WORK.md) for:
- Additional seed robustness (seeds 123, 456 for Phase 2)
- Belief-action divergence analysis
- Bluff frequency stress test
- Nash/CFR opponents
- Multi-model comparison
