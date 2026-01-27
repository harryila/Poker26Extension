# Research Pipeline: LLM Belief Modeling Experiments

This document outlines the concrete steps for running the LLM belief modeling research experiments using Llama 3.1 70B.

## Overview

The research pipeline answers a key question:

> **Do LLM poker agents actually use betting history to form beliefs, or do they rely only on card information?**

Before running large-scale experiments, we first run a **sanity check** to verify the basic scientific signal is present.

---

## Phase 1: Mini Sanity Grid (First!)

**Purpose:** Verify the LLM is actually using action history, not just card information.

### Step 1A: Run Small Experiment Grid

Run heads-up fixed-limit with beliefs enabled:

```bash
# Llama 3.1 70B, temp 0.0, seed 42
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

# Llama 3.1 70B, temp 0.0, seed 123
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent call \
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
    --opponent call \
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
    --opponent call \
    --hands 50 \
    --seed 123 \
    --temperature 0.2 \
    --elicit-beliefs \
    --out logs/sanity_70b_t02_s123.jsonl \
    -v
```

**Grid summary:**
- Model: Llama 3.1 70B
- Temperatures: 0.0, 0.2
- Seeds: 42, 123
- Hands per condition: 50
- Total: 4 runs × 50 hands = 200 hands

### Step 1B: Enrich Logs with Oracle Posteriors

The dataset builder computes **both** `oracle_card_only` and `oracle_strategy_aware` in a single pass:

```bash
# Enrich all sanity logs (each produces both oracle types)
python -m analysis.build_dataset logs/sanity_70b_t0_s42.jsonl logs/sanity_70b_t0_s42_enriched.jsonl --opponent default
python -m analysis.build_dataset logs/sanity_70b_t0_s123.jsonl logs/sanity_70b_t0_s123_enriched.jsonl --opponent default
python -m analysis.build_dataset logs/sanity_70b_t02_s42.jsonl logs/sanity_70b_t02_s42_enriched.jsonl --opponent default
python -m analysis.build_dataset logs/sanity_70b_t02_s123.jsonl logs/sanity_70b_t02_s123_enriched.jsonl --opponent default
```

Each enriched log will contain:
- `oracle_card_only`: Posterior ignoring betting history (blockers only)
- `oracle_strategy_aware`: Posterior using opponent behavior model

### Step 1C: Compute Sanity Check Metrics

Compute the key comparison:

```
mean JS(LLM, CardOnly) vs mean JS(LLM, StrategyAware)
```

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

print(f"Mean JS(LLM, CardOnly):     {np.mean(js_to_card_only):.4f}")
print(f"Mean JS(LLM, StrategyAware): {np.mean(js_to_strat_aware):.4f}")

if np.mean(js_to_strat_aware) < np.mean(js_to_card_only):
    print("✓ LLM closer to StrategyAware - GREEN LIGHT to scale up!")
else:
    print("✗ LLM closer to CardOnly - investigate prompt/model")
```

**Decision rule:**

| Result | Interpretation | Action |
|--------|----------------|--------|
| LLM closer to **CardOnly** | Model ignores betting history | Investigate prompt, may need revision |
| LLM closer to **StrategyAware** | Model uses action history | **Green light:** scale up |
| Similar distance to both | Ambiguous signal | Run more hands or investigate |

This step is cheap (~200 hands total) and prevents wasting compute on uninformative experiments.

---

## Phase 2: Baseline Dataset Grid (After Sanity Check Passes)

**Only proceed here if Phase 1 shows LLM is closer to StrategyAware.**

### Step 2A: Full Experiment Grid

Scale up to paper-grade dataset:

```bash
# Configuration grid:
# - Model: Llama 3.1 70B
# - Temperatures: 0.0, 0.2
# - Seeds: 42, 123, 456
# - Hands: 1000 per condition (scale to 5000 if stable)

# Example: temp 0.0, seed 42, 1k hands
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent call \
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

```bash
python -m analysis.build_dataset logs/baseline_70b_t0_s42.jsonl logs/baseline_70b_t0_s42_enriched.jsonl --opponent default
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

Repeat enrichment/metrics across different opponent presets:

```bash
# Tight-passive opponent
python -m analysis.build_dataset logs/baseline.jsonl logs/baseline_tight_passive.jsonl --opponent tight_passive

# Loose-aggressive opponent
python -m analysis.build_dataset logs/baseline.jsonl logs/baseline_loose_aggressive.jsonl --opponent loose_aggressive

# ... repeat for other presets ...
```

**Available opponent presets:**
- `default` (balanced)
- `tight_passive`
- `tight_aggressive`
- `loose_passive`
- `loose_aggressive`

### Step 4B: Qualitative Stability

Show that core results are stable across opponent models. Small quantitative differences are expected; large qualitative changes would be concerning.

---

## Quick Reference: Command Flags

### Experiment Flags

| Flag | Description |
|------|-------------|
| `--agent hf` | Use HuggingFace LLM agent |
| `--hf-model MODEL_ID` | Specify model (default: 8B, research: 70B) |
| `--opponent call` | Opponent always calls (simple, interpretable) |
| `--hands N` | Number of hands to play |
| `--seed N` | Random seed for reproducibility |
| `--temperature T` | Generation temperature (0.0 = deterministic) |
| `--elicit-beliefs` | Enable belief elicitation |
| `--out FILE` | Output JSONL path |
| `-v` | Verbose output |

### Dataset Builder Flags

| Flag | Description |
|------|-------------|
| `--opponent MODEL` | Opponent model for posteriors (default, card_only, tight_passive, etc.) |

---

## Sanity Trap to Avoid

Since coherence is now perfect (repair_distance ~0), you might assume beliefs are high-quality. **Don't.**

The critical question isn't "are beliefs valid probability distributions?" but **"are beliefs informed by the right information?"**

A model that outputs perfect probabilities but ignores betting history is scientifically uninteresting.

**Always run the sanity check first.**

---

## File Structure After Running Pipeline

```
logs/
├── sanity_70b_t0_s42.jsonl            # Raw experiment logs
├── sanity_70b_t0_s42_enriched.jsonl   # Enriched (contains both oracle_card_only AND oracle_strategy_aware)
├── sanity_70b_t0_s123.jsonl
├── sanity_70b_t0_s123_enriched.jsonl
├── sanity_70b_t02_s42.jsonl
├── sanity_70b_t02_s42_enriched.jsonl
├── sanity_70b_t02_s123.jsonl
├── sanity_70b_t02_s123_enriched.jsonl
├── baseline_70b_t0_s42.jsonl
├── baseline_70b_t0_s42_enriched.jsonl
└── ...

results/
├── sanity_check_comparison.csv       # CardOnly vs StrategyAware JS distances
├── pce_distribution.csv
├── update_coherence.csv
└── belief_action.csv
```
