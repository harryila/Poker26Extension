# Deferred Experiments and Extensions

This document enumerates experiments, ablations, and analyses that were discussed during development but are **intentionally deferred**. Each item is categorized by priority and intended placement (main paper, appendix, or future work), with a brief justification.

The goal is to maintain a disciplined scope for the current paper while preserving a clear roadmap for extensions.

---

## Tier 1: High-Value, Optional for This Paper (Appendix Candidates)

These items strengthen robustness or interpretability, but are not required to establish the core claims of the paper. They are good appendix material if time/compute permits.

### 1. Bluff Frequency Stress Test (Opponent Distribution Shift)

**Description:**  
Run the same Phase 1A / Phase 2 setup using an alternative opponent preset identical to `informative_v2` except with a higher bluff frequency (e.g., `bluff_freq = 0.10` instead of `0.02`).

**Purpose:**  
Test whether the LLM's belief updates are brittle to distribution shift—specifically, whether it implicitly equates "aggression ⇒ strength" rather than performing robust Bayesian updating.

**Why This Is Interesting:**
- Distinguishes Bayesian failure from heuristic overfitting
- Tests whether action-conditioning is fragile or principled
- Clean robustness angle reviewers often like

**Why Deferred:**
- Changes the environment distribution (fold rates, reached states)
- Not required to establish base-rate neglect
- Better framed as robustness/appendix

**Suggested Placement:** Appendix (if run) or explicitly listed as future work

**Command (if run):**
```bash
# Would require adding a new preset to opponent_model.py first
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset high_bluff \  # hypothetical preset
    --hands 200 \
    --seed 42 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/ablation_high_bluff.jsonl \
    -v
```

---

### 2. Additional Seed Robustness Beyond Phase 2b

**Description:**  
Run additional short (150–300 hand) experiments with new random seeds beyond the current data.

**Purpose:**  
Demonstrate that the "LLM closer to CardOnly than StrategyAware" result holds across random seeds.

**Why This Is Interesting:**
- Improves confidence that results are not seed-specific
- Reviewer-friendly robustness check

**Why Deferred:**
- Core effect already replicated across multiple runs (seeds 42, 123 in Phase 1A; seed 42 in Phase 2)
- Diminishing scientific returns vs compute cost (~33 hours per 1000-hand run)

**Suggested Placement:** Appendix robustness table (if expanded), otherwise future work

**Command (if run):**
```bash
for seed in 456 789; do
    python run_experiment.py \
        --agent hf \
        --hf-model meta-llama/Llama-3.1-70B-Instruct \
        --opponent threshold \
        --opponent-preset informative_v2 \
        --hands 200 \
        --seed $seed \
        --temperature 0.0 \
        --elicit-beliefs \
        --out logs/robustness_seed${seed}.jsonl \
        -v
done
```

---

## Tier 2: Medium-Value, Conceptually Important but Complex (Likely Follow-Up Paper)

These items are scientifically deep, but substantially increase implementation complexity or paper scope. They are better suited for a follow-up or extension paper.

### 3. Belief–Action Gap (Knowing–Doing Analysis)

**Description:**  
Compare the LLM's stated beliefs with action-implied beliefs, and quantify regret under oracle beliefs.

**Purpose:**  
Test whether the agent:
- Acts well despite holding incorrect beliefs
- Or whether belief errors translate into decision quality degradation

**Why This Is Interesting:**
- Connects to "knowing–doing gap" literature
- Bridges interpretability and performance
- Very strong conceptual contribution if fully developed

**Why Deferred:**
- Requires Q-value estimation or counterfactual evaluation
- Significant engineering and modeling assumptions
- Not necessary to support the paper's main claim about belief formation

**Suggested Placement:** Explicitly listed as future work (potential follow-up paper)

**Existing Infrastructure:**
- `analysis/metrics/belief_action.py` — Has placeholder for belief-action divergence
- `analysis/implied_belief/inverse.py` — Has infrastructure for inferring action-implied beliefs

---

### 4. Street-Wise Counterfactual Belief Updating

**Description:**  
Measure belief update quality separately for:
- Public card reveals (flop/turn/river)
- Opponent betting actions

Including counterfactual "what should have updated" vs "what did update."

**Purpose:**  
Isolate whether belief failures come from:
- Misunderstanding combinatorics
- Ignoring action history
- Or over-weighting private heuristics

**Why This Is Interesting:**
- Fine-grained diagnosis of belief formation
- Strong interpretability contribution

**Why Deferred:**
- Requires careful counterfactual construction
- High analysis complexity
- Not required to establish base-rate neglect

**Note:** Phase 3's `compute_update_coherence.py` provides a **partial** answer to this (separates CARD_REVEAL vs ACTION updates), but full counterfactual analysis would require oracle-expected updates, not just magnitude comparison.

**Suggested Placement:** Future work

---

### 5. Prompt Engineering Interventions

**Description:**  
Test whether explicit combinatorial guidance in the belief prompt improves calibration:
- "Note: ~66% of hands are 'trash' due to combinatorics"
- "Premium pairs (AA, KK, QQ) are only ~1.4% of hands"

**Purpose:**  
Determine whether base-rate neglect is due to:
- Lack of domain knowledge
- Inability to apply known knowledge
- Or fundamental reasoning limitations

**Why This Is Interesting:**
- Directly tests interventions
- Could lead to practical improvements

**Why Deferred:**
- Changes the experimental setup (no longer testing "out of the box" LLM)
- Opens scope to prompt optimization research
- Not required for the diagnostic contribution

**Suggested Placement:** Future work / separate line of research

---

## Tier 3: Low Priority / Out of Scope for Current Line

These are interesting but would substantially alter the research question.

### 6. Nash / CFR / Equilibrium-Based Opponents

**Description:**  
Replace threshold-based opponents with CFR-trained or equilibrium approximations.

**Purpose:**  
Test belief modeling against theoretically optimal play.

**Why This Is Interesting:**
- Strong theoretical grounding
- Appealing to game-theory reviewers

**Why Explicitly Out of Scope:**
- Changes the nature of the oracle (no longer parametric)
- Shifts paper from belief modeling to game-theoretic optimality
- Large increase in infrastructure and explanation burden

**Suggested Placement:** Out of scope for this paper; separate project

---

### 7. Multi-Model Comparison (GPT-4, Claude, etc.)

**Description:**  
Run the same experimental protocol on other frontier LLMs.

**Purpose:**  
Determine whether the findings are Llama-specific or general to LLMs.

**Why This Is Interesting:**
- Broader applicability
- Comparative analysis

**Why Deferred:**
- API costs
- Changes scope from "diagnosis" to "benchmark"
- Not required for the mechanistic contribution

**Suggested Placement:** Future work / separate benchmark paper

---

## Summary Table

| Item | Priority | Placement | Required? |
|------|----------|-----------|-----------|
| Bluff frequency stress test | High | Appendix / robustness | No |
| Extra seed robustness | High | Appendix | No |
| Belief–action gap | Medium | Future work | No |
| Counterfactual street updates | Medium | Future work | No |
| Prompt interventions | Medium | Future work | No |
| Nash/CFR opponents | Low | Out of scope | No |
| Multi-model comparison | Low | Future work | No |

---

## Scope Statement (Suggested Language for Paper)

> This work focuses on **belief formation and calibration** rather than optimal gameplay. Several extensions—such as robustness to bluffing distributions, belief–action divergence, and equilibrium-based opponents—are left to future work. Our goal is to diagnose *how* LLMs form beliefs under uncertainty, not to optimize their strategic performance.

---

## What This Paper Does Claim

For clarity, the current paper makes the following claims, all of which are fully supported by the existing data:

1. **Capability threshold:** 8B is degenerate; 70B shows non-trivial but flawed beliefs
2. **Distributional misalignment:** 70B beliefs are closer to CardOnly (combo-counting) than StrategyAware (Bayesian)
3. **Partial sensitivity:** LLM shifts in correct direction after aggression, but with wrong magnitude
4. **Base-rate neglect:** ~4x underestimate of trash hands, ~5x overestimate of strong hands
5. **Miscalibrated updating:** LLM over-updates by 3-11x with near-zero correlation to oracle updates

These claims are supported by:
- N=1,084 valid beliefs across Phase 1A + Phase 2
- Bootstrap CIs on all key metrics
- L1 metrics confirming robustness to normalization
- Update coherence analysis diagnosing the failure mode
