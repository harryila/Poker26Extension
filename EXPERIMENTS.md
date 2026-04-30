# Recommended Experimental Suite for `xpoker2026Extension`

> **Command given:** "based on poker26's paper (check in iclr26/) and the comments/reviews I pasted earlier, and now the new built/upgraded repo xpoker2026Extension/, what would be the appropriate tests to run, and why, and are they equally representative with the poker26 runs?"
>
> **Date:** April 22, 2026

---

## Ground Truth from the Published Paper

**Paper's design parameters (poker26):**
- Llama 3.1 70B Instruct, heads-up Fixed-Limit Hold'em (PokerKit), blinds 1/2, small bet 2 / big bet 4
- Opponent: `ThresholdAgent` with `informative_v2` preset (aggression=0.85, fold=0.55, bluff=0.02)
- 14-bucket compact belief schema (`buckets_14_v1`)
- **1,084 valid belief elicitations** total

**Two-tier sample structure** (from existing logs):

| Tier | Hands/cell | Seeds | Temperatures | Total hands |
|------|-----------|-------|--------------|-------------|
| sanity | 50 | 42, 123 | 0.0, 0.2 | 4 cells × 50 = 200 |
| phase2 | ~350 | 42 | 0.0, 0.2 | 2 cells × 350 = 692 |
| 8B sanity | 50 | 42 | 0.0 | 1 cell |

**Existing logs in the repo:**

| Log | Hands | Decisions | Parsed beliefs |
|-----|------:|----------:|---------------:|
| `phase2_70b_t0_s42_informative_v2.jsonl`  | 366 | 2,546 | 375 |
| `phase2_70b_t02_s42_informative_v2.jsonl` | 326 | 2,345 | 476 |
| `sanity_70b_t0_s42_informative.jsonl`     |  50 |   384 |  49 |
| `sanity_70b_t0_s123_informative.jsonl`    |  50 |   353 |  56 |
| `sanity_70b_t02_s42_informative.jsonl`    |  50 |   331 |  63 |
| `sanity_70b_t02_s123_informative.jsonl`   |  50 |   333 |  82 |
| `sanity_8b_t0_s42.jsonl`                  |  50 |   450 | 200 |

To stay **equally representative** for the paper-replication parts of the new suite, every new condition that compares against the published numbers must use the same env/opponent/seed/temp/hand-count grid and the same statistical machinery (hand-clustered bootstrap, parsed-only filter).

---

## Reviewer Concerns → Test Tier Mapping

| Reviewer concern | # reviewers | Test tier |
|------------------|------------:|-----------|
| Try Chain-of-Thought prompting | 3 | Tier 1 |
| Test other model families (GPT, Claude, Qwen, Mistral, Gemini) | 3 | Tier 2 |
| Address scaling / emergence (8B → 70B → frontier) | 1 | Tier 3 |
| Opponent-model uncertainty / mismatch | 1 | Tier 4 |
| Internal vs verbalized calibration (logprobs / interp) | 2 | Tier 5 |
| Equity threshold isn't EV (pot-odds analysis) | 1 | Tier 6 |
| Soften general claims (writing only) | 2 | (writing change) |
| Strengthen statistical validity (clustered bootstraps) | 1 | Already in base |

---

## Test Tiers

### Tier 0 — Replication smoke test (NOT optional; do this first)

**Why:** The extension touched 15+ analysis/env files that were previously identical to base. Before trusting any new finding, confirm the extension reproduces poker26's headline numbers when run on poker26's exact logs.

**What:** Run the extension's analysis pipeline on the published `phase2_70b_t0_s42_informative_v2_enriched.jsonl` and verify:
- `mean_js_to_sa = 0.014 ± 0.003`
- Base-rate-neglect ratio ≈ 4×
- Update correlation r ≈ 0.06

**Commands:**

```bash
cd xpoker2026Extension
python -m analysis.compute_pce_distribution \
  --input ../logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl \
  --output /tmp/pce_check.csv
python -m analysis.compute_update_coherence \
  --input ../logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl \
  --output /tmp/uc_check.csv
```

**Pass criterion:** Numbers within ±2× the published bootstrap CI. If not, the extension's modified analysis is producing different metrics — investigate before running anything else.

---

### Tier 1 — CoT replication (THE primary reviewer ask)

**Reviewer concern:** *"Did you attempt any Chain-of-Thought (CoT) prompting? Specifically, does asking the model to explicitly reason about the opponent's range before outputting the probability distribution improve the JS distance to the StrategyAware oracle?"* — asked by 3 of 4 reviewers.

**Why this is the most important tier:** Identical model, identical opponent, identical seeds → only the prompt differs. Direct apples-to-apples comparison with the published numbers.

**Conditions** mirror the sanity grid + phase2 grid exactly:

```bash
# Sanity-tier CoT (4 cells × 50 hands)
for SEED in 42 123; do
  for TEMP in 0.0 0.2; do
    python run_experiment.py --agent hf --hf-model llama-70b \
      --opponent threshold --opponent-preset informative_v2 \
      --hands 50 --seed $SEED --temperature $TEMP \
      --elicit-beliefs --cot --capture-logprobs \
      --out logs/cot_sanity_70b_t${TEMP/./}_s${SEED}_informative_v2.jsonl -v
  done
done

# Phase2-tier CoT (2 cells × 350 hands at seed 42)
for TEMP in 0.0 0.2; do
  python run_experiment.py --agent hf --hf-model llama-70b \
    --opponent threshold --opponent-preset informative_v2 \
    --hands 350 --seed 42 --temperature $TEMP \
    --elicit-beliefs --cot --capture-logprobs \
    --out logs/cot_phase2_70b_t${TEMP/./}_s42_informative_v2.jsonl -v
done

# Enrich + analyze
for f in logs/cot_*.jsonl; do
  python -m analysis.build_dataset --input $f \
    --output ${f%.jsonl}_enriched.jsonl --opponent-preset informative_v2
done
python -m analysis.analyze_cot \
  --inputs logs/cot_phase2_*_enriched.jsonl logs/phase2_*_enriched.jsonl \
  --output results/cot_vs_direct.json
```

**Headline outputs:**
- Δ JS-to-StrategyAware (CoT vs direct)
- Δ base-rate-neglect ratio
- Δ update correlation r
- Fraction of CoT runs that mention opponent ranges / pot odds (`analyze_cot.score_reasoning`)

**Why equally representative:** Identical hand grid, identical seeds, identical opponent. The only nuisance variable is the prompt. With the same ~1,000 decisions, bootstrap CIs are directly comparable.

---

### Tier 2 — Cross-family generalization (THE second reviewer ask)

**Reviewer concern:** *"Test GPT-4o, Claude 3.5, Qwen, or Mistral to see if 'belief inertia' is a universal LLM trait or specific to Llama."* — asked by 3 of 4 reviewers.

**Conditions** (sanity-tier hand budget per model since API calls cost money):

```bash
# Open-weights other families
for MODEL in qwen-7b mistral-7b; do
  python run_experiment.py --agent hf --hf-model $MODEL \
    --opponent threshold --opponent-preset informative_v2 \
    --hands 50 --seed 42 --temperature 0.0 \
    --elicit-beliefs --capture-logprobs \
    --out logs/family_${MODEL}_t0_s42_informative_v2.jsonl -v
done

# Closed-source frontier (one each)
python run_experiment.py --agent api --api-provider openai \
  --api-model gpt-4o \
  --opponent threshold --opponent-preset informative_v2 \
  --hands 50 --seed 42 --temperature 0.0 --elicit-beliefs \
  --out logs/family_gpt4o_t0_s42_informative_v2.jsonl -v

python run_experiment.py --agent api --api-provider anthropic \
  --api-model claude-sonnet-4-20250514 \
  --opponent threshold --opponent-preset informative_v2 \
  --hands 50 --seed 42 --temperature 0.0 --elicit-beliefs \
  --out logs/family_claude_t0_s42_informative_v2.jsonl -v

python run_experiment.py --agent api --api-provider google \
  --api-model gemini-2.0-flash \
  --opponent threshold --opponent-preset informative_v2 \
  --hands 50 --seed 42 --temperature 0.0 --elicit-beliefs \
  --out logs/family_gemini_t0_s42_informative_v2.jsonl -v
```

**Why ~50 hands per model is OK:** That gives ~150–250 parsed beliefs per cell, enough for "is the model's mean JS distance > the published 0.014 ± 0.003 threshold?" with hand-clustered CIs. It's *not* enough to claim equivalence within tight bounds — but it's enough to test the universality claim, which is what the reviewers asked.

**Equally representative caveat:** Sample size per cell is smaller than the paper's pooled total, so individual model claims should be reported with their own CIs, not pooled. The reviewers asked "is this Llama-specific?", not "give me a paper-quality estimate for every model" — directional answers are what they want.

---

### Tier 3 — Scaling / emergence (the 8B → 70B → frontier hypothesis)

**Reviewer concern:** *"The sharp contrast between 8B (complete failure) and 70B (miscalibrated reasoning) suggests a capability phase transition."*

**Conditions** (matched to the existing 8B sanity cell):

```bash
# Existing 8B sanity log already exists: logs/sanity_8b_t0_s42_enriched.jsonl
# Existing 70B sanity log already exists: logs/sanity_70b_t0_s42_informative_enriched.jsonl
# New: a frontier model on the same 50-hand grid for direct comparison
python run_experiment.py --agent api --api-provider openai \
  --api-model gpt-4o \
  --opponent threshold --opponent-preset informative_v2 \
  --hands 50 --seed 42 --temperature 0.0 --elicit-beliefs \
  --out logs/scale_gpt4o_t0_s42_informative_v2.jsonl -v
```

**Why equally representative:** Same 50-hand grid as the paper's existing 8B sanity. Three points on the capability axis (8B, 70B, frontier) using identical decision sets where possible.

---

### Tier 4 — Opponent-model robustness (reviewer's opponent-mismatch concern)

**Reviewer concern:** *"StrategyAware is a normative target only if the agent is intended to share the same opponent model; otherwise the gap can reflect opponent-model mismatch rather than failure to condition on actions."*

**Conditions** (vary opponent only, hold model + seeds fixed):

```bash
for PRESET in informative_v2 tight_aggressive loose_aggressive loose_passive default; do
  python run_experiment.py --agent hf --hf-model llama-70b \
    --opponent threshold --opponent-preset $PRESET \
    --hands 50 --seed 42 --temperature 0.0 --elicit-beliefs \
    --out logs/opp_${PRESET}_70b_t0_s42.jsonl -v
  python -m analysis.build_dataset \
    --input logs/opp_${PRESET}_70b_t0_s42.jsonl \
    --output logs/opp_${PRESET}_70b_t0_s42_enriched.jsonl \
    --opponent-preset $PRESET
done
```

**Headline output:** Is the JS-to-StrategyAware gap stable across opponent types (the gap is real) or does it vanish when the opponent is "easier" to model (the gap was opponent-mismatch)?

**Why equally representative:** Same model, same seed, same hand count. Only the opponent and matching oracle change. Direct sensitivity analysis.

---

### Tier 5 — Internal vs verbalized calibration (the logprobs/interp concern)

**Reviewer concern:** *"How do you rule out the possibility that the model is calibrated internally but uncalibrated in its text generation? LLMs are notoriously poor at producing calibrated probabilities via natural language generation."*

**This is FREE if you ran Tier 1 with `--capture-logprobs` enabled** (which is included in the Tier 1 commands above). Plus run the existing logit-lens analysis on the same logs:

```bash
# Re-run a subset of phase2 with full interp capture (HF only, slower but rich)
python run_experiment.py --agent hf --hf-model llama-70b \
  --opponent threshold --opponent-preset informative_v2 \
  --hands 100 --seed 42 --temperature 0.0 \
  --elicit-beliefs --cot --capture-logprobs --logit-lens \
  --out logs/interp_70b_t0_s42_informative_v2.jsonl -v

python -m analysis.analyze_logit_lens \
  --input logs/interp_70b_t0_s42_informative_v2_logit_lens.jsonl \
  --output results/logit_lens.json
python -m analysis.analyze_attention \
  --input logs/interp_70b_t0_s42_informative_v2_attention.jsonl \
  --output results/attention.json
```

**Headline outputs:**
1. Compare token-level confidence (mean logprob of belief output) between CoT and direct → does CoT make the text more calibrated?
2. Logit-lens crystallization layer → does the model "know" the answer earlier than the text suggests?
3. Linear probe accuracy on hidden states for "is this trash hand?" → can a probe extract the right answer even when the verbalized output is wrong?

**Why equally representative:** Same env/opponent/seeds as the paper; just adds orthogonal measurement on top.

---

### Tier 6 — EV / pot-odds replacement of the equity-threshold claim

**Reviewer concern:** *"Some interpretations are also overstated, especially claims that actions are 'clearly correct' based on equity thresholds without an EV or pot-odds analysis."*

**This runs on existing logs — no new experiments needed.** Use the new `pot_odds_analysis.py`:

```bash
# Reproduce the "46.6% equity sacrificed" claim with a proper EV benchmark
python -m analysis.pot_odds_analysis \
  --input logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl \
  --output results/pot_odds_phase2_t0.csv \
  --num-rollouts 200 --samples-per-bucket 5 \
  --opponent-preset informative_v2 -v

# Same for CoT and other model conditions
python -m analysis.pot_odds_analysis \
  --input logs/cot_phase2_70b_t0_s42_informative_v2_enriched.jsonl \
  --output results/pot_odds_cot_t0.csv \
  --num-rollouts 200 --samples-per-bucket 5 \
  --opponent-preset informative_v2 -v
```

**Headline outputs:**
- `frac_threshold_agrees_with_ev_truth` — quantifies how often the simple equity rule and the EV-best action diverge (the reviewer's exact concern)
- `frac_ev_truth_optimal` — the rate at which the agent's chosen action was actually EV-optimal
- `frac_ev_belief_optimal` — internal-consistency check: did the agent play consistently with its OWN stated belief?
- `mean_ev_truth_regret` — average chips-per-decision left on the table

This **directly replaces** the paper's "46.6% equity, calling was clearly correct" framing with a defensible "X% of decisions were EV-suboptimal, with mean regret of Y chips" number.

---

## Summary Table — What Each Tier Costs and Answers

| Tier | What changes | Budget | Reviewer concern addressed |
|------|--------------|-------:|----------------------------|
| 0 | Re-run analysis on existing logs | minutes | Sanity / regression check |
| 1 | Llama 70B + `--cot`, paper grid | ~700 hands HF | CoT (3 reviewers) |
| 2 | 5 model families, 50-hand grid | ~250 hands HF + ~150 API calls | Multi-family generalization (3 reviewers) |
| 3 | Frontier model on 8B/70B grid | 50 API calls | Emergence/scaling (1 reviewer) |
| 4 | 5 opponent presets, fixed model | 250 hands HF | Opponent-model uncertainty (1 reviewer) |
| 5 | Logprobs + logit lens on Tier 1 | included in Tier 1 | Internal vs verbalized calibration (2 reviewers) |
| 6 | Pot-odds analysis on existing logs | minutes | EV / clearly-correct framing (1 reviewer) |

---

## Are These "Equally Representative" Compared to poker26?

**Yes, with one important caveat.** Equal representativity means:

1. **Same env config + opponent + seeds + temps + hand counts as the paper** for all tiers that compare directly to published numbers (Tiers 0, 1, 4, 5)
2. **Same statistical machinery** — clustered bootstrap, parsed-only filter, JS distance via scipy. Already in the extension.
3. **Same belief schema and parser** — `buckets_14_v1` in `compact` format, unchanged from paper

**Where they're NOT equally representative — and why that's deliberate:**

- Tiers 2, 3, 6 use the smaller "sanity" hand budget per cell because (a) API calls cost real money and (b) the reviewer questions there are directional ("is this universal?", "does scale matter?", "does EV say something different?"), not paper-quality estimation.
- Each cell still produces individually-bootstrappable JS distances with their own CIs — just wider CIs than the paper's pooled 1,084-belief estimate.

**The honest framing for a follow-up paper:**
- **Tier 1** is the new "headline" experiment with paper-grade sample size
- **Tiers 2–4** are robustness checks
- **Tiers 5–6** are mechanistic / methodological add-ons that strengthen the diagnostic story without re-running anything

---

## Suggested Run Order

1. **Tier 0** (5 min) — verify extension reproduces paper numbers
2. **Tier 6** (10 min) — re-frame the existing "clearly correct" claim with EV
3. **Tier 1** (overnight on a 70B-capable GPU) — the headline new experiment
4. **Tier 4** (overnight) — opponent robustness
5. **Tier 2** (one day, depends on API rate limits + budget) — cross-family
6. **Tier 5** (re-runs of Tier 1 logs with `--logit-lens`, ~6 hours)
7. **Tier 3** (~30 min API calls) — frontier scaling point

Tiers 1, 4, 5 can run sequentially on the same GPU. Tiers 2, 3, 6 are CPU/API-bound and can interleave.
