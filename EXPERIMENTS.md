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
| Test other model families (GPT, Claude, Qwen, Mistral, Gemini) | 3 | Tier 1A (open-weights) + Tier 2 (API) |
| Address scaling / emergence (8B → 70B → frontier) | 1 | Tier 1A.small + Tier 3 |
| Opponent-model uncertainty / mismatch | 1 | Tier 4 |
| Internal vs verbalized calibration (logprobs / interp) | 2 | Tier 5 |
| Equity threshold isn't EV (pot-odds analysis) | 1 | Tier 6 |
| Soften general claims (writing only) | 2 | (writing change) |
| Strengthen statistical validity (clustered bootstraps) | 1 | Already in base + Tier 1A (3 seeds) |

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

### Tier 1A — Scaled non-CoT baseline (BEFORE CoT)

**Why this comes first:** Before adding CoT (the new-prompt confound), we need to know how the *original* belief-elicitation prompt behaves across **(a) more seeds** and **(b) other model families at matched scale**. This addresses two reviewer concerns at once: scaling/emergence and family generalization, *without* introducing a prompt change. Any CoT effect in Tier 1 can then be measured against this stronger baseline rather than against the single-seed paper anchor.

**Two parallel sub-tiers, run on the same env / opponent / belief schema as the paper:**

#### Sub-tier 1A.large — ~70B class (paper-scale)

Tests whether belief inertia is Llama-3.1-specific or persists across (i) the next Llama generation and (ii) a different-family model at matched scale.

| Model | Registry name | Params | Vintage |
|-------|---------------|-------:|---------|
| Llama 3.1 70B Instruct (paper anchor) | `llama-70b` | 70B | Jul 2024 |
| Llama 3.3 70B Instruct | `llama-3.3-70b` | 70B | Dec 2024 |
| Qwen 2.5 72B Instruct | `qwen-72b` | 72B | Sep 2024 |

- **Grid:** 3 models × 3 seeds (42, 123, 456) × 2 temps (0.0, 0.2) = **18 cells**
- **Hands per cell:** 500 for `llama-70b` (anchor, target ~3× paper sample for tighter CIs), 350 for `llama-3.3-70b` and `qwen-72b` (paper-quality)
- **Total:** ~7,200 hands of HF compute
- **Captures:** beliefs + per-token logprobs (free with `--capture-logprobs`)

#### Sub-tier 1A.small — 8B class (capability-floor, parameter-matched)

Tests whether the 8B failure mode reported in the appendix is Llama-3.1-specific or a general property of small dense models. All three models are exactly 8B parameters so any difference reflects family / training, not capacity.

| Model | Registry name | HF model_id | Params | Vintage |
|-------|---------------|-------------|-------:|---------|
| Llama 3.1 8B Instruct | `llama-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Jul 2024 |
| Qwen 3 8B | `qwen-8b` | `Qwen/Qwen3-8B` | 8B | Apr 2025 |
| Ministral 8B Instruct (2410) | `ministral-8b` | `mistralai/Ministral-8B-Instruct-2410` | 8B | Oct 2024 |

- **Grid:** 3 models × 3 seeds × 2 temps = **18 cells**
- **Hands per cell:** 100 (8B inference is fast; the original 8B sanity used 50 hands and was clearly degenerate, 100 is enough to confirm or rule out the same pattern in Qwen 3 8B and Ministral 8B)
- **Total:** ~1,800 hands of HF compute (much faster wall-clock than the 70B tier)

**Qwen 3 thinking-mode caveat:** Qwen 3's chat template has `enable_thinking=True` by default, which would silently inject internal CoT into every response and confound the non-CoT baseline. The HFAgent now gates this on `self.cot_mode`, so non-CoT runs of `qwen-8b` correctly disable thinking. This is logged as `enable_thinking=False` in the agent config of every Tier 1A run. When `--cot` is passed (Tier 1), thinking is re-enabled.

**Vintage caveat:** The three 8B models span Jul 2024 → Apr 2025, vs the 70B tier which spans Jul–Dec 2024. This is unavoidable: there is no Llama 3.1-era exact-8B Qwen or Mistral release. The matched-parameter design keeps the comparison cleaner than mixing 7B and 8B; vintage drift is reported as a caveat in the writeup.

**Commands** — recommended, dedicated per-tier scripts that auto-launch in tmux and run analysis after each model so you get incremental results:

```bash
cd xpoker2026Extension

# Tier 1A.small (8B class) — ~3-6h on 1xH100/A100
bash scripts/run_tier1a_small.sh
#   - Creates and attaches you to tmux session 'poker_tier1a_small'
#   - You manually Ctrl-B D to detach when you want to leave
#   - Runs llama-8b -> qwen-8b -> ministral-8b sequentially
#   - Per-model: run cells, enrich, run PCE+UC; then cross-model pool
#   - Outputs in results/tier1a_small/

# Tier 1A.large (70B class) — overnight on 1xH100 80GB or multi-GPU
bash scripts/run_tier1a_large.sh
#   - Same pattern, session 'poker_tier1a_large'
#   - Runs llama-70b (500 hands/cell) -> llama-3.3-70b -> qwen-72b (350 each)
#   - Outputs in results/tier1a_large/

# Alternative: original "do everything" script (no tmux, no per-model analysis)
bash scripts/run_scaled_baseline.sh                  # all 36 cells (both tiers)
bash scripts/run_scaled_baseline.sh large            # only 70B tier
bash scripts/run_scaled_baseline.sh small            # only 8B tier
bash scripts/run_scaled_baseline.sh llama-70b        # one model only
# then:
bash scripts/enrich_scaled_baseline.sh               # build oracle posteriors
bash scripts/analyze_scaled_baseline.sh              # PCE + UC + pot-odds, grouped by tier
```

**Headline outputs (per tier):**
- Per-model JS-to-StrategyAware (mean ± 95% clustered bootstrap CI)
- Cross-seed variance: is the paper's 0.014 stable across 3 seeds, or seed-dependent?
- Cross-family agreement: do Qwen 72B / Llama 3.3 70B reproduce Llama 3.1 70B's gap?
- Small-tier degeneracy fingerprint: action diversity, parsed-belief rate, JS distance to uniform

**Why equally representative:**
- Same env, same opponent (`informative_v2`), same belief schema (`buckets_14_v1`), same statistical pipeline (clustered bootstrap, parsed-only filter) as the paper
- 70B-class hand counts ≥ paper's per-cell sample; pooled CIs will be tighter than paper's
- 8B-class hand counts at 100/cell are 2× the original 8B sanity, which itself produced the appendix claim

#### Tier 1A.future (NOT run now — noted for later)

A "newer-architecture, mid-scale" tier where all three families have a dense model at roughly comparable scale. Would test whether belief inertia persists in newer training recipes.

| Model | Params | Notes |
|-------|-------:|-------|
| Qwen 3 32B | 32B | Apr 2025, largest dense Qwen 3 |
| Gemma 4 31B | 31B | 2026, latest dense Gemma |
| Mistral Small 24B | 24B | 2025, ~25% smaller than the others |

Scale span 24–32B (1.3×) is tight enough to call "matched mid-scale". Not run yet because (a) Tier 1A.large/small already covers the reviewers' family-generalization ask, and (b) downloading/running three new model families is significant additional infra work. Re-evaluate after Tier 1A results.

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
| 1A.large | 3 models × 3 seeds × 2 temps at ~70B | ~7,200 hands HF | Cross-family at scale + seed stability |
| 1A.small | 3 models × 3 seeds × 2 temps at ~7-8B | ~1,800 hands HF | Capability-floor generalization |
| 1 | Llama 70B + `--cot`, paper grid | ~700 hands HF | CoT (3 reviewers) |
| 2 | 3 closed-source frontier APIs, 50-hand grid | ~150 API calls | API-only family generalization |
| 3 | Frontier model on 8B/70B grid | 50 API calls | Emergence/scaling endpoint |
| 4 | 5 opponent presets, fixed model | 250 hands HF | Opponent-model uncertainty (1 reviewer) |
| 5 | Logprobs + logit lens on Tier 1 | included in Tier 1 | Internal vs verbalized calibration (2 reviewers) |
| 6 | Pot-odds analysis on existing logs | minutes | EV / clearly-correct framing (1 reviewer) |
| 1A.future | Qwen 3 32B + Gemma 4 31B + Mistral 24B | ~3,600 hands HF | Newer-architecture mid-scale (deferred) |

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

1. **Tier 0** (5 min) — verify extension reproduces paper numbers ✓ done
2. **Tier 6** (10 min) — re-frame the existing "clearly correct" claim with EV
3. **Tier 1A.large** (overnight on a 70B-capable GPU) — scaled non-CoT baseline at 70B class
4. **Tier 1A.small** (a few hours on the same GPU) — 8B-class capability-floor sweep
5. **Tier 1** (overnight on a 70B-capable GPU) — CoT, the headline new experiment
6. **Tier 4** (overnight) — opponent robustness
7. **Tier 2** (one day, depends on API rate limits + budget) — cross-family API models
8. **Tier 5** (re-runs of Tier 1 logs with `--logit-lens`, ~6 hours)
9. **Tier 3** (~30 min API calls) — frontier scaling point
10. **Tier 1A.future** (deferred) — 24-32B mid-scale across Qwen / Gemma / Mistral

Tiers 1A.large, 1A.small, 1, 4, 5 can run sequentially on the same GPU. Tiers 2, 3, 6 are CPU/API-bound and can interleave.
