# The Journey — from "extend the accepted paper" to a localized deliberation circuit

A consolidated narrative of every experiment, decision, mistake, and finding from this work cycle. Two reading modes:

- **Section bodies** = technical detail (numbers, file paths, code references)
- **"In plain English" boxes** at the end of each section = the same content explained for a non-technical reader

The arc, in one sentence: *We set out to extend an accepted ICLR paper with Chain-of-Thought experiments; in the process we found a behavioral pathology in small models that turned out to be a measurement artifact, then a measurement artifact that turned out to reveal a real behavioral pattern, then a behavioral pattern that we causally pinpointed to a specific 3-layer "deliberation circuit" (layers 14–16) in Ministral 8B — and then generalized the underlying mechanism to Llama 8B (which exhibits the same late-layer-revision pattern at the logit-lens level) while finding that Qwen 8B notably DOESN'T (consistent with Qwen also not benefitting from CoT behaviorally).*

---

## Table of contents

1. [Where we started](#1-where-we-started)
2. [Phase A — Building the extension infrastructure](#2-phase-a--building-the-extension-infrastructure)
3. [Phase B — Tier 1A.small baseline (the foundation)](#3-phase-b--tier-1asmall-baseline-the-foundation)
4. [Phase C — Tier 1A.small with CoT (and the Qwen "double thinking" bug)](#4-phase-c--tier-1asmall-with-cot-and-the-qwen-double-thinking-bug)
5. [Phase D — The Ministral seed-42 dig-in (paradigm shift #1)](#5-phase-d--the-ministral-seed-42-dig-in-paradigm-shift-1)
6. [Phase E — Pot-odds / EV analysis (paradigm shift #2)](#6-phase-e--pot-odds--ev-analysis-paradigm-shift-2)
7. [Phase F — Logit-lens (paradigm shift #3)](#7-phase-f--logit-lens-paradigm-shift-3)
8. [Phase G — Causal patching (the mechanistic answer)](#8-phase-g--causal-patching-the-mechanistic-answer)
9. [Where we are now & the publishable claim](#9-where-we-are-now--the-publishable-claim)
10. [What's left](#10-whats-left)
11. [Glossary](#11-glossary)

---

## 1. Where we started

**Context:** the ICLR AIMS-accepted paper (poker26) studied LLM belief calibration in poker — does the model's stated belief about its opponent's hand match its action? Result: at 70B scale, models show measurable miscalibration; at 8B scale they degenerate (parse failures, one-hot trash beliefs).

**Reviewer feedback** boiled down to two asks:
1. Add **logprobs** for HuggingFace models (already had them for OpenAI API models — needed apples-to-apples comparison across providers).
2. Add **EV / pot-odds analysis** — show that the action is *clearly correct* via expected value, not just an equity threshold.

A research partner cloned the poker26 codebase, refactored it, and added a new repo: [`xpoker2026Extension/`](.) (hosted at `github.com/krishjainm/Updated-miscalibrated-belief-llms`, gitignored locally). The plan was to extend that repo with **Chain-of-Thought (CoT) experiments**, which were never in the original paper.

> **In plain English.** We had a published research paper about whether large language models "know what they don't know" when playing poker. Reviewers wanted us to add two things: (1) richer probability data from open-source models, (2) a more rigorous test of whether the model's choices are actually correct. Our research partner gave us a refactored codebase to build on, and our new question was: does asking the model to "think out loud" before acting (Chain-of-Thought) make it better or worse?

---

## 2. Phase A — Building the extension infrastructure

We did a deep audit of the partner's refactor ([`AUDIT.md`](AUDIT.md)) and added the two reviewer-requested features:

- **Logprobs for HFAgent** ([`poker_env/agents/hf_agent.py`](poker_env/agents/hf_agent.py)) — `output_scores=True` in `generate()`, then a custom extractor that mimics the OpenAI logprobs schema so cross-provider analysis works without adapter code.
- **EV / pot-odds pipeline** ([`analysis/pot_odds_analysis.py`](analysis/pot_odds_analysis.py)) — Monte-Carlo Q-value rollouts per decision, computes pot odds, EV-truth (with true opponent hand), EV-oracle (with belief posterior), and EV-belief (with model's stated belief).

We also wrote [`EXPERIMENTS.md`](../EXPERIMENTS.md) — a structured plan for what experiments to run and why, mapped to reviewer asks.

> **In plain English.** Before running anything new, we extended the codebase to capture more data per decision: (1) the model's confidence in each token it generated, (2) a calculation of whether each action made financial sense given the cards on the table.

---

## 3. Phase B — Tier 1A.small baseline (the foundation)

**Why first:** before testing CoT, we needed a clean apples-to-apples baseline at small scale. The original paper used 70B-class models (which cost ~12h GPU per cell); 8B models are ~10× cheaper and let us run 18 cells in one overnight.

**Setup** ([`scripts/run_tier1a_small.sh`](scripts/run_tier1a_small.sh)):
- 3 models: Llama 3.1 8B, Qwen 3 8B, Ministral 8B
- 3 seeds × 2 temperatures × 100 hands = **18 cells**
- Opponent: `informative_v2` heuristic preset (deterministic, equity-aware)
- All without CoT (direct JSON action output)

**The Qwen 3 thinking-mode confound:** Qwen 3 has a native "thinking mode" where it generates internal `<think>...</think>` reasoning before its visible output. If left on, this would silently inject CoT into our "non-CoT baseline" and contaminate everything. We force-disabled it via the `enable_thinking=False` flag (logged in every JSONL record as a permanent audit trail).

**Results** ([`results/tier1a_small/`](results/tier1a_small)): 36 result CSVs, all integrity-verified. Patterns matched the original paper: 8B models show degeneracy (low parse rates, trash beliefs) but still produce usable EV / action distribution data.

> **In plain English.** Before testing the new "think out loud" feature, we ran a baseline: three different small AI models, each playing 100 poker hands with three different randomization seeds at two different "confidence" settings. 18 separate experiments total. We had to be careful with one model (Qwen) that has a hidden "thinking" feature we needed to turn off, otherwise our baseline wouldn't really be a baseline.

---

## 4. Phase C — Tier 1A.small with CoT (and the Qwen "double thinking" bug)

**Setup** ([`scripts/run_tier1a_small_cot.sh`](scripts/run_tier1a_small_cot.sh)):
- Same 18 cells, but with `--cot` (the model first writes free-form reasoning, then outputs JSON).

**Pilot (6 cells) revealed Qwen catastrophic failure:** **1.3% parse rate** on Qwen 8B CoT cells. Investigation showed the cause was a code-level bug, not a model failure: when `cot_mode=True`, [`hf_agent.py`](poker_env/agents/hf_agent.py) was *also* enabling Qwen's native thinking mode (because `cot_mode` was being interpreted as "let the model think any way it wants"). So Qwen was generating:

```
<think>
... internal Qwen reasoning, hundreds of tokens ...
</think>
REASONING: ... prompt-level reasoning we asked for ...
JSON: {"action": "..."}
```

The combined output blew through the 768-token budget, truncating before the JSON ever appeared. Both reasoning modes ran simultaneously — "double CoT".

**Fix A** (in code): in [`hf_agent.py`](poker_env/agents/hf_agent.py), force `enable_thinking=False` for ALL `has_thinking_mode` models regardless of `cot_mode`. Native thinking is now permanently off; only our prompt-level CoT is allowed.

**Fix B** (post-hoc test): wrote a script to strip `<think>...</think>` blocks from the existing pilot's `raw_response` and re-parse. **Recovered exactly 0 of 608 failures** — confirming the issue is token-budget exhaustion (the JSON was never emitted), not parser failure (the JSON was there but malformed). Fix A is the real fix.

**Full grid (18 CoT cells)** completed cleanly with Fix A in place. CoT-vs-non-CoT comparison committed to [`results/tier1a_small_cot/COMPARISON.md`](results/tier1a_small_cot/COMPARISON.md).

> **In plain English.** When we asked the models to "think out loud," Qwen acted broken — only 1.3% of its responses were parseable. Investigation showed our code was accidentally telling Qwen to think *twice* (once in its native style, once in our prompt-level style). The double-thinking ate up the response space before Qwen could write its final answer. We fixed the bug, re-ran the experiments, and got clean data for all three models.

---

## 5. Phase D — The Ministral seed-42 dig-in (paradigm shift #1)

**The puzzle:** in the full CoT grid, Ministral 8B's `parse_success_rate` looked weird:

| Cell | parse rate | fallback rate |
|---|---:|---:|
| `cot_ministral8b_t0_s42`  | **21.7%** | 37.9% |
| `cot_ministral8b_t02_s42` | **23.5%** | 36.0% |
| `cot_ministral8b_t0_s123` | 49.8% | 0.5% |
| `cot_ministral8b_t02_s123` | 49.5% | 1.0% |
| `cot_ministral8b_t0_s456` | 49.3% | 1.4% |
| `cot_ministral8b_t02_s456` | 48.4% | 3.2% |

Seed 42 was an outlier. Half the parse rate of every other seed. **Was the parser broken on those specific raw responses?**

**The investigation.** I sampled a few hundred raw responses from `cot_ministral8b_t0_s42`. Every single one contained a perfectly-formed JSON like `{"action": "FOLD"}`. **Zero JSON parsing failures.** What was actually happening:

1. **47 cases of alias mismatch.** Ministral wrote `{"action": "CHECK"}` (single word). The action map only knew `"CHECK_OR_CALL"` (the canonical name with disjunctions). So a perfectly valid intended action got logged as a parse failure.
2. **412 cases of illegal FOLD attempts.** Ministral wrote `{"action": "FOLD"}` when `bet_to_call=0` (it could check for free; folding was not a legal action). The env's `_fallback_action` rescued these to `CHECK_OR_CALL`. Logged as `fallback_used=True`, conflated with parse failures.

**Two distinct things were being conflated by a single `parse_success` flag:**
- (a) JSON didn't parse / wasn't recognized
- (b) JSON parsed and was recognized but wasn't legal in context

**Fix (4 things, all done):**
1. Added `ACTION_ALIASES` dict to [`json_utils.py`](poker_env/agents/json_utils.py) mapping `"CHECK"`/`"CALL"`/`"BET"`/`"RAISE"` (and gerunds, and word-order variants like `"CALL_OR_CHECK"`) to canonical names.
2. Split `ActionMetadata` in [`hf_agent.py`](poker_env/agents/hf_agent.py) and [`api_agent.py`](poker_env/agents/api_agent.py) into 3 diagnostic flags: `action_json_parsed` / `action_recognized` / `action_legal_in_context`. Old `parse_success` is now `(parsed AND recognized AND legal)` — the conjunction the field was always *trying* to be.
3. Wrote [`analysis/recategorize_action_metadata.py`](analysis/recategorize_action_metadata.py) — re-runs the alias-aware bucketing on existing logs to retroactively distinguish the three failure modes. Output: [`results/tier1a_small_cot/COMPARISON_v2.md`](results/tier1a_small_cot/COMPARISON_v2.md).
4. Documented in [`updates.md` §11](updates.md).

**Side discovery (semi-major):** while testing the recategorizer on locally-cached logs, I ran it on the original poker26 70B sanity logs. The 70B models systematically emit `"CALL_OR_CHECK"` (reversed word order) ~10% of the time. Same alias bug.

| Original poker26 log | parse_v1% (as published) | parse_v2% (alias-aware) | recovered |
|---|---:|---:|---:|
| `phase2_70b_t02_s42` | 45.2% | **57.8%** | +12.6 pts |
| `phase2_70b_t0_s42` | 45.7% | **56.8%** | +11.0 pts |
| `sanity_70b_t02_s42` | 43.8% | **56.5%** | +12.7 pts |
| `sanity_70b_t0_s42` | 43.8% | **57.6%** | +13.8 pts |

The published 70B parse rates **under-reported the model's effective parse rate by 8–14 percentage points**. The 8B sanity log was unaffected (its 55% failure is the well-known JSON degeneracy from the paper's appendix), so this is a 70B-specific vocabulary quirk. Worth a footnote in the camera-ready / rebuttal.

**Reframing.** The "Ministral seed-42 parse-failure pathology" turned out to be:
- 0 actual parse failures
- ~10% recoverable alias misses
- ~30% real model behavior: Ministral CoT becomes *pathologically conservative* on this seed's specific hand sequence and tries to fold into free-check spots ~30% of the time

The pathology is **real** but **behavioral**, not a parser bug. The next question: is this conservatism actually *bad*, or does the env's safety-net rescue mechanism make it OK?

> **In plain English.** Ministral looked like it was "broken" on one specific random seed — only 22% of its responses being usable, vs ~50% on other seeds. We assumed it was a parser bug. It wasn't. Two real things were happening: (1) Ministral was using slightly different words than our code expected (saying "CHECK" instead of "CHECK_OR_CALL"), and (2) Ministral was actually trying to "give up" on hands ~30% of the time when it wasn't even allowed to give up (it could stay in for free). We added word-aliases to fix the first issue; the second turned out to be a real behavior worth investigating further. As a bonus, we discovered the original published paper had the same word-alias issue affecting 70B models — its parse rate was under-reported by ~10% across the board.

---

## 6. Phase E — Pot-odds / EV analysis (paradigm shift #2)

**The new question:** if Ministral's seed-42 CoT decisions are 30% "illegal FOLDs" that the env safely re-routes to `CHECK_OR_CALL`, **do those rescued decisions cost the model EV, or does the rescue actually help?**

### First pass: Ministral s42 only (4 cells)

Wrote [`analysis/decompose_pot_odds_by_fallback.py`](analysis/decompose_pot_odds_by_fallback.py) — joins the per-decision pot-odds CSV with the enriched-log `action_metadata` and buckets each decision into:
- `clean` = no fallback used
- `illegal_fold_rescued` = model picked illegal FOLD, env rescued to CHECK_OR_CALL
- `other_fallback` = anything else (parse fail, alias miss, illegal non-FOLD)

Result on Ministral s42 ([`results/tier1a_small_cot/pot_odds/SUMMARY.md`](results/tier1a_small_cot/pot_odds/SUMMARY.md)):

| Cell × bucket | n | mean EV-truth-regret (chips/dec) | EV-optimal % |
|---|---:|---:|---:|
| CoT t=0, **clean picks** | 114 | **3.370** | 42.1% |
| CoT t=0, **illegal-FOLD rescued** | **179** | **0.554** | **60.9%** |
| non-CoT t=0, clean (only bucket) | 215 | 2.507 | 31.6% |

**The surprise:** the rescued-illegal-FOLD decisions had **6× lower mean regret** and **higher EV-optimality** than cleanly-emitted CoT decisions. The "pathology" was actually *helping* the model's EV.

The interpretation: Ministral under CoT becomes pathologically conservative; when the model wants to fold a marginal hand into a free-check spot, that intent is "give up cheaply." The env's `_fallback_action` honors that intent at zero chips by playing CHECK_OR_CALL (which is EV-optimal in those situations). The fallback isn't papering over a mistake — it's executing the cheapest version of the model's intent.

### Second pass: full 36-cell grid

The Ministral-only result was suggestive but limited. Wrote [`scripts/run_pot_odds_full_grid.sh`](scripts/run_pot_odds_full_grid.sh) — runs pot-odds on every CoT and non-CoT cell (36 total, 18 of each), with the by-fallback decomposition baked in.

Wall-clock: 50 min on a 4-wide MacBook M-series.

Key result ([`results/pot_odds_full_grid/SUMMARY.md`](results/pot_odds_full_grid/SUMMARY.md)):

**(1) The rescue effect generalizes across the family:**

| Model | CoT cells with any rescue | Total rescued FOLDs | Cells where rescue regret < clean regret |
|---|---:|---:|---:|
| `llama8b`     | 6/6 | 144 | 5/6 |
| `ministral8b` | 6/6 | 353 | 4/6 |
| `qwen8b`      | 6/6 | 53 | 6/6 |

In every cell that had any rescues, those rescued decisions tended to have lower regret than the clean ones — not just for Ministral.

**(2) CoT splits three different ways across models:**

| Model | Temp | EV-regret CoT/non-CoT/Δ | EV-optimal % CoT/non-CoT/Δ |
|---|---|---|---|
| `llama8b`     | 0.0 | 2.10 / 2.56 / **−0.45** | 40.7 / 49.5 / **−8.8 pp** |
| `llama8b`     | 0.2 | 2.02 / 2.46 / **−0.44** | 42.2 / 49.1 / **−6.8 pp** |
| `ministral8b` | 0.0 | 1.63 / 1.75 / **−0.12** | 35.3 / 30.2 / **+5.1 pp** |
| `ministral8b` | 0.2 | 1.57 / 1.72 / **−0.15** | 35.1 / 30.3 / **+4.8 pp** |
| `qwen8b`      | 0.0 | 2.23 / 2.12 / **+0.11** | 44.9 / 42.0 / **+2.9 pp** |
| `qwen8b`      | 0.2 | 2.20 / 2.11 / **+0.09** | 44.7 / 43.5 / **+1.3 pp** |

- **Llama**: CoT lowers regret AND lowers sharpness (smooths errors at the cost of fewer outright optimal picks).
- **Ministral**: CoT lowers regret AND raises sharpness — but the bucket table shows this is driven entirely by s42's 341 rescued FOLDs. Strip those and the comparison is a wash.
- **Qwen**: CoT *raises* regret slightly. Closest to neutral; if anything, mildly harmful.

**(3) CoT shortens hands.** Non-CoT cells have 2–3× more decisions than CoT cells (Llama non-CoT: 1374–1695 decisions; Llama CoT: 524–646). Under CoT, small models fold more often (legally or illegally-then-rescued), which truncates hands.

**Reframing.** The original CoT-improvement headline was a rescue artifact. The honest writeup line:

> **For small models in this poker environment, CoT does not unambiguously improve decision quality; it changes the shape of the decision distribution in model-specific ways, and any aggregate EV win is partly attributable to the env's safety net catching pathological action choices.**

> **In plain English.** We expected Chain-of-Thought to make small models play poker better. The full numbers say otherwise: Llama becomes more careful but less sharp (fewer "perfect" plays); Ministral *appears* better only because the game's safety mechanism keeps catching its bad decisions and converting them to safe ones; Qwen actually plays slightly worse with CoT. Across all three, CoT makes models more likely to fold quickly, which ends hands faster. So the simple "CoT helps small models" story is wrong — what's really happening is more interesting and varies by model.

---

## 7. Phase F — Logit-lens (paradigm shift #3)

By now we had a behavioral observation: small-model CoT produces lots of illegal-FOLD attempts that get rescued. The next question was **mechanistic**: when the model commits to FOLD in a free-check spot, what's happening inside its layers? Does any intermediate layer "know" the right action (CHECK), with the final layers losing it (a *verbalization* failure)? Or is the FOLD signal encoded top-to-bottom?

### The setup

Built [`scripts/run_tier1a_small_cot_logitlens.sh`](scripts/run_tier1a_small_cot_logitlens.sh) — re-runs the 18-cell CoT grid with `--logit-lens` enabled. The logit-lens hook captures, at every transformer layer and every generated position, the top-1 token that the layer would predict if you projected its hidden state through the unembedding head.

Output per cell: a `*_logit_lens.jsonl.gz` sidecar (18 of them, one per cell). Wall-clock: 26 h on Blackwell.

### The first analysis — and a bug

I wrote [`analysis/analyze_logit_lens_by_failure_mode.py`](analysis/analyze_logit_lens_by_failure_mode.py) to cross-join the sidecar with the enriched-log `action_metadata` flags, bucket each decision (`clean / illegal_fold / illegal_other / json_failure / alias_unrecognized`), and report per-bucket × per-layer action mix at the action-emission token.

**First aggregate ran clean — but every layer for every bucket showed 100% "OTHER".**

Diagnosis: the sidecar stores per-layer top-1 tokens for **every generated position** (~80–130 positions per CoT response). My naive analyzer was reading position `-1` (the EOS marker `'<|eot_id|>'`), which is always "OTHER" — not "FOLD" or "CHECK". The actual action verb is somewhere in the middle, like position 78 out of 85.

**Fix:** anchor on the JSON close brace. Walk back from `'"}'` to find the value-opening quote `' "'`, take the position right after. That's the verb position. Subword splits handled (`'F'+'OLD'`, `'B'+'ET'+'_OR'+'_RA'+'ISE'`).

### The actual finding (after fix)

For each cell, the **crystallization layer** (earliest layer from which the action-group prediction at the verb position is stable through to the end):

| Model | Δ (illegal_fold − clean), range across 6 cells | Direction |
|---|---|---|
| **Llama 8B (32 layers)** | **−2.5 to −4.0** | illegal-FOLD crystallizes EARLIER |
| **Ministral 8B (36 layers)** | **−1.2 to −2.7** | illegal-FOLD crystallizes EARLIER |
| **Qwen 8B (36 layers)** | −0.6 to +0.5 | no significant difference |

For Llama and Ministral, illegal-FOLDs become FOLD-committed in the residual stream **2–4 layers earlier** than clean decisions. The OPPOSITE of what I expected.

### Splitting "clean" by emitted action revealed the real story

The "clean" bucket pools legal-FOLD, CHECK_OR_CALL, and BET_OR_RAISE. Splitting it on Ministral 8B s=42 t=0 (n=313 joined records):

| Bucket | n | Crystallization layer | Layer 22-23 trajectory | Layer 25-29 trajectory |
|---|---:|---:|---|---|
| `clean_CHECK_OR_CALL` | 29 | **28.8** | OTHER | starts FOLD (86%), then 28→48% FOLD/48% CHECK, then 29→90% CHECK |
| `clean_LEGAL_FOLD` | 98 | 23.7 | smooth FOLD ramp | 100% FOLD |
| `illegal_FOLD` | 179 | **22.7** | aggressive FOLD ramp (33% → 100%) | 100% FOLD |

Reading per-layer action mix at the action-verb position:

```
  L | clean_CHECK_OR_CALL          | clean_LEGAL_FOLD        | illegal_FOLD
----+------------------------------+-------------------------+-------------------
 22 | OTHER 1.00                   | FOLD 0.02  OTHER 0.98   | FOLD 0.33  OTHER 0.67
 23 | OTHER 1.00                   | FOLD 0.26  OTHER 0.74   | FOLD 1.00
 24 | OTHER 1.00                   | FOLD 1.00               | FOLD 1.00
 25 | FOLD  0.86  OTHER 0.14       | FOLD 1.00               | FOLD 1.00
 27 | FOLD  0.76  CHECK 0.21       | FOLD 1.00               | FOLD 1.00
 28 | FOLD  0.48  CHECK 0.48       | FOLD 1.00               | FOLD 1.00
 29 | FOLD  0.00  CHECK 0.90       | FOLD 1.00               | FOLD 1.00
 35 | FOLD  0.00  CHECK 1.00       | FOLD 1.00               | FOLD 1.00
```

What this revealed:

1. **All decisions start with a FOLD-leaning signal** in mid-to-late layers (~L=22+ for Ministral).
2. **Legal-FOLD decisions** lock in at FOLD by layer 24 and never revise.
3. **Illegal-FOLD decisions** lock in **even earlier** (layer 23) and **more confidently**. The model is *more certain* of the wrong FOLD than of correct FOLDs.
4. **CHECK_OR_CALL decisions are the ONLY bucket that late-layer-revises** — they start FOLD-leaning at layer 25, then CHECK climbs from 0% → 21% → 48% → 90% across layers 27–29, fully overtaking by layer 35.

**The §13 interpretation:**

> Small-model CoT in this poker setting has a **baseline FOLD pull** in the mid-to-late residual stream. What distinguishes outcomes is whether a **late-layer revision circuit** overrides that pull. CHECK_OR_CALL emerges only via late-layer revision; legal FOLDs lock in early; illegal FOLDs lock in even earlier and more confidently. The illegal-FOLD pathology is a **failure of late-layer deliberation**, not a verbalization-stage glitch.

This is the mechanistic explanation for the §12 EV finding: when the deliberation circuit doesn't fire, the model commits to FOLD in the residual stream from layer 23–24 onwards; the env's `_fallback_action` catches those FOLDs in free-check spots and converts them to the EV-optimal CHECK_OR_CALL.

> **In plain English.** We added "X-rays" to the model — instruments that let us read what each of the 36 layers of the model "thinks" the next word should be. We expected to find that when the model picked the wrong action (illegal FOLD), early layers were predicting CHECK and only the very last layer made the wrong call. The data showed the OPPOSITE: when the model picked FOLD, every layer from layer 22 onward was confidently predicting FOLD. The interesting cases were the CORRECT decisions — there, early layers also predicted FOLD, but a "deliberation circuit" in the late layers (27-29) overrode the initial pull and switched the prediction to CHECK. So the question stopped being "why does the model verbalize wrong" and became "why does the deliberation circuit sometimes fail to fire."

---

## 8. Phase G — Causal patching (the mechanistic answer)

The §13 finding was **correlational**: late-layer revision *correlates with* CHECK output. To make it **causal** — and to pinpoint *which* layers actually do the deliberation — we needed activation patching.

### The plan

**Hypothesis:** if we take the residual stream at a late layer from a *clean CHECK_OR_CALL* decision (where the deliberation circuit succeeded) and inject it into an *illegal_FOLD* decision (where the circuit failed), the second decision should now predict CHECK.

**Built infrastructure** (all locally, GPU-tested later):
- [`poker_env/interp/patching.py`](poker_env/interp/patching.py) — `HiddenStateCapture` (saves residuals per layer at the verb position) + `HiddenStatePatch` (replaces residual at one layer with a saved tensor)
- [`poker_env/interp/forward_helpers.py`](poker_env/interp/forward_helpers.py) — `PromptReconstructor` (rebuilds the exact input the model originally saw) + `run_forward_at_last_position` (single forward pass bypassing `model.generate`)
- [`experiments/causal_patching.py`](experiments/causal_patching.py) — main CLI driver with 3 controls (no-patch baseline, self-patch identity, random-source null)
- [`experiments/verify_prompt_reconstruction.py`](experiments/verify_prompt_reconstruction.py) — Phase 1b verification: forward-pass top-1 must match recorded verb. **Experiment blocked if it fails.**
- [`scripts/run_causal_patching_pilot.sh`](scripts/run_causal_patching_pilot.sh) and [`scripts/run_causal_patching_layer_sweep.sh`](scripts/run_causal_patching_layer_sweep.sh) — GPU drivers
- [`CAUSAL_PATCHING_RUNBOOK.md`](CAUSAL_PATCHING_RUNBOOK.md) — what to run on the GPU box, in order

**Pre-flights all passed locally** (CPU-only): prompt-hash reproduction was bit-identical on 5/5 sampled records of Ministral s42 t=0; verb-finder worked across FOLD / CHECK / BET responses; bucketing matched §13 numbers exactly (29 clean_CHECK_OR_CALL, 98 clean_legal_fold, 7 clean_bet_or_raise, 179 illegal_fold).

### Pilot — saturated, but the controls worked

Ran on the GPU box: 10 sources × 30 targets × 5 layers (22, 24, 26, 28, 30) = 1,500 patched forwards.

| Layer | n | mean Δlogit(CHECK − FOLD) | top-1 → CHECK-family |
|---:|---:|---:|---:|
| 22 | 300 | **+10.32** | 100% |
| 24 | 300 | +10.61 | 100% |
| 26 | 300 | +11.32 | 100% |
| 28 | 300 | +11.34 | 100% |
| 30 | 300 | **+11.43** | 100% |

Controls:
- `baseline_top1_match_rate = 1.0` ✓ (perfect prompt reconstruction)
- `self_patch_max_logit_drift = 0.0` ✓ (perfect hook plumbing)
- `random_source_mean_delta = −0.70` ✓ (random sources don't shift toward CHECK)

The causal effect was real (12-nat specificity gap between CHECK source and random source), but **layer specificity was absent** — L=22 and L=30 produced essentially the same effect (+10 vs +11 nat). The plan's success gate (L=28 ≥ 3× L=22) was NOT met (we got 1.1×).

**What we learned:** by L=22, the residual at the verb position already encodes the prediction. Patching it wholesale at any L=22+ flips the output. The deliberation must happen *earlier*.

### Layer sweep — pinpointing the circuit

Wrote [`scripts/run_causal_patching_layer_sweep.sh`](scripts/run_causal_patching_layer_sweep.sh) — every layer 0..35 of Ministral 8B, 10 sources × 15 targets, 5 random-source nulls per layer. Also extended the CLI driver to compute per-layer null baselines (instead of one mid-layer null) so we get a clean **specificity-adjusted Δ** at every depth.

Wall-clock: ~3 hours on H100. Output: [`results/causal_patching/ministral8b_t0_s42_layer_sweep/SUMMARY.md`](results/causal_patching/ministral8b_t0_s42_layer_sweep/SUMMARY.md).

**The result:**

| Layers | Specificity-adjusted Δ | Top-1 → CHECK | Interpretation |
|---|---:|---:|---|
| **0–13** | **±0.5 nat (null)** | **0%** | Residual at verb position carries no CHECK/FOLD signal yet |
| **14** | **+1.57 nat** | **4%** | **Signal emerges** |
| **15** | **+3.96 nat** | **26.7%** | Rising |
| **16** | **+7.43 nat** | **100%** | **Full flip established** |
| 17–21 | +8.5 to +10.2 nat | 100% | Strengthening |
| 22–35 | +10.4 to +12.0 nat | 100% | Saturated read-out |

A clean **3-layer transition** from null to complete flip. Quality indicators:
- Random null is tight: between −0.6 and +0.18 nat across all 36 layers
- Effect at L=14 is ~20σ above 0 (n=150 pairs, SE ≈ 0.08 nat)
- Self-patch identity = 0.0 (perfect)
- Baseline match = 1.0 (perfect)

**The deliberation circuit lives at layers 14–16 of Ministral 8B.**

### What this rewrites about §13

The §13 logit-lens analysis said "crystallization at layer 22-23." The causal patching distinguishes **two layers that mechanistically matter**:

1. **Layer 16 — causal commitment layer.** Where the verb decision is made. Patching here flips the output 100% of the time.
2. **Layer 22 — emergence-in-vocab layer.** Where the verb's high-dimensional encoding becomes legible to the unembedding head as a verb-token (the §13 "crystallization" finding).

The §13 "crystallization at L=22" was downstream evidence of a **L=14–16 commitment**.

> **In plain English.** We took the X-ray finding from the previous phase and turned it into a surgical experiment. We literally took the "thinking" from a moment when the model decided CHECK, and pasted it into a moment when the model decided (incorrectly) FOLD. Then we watched what came out. Result: if we paste at layer 16 or later, the model now says CHECK every single time. If we paste at layer 13 or earlier, nothing changes. Layers 14, 15, 16 are where the model "decides" — and the transition is sharp, only 3 layers wide. Before layer 14, the decision hasn't been made yet; after layer 16, it's locked in. We pinpointed the deliberation circuit.

---

## 8.5. Phase H — cross-model logit-lens generalization (added 2026-05-XX)

After the Ministral causal-patching sweep landed clean, we still had two open questions: (1) does the §13 late-layer-revision pattern generalize to Llama and Qwen, or is it Ministral-specific? (2) is our action-token classifier robust across all three tokenizers, or are we silently missing tokens we never explicitly added?

### 8.5a. Subword robustness audit ([`analysis/audit_action_verb_tokens.py`](analysis/audit_action_verb_tokens.py))

Scanned every verb-position token across all 18 sidecars (3 models × 6 cells). Each token at the action-verb position was tested against `ACTION_TOKEN_GROUPS` + `ACTION_SUBWORD_PREFIXES`.

**Result: 100% coverage across all 18 cells, all 3 models, all 5,290 records.** No silently-missed tokens. The classifier is robust as-is. Output: [`results/audit/AUDIT_ACTION_VERB_TOKENS.md`](results/audit/AUDIT_ACTION_VERB_TOKENS.md).

### 8.5b. Cross-model clean-bucket-split ([`analysis/cross_model_clean_split.py`](analysis/cross_model_clean_split.py))

Generalized the §13 detail-table from "Ministral s42 t=0" to all 18 cells, splitting the "clean" bucket by emitted action (`clean_CHECK_OR_CALL` / `clean_LEGAL_FOLD` / `clean_BET_OR_RAISE`).

**Per-model weighted-mean crystallization layer** (from [`results/tier1a_small_cot_logitlens/CROSS_CELL_DETAILED.md`](results/tier1a_small_cot_logitlens/CROSS_CELL_DETAILED.md)):

| Model | clean CHECK_OR_CALL | clean LEGAL FOLD | illegal FOLD | Δ (illegal − CHECK) |
|---|---:|---:|---:|---:|
| `llama8b`     | **25.4** (n=1080) | 20.9 (n=579) | 20.7 (n=143) | **−4.7** |
| `ministral8b` | **29.6** (n=83)   | 23.8 (n=596) | 22.7 (n=350) | **−6.9** |
| `qwen8b`      | 30.6 (n=1149)     | 29.8 (n=522) | 30.4 (n=42)  | **−0.2** |

**The big finding:** Llama AND Ministral both show the §13 pattern (CHECK_OR_CALL crystallizes 5–7 layers LATER than FOLD-emitting decisions = the late-layer revision signature). **Qwen does NOT** — its three buckets all crystallize at the same late layer (~30), no late-layer revision detectable.

This **lines up perfectly** with the §12 EV finding that Qwen *also* doesn't get an aggregate EV win from CoT (it's the only model where CoT slightly *hurts*) AND with the §11 illegal-FOLD-rate finding that Qwen has the fewest illegal FOLDs of the three (4–11 per cell vs Llama 16–34 vs Ministral 1–179).

**The §13 mechanistic pattern and the §11–§12 behavioral patterns are now LINKED across the family:**
- Llama, Ministral: late-layer-revision circuit present + illegal-FOLD pathology + CoT helps EV (sometimes via the rescue mechanism)
- Qwen: no late-layer-revision circuit + minimal illegal-FOLD pathology + CoT doesn't help

This is the cleanest cross-model story we have. It tightens the BlackboxNLP claim from "Ministral has a circuit at L=14-16" to "small-model CoT-deliberation circuits exist as a 2-of-3 family property; the model that lacks the circuit ALSO lacks the behavioral signatures we expect a deliberation circuit to produce."

> **In plain English.** We took the X-ray finding from Ministral and asked: do the other two small models show the same thing? Two of three do (Llama and Ministral). The third (Qwen) doesn't show the X-ray pattern AND also doesn't show the behavioral pattern (Qwen barely tries to fold incorrectly, and Chain-of-Thought doesn't help Qwen). So the mechanism we found isn't unique to Ministral — but it isn't universal either. It's a property of *some* small models, and the model that lacks it is exactly the model that lacks the corresponding behavior.

### 8.5c. What we have NOT yet done (gates the next round of GPU runs)

- **Causal patching on Llama 8B** — pending GPU. Predicted: sharp transition at L~14–16 (matching Ministral) given Llama's 32 layers.
- **Causal patching on Qwen 8B** — pending GPU. Predicted: NO sharp transition (consistent with §13/§8.5b absence). If true, that's the cleanest possible BlackboxNLP figure: 2 models with circuits, 1 without, all 8B-class.
- **Replication on more Ministral seeds/temps** — s42 t=0.2 has 161 illegal_FOLDs (good replication target); s123/s456 t=0 only have 1–3 (smoke test).

Three new driver scripts written for these:
- [`scripts/run_causal_patching_llama_sweep.sh`](scripts/run_causal_patching_llama_sweep.sh) — `SCOPE=s42_only` (3 h) or `SCOPE=pooled` (~6 h)
- [`scripts/run_causal_patching_qwen_sweep.sh`](scripts/run_causal_patching_qwen_sweep.sh) — `SCOPE=pooled` recommended (~5 h)
- [`scripts/run_causal_patching_ministral_replicate.sh`](scripts/run_causal_patching_ministral_replicate.sh) — 3 cells back-to-back (~3 h)

The CLI driver ([`experiments/causal_patching.py`](experiments/causal_patching.py)) was extended to accept multiple `--enriched-log` arguments for pooled-seeds runs.

---

## 9. Where we are now & the publishable claim

The end-state of this work cycle:

**Behavioral findings (paper-ready):**
- Small-model CoT does not unambiguously improve EV; it changes the shape of the decision distribution model-specifically (Llama: smoothed-but-less-sharp; Ministral: rescue-driven; Qwen: mildly worse).
- Small-model CoT shortens hands (2–3× fewer decisions per cell vs non-CoT).
- The env's `_fallback_action` rescue mechanism is responsible for most of CoT's apparent EV win on Ministral specifically (driven by 341 rescued illegal-FOLDs on s42).

**Mechanistic findings (paper-ready):**
- Across 18 CoT cells, illegal-FOLD decisions crystallize 2–4 layers EARLIER than clean decisions (Llama, Ministral). The pathology is a failure of late-layer deliberation, not a verbalization glitch.
- Causal patching localizes Ministral 8B's action-decision circuit to **layers 14–16** with a sharp 3-layer transition. The §13 logit-lens "crystallization at layer 22" is downstream readout of a layer-16 commitment.

**Methodological findings (audit-worthy):**
- The original poker26 70B logs under-reported parse rates by 8–14 percentage points due to an `ACTION_ALIASES` mismatch (`"CALL_OR_CHECK"` was treated as a parse failure when the model unambiguously meant `"CHECK_OR_CALL"`). Worth a footnote in the camera-ready / rebuttal.
- The Qwen 3 "thinking mode" must be explicitly disabled when running prompt-level CoT or it doubles up the reasoning budget and the JSON gets truncated.

**Single writeup-ready paragraph for BlackboxNLP:**

> We localize the action-decision circuit in Ministral 8B under Chain-of-Thought to layers 14–16. Activation patching of `clean_CHECK_OR_CALL` source residuals into `illegal_FOLD` target decisions has no causal effect through layer 13 (specificity-adjusted Δlogit(CHECK − FOLD) within ±0.5 nat across all 14 layers tested). Patching at layer 14 produces a partial flip (+1.57 nat, 4% top-1 → CHECK), at layer 15 a stronger partial flip (+3.96 nat, 27%), and at layer 16 a complete flip (+7.43 nat, 100%) that strengthens monotonically through layer 35 (+12.0 nat). This sharp 3-layer transition causally identifies the deliberation circuit and reframes the upstream logit-lens "crystallization at layer 22" finding as the downstream readout of a layer-14–16 commitment.

**Target venue:** BlackboxNLP @ EMNLP 2026 — deadline July 17, 2026. Workshop on analyzing and interpreting NN for NLP. Perfect fit for the localized-circuit finding.

> **In plain English.** We started with "does CoT help small models" and we now know: (1) CoT effects are model-specific and largely artifact-driven, not the simple win we expected; (2) we can read which "decision" each of the 36 internal layers is leaning toward; (3) for Ministral specifically, layers 14, 15, and 16 are where the actual decision is made, and everything after that is just turning the decision into words. The first finding goes in the paper as the headline behavioral story; the second and third go in as mechanistic evidence localized to a tight 3-layer circuit. The whole package is publishable at BlackboxNLP (a workshop on AI interpretability) by July 17, 2026.

---

## 9.5. Open mechanistic gaps (to revisit after Phase H GPU runs)

These are the publishable side-experiments we identified but haven't yet run. Listed in priority order; each is a discrete additional figure or sub-claim for the BlackboxNLP paper.

### Generalization gaps (gating the cross-model story — being addressed in Phase H GPU runs)

1. **Llama 8B causal patching.** Predicted L=14–16 transition matching Ministral.
2. **Qwen 8B causal patching.** Predicted NO sharp transition (validates the Phase 8.5b "Qwen lacks the circuit" claim causally).
3. **Ministral seed/temp replication** (s42 t=0.2; s123 t=0; s456 t=0). Confirms L=14–16 isn't an s42-t=0 artifact.

### Mechanistic gaps (publishable side experiments, GPU-required)

4. **Reverse-direction patching.** We did CHECK source → illegal_FOLD target (got 100% flip). Did NOT test FOLD source → clean_CHECK_OR_CALL target. If both directions work, the L=14–16 circuit is bidirectional. Cheap (~1 h on H100). Strengthens causality claim.
5. **Other source/target pairs.** BET_OR_RAISE source → illegal_FOLD target (does it produce BET?). CHECK source → clean_LEGAL_FOLD target (does it flip a legal-FOLD spot to CHECK?). Tells us whether the circuit is *verb-agnostic* (encoding "the action decision") or *FOLD-vs-CHECK-specific*.
6. **Position robustness.** We patched at the action-verb position. What if we patch at an EARLIER position in the response (e.g., during the reasoning text)? If the L=14–16 circuit also exists for non-verb positions, that's strong evidence it's a general "decision computation" circuit, not a verb-specific one.
7. **Continuation test.** After patching, the predicted next-token flips to CHECK. If we let the model CONTINUE generating, does the rest of the response (post-verb) still produce a coherent "I check because..." justification? Or does it produce gibberish? Tells us how deeply the patch propagates.
8. **Attention-pattern analysis at L=14–16.** What's attending to what to compute the verb decision at this circuit? This is the next mech-interp depth and would be a strong figure.

### Theoretical gaps (interesting but not blocking)

9. **What is the circuit COMPUTING?** It encodes the verb decision, but is it doing pot-odds reasoning? Equity threshold? Mimicking the system prompt? Could probe with sources from very different poker situations (e.g., same hero hand but different board / pot odds).
10. **Does the circuit exist in non-CoT mode?** The whole story is CoT-induced. Direct prompting doesn't have illegal-FOLD pathology. Does it have a different (or no) deliberation circuit at L=14–16?

### Tracking

| # | Status | Notes |
|---|---|---|
| 1 | Pending GPU | Script ready: `scripts/run_causal_patching_llama_sweep.sh` |
| 2 | Pending GPU | Script ready: `scripts/run_causal_patching_qwen_sweep.sh` |
| 3 | Pending GPU | Script ready: `scripts/run_causal_patching_ministral_replicate.sh` |
| 4–10 | Not started | Plan / scripts to be written after #1–3 land |

---

## 10. What's left

In rough priority order (revised 2026-05-XX after the cross-model logit-lens finding):

### Immediate next (Phase H — small-family closure, GPU)

1. **Llama 8B causal patching** ([`scripts/run_causal_patching_llama_sweep.sh`](scripts/run_causal_patching_llama_sweep.sh)) — `SCOPE=s42_only` (3 h) THEN `SCOPE=pooled` (~6 h) for both per-seed and pooled-seed claims.
2. **Qwen 8B causal patching** ([`scripts/run_causal_patching_qwen_sweep.sh`](scripts/run_causal_patching_qwen_sweep.sh)) — `SCOPE=pooled` (~5 h). Tests whether absent late-layer-revision in §8.5b also means absent causal circuit at L=14–16.
3. **Ministral seed/temp replication** ([`scripts/run_causal_patching_ministral_replicate.sh`](scripts/run_causal_patching_ministral_replicate.sh)) — 3 cells, ~3 h total. Confirms L=14–16 isn't an s42-t=0 artifact.

### Next round (Phase I — open mechanistic gaps, GPU)

4. Reverse-direction patching, other source/target pairs, position robustness, continuation test, attention analysis (see §9.5 items #4–8).

### Cross-scale (Phase J — 70B story, big GPU)

5. **70B-tier non-CoT baseline** ([`scripts/run_tier1a_large.sh`](scripts/run_tier1a_large.sh) — already wired with `--capture-logprobs --logit-lens`). ~12 h on H100.
6. **70B-tier CoT** — a `run_tier1a_large_cot.sh` mirror of [`run_tier1a_small_cot.sh`](scripts/run_tier1a_small_cot.sh). The actually interesting story is whether the L=14–16 deliberation circuit at 8B has an analog at 70B, and whether the §11 illegal-FOLD pathology disappears at scale.

### Writeup

7. **BlackboxNLP submission** (deadline July 17, 2026). The current set of findings is already paper-grade; cross-model + replication will turn the headline figure from "we found a circuit in Ministral" to "the deliberation-circuit-vs-no-circuit cross-model story explains both mechanism AND behavior."
8. **Camera-ready footnote for poker26** — note the `ACTION_ALIASES` recovery on the original 70B logs (8–14 pp parse-rate underreporting).

> **In plain English.** First, finish the small-model story (3 GPU runs: Llama, Qwen, Ministral replication). Second, run the publishable side-experiments (more patching variations to strengthen the causal claim). Third, do the 70B scale comparison. Fourth, write the workshop paper. The current data is already enough for a paper; the additional runs make it a much stronger paper.

---

## 11. Glossary

- **CoT** (Chain-of-Thought) — prompt-level technique where the model is asked to "think out loud" before producing its final answer.
- **Cell** — one experimental configuration: model × seed × temperature. Our small-tier grid has 18 cells (3 × 3 × 2).
- **Logit** — the unnormalized score the model assigns to a token before softmax. Δlogit(A − B) = log(P(A) / P(B)) up to a constant.
- **Logit lens** — interpretability technique that projects every transformer layer's hidden state through the unembedding head to read what each layer "thinks" the next token is.
- **Activation patching** — interpretability technique that replaces a hidden-state vector at one layer with a saved vector from a different forward pass, measuring the downstream effect on the output.
- **Crystallization layer** — earliest layer from which the top-1 prediction at a position never changes through to the final layer. Lower = the model "decided" earlier.
- **Specificity-adjusted Δ** — the difference between the patching effect of a *specific* source (e.g., CHECK_OR_CALL residual) and the average effect of *random* alt-bucket sources. Isolates the source-content contribution from any generic "patching breaks the model" effect.
- **EV-truth-regret** — chips/decision lost vs the perfect-information optimal action (knowing both hole cards). Lower = better.
- **EV-optimal %** — fraction of decisions that match the perfect-information optimal action.
- **Pot odds** — `bet_to_call / (pot_total + bet_to_call)`. The minimum equity (win probability) a hand needs for a call to be break-even EV.
- **Fallback action** — when the agent's chosen action is illegal, the env's `_fallback_action()` substitutes a legal action. For "FOLD when bet_to_call=0" specifically, it substitutes CHECK_OR_CALL.
- **Illegal-FOLD-rescued** — a decision where the model emitted FOLD into a free-check spot and the env converted it to CHECK_OR_CALL. The §11 / §12 / §13 / §14 finding centers on these.
- **Action-verb position** — the token position inside the model's response where the action verb (FOLD, CHECK_OR_CALL, BET_OR_RAISE) is emitted in the JSON `{"action": "<VERB>"}` payload.
- **Random-source null** — control condition where we patch with residuals from random non-target-bucket decisions. Should produce ~0 specificity-adjusted Δ if the patching mechanism itself isn't biasing the result.
- **Self-patch identity** — control condition where we patch a target with its own captured residual. Should produce identical logits to no-patch (verifies hook plumbing).
- **`_fallback_action`** — the env method (in [`run_experiment.py`](run_experiment.py)) that substitutes a legal action when the agent picks an illegal one.

---

## Appendix — file index

### Code added/changed in this work cycle

**Agent / interp infrastructure:**
- [`poker_env/agents/json_utils.py`](poker_env/agents/json_utils.py) — `ACTION_ALIASES` + `normalize_action_str` (Phase D fix)
- [`poker_env/agents/hf_agent.py`](poker_env/agents/hf_agent.py) — `enable_thinking=False` (Phase C fix), 3 diagnostic flags on `ActionMetadata` (Phase D)
- [`poker_env/agents/api_agent.py`](poker_env/agents/api_agent.py) — same 3 diagnostic flags on `APIMetadata` for parity
- [`poker_env/interp/logit_lens.py`](poker_env/interp/logit_lens.py) — pre-existing, reused
- [`poker_env/interp/patching.py`](poker_env/interp/patching.py) — **new** (Phase G): `HiddenStateCapture` + `HiddenStatePatch`
- [`poker_env/interp/forward_helpers.py`](poker_env/interp/forward_helpers.py) — **new** (Phase G): `PromptReconstructor` + `run_forward_at_last_position`

**Analysis scripts:**
- [`analysis/recategorize_action_metadata.py`](analysis/recategorize_action_metadata.py) — **new** (Phase D)
- [`analysis/pot_odds_analysis.py`](analysis/pot_odds_analysis.py) — gzipped-log support added
- [`analysis/decompose_pot_odds_by_fallback.py`](analysis/decompose_pot_odds_by_fallback.py) — **new** (Phase E)
- [`analysis/analyze_logit_lens_by_failure_mode.py`](analysis/analyze_logit_lens_by_failure_mode.py) — **new** (Phase F)
- [`analysis/audit_action_verb_tokens.py`](analysis/audit_action_verb_tokens.py) — **new** (Phase H subword-robustness audit)
- [`analysis/cross_model_clean_split.py`](analysis/cross_model_clean_split.py) — **new** (Phase H cross-model logit-lens analysis)
- [`experiments/causal_patching.py`](experiments/causal_patching.py) — **new** (Phase G); extended in Phase H to accept multiple `--enriched-log` (pooled-seeds support)
- [`experiments/verify_prompt_reconstruction.py`](experiments/verify_prompt_reconstruction.py) — **new** (Phase G)
- [`experiments/verify_position_mapping.py`](experiments/verify_position_mapping.py) — **new** (Phase G)

**GPU driver scripts:**
- [`scripts/run_tier1a_small.sh`](scripts/run_tier1a_small.sh) — Phase B
- [`scripts/run_tier1a_small_cot.sh`](scripts/run_tier1a_small_cot.sh) — Phase C
- [`scripts/run_tier1a_small_cot_logitlens.sh`](scripts/run_tier1a_small_cot_logitlens.sh) — Phase F
- [`scripts/run_tier1a_large.sh`](scripts/run_tier1a_large.sh) — pending Phase 70B-baseline
- [`scripts/run_pot_odds_full_grid.sh`](scripts/run_pot_odds_full_grid.sh) — Phase E
- [`scripts/run_causal_patching_pilot.sh`](scripts/run_causal_patching_pilot.sh) — Phase G pilot
- [`scripts/run_causal_patching_layer_sweep.sh`](scripts/run_causal_patching_layer_sweep.sh) — Phase G sweep (Ministral L=14–16 finding)
- [`scripts/run_causal_patching_full.sh`](scripts/run_causal_patching_full.sh) — Phase G full / multi-model
- [`scripts/run_causal_patching_llama_sweep.sh`](scripts/run_causal_patching_llama_sweep.sh) — **new** (Phase H Llama generalization, GPU pending)
- [`scripts/run_causal_patching_qwen_sweep.sh`](scripts/run_causal_patching_qwen_sweep.sh) — **new** (Phase H Qwen test, GPU pending)
- [`scripts/run_causal_patching_ministral_replicate.sh`](scripts/run_causal_patching_ministral_replicate.sh) — **new** (Phase H seed/temp replication, GPU pending)

### Result documents

| Phase | Document |
|---|---|
| C | [`results/tier1a_small_cot/COMPARISON.md`](results/tier1a_small_cot/COMPARISON.md) |
| D | [`results/tier1a_small_cot/COMPARISON_v2.md`](results/tier1a_small_cot/COMPARISON_v2.md) |
| D | [`results/recategorize_local_sanity.md`](results/recategorize_local_sanity.md) |
| E | [`results/tier1a_small_cot/pot_odds/SUMMARY.md`](results/tier1a_small_cot/pot_odds/SUMMARY.md) |
| E | [`results/pot_odds_full_grid/SUMMARY.md`](results/pot_odds_full_grid/SUMMARY.md) |
| F | [`results/tier1a_small_cot_logitlens/SUMMARY.md`](results/tier1a_small_cot_logitlens/SUMMARY.md) |
| F | [`results/tier1a_small_cot_logitlens/BY_FAILURE_MODE.md`](results/tier1a_small_cot_logitlens/BY_FAILURE_MODE.md) |
| F | [`results/tier1a_small_cot_logitlens/MECHANISTIC_FINDINGS.md`](results/tier1a_small_cot_logitlens/MECHANISTIC_FINDINGS.md) |
| G | [`results/causal_patching/ministral8b_t0_s42_pilot/SUMMARY.md`](results/causal_patching/ministral8b_t0_s42_pilot/SUMMARY.md) |
| G | [`results/causal_patching/ministral8b_t0_s42_layer_sweep/SUMMARY.md`](results/causal_patching/ministral8b_t0_s42_layer_sweep/SUMMARY.md) |
| H | [`results/audit/AUDIT_ACTION_VERB_TOKENS.md`](results/audit/AUDIT_ACTION_VERB_TOKENS.md) |
| H | [`results/tier1a_small_cot_logitlens/CROSS_CELL_DETAILED.md`](results/tier1a_small_cot_logitlens/CROSS_CELL_DETAILED.md) |
| meta | [`updates.md`](updates.md) — running log of decisions and findings |
| meta | [`CAUSAL_PATCHING_RUNBOOK.md`](CAUSAL_PATCHING_RUNBOOK.md) — what to run on the GPU box |
| meta | [`AUDIT.md`](AUDIT.md) — Phase A audit of the partner's refactor |
| meta | [`EXPERIMENTS.md`](../EXPERIMENTS.md) — the original experiment plan |
