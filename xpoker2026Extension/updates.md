# Updates — Tier 1A.small results audit

> **Date:** 2026-05-02 (initial); 2026-05-03 (CoT pilot follow-up + parser bug); 2026-05-03 (Ministral action-parser dig-in + alias fix; see §11)
> **Run pulled from GPU box** in commits:
> - `f710f59` (`tier1a_small: complete 8B-family baseline + rolling-cache runbook`) — non-CoT baseline
> - `e30e01d` (CoT pilot: 6 cells, 3 models × 2 temps × 1 seed) — CoT pilot

This document captures the post-pull audit of the Tier 1A.small (8B-class) experiment, the small-model CoT pilot we ran on top of it, and a parser/token-budget bug we discovered while interpreting the pilot.

---

## 1. Data integrity (non-CoT baseline)

The GPU run wrote logs in gzip-compressed JSONL because the uncompressed total (~600 MB) was too large to commit cleanly. Verification of the pulled artifacts:

| Check | Result |
|---|---|
| 36 expected log files (3 models × 6 cells × {raw, enriched}) | ✅ all present in `logs/scaled_*.jsonl.gz` |
| `gzip -t` on every `.gz` | ✅ all pass — no corruption from transfer |
| Per-model result CSVs in `results/tier1a_small/` | ✅ `pce_<tag>_{records,summary}.csv`, `uc_<tag>.csv`, `uc_<tag>_summary.json`, plus `pce_pool_*` |
| Logs decompress to valid JSONL (each line parses) | ✅ verified by sampling first record + sample decisions across all models |

**Conclusion:** the compression was the right call and lossless; everything is readable and the analysis pipeline ran successfully on the GPU box before the push.

---

## 2. Agent config verification (the critical check)

The Tier 1A design depends on Qwen 3 8B running with thinking mode disabled, otherwise the cross-family baseline is invalid (Qwen would silently do internal CoT). Pulled the `agent_configs` block from the first record of each `t0_s42` log:

| Model | `cot_mode` | `has_thinking_mode` | `enable_thinking` | Status |
|---|---|---|---|---|
| `llama-8b` | `False` | `False` | `None` | ✅ no thinking flag (model has no such mode) |
| **`qwen-8b`** | **`False`** | **`True`** | **`False`** | ✅ **thinking explicitly disabled** |
| `ministral-8b` | `False` | `False` | `None` | ✅ no thinking flag |

The Qwen 3 thinking-mode gating logic worked exactly as designed end-to-end — the registry flag was respected, the HFAgent passed it through, and it landed in the JSONL as a permanent audit trail.

---

## 3. Headline numbers (non-CoT baseline)

Pooled across all 6 cells per model (3 seeds × 2 temps):

| Model | Decisions | Belief attempts | Parse OK | Trash argmax | Trash one-hot |
|---|---:|---:|---:|---:|---:|
| `llama-8b` | 8,989 | 4,949 | **100.0%** | **100.0%** | **100.0%** |
| `qwen-8b` | 3,571 | 2,075 | **100.0%** | **100.0%** | **100.0%** |
| `ministral-8b` | 1,644 | 844 | **5.8%** | 51.0% (of 49) | 0.0% |

**Action distributions (sanity-check that the action policy still works, even when belief reasoning fails):**
- `llama-8b`: BET 67%, CALL 30%, FOLD 3% — extremely aggressive
- `qwen-8b`: BET 54%, CALL 29%, FOLD 17%
- `ministral-8b`: BET 46%, FOLD 37%, CALL 18% — fold-heavy

So all three models DO produce actions; the failure is specifically in belief elicitation.

---

## 4. The trash-collapse — direct evidence from the logs

Both `llama-8b` and `qwen-8b` produce literally this belief on every single elicitation across all 6 cells:

```json
{"schema":"buckets_14_v1","probs":[0,0,0,0,0,0,0,0,0,0,0,0,0,1.0]}
```

100% of probability mass on bucket 13 ("trash"). 4,949 + 2,075 = 7,024 such beliefs, identical.

`ministral-8b` produces a different failure mode — when it tries, it outputs:

```json
{"schema":"buckets_14_v1","probs":[0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
```

All zeros, fails normalization, parser rejects. Happens 94.2% of the time.

---

## 5. Cross-check vs. poker26's existing 8B sanity log

Re-ran the same diagnostic on the paper's `logs/sanity_8b_t0_s42_enriched.jsonl` (50 hands, t=0, seed 42 — the original 8B sanity from the published paper):

| Metric | poker26 sanity 8B | tier1a `llama-8b` (this run) |
|---|---|---|
| Parse rate | 200/200 (100%) | 4,949/4,949 (100%) |
| Trash one-hot rate | 200/200 (100%) | 4,949/4,949 (100%) |

**Exact reproduction** at ~25× the sample size. The new pipeline behaves identically to the published one for this model, and the 8B trash-collapse is deterministic — it persists across all 3 seeds (42, 123, 456) and both temperatures (0.0, 0.2).

---

## 6. CoT pilot — does CoT rescue the 8B trash-collapse?

Ran a 6-cell pilot (3 models × 2 temps × seed 42 × 50 hands) at commit `e30e01d`. Same opponent, same hands, same seed — only `--cot` was added. Side-by-side numbers from `results/tier1a_small_cot_pilot/SUMMARY.md`:

| Model | Mode | Belief attempts | Parse rate | Trash one-hot (of parsed) | Mean reasoning chars |
|---|---|---:|---:|---:|---:|
| `llama-8b` | no_cot | 1,833 | **100.0%** | **100.0%** | 0 |
| | **cot** | 348 | **99.7%** | **0.6%** | 809 |
| | **delta** | | -0.3p | **-99.4p** | |
| `qwen-8b` | no_cot | 579 | 100.0% | 100.0% | 0 |
| | **cot** | 608 | **1.3%** ← *parser/budget bug, see §7* | 0.0% (n=8) | 328 |
| | **delta** | | -98.7p | -100.0p | |
| `ministral-8b` | no_cot | 444 | 5.0% | 0.0% | 0 |
| | **cot** | 309 | **99.0%** | 0.0% | 552 |
| | **delta** | | **+94.1p** | +0.0p | |

### Three model-specific stories

**Llama 8B — CoT *rescues* belief reasoning.** Without CoT, 100% trash-collapse. With CoT, only 2 of 347 parsed beliefs are one-hot trash. The model now produces ~800 chars of high-quality reasoning per decision (mentions opponent actions in 100% of traces, board texture 96–97%, hand combos 98–99%; avg quality score 0.98 from `analyze_cot_llama8b.json`). Action policy also shifts substantially: BET 67% / CALL 30% / FOLD 3% (aggro-bot) → CALL 53% / BET 31% / FOLD 16% (more conservative). **But the resulting beliefs are still nowhere near calibrated:** mean JS to StrategyAware is 0.44–0.47, vs the paper's 70B no-CoT anchor of 0.014. CoT enables 8B to *attempt* the task; it does not enable 8B to *succeed* at it.

**Ministral 8B — CoT *rescues* format compliance.** Without CoT, only 5% of belief attempts produce valid JSON. With CoT, 99% do. The reasoning step seems to act as a "warmup" that gets the model into a state where it can then produce structured output. Reasoning quality is high (0.99 quality score, 552 chars). JS distances are still bad (0.52–0.57 — even worse than Llama's CoT numbers), so we have the same pattern: CoT enables the model to attempt the task, but the resulting distributions aren't well-calibrated.

**Qwen 8B — CoT *appears* to break the model. It does not.** This is a parser/token-budget bug, see §7 immediately below.

### What this would mean for the paper *if the Qwen number were real*

A two-of-three "CoT helps" story would be a much stronger paper result than the original null. It is also exactly the kind of finding that needs sharper experimental design before publication — see §7 and §8.

---

## 7. The Qwen 8B "regression" is a parser/token-budget bug, not a model failure

When I dug into the Qwen 8B CoT raw responses, I found something suspicious: the SUMMARY says "1.3% parse rate" but `analyze_cot_qwen8b.json` reports an *action* parse rate of 2.4% with a 57% fallback rate. So both action and belief parsing crater simultaneously. That's not a normal failure mode for a competent 8B model.

I sampled raw belief responses and ran a structural diagnostic across all 6 CoT pilot cells. Here are the per-file numbers:

| File | Belief attempts | Parse OK | Mean raw chars | Has `<think>` open | Has `</think>` close |
|---|---:|---:|---:|---:|---:|
| `cot_pilot_qwen8b_t0_s42_*` | 306 | 5 (1.6%) | 2,603 | **306 (100%)** | **6 (1.96%)** |
| `cot_pilot_qwen8b_t02_s42_*` | 302 | 3 (1.0%) | 2,603 | **302 (100%)** | **8 (2.65%)** |
| `cot_pilot_llama8b_t0_s42_*` | 184 | 184 (100%) | 545 | 0 | 0 |
| `cot_pilot_llama8b_t02_s42_*` | 164 | 163 (99.4%) | 532 | 0 | 0 |
| `cot_pilot_ministral8b_t0_s42_*` | 155 | 153 (98.7%) | 425 | 0 | 0 |
| `cot_pilot_ministral8b_t02_s42_*` | 154 | 153 (99.4%) | 412 | 0 | 0 |

**Every single Qwen 8B belief response opens a `<think>` block, and 98% of them never close it.** Sampling shows what's happening: the script enables Qwen 3's *native* thinking mode (because `enable_thinking=cot_mode` and `cot_mode=True`), the model generates 1,500–3,000 characters of internal reasoning inside `<think>...</think>`, and then **runs out of token budget before ever closing the tag** — so we never get the `REASONING:` / `JSON:` block we asked for. The default belief budget under CoT is `DEFAULT_COT_BELIEF_MAX_TOKENS = 768`, and 768 tokens isn't even close to enough for a model that's been told to think internally first.

In other words: **the model is doing exactly what we asked it to do; we just starved its token budget.** Llama 8B and Ministral 8B don't have a native thinking mode, so all of their CoT output is the prompt-level REASONING/JSON we want, and 768 tokens fits comfortably.

There is also a secondary parser bug: even when `<think>` *is* properly closed, `parse_cot_response` would capture the entire `<think>` block as part of the reasoning text (it falls back to "everything before the first `{`"), which inflates reasoning-quality metrics for any thinking-mode model. Not fatal but worth fixing.

### What the bug is specifically

There are actually two design choices interacting badly:

1. **`hf_agent.py` lines 429–430** unconditionally enable Qwen 3's native thinking mode whenever `cot_mode=True`:
    ```python
    if self.has_thinking_mode:
        chat_kwargs["enable_thinking"] = bool(self.cot_mode)
    ```
   This ties native thinking to prompt-level CoT 1:1, which means under `--cot` the model is asked to reason *twice* — internally (in `<think>...</think>`) and then again at the prompt level (`REASONING: ... JSON: ...`). That's redundant *and* it makes token budgets way larger than the script anticipates.

2. **`config.py` line 105** sets `DEFAULT_COT_BELIEF_MAX_TOKENS = 768` globally, with no per-model override. 768 is fine for non-thinking models but ~1/4 of what Qwen 3 thinking mode actually needs.

3. **`json_utils.py` `parse_cot_response`** has no special handling for `<think>...</think>` blocks. If the block is closed properly, the function still treats `<think>analysis</think>REASONING:...JSON:{}` as if the entire prefix (including the `<think>` text) is reasoning.

### Recommended fixes (in priority order)

A. **Decouple `enable_thinking` from `cot_mode`.** Best option scientifically. Keep `enable_thinking=False` always, even with `--cot`. Let prompt-level CoT do the CoT work. This makes Qwen 3 directly comparable to Llama and Ministral under both `--cot` and non-`--cot`, removes the budget blowup, and removes the "double CoT" confound. Trade-off: we lose the ability to study Qwen 3's *native* thinking specifically, which arguably is a fourth condition we'd want anyway — see §8.

B. **Strip `<think>...</think>` in `parse_cot_response`** before the reasoning/JSON split, so the captured `cot_reasoning` is the explicit prompt-level reasoning only. Cheap, low-risk fix.

C. **Per-model token-budget overrides** in `MODEL_REGISTRY` (e.g. `cot_belief_max_tokens: 4096` for thinking-mode models). Required only if we choose to keep native thinking on — which I don't recommend.

The audit trail in the JSONL (`agent_configs` block) already correctly records `enable_thinking=True` for the pilot, so any re-run with the fix produces a directly comparable artifact.

### Status of fixes (2026-05-03)

**Fix A — DONE in code.** Three files changed:

| File | Change |
|---|---|
| `poker_env/agents/hf_agent.py` | `_generate` now hard-codes `chat_kwargs["enable_thinking"] = False` for any thinking-mode model, regardless of `cot_mode`. `get_config()` reflects this in the JSONL audit trail. Comment in `__init__` documents the rationale. |
| `poker_env/config.py` | Registry doc-comment updated to describe the new "always off" policy. |
| `scripts/run_tier1a_small{,_cot_pilot}.sh` | Pre-flight checks updated to assert that the registry flag is set, but reflect that `enable_thinking=False` will hold under both `cot_mode` values. |

Verified by running `HFAgent(model_id="qwen-8b", cot_mode=True).get_config()` (with mocked tokenizer/model) — `enable_thinking` is `False` for both `cot_mode={False, True}`. Llama and Ministral are unaffected (they don't have a thinking-mode flag, so `enable_thinking` is `None`, as before).

The full small-tier CoT grid (`run_tier1a_small_cot_pilot.sh` re-run, or a copy with the 18-cell grid) will now produce a clean Qwen 8B baseline directly comparable to Llama and Ministral.

**Fix B — tested post-hoc on the existing Qwen 8B pilot data.** Wrote `analysis/reparse_qwen_cot_pilot.py` that strips `<think>...</think>` blocks from the recorded `belief_metadata.raw_response` and re-runs `extract_json`. Result:

| File | Belief attempts | Original parse OK | After Fix B parse OK | Newly recovered | Unterminated `<think>` (lost) |
|---|---:|---:|---:|---:|---:|
| `cot_pilot_qwen8b_t0_s42` | 306 | 5 | 5 | **0** | 300 |
| `cot_pilot_qwen8b_t02_s42` | 302 | 3 | 3 | **0** | 294 |
| **TOTAL** | **608** | **8 (1.32%)** | **8 (1.32%)** | **0** | **594 (97.7%)** |

**Fix B alone recovers exactly 0 additional decisions on the existing data.** The reason is mechanical: 594 of 608 attempts have `<think>` opened but never closed (because the 768-token belief budget was exhausted before `</think>` was emitted), so there is no JSON for any parser to find. The 14 cases where `</think>` did close were already parsed by the original parser (because `extract_json` looks for `{...}` anywhere in the response, including after `</think>`); the only 6 that were "still broken" after Fix B failed for other reasons (`no_json_extracted` — closed `<think>` block but the model still didn't emit a complete JSON afterwards).

**Conclusion of the post-hoc test:** Fix B is *not sufficient* to rescue the existing Qwen 8B CoT pilot. Fix A — re-running with `enable_thinking=False` — is required. Fix B remains worth implementing in code separately for cleanliness of `cot_reasoning` capture in any future thinking-mode runs (e.g. if we ever explicitly enable native thinking as a deliberate experimental condition), but it is not on the critical path for the small-tier CoT story.

Output JSON: `results/tier1a_small_cot_pilot/fix_b_reparse_qwen8b.json`.

---

## 8. What this means for the paper

### The capability-floor claim is unchanged and stronger
The non-CoT result still stands: at 8B, belief elicitation fails categorically across three independently-trained model families. That claim is independent of the CoT pilot.

### The CoT result is genuinely interesting but needs one more pilot before it's reportable
The Llama-8B and Ministral-8B numbers are large effects in opposite directions (Llama: trash-collapse rescue; Ministral: format-compliance rescue) and both come with high-quality reasoning traces. That's a publishable finding.

But before reporting the Qwen number we *must* re-run Qwen 8B CoT with one of the fixes above. Otherwise we'd be claiming "CoT breaks Qwen 8B" when what we'd actually be reporting is "we under-sized our token budget for thinking-mode models."

### Suggested next pilot
Add a 2-cell mini-pilot for Qwen 8B CoT with `enable_thinking=False` (50 hands, t∈{0, 0.2}, seed 42). That gives us a clean Qwen-CoT data point comparable to Llama and Ministral. Independently, optionally add a 2-cell condition for Qwen 8B with `enable_thinking=True` *and* a 4096-token belief budget — this would let us report on Qwen 3's native thinking mode separately, which is interesting in its own right (it's a different reasoning mechanism than prompt-level CoT).

### Defensible claim once Qwen is re-run cleanly

> "At 8B, two of three model families respond to Chain-of-Thought prompting with categorical changes to belief output. Llama 3.1 8B's trash-collapse rate drops from 100% to <1% under CoT (n = 347 parsed beliefs); Ministral 8B's belief format-compliance rate rises from 5% to 99%. In both cases the resulting belief distributions remain ~30× further from the StrategyAware oracle than the 70B no-CoT baseline (mean JS-to-SA 0.44–0.57 vs 0.014), so CoT enables 8B models to *attempt* the elicitation task without enabling them to *succeed* at it. This is consistent with a verbalized-vs-internal calibration gap: at 8B, the model can produce well-formed reasoning text about opponent actions, board texture, and hand combos (mean quality score 0.98–0.99), but the resulting distribution does not reflect that reasoning."

That's a much richer story than the original null and directly addresses the reviewer concerns about (i) verbalized-vs-internal reasoning and (ii) whether CoT helps small models.

---

## 9. What's left to do

In rough priority order:

1. ~~**Fix `<think>` handling.**~~ **DONE in code (Fix A).** Native thinking is now permanently OFF in `HFAgent` for thinking-mode models, regardless of `cot_mode`. The full small-tier CoT grid will pick this up automatically. (Fix B in code is deferred — empirically zero rescue on the existing Qwen run; only worth adding for hygiene if we later enable native thinking deliberately.)
2. ~~**Re-run Qwen 8B CoT.**~~ Subsumed into the full small-tier CoT grid — done.
3. ~~**Run the full small-tier CoT grid**~~ — done in commit covered by §6/§7. Ministral seed-42 dig-in done in §11.
4. ~~**Run `analysis/recategorize_action_metadata.py` on the full CoT grid**~~ — **DONE**. Outputs in `results/tier1a_small_cot/{comparison_v2.json,COMPARISON_v2.md}`. Headline numbers now in §11 and ready for the paper writeup.
5. **Optionally** also run the recategorizer on the original poker26 70B logs (camera-ready footnote: ~10% of action-pipeline parse failures were alias-mismatches, not real failures).
6. **Add a degeneracy-report analysis script** (`analysis/degeneracy_report.py`) that surfaces the metrics that actually distinguish 8B failure modes (parse rate, trash-one-hot rate, action entropy, action distribution by hand strength). The current `compute_pce_distribution.py` outputs are technically valid but the wrong instrument for this regime.
7. **Skip JS-distance comparison for small tier in the writeup** — explain that the metric requires a non-degenerate belief distribution to be meaningful.
8. **Run the 70B tier (`run_tier1a_large.sh`)** — that's where the *interesting* miscalibration story (the original poker26 finding) plays out. With the small-tier CoT result clean, the eventual 70B-CoT vs 70B-direct comparison becomes much sharper.

---

## 10. File index

**Result CSVs** (`xpoker2026Extension/results/tier1a_small/` — non-CoT baseline):
- `pce_llama8b_{records,summary}.csv`
- `pce_qwen8b_{records,summary}.csv`
- `pce_ministral8b_{records,summary}.csv`
- `pce_pool_{records,summary}.csv` — pooled cross-model
- `uc_llama8b.csv`, `uc_llama8b_summary.json` (and same for `qwen8b`, `ministral8b`)

**Result CSVs** (`xpoker2026Extension/results/tier1a_small_cot_pilot/` — CoT pilot):
- `SUMMARY.md` — human-readable rescue verdict per model
- `degeneracy_diff.json` — full numerical diff vs non-CoT baseline
- `analyze_cot_<llama8b,qwen8b,ministral8b>.json` — CoT-specific metrics (reasoning length, quality, JS distances, parse/fallback rates)
- `pce_<llama8b,qwen8b,ministral8b>_{records,summary}.csv` — same PCE pipeline as baseline, now under CoT
- `fix_b_reparse_qwen8b.json` — post-hoc result of applying Fix B (strip `<think>`) to the existing Qwen 8B CoT logs (0 newly recovered out of 608, confirming Fix A is the real bottleneck)

**Raw + enriched logs** (`xpoker2026Extension/logs/`):
- Non-CoT: `scaled_<tag>_{t0,t02}_s{42,123,456}_informative_v2{,_enriched}.jsonl.gz` (36 files, 12 per model)
- CoT pilot: `cot_pilot_<tag>_{t0,t02}_s42_informative_v2{,_enriched}.jsonl.gz` (12 files, 4 per model)
- All gzipped, all integrity-verified

---

## 11. Ministral seed-42 action-parser dig-in (2026-05-03)

### Why we dug in
The full-grid CoT run (`results/tier1a_small_cot/COMPARISON.md`) reported Ministral 8B's action `parse_success_rate` cratering on seed 42 only:

| Cell | parse_success_rate | fallback_rate |
|---|---:|---:|
| `cot_ministral8b_t0_s42`  | 21.7% | 37.9% |
| `cot_ministral8b_t02_s42` | 23.5% | 36.0% |
| `cot_ministral8b_t0_s123` | 49.8% | 0.5%  |
| `cot_ministral8b_t02_s123`| 49.5% | 1.0%  |
| `cot_ministral8b_t0_s456` | 49.3% | 1.4%  |
| `cot_ministral8b_t02_s456`| 48.4% | 3.2%  |

Other seeds were ~98% on both metrics. The user asked me to sample raw `raw_response` values from the s42 cells and figure out whether this was a one-line parser fix or a model quirk.

### What it actually was
**It was not a JSON parser problem.** Re-running JSON extraction on the s42 raw responses succeeded essentially 100% of the time. The 37%-of-decisions "fallback" was caused by two things the old `ActionMetadata.parse_success` flag was silently conflating:

1. **Action-name alias mismatch (~47 cases on Ministral s42 alone).** Ministral emitted `{"action": "CHECK"}` and `{"action": "RAISE"}` — colloquial single-word forms that the action map only accepted as `CHECK_OR_CALL` / `BET_OR_RAISE`. So a perfectly-formed JSON response with a clearly-intended legal action got logged as `parse_success=False`. **In-code fix.**
2. **Illegal action attempts (~412 cases across the s42 cells, overwhelmingly Ministral).** The model picked `{"action": "FOLD"}` when `bet_to_call=0` made `FOLD` not a legal action (because `CHECK_OR_CALL` is free). This is a *model behavior* observation, not a parser bug — Ministral 8B under CoT exhibits an extreme conservative bias on a particular seed's hand sequence.

### Actual numbers from the full small-tier CoT grid (18 cells, n=8,749 decisions)

Ran `analysis/recategorize_action_metadata.py` on every `cot_*_enriched.jsonl.gz` and matching `scaled_*_enriched.jsonl.gz` baseline locally. Output: `results/tier1a_small_cot/{comparison_v2.json,COMPARISON_v2.md}`.

**Headline: the alias fix is real but modest in the CoT data; the illegal-FOLD behavior is the dominant story.**

Per-cell parse-rate breakdown (from `COMPARISON_v2.md`):

| Cell | n | parse_v1% | parse_v2% | Δ alias-recover | JSON-fail% | **Illegal%** |
|---|---:|---:|---:|---:|---:|---:|
| `cot_llama8b_t0_s42`     | 610 | 54.8 | 54.8 | +0.0 | 42.6 | **2.6** |
| `cot_llama8b_t0_s123`    | 596 | 53.2 | 53.2 | +0.0 | 41.1 | **5.7** |
| `cot_llama8b_t0_s456`    | 524 | 56.9 | 56.9 | +0.0 | 39.7 | **3.4** |
| `cot_llama8b_t02_s42`    | 646 | 53.1 | 53.1 | +0.0 | 41.5 | **5.4** |
| `cot_llama8b_t02_s123`   | 528 | 55.5 | 55.5 | +0.0 | 40.9 | **3.6** |
| `cot_llama8b_t02_s456`   | 542 | 56.5 | 56.5 | +0.0 | 39.5 | **4.1** |
| `cot_qwen8b_t0_s42`      | 796 | 56.4 | 56.4 | +0.0 | 42.8 | **0.8** |
| `cot_qwen8b_t0_s123`     | 603 | 56.4 | 56.4 | +0.0 | 42.0 | **1.7** |
| `cot_qwen8b_t0_s456`     | 616 | 56.0 | 56.0 | +0.0 | 42.0 | **1.9** |
| `cot_qwen8b_t02_s42`     | 804 | 56.8 | 56.8 | +0.0 | 42.3 | **0.9** |
| `cot_qwen8b_t02_s123`    | 612 | 54.2 | 54.2 | +0.0 | 43.6 | **2.1** |
| `cot_qwen8b_t02_s456`    | 568 | 56.3 | 56.3 | +0.0 | 41.7 | **1.9** |
| `cot_ministral8b_t0_s42` | **525** | **21.7** | **25.5** | **+3.8** | 40.4 | **34.1** |
| `cot_ministral8b_t0_s123`| 203 | 49.8 | 49.8 | +0.0 | 49.8 | **0.5** |
| `cot_ministral8b_t0_s456`| 209 | 49.3 | 49.3 | +0.0 | 49.3 | **1.4** |
| `cot_ministral8b_t02_s42`| **519** | **23.5** | **28.7** | **+5.2** | 40.5 | **30.8** |
| `cot_ministral8b_t02_s123`| 206| 49.5 | 49.5 | +0.0 | 49.5 | **1.0** |
| `cot_ministral8b_t02_s456`| 221| 48.4 | 48.9 | +0.5 | 48.4 | **2.7** |

CoT-vs-non-CoT delta in `parse_success_v2` (from the bottom of `COMPARISON_v2.md`):

* **17 of 18 cells: |Δ| ≤ 3.0 pts** (CoT parse rate is essentially the same as the non-CoT baseline once aliases are accounted for).
* **2 outliers: `ministral8b_t0_s42` (−26.3 pts) and `ministral8b_t02_s42` (−24.7 pts)**. Both come almost entirely from illegal-FOLD attempts (34.1% / 30.8% of decisions in those two cells).

### The illegal-FOLD behavior is purely CoT-induced (the cleanest finding here)

| Condition | Total cells | Illegal-action rate (per cell, range) | Illegal-action rate (mean) |
|---|---:|---|---:|
| **Non-CoT baseline** (`scaled_*`)   | 18 | **0.0% in every single cell** | **0.0%** |
| **CoT** (`cot_*`)                   | 18 | 0.5% – 34.1% | 5.5% (median 2.4%) |

Across **8,749 non-CoT decisions** in the matching 18 baseline cells, **zero** were "recognized but illegal" actions. Across **9,929 CoT decisions** in the same model × seed × temperature crossings, **537 were illegal** (overwhelmingly FOLD chosen when `bet_to_call=0`). This is a categorical behavioral difference, not a parser issue.

The illegal-FOLD attempts pattern by model is also worth noting: **Llama 8B does it 16-35 times per cell consistently across all six seeds × temps** (so it's a baseline CoT-induced bias), **Qwen 8B keeps it under 13 per cell across all six** (most CoT-resistant of the three), and **Ministral 8B is bimodal** — 1–6 per cell on seeds 123/456, then 160/179 per cell on seed 42 specifically. So:
* Llama: small consistent CoT-conservatism shift, all seeds.
* Qwen: barely any CoT-conservatism shift, all seeds.
* Ministral: small shift on most seeds, **catastrophic** shift on a single bad-luck seed.

### Pot-odds / EV decomposition rewrites the framing (2026-05-04)

After running `analysis.pot_odds_analysis` + `analysis.decompose_pot_odds_by_fallback` on the 4 Ministral s42 cells (CoT t=0, CoT t=0.2, baseline t=0, baseline t=0.2), the simple "CoT introduces a behavioral pathology" framing is **wrong**. Full numbers in `results/tier1a_small_cot/pot_odds/SUMMARY.md`. Headline:

| Cell × bucket | n | mean EV-truth-regret (chips/dec) | EV-optimal% |
|---|---:|---:|---:|
| CoT t=0, **clean** picks (no fallback) | 114 | **3.370** | 42.1% |
| CoT t=0, **illegal-FOLD rescued** | **179** | **0.554** | **60.9%** |
| CoT t=0.2, clean | 122 | 3.002 | 44.3% |
| CoT t=0.2, illegal-FOLD rescued | 160 | 0.586 | 51.2% |
| non-CoT t=0, clean (the only bucket) | 215 | 2.507 | 31.6% |
| non-CoT t=0.2, clean | 229 | 2.387 | 33.2% |

Two surprises here:

1. **Within each CoT cell, the illegal-FOLD-rescued bucket has 6× lower mean regret than the clean bucket.** The "pathology" decisions are paradoxically the *best* plays in the cell — the model's intent (give up on a marginal hand) is correct; its action choice (FOLD) is illegal; our `_fallback_action` plays the EV-optimal CHECK_OR_CALL on its behalf. The fallback is doing the model a favor, not papering over a mistake.

2. **Compared to the non-CoT baseline, CoT makes Ministral's *deliberate* picks WORSE by 0.6–0.9 chips/decision** (3.37 vs 2.51 at t=0). The aggregate "CoT improves Ministral" headline (mean regret 1.27 vs 1.69) is entirely a rescue artifact: subtract the rescue bucket and CoT-clean is worse than non-CoT-clean. So:

> CoT does not unambiguously help Ministral 8B's decisions. It primarily *redirects* a fraction of decisions through the env's safety net, where they happen to land on the EV-optimal action. Without that net (e.g. in an env that respected the literal FOLD), CoT would cost Ministral measurable EV.

### Updated reportable framings

| Was (§11 first pass) | Is (after pot-odds decomp) |
|---|---|
| "CoT introduces an illegal-action behavioral artifact at 8B" | "CoT shifts small-model action distributions toward conservative play, including illegal FOLDs into free-check spots; the env safety net rescues these to the EV-optimal action" |
| "CoT improves Ministral aggregate decision quality" | "CoT's apparent improvement on aggregate EV is entirely attributable to the fallback-rescue mechanism; on cleanly-emitted actions, CoT is worse than direct prompting" |
| "Ministral seed-42 outlier is a pathology" | "Ministral seed-42 is a stress test of the fallback rescue: a single bad-luck hand sequence triggers 339 illegal-FOLD attempts, all of which the safety net catches" |

### Open analyses (now wired up; see §12 for status)

- ~~Same decomposition on Llama and Qwen s42 CoT cells.~~ — superseded by the **full-grid** pot-odds run below (covers all 36 cells, not just s42).
- ~~Same decomposition across all 18 CoT cells.~~ — superseded by the full-grid run, which cross-joins per-cell pot-odds output with `action_metadata.fallback_used` to get the `clean / illegal_fold_rescued / other_fallback` decomposition for every cell.

This is a defensible, publishable observation that the original `parse_success_rate` flag was hiding. Suggested writeup framing: see "Updated reportable framings" table above — the previous "CoT introduces a behavioral artifact" line is preserved in the table for contrast but is no longer the right characterization on its own.

### Side discovery from running the recategorizer on the original poker26 70B sanity logs
While verifying the script on data we already had locally, I discovered the same alias issue is also present in the **original poker26 70B sanity logs** that backed the published paper:

| Log | n decisions | parse_v1% (as logged) | parse_v2% (alias-aware) | recovered |
|---|---:|---:|---:|---:|
| `phase2_70b_t02_s42_informative_v2_enriched`  | 2,345 | 45.2 | **57.8** | **+12.6 pts** |
| `phase2_70b_t0_s42_informative_v2_enriched`   | 2,546 | 45.7 | **56.8** | **+11.0 pts** |
| `sanity_70b_t02_s42_informative_enriched`     |   331 | 43.8 | **56.5** | **+12.7 pts** |
| `sanity_70b_t0_s42_informative_enriched`      |   384 | 43.8 | **57.6** | **+13.8 pts** |
| `sanity_70b_t02_s123_informative_enriched`    |   333 | 47.7 | **55.9** |  +8.1 pts |
| `sanity_70b_t0_s123_informative_enriched`     |   353 | 47.0 | **56.1** |  +9.1 pts |
| `sanity_8b_t0_s42_enriched` (8B, control)     |   450 | 44.4 | 44.4    |  +0.0 pts |

The 70B models systematically emit `CALL_OR_CHECK` (reversed word order) and `CALL_OR_CALL` (duplicate) ~10% of the time. Both unambiguously refer to `CHECK_OR_CALL`. The 8B sanity log is unaffected (its 55.6% failure is the well-known JSON-degeneracy from the paper appendix), which is exactly the right control: the alias issue is a 70B-cohort vocabulary quirk, not an 8B problem.

This means **roughly 10% of decisions in the published 70B sanity logs were miscategorized as parse failures** when the model was actually expressing a valid action. The headline 70B *belief* numbers are unaffected (those use a separate parser path), but the *action-pipeline* numbers were under-reporting effective parse rate by ~10 pts. Worth a footnote in the rebuttal/camera-ready.

### Code changes (in this commit)

| File | Change | Why |
|---|---|---|
| `poker_env/agents/json_utils.py` | New `ACTION_ALIASES` dict + `normalize_action_str()`. | Single source of truth for "the model meant the right action, just used a colloquial name". Maps `CHECK`, `CALL`, `BET`, `RAISE`, the gerunds, `FOLDING`, plus `CALL_OR_CHECK` / `CALL_OR_CALL` / `CHECK_OR_CHECK` / `RAISE_OR_BET`. Deliberately does **not** alias `CALL_OR_RAISE` (genuinely ambiguous). |
| `poker_env/agents/hf_agent.py` | `act_with_metadata` now applies `normalize_action_str` before the `action_map` lookup. `ActionMetadata` gains three diagnostic fields: `action_json_parsed`, `action_recognized`, `action_legal_in_context`. `_fallback_action` accepts and propagates them. `to_dict` only emits them when populated, so old-log analysis stays backward-compatible. | Decouples the three orthogonal failure modes the old `parse_success` was hiding. |
| `poker_env/agents/api_agent.py` | Same alias normalization + same three new diagnostic fields on `APIMetadata`, for parity. | Closed-model logs get the same audit trail going forward. |
| `analysis/recategorize_action_metadata.py` (new) | Standalone, log-only re-attribution: re-parses every decision in a glob of JSONL logs with the alias-aware logic, splits failures into `JSON-fail` / `Alias-unk` / `Illegal-in-context`, writes `comparison_v2.json` and `COMPARISON_v2.md`. Optional `--baseline-glob` produces a CoT-vs-baseline column. | Lets us re-categorize already-collected data (including the original poker26 logs) without a re-run. |
| `results/recategorize_local_sanity.{json,md}` | Output of running the script on every `*_enriched.jsonl` we have locally. | Local-reproducible artifact of the discovery above. |

### Verification
- `normalize_action_str()` smoke-tested on all 11 alias variants + canonical names + 1 unknown — all map correctly, unknowns pass through.
- End-to-end mock test of `HFAgent.act_with_metadata` on 7 cases (canonical, all 4 aliases, illegal-FOLD-when-CHECK-is-legal, unrecognized-string, malformed-JSON). All pass; the new diagnostic fields correctly distinguish all three failure modes.
- `analysis/recategorize_action_metadata.py` smoke-tested on a synthetic 6-record log designed to exercise every category. All counts (`recovered_by_alias_norm`, `json_parse_failures_v2`, `alias_unrecognized_v2`, `illegal_in_context_v2`) come out exactly as designed.
- Then run on real local data (sanity_70b_*, sanity_8b_*, phase2_70b_*); the unprompted ~10pt recovery on every 70B cell with zero recovery on the 8B control cleanly cross-validates the script.

### What this changes for the small-tier CoT story

For Ministral s42 specifically (now from the actual recategorized logs, not estimates):

| Quantity | What it measures | Ministral s42, t=0 (n=525) | Ministral s42, t=0.2 (n=519) |
|---|---|---:|---:|
| `parse_success_v1` | Old combined flag (json AND recognized AND legal) | 21.7% | 23.5% |
| `parse_success_v2` | Same, after alias normalization | **25.5%** | **28.7%** |
| `recovered_by_alias_norm` | Decisions rescued purely by the in-code alias fix | +3.8 pts (~20 cases) | +5.2 pts (~27 cases) |
| `json_parse_failures_v2` | True JSON parse failures | 40.4% | 40.5% |
| `illegal_action_attempt_rate` | Recognized action not in `legal_actions` (almost all illegal FOLD) | **34.1%** (179 of 525) | **30.8%** (160 of 519) |

The remaining gap is the actual model-behavior story: **on this seed's specific hand sequence, Ministral 8B under CoT becomes pathologically conservative and tries to FOLD into free-check spots ~30% of the time.** That's a real, novel small-tier observation and belongs in the writeup as a behavioral finding (not as a parser bug). Per-cell numbers across the full grid are in `results/tier1a_small_cot/comparison_v2.json`.

### How to reproduce (the raw `cot_*.jsonl.gz` logs live in `xpoker2026Extension/logs/` — gitignored, ~1.2 GB)

```bash
cd xpoker2026Extension
python -m analysis.recategorize_action_metadata \
  --logs-glob 'logs/cot_llama8b_*_enriched.jsonl.gz' \
  --logs-glob 'logs/cot_qwen8b_*_enriched.jsonl.gz' \
  --logs-glob 'logs/cot_ministral8b_*_enriched.jsonl.gz' \
  --baseline-glob 'logs/scaled_*_enriched.jsonl.gz' \
  --json-out results/tier1a_small_cot/comparison_v2.json \
  --md-out   results/tier1a_small_cot/COMPARISON_v2.md
```

The script reads `.jsonl` and `.jsonl.gz` transparently (`_open_log`). Runtime ~21 s for 36 files / ~18 k decisions on local laptop.

### Bottom line
- The "Ministral seed-42 action-parser failure" was **two distinct things** the old metadata flag was conflating: ~47 cases of fixable alias mismatch (now fixed in code) and ~412 cases of model-behavior illegal-FOLD attempts (now reportable as a behavioral observation).
- The same alias-mismatch bug is present in the original poker26 70B sanity logs and silently miscategorized ~10% of decisions per cell as parse failures.
- All four follow-ups from the dig-in are landed in this commit (alias fix; metadata semantic split; recategorize script; this writeup).

---

## 12. Logit-lens × failure-mode mechanistic add-on + full-grid pot-odds (2026-05-04)

Three pieces landed together to broaden the §11 finding from a Ministral-only observation to a generalisable claim.

### 12a. `analysis/analyze_logit_lens_by_failure_mode.py` (new — answers the actual mechanistic question)

`analysis/analyze_logit_lens.py` reduces a logit-lens sidecar to a single per-cell descriptive summary (mean entropy curve, mean crystallization layer). That tells us nothing about *why* the model emitted FOLD into a free-check spot. The new script joins the sidecar (keyed on `hand_id, decision_idx`) with the enriched log's new diagnostic flags, buckets decisions into `{clean, illegal_fold, illegal_other, alias_unrecognized, json_failure}`, and per bucket reports per-layer mapped action group at the action-emission token (`FOLD / CHECK / CALL / BET / RAISE / OTHER`).

Read: if the `illegal_fold` bucket shows `CHECK` dominating early layers and `FOLD` only crossing 0.5 in the late layers, that's a **verbalization-stage failure** (the model "knew" the right call). If `FOLD` dominates from layer 0, the model is FOLD-committed top to bottom — a deeper representational issue.

Smoke-tested on synthetic data: a hand-crafted `illegal_fold` record with layers 0–23 saying `CHECK` and layers 24–31 saying `FOLD` correctly produces `crystallization_layer.mean = 24.0`, layer-0 action mix `{CHECK: 1.0}`, layer-31 action mix `{FOLD: 1.0}`. Real-data validation will follow once the GPU run produces the sidecars.

### 12b. `scripts/run_tier1a_small_cot_logitlens.sh` is now overridable

Default still 3 cells (cheapest mechanistic answer, ~3 h on H100), but `SEEDS` and `TEMPS` are now env-var-overridable — pass `SEEDS="42" TEMPS="0.0 0.2"` for the recommended 6-cell version (~6 h, captures all 339 Ministral illegal FOLDs + ~50 Llama + ~10 Qwen) or `SEEDS="42 123 456" TEMPS="0.0 0.2"` for the full 18-cell mirror (~18 h). Phase 3b in the script auto-runs `analyze_logit_lens_by_failure_mode.py` per cell and appends to `results/tier1a_small_cot_logitlens/BY_FAILURE_MODE.md`.

### 12c. `scripts/run_pot_odds_full_grid.sh` (new) — pot-odds + EV + by-fallback decomposition for ALL 36 cells

The §11 pot-odds analysis only covered 4 cells (Ministral s42 × {CoT, baseline} × {t=0, t=0.2}) because that was where the illegal-FOLD pathology was concentrated. The new driver runs `analysis/pot_odds_analysis` on every CoT and non-CoT enriched log (36 cells total) and aggregates into `results/pot_odds_full_grid/SUMMARY.md`. Wall-clock 50 min on a 4-wide Mac M-series with `--num-rollouts 30 --samples-per-bucket 2 --skip-belief-ev`. Run completed 2026-05-04.

#### Three findings the full grid produced

**(1) The illegal-FOLD rescue effect generalizes across the small-model family.**

| Model | CoT cells with any rescue | Total rescued FOLDs | Cells where rescue regret < clean regret |
|---|---:|---:|---:|
| `llama8b`     | 6/6 | 144 | 5/6 |
| `ministral8b` | 6/6 | 353 | 4/6 |
| `qwen8b`      | 6/6 |  53 | 6/6 |

The §11.6 Ministral-only finding (rescued FOLDs have lower mean regret than cleanly-emitted actions) holds for all three small models in the cells where rescues happen. The mechanism is general: when small-model CoT emits FOLD into a free-check spot, it's expressing a "this hand is weak, defend" intent that the env's `_fallback_action` honors at zero chips by playing CHECK_OR_CALL.

**(2) CoT's effect splits three different ways across models** (writeup-ready table).

| Model | Temp | EV-regret CoT/non-CoT/Δ | EV-optimal % CoT/non-CoT/Δ |
|---|---|---|---|
| `llama8b`     | 0.0 | 2.10 / 2.56 / **−0.45** | 40.7 / 49.5 / **−8.8 pp** |
| `llama8b`     | 0.2 | 2.02 / 2.46 / **−0.44** | 42.2 / 49.1 / **−6.8 pp** |
| `ministral8b` | 0.0 | 1.63 / 1.75 / **−0.12** | 35.3 / 30.2 / **+5.1 pp** |
| `ministral8b` | 0.2 | 1.57 / 1.72 / **−0.15** | 35.1 / 30.3 / **+4.8 pp** |
| `qwen8b`      | 0.0 | 2.23 / 2.12 / **+0.11** | 44.9 / 42.0 / **+2.9 pp** |
| `qwen8b`      | 0.2 | 2.20 / 2.11 / **+0.09** | 44.7 / 43.5 / **+1.3 pp** |

- **Llama**: CoT cuts mean EV-regret by ~0.45 chips/decision (real, both temps), BUT also drops EV-optimal% by 7–9 pp. CoT smooths Llama's wrong picks (closer to optimal when it's wrong) at the cost of making fewer outright optimal picks. Net EV per decision improves; sharpness regresses.
- **Ministral**: CoT cuts regret by 0.12–0.15 and raises optimal% by ~5 pp. Reads like a clean win — but the per-cell bucket table shows this is driven entirely by the s42 rescue mechanism (179 + 162 rescued FOLDs out of 353 total). On the other 4 Ministral cells the rescue count drops to 1–6 and the CoT-vs-non-CoT clean-regret comparison is essentially a wash (1.7–1.9 vs 1.7–1.9). **CoT does not robustly help Ministral; the rescue mechanism does.**
- **Qwen**: CoT *increases* mean EV-regret by ~0.1 and only marginally raises optimal% (1–3 pp). Closest to neutral; if anything mildly harmful.

**(3) CoT shortens hands.** Non-CoT cells have 2–3× more decisions per cell than CoT cells (Llama non-CoT 1374–1695 decisions vs Llama CoT 524–646). Under CoT, small models fold more often (whether legally or illegally-then-rescued), which truncates hands and shrinks the decision count. This is itself a behavioral observation independent of the EV story.

#### Reportable framing (writeup-ready)

The §11.6 reframing ("CoT's apparent EV improvement on Ministral is a rescue artifact") generalizes to a stronger claim:

> **For small models in this poker environment, CoT does not unambiguously improve decision quality; it changes the shape of the decision distribution in model-specific ways, and any aggregate EV win is partly attributable to the env's safety net catching pathological action choices.**

This is verifiable from `results/pot_odds_full_grid/SUMMARY.md` without a re-run.

### 12d. What's left

- 70B-tier non-CoT baseline (`scripts/run_tier1a_large.sh`, now also captures logprobs + logit-lens by default — see §9 item 8).
- 70B-tier CoT (a `run_tier1a_large_cot.sh` mirror is the natural next script, modeled on `run_tier1a_small_cot.sh`).
- ~~Run `analyze_logit_lens_by_failure_mode.py` on the small-tier logit-lens sidecars.~~ **DONE** — see §13.

---

## 13. Logit-lens mechanistic findings (2026-05-06, full 18-cell run)

`scripts/run_tier1a_small_cot_logitlens.sh` was run with `SEEDS="42 123 456" TEMPS="0.0 0.2"` on a Blackwell GPU box. Wall-clock ~26 h, all 18 cells produced raw logs, enriched logs, logit-lens sidecars, and (bonus, not requested) per-decision hidden-state files. Pulled clean to local: 54 logit-lens log files (18 raw + 18 enriched + 18 sidecars), 18 entropy PNGs, 18 by-failure-mode JSONs, plus aggregate `SUMMARY.md` and `BY_FAILURE_MODE.md`.

### 13a. Bug fix in the failure-mode analyzer (caught while reading the first aggregate)

The first aggregation pass produced 100% "OTHER" at every layer for every bucket. Diagnosis: the sidecar stores per-layer top-1 tokens for **every generated token** (~80–130 positions per response), not just the action-verb position. My initial script naively read position `-1` (the EOS marker `'<|eot_id|>'`) instead of finding the actual action-verb token inside the JSON payload.

Fix in `analysis/analyze_logit_lens_by_failure_mode.py::_find_action_position`:
1. Find the LAST `'"}'` close-brace token in the final layer (search the last 10 positions).
2. Walk back from it looking for the value-opening `' "'` quote (skipping `'":' ` action-key separators).
3. The action verb is at `open_pos + 1` — works for full-word verbs (`'CHECK'`, `'FOLD'`) and subword splits (`'F'+'OLD'`, `'B'+'ET'+'_OR'+'_RA'+'ISE'`) alike.

Subword matching was added to the per-layer scan (e.g. `'F'` and `'fol'` count as FOLD when checking layer projections at the verb position) so that mid-layer subword fragments don't get classified as "OTHER".

Verified by inspection of 12 records across CHECK_OR_CALL / BET_OR_RAISE / FOLD / json-failure cases — anchor lands on the correct verb 100% of the time, including for json-failures (correctly returns None → contributes no layer trajectory).

### 13b. Headline mechanistic finding

For each cell, the **crystallization layer** (action-group axis) of the action prediction at the action-verb position:

| Model | Δ (illegal_fold − clean), range across 6 cells | Direction |
|---|---|---|
| **Llama 8B**     | **−2.5 to −4.0** | illegal-FOLD crystallizes EARLIER |
| **Ministral 8B** | **−1.2 to −2.7** | illegal-FOLD crystallizes EARLIER |
| **Qwen 8B**      | −0.6 to +0.5     | no significant difference |

For Llama and Ministral, **illegal-FOLD decisions become FOLD-committed in the residual stream 2–4 layers earlier than clean decisions.** Consistent across all 12 (Llama+Ministral) cells.

### 13c. The deeper story — split clean by emitted action

The "clean" bucket pools legal FOLD, CHECK_OR_CALL, and BET_OR_RAISE. Splitting it on Ministral 8B t=0 s=42 (n=313 joined records) shows **the actual mechanism**:

| Bucket | n | Crystallization layer |
|---|---:|---:|
| `clean_CHECK_OR_CALL` |  29 | **28.8** |
| `clean_BET_OR_RAISE`  |   7 | 25.7 |
| `clean_LEGAL_FOLD`    |  98 | 23.7 |
| `illegal_FOLD`        | 179 | **22.7** |

Per-layer action-group mix at the action-verb position (Ministral, layers 22–35):

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

What this shows:

1. **All decisions start neutral** (layers 0–21: 100% OTHER — residual stream not yet vocab-aligned).
2. **By layer 22–23, a FOLD-leaning signal emerges in the residual stream for every bucket** — including decisions that will eventually emit CHECK_OR_CALL.
3. **Legal-FOLD decisions** lock in at FOLD by layer 24 and never revise.
4. **Illegal-FOLD decisions** lock in **one layer earlier** (layer 23) and are MORE confidently FOLD (100% at layer 23 vs 26% for legal FOLDs at the same layer). The model is *more certain* of the wrong FOLD than of correct FOLDs.
5. **CHECK_OR_CALL decisions are the only bucket that late-layer-revises** — they start FOLD (86% at layer 25), then CHECK climbs from 0% → 21% → 48% → 90% across layers 27–29, fully overtaking by layer 35.

### 13d. The mechanistic claim (the writeup line)

This is the **opposite** of the verbalization-failure hypothesis that motivated the run. The data does NOT show "early/mid layers say CHECK, last layer flips to FOLD". It shows:

> **Small-model CoT in this poker setting has a baseline FOLD pull in the mid-to-late residual stream (~layer 22+ for Llama and Ministral). What distinguishes outcomes is whether the late-layer deliberation circuit overrides that pull. CHECK_OR_CALL emerges only via late-layer revision; legal FOLDs lock in early; illegal FOLDs lock in even earlier and more confidently. The illegal-FOLD pathology is a failure of late-layer deliberation, not a verbalization-stage glitch.**

This is the **mechanistic explanation** for the §12c full-grid pot-odds finding ("CoT's apparent EV improvement on Ministral is a rescue artifact"). When the deliberation circuit doesn't fire, the model commits to FOLD in the residual stream from layer 23–24 onwards; the env's `_fallback_action` catches those FOLDs in free-check spots and converts them to the EV-optimal CHECK_OR_CALL.

### 13e. Why Qwen is different

Qwen's illegal-FOLD count is 4–11 per cell (vs Llama's 16–34, Ministral's 1–179). At those n, Δ ≈ 0 is consistent with both "Qwen's deliberation circuit is more uniform" and "underpowered to detect the effect". Combined with the §12c finding that Qwen *doesn't* get an aggregate EV win from CoT — slightly worse, in fact — a more uniform layer structure (less reliance on a brittle late-layer override) is the most consistent interpretation.

### 13f. Files & follow-ups

- Full per-cell × per-bucket × per-layer table: `results/tier1a_small_cot_logitlens/BY_FAILURE_MODE.md` (18 cells × 5 buckets × 32–36 layers, 2437 lines).
- Distilled findings doc: `results/tier1a_small_cot_logitlens/MECHANISTIC_FINDINGS.md`.
- Per-cell JSON: `results/tier1a_small_cot_logitlens/by_failure_mode_<cell>.json`.
- Hidden-state sidecars (`_hiddens.jsonl`) are present in `logs/` but not yet consumed — they enable causal layer-patching experiments as a follow-up.
- Open follow-ups (lower priority but publishable):
  - **Causal patching**: take a CHECK_OR_CALL decision's layer-25–29 hidden state and patch it into an illegal-FOLD decision; does the model emit CHECK?
  - **Attention-pattern study at layers 25–29 of Ministral on s42** — what attends to what to flip the residual from FOLD-leaning to CHECK-leaning?
  - 70B-tier logit-lens (already wired into `run_tier1a_large.sh`) — does the same baseline-FOLD-pull / late-layer-revision dynamic appear at scale?
