# Updates — Tier 1A.small results audit

> **Date:** 2026-05-02 (initial); 2026-05-03 (CoT pilot follow-up + parser bug)
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
2. **Re-run Qwen 8B CoT** as part of the full small-tier CoT grid (next item) — no need for a separate 2-cell sanity since the change is gated by registry flag and was verified with a unit-style smoke test of `get_config()`.
3. **Run the full small-tier CoT grid** (3 models × 3 seeds × 2 temps = 18 cells, mirroring `run_tier1a_small.sh` but with `--cot`). Llama's −99.4-point shift in trash-one-hot rate is way past the "non-trivial CoT effect" threshold, so the full grid is justified. Recommended as a copy of `run_tier1a_small.sh` with `--cot` added to the `run_experiment.py` invocation.
4. **Add a degeneracy-report analysis script** (`analysis/degeneracy_report.py`) that surfaces the metrics that actually distinguish 8B failure modes (parse rate, trash-one-hot rate, action entropy, action distribution by hand strength). The current `compute_pce_distribution.py` outputs are technically valid but the wrong instrument for this regime.
5. **Skip JS-distance comparison for small tier in the writeup** — explain that the metric requires a non-degenerate belief distribution to be meaningful.
6. **Run the 70B tier (`run_tier1a_large.sh`)** — that's where the *interesting* miscalibration story (the original poker26 finding) plays out. With the small-tier CoT result clean, the eventual 70B-CoT vs 70B-direct comparison becomes much sharper.

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
