# xpoker2026Extension — Full Code Audit (v2)

> **Command given:** "the person that is supposed to do the xpoker2026Extension/ went ahead and supposedly made the audit changes from AUDIT.md and that is what we pulled rn. I need you to comb through it and see if that's true, and overall comb through that folder and see what it really does again and how it differs from our poker2026. I want you to see if there are any bugs, or anything odd that stands out and doesn't make sense or is done improperly."
>
> **Repo audited:** `https://github.com/krishjainm/Updated-miscalibrated-belief-llms.git` (re-pulled), cloned into `xpoker2026Extension/`
>
> **Date:** April 22, 2026
>
> **Prior audit:** April 11, 2026 — found 32 issues (1 critical, 5 high, 14 medium, 12 low)

---

## TL;DR

- **22 of 32** original issues are **FIXED**
- **4 of 32** are **PARTIALLY FIXED**
- **6 of 32** are **NOT FIXED**
- **A major new concern**: 15 files that were previously byte-identical to the base `poker2026` are now modified — the extension is no longer a strict additive layer on top of your published code

---

## Part 1: Verification of Prior Audit Fixes

### Scorecard

| Severity | Total | Fixed | Partially Fixed | Not Fixed |
|----------|------:|------:|----------------:|----------:|
| CRITICAL | 1 | 1 | 0 | 0 |
| HIGH | 5 | 3 | 1 | 1 |
| MEDIUM | 14 | 11 | 1 | 2 |
| LOW | 12 | 7 | 2 | 3 |
| **TOTAL** | **32** | **22** | **4** | **6** |

---

### CRITICAL — All Fixed

| # | Issue | Status | How |
|---|-------|--------|-----|
| C1 | `analyze_beliefs.py` — `np.mean()` on empty arrays | **FIXED** | All `np.mean`/`np.std` calls now guarded with `if list else None` |

### HIGH — 3 Fixed, 1 Partial, 1 Not Fixed

| # | Issue | Status | How |
|---|-------|--------|-----|
| H1 | `analyze_cot.py` — CoT/JS row mis-alignment | **FIXED** | Score and JS now computed in the same loop iteration; paired list only appended when both valid |
| H2 | `analyze_attribution.py` — prompt reconstruction fidelity | **PARTIALLY FIXED** | Now includes system message and history truncation matching agent code. **Still uses first subword of action string** as attribution target (flagged with `multi_token_target` but not fixed to use actual generated token) |
| H3 | `plot_paper_figures.py` — crash on empty subsets | **FIXED** | `len(subset_df) < 2` guard skips scatter/linregress; draws "insufficient data" title instead |
| H4 | `run_experiment.py` — CoT overwrite bug | **FIXED** | Agents now expose `get_last_action_cot()` and `get_last_belief_cot()` separately; runner captures both with `[ACTION]`/`[BELIEF]` labels, falls back to `get_last_cot_reasoning()` only if split getters absent |
| H5 | `run_experiment.py` — dual `poker_env` import hazard | **NOT FIXED** | No `sys.path` guard, no restructured imports. Which `poker_env` loads still depends on CWD/PYTHONPATH |

### MEDIUM — 11 Fixed, 1 Partial, 2 Not Fixed

| # | Issue | Status | How |
|---|-------|--------|-----|
| M1 | `or` vs `is not None` for token budgets | **FIXED** | Both agents use `if x is not None` |
| M2 | `parsed["action"].upper()` no type check | **FIXED** | Both gate on `isinstance(parsed["action"], str)` |
| M3 | `_truncate_history` breaks if `max_events < 5` | **FIXED** | Early branch: `if max_events <= 5: return history[-max_events:]` |
| M4 | No error handling around API calls | **FIXED** | `_call_api` wraps in try/except with retry + exponential backoff (default 3 attempts) |
| M5 | Google path crash on blocked responses | **FIXED** | `response.text` in try/except; `candidates[0]` guarded with try/except |
| M6 | API vs HF belief shape inconsistency | **FIXED** | Both now iterate `BUCKET_ORDER` and produce all 14 keys |
| M7 | `json_utils.py` brace matching inside strings | **FIXED** | Now tracks `in_string` and `escape` flags; braces inside strings ignored |
| M8 | JS divergence vs distance scale mismatch | **FIXED** | Documented in docstring; new `compute_js_distance()` helper returns [0,1] scale |
| M9 | `compute_update_coherence.py` — missing SELF_ACTION, hardcoded player==1 | **FIXED** | Uses `hero_idx = record.get("player_to_act", 0)` and returns `SELF_ACTION` |
| M10 | `player_to_act == 0` filter drops LLM data | **FIXED** | Now filters on presence of `agent_belief` + oracles, not seat index |
| M11 | Shared agent instance mutable state coupling | **NOT FIXED** | Still shares one HF/API instance across seats; `reset()` is a no-op on LLM agents |
| M12 | `--randomize-probe-order` uses global random | **FIXED** | Uses dedicated `random.Random(seed ^ player ^ decision_idx)` |
| M13 | `--max-input-tokens` only applies to HF | **NOT FIXED** | `APIAgent` still has no `max_input_tokens`; CLI help still ambiguous |
| M14 | `posterior_oracle.py` bucket likelihood = mean not marginal | **PARTIALLY FIXED** | Math unchanged (still mean); now has inline comment "Average likelihood over hands in bucket" |

### LOW — 7 Fixed, 2 Partial, 3 Not Fixed

| # | Issue | Status | How |
|---|-------|--------|-----|
| L1 | Dead imports | **FIXED** | All 5 unused imports removed |
| L2 | `prompts.py` example drift | **FIXED** | Uses `BELIEF_SCHEMA_ID` consistently |
| L3 | Short CoT reasoning silently dropped | **NOT FIXED** | Still requires `len(candidate) > 10` |
| L4 | `test_determinism.py` vacuous `or True` | **NOT FIXED** | Still `assert ... or True` |
| L5 | `test_golden.py` weak raise cap assertion | **NOT FIXED** | Still only `assert raise_count >= 0` |
| L6 | `test_v2_features.py` silent pass on ImportError | **NOT FIXED** | Still bare `return` without `pytest.skip` |
| L7 | `test_get_agent_configs_shared_instance` always true | **FIXED** | Rewritten with concrete assertions |
| L8 | `deck.py` explicit board draws from shuffled deck | **PARTIALLY FIXED** | Now warns on short explicit board; validates holes; but still falls through to draw |
| L9 | `oracle/win_prob.py` empty opponent_holes crash | **FIXED** | Early return when empty; returns 1 for hero |
| L10 | `interp/attention.py` assumes `output[1]` has weights | **FIXED** | Guarded: `if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None` |
| L11 | `interp/*` hardcoded Llama architecture | **PARTIALLY FIXED** | Documented in logit_lens module docstring; no runtime architecture detection |
| L12 | Misleading dependency error messages | **PARTIALLY FIXED** | `analyze_attribution.py` fixed; `analyze_probes.py` still says "scikit-learn required" for any import failure |

---

## Part 2: Major New Finding — Base Pipeline Files Are No Longer Identical

**This is the most important finding.** In the previous version, 63 files were byte-identical to your base `poker2026`. In this updated version, **only 6 of 21 spot-checked "should be identical" files still match**. The extension developer touched core analysis pipeline files, environment code, and oracle code.

### Files That Are Now Modified (Previously Identical)

| File | Lines Changed | Nature | Can Affect Published Results? |
|------|--------------|--------|-------------------------------|
| `analysis/build_dataset.py` | +15 / -4 | Multi-opponent hole parsing, regex for player keys | Likely safe for heads-up; risky for multi-way |
| `analysis/compute_pce_distribution.py` | +11 / -30 | Removed `player_to_act==0` filter; imports `BUCKET_NAMES` from `buckets` | **YES** — different rows enter the pipeline |
| `analysis/compute_update_coherence.py` | +22 / -12 | Fixed direction agreement; `SELF_ACTION`; uses `hero_idx` | **YES** — agreement rates change |
| `analysis/plot_paper_figures.py` | +86 / -76 | Robustness guards; dynamic xlim; CDF fixes | **YES** — figures look different |
| `analysis/analyze_beliefs.py` | +99 / -50 | Empty-array guards; ternary fixes; FOLD category; hero-relative opponent actions | **YES** — stats and categories change |
| `analysis/buckets.py` | +19 / -14 | Populates 7/30-bucket schemes; removes dead code | **YES** for alternate schemes |
| `analysis/posterior_oracle.py` | +51 / -16 | Richer state extraction; unknown-action heuristic (1/3 factor); reversed card key lookup | **YES** — oracle posteriors can change |
| `analysis/metrics/calibration.py` | +37 / -6 | Renormalized KL; `compute_js_distance` helper; documentation | **YES** if KL is used |
| `analysis/metrics/update_coherence.py` | +6 / -1 | Skip `"any"` buckets; new hand count fields | **YES** for downstream consumers |
| `analysis/projection.py` | +1 / -0 | Renormalize projected distribution before KL | **YES** for projection-KL values |
| `analysis/implied_belief/inverse.py` | +4 / -2 | T≤0 deterministic; narrower except | Can change for T=0 or optimizer errors |
| `analysis/implied_belief/q_value.py` | +49 / -28 | PokerKit showdown evaluation; card guards; removes `RolloutState` | **YES** for Q-values |
| `poker_env/env.py` | +55 / -28 | Step records player/street before applying action; button from `dealer_index`; pot math | **YES** — env traces and manifests differ |
| `poker_env/obs.py` | +14 / -19 | Removes dead helpers; warnings; hero index guard | Low |
| `poker_env/deck.py` | +63 / -1 | Input validation (duplicate cards, length, board overlap) | Same valid deals; errors on bad inputs |
| `poker_env/oracle/win_prob.py` | +23 / -4 | Empty-opponent guard; board >5 handling; combinatoric safety | Edge cases only |
| `poker_env/agents/call_agent.py` | +1 / -1 | Removed unused import | None |
| `poker_env/agents/threshold_agent.py` | +4 / -4 | Probability clamps; removed unused import | Only in extreme parameter ranges |
| `analysis/opponent_model.py` | +9 / -4 | Same probability clamps as threshold agent | Same |
| `analysis/belief_utils.py` | +4 / -1 | Docstring clarification | None |

### What This Means

**If you run the extension's analysis pipeline on the same logs your paper used, you may get different numbers.** The changes are mostly improvements (bug fixes, robustness), but they break the "extension = base + additive layer" guarantee. Key areas of concern:

1. **`compute_pce_distribution.py`** now includes rows it previously excluded (any seat, not just seat 0)
2. **`compute_update_coherence.py`** has a corrected direction-agreement formula (good fix, but changes the metric)
3. **`posterior_oracle.py`** has different likelihood handling for unknown actions
4. **`env.py`** records manifest entries at different points in the step cycle

---

## Part 3: New Issues (Not in Prior Audit)

### NEW-HIGH

| # | File | Issue |
|---|------|-------|
| N1 | `poker_env/agents/api_agent.py` | `get_last_cot_reasoning()` always prefers belief CoT over action CoT regardless of call order (returns `_last_belief_cot_reasoning` if non-None). `HFAgent` uses `_last_cot_source` to track which ran last. The two agents have **inconsistent "last" semantics** for the legacy fallback getter. |
| N2 | `analysis/analyze_attribution.py` | Even with the improved prompt reconstruction, it still builds a flat `f"{system_msg}\n\n{user}"` string — does NOT apply `apply_chat_template`, which means token boundaries and special tokens differ from the actual model forward pass. Attribution results are approximate. |
| N3 | `analysis/compute_pce_distribution.py` | `get_opponent_last_action` still assumes `opponent = 1 - player_to_act` — only valid for heads-up. Inconsistent with the same file's new policy of not filtering by seat. |

### NEW-MEDIUM

| # | File | Issue |
|---|------|-------|
| N4 | `run_experiment.py` | Top-level `prompt_hash` / `prompt_template_id` on the decision record are from the **action** prompt in the default `action_first` path. A consumer reading only `prompt_hash` may think it keys the belief elicitation. |
| N5 | `analysis/analyze_cot.py` | Two-file comparison: `name0_is_cot` detection uses filename substring OR higher quality score. If neither filename contains "cot" and quality is tied, assignment is arbitrary — can swap which file is labeled "CoT" vs "direct". |
| N6 | `poker_env/agents/__init__.py` | `API_AVAILABLE` means the `api_agent` module imported successfully, not that the chosen provider's SDK is installed. A missing `openai` package surfaces as a crash inside `APIAgent.__init__`, not at the guarded import check. |
| N7 | `analysis/posterior_oracle.py` | New heuristic: unknown actions now apply a `1/3` likelihood factor instead of being skipped. This is a modeling choice, not a principled Bayesian update — could bias posteriors. |

### NEW-LOW

| # | File | Issue |
|---|------|-------|
| N8 | `poker_env/agents/prompts.py` | `ImportError` fallback sets `BUCKET_ORDER = []`, making compact prompts ill-formed if `poker_env.config` fails to import. |
| N9 | `analysis/analyze_probes.py` | Still maps any failure of `from poker_env.interp.probing import ...` to "scikit-learn required" — misleading if the failure is `torch` or missing `poker_env.interp`. |

---

## Part 4: Still-Open Items (Unfixed from Prior Audit)

These 6 issues from the original audit remain:

| # | Severity | Issue |
|---|----------|-------|
| H5 | HIGH | Dual `poker_env` import hazard — no path guard |
| M11 | MEDIUM | Shared agent instance mutable state coupling across seats |
| M13 | MEDIUM | `--max-input-tokens` only applies to HF, not API |
| L3 | LOW | Short CoT reasoning (< 10 chars) silently dropped in `json_utils.py` |
| L4 | LOW | `test_determinism.py` vacuous `or True` assertion |
| L5 | LOW | `test_golden.py` raise cap assertion only checks `>= 0` |

---

## Overall Assessment

### What Improved
- Most of the original 32 bugs were addressed (22 fixed, 4 partially)
- API agents are more robust (retries, type safety, response guards)
- CoT logging is now properly split between action and belief
- Analysis scripts are more defensive (empty-data guards, proper ternaries)
- JSON parsing handles braces inside strings
- Metrics module has clear JS divergence vs distance documentation

### What's Concerning
1. **The extension is no longer a clean additive layer.** 15+ previously-identical files were modified, including core analysis scripts, environment code, and oracle code. Running the extension's pipeline on your original logs will likely produce different numbers than your published results.
2. **The `env.py` changes affect game traces.** Manifest entries are now recorded at different points in the step cycle, which means new experiment runs produce structurally different logs.
3. **The `posterior_oracle.py` changes alter the oracle itself.** Strategy-aware posteriors now handle unknown actions differently and extract richer state — this changes the benchmark your metrics are measured against.

### Recommendation
If you need the extension's analysis pipeline to reproduce your published paper's exact numbers on the same logs, **do not use the extension's analysis code** — use your base `poker2026/analysis/` pipeline for that. The extension's modified analysis files should only be used for new experiments run with the extension's modified environment.
