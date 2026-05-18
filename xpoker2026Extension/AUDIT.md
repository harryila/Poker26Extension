# Codebase + claims audit, pre-paper-writing

> Pulled together 2026-05-18. Three deep parallel sweeps: (i) every experiment
> driver inspected for hardcoded values / control correctness / metric
> definitions, (ii) all 139 result-directory SUMMARYs inventoried and
> cross-checked against the numerical claims in `updates.md` §14–§22,
> (iii) full design coherence + dependency-graph audit of EXPERIMENTS.md /
> JOURNEY.md / EXPERIMENTS_QUEUE.md / updates.md. This file consolidates
> findings into a paper-writing decision support document.
>
> **Bottom-line verdict**: the project's claims are largely backed by the
> experiments cited and the numbers in `updates.md` largely match the
> SUMMARYs. There are **8 specific issues** to address before the paper,
> ranging from minor docstring imprecision to one wording overclaim that
> a reviewer will catch. Section 4 lists them in priority order.
>
> **POST-AUDIT STATUS (2026-05-18, commit pending):** all 8 issues have been
> addressed. R1/R3/R4 fixed in `updates.md §22h` (claims now scoped to
> tested cells). M2/M3 fixed in `causal_patching.py` and
> `component_patching.py` docstrings + the one misleading wrapper. M1 + M5
> disclosed in `causal_patching.py` module docstring. M4 fixed in
> `direction_probe_baselines._pot_size`. M7+M8 added as CLI flags
> (`--n-permutation-trials`, `--also-fixed-threshold-random`) plumbed
> through `run_a3_cleanup.sh`. Naming clarification documented in
> `updates.md §22j`. See `updates.md §22k` for the per-item fix ledger.
> An optional A3 cleanup re-run with the new flags is queued (~15 min CPU
> on the GPU box, no model load); other optional reruns are decision-
> dependent (see Section 6 of this file).

---

## 1. Formal metric definitions (what the code actually computes)

| Metric | Exact computation | Where |
|---|---|---|
| `spec-adj Δ` (causal_patching headline) | `mean_over_(src×tgt) Δ_pairs  −  mean_over_random_sources Δ_random_target0`, where `Δ = score_logits(patched) − score_logits(baseline)` and `score_logits = LSE(CHECK_CALL family logits) − LSE(FOLD family logits)` on **raw logits** (not softmaxed) | `causal_patching.py` lines 192–207, 600–650 |
| `baseline_top1_match_rate` | Fraction of targets whose no-patch top-1 token id equals `tokenizer(raw_response)["input_ids"][verb_resp_idx]` | `causal_patching.py` lines 493–519 |
| `top-1 → FAMILY` fractions | Fraction of patched forwards whose argmax token id is in the family's single-token-id set | `causal_patching.py` lines 705–715 |
| `mean Δ(CHECK − FOLD)` (per row in by_pair.csv) | Same LSE-on-raw-logits Δ as above | same |
| `R² in-sample` (belief probe) | `sklearn Ridge(α=1.0).score(X, Y)` — uniform-weighted multi-output R² | `belief_direction_probe.py` line 208 |
| `R² held-out` (5-fold CV) | KFold; per-fold: fit Ridge on train indices, predict test; per-output R² then averaged across outputs; reported = mean±std across folds | `belief_direction_probe.py` lines 227–247 |
| `cos(w_verb, principal belief)` | `weight_vec` (cached from probe NPZ) dotted with `Vt[0]` from SVD of `W ∈ R^{14×d}` (Ridge coef matrix) | `belief_direction_probe.py` lines 254–277 |
| `learned probe CV acc` | StratifiedKFold(5) + LogisticRegression(C=0.01) over balanced/unbalanced residuals | `direction_probe_baselines.py` |
| `permuted-label CV acc` | Same as above but with `y_shuffled` (single shuffle, not multiple trials) | same |
| `random-direction acc` (best threshold) | For each of N random unit vectors, scan all possible split thresholds on 1-D projection and report MAX accuracy (oracle threshold — upper bound) | `direction_probe_baselines.py` lines 155–191 |
| `head ablation Δ` | `score_logits(zero-ablated)` − `score_logits(baseline)` for each target, averaged per head set | `head_ablation.py` |
| `attention-mask Δ` | `score_logits(legal-actions-tokens masked)` − `score_logits(baseline)`, eager attention, all heads & layers | `attention_mask_ablation.py` |

**Two of these are NOT what their docstrings literally say:**

1. `score_logits` docstring says "log-prob (softmax-normalized) per family." Actual: `logsumexp` on **raw logits** (not normalized by partition function). The Δ across two states (patched vs baseline) IS still interpretable as a logit shift — but the wording "log-prob" is misleading. Reviewers familiar with mech-interp will read `LSE(logits)` as "log of unnormalized probability mass" which differs from the docstring claim by an additive constant per token position. **Fix: docstring update; the numbers stand.**
2. `causal_patching` docstring claims self-patch identity tolerated to `1e-4`. Actual code warns only when drift > `1e-2` (100× looser). Practically nothing in our runs exceeds even `1e-4`, but if a future cell drifts at `1e-3` the gate won't fire. **Fix: tighten code or relax docstring to match.**

---

## 2. Results inventory + cell-count table

### 2.1 Bulk inventory

- **139 primary mechanistic summary artifacts** under `results/` (121 `SUMMARY.md` + 18 `SUMMARY_components.md`).
- 64 `causal_patching/` subdirs cover: 8 pooled CoT sweeps (Llama/Ministral/Qwen × {layer sweep, reverse pilot}), 9 per-seed cells (3 models × {s42, s123, s456}), 18 verb-pair filtered cells, 12 Tier 4 mech cells (Qwen 4 + Ministral 4 + Llama 3 + 1 missing), 4 component/head subset cells, plus zero-ablation and verb-generality variants.
- All `causal_patching/` cells write `by_pair.csv` so re-aggregation is possible.
- All `head_ablation/` cells write `by_target.csv`.

### 2.2 Paper-critical cell-count table

| Claim cluster | Cells | Pairs/samples per cell | Seeds | Backing dirs |
|---|---|---|---|---|
| Cross-model L\* exists (Phase H/I) | 3 (Llama L=14, Ministral L=16, Qwen L=23) at residual level | 10×30 = 300 | 3 pooled | `{llama,ministral,qwen}8b_t0_pooled_layer_sweep/` |
| Cross-seed L\* concordance (Qwen) | 3 (s42, s123, s456) | 10×{4,9,11} | 1/cell | `qwen8b_t0_s*_replicate/` |
| Cross-seed L\* concordance (Llama) | 3 (s42, s123, s456) | 10×30 | 1/cell | `llama8b_t0_s42_layer_sweep/`, `s123_replicate/`, `s456_replicate/` *(note: s42 is "layer_sweep" not "replicate" — naming inconsistency)* |
| Llama head decomposition | 1 pooled + 3 per-seed | 10×30 | 3 | `llama8b_l14_components{,_s42,_s123,_s456}/` |
| Llama head triplet/quartet | 2 (triplet h5/h23/h24, quartet h2/h5/h23/h24) | 10×30 | pooled 3 | `llama8b_l14_head_{triplet,quartet}/` |
| Ministral head decomposition | 4 cells (L=14/15/16/17 components) + triplet + sextet | 10×30 | pooled 3 | `ministral8b_l1{4,5,6,7}_components/`, `_l16_head_{triplet,sextet}/` |
| Qwen head decomposition | 2 cells (L=23, L=24 components) | 10×30 | pooled 3 | `qwen8b_l2{3,4}_components/` |
| Verb-pair circuit (Phase I C1) | 6 directions × 3 models (partial) | 10×30 | 3 | `*_verb_generality_*/`, `*_verbpair_*/` |
| Non-CoT layer sweep | 3 (1 per model, s42 only) | 10×~10 | 1 | `*_nocot_layer_sweep_s42*/` |
| Non-CoT verb-pair matrix | 8 cells (4 dirs × 2 models = Llama, Qwen; Ministral skipped) | 10×~5 | 1 | `{llama,qwen}8b_nocot_verbpair_*_filtered/` |
| Non-CoT clean→clean | 2 (Llama L=14, Qwen L=23) | 10×~30 | 1 | `*_nocot_clean_to_clean_l*/` |
| Per-seed Llama heads at L=14 | 3 (s42, s123, s456) | 10×30 | 1/cell | `llama8b_l14_components_s*/` |
| Reverse-direction Llama L=14 | 1 | 10×30 | 3 pooled | `llama8b_l14_components_reverse/` |
| Direction probe (verb dir) | 3 (Llama L=14, Ministral L=16, Qwen L=23) | n=596–600 residuals | 3 pooled | `direction_probe/*_l*/` |
| Direction probe baselines (Phase O original) | 3 | same residuals | 3 pooled | `direction_probe_baselines/*8b_l*.md` |
| Direction probe baselines (Phase P cleanup) | 3 | same residuals (with balancing) | 3 pooled | `direction_probe_baselines/*_phaseP.md` |
| Direction probe nocot | 2 (Llama, Qwen) | n≈110-300 | 1 | `direction_probe_nocot/*_l*/` |
| Direction probe residual_top1 labels | 3 | same residuals | 3 pooled | `direction_probe_residual_top1/*_l*/` |
| Position sweep | 3 (Llama L=14, Ministral L=16, Qwen L=23) | n=50/bucket × multiple rel_pos | 3 pooled | `position_sweep/*_l*/` |
| Attention patterns (Phase L + extended) | 6 cells (3 models × 2 sample sizes) | n=50 or 200/bucket | 3 pooled | `attention_patterns/*/` |
| Head ablation on clean_check_or_call (Phase O A1) | 3 | n=50/cell | 3 pooled | `head_ablation/{llama,ministral,qwen}8b_l*/` |
| Head ablation on illegal_fold (Phase P §2) | 3 | n=50 (L,M), n=24 (Qwen) | 3 pooled | `head_ablation/*_illegal_fold/` |
| Attention mask ablation (A2) | 3 | n=33-50 | 3 pooled | `attn_mask_ablation/{llama,ministral,qwen}8b/` |
| CoT magnitude analysis (B2) | 2 (Llama, Qwen) | from cached residuals | 3 pooled | `cot_magnitude_analysis/{llama,qwen}8b_l*.md` |
| Belief probe in-sample (Phase O B3) | 3 (oracle_strategy_aware only) | n=300/cell | 3 pooled | `belief_direction_probe/*_l*/` |
| Belief probe held-out (Phase P §3a) | 3 | n=300, 5-fold CV | 3 pooled | `belief_direction_probe/*_l*_heldout/` |
| Belief probe agent_belief (Phase P §3b) | 3 | n=300, 5-fold CV | 3 pooled | `belief_direction_probe/*_l*_agent_belief/` |
| Mode-balanced direction probe | **0** (shelved; CoT/non-CoT logs share zero `(hand_id, decision_idx)` pairs) | — | — | — |
| Tier 0 smoke test | 1 (70B paper anchor) | 371 PCE records, 58 UC records | s42 | `tier0_smoke_test/` |
| Tier 4 behavioral | 15 (5 presets × 3 models, 50 hands each) | n=95–470 decisions/cell | s42 | `tier4_opponent/*/` |
| Tier 4 mech patching | **11 of 12** (Qwen 4, Ministral 4, Llama 3) | 10×27–30 | s42 | `causal_patching/tier4_*_l*/` |

**Cell-count totals**: about **115 distinct patching / probe / ablation cells**, plus the 15 Tier 4 behavioral cells. Most paper-critical headline claims are backed by 3-seed pooled data with n=300 patching pairs per cell.

---

## 3. Claim-to-evidence verification

### 3.1 GREEN flags — exactly verified against SUMMARY files

These claims in `updates.md` match their backing SUMMARYs exactly (within rounding):

- **§22b** A3 cleanup baselines — all 9 numbers (learned/permuted/random for 3 models) match `direction_probe_baselines/*_phaseP.md` to 3 decimals.
- **§22c** A1 illegal_fold ablation — all 12 head-set rows match `head_ablation/*_illegal_fold/SUMMARY.md`.
- **§22d** B3 held-out R² + agent_belief — all 12 R² values match their respective SUMMARYs to 3 decimals.
- **§22d** Six belief × verb cosines match to 4 decimals.
- **§22e** Qwen Tier 4 patching: all 4 cells' spec-adj Δ, top-1 → CHECK, and random-null values match SUMMARYs to 2 decimals.
- **§19b** Original Llama clean_CC head ablation Δ = −1.886 matches `head_ablation/llama8b_l14/SUMMARY.md`.
- **§14b** Qwen cross-seed replicate table (per-layer fractions and Δ values) matches `qwen8b_t0_s{42,123,456}_replicate/SUMMARY.md`.
- **§14b** Pooled Qwen layer sweep L22/L23 values (76.2%/+14.19; 100%/+18.33) match `qwen8b_t0_pooled_layer_sweep/SUMMARY.md` exactly.
- **§14c** Llama L=14 clean spec-adj +6.48 matches `llama8b_t0_pooled_layer_sweep/SUMMARY.md` (actual +6.484).
- **§14e** Llama L=14 component decomposition (residual +7.90, attn 49%, heads 18/35/20%) matches `llama8b_l14_components/SUMMARY_components.md`.
- **§15b** Llama triplet patch (+5.17 Δ, 65% ratio, 48.7% top-1 CHECK) matches `llama8b_l14_head_triplet/SUMMARY_components.md`.
- **§16a** Ministral L=16 triplet (43% ratio, +3.39 Δ, 3.0% top-1 CHECK) matches `ministral8b_l16_head_triplet/SUMMARY_components.md`.
- **§18c** Llama position sweep at rel_pos −10 (LF −1.06, iF −1.00) matches `position_sweep/llama8b_l14/SUMMARY.md`.
- **§20a** Tier 0 PASS: |js_difference| = 0.0163 matches `tier0_smoke_test/pce_check_summary.csv` OVERALL row; mean correlation = 0.080 matches `uc_check.csv` aggregation.

### 3.2 RED flags — claims that need attention before the paper

| # | § | Claim text | What evidence supports | Gap | Severity |
|---|---|---|---|---|---|
| R1 | §22h.8 | "Qwen's L=23 verb-pair circuit is opponent-invariant" (global phrasing) | 4/4 Qwen Tier 4 cells; **Llama 3/4 (with bad baselines), Ministral 4/4 (weak + variable)** | Reads as universal, only Qwen is bulletproof. **Must say "in Qwen 8B" explicitly.** | HIGH |
| R2 | §22h.4 | "Verbalized belief decoupled from L* residual" | Held-out R² ≤ 0.14 (Llama) / negative (Ministral, Qwen) on agent_belief via 5-fold ridge | "Decoupled" is strong; correct statement is **"not linearly decodable via ridge regression at L\* with n=300"** — could be nonlinear, could be at different layer | MEDIUM |
| R3 | §22h.5 | Verb ⊥ belief "all three models, both belief sources" | Cos ≤ 0.05 across all 6 cells. But **Qwen × agent_belief held-out R² = −2** so the "belief subspace" is essentially undefined — orthogonality is degenerate here | Add: "for Qwen × agent_belief, the belief subspace itself is undefined (held-out R² < 0), so orthogonality is uninformative" | MEDIUM |
| R4 | §15a / §17i / §22h | "Failure mode is CoT-conditional" interpreted as "CIRCUIT is CoT-conditional" | A3 audit (§15a) shows zero illegal_FOLD in non-CoT across 18 cells. §17c shows Qwen non-CoT clean→clean DOES work (circuit is intrinsic in Qwen) | Headline claim must distinguish **"failure mode requires CoT"** from **"circuit requires CoT"**. EXPERIMENTS_QUEUE explicitly flags this as the analytical hole. | HIGH |
| R5 | §22e + §22h.8 | "spec-adj Δ = +20.10 ± 0.07 across 4 Qwen presets" | 3 cells identical (+20.102), default (+19.914) is 0.188 nats away. Actual sample sd ≈ 0.081, range 0.188. | "±0.07" understates the gap. Use "+20.10 ± 0.10 (range +19.91 to +20.10)" or report both | LOW |
| R6 | §19a (pre-§22 text) | Qwen R² = 0.999 → "belief highly decodable in Qwen" | Held-out R² = 0.089 (§22d). The 0.999 was overfitting at d=4096, n=300, α=1.0 | §22d already revises this; just make sure paper text uses §22 numbers not §19a numbers | MEDIUM (revised in §22d) |
| R7 | §16d / §16c | "Three sparse-attention circuits across 3 models" / "verb-flip threshold ≈ +4-5 nats shared" | Llama h5/h23/h24 sparse, Ministral h22 + long tail (additive sextet), Qwen distributed across L=23. Threshold cited from 2 models (§16e caveat) | Re-word: "Llama narrow-and-deep, Ministral wide-and-shallow, Qwen distributed" — don't lump as "three sparse circuits". | MEDIUM |
| R8 | §22h.6 | "Compute-then-commit two-stage circuit confirmed across all three models" | Llama L=14/15 evidence clear (§15c, §17g). Ministral L=16/17 + Qwen L=23/24 component cells exist but show **less clean** separation | OK to claim; add "with Qwen showing a more distributed compute phase" | LOW |

### 3.3 Methodological concerns in the patching infrastructure

| # | Concern | Where | Impact |
|---|---|---|---|
| M1 | **Random-null Δ is computed against ONE target only** (`targets_prep[0]`), while patched Δ is averaged over 10×30 = 300 pairs | `causal_patching.py` lines 625–650 | spec-adj Δ subtracts a noisier baseline than the headline. For Qwen at +20 nats vs +3 null this is irrelevant; for Llama/Ministral Tier 4 at +0.3-0.7 nats vs ±0.5 nat null, the random-null noise is comparable to the signal. **Disclose in methods section.** |
| M2 | **`score_logits` says "log-prob" in docstring but computes LSE on raw logits** | `causal_patching.py` line 30, 197 | Numbers are correct (Δ is interpretable as logit shift); just fix wording in code AND paper to say "logit aggregate" / "summed log-mass" / etc., NOT "log-prob" | 
| M3 | **`component_patching.py` docstring + wrapper text promise spec-adj Δ but the code only computes `ratio_to_residual`** (component Δ / residual-mode Δ) | `component_patching.py` line 55, wrappers | Paper should describe the COMPONENT decomposition as "ratio to residual-mode effect" not "specificity-adjusted Δ" |
| M4 | **`_pot_size` cross-task feature reads `obs.get("pot", 0)` but enriched logs use `pot_total`** | `direction_probe_baselines.py` line 257 | Bug in unused feature. Phase P A3 cleanup used `position`, not `pot_size`, so no published result is affected. Document or fix. |
| M5 | **Self-patch identity tolerance: docstring says `1e-4`, code uses `1e-2` warning threshold (no hard fail)** | `causal_patching.py` line 583 | All observed self-patch drifts are 0.000 in the SUMMARYs we read, so no published result is affected — but the gate is laxer than the doc claims |
| M6 | **`baseline_top1_match_rate` Tier 4 Llama at 0.57-0.81, well below canonical 0.95** | `tier4_*_llama_*/SUMMARY.md` | Workaround was to drop `--baseline-tolerance-frac` to 0.50 in the wrapper. This is a real measurement issue: PromptReconstructor doesn't byte-identically reproduce Llama on opp-preset chat-template enriched logs. **§22f already documents this. Llama Tier 4 patching numbers should NOT appear in the paper as opponent-stability evidence.** |
| M7 | **Permuted-label control is ONE shuffle, not N trials** | `direction_probe_baselines.py` | A single permutation might be lucky/unlucky. Practically the gap (learned 0.99 vs permuted 0.48) is too large to be permutation noise, but for tighter Δ a multi-permutation null would be better. LOW priority. |
| M8 | **Random-direction "accuracy" uses oracle threshold per trial** | `direction_probe_baselines.py` lines 155–191 | Best-threshold-per-trial is the *upper bound* on what random projections can achieve, not the expected accuracy. The reported 0.76-0.88 is therefore inflated relative to a fixed-threshold random control. Headline still holds (learned 0.99 ≫ random 0.88) but the gap is smaller than the wording implies. |

---

## 4. Dependency graph — how findings build on each other

### 4.1 Compact textual DAG (40 nodes)

```
[F02] Tier 0 PASS — extension reproduces 70B anchor (§20a) [STRONG]
  └─> All downstream analysis can be trusted to use the same pipeline as paper

[F00] L* causal mediation per family (Phase G/H, §14) [STRONG]
  parents: F02
  └─> [F03] Per-model L* exists at residual level
  └─> [F04] Content-addressable (zero-patch null) §14c [STRONG]
  └─> [F05] Verb-general (verb-pair test) §14d [MODERATE]
  └─> [F06] Llama L=14 attn dominates, sparse triplet §14e/§16 [STRONG]
        └─> [F07] Triplet + quartet partially flips verb §15b/§16 [MODERATE]
        └─> [F08] L=14 compute vs L=15 commit §15c [MODERATE]
        └─> [F09] Per-seed head identity stable §17d [STRONG]
        └─> [F10] Reverse direction same heads, opposite sign §17e [STRONG]
        └─> [F25] CHECK-baseline head ablation: no necessity (redundant) §19b [MODERATE]
              └─> [F29] illegal_fold ablation: STILL no necessity, sign FLIPS — heads encode "current verb" §22c [STRONG]
  └─> [F11] Ministral L=16 wide-and-shallow heads §16 [STRONG]
  └─> [F19] Qwen L=23 distributed, residual flow-through §17g [STRONG]

[F12] A3 audit: zero illegal_FOLD in non-CoT (18 cells) §15a [STRONG]
  └─> [F15] Non-CoT clean→clean: Qwen intrinsic (Δ +20.21, 88.7%), Llama inconclusive §17c [MODERATE; Llama is weak]
        └─> [F20] Non-CoT prior-dominated (Llama, Ministral) vs content-faithful (Qwen) §18a [MODERATE]
              └─> [F26] CoT vs non-CoT centroid distance 3× wider in Llama §19e [STRONG-Llama only]

[F14] Linear verb direction at L* exists; CV ~99% §17b → [F31] balanced baselines confirm §22b [STRONG]
  └─> [F22] Position sweep: decision crystallizes within last ~10-20 tokens §18c [MODERATE]
  └─> [F21] CoT vs non-CoT verb-axis cosine 0.27-0.43 §18b [MODERATE]
        └─> [F33] Mode-balanced probe SHELVED — no matched hand_ids §22g [N/A]
  └─> [F24] Belief decodable + verb ⊥ belief (in-sample) §19a [WEAK — overfit in Qwen]
        └─> [F30] Held-out R² REVISES: Llama/Mn moderate (~0.3), Qwen chance (0.09) §22d [STRONG revision]
        └─> [F30b] agent_belief: held-out R² 0.14/-0.18/-2.0 — verbalized belief decoupled §22d [STRONG-new]
        └─> [F30c] cos ≤ 0.05 across all 6 (model × belief source) cells §22d [STRONG with degenerate caveat for Qwen×agent]

[F13] Dominant heads attend legal-action verb fragments §17a → [F23] extended to 200/bucket §18h [MODERATE]

[F27] A2 attention-mask: 8-10pp drop in Llama+Ministral, null in Qwen §19c [MODERATE]
[F28] C3 commit-layer components consistent across 3 models §19f [MODERATE]

[F32] Tier 4 behavioral 15 cells §20b [STRONG] → [F34] Tier 4 mech patching:
        Qwen invariant +20.10 ± 0.10 across 4 presets §22e [STRONG-Qwen-only]
        Ministral WEAK + variable Δ 0.16-0.70 §22e [MODERATE]
        Llama unreliable due to baseline reconstruction breakdown §22f [SHELF]
```

### 4.2 Most paper-critical chain (the headline narrative)

```
Tier 0 PASS (foundation)
  ↓
L* exists at residual level in 3 models (Phase H sweeps, 3 seeds, ~300 pairs/cell)
  ↓
Circuit is content-addressable (zero-patch null) + verb-general (verb-pair)
  ↓
At L*, an attention-mediated sub-circuit SUFFICIENTLY encodes the verb decision
  (head decomposition, 10×30 pairs/cell, 3 seeds pooled)
  ↓
The encoding is REDUNDANT (no single-path necessity) — both saturated AND
  near-threshold ablation tests are null
  ↓
The same L* hosts a linear verb direction (CV ~99% with balanced baselines)
  ↓
The same L* residual partially decodes the strategy-aware oracle belief
  (held-out R² 0.30 in Llama+Ministral, ~0 in Qwen)
  ↓
The verb-decision direction is ORTHOGONAL to the belief subspace
  (cos ≤ 0.05 across all 6 model × belief source cells)
  ↓
The model's OWN verbalized belief is essentially not encoded at L*
  (held-out R² ≤ 0.14, even negative) — verbalized vs represented decouple
  ↓
The L* circuit shows compute-then-commit two-stage structure (compute L*,
  commit L*+1) in all 3 models, with Qwen most distributed
  ↓
Opponent-invariance:
  - Qwen L=23 saturated and constant across 4 presets [STRONG]
  - Ministral L=16 weak but opponent-conditional [MODERATE]
  - Llama L=14 not measurable due to reconstructor mismatch [GAP]
```

This is the writeup spine. Every step is backed by a SUMMARY.md with verified numbers, with the eight caveats from §3 above to fold into the methods section.

---

## 5. Plan vs delivery (EXPERIMENTS.md tiers → status)

| Original Tier | Plan | Status | Notes |
|---|---|---|---|
| Tier 0 | 70B anchor reproduction | ✅ done | §20a |
| Tier 1A.small | 3×8B × 18 cells CoT | ✅ done | Phases B-C |
| Tier 1A.large | 3×70B × 3 seeds × 2 temps | ❌ skipped | Focus shifted to 8B mech |
| Tier 1 (Llama 70B CoT vs direct) | ❌ skipped | Cost/scope | |
| Tier 2 (API frontier) | ❌ skipped | Cost | |
| Tier 3 (8B → frontier scaling) | ❌ skipped | Deferred to revisions only |
| Tier 4 (opponent presets) | ✅ behavioral done; ⚠️ mech 11/12 (Llama unreliable) | §20b–c, §22e–f |
| Tier 5 (logit lens) | ✅ done | §13, §17–§18 (extensive) |
| Tier 6 (pot-odds analysis) | ✅ done | §12 |
| Queue: non-CoT clean→clean | ✅ Qwen, ⚠️ Llama inconclusive | §17c, §17i |
| Queue: Qwen B1 at L=23 | ✅ done | §17g |
| Queue: reverse Llama L=14 | ✅ done | §17e |
| Queue: per-seed Llama heads | ✅ done | §17d |
| Queue: mode-balanced probe | ❌ infeasible (no matched hand IDs) | §22g |

The paper scope is 8B-mech-focused. The original Tier 1A.large/Tier 1/Tier 2/Tier 3 were ambitions in EXPERIMENTS.md that were principled-cut once 8B mech started producing results. Worth a single paragraph in the paper noting "scaled-up replication at 70B+ is left to follow-up work."

---

## 6. Code-level action items (before paper submission)

Ranked by priority. The paper can go out with all of these as-is (none invalidate a result), but addressing them tightens the methods section.

### MUST-FIX before paper (3 items)

1. **R1 / R4 / R3: Tighten language in §22h's "Combined claims" to not over-generalize.**
   - "Qwen's L=23 circuit is opponent-invariant" → "In Qwen 8B at L=23, the verb-pair patching effect is opponent-invariant across 4 presets (spec-adj Δ +20.10, range 0.19 nats). Ministral shows a weaker and opponent-conditional effect; for Llama, the prompt-reconstructor does not byte-identically reproduce opp-preset chat templates (baseline match 0.57–0.81 vs canonical ≥0.95) so Llama Tier 4 patching is not interpretable."
   - "Failure mode is CoT-conditional" must not become "circuit is CoT-conditional" anywhere in the writeup. Qwen's non-CoT clean→clean §17c result is the critical counterevidence; cite it.
   - "Verb ⊥ belief in all 3 models, both belief sources" → add "for Qwen × agent_belief, the belief subspace itself is undefined (held-out R² < 0), so orthogonality there is uninformative."

2. **M2: Fix `score_logits` terminology everywhere.** Change "log-prob" wording in code docstring AND paper text to "logit aggregate" or "summed unnormalized log-mass." Numbers stand; vocabulary tightens.

3. **M3: Component decomposition wording.** Change "specificity-adjusted Δ" in `component_patching.py` docstring + wrapper text to "ratio to residual-mode effect" — that's what the code actually outputs.

### SHOULD-FIX (5 items)

4. **M1: Document the random-null asymmetry.** In methods section: "spec-adj Δ subtracts a random-null estimated against the first target only; this is documented in the code and is appropriate for the high-spec-adj-Δ Qwen result but should be noted as a noise floor for the lower-Δ Ministral Tier 4 cells (where random-null variance is comparable to signal)."

5. **M4: Either fix `_pot_size` to read `pot_total`, or remove the option from CLI choices.** Unused feature, but published code shouldn't ship dead branches.

6. **M5: Self-patch tolerance docstring vs code.** Either change docstring to say "1e-2" or tighten code to fail at "1e-4."

7. **M7+M8: Direction-probe baseline controls.** Add multi-permutation null and fixed-random-direction (not best-threshold) control as additional rows in `direction_probe_baselines.py` output. This strengthens §22b without weakening any result.

8. **Naming inconsistency: `llama8b_t0_s42_layer_sweep` vs `llama8b_t0_s{123,456}_replicate`.** §14 narrative implies all three are "per-seed replicates" but the directory names differ. Add a one-line note in updates.md (or rename) so a reader doesn't think there's a missing cell.

### NICE-TO-HAVE (not blockers)

- Re-run mode-balanced probe with matched-seed inference (would tighten §18b from "0.27/0.34 unmatched" to "X.XX matched"). 3-4 h GPU per model.
- Get Llama × informative_v2 Tier 4 cell to run (requires diagnosing the chat-template reconstructor mismatch). The other 3 Llama Tier 4 cells are unreliable; this would be too without fixing the root cause.
- Tier 1A.large or Tier 2 if you want to claim cross-scale generalization. Big GPU investment.

---

## 7. What this audit found NOT to worry about

To be explicit about what this audit cleared:

- **No fabricated numbers**: every numeric claim spot-checked traced to a real SUMMARY.md row.
- **No mock data**: drivers compute on real logs throughout.
- **No silent skip-on-no-data fallbacks** that would mask missing experiments. Cells that didn't run produced clear `[abort]` / `[fail]` messages and no SUMMARY.
- **No misnamed directories** that could be confused as different experiments (the `s42_layer_sweep` vs `s*_replicate` is the only naming hiccup).
- **Bucket classification (`classify_decision`)** correctly distinguishes `clean_check_or_call`, `clean_legal_fold`, `illegal_fold`, etc. according to the enriched-log schema.
- **Control invocations** (baseline, self-patch, random-null) are real forward passes through the model, not hardcoded numbers.
- **Held-out R² in B3 follow-up** uses true train-test split with no leakage (verified in code).
- **Tier 0 PASS is real**: extension reproduces the published 70B anchor's `|js_difference|` to within 0.002 nats of the published 0.014 ± 0.003 number.

---

## 8. TL;DR for the paper

- **The headline mechanistic claims (L\* circuit exists, head decomposition, redundancy, belief-verb orthogonality, opponent-invariance in Qwen) are all backed by real code, real forward passes, real residuals, and verified numbers.**
- **8 specific wording / control / docstring issues need addressing** before the paper goes out (Section 6), but none invalidate a published result.
- **The biggest risk to the writeup is over-generalization**: Qwen results read as universal, "circuit is CoT-conditional" overstated, agent_belief orthogonality vacuous for Qwen. All three are fixable with careful prose; the underlying experiments are sound.
- **Outstanding gaps that are honest limitations**: Llama Tier 4 patching (reconstructor mismatch), mode-balanced direction probe (no matched IDs), full 70B replication (out of scope by design).

This codebase is **not making things up**. It's making more careful claims than the writeup currently does in 3-4 places. Fix the wording, ship the paper.
