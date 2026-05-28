# Phase Q post-run audit (commit `6fe23d4` + Ministral L16 inference ablation)

**Date:** 2026-05-27
**Scope:** Skeptical re-read of every Phase Q result, code path, and claim that
landed on `main` after the GPU run.

This file documents (a) what is **defensible** as-is, (b) what was **broken or
mis-framed** in the original post-run write-up, and (c) the **fixes and
follow-up reruns** queued for the next GPU session.

---

## TL;DR — what to claim in the paper

> L\* is **sufficient** and **verb-general** (CHECK / FOLD / BET subspace
> swap) in **all three 8B models**. **Necessity, opponent-invariance, and
> mode-stability** are strongest in **Qwen 8B** (L=23), partial in **Llama 8B**
> with a parse-damage confound (L=14), and **weak in Ministral 8B** (L=16).
> The three 8B architectures sit at three points along a circuit-consolidation
> spectrum.

This replaces the earlier "all 3 are equivalent" framing.

---

## Cross-model consolidation gradient

| Property                                | Qwen 8B (L=23) | Llama 8B (L=14)         | Ministral 8B (L=16) |
|-----------------------------------------|----------------|-------------------------|---------------------|
| L\* sufficiency, top-1 → CHECK (forward)| **100%**       | 79% at L=14; 100% at L=15 | **100%**          |
| L\* spec-adj Δ (forward)                | **+18.3 nats** | +6.5 (L=14) / +10.2 (L=15) | +7.4 nats        |
| Reverse FOLD→CHECK, top-1 → FOLD        | **96%**        | 59%                      | 100%               |
| Reverse spec-adj Δ                      | **−26.8 nats** | −9.4                     | −2.2               |
| BET→illegal_FOLD, top-1 → BET_RAISE     | 80% (+ 20% CHECK) | **94%**               | **100%**           |
| Continuation regen_ablated FOLD→CHECK   | **22/24 = 92%**| 22/24 = 92% (with **46% parse fail**) | 4/25 = 16%   |
| Continuation regen_baseline parse_OK     | 23/24          | **14/25**                | 25/25              |
| Continuation patch flip → CHECK         | **120/120 = 100%** | 58/125 = 46%         | **125/125 = 100%** |
| Mode-balanced cos(w_CoT, w_nonCoT)      | **+0.51** (n=110, both modes own labels) | +0.33 (n=99, fallback labels) | +0.095 (**n=16, NaN CV**) |
| Opp-invariance (Tier 4 spec-adj)        | **+19.3 nats** | ≈0 (noise floor)         | ≈0.7               |

**Reading the row "L\* spec-adj Δ (forward)":** Qwen at its saturation layer is
~3× cleaner than Ministral at *its* saturation layer; Llama at L=14 (the
nominal L\*) is one layer **before** saturation, which propagates through
every Phase Q cell.

---

## Major issues uncovered

### 1. Llama L\*=14 is one layer BEFORE saturation (was: not flagged)

From `results/causal_patching/llama8b_t0_pooled_layer_sweep/SUMMARY.md`:

| Layer | top-1 → CHECK | spec-adj Δ |
|------:|--------------:|-----------:|
| 13    | 9%            | +2.49      |
| **14**| **79%**       | **+6.48**  |
| 15    | 100%          | +10.24     |
| 16    | 100%          | +11.30     |

Every Phase Q cell that uses `LAYER=14` for Llama (continuation, context-
stratified, BET→illegal_FOLD, reverse, head-ablation triplet) is being
evaluated at the transition layer, not at full saturation. This explains the
softer Llama numbers vs Qwen and Ministral and the patch-flip rate of only
46% in continuation.

**Action:** keep L=14 as the canonical L\* (Phase K choice), but document the
saturation-layer asymmetry in methods. **Do not** silently re-pick L=15 only
for the cells where it improves — that is post-hoc tuning. If we run Llama at
L=15 we must run it at L=15 across **all** Phase Q cells, then report both
side-by-side.

### 2. Ministral inference ablation was incoherent (`commit 114e7d0` was a partial fix)

`results/inference_head_ablation/ministral8b_l16_cot/`:

- baseline parseable JSON on **recorded illegal_fold pool**: **4/80 = 5%**.
- triplet: 1/80 = 1.25% (parse-OK)
- control: 14/80 = 17.5% (parse-OK)

The previous SUMMARY reported "triplet illegal_FOLD 42.5%, baseline 38.0%,
+4.5pp" without disclosing that 76/80 baseline records have already failed to
regenerate parseable JSON at all. The "illegal_fold rate" is functionally
**measuring whether HFAgent's fallback fired** (every fallback resolves to a
default action that happens to be CHECK_OR_CALL when CHECK_OR_CALL is legal —
hence `replay_action: "CHECK_OR_CALL"` AND `replay_bucket: "illegal_fold"`
simultaneously, because `_reparse_one(raw_response)` finds "FOLD" in the
truncated raw text).

The result: this experiment has been measuring HFAgent's regen-fidelity, not
the heads' necessity.

**Comparison with continuation on the same model**: continuation's
`regenerate_baseline` regenerates 25/25 of its illegal_fold targets cleanly
and 24/25 still emit FOLD. So the SAME ablation hook on a different prompt
pipeline produces 100% parseable baselines.

**Root cause:** HFAgent's prompt assembly path (history truncation policy +
prompt cache + dtype + chat-template branch) drifts from
`PromptReconstructor.build()` enough to change generation outcome at T=0 on
~95% of records. We did not pin down which specific element diverges; the
fix is to bypass HFAgent entirely.

**Fix (committed):** new `--pipeline recon` flag in
`experiments/inference_head_ablation.py` that uses
`PromptReconstructor + raw model.generate(do_sample=False)` — identical to
`continuation_after_patch._full_generate`. New `--filter-recorded-bucket
illegal_fold` flag restricts the pool to recorded illegal_fold targets so the
inference and continuation flip-rate numbers are apples-to-apples.

### 3. Ministral has weak L\* necessity (was: implied as "cleanest sufficient")

Re-counted from `results/continuation_after_patch/ministral8b_l16/examples.jsonl`
(the now-fixed continuation experiment):

- 25 recorded illegal_fold targets, all FOLD verbs.
- regenerate_baseline: 24 FOLD, 1 BET_OR_RAISE → 4% baseline flip (T=0 not
  fully deterministic).
- regenerate_ablated: 21 FOLD, 3 CHECK_OR_CALL, 1 BET_OR_RAISE → 16% flip;
  net ablation effect = **+12 pp**.
- patch flip (verb-only): **125/125 = 100% to CHECK** ✅ (unchanged
  sufficiency claim).

So: **Ministral is the strongest sufficient cell (saturated at L=16, +11.5
nat spec-adj plateau, 100% patch flip) AND the weakest necessity cell (12pp
flip under triplet ablation).** This is a real finding, not a bug.

### 4. Llama's continuation numbers conflate "necessity" and "general damage"

Re-counted from `results/continuation_after_patch/llama8b_l14/examples.jsonl`:

- 24 recorded illegal_fold targets parsed FOLD; 1 was UNK at recording.
- regen_baseline parse_OK: **14/25** — Llama already fails to regenerate
  parseable JSON 44% of the time at T=0. This is the chat-template `Today
  Date:` drift documented in §22m.2. (We knew about this for Tier 4 already.)
- regen_ablated parse_OK: 11/25 — adds another 3 parse failures.
- regen_ablated FOLD→non-FOLD verb flip: 22/24 = 92%, but **22-7 = 15
  additional flips over baseline (75% baseline)** ⇒ **+17 pp** *if you
  trust the baseline*.
- Patch flip (verb-only): 58/125 = **46% to CHECK**, 62/125 stay FOLD.

The patch flip of only 46% on Llama L=14 is genuinely surprising. At L=14 the
forward pooled sweep showed top-1 → CHECK = 79%; the continuation patch
sample of n=125 showing 46% suggests either (a) the source residuals sampled
in continuation underperform the n_source=10 pool used in pooled sweeps or
(b) the verb-only single-forward measurement is noisier than the
n_source × n_target full grid. Either way the continuation patch headline
should be reported with the 46% number, not the pooled-sweep 79%.

**Action:** flag in the writeup that Llama L=14 patch flip on continuation is
46%, and that the necessity claim is confounded by parse damage. Frame Llama
as the "intermediate consolidation" cell; let Qwen carry the cleanest
necessity claim.

### 5. Context-stratified result is "stable across streets", not "context-modulated"

The auto-summary rule "spread > 1 nat ⇒ context-modulated" fires misleadingly
because top-1 is at 86–100% in every stratum. The cleanest reading of
`results/context_stratified_patching/{model}_*/SUMMARY.md`:

- Llama L=14: top-1 flip 86–95% across PREFLOP / FLOP / TURN / RIVER (4
  strata).
- Qwen L=23: top-1 flip 100% across all 4 strata.
- Ministral L=16: top-1 flip 100% across the 2 viable strata (FLOP, PREFLOP).

i.e. **the categorical L\* sufficiency is invariant across streets in all 3
models.** The magnitude of the residual shift (mean Δ) varies, but that's
expected because base-rate logit gaps differ by street.

**The original equity-stratification question is still open.** The
`STRATIFY_BY=pot_odds_quartile` path in `experiments/context_stratified_
patching.py` (committed) filters to facing-bet decisions only and bins by pot
odds. This is the right experiment for "does L\* fire downstream of equity
computation". Queued in `scripts/run_phase_q_audit_rerun.sh`.

### 6. Tier 4 uses `target_bucket=clean_legal_fold`, not `illegal_fold`

`scripts/run_tier4_patching.sh:176` — `--target-bucket clean_legal_fold`. This
is a **different baseline from Phase K's main patching cells**, which all use
`illegal_fold` targets. Tier 4 measures: "given a deck where the model
**legitimately wanted to fold** (and FOLD was legal), does patching a CHECK
residual flip it across opponent presets?"

Numbers across presets:

- Qwen: spec-adj **+19.3 / +20** nats, top-1 → CHECK 92–100% — robust.
- Llama: spec-adj at noise floor (~0.5 nats); some top-1 → CHECK reported
  but likely measurement floor.
- Ministral: spec-adj <1 nat, top-1 → BET_OR_RAISE 38–42% in some presets —
  patch sometimes pushes the model TOWARD a BET, opposite of the intended
  CHECK takeover. Inconsistent with the patching at illegal_fold targets.

**Action:** add a methods note explaining the target-bucket choice and why
the Tier 4 numbers are smaller than Phase K's (different counterfactual,
different Δ-scale). The Tier 4 conclusion is "Qwen circuit is opponent-
invariant; Llama and Ministral are not measurable with this setup".

### 7. Mode-balanced probe: Ministral broken, Llama fallback, Qwen clean

From `results/mode_balanced_probe/`:

- **Qwen** L=23: 110 matched-classified pairs, both modes own labels, CV
  accuracy 100%, **cos(w_CoT, w_nonCoT) = +0.51**, centroid cos +0.60. Clean.
- **Llama** L=14: 99 pairs but `label_source=cot` fallback (one mode had
  degenerate label distribution); CV accuracy 99% / 97%; cos +0.33. Usable
  with caveat.
- **Ministral** L=16: only **16 pairs** survived classification; **CV
  accuracy = NaN ± NaN**; cos +0.095 (essentially zero). **Unreportable.**

**Action:** drop Ministral from the mode-balanced cell. Caveat Llama's
fallback. Report Qwen as the primary mode-stability evidence.

---

## Reconciling the 1.25% vs 16% flip rate

Before the audit rerun, the same hook on the same model produced **two
different numbers** for "fraction of illegal_fold records where ablation
flips the verb to CHECK":

| Experiment | Pool | Pipeline | Net flip rate (Ministral, triplet) |
|------------|------|----------|------------------------------------|
| `inference_head_ablation` (legacy)         | 80 of 200 mixed; HFAgent regen fidelity 5% | HFAgent | 1.25% |
| `continuation_after_patch.regenerate_ablated` | 25 illegal_fold; recon + raw generate; baseline parse 100% | recon | **16%** (net +12pp over baseline 4%) |

The continuation-side number (16% / +12pp) is the trustworthy one because
its baseline regeneration parses 100% of records cleanly and we can compute a
true differential.

After the audit rerun (`scripts/run_phase_q_audit_rerun.sh`) the inference
ablation will use the same pipeline and pool filter, and the two numbers
should agree. If they don't, we have a residual bug to chase.

---

## Code/script changes in this audit

- `experiments/inference_head_ablation.py` — added `--pipeline recon` (default)
  and `--filter-recorded-bucket`; tracks verb regex (FOLD / CHECK_OR_CALL /
  BET_OR_RAISE / UNK) and reports flip rate on recorded-FOLD subset; saves
  `raw_response` per row.
- `experiments/continuation_after_patch.py` — added per-mode verb counts and
  flip-rate breakdown for recorded-FOLD targets; SUMMARY.md gains a "Verb
  distribution" and "Flip rate on recorded-FOLD targets" section.
- `scripts/run_inference_head_ablation.sh` — per-model selector
  (`MODEL=ministral|llama|qwen`), pipeline + filter env vars, FORCE_RERUN
  guard.
- `scripts/run_context_stratified_patching.sh` — `OUT_SUFFIX` env so multiple
  stratifications can coexist (`_pot_odds`, etc).
- `scripts/run_phase_q_audit_rerun.sh` — orchestrator for items 1, 2, 3 above.

## Code paths NOT changed (deliberately)

- Reverse FOLD→CHECK and BET→illegal_FOLD patching summaries (`commit
  6fe23d4`) — already valid, do not re-run.
- Tier 4 — methodologically OK but documented better below.
- Mode-balanced probe — Ministral isn't worth re-running (small pool is a
  data fact, not a code bug); Llama fallback is documented.

---

## What to run on the next GPU box

After disk increase (HF cache ≥ 35GB):

```bash
cd xpoker2026Extension
git pull origin main
export HF_HOME=/workspace/huggingface HF_TOKEN=...

# Audit rerun: (1) inference ablation recon pipeline, (2) continuation with
# verb breakdown, (3) context-stratified by pot_odds_quartile.
FORCE_RERUN=1 CONTINUE_TOKENS=180 \
  bash scripts/run_phase_q_audit_rerun.sh
```

Expected new artefacts:

```
results/inference_head_ablation/{ministral,llama,qwen}8b_l*_recon_illegal_fold/
results/continuation_after_patch/{ministral,llama,qwen}8b_l*/SUMMARY.md  (overwritten)
results/context_stratified_patching/{ministral,llama,qwen}8b_l*_pot_odds/
```

Total wall time estimate: ~3 GPU-hours across all 3 models.

---

## What the writeup should now claim

1. **Sufficiency** (universal across 3 models): single-residual L\* patch
   from a CHECK source flips an illegal_FOLD target's verb at 100% in
   Ministral / Qwen, 79% in Llama L=14 / 100% in Llama L=15. Verb-generality
   replicated for BET → illegal_FOLD (80–100% top-1 BET_RAISE).

2. **Necessity** (gradient): triplet head-ablation during full CoT decoding
   flips +58 pp of recorded-FOLD verbs to CHECK in Qwen, ≈+17 pp in Llama
   *with parse-failure damage*, +12 pp in Ministral (small but real).

3. **Opponent-invariance** (Qwen-only, by Tier 4): Qwen's L\*=23 produces
   spec-adj +19–20 nats / 92–100% top-1 → CHECK across 6 opponent presets;
   Llama and Ministral fail this test.

4. **Mode-stability** (Qwen-only, by §18b/§22j): cos(w_CoT, w_nonCoT) = +0.51
   matched, +0.34 unmatched in Qwen; Llama uses fallback labels (cos +0.33);
   Ministral has too few matched pairs to compute (n=16, CV NaN).

5. **L\* fires regardless of street** (3 models, 2–4 strata): top-1 → CHECK
   stays 86–100% across PREFLOP / FLOP / TURN / RIVER. The
   pot-odds-quartile-on-facing-bet experiment (queued) will say whether L\*
   is downstream of equity computation specifically.

6. **The L\* patch is surgical, not a coherence break**: continuation after
   verb-only patch keeps 100% coherent CoT+JSON in all 3 models — but **does
   not propagate the verb change through full re-decoding** (regenerate +
   ablated does, with model-specific caveats). The verb is encoded modularly,
   separable from JSON formatting.

7. **Three 8B models lie at three points along a circuit-consolidation
   spectrum.** Qwen has the deepest consolidation (sufficient + necessary +
   opponent-invariant + mode-stable); Llama is intermediate with
   parse-damage confounds at the transition layer; Ministral has clean
   sufficiency but circuit-redundant necessity.

This is the honest cross-model story.
