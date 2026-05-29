# Phase Q post-run audit (through commits `dd1a534` audit-rerun + `185800a` missing-four)

**Date:** 2026-05-27 (updated 2026-05-28 after the recon-pipeline rerun and the
four missing tasks landed)
**Scope:** Skeptical re-read of every Phase Q result, code path, and claim that
landed on `main` after the GPU runs.

This file documents (a) what is **defensible** as-is, (b) what was **broken or
mis-framed** in the original post-run write-up, and (c) the **fixes and
follow-up reruns** queued for the next GPU session.

> **See also:** [`REDO_PLAN.md`](REDO_PLAN.md) for the prioritized rerun
> list (P0/P1/P2), the disciplined layer/head choice per cell, and the
> archive/keep/delete decisions.

> **2026-05-28 update banner.** The recon-pipeline reruns changed several
> headline numbers (the consolidation-gradient table below was refreshed) and
> closed P0.2 / P1.2 / P1.3. Two qualitative shifts worth flagging:
> 1. **Llama's continuation "parse-damage" confound did NOT reproduce.** Under
>    the recon pipeline Llama regenerates 100% coherent JSON (0% parse-fail).
>    The real Llama issue is **verb drift on plain regeneration** (baseline
>    regen already flips 56% of recorded FOLDs), not broken JSON. See §4.
> 2. **Ministral has NO sparse-head story at L=16, even at 6 heads.** The
>    sextet ablation is a clean null (§9). L=16 is residual flow-through.
> A new data-generation issue was also found in **Tier 4** (§12): for Llama and
> Qwen, `tight_aggressive` and `loose_aggressive` are byte-identical opponent
> distributions — effectively 4 distinct presets, not 5.

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
| BET→illegal_FOLD, top-1 → BET_RAISE     | 80% (+ 20% CHECK) | 94% (L=14) / **100% (L=15)** | **100%**       |
| Continuation regen_ablated FOLD→CHECK   | **20/24 = 83.3%** | 14/25 = 56% (net **+0pp** over baseline) | 3/25 = 12% (net **+12pp**) |
| Continuation regen_baseline FOLD→CHECK  | 6/24 = 25%     | **14/25 = 56%** (verb drift) | 0/25 = 0%        |
| Continuation parse-fail (regen, either) | **0%**         | **0%** (confound did NOT reproduce) | **0%**       |
| Continuation patch flip → CHECK         | **120/120 = 100%** | 105/125 = 84%        | **125/125 = 100%** |
| Inference ablation (recon, illegal_fold pool) triplet net FOLD→CHECK | **+58.3pp** (33→92%) | +23.5pp (74→97% any-flip; FOLD→CHECK ≈flat) | −2.5pp (control flips MORE) |
| Mode-balanced cos(w_CoT, w_nonCoT)      | **+0.51** (n=110, both modes own labels) | +0.33 (n=99, fallback labels) | +0.095 (**n=16, NaN CV**) |
| Compute band (where attn injects verb)  | **distributed attn L18–20** (no sparse head; max 17%) | **sparse 3-head circuit @ L14** (h5,h23,h24) | residual flow-through @ L16 (sextet null) |
| Opp-invariance (Tier 4 spec-adj, distinct-seed) | **+5 to +20 nats across 5 distinct presets** | ≈0 (noise floor, baseline_match 0.56–0.63) | ≈0.7–1.4 (fails) |

**Reading the row "L\* spec-adj Δ (forward)":** Qwen at its saturation layer is
~3× cleaner than Ministral at *its* saturation layer; Llama at L=14 (the
nominal L\*) is one layer **before** saturation, which propagates through
every Phase Q cell.

---

## Major issues uncovered

### 1. Llama L\*=14 is one layer BEFORE saturation, BUT it's the correct head-discovery layer

From `results/causal_patching/llama8b_t0_pooled_layer_sweep/SUMMARY.md`:

| Layer | top-1 → CHECK | spec-adj Δ |
|------:|--------------:|-----------:|
| 13    | 9%            | +2.49      |
| **14**| **79%**       | **+6.48**  |
| 15    | 100%          | +10.24     |
| 16    | 100%          | +11.30     |

Every Phase Q **patching cell** that uses `LAYER=14` for Llama is being
evaluated at the transition layer. The continuation patch flip rate of 46%
comes from this.

But the project's own component decompositions
(`results/causal_patching/llama8b_l14_components_*/SUMMARY_components.md` vs
`llama8b_l15_components/SUMMARY_components.md`) establish that **L=14 is
where the verb-producing heads live** — `heads_05_23_24` carry 65% of the
residual effect with 49% top-1 → CHECK at L=14, and **NO sparse head subset
at L=15 carries any meaningful effect** (best individual head 17%, no
triplet > 19%). At L=15 the verb signal arrives via residual flow-through
from L=14's heads.

**Disciplined action (see REDO_PLAN.md P0.3):**
- **Patching cells** (residual-level sufficiency): run BOTH L=14 and L=15
  for Llama. Report side-by-side as "transition (L=14) → saturation (L=15)
  in the residual stream."
- **Head ablation cells**: keep at L=14. Llama L=15 has no sparse head
  story; ablating arbitrary heads at L=15 isn't a meaningful necessity
  test.

This is *not* post-hoc cherry-picking — it's matching layer to question.
Ministral at L=14 is also "wrong layer" by the same logic
(`ministral8b_l14_components/SUMMARY_components.md`: residual only 2%
top-1 → CHECK). Ministral's compute and commit collapse onto the same
layer (L=16); Qwen's compute is distributed across L=19–22 with commit at
L=23. See `updates.md` §17g for the full Phase L analysis we already did.

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

### 4. Llama's continuation: verb DRIFT on regeneration, NOT parse damage [REVISED 2026-05-28]

The original audit (HFAgent-era continuation) reported Llama failing to
regenerate parseable JSON 44% of the time and framed the necessity result as
"confounded by parse damage." **The recon-pipeline rerun
(`results/continuation_after_patch/llama8b_l14/`, commit `dd1a534`) does NOT
reproduce parse damage.** Re-counted from the committed `summary.json`:

- 25 recorded illegal_fold targets, all FOLD.
- **All four modes regenerate 100% coherent CoT+JSON; parse-fail = 0/25.**
- regen_baseline verbs: 6 FOLD / 14 CHECK_OR_CALL / 5 BET_OR_RAISE → **baseline
  already flips 56% (14/25) of recorded FOLDs to CHECK without any ablation.**
- regen_ablated verbs: 1 FOLD / 14 CHECK_OR_CALL / 10 BET_OR_RAISE → FOLD→CHECK
  still 56% (14/25); net ablation effect over baseline = **+0 pp** (ablation
  shifts mass FOLD→BET, not FOLD→CHECK).
- Patch flip (verb-only, source×target): **105/125 = 84% to CHECK** (up from
  the stale 46% in the HFAgent-era run).

**Corrected interpretation.** Llama's L=14 continuation **cannot measure
necessity** because the *baseline regeneration itself* already drifts the verb
56% of the time (chat-template / `Today Date:` regeneration non-determinism in
the VERB token — distinct from the now-disproven JSON-parse damage). With a
baseline that high, the ablated condition has no headroom to show a clean
differential. Sufficiency, by contrast, is strong: 84% patch flip and (at L=15)
100% top-1 → CHECK.

**Where Llama necessity DOES show up:** the inference head-ablation recon cell
on the illegal_fold pool (`results/inference_head_ablation/llama8b_l14_recon_illegal_fold/`)
gives triplet any-flip 97.1% vs baseline 73.5% (**+23.5 pp**), and the L=15
negative control (§10) confirms this is specific to the L=14 compute layer.

**Action:** report Llama as the "intermediate consolidation" cell using the
**inference-ablation** necessity number (+23.5 pp at L=14), not the
continuation number. Explicitly note that Llama continuation necessity is
**unmeasurable** due to 56% baseline regeneration drift, and drop the old
"parse damage" framing entirely. Let Qwen carry the cleanest necessity claim.

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

### 7. Head set choice per model — only Llama L=14 has a true sparse circuit

From the existing component decompositions and `updates.md` §17g:

| Model/Layer | Best head set crossing verb-flip threshold | Joint top-1 → CHECK | Status |
|-------------|-------------------------------------------|----------------------|--------|
| **Llama L=14**  | sparse triplet `[5, 23, 24]`           | 49% (`heads_05_23_24`) | **Real sparse story** |
| Llama L=14 quartet | `[2, 5, 23, 24]`                    | 69% (`heads_02_05_23_24`) | Bigger version |
| Llama L=15      | none                                   | best single head 3% | **No sparse story** |
| Ministral L=14  | none — residual itself is 2% top-1     | n/a | **Wrong layer** |
| **Ministral L=16** | sextet `[9, 15, 22, 24, 30, 31]`    | 37% (`heads_09_15_22_24_30_31`) | **Wide-and-shallow** |
| Ministral L=16  | triplet `[22, 9, 15]`                  | **3%** (`heads_09_15_22`) | **Sub-circuit** |
| Qwen L=23       | none — residual flow-through 82%       | best single head 11%, includes negative contributors | **No sparse story** |

**The current canonical head sets are NOT all matched to the real circuit:**

- **Llama** triplet `[5, 23, 24]` at L=14 is the genuine sparse circuit. ✅
- **Ministral** triplet `[22, 9, 15]` at L=16 is **sub-circuit** — it only
  flips 3% of patched targets. We hypothesized the sextet was the proper
  Ministral analogue of Llama's triplet and added it as a new condition
  (`DEFAULT_HEAD_SETS["ministral"]["extended"]`). **The sextet hypothesis was
  tested behaviorally (P0.2) and is a clean NULL — see §9.** The sextet adds
  only +3.7 pp any-flip over the triplet and is still far below the control's
  43.8% flip. NB: the "37%" in the table is the *component-patch joint
  top-1 → CHECK* (signal the 6 heads carry when their contribution is
  injected); it does NOT survive as behavioral *necessity* when those heads
  are ablated during generation. Ministral L=16 is residual flow-through.
- **Qwen** triplet `[26, 28, 30]` at L=23 is **arbitrary** — Qwen has no
  sparse head story at L=23 (`heads_26+28+30` includes h28 with NEGATIVE
  contribution). Reported for cross-model surface comparison only. The
  cleanest Qwen "necessity" test would be either (a) component
  decomposition at L=22 to find the compute layer's heads (see
  REDO_PLAN.md P1.2), or (b) layer-level (whole-attention) ablation.

**Disciplined action (see REDO_PLAN.md P0.2):** run Ministral inference
ablation with the sextet as well as the triplet, and report both.

### 8. Mode-balanced probe: Ministral broken, Llama fallback, Qwen clean

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

### 9. Ministral sextet ablation is a clean NULL — L=16 is residual flow-through [NEW 2026-05-28]

P0.2 ran the sextet `[9,15,22,24,30,31]` as a behavioral inference-ablation
condition alongside triplet/control on the recorded illegal_fold pool (n=80,
recon pipeline). From
`results/inference_head_ablation/ministral8b_l16_recon_illegal_fold_sextet/SUMMARY.md`:

| Condition | any FOLD-flip | FOLD→CHECK | parse-fail |
|-----------|--------------:|----------:|-----------:|
| baseline  | 21.2%         | 11.2%     | 0% |
| triplet   | 18.8%         | 17.5%     | 0% |
| **sextet (extended)** | **22.5%** | **21.2%** | 0% |
| control `[0,1,2]` | **43.8%** | **42.5%** | 0% |

The sextet adds only **+3.7 pp** any-flip over the triplet (noise) and is
**less than the control's 43.8%**. The control (heads 0–2) flips ~2× more than
either hypothesized head set. This is decisive: **Ministral L=16 has no sparse
head story at 3 OR 6 heads.** The verb signal at L=16 is residual
flow-through; ablating any small head subset is dominated by the generic
attention-reweighting disruption captured by the control.

**Conclusion for the paper.** Drop the "Ministral sextet is the proper
analogue of Llama's triplet" hypothesis. Ministral's necessity story is
"distributed / residual flow-through at the saturation layer; no localizable
attention-head circuit at L=16." This is the honest null, and it sharpens the
gradient: only Llama has a sparse head circuit at its compute layer.

### 10. Llama L=15 negative control PASSES — confirms L=14 compute / L=15 saturation [NEW 2026-05-28]

P1.3 applied the SAME Llama triplet `[5,23,24]` at **L=15** (the saturation
layer, where the component decomposition found no sparse head story). From
`results/inference_head_ablation/llama8b_l15_recon_illegal_fold_negctrl/SUMMARY.md`
(recon, illegal_fold pool, n=68):

| Condition | any FOLD-flip | FOLD→CHECK |
|-----------|--------------:|----------:|
| baseline  | 73.5%         | 50.0%     |
| triplet @ L=15 | 75.0%    | 36.8%     |
| control @ L=15 | 75.0%    | 54.4%     |

Triplet at L=15 = +1.5 pp any-flip over baseline (noise), indistinguishable
from control. **This is a textbook negative control:** the triplet that
carries +23.5 pp necessity at L=14 carries nothing at L=15. It confirms the
compute(L=14) vs saturation(L=15) split is real and head-specific, not an
artifact of ablating any heads anywhere. Cite alongside §1.

### 11. Qwen compute band located = attention across L18–L20 (distributed, no sparse head) [UPDATED 2026-05-29]

P1.2 ran `experiments.component_patching` at L=22 (pooled 3 seeds,
clean_check_or_call → illegal_fold). From
`results/causal_patching/qwen8b_l22_components/SUMMARY_components.md`:

| Component | mean Δ(CHECK−FOLD) | ratio to residual | top-1 → CHECK |
|-----------|-------------------:|------------------:|--------------:|
| `residual`| +19.39 nats        | 100%              | 76.2% |
| `attn`    | +1.59 nats         | 8%                | 2.5%  |
| `mlp`     | −0.74 nats         | −4%               | 0.0%  |
| best head (`head_00`) | +4.59  | 24%               | 3.8%  |

Patching the full residual at L=22 reproduces the verb flip (+19.4 nats, 76%
top-1 → CHECK), but the layer's **own attention contributes 8% and its MLP
−4%**; no single head exceeds 24% and the next-largest are <17%. So **L=22 is
residual flow-through, like L=23** — the verb signal arrives via the residual
stream and L=22 passes it through. **Qwen's compute layer (where attn/MLP
inject the signal) is at L<22 and is not yet localized.**

**Follow-up COMPLETE [2026-05-29].** `scripts/run_qwen_compute_layer_sweep.sh`
ran the same decomposition at L=18–21
(`results/causal_patching/qwen8b_l{18,19,20,21}_components/`). Attention's share
of the residual effect, by layer:

| Layer | residual Δ (nats) | **attn ratio** | mlp ratio | top-1 → CHECK (residual) | max single head |
|------:|------------------:|---------------:|----------:|-------------------------:|----------------:|
| L18 | +1.19  | **92%** | 64% | 0.0%  | (all small) |
| L19 | +4.83  | **78%** | 18% | 1.2%  | head_31 = 17% |
| L20 | +9.17  | 39%     | 11% | 27.1% | head_29 = 30% / head_09 = −19% |
| L21 | +13.25 | 1%      | −17% | 55.8% | <3% |
| L22 | +19.39 | 8%      | −4% | 76.2% | head_00 = 24% |
| L23 | +26.83 | (flow-through) | | 100% | — |

**Qwen's compute band is L18–L20: attention injects the signal there (92% of
the small effect at L18, 78% at L19), and from L21 onward it is residual
flow-through (attn ≈ 0–8%).** Crucially the compute is **distributed across
heads** — the largest single head is only 17% (L19) and L20 has a cancelling
+30%/−19% pair — so unlike Llama L=14 there is **no sparse necessary triplet**.
This is the mechanistic signature of Qwen's deeper consolidation: the verb
decision is computed by a *distributed* attention sub-network across three
layers, not a 3-head circuit.

**Necessity (TASK 3, scripted):** because there is no sparse head set, Qwen's
necessity test is a **whole-attention-block ablation** at L18/19/20 (+ saturation
L23 and control L8), plus a concentrated top-positive-head set at L19/L20 —
`scripts/run_qwen_necessity_ablation.sh`, wired into `run_phase_q_final_gpu.sh`.
A genuine necessity result is high FOLD→CHECK flip with LOW parse_fail localized
to the compute band; whole-attention ablation is blunt, so the SUMMARY reports
flip vs parse_fail side-by-side to separate FOLD-specific necessity from generic
CoT damage.

### 12. Tier 4 preset duplication: Llama/Qwen have 4 distinct opponent distributions, not 5 [NEW 2026-05-28]

While auditing the Tier 4 L=15 cells we found
`tier4_loose_aggressive_llama_l15` and `tier4_tight_aggressive_llama_l15`
report **bit-identical aggregates** (mean Δ, top-1 fractions, every per-pair
value to 4 dp) despite different hand IDs. Root cause is upstream in the Tier 4
behavioral generation, **not** in patching:

- `default`          = {aggression 0.40, fold_threshold 0.30, bluff_freq 0.10}
- `informative_v2`   = {aggression 0.85, fold_threshold 0.55, bluff_freq 0.02}
- `tight_aggressive` = {aggression 0.60, fold_threshold 0.40, bluff_freq 0.08}
- `loose_aggressive` = {aggression 0.60, fold_threshold 0.20, bluff_freq 0.15}
- `loose_passive`    = {aggression 0.20, fold_threshold 0.20, bluff_freq 0.05}

Several of these bind only when a hand's strength lands in a narrow band. The
opponent's RNG is seeded `base_seed + player_index = 43` for **every** preset
(`run_experiment.create_agents`). With a shared RNG stream + a fixed 50-hand
deck, when the policy differences don't bind the opponent plays an IDENTICAL
action sequence — so the hero sees identical states (identical prompt_hash
sequences; even the whole game tree is identical, only the `hand_id` label
differs). The collapse is **model-dependent** (it depends on the hero's own
trajectory, which routes the opponent's strengths into/out of the divergent
bands).

**Full cross-preset overlap matrix** (committed, reproducible, CPU-only:
`python -m experiments.diagnose_tier4_preset_overlap` →
[`results/diagnostics/tier4_preset_overlap/SUMMARY.md`](results/diagnostics/tier4_preset_overlap/SUMMARY.md)):

| Model | collapsed (byte-identical) group | **distinct distributions** |
|-------|----------------------------------|----------------------------|
| llama-8b  | `tight_aggressive ≡ loose_aggressive`            | **4** {default, informative_v2, aggressive_twin, loose_passive} |
| qwen-8b   | `informative_v2 ≡ tight_aggressive ≡ loose_aggressive` | **3** {default, aggressive_cluster, loose_passive} |
| ministral-8b | `informative_v2 ≡ loose_aggressive`           | **4** {default, tight_aggressive, info≡loose_agg, loose_passive} |

**Implication — worse than first thought: NO model has 5 distinct presets, and
Qwen (the opponent-invariance headline model) has only 3.** Any
preset-to-preset comparison WITHIN a collapsed group is trivially true
(identical inputs) and must NOT be counted as independent invariance evidence.
The honest opponent-invariance claim is "invariant across **3** (Qwen) /
**4** (Llama, Ministral) genuinely distinct opponent distributions." This also
**corrects an earlier draft of this section** that claimed Ministral had 5
distinct presets — it has 4 (its collapse is `informative_v2 ≡ loose_aggressive`,
a different pair than Llama/Qwen).

**Fixes shipped (this commit):**
- `run_experiment.py` — new opt-in `--opponent-seed` override. Default is
  unchanged (preserves Tier 1–3 reproducibility); passing a distinct
  per-preset value decorrelates the RNG streams so presets can no longer
  collapse. Documented in `create_agents`' docstring.
- `scripts/run_tier4_regen_distinct_presets.sh` — **OPTIONAL** regeneration of
  Llama/Qwen presets with distinct opponent seeds + re-patch at L*. Outputs
  suffixed `_distinctseed`; originals preserved.

**Recommendation: regenerate (now warranted), AND document.** Originally this
was "documentation only," on the logic that tight/loose_aggressive are
near-identical by design. But the full matrix shows the collapse is broader
(Qwen down to **3** distinct distributions, including the headline-aggressive
preset folding into informative_v2). With distinct per-preset opponent seeds,
each preset draws an INDEPENDENT sample even where policies are similar, which
turns the opponent-invariance test from "3 distinct populations" back into a
genuine 5-population test. So regeneration meaningfully strengthens the claim,
not just cosmetically. Run `scripts/run_tier4_regen_distinct_presets.sh` for
all 3 models (it is in the required GPU list below). Until then, the interim
writeup move is to **report each collapsed group as a single cell** (Llama/Min:
4 cells; Qwen: 3 cells).

**Distinct-seed regeneration COMPLETE [2026-05-29].** Ran
`scripts/run_tier4_regen_distinct_presets.sh` for all 3 models with a distinct
per-preset opponent seed (1043/2043/3043/4043/5043), re-enriched, re-patched at
each L*, and re-ran the overlap diagnostic on the `_distinctseed` logs
([`results/diagnostics/tier4_preset_overlap_distinctseed/SUMMARY.md`](results/diagnostics/tier4_preset_overlap_distinctseed/SUMMARY.md)).
Decorrelation outcome:

| Model | distinct distributions BEFORE | AFTER (distinct seed) | residual collapse |
|-------|------------------------------:|----------------------:|-------------------|
| qwen-8b   | 3 | **5** ✓ | none |
| llama-8b  | 4 | **5** ✓ | none |
| ministral-8b | 4 | **4** | `default ≡ informative_v2` |

Qwen and Llama now have **5 genuinely-distinct, independently-sampled** opponent
distributions — the seed fix fully decorrelated them. Ministral still has one
residual collapse (`default ≡ informative_v2`), but for a **different, benign
reason**: Ministral folds early (those two presets average only 2 decisions/hand,
49 distinct prompts each), so the opponent rarely acts and its policy never binds
— the collapse is a property of Ministral's tight play, not the shared RNG. Report
Ministral as 4 distinct cells; Qwen and Llama as 5.

**Re-patched opponent-invariance on the now-distinct presets**
(`results/causal_patching/tier4_*_*_l*_distinctseed/`, spec-adj Δ / top-1→CHECK):

| Preset | Qwen L23 | Llama L15 | Ministral L16 |
|--------|---------:|----------:|--------------:|
| default          | +5.1 / 92%  | −1.0 / 98%* | (skipped, <5 LF) |
| informative_v2   | +8.4 / 100% | (skipped)   | (skipped, <5 LF) |
| tight_aggressive | +13.0 / 61% | (skipped)   | +1.4 / 3%  (n=60) |
| loose_aggressive | +18.3 / 100%| −0.2 / 70%* | +0.9 / 19% |
| loose_passive    | +20.5 / 100%| (skipped)   | +0.9 / 9%  |

\* Llama baseline_top1_match is 0.56–0.63 (action-token instability), so its
spec-adj sits at the noise floor regardless of preset — Tier 4 remains
unmeasurable for Llama, consistent with the rest of Phase Q. (Skipped Llama/
Ministral cells had <5 clean_legal_fold targets in the regenerated logs.)

**Net: Qwen's opponent-invariance now holds across 5 genuinely-distinct presets**
(spec-adj +5 to +20 nats, top-1→CHECK 61–100%, baseline_match 1.00) — a strictly
stronger claim than the pre-fix "3 distinct." Llama = noise floor, Ministral =
fails (~1 nat). This supersedes the interim "report collapsed groups as one cell"
guidance for Qwen and Llama.

**Tangential confirmation (Llama Tier 4 L=15 baseline_top1_match).** Across
presets the no-patch baseline match is 0.23 (informative_v2) – 0.67 (default),
reflecting the known Llama action-token top-1 instability, NOT a prompt-recon
bug: `experiments.verify_prompt_reconstruction` passes 30/30 at 0.50-nat
tolerance for every preset (`results/diagnostics/tier4_llama_l15_regen/`).
informative_v2 had 8/30 "TIE" (within tolerance), consistent with bf16 ULP
noise on the verb token at logits ~25.

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

## Files marked superseded (kept on disk, do not cite)

- [`results/inference_head_ablation/ministral8b_l16_cot/SUPERSEDED.md`](results/inference_head_ablation/ministral8b_l16_cot/SUPERSEDED.md)
  — confounded by HFAgent regen-fidelity (5% baseline parse_OK). Replace
  with P0.1 recon-pipeline rerun.
- [`results/mode_balanced_probe/ministral8b_l16/SUPERSEDED.md`](results/mode_balanced_probe/ministral8b_l16/SUPERSEDED.md)
  — n=16 matched-classified pairs, CV NaN. Drop from writeup.

We do NOT delete these directories — they document real findings (regen
drift, Ministral's bucket-skew) that belong in the limitations section.

## Code/script changes in this audit

- `experiments/inference_head_ablation.py` — added `--pipeline recon` (default)
  and `--filter-recorded-bucket`; tracks verb regex (FOLD / CHECK_OR_CALL /
  BET_OR_RAISE / UNK) and reports flip rate on recorded-FOLD subset; saves
  `raw_response` per row. **DEFAULT_HEAD_SETS now also exposes `extended`**
  (Ministral sextet `[9,15,22,24,30,31]`; Llama and Qwen unchanged), and
  `--conditions extended` adds a new ablation cell in the same run.
- `experiments/continuation_after_patch.py` — added per-mode verb counts and
  flip-rate breakdown for recorded-FOLD targets; SUMMARY.md gains a "Verb
  distribution" and "Flip rate on recorded-FOLD targets" section.
- `scripts/run_inference_head_ablation.sh` — per-model selector
  (`MODEL=ministral|llama|qwen`), pipeline + filter env vars, FORCE_RERUN
  guard.
- `scripts/run_context_stratified_patching.sh` — `LAYER` and `OUT_SUFFIX`
  env so multiple layers/stratifications can coexist (`_pot_odds`,
  `llama8b_l15_street`, etc).
- `scripts/run_phase_q_audit_rerun.sh` — orchestrator for items 1, 2, 3 above.
- `scripts/run_phase_q_llama_l15_parallel.sh` — runs the Llama
  patching cells (reverse, BET, context-stratified) at L=15 for the
  saturation-layer parallel comparison.
- `scripts/run_phase_q_audit_and_l15_serial.sh` + `scripts/run_phase_q_missing_four.sh`
  — serial orchestrators that produced the `dd1a534` and `185800a` results
  (recon ablation, continuation breakdown, pot_odds, Llama L=15 cells,
  Ministral sextet, Qwen L=22 components, Llama L=15 negctrl, Tier 4 regen).
- **`run_experiment.py`** — new opt-in `--opponent-seed` override so opponent
  presets no longer share an RNG stream (root-cause fix for §12). Default
  behavior unchanged.
- **`scripts/run_tier4_regen_distinct_presets.sh`** — Tier 4 regeneration with
  decorrelated per-preset opponent seeds (§12). RAN 2026-05-29 (all 3 models).
- **`scripts/run_qwen_compute_layer_sweep.sh`** — Qwen L=18–21 component
  decomposition (§11). RAN 2026-05-29: located compute band = attn L18–20.
- **`scripts/run_qwen_necessity_ablation.sh`** + `inference_head_ablation.py`
  `--head-sets name:all` support — whole-attention-block necessity test at
  Qwen's compute band (§11, TASK 3), since the distributed circuit has no
  sparse triplet to ablate.
- **`experiments/diagnose_tier4_preset_overlap.py`** — added `--log-suffix` so
  the diagnostic can target the `_distinctseed` regenerated logs (§12).

## Code paths NOT changed (deliberately)

- Reverse FOLD→CHECK and BET→illegal_FOLD patching summaries (`commit
  6fe23d4`) — already valid, do not re-run.
- Tier 4 *patching* code is OK; the issue was upstream *data generation*
  (§12 preset-duplication), now fixed opt-in via `--opponent-seed` AND
  regenerated with distinct seeds (Qwen/Llama → 5 distinct, Ministral → 4).
- Mode-balanced probe — Ministral isn't worth re-running (small pool is a
  data fact, not a code bug); Llama fallback is documented.

---

## What to run on the next GPU box

See [`REDO_PLAN.md`](REDO_PLAN.md) for the full prioritized list (P0/P1/P2)
with GPU-hour estimates. The one-shot orchestrator:

```bash
cd xpoker2026Extension
git pull origin main
export HF_HOME=/workspace/huggingface HF_TOKEN=...

# P0.1 (inference recon) + P0.4 (continuation breakdown) + P1.1 (pot_odds)
FORCE_RERUN=1 CONTINUE_TOKENS=180 bash scripts/run_phase_q_audit_rerun.sh

# P0.3 (Llama L=15 patching parallel set)
FORCE_RERUN=1 bash scripts/run_phase_q_llama_l15_parallel.sh

# P0.2 (Ministral sextet ablation condition)
MODEL=ministral PIPELINE=recon \
  FILTER_RECORDED_BUCKET=illegal_fold \
  FORCE_RERUN=1 \
  bash scripts/run_inference_head_ablation.sh \
  --conditions baseline triplet extended control
```

**Status 2026-05-29: P0.1–P0.4, P1.1–P1.3, Tier 4@L=15, the Qwen compute-layer
sweep (§11), and the Tier 4 distinct-seed regeneration (§12) have ALL landed**
(`dd1a534`, `185800a`, `8ce0b47`, `2b88d27`). The only remaining GPU task is
**TASK 3 — the Qwen necessity ablation** (now unblocked + scripted), folded into
the same orchestrator:

```bash
cd xpoker2026Extension && git pull origin main
export HF_HOME=/workspace/huggingface HF_TOKEN=...
bash scripts/run_phase_q_final_gpu.sh   # detaches into tmux 'poker_phase_q_final'
```

On re-run that orchestrator skips the already-written sweep/regen (FORCE_RERUN
guards) and runs **TASK 3** directly. Or run TASK 3 standalone (Qwen weights
only, ~20–40 min):

```bash
cd xpoker2026Extension && export HF_HOME=/workspace/huggingface HF_TOKEN=...
FORCE_RERUN=1 bash scripts/run_qwen_necessity_ablation.sh
```

TASK 3 = whole-attention-block ablation at Qwen's compute band L18/19/20 (+
saturation L23, control L8) and a concentrated top-head set at L19/L20, on the
illegal_fold pool. Reads flip→CHECK vs parse_fail to separate FOLD-specific
necessity from generic CoT damage (the test is blunt because the circuit is
distributed — §11).

Local (no-GPU) verification committed:
`results/diagnostics/tier4_preset_overlap_distinctseed/SUMMARY.md`
(Qwen/Llama 5 distinct, Ministral 4 — run via
`python -m experiments.diagnose_tier4_preset_overlap --log-suffix _distinctseed`).

---

## What the writeup should now claim

1. **Sufficiency** (universal across 3 models): single-residual L\* patch
   from a CHECK source flips an illegal_FOLD target's verb at 100% in
   Ministral / Qwen, 79% in Llama L=14 / 100% in Llama L=15. Verb-generality
   replicated for BET → illegal_FOLD (80–100% top-1 BET_RAISE).

2. **Necessity** (gradient, from the recon-pipeline inference ablation on the
   illegal_fold pool): triplet head-ablation during full CoT decoding flips
   **+58 pp** of recorded-FOLD verbs to CHECK in Qwen (33%→92% any-flip), the
   L=14 triplet adds **+23.5 pp** any-flip in Llama (and its L=15 negative
   control is null, §10), and Ministral shows **no head-localized necessity at
   3 or 6 heads** (sextet null, control flips more — §9). Report Llama
   necessity from inference ablation, **not** continuation (continuation
   baseline regen-drift is 56%, §4). Ministral is "distributed / residual
   flow-through; no sparse circuit at L=16."

3. **Opponent-invariance** (Qwen-only, by Tier 4): after the distinct-seed
   regeneration (§12), Qwen's L\*=23 produces spec-adj **+5 to +20 nats /
   61–100% top-1 → CHECK across all 5 genuinely-distinct opponent presets**
   (baseline_match 1.00); Llama is at the noise floor (baseline_match 0.56–0.63)
   and Ministral fails (~0.9–1.4 nats, top-1→CHECK 3–19%). The earlier RNG
   collapse is **resolved** — Qwen/Llama now have 5 distinct distributions,
   Ministral 4 (`default ≡ informative_v2`, a benign fold-early collapse). Note
   the target_bucket is `clean_legal_fold` (§6), so Δ-magnitudes differ from
   Phase K.

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
   sufficiency but a wider, longer-tailed head circuit.

8. **Only Llama has a sparse head circuit; Ministral and Qwen are residual
   flow-through at the saturation layer** [revised after P0.2/P1.2/P1.3]:
   - **Llama L=14** — genuine sparse triplet `[5,23,24]`: +23.5 pp behavioral
     necessity, and the L=15 negative control is null (§10).
   - **Ministral L=16** — **no** sparse circuit at 3 or 6 heads; the sextet
     ablation is a clean null and the control flips more (§9). Distributed /
     residual flow-through.
   - **Qwen** — compute is **distributed attention across L18–L20** (§11): attn
     carries 92%/78% of the (small) residual effect at L18/L19, but the largest
     single head is only 17% and L20 has a cancelling +30%/−19% pair. By L21–23
     it is residual flow-through (attn ≈0–8%). So Qwen HAS a localizable
     *compute band*, but it is a distributed multi-layer attention sub-network,
     **not** a sparse head circuit. Necessity tested by whole-attention ablation
     (TASK 3).

   So the gradient is sharper than "head count scales": **Llama is the only 8B
   with a localizable *sparse* attention-head circuit at its compute layer;
   Qwen consolidates the verb into a distributed attention band (L18–20) that
   then rides the residual stream; Ministral commits it via the residual stream
   with no localizable heads at its saturation layer.**

This is the honest cross-model story.
