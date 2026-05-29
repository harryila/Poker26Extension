# Claims, identification assumptions, and limitations (writeup reference)

Companion to `AUDIT_FINDINGS.md`. Structures the paper's causal claims per the disclosure
norm of *"Mechanistic Interpretability Must Disclose Identification Assumptions for Causal
Claims"* (arXiv 2605.08012): for each claim state (a) whether it is causal, (b) the
identification strategy, (c) the assumptions, (d) at least one stress test, (e) how the
conclusion shifts if the assumption fails. **Validation is not identification.**

Status legend: ✅ supported by committed evidence · 🟡 supported with a stated caveat ·
⏳ requires a queued GPU experiment (see `REDO_PLAN.md` / Phase B–D).

---

## 0. Reframed headline (what leads the paper)

**Old framing (demote):** "LLMs encode a linear *belief/decision direction*" — led with the
linear probe and the CoT/non-CoT cosine. This is the WEAKEST evidence (correlational, and
the verb label is collinear with a game-state feature; see §3).

**New framing (lead):** "A single residual-stream **intervention** at a model-specific layer
L\* is **causally sufficient** to flip a poker agent's committed action (FOLD→CHECK), in all
three 8B models; the *necessity* of that locus follows a cross-family **consolidation
gradient** (sparse head circuit in Llama → distributed L19 attention in Qwen → residual
flow-through in Ministral)." Probe geometry and mode-stability are **supporting** evidence,
explicitly de-confounded (§3).

---

## 1. CLAIM: Causal sufficiency of L\* (HEADLINE)

> Patching a single CHECK/CALL-source residual at L\* into an illegal_fold target flips the
> model's next-token verb to CHECK at **100% top-1** (all 3 models; Llama 79% at L14 / 100%
> at L15).

- **Causal?** Yes — interchange intervention (do-operation on one residual vector).
- **Identification strategy:** activation patching with a **specificity-adjusted** null
  (random alt-bucket source at the same layer) à la Zhang & Nanda 2024 (arXiv 2309.16042).
  Top-1→CHECK read at the position that *predicts* the verb, input truncated strictly before
  the verb's first character (no source-token copy path; verified `forward_helpers.py:301-302`).
- **Assumptions:** (i) the patched position precedes verb commitment — ✅ verified; (ii) the
  effect is not a generic "any large injection flips it" artifact — addressed by the null
  subtraction; (iii) the target pool is representative — ⚠️ illegal_fold is rare.
- **Stress tests (done):**
  - 3-seed replication: top-1→CHECK = **100% / 100% / 100%** (`SUFFICIENCY_CI.md`).
  - Target-CLUSTERED bootstrap (independent unit = target decision, n=24, not 240 pairs):
    top-1→CHECK CI **[100%, 100%]**; mean Δ +26.83 nats CI [+26.04, +27.56].
  - Verb-generality: BET→illegal_FOLD flips to BET 80–100%; reverse FOLD→CHECK flips to FOLD.
- **If assumptions fail:** the magnitude (spec-adj nats) is seed-sensitive — per-seed range
  **[+15.6, +25.1]**, null varies ~5× by seed (`SUFFICIENCY_CI.md`). ⇒ **Report the categorical
  100% (robust) and the spec-adj as a RANGE, never the single pooled +18.3.** The categorical
  claim does not depend on the magnitude.
- **Confound check (the bet_to_call issue, §3):** the sufficiency targets are illegal_fold
  (bet=0) and sources are mixed-bet; a facing-bet source would push toward FOLD if the effect
  were bet-context, yet flips are uniformly to CHECK ⇒ sufficiency is **not** a bet-context
  artifact. ⏳ The clean confirmation is the bet-matched patch (Phase B2).

## 2. CLAIM: Necessity gradient (Llama sparse / Qwen L19 / Ministral none)

> Llama: a sparse triplet `[5,23,24]` @ L14 is behaviorally necessary (+23.5pp, McNemar
> p=2.1e-5 vs head-control; L15 negative control null). Qwen: necessity localizes to **L19
> attention** (+42.7pp, McNemar p=3.8e-5 vs an early L8 control; 5-head subset p=1e-3; L23
> saturation NS). Ministral: **no** head-localized necessity at 3 or 6 heads (control flips
> MORE, ✗ reversed p=8e-4).

- **Causal?** Yes — zero-ablation of attention components during generation.
- **Identification strategy:** **paired McNemar** of each ablation against a CONTROL ablation
  on the SAME records (Llama/Ministral: within-cell head control `[0,1,2]`; Qwen: cross-layer
  L8 whole-attention control), direction-aware (`necessity_significance.py`). All four
  `SIGNIFICANCE_*.md` reproduce exactly from committed `*_rows.jsonl`.
- **Assumptions & stress tests:**
  - (i) **Low baseline regeneration drift**, else ablation has no headroom. `REGEN_DRIFT.md`:
    Qwen clean_legal_fold = **15.3%** (OK) — this is *why* the headline uses that pool; Llama
    illegal_fold = **73.5%** (so Llama necessity is read only as the control-paired delta, and
    Llama *continuation* necessity is reported as **unmeasurable**). parse_fail = 0% everywhere.
  - (ii) **Seed consistency:** Qwen L19 net per base seed = +53/+40/+35pp (always the peak,
    L19>L23 in all 3); not a pooling artifact.
  - (iii) **Control adequacy** — STRESS THIS: the Qwen L8 control is a single early-vs-deep
    comparison; whole-attention ablation at L8 already yields +20.7pp (a generic
    "remove-attention→fold-less" floor). 🟡 Mitigants: L20/L23 are NOT elevated (non-monotonic
    in depth) and a 5-head L19 subset alone beats L8. ⏳ Cleaner: a **same-depth random-head
    control at L19** (Phase B4) — if L19's named heads beat random L19 heads, depth is ruled out.
- **If assumptions fail:** without a same-depth control the Qwen necessity reads as "deeper
  layers matter more," not "L19 specifically." B4 closes this. Llama's necessity is robust
  (clean-null control); Ministral's null is robust (control flips more).
- **Caveat (single-seed):** 🟡 Ministral's necessity pool is **78/80 base-seed-42**
  (`AUDIT_FINDINGS` + agent check); the null is real but effectively single-seed. ⏳ Phase B3
  rebalances to 3 seeds.

## 3. CLAIM: The direction is a *decision* representation, not just "facing a bet"

> The CHECK/CALL-vs-FOLD verb label is near-collinear with the observable `bet_to_call>0`
> (illegal_fold⟺bet=0, clean_legal_fold⟺bet>0), and a probe on `bet_to_call>0` matches the
> verb probe's accuracy (Llama .988=.988, Qwen .998/1.000). **Is the direction just the
> bet-context axis?**

- **Causal?** Partly: the probe is correlational; the patch (§1) is causal.
- **Identification strategy + stress test (done, CPU — `CONFOUND_PROJECTION.md`):** a pure
  bet-axis would place illegal_fold (bet=0) WITH check (bet=0). It does NOT — both fold types
  project to the same (fold) side, illegal_fold projects opposite to check, and
  **cos(w, verb-axis) ≈ +0.95–0.99 ≫ |cos(w, bet-axis)| ≈ 0.3–0.4** for all 3 models. The
  decision direction is **verb-aligned, not bet-aligned.**
- **Assumption being stressed:** the probe was *trained* on data including these illegal_folds,
  so this is *necessary* (a pure bet-detector could not place illegal_fold correctly) but not
  fully *sufficient* evidence.
- **If it fails / the clean test:** ⏳ a **held-out bet-matched probe** (train CALL@bet>0 vs
  FOLD@bet>0; and a bet-balanced probe) + the **bet-matched patch** (Phase B1/B2). Given the
  geometry, the expected outcome is that the direction survives bet-matching. **Until B1/B2
  land, state the claim as: "the decision axis is verb-aligned (geometry); a held-out
  bet-matched test is pending."**
- **Downgraded items:** the **mode-stability cosine (+0.51 Qwen)** compares two verb-aligned
  directions, so it is more defensible than feared, BUT (a) Llama uses a CoT-label fallback
  (different question — keep as caveat), and (b) the **Ministral probe is degenerate** (verb
  acc 0.900 == permuted-label floor; a random direction scores 0.946) → **drop Ministral from
  all probe/cosine claims**, keep it only for the causal sufficiency + necessity-null story.

## 4. CLAIM: Opponent-invariance (Qwen-only, Tier 4) — 🟡 most confound-exposed

> Qwen L23 patch flips clean_legal_fold targets to CHECK across 5 distinct opponent presets
> (+5 to +20 nats); Llama at noise floor; Ministral fails.

- **Identification caveat to DISCLOSE:** Tier-4 targets are clean_legal_fold (**bet>0**) and
  sources are mostly bet=0 → this cell is **cross-regime**, so it is the cell where the
  bet-context confound is most plausible. ⏳ Re-run the Tier-4 patch with **bet-matched sources
  (CALL@bet>0)** before making the invariance claim load-bearing. Also: the distinct-seed
  regeneration (§12) is verified on disk; report Qwen 5 / Llama 5 / Ministral 4 distinct presets.

---

## Summary of what changes in the writeup
1. Lead with causal sufficiency; demote probe/cosine to supporting + de-confounded.
2. Report sufficiency as categorical-100% + spec-adj RANGE (not pooled point).
3. Cite `CONFOUND_PROJECTION.md` to pre-empt the bet-context objection; promise/【then deliver】
   the bet-matched probe+patch (B1/B2) as the clean confirmation.
4. Drop Ministral from probe/cosine claims (degenerate probe); keep its causal null.
5. Add a same-depth L19 control (B4) so Qwen necessity is "L19-specific," not "deeper-matters."
6. Add an explicit "Identification assumptions" subsection (this file) — ahead of the 2605.08012 curve.
7. Tier-4 invariance: disclose cross-regime exposure; confirm with bet-matched sources.
