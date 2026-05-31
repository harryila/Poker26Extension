# Claims, identification assumptions, and limitations (writeup reference)

Companion to `AUDIT_FINDINGS.md`. Structures the paper's causal claims per the disclosure
norm of *"Mechanistic Interpretability Must Disclose Identification Assumptions for Causal
Claims"* (arXiv 2605.08012): for each claim state (a) whether it is causal, (b) the
identification strategy, (c) the assumptions, (d) at least one stress test, (e) how the
conclusion shifts if the assumption fails. **Validation is not identification.**

Status legend: ✅ supported by committed evidence · 🟡 supported with a stated caveat ·
⏳ requires a queued GPU experiment (see `REDO_PLAN.md` / Phase B–D).

---

## RESULTS LANDED (2026-05-29 — Phase B/C/D GPU follow-ups)

**Rejection-proofing (Phase B) — the crux is resolved:**
- ✅ **Bet-matched PROBE (B1, all 3 models):** with bet held constant or balanced, the verb is
  still decodable far above the permuted floor — bet-balanced acc **Qwen 1.000 / Llama 0.996 /
  Ministral 1.000** (floors ~0.50). The "decision direction = facing-a-bet" objection is dead.
- ✅ **Bet-matched PATCH (B2):** sufficiency SURVIVES bet held constant. The previously-vulnerable
  Tier-4-style cell — Qwen L23 facing (CALL@bet>0 → legal_fold@bet>0) — is **100% top-1→CHECK,
  spec-adj +19.6**. Llama L15 facing 100%/+13.0; Ministral 86–100%. (Llama L14 facing 100%, L14
  nobet 69% — L14 is pre-saturation, consistent.)
- ✅ **Ministral 3-seed sufficiency:** L16 pooled **100% / +8.0** (was single-seed s42 +7.4).
- 🟡 **B4 same-depth control sharpens Qwen necessity:** at L19, the named top-5 heads flip MORE
  than random L19 5-head sets but only **marginally (p=0.06 vs best random draw)**; all L19 sets
  flip ~20–37 pp. ⇒ Qwen necessity is **L19-LAYER-specific but DISTRIBUTED across L19 heads**, NOT
  a sparse head circuit. (The earlier L8 cross-layer control overstated head-specificity.) This
  *strengthens* the gradient: Llama = sparse specific heads; Qwen = distributed-within-L19.
- 🟡 **Ministral null reconfirmed at n=150** (control flips MORE, p<1e-5) but the illegal_fold pool
  is **147/150 seed-42** — illegal-fold events concentrate in seed 42 for Ministral; can't be
  seed-balanced (data property, not a bug). The null is robust regardless.
- Integrity: all SIGNIFICANCE_*.md reproduce EXACTLY from the pulled rows.

**Novelty (Phase C):**
- ✅/🟡 **Encode-vs-decode (C1, all 3 models):** the residual linearly encodes the correct Bayesian
  trash mass (probe |err| **0.006–0.027**) while the model's STATED belief severely under-reports it
  (|err| **0.32–0.63**; Ministral states 0.049 vs truth 0.682). JS(stated,oracle) 0.20–0.30.
  → "the correct posterior is linearly available at the decision layer; the verbalized belief
  discards it." **CAVEAT (must control):** the oracle is a deterministic function of the prompt
  inputs, which the residual encodes — so decodability could be trivial input-presence, not
  "computed." Needs the **early-layer control** (decode oracle at L2 vs L*); claim only "linearly
  available + discarded by readout" until then, not "knows."
- ❌ **Steering de-bias (C2): NULL.** Adding the trash direction at L23 (last_only) does NOT beat
  the random-direction control (alpha-8: trash −6.7 vs control −3.2; top-1→CHECK 0% throughout).
  Likely because L23 is the commit layer and the direction acts like a norm bump. Needs rework
  (steer at the L19 compute layer; tune alpha; target base-rate-neglect spots; belief-JS readout).
- ❌ **Behavior-at-scale (C3): UNUSABLE as run — root-caused.** The env is CORRECT (random-vs-random
  play gives varied deltas +2/−10/+16/+20/… and correctly awards pots). The constant −4/hand was a
  **behavioral degeneracy of the non-CoT agent**: it ran **cot_mode=False** (mismatch — the circuit
  was characterized on CoT) and the non-CoT Qwen plays a fixed losing line (limp/bet then FOLD to the
  aggressive opponent's raise, contributing exactly 4 every hand → −4), so there is no win-rate
  headroom to detect a circuit effect. The steer run additionally induced **60% fallbacks** (alpha
  too large, all-positions). FIX (in v2): run with `--cot` (varied play, as in the original logs);
  steer with small alpha + last_only + a parse-rate guard. No env fix needed.

**Depth (Phase D):**
- 🟡 **D1 SVD:** residual at L18–23 is ~20-rank (90% var), but the decision direction is NOT in the
  top singular vectors (cos 0.07–0.13) — it is a **low-variance functional direction**, not a
  dominant axis of variation. Exploratory; mildly interesting, not load-bearing.

**Net:** the paper's central confound is closed (B1+B2). The new headline is C1 (encode-vs-decode),
pending its early-layer control. C2/C3 need rework before any steering/behavioral claim.

---

## RESULTS LANDED v2-batch (2026-05-29 — C1-control, C2-steering, C3-CoT) + adversarial verification

A 3-investigator → 3-skeptic workflow (all skeptics tried hard to refute; all AGREED) plus my
own decisive cross-checks settled the three v2 follow-ups:

- ❌→🔧 **Steering (C2): ARTIFACT, not a genuine null.** The steered vector
  (`steer_trash_direction.npz`) is the residual→oracle-trash RIDGE direction, **orthogonal to
  the causal decision axis** (cos to weight_vec/centroid_diff = 0.006; Cohen's d 0.93 vs
  10.7–14.4), and alpha 2/4/8 = **6–25× the natural check−fold gap** (=2/4/8× the residual norm
  158). The random control swung the logit as much as the trash direction, and generation
  parse-rate collapsed 1.0→0.0 identically for both = direction-independent norm destruction.
  **The corrected vector was never actually run** — so "it will steer" is a prediction, and ADD-
  steering is mechanistically weaker than the REPLACE-patching that flips 100%. → re-run P0 below.

- ❌ **Encode-vs-decode "knows" (C1): INPUT-PRESENCE — claim is DEAD (verifier: understated).**
  A 4-number readout of opponent BET/RAISE/CALL/CHECK counts recovers oracle trash mass at
  R²=0.962/0.960/0.974 (Llama/Qwen/Ministral) — **above the residual probe at every layer**
  (L2≈L*). A parameter-free group-mean over 27/58 count-tuples reproduces R²≈0.97 (rules out
  leakage); partialling opp-counts out of both sides leaves the best probe (Ministral L16) at
  residual R²=**−1.21**. The oracle posterior is a near-deterministic function of prompt token
  counts. **DROP "the model computes the Bayesian posterior."** The surviving, quantified finding:
  *the information sufficient for the posterior is linearly present in the prompt yet the
  VERBALIZED belief discards it* (stated trash abs-err 0.32/0.56/0.63; JS 0.20/0.25/0.30 nats;
  stated belief uncorrelated/anti-correlated with oracle). Real, but a property of the prompt +
  readout, not an internal computed representation.

- 🔧 **C3 gameplay: NEEDS REDESIGN + a real reinterpretation.** Net chips is dead-noisy (paired
  diff −0.127/hand, SE 0.78). On 670 byte-matched spots: **fold-rate UNCHANGED** (0.556→0.562,
  McNemar p≈1) while **bet/raise DOUBLES** (0.449→0.919, z=7.1; confirmed at the byte-identical
  first decision z≈8.9 with FOLD=0 in BOTH conditions). So in free play L19[31,3,21,1,0] reads as
  an **aggression/bet-suppression component, not a fold circuit**. CoT fixed the v1 degeneracy
  (31 distinct deltas); the steer arm is broken (1791/1791 forced fallbacks = over-steer).

### MY CROSS-CHECK reconciling C3 with the necessity headline (load-bearing):
When L19 ablation flips a *recorded* FOLD (inference-ablation pool), where does it go?
- Qwen L19 whole-attn: 70/127 flip → **57% CHECK, 43% BET**.  L19 top-5 heads: **68% CHECK, 32% BET**.  L23: 76% CHECK.
⇒ The two pipelines are BOTH right on their own pools: on *recorded-fold* spots ablation reduces
fold (majority to CHECK), but in *free play across all spots* the dominant effect is check→bet
with no net fold reduction. **Honest reframe: L19 modulates the fold-vs-(check/bet) decision on
fold-committed spots and suppresses aggression in general play; "necessary for FOLD" is
pool-dependent and too strong.** A fixed facing-bet battery (P1) is needed to settle it cleanly.

### What survives as the paper backbone (post-v2):
✅ Causal **sufficiency** (REPLACE-patch flips FOLD→CHECK ~100%, 3 seeds, +18.3 nats) — headline.
✅ **Bet-matched control** (decode-the-decision-not-the-input, acc 1.00 vs 0.5 floor) — the
   methodological contribution that beats the only direct competitor (arXiv 2512.23722, workshop).
🟡 **Necessity gradient** — DOWNGRADE from "circuit" to "layer localization" until hardened: Qwen
   L19 is p=3.8e-5 vs an early-layer control but only **marginal vs same-depth random heads**
   (p≈0.06) — "you localized a layer, not a circuit." Reframe as a *non-universal* consolidation
   gradient (Llama sparse → Qwen graded/distributed → Ministral flow-through).
DROP: "computes Bayesian posterior / world model"; "L19 is a fold circuit"; net-chips as a readout.

### Venue (per literature scan): borderline main-track.
Strong workshop / mid-tier accept as-is (beats 2512.23722 on causal rigor). Two experiments lift
it to credible main-track: (P0) harden necessity into a circuit claim; (P0→P2) land ONE working
steering/de-bias payoff. Frame as "a causally sufficient decision direction + a non-universal
necessity gradient," lineage LLMs-know-more-than-they-show (2410.02707) / verbalized-vs-internal
(2603.25052); demote the world-model story to an honest input-presence negative (2509.13316).

---

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
