# Causal-patching experiments — queued + shelved

Living document. Captures (i) the experiments we're actively queuing for the
GPU box now, (ii) the shelf of experiments we considered and deliberately
deferred, with the trigger condition that would move each shelf item into the
active queue. Sister document to `JOURNEY.md` and `updates.md` (which record
what *has* been run); this records what *will* be run, *might* be run, and
*won't* be run, and why.

> **Last refresh:** 2026-05-10, after the Phase J batch (B1-Ministral-L=14 +
> Llama B1.5 triplet + Ministral B1.5 triplet + Llama L=15 + A3 audit) ran
> on the GPU box. See `updates.md` §15 for the results writeup. The queue
> below is the next batch — and the new top priority is the Ministral
> component sweep at the *saturation* layer (L=15 or L=16), since the L=14
> sweep we just ran was at the wrong layer for cross-model comparison.

---

## How to use this document

- **Queued (active)** items are the next things to run. Each one has a script,
  a target output directory, and a one-line headline metric to read on the
  resulting `SUMMARY.md`.
- **Shelved (deferred)** items are not currently planned. Each lists *what
  would have to be true* for us to move it into the active queue (a result, a
  reviewer comment, or a time budget).
- We do **not** add items to "queued" without a matching script and runbook
  line. If a script is missing, the item belongs on the shelf.

---

## Recently shipped — Phase J (this round)

These were in the prior "Queued" list and have now run; full results +
caveats are in `updates.md` §15.

| Code | Script | Status | Headline result |
|------|--------|--------|------------------|
| A3 (audit) | `nocot_parity_a3/illegal_fold_audit.txt` + `SUMMARY.md` | ✅ negative finding (paper-banner) | Across **18 non-CoT conditions (3 models × 3 seeds × 2 temps)**, illegal_fold count = **0**. The pathology is 100% conditional on CoT. **Reframes the entire L\* claim from "decision circuit" to "CoT-induced deliberation circuit."** No re-run possible or needed. |
| B1.5 | `run_causal_patching_component_l14_extras.sh` (cell 2) | ✅ Llama L=14 triplet | `heads_05_23_24` jointly: ratio 65%, top-1 → CHECK 48.7% (vs residual's 79%). Slightly subadditive (linear sum 73%). The triplet jointly DOES clear the verb-flip threshold; no single head does. |
| B1 | `run_causal_patching_component_l14_extras.sh` (cell 1) | ⚠️ Ministral L=14 — wrong layer | Ministral's residual patch at L=14 only flips 2% top-1 → CHECK; L=14 is the *start* of Ministral's flip, not saturation. MLP at L=14 is *anti-CHECK* (-34%). Top heads h20 (32%), h21 (25%), h09 (12%) — different indices than Llama. **Re-queue at L=15 or L=16.** |
| B1.5 | `run_causal_patching_component_l14_extras.sh` (cell 3) | ⚠️ Ministral L=14 — wrong layer | `heads_09_20_21`: 72% ratio, 0% top-1 → CHECK. Almost perfectly additive (linear sum 69%) but residual itself doesn't flip at L=14, so the "circuit-completeness" question can't be answered here. |
| L=15 sweep | `LAYER=15 bash scripts/run_causal_patching_component_l14.sh` | ✅ Llama L=15 components | residual=100% flip; attn=17%, mlp=9%; max single head 19%. **Neither sublayer carries the signal locally — L=15 is *commitment*, L=14 is *computation*.** Two-layer two-stage circuit story confirmed. |

---

## Queued — to run on the GPU box, in this priority order

Three items, all small. The cross-model head story needs Ministral at the
*saturated* layer (L=14 was the wrong choice for cross-model comparison),
and the Llama triplet → quartet extension closes the gap from 65% to
hopefully ≈ residual.

| # | Code | Command | Wall-clock | Headline reading |
|---|------|---------|-----------:|------------------|
| 1 | B1-Ministral-L15 | `LAYER=15 SKIP_LLAMA_B1_5=1 bash scripts/run_causal_patching_component_l14_extras.sh` | ~50-60 min | Component sweep at Ministral's *transition* layer (verb-generality showed RAISE flips at L=16, so L=15 is mid-flip). Identify the top-3 positive heads. |
| 2 | B1-Ministral-L16 | `LAYER=16 SKIP_LLAMA_B1_5=1 bash scripts/run_causal_patching_component_l14_extras.sh` | ~50-60 min | Component sweep at Ministral's *saturated* layer. **This is the proper analog of Llama L=14.** Identify the top-3 positive heads. |
| 3 | B1.5-Ministral | After #2 completes, with chosen triplet | ~10 min | Triplet patch at the layer that flips. Direct cross-model analog of Llama's `heads_05_23_24` finding. |
| 4 | B1.5+ Llama quartet | One-shot `python -m experiments.component_patching ...` | ~10 min | Does adding head_02 to the Llama triplet close the gap from 65% to ≈100%? |

### Cell 1 — Ministral B1 at L=15 (transition layer)

> **Why:** Ministral's L=14 component sweep showed only 2% verb-flip at
> the residual level — L=14 is the *start* of Ministral's flip, not
> saturation. The verb-generality result (§14d) showed BET_RAISE flips at
> Ministral L=16 (96%), with L=15 being the clean-decision-mid-transition
> layer. We want a clean component decomposition at the layer where the
> flip is *happening*, not before or after.

- **Command:**
  ```
  LAYER=15 SKIP_LLAMA_B1_5=1 \
      bash scripts/run_causal_patching_component_l14_extras.sh
  ```
- **Output:** `results/causal_patching/ministral8b_l15_components/`
- **Read:** the per-head ratio column. Identify top-3 positive heads
  (those will be candidates for cell 3's triplet test).

### Cell 2 — Ministral B1 at L=16 (saturated layer)

> **Why:** L=16 is where Ministral's verb is *committed* (per the existing
> sweeps and the C1 verb-generality result). The proper cross-model analog
> of Llama L=14 is whichever Ministral layer has the highest residual-
> level top-1 → CHECK rate; based on existing data, that's L=16.

- **Command:**
  ```
  LAYER=16 SKIP_LLAMA_B1_5=1 \
      bash scripts/run_causal_patching_component_l14_extras.sh
  ```
- **Output:** `results/causal_patching/ministral8b_l16_components/`
- **Read:** confirm residual-level flip rate is high (≥80%). If yes,
  identify top-3 positive heads. If no (still <50%), Ministral's true
  saturation is even later (L=17+) and we need to extend.

### Cell 3 — Ministral B1.5 at the saturated layer

> **Why:** Cross-model triplet test, properly. The Ministral L=14 triplet
> result (§14e) was at the wrong layer; we re-do it at the saturation
> layer with the heads identified in cell 2 (or cell 1 if cell 2 says
> L=16 isn't yet saturated).

- **After cell 2 completes**, read `ministral8b_l16_components/SUMMARY_components.md`
  and pick the three highest-ratio positive heads (call them `<a> <b> <c>`).
- **Command:**
  ```
  LAYER=16 MINISTRAL_TRIPLET="<a> <b> <c>" \
      SKIP_MINISTRAL_B1=1 SKIP_LLAMA_B1_5=1 \
      bash scripts/run_causal_patching_component_l14_extras.sh
  ```
- **Output:** `results/causal_patching/ministral8b_l16_head_triplet/`
- **Three possible outcomes** (same as Llama B1.5):
  - **triplet ≈ residual AND verb-flip clears threshold**: triplet IS the
    circuit; cross-model symmetric sparse-attention story confirmed.
  - **triplet ≈ linear sum AND verb-flip clears threshold partially**:
    Ministral analog of Llama's "subadditive but jointly clears threshold"
    finding — paper-banner cross-model.
  - **triplet doesn't flip the verb**: dense-attention story for Ministral;
    Llama-specific sparse-head finding.

### Cell 4 — Llama L=14 quartet (extending the triplet)

> **Why:** Llama's triplet `heads_05_23_24` reaches 65% of residual / 49%
> verb-flip; head_02's individual ratio was 10% (next-highest after the
> triplet). Adding it should close the gap.

- **Command** (one-shot, no script needed):
  ```
  python -m experiments.component_patching \
      --enriched-log logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
                     logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
                     logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz \
      --source-bucket clean_check_or_call \
      --target-bucket illegal_fold \
      --layer 14 \
      --components residual attn mlp head_subset head \
      --head-indices 2 5 23 24 \
      --n-source 10 --n-target 30 \
      --seed 42 \
      --out-dir results/causal_patching/llama8b_l14_head_quartet \
      --device cuda --dtype bfloat16
  ```
- **Output:** `results/causal_patching/llama8b_l14_head_quartet/`
  (one row labeled `heads_02_05_23_24`)
- **Pass criteria**:
  - **quartet ≈ residual AND verb-flip ≈ 79%**: 4 heads ARE the circuit;
    strongest possible per-head story.
  - **quartet > triplet but < residual**: the gap closes monotonically;
    the circuit is "weighted-combination over a small set" (5–10 heads).
    Cite both numbers.
  - **quartet ≈ triplet**: head_02 doesn't add anything; the 35% of
    residual outside the triplet lives in residual flow-through, not
    in additional heads. Implication: the per-head story tops out at the
    triplet.

---

## Shelf — considered, deferred, with trigger conditions

Each item below is **not** scheduled, but is an obvious next step under some
condition. We track them here so they don't get lost.

### B2 — Linear "decision direction" probe at L*=14
> *Trigger:* B1 finishes and shows a sparse head profile. Then a single
> direction probe is the natural follow-up — does one linear projection
> of the L=14 residual explain >70% of the patching variance? If yes,
> single-direction story for the paper. Cost: ~2-3 h, mostly free if we
> cached residuals during B1.

### B3 — Belief-conditional patching
> *Trigger:* B2 shows a clear decision direction. Then we ask whether
> *patching* the model's belief vector (from the analysis pipeline's
> oracle posteriors) at L=14 changes downstream behavior in the same way.
> This is the bridge from causal-patching results to the paper's
> belief-inertia narrative. Cost: ~3-5 days new code, paper-banner result
> if it works.

### A2 — Per-seed reverse-direction cells (all 3 models)
> *Trigger:* a reviewer asks for cross-seed evidence specifically in the
> reverse direction (currently we only have per-seed forward + pooled
> reverse). The pooled reverse pilot already shows the predicted clean
> reversal in all three models, so per-seed reverse is precision-of-claim,
> not a new finding. Cost: ~6 h on H100 if requested.

### C2 — Hand-strength specificity slice
> *Trigger:* C1 shows verb-general L*=14 (not just fold-or-not). Then
> the next question is "is the decision conditioned on hand strength,
> or pure verb?" Slice `clean_check_or_call` by strong / marginal / weak
> hand and re-patch at L=14. If the patching effect depends on
> source-target hand-strength matching, the circuit is conditional;
> if not, verb-only. Cost: ~1 day code + ~6 h compute.

### C3 — Opponent-preset robustness (Tier 4 from EXPERIMENTS.md)
> *Trigger:* a reviewer specifically raises opponent-mismatch as the
> mechanism behind L*=14, OR we have time and want to test whether the
> circuit is opponent-conditional. Requires running new baseline cells
> with different opponent presets first (1-2 days). Cost: ~3 days end-to-end.

### D1 — Position-generality (patch at belief vs at action verb)
> *Trigger:* B1 succeeds and we want to know whether the same heads at
> L=14 are also active when the model emits its belief distribution
> (earlier in the sequence) or only at the action-verb position.
> Cheapest way to test "is the decision encoded throughout the response,
> or only at the action point?" Cost: ~half a day code + ~4 h compute.

### Tier-3 — 70B / frontier cross-scale patching
> *Trigger:* none of the 8B-class results are surprising and we want to
> see whether the same circuit shows up at 70B. Likely requires multi-GPU
> or shifting to API-only behavioral checks. Big infra lift. *Probably
> never run unless the paper goes for revisions and a reviewer asks
> specifically.*

### ACDC-style automated circuit discovery
> *Trigger:* B1 + C1 succeed and there is appetite to formalize the
> circuit graph automatically. The paper does not require this; it would
> be a follow-on workshop submission. Cost: ~2 weeks (entirely new
> infrastructure).

### Path patching (upstream contribution decomposition)
> *Trigger:* B1 finds a sparse head set. Then path patching identifies
> which earlier-layer heads/MLPs feed *into* those L=14 heads. Standard
> mech-interp methodology, ~1 week of code, ~1 day of compute.

### Probe for opponent's strategy in residual at L=14
> *Trigger:* C3 shows opponent-conditioning. Then the question is whether
> the model has internalized the opponent's preset as a linear direction
> in the residual. Cost: ~half a day if probe infra exists, ~3 days if not.

### Out-of-distribution robustness (4-card / Omaha-ish hands)
> *Trigger:* a reviewer asks "does the circuit generalize beyond NLHE?"
> Requires building a new env or major adapter work. *Probably never
> run.*

---

## Things we are explicitly **not** going to do

These were considered and ruled out for principled reasons, not just time:

- **More 8B model families** (Gemma, Phi, etc.). Tier 1A.small already
  established the 3-family pattern; adding more models without theory
  predicting their behavior is data fishing.
- **Tighter-than-0.10-nat pre-flight gate.** The gate is bf16-precision-
  justified. Tightening it produces a worse (false-positive-prone) gate
  with no scientific justification. Documented in
  `experiments/verify_prompt_reconstruction.py`.
- **Re-run the existing pooled sweeps with more random-source samples.**
  The 3-seed pooled estimates already converge tightly; specificity-adjusted
  Δ wouldn't change qualitatively with more null sources.
- **Resampling existing results to chase higher-than-saturation top-1
  rates.** s456 saturates at 90% (16/18) due to small N; this is a
  reporting issue, not a measurement issue.

---

## Cross-references

- Methods log: `logs/preflight_relaxed_gate.txt` (every enriched log used
  by the chain passes the relaxed pre-flight gate)
- Verifier source: `experiments/verify_prompt_reconstruction.py`
- Driver source: `experiments/causal_patching.py`
- Patching primitives: `poker_env/interp/patching.py`
- Existing per-cell results: `results/causal_patching/`
- Sister document (what HAS been done): `JOURNEY.md`

---

## Update protocol

- **Move to "queued":** add the script path, expected wall-clock, and a
  one-line headline metric. Do *not* queue without a script.
- **Move from "queued" to "done":** delete from this file; add a one-line
  entry to `JOURNEY.md` with a link to the resulting SUMMARY.md.
- **Move from "shelf" to "queued":** state the trigger condition that fired
  in the queued entry.
- **Add to "shelf":** include the trigger condition. Items without trigger
  conditions go to "explicitly not going to do".
