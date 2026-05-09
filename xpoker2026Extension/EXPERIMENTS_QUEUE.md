# Causal-patching experiments — queued + shelved

Living document. Captures (i) the experiments we're actively queuing for the
GPU box now, (ii) the shelf of experiments we considered and deliberately
deferred, with the trigger condition that would move each shelf item into the
active queue. Sister document to `JOURNEY.md` and `updates.md` (which record
what *has* been run); this records what *will* be run, *might* be run, and
*won't* be run, and why.

> **Last refresh:** 2026-05-09, after the Phase I batch (A1/D2/C1/B1) ran on
> the GPU box. See `updates.md` §14 for the results writeup. The queue below
> is the next batch.

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

## Recently shipped (since previous queue refresh)

These were in the prior "Queued" list and have now run; full results +
caveats are in `updates.md` §14.

| Code | Script | Status | Headline result |
|------|--------|--------|------------------|
| A1 | `scripts/run_causal_patching_qwen_seeds_replicate.sh` | ✅ all 3 cells | All three Qwen seeds show the same gradual L=19→23 ramp; cross-seed concordance confirmed. Caveat: s42 has n_target=4 so steps are coarse-grained. |
| D2 | `scripts/run_causal_patching_zero_ablation.sh` | ✅ all 3 cells | Zero-patch flips 0% top-1 → CHECK across all three models, all tested layers. **Circuit is content-addressable, not load-bearing.** |
| C1 | `scripts/run_causal_patching_verb_generality.sh` | ✅ all 3 cells | RAISE→CHECK flips at L*+1 to L*+2 in all three models — **L\* is a general decision circuit with two-stage internal structure** (FOLD-vs-not at L\*; BET-vs-CHECK 1-2 layers later). |
| B1 | `scripts/run_causal_patching_component_l14.sh` | ⚠️ Llama only | At Llama L=14: MLP irrelevant (-6%), attn ≈ 49% of residual, three heads (h5/h23/h24) carry ≈ 73% of per-head signal. Sparse-triplet but no single head clears the verb-flip threshold. Ministral B1 not yet run (gated on `RUN_MINISTRAL=1`). |
| A3 | `scripts/run_causal_patching_nocot_parity.sh` | ❌ auto-skipped | Non-CoT `scaled_*_enriched.jsonl` baseline logs not present at the expected paths on the GPU box. Re-queued below; needs an inputs check. |

---

## Queued — to run on the GPU box, in this priority order

Three items, in dependency order. Each has a wrapper script and a one-line
headline metric.

| # | Code | Script | Wall-clock | Headline reading |
|---|------|--------|-----------:|------------------|
| 1 | B1-Ministral + B1.5 | `scripts/run_causal_patching_component_l14_extras.sh` | ~60-90 min | Cell 1: does Ministral L=14 also show a sparse-triplet head pattern? Cell 2: does the Llama h5+h23+h24 triplet jointly clear the verb-flip threshold (linearity test)? Cell 3 (conditional): same triplet test for Ministral's top-3 heads. |
| 2 | A3 | `scripts/run_causal_patching_nocot_parity.sh` | ~3-6 h *if logs exist* | Re-queue. Run an `ls logs/scaled_*_informative_v2_enriched.jsonl` first; if present, fire the script. Outcome either way is paper-worthy: if L\* persists, circuit is intrinsic; if it shifts, CoT mechanistically reshapes internals. |
| 3 | B1 at L=15 (Llama) | `LAYER=15 bash scripts/run_causal_patching_component_l14.sh` | ~50 min | Verb-generality found that RAISE flips at L=15 in Llama (94.7%), not L=14 (44%). Repeating the component sweep at L=15 may show a *different* head pattern — possibly the BET-vs-CHECK heads if the two-stage decision story is right. |

### Cell 1.1 — Ministral B1 (component sweep at L=14)

> **Why:** Cross-model check on the head story. The Llama L=14 result
> identified head_05 / head_23 / head_24 as carrying ≈ 73% of the per-head
> signal. If Ministral also shows a sparse triplet at L=14 (even at
> different head indices), the localized-attention story becomes
> architecturally meaningful, not Llama-specific.

- **Script:** `bash scripts/run_causal_patching_component_l14_extras.sh`
  (cell 1; the script also runs cell 2 in the same invocation)
- **Inputs:** Ministral pooled enriched logs (s42 + s123 + s456)
- **Outputs:** `results/causal_patching/ministral8b_l14_components/`
- **Pass criterion:** if 2-4 heads each carry > 10% of residual magnitude,
  same-shape sparse-triplet across models. If all 32 heads contribute
  ±3% individually, dense-attention story (Llama-specific sparse triplet).

### Cell 1.2 — Llama B1.5 (head-triplet patch at L=14)

> **Why:** The Llama per-head sweep gave ≈ 73% of residual when summing
> the three top-head ratios (head_05 18% + head_23 35% + head_24 20%).
> But no single head individually flipped the verb. The triplet patch
> tests three things in one row:
>   - Does the joint triplet patch ≈ the linear sum (additivity)?
>   - Does the joint triplet patch ≈ the attn-only patch (49%)?
>   - Does the joint triplet flip the verb at the same rate as the
>     full residual?

- **Script:** same as cell 1.1 — both fire in one invocation.
- **Inputs:** Llama pooled enriched logs.
- **Outputs:** `results/causal_patching/llama8b_l14_head_triplet/`
  (the table will have rows: residual / attn / mlp / heads_05_23_24 /
  head_05 / head_23 / head_24)
- **Pass criteria** (three possible outcomes):
  - **heads_05_23_24 ≈ 73% of residual AND verb-flip ≥ 50%**: linearity holds,
    triplet IS the circuit, paper-banner.
  - **heads_05_23_24 ≈ residual (close to 100%) AND verb-flip ≈ 79%**: the
    triplet alone reproduces the full residual effect — even stronger
    paper-banner: the circuit IS the triplet.
  - **heads_05_23_24 < linear sum**: heads interfere; the per-head
    decomposition was misleading; no clean head story; paper says
    "weighted-combination across many heads."

### Cell 1.3 — Ministral B1.5 (conditional)

> **Why:** Cross-model triplet test. Only fires if cell 1.1 (Ministral B1)
> identifies a clear top-3 set of heads.

- **Trigger:** after cell 1.1 completes, read its SUMMARY_components.md and
  identify Ministral's three highest-ratio heads. Then re-run with
  `MINISTRAL_TRIPLET="<a> <b> <c>"`:
  ```
  MINISTRAL_TRIPLET="<a> <b> <c>" \
      SKIP_MINISTRAL_B1=1 SKIP_LLAMA_B1_5=1 \
      bash scripts/run_causal_patching_component_l14_extras.sh
  ```
- **Outputs:** `results/causal_patching/ministral8b_l14_head_triplet/`
- **Same three pass criteria as Llama B1.5.**

### Cell 2 — A3 (non-CoT patching parity)

> **Re-queued from the previous batch.** Auto-skipped because
> `logs/scaled_<model>8b_t0_s42_informative_v2_enriched.jsonl` were not
> present on the GPU box where the queue ran.

- **Inputs check first:** on the GPU box, run
  `ls logs/scaled_*_informative_v2_enriched.jsonl` and confirm at least
  Llama+Ministral have files. If they don't:
    - Either run `bash scripts/run_tier1a_small.sh` first (~3-6 h to
      regenerate the non-CoT baseline) — large compute outlay.
    - Or pass `LOGS=...` to the script with whatever non-CoT enriched logs
      do exist.
    - Or skip A3 entirely and rely on the existing CoT-only story (the
      paper still works without A3, just one fewer paragraph).
- **Then:** `bash scripts/run_causal_patching_nocot_parity.sh`
- **Outputs:** `results/causal_patching/{llama,ministral,qwen}8b_nocot_parity/`
- **Pass criterion (for "circuit is intrinsic"):** L\* boundary in non-CoT
  matches L\* boundary in CoT (within ±1-2 layers) for Llama and Ministral.

### Cell 3 — B1 at L=15 (Llama)

> **Why:** Verb-generality (C1) found that RAISE→CHECK *flips* at L=15 in
> Llama (94.7% top-1 → BET_RAISE), not at L=14 (only 44%). The component
> profile at L=15 might therefore be different from L=14:
>   - If L=15's heads ⊃ L=14's heads + new heads: staged-decision picture
>     (FOLD-vs-not heads at L=14, BET-vs-CHECK heads at L=15).
>   - If L=15's heads ≈ L=14's heads with bigger ratios: same circuit,
>     just one layer further into saturation.
>   - If L=15's heads ⊥ L=14's heads: completely separate verb circuits
>     at adjacent layers — surprising, paper-worthy if true.

- **Script:** `LAYER=15 bash scripts/run_causal_patching_component_l14.sh`
  (the existing script's `LAYER` env knob covers this — no new script
  needed)
- **Outputs:** `results/causal_patching/llama8b_l15_components/`
- **Pass criterion:** report whichever of the three patterns shows up;
  no a priori prediction.

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
