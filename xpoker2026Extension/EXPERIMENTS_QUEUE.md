# Causal-patching experiments — queued + shelved

Living document. Captures (i) the experiments we're actively queuing for the
GPU box now, (ii) the shelf of experiments we considered and deliberately
deferred, with the trigger condition that would move each shelf item into the
active queue. Sister document to `JOURNEY.md` and `updates.md` (which record
what *has* been run); this records what *will* be run, *might* be run, and
*won't* be run, and why.

> **Last refresh:** 2026-05-10, after the Phase K closing batch (Ministral
> L=15 + L=16 component sweeps + L=16 triplet/sextet + Llama quartet) ran
> on the GPU box. See `updates.md` §16 for the results writeup. The
> cross-model head-circuit story is now complete; **the queue below is
> very small** — we are at a natural stopping point for the patching
> experiments, and most remaining items are paper-writeup tasks rather
> than new GPU runs.

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

## Recently shipped — Phase K (closing batch)

Full results + caveats in `updates.md` §16. The cross-model head-circuit
comparison is now complete.

| Code | Script | Status | Headline result |
|------|--------|--------|------------------|
| Cell 1 (B1 Ministral L=15) | `LAYER=15 SKIP_LLAMA_B1_5=1 bash scripts/run_causal_patching_component_l14_extras.sh` | ✅ | residual 36% (transition layer); attn 19%, mlp 5%; top heads h21/h30/h15/h08/h13 at 11–23%. *Sparser-ish but not committed.* |
| Cell 2 (B1 Ministral L=16) | `LAYER=16 SKIP_LLAMA_B1_5=1 bash scripts/run_causal_patching_component_l14_extras.sh` | ✅ | **Saturation layer**: residual 100% flip; attn 38%, mlp 13%; head_22 dominates at 31%, long tail. The proper cross-model analog of Llama L=14. |
| Cell 3 (B1.5 Ministral L=16 triplet) | `MINISTRAL_TRIPLET="22 9 15" LAYER=16 SKIP_MINISTRAL_B1=1 SKIP_LLAMA_B1_5=1 ...` | ✅ | `heads_09_15_22`: ratio 43%, top-1 → CHECK **3%**. Exactly additive (linear sum 45%). Triplet ≈ attn-only — does NOT clear verb-flip threshold. |
| Bonus (Ministral L=16 sextet) | same script with `MINISTRAL_TRIPLET="22 9 15 30 31 24"` | ✅ | `heads_09_15_22_24_30_31`: ratio 55%, top-1 → CHECK **37%**. Still additive (linear sum 54%). **Adding three small heads (h24/30/31, each ~3%) is what crosses the verb-flip threshold.** |
| Cell 4 (B1.5+ Llama quartet) | one-shot `python -m experiments.component_patching ...` | ✅ | `heads_02_05_23_24`: ratio 77%, top-1 → CHECK 69%. Adding head_02 closes most of the gap from triplet's 65% / 49%; mildly subadditive (linear sum 83%). |

### Cross-model head-circuit summary (paper-ready)

| Model | L\* | residual | attn-only | top single head | top-3 triplet | top-4 quartet | top-6 sextet |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Llama 8B** | 14 | 100% / 79% | 49% / 14% | h23 35% / 1% | **65% / 49%** | **77% / 69%** | (not tested) |
| **Ministral 8B** | 16 | 100% / 100% | 38% / 6% | h22 31% / 0% | 43% / 3% | (n/a) | **55% / 37%** |
| **Qwen 8B** | 22-23 | 100% / 100% | (not measured) | — | — | — | — |

(Format: ratio-to-residual / top-1 → CHECK)

**Cross-model takeaway**: Llama is "narrow-and-deep" (3-4 heads, mildly subadditive); Ministral is "wide-and-shallow" (≥6 heads, perfectly additive). Both share an approximate verb-flip threshold around Δ ≈ +4-5 nats, but Llama has a soft sigmoid (residual at +7.90 → 79% only) while Ministral has a steep one (residual at +7.81 → 100%).

---

## Status: cross-model patching story is solid; ONE substantive analytical hole remains

After Phase K, the patching-experiments story is mostly complete:

1. Cross-seed concordance in all three models (Phase H + Phase I A1)
2. Forward + reverse patching in all three models (earlier phases)
3. Content-addressable circuit (Phase I D2)
4. Verb-general circuit with two-stage internal structure (Phase I C1)
5. Llama L=14 head decomposition: narrow-and-deep three-head triplet,
   head_02 quartet closes most of the gap (Phase I B1, Phase J B1.5, Phase K cell 4)
6. Llama L=15 commitment-vs-computation distinction (Phase J)
7. Ministral L=16 head decomposition: wide-and-shallow, dominant head +
   long tail, additive over six heads (Phase K cells 2/3 + sextet)
8. **A3 negative finding**: the FAILURE MODE is 100% CoT-conditional (Phase J)

**Key analytical hole**: claim 8 demonstrates the failure-mode is
CoT-conditional, but does NOT demonstrate that the *circuit itself* is
CoT-conditional. Those are different claims, and a reviewer will catch
the slippage. The non-CoT clean→clean experiment below distinguishes
them; it is the highest-value remaining experiment.

---

## Queued — in priority order

| # | Code | Script / Command | Wall-clock | Why |
|---|------|-----------------|-----------:|-----|
| **1** | **Non-CoT clean→clean (CRITICAL)** | `bash scripts/run_causal_patching_nocot_clean_to_clean.sh` | ~60-80 min | **Closes the analytical hole between "failure mode is CoT-conditional" (demonstrated) and "circuit is CoT-conditional" (NOT demonstrated). Paper-changing either way.** |
| 2 | Qwen B1 at L=23 | one-shot `python -m experiments.component_patching ...` (see below) | ~50 min | Closes the 3-model head-decomposition picture. Predicted dense distribution; cheap to confirm. |
| 3 | Reverse-direction component decomposition | `--source-bucket clean_legal_fold --target-bucket clean_check_or_call --layer 14` | ~50 min | Are h5/h23/h24 the same heads in the FOLD→CHECK direction? Tests bidirectional vs unidirectional encoding. |
| 4 | Per-seed head consistency at Llama L=14 | three separate `component_patching` invocations on s42, s123, s456 | ~150 min | Are h5/h23/h24 robust to seed, or did pooling smear it? Reviewer-defense robustness check. |

### Item 1 — Non-CoT clean→clean (the critical experiment)

> **Why this is the highest-priority remaining experiment.**
>
> A3 demonstrated:
>     The illegal_fold FAILURE MODE is 100% CoT-conditional. ✅
>
> A3 did NOT demonstrate (despite our previous writeup implying it):
>     The L\* CIRCUIT itself is CoT-conditional. ❌
>
> Those are different claims. The circuit could exist intrinsically and just never be traversed pathologically without CoT. To distinguish, we patch a clean_check_or_call source's L\* residual into a clean_legal_fold target's forward pass — both in non-CoT mode. The verb either flips or it doesn't.

- **Script:** `bash scripts/run_causal_patching_nocot_clean_to_clean.sh`
- **Inputs:** `logs/scaled_<model>8b_t0_s42_informative_v2_enriched.jsonl`
  (audit confirmed Llama and Qwen have ≥3 of each clean bucket; Ministral
  is too sparse and is auto-skipped)
- **Layer per model:** Llama L=14, Qwen L=23. Single layer (not a sweep);
  the question is "does the circuit work at the same L\*?", not "is there
  a different L\*?"
- **Outputs:**
  - `results/causal_patching/llama8b_nocot_clean_to_clean_l14/SUMMARY.md`
  - `results/causal_patching/qwen8b_nocot_clean_to_clean_l23/SUMMARY.md`
- **Three possible outcomes** (each is paper-grade — there is no failed-
  experiment result here):
  - **Verb flips ≥50% AND spec-adj Δ ≥ +5 nats**: circuit IS intrinsic.
    Strong rephrasing for the paper:
    *"L\* is the model's intrinsic action-decision circuit; CoT exposes
    a pathway through it (the FOLD-pull-then-stuck failure) that one-shot
    decoding never traverses."*
  - **Verb flips <10% AND spec-adj Δ < +2 nats**: circuit IS CoT-induced.
    Original "deliberation-circuit" framing is correct as written.
  - **Anything in between**: nuanced; weakened-but-present circuit in
    non-CoT. Report both numbers; discuss attenuation.
- **Compare against** the corresponding CoT pooled-sweep number at the
  same L\* (Llama CoT L=14: spec-adj Δ ≈ +6.48, flip ≈ 79%; Qwen CoT
  L=23: spec-adj Δ ≈ +18.3, flip ≈ 100%).

### Item 2 — Qwen B1 at L=23 (the open cell)

> **Why this is queued**: Llama and Ministral got head-decomposition;
> Qwen did not. The cross-model table currently has 2/3 rows. A
> reviewer will ask "why did you only do this for two models?"
>
> **Why it's lower-priority than Item 1**: Qwen's distributedness is
> already documented at the residual-stream level. The per-head story
> is highly likely to look "dense, no triplet" — predicted by the
> existing data. Running the experiment confirms a prediction; it
> doesn't change the paper's claim.

- **Command** (one-shot direct invocation):
  ```
  python -m experiments.component_patching \
      --enriched-log logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
                     logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
                     logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz \
      --source-bucket clean_check_or_call \
      --target-bucket illegal_fold \
      --layer 23 \
      --components residual attn mlp head \
      --head-indices all \
      --n-source 10 --n-target 30 \
      --seed 42 \
      --out-dir results/causal_patching/qwen8b_l23_components \
      --device cuda --dtype bfloat16
  ```
- **Pass criterion (predicted)**: residual 100% / 100% flip; attn modest
  (~30-50%); mlp small; **no head with ratio >15%**, all 32 heads in a
  ±5% range. The "dense distribution" outcome — completes the cross-model
  table with three qualitatively different head structures.
- **If a sparse triplet does show up**: surprising, paper-relevant; run
  the triplet-patch protocol on it.

### Item 3 — Reverse-direction component decomposition (Llama L=14)

> **Why**: We've decomposed the FORWARD (CHECK → FOLD-target) circuit at
> Llama L=14 and identified h5/h23/h24. Are those the SAME heads in the
> REVERSE direction (FOLD → CHECK-target)? If yes, the heads encode
> "CHECK content" bidirectionally. If different, there are *separate*
> CHECK-encoding and FOLD-encoding heads — which would itself be
> paper-worthy because it implies the circuit is verb-direction-specific
> at the head level. Either outcome is informative.

- **Command** (one-shot, mirrors Phase K cell 4):
  ```
  python -m experiments.component_patching \
      --enriched-log logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
                     logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
                     logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz \
      --source-bucket clean_legal_fold \
      --target-bucket clean_check_or_call \
      --layer 14 \
      --components residual attn mlp head \
      --head-indices all \
      --n-source 10 --n-target 30 \
      --seed 42 \
      --out-dir results/causal_patching/llama8b_l14_components_reverse \
      --device cuda --dtype bfloat16
  ```
- **Headline reading**: identify top-3 negative-direction heads (the ones
  whose patch most pushes top-1 → FOLD-family). If they're {5, 23, 24}
  with sign flipped: same heads, bidirectional encoding. If they're a
  different set: separate CHECK-encoding and FOLD-encoding heads.

### Item 4 — Per-seed head consistency at Llama L=14

> **Why**: All B1/B1.5 results are pooled over s42 + s123 + s456. Pooling
> smears within-seed signal. We claim h5/h23/h24 are "the heads"; a
> defensive reviewer will ask "are those heads consistent across seeds,
> or is the pooled finding driven by one cell?" Run B1 separately on
> each seed at Llama L=14, compare the top-3 sets.

- **Three commands** (or wrap them in a small script — three sequential
  invocations of `experiments.component_patching` with one log each;
  ~50 min × 3 = 150 min total):
- **Pass criterion**: top-3 heads are the same {5, 23, 24} in all three
  per-seed runs (within ±1 head). If yes, robust per-seed. If different
  heads dominate per seed, the pooled finding is real but the per-seed
  story is messier — say so in the paper.

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
