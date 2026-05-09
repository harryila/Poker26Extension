# Causal-patching experiments — queued + shelved

Living document. Captures (i) the experiments we're actively queuing for the
GPU box now, (ii) the shelf of experiments we considered and deliberately
deferred, with the trigger condition that would move each shelf item into the
active queue. Sister document to `JOURNEY.md`, which records what *has* been
run; this records what *will* be run, *might* be run, and *won't* be run, and
why.

> **Last refresh:** May 2026, after the Llama per-seed replication closed the
> cross-seed Llama story. Forward + reverse pilots, Llama+Ministral per-seed
> replication, and a methods-ready relaxed pre-flight gate
> (`logs/preflight_relaxed_gate.txt`) are all complete.

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

## Queued — to run on the GPU box, in this priority order

The five items below should be runnable end-to-end with one `git pull` on the
GPU box. Each line shows the script, the expected wall-clock on a single
H100, and the headline reading the resulting SUMMARY.md should produce.

| # | Code | Script | Wall-clock | Headline reading |
|---|------|--------|-----------:|------------------|
| 1 | A1 | `scripts/run_causal_patching_qwen_seeds_replicate.sh` | ~2 h | Per-seed Qwen sweeps; expected: gradual ramp L=19→23, top-1 → CHECK climbs from <2% to 100%, **same shape across all three seeds**. |
| 2 | D2 | `scripts/run_causal_patching_zero_ablation.sh` | ~30 min | Zero-ablation control at L*=14 in Llama and L*=14 in Ministral; if zero-patch flips the verb, the layer is *load-bearing*; if not, the existing result is *content-addressable* rather than just "a clean signal at this layer matters". |
| 3 | A3 | `scripts/run_causal_patching_nocot_parity.sh` | ~3-6 h | Same protocol on non-CoT (`scaled_*_enriched.jsonl`) baseline logs in all three models; expected: L*=14 boundary persists in Llama+Ministral; uncertain for Qwen. |
| 4 | C1 | `scripts/run_causal_patching_verb_generality.sh` | ~3-4 h | RAISE source ↔ CHECK target patching at L*=14 (and a small layer sweep around it); tests whether L*=14 is a *fold-or-not* or *general decision* circuit. Read `top-1 → BET_RAISE-family` column. |
| 5 | B1 | `scripts/run_causal_patching_component_l14.sh` | ~6-12 h | Llama L=14 broken into attention-only / MLP-only / per-attention-head patches; if attn-only ≈ full-residual *and* a small set of heads dominates, that's the headline mech-interp result. |

### A1 — Qwen per-seed replication

> **Why:** Cross-model symmetry. Llama and Ministral both have 4 cells of
> per-seed evidence; Qwen has only the 3-seed pooled sweep. The forward
> pooled sweep already shows Qwen has a qualitatively different signature
> (gradual L=19→23 ramp vs Llama/Ministral's sharp 2-layer flip at L=14),
> but the per-seed cells confirm that the *gradual* shape itself is
> reproducible across seeds and not a pooling artifact. This is what lets
> us write the cross-model story symmetrically.

- **Inputs:** `logs/cot_qwen8b_t0_{s42,s123,s456}_informative_v2_logitlens_enriched.jsonl.gz`
- **Layers:** all 36 (matches Ministral replicate; Qwen has 36 layers)
- **n_source / n_target:** 10 / max-available
- **Outputs:** `results/causal_patching/qwen8b_t0_{s42,s123,s456}_replicate/`
- **Pass criterion:** all three SUMMARY.md tables show top-1 → CHECK climbing
  from <2% at L≤19 to ≥75% at L=22 to 100% at L=23, in roughly the same
  shape, with specificity-adjusted Δ rising monotonically.

### D2 — Zero-ablation control

> **Why:** Distinguishes two interpretations of the existing result.
> "Patching a clean source residual into the target flips the verb" is
> consistent with both *(a) the layer-L state matters causally* and *(b) any
> well-formed signal at layer L flips the verb*. Zero-patch (replace the
> residual with the all-zeros tensor) tests (a) directly: if zero flips the
> verb, the circuit *uses* layer L; if zero does *not* flip the verb but
> clean-source patches do, the circuit is content-addressable, not just
> layer-load-bearing.

- **Inputs:** Llama and Ministral pooled enriched logs
- **Layers:** L*-2, L*-1, L*, L*+1, L*+4 (5 layers each model)
- **Mode:** `--zero-ablation` flag added to `experiments/causal_patching.py`
  (replaces the source residual with zeros instead of a sampled source)
- **Outputs:** `results/causal_patching/{llama,ministral}_zero_ablation/`
- **Pass criterion (for "circuit is content-addressable"):** zero-ablation
  Δlogit(CHECK − FOLD) at L=14 should be much smaller than the +10 to +14
  nat shift from the clean-source patch. If it's similar, the layer is
  load-bearing in a directional sense; if it's near 0, the circuit is fully
  content-addressable.

### A3 — Non-CoT patching parity

> **Why:** All current causal patching is on CoT runs. Question: is L*=14
> the model's *natural* decision circuit, or is it specifically the
> deliberation circuit that the CoT prompt induces? If L*=14 boundary
> persists in non-CoT runs, the circuit is the model's, not the prompt's.
> If L* shifts or vanishes, that's a finding about what CoT actually does
> mechanistically.

- **Inputs:** `logs/scaled_<model>8b_t0_s42_informative_v2_enriched.jsonl`
  (one per model — Tier-1A.small produced these)
- **Layers:** small sweep around the CoT-derived L* (Llama: 10-18,
  Ministral: 12-20, Qwen: 16-24)
- **Outputs:** `results/causal_patching/{llama,ministral,qwen}8b_nocot_parity/`
- **Caveat:** non-CoT runs may have far fewer `illegal_fold` decisions than
  CoT runs (no chain-of-thought to "miss" with). Script auto-detects
  available targets and skips cells with <3 illegal_FOLDs. If a model has
  insufficient targets, this just prints `[skip]` for that cell and the
  rest still run.
- **Pass criterion:** if L*=14 boundary persists in Llama/Ministral non-CoT,
  banner result. If it shifts, banner result of a different kind. There is
  no failure mode for this experiment given the controls run first.

### C1 — Verb-generality (RAISE included)

> **Why:** Currently CHECK_CALL ↔ FOLD is the only binary tested. If L*=14
> is a *general* decision circuit, it should also mediate RAISE source ↔
> CHECK target swaps. If L*=14 is *fold-or-not specific*, RAISE patches
> won't flip the verb at the same layer.

- **Inputs:** Same pooled enriched logs as the original layer sweep
  (`cot_<model>8b_t0_s{42,123,456}_*_enriched.jsonl.gz`)
- **Source bucket:** `clean_bet_or_raise`
- **Target bucket:** `clean_check_or_call`
- **Layers:** {L*−2, L*−1, L*, L*+2, L*+5} per model
- **SUMMARY.md change:** the existing SUMMARY only prints CHECK and FOLD
  fractions. For verb-generality, the headline column is *top-1 →
  BET_RAISE-family*. Driver tweaked to always print all three group
  columns — backward-compatible, prior runs just had 0% in that column.
- **Outputs:**
  `results/causal_patching/{llama,ministral,qwen}8b_verb_generality_raise_to_check/`
- **Pass criterion (for "general decision circuit"):** at L=L*, top-1 →
  BET_RAISE rises to >70% at saturation; at L<L*, no flip. Same boundary
  layer as the CHECK/FOLD experiment.

### B1 — Component-level patching at L*=14 (Llama first)

> **Why:** Standard mech-interp depth-of-detail. The current result says
> "patching the residual at L=14 flips the verb." The next-resolution
> question is "*which sublayer at L=14 carries the signal?*" — attention or
> MLP — and within attention, "*which heads?*" If a small set of heads
> dominates, that's the strongest version of the result.

- **Implementation:** new patching primitives in
  `poker_env/interp/patching.py`:
    - `HiddenStatePatchAttnOnly` (replaces only `self_attn` output last position)
    - `HiddenStatePatchMLPOnly` (replaces only `mlp` output last position)
    - `HiddenStatePatchAttnHeadSubset` (replaces specific head slices in the
      pre-`o_proj` per-head concat)
  Plus capture analogs: `HiddenStateCaptureAttnAndMLP` and
  `HiddenStateCaptureAttnHeads`.
- **Driver:** `--component {residual,attn,mlp,head}` flag and `--head-indices`
  list.
- **Cells (Llama, L=14):**
    - attn-only sweep over L=12-16
    - MLP-only sweep over L=12-16
    - per-head sweep at L=14 (32 heads, one head at a time)
- **Outputs:** `results/causal_patching/llama8b_l14_component_<mode>/`
  (one per mode + a top-level `SUMMARY_components.md` aggregating)
- **Pass criterion (for "head-localized circuit"):** attn-only effect ≈
  full-residual effect (≥80% of the magnitude); MLP-only effect <20% of
  the magnitude; per-head sweep produces a sparse profile (a small
  number of heads each contributing >10% of the full attention effect).
- **Stretch — Ministral:** if Llama shows a sparse head profile, repeat
  the per-head sweep on Ministral L=14 to test cross-model head
  consistency. (~2 h additional.)

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
