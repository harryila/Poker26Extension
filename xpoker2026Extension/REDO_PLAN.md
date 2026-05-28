# Phase Q redo plan — layer / head set discipline

Companion to `AUDIT_FINDINGS.md`. After the post-`6fe23d4` audit, we want
every Phase Q cell to (a) use the principled layer for its question, and
(b) use a head set that actually carries the verb signal at that layer.
This file lists exactly what to rerun, what to keep, and what to mark
superseded.

---

## TL;DR — disciplined layer/head choices

| Cell type             | Llama  | Ministral | Qwen   | Notes |
|-----------------------|--------|-----------|--------|-------|
| **Patching (residual)**   | L=14 *and* L=15 | L=16 | L=23 | L=14 is Llama's transition layer; L=15 is saturation. We keep L=14 as canonical and add L=15 as parallel. |
| **Head ablation**         | **L=14, triplet `[5,23,24]`** | **L=16, sextet `[9,15,22,24,30,31]`** | L=23, triplet `[26,28,30]` (caveat: arbitrary) | Llama L=14 is the only layer/model with a true sparse head story (`heads_05_23_24` carry 65% / 49% top-1). Ministral's triplet is sub-circuit (3% top-1 flip) — switch to the sextet (37% top-1 flip). Qwen has **no** sparse head story at L=23 (residual flow-through carries 82%); the triplet is reported for cross-model surface comparison only. |
| **Mode-balanced probe**   | L=14 *(label_source=cot fallback)* | **drop — n=16, NaN** | L=23 | Cite Qwen as the clean cell; caveat Llama; do not cite Ministral. |

**Why Llama keeps two layers:** L=14 is where the verb-producing heads live
(sparse triplet structure). L=15+ residual carries the signal but the
heads at L=15 don't *produce* it — they just pass it through. Patching at
L=15 measures "is the verb already in the residual" (yes, 100%); patching
at L=14 measures "do these specific heads encode it" (yes, 79%, can be
boosted to 100% by adding more layers). Both are valid; reporting both
prevents post-hoc cherry-picking.

---

## P0 (must-do before writeup)

### P0.1 — Inference head ablation, recon pipeline, all 3 models

Replaces `results/inference_head_ablation/ministral8b_l16_cot/` (now marked
`SUPERSEDED.md`).

```bash
MODELS="ministral llama qwen" \
FORCE_RERUN=1 \
bash scripts/run_phase_q_audit_rerun.sh
```

Produces:

```
results/inference_head_ablation/{ministral,llama,qwen}8b_l*_recon_illegal_fold/
```

GPU: ~3 hours total (3 models × 80 records × 3 conditions × 3 forward generates ≈ 2000 generates).

**Cross-check:** `regenerate_ablated` flip rate in
`results/continuation_after_patch/<model>/SUMMARY.md` (already on `main`)
should agree within sampling noise:

| Model     | Continuation regen_ablated FOLD→CHECK | Inference recon (predicted) |
|-----------|--------------------------------------|-----------------------------|
| Ministral | 4/25 = 16% (net +12pp)               | should be ~12–20% over baseline |
| Llama     | 22/24 = 92% (with parse damage)      | should match (with parse-fail column) |
| Qwen      | 22/24 = 92% (net +58pp)              | should match |

If they don't agree → there's a residual prompt-pipeline divergence to
chase before submission.

### P0.2 — Ministral inference ablation with the SEXTET head set

The current canonical head set for Ministral L=16 is the triplet
`[22, 9, 15]`, which only flips **3% of patched targets to CHECK** in the
component decomposition (`results/causal_patching/ministral8b_l16_head_triplet/SUMMARY_components.md`,
also documented in `updates.md` §16). The sextet
`[9, 15, 22, 24, 30, 31]` flips **37%** (`results/causal_patching/ministral8b_l16_head_sextet/SUMMARY_components.md`).

So zeroing the triplet shouldn't have a strong necessity effect (we're
ablating a sub-circuit). The sextet is the proper Ministral analogue of
Llama's L=14 triplet — they're matched on "smallest head set that
demonstrably carries the verb signal at this layer".

After P0.1 lands, add the extended condition:

```bash
MODEL=ministral PIPELINE=recon \
  FILTER_RECORDED_BUCKET=illegal_fold \
  CONDITIONS="baseline triplet extended control" \
  FORCE_RERUN=1 \
  bash scripts/run_inference_head_ablation.sh
```

(Extended-set support is already wired into
`experiments/inference_head_ablation.py:DEFAULT_HEAD_SETS["ministral"]["extended"]`
and the `--conditions extended` flag.)

GPU: ~30 min for the extra condition.

### P0.3 — Llama L=15 patching parallel set

Patching cells (no head ablation) at the saturation layer:

```bash
FORCE_RERUN=1 bash scripts/run_phase_q_llama_l15_parallel.sh
```

Produces:

```
results/causal_patching/llama8b_reverse_fold_to_check_l15/
results/causal_patching/llama8b_bet_to_illegal_fold_l15/
results/context_stratified_patching/llama8b_l15_street/
```

GPU: ~1 hour.

**Reporting:** show L=14 *and* L=15 side-by-side in the Llama row of every
cross-model table. The story is "transition (L=14) → saturation (L=15)"
across the residual stream; the heads at L=14 produce the signal.

### P0.4 — Continuation rerun with verb-flip breakdown

The new `experiments/continuation_after_patch.py` adds:
- per-mode verb distribution (FOLD / CHECK_OR_CALL / BET_OR_RAISE / UNK)
- flip rate on recorded-FOLD targets with parse-fail column
- patch-flip rate on source×target pairs

The on-`main` SUMMARY.md files don't have this breakdown yet. Re-run to
overwrite:

```bash
MODELS="ministral llama qwen" \
CONTINUE_TOKENS=180 \
FORCE_RERUN=1 \
bash scripts/run_phase_q_audit_rerun.sh   # already includes step (2)
```

(P0.1 and P0.4 share the same orchestrator; one invocation does both.)

GPU: included in P0.1's 3 hours.

---

## P1 (should-do before writeup)

### P1.1 — Context-stratified by `pot_odds_quartile` on facing-bet pool

Answers the original equity-stratification question (the `street`
stratification answered a different one — see `AUDIT_FINDINGS.md` §5).

```bash
MODELS="ministral llama qwen" \
FORCE_RERUN=1 \
bash scripts/run_phase_q_audit_rerun.sh   # already includes step (3)
```

Produces `results/context_stratified_patching/<model>8b_l*_pot_odds/`.

GPU: ~30 min total.

### P1.2 — Qwen component decomposition at L=22

Phase L (`updates.md` §17g) established that Qwen's "computation" is
distributed across L=19–22, not at L=23. We have a component decomposition
at L=23 (saturation, 10% attn ratio) and at L=24 — but **not at L=22**, the
predicted compute layer. Filling that gap is one cell of `experiments/
component_patching.py`:

```bash
python -m experiments.component_patching \
  --enriched-log logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
                 logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz \
                 logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz \
  --layer 22 --source-bucket clean_check_or_call --target-bucket illegal_fold \
  --n-source 10 --n-target 30 --seed 42 \
  --out-dir results/causal_patching/qwen8b_l22_components \
  --device cuda --dtype bfloat16
```

(Pseudocode — verify the exact flag names against `experiments/component_patching.py`.)

If a sparse head set exists at Qwen L=22, it would replace `[26, 28, 30]`
as the canonical Qwen ablation set (with L=22, not L=23, as the matched
ablation layer).

GPU: ~20 min.

### P1.3 — Llama L=15 inference ablation (negative control only)

To make explicit that the heads at L=15 don't carry the signal, run a
single-condition ablation at Llama L=15 with the L=14 triplet (transferred
to L=15 layer-index) and confirm it has near-zero effect:

```bash
MODEL=llama LAYER=15 PIPELINE=recon \
  FILTER_RECORDED_BUCKET=illegal_fold \
  N_DECISIONS=50 \
  HEAD_SET_OVERRIDE="5 23 24" \
  FORCE_RERUN=1 \
  bash scripts/run_inference_head_ablation.sh
```

(Requires adding `--head-indices` CLI flag — currently the head set is
hardcoded by model defaults. ~10 lines of code; see TODO note in script.)

This is a P1 because it's a negative-control plot, not a headline result.
GPU: ~30 min.

---

## P2 (nice-to-have)

### P2.1 — Tier 4 at Llama L=15

Tier 4 has Llama L=14 already at the noise floor. Re-running at L=15 isn't
expected to help (Llama's opp-invariance failure is data-distribution, not
layer choice). Skip unless reviewer asks.

### P2.2 — Layer-level (whole-attention) ablation for Qwen

Qwen has no sparse head story → cleanest "necessity" test would be to
zero the entire L=23 (or L=22) attention block. Currently
`AttnHeadZeroAblation` zeros a head subset; would need a small extension
to support `head_indices=range(num_heads)` or a dedicated
`AttnBlockZeroAblation`. ~30 lines.

GPU: ~30 min.

---

## What to keep / archive / delete

### Keep as-is (data is correct)

- `results/causal_patching/{model}8b_*` — all the layer sweeps,
  reverse/BET/Tier 4/component decompositions. These are correct.
- `results/causal_patching/{ministral,llama,qwen}8b_l*_components/` and
  `*_head_triplet/`, `*_head_quartet/`, `*_head_sextet/` — head discovery.
- `results/causal_patching/llama8b_reverse_fold_to_check_l14/`,
  `llama8b_bet_to_illegal_fold_l14/`, etc. (commit `6fe23d4`) — keep; add
  L=15 parallels via P0.3.
- `results/continuation_after_patch/{model}8b_l*/` — re-run via P0.4 to
  add the new SUMMARY breakdown; the JSONL data is correct.
- `results/mode_balanced_probe/{llama,qwen}8b_l*/` — keep.
- `results/context_stratified_patching/{model}8b_l*/` — keep `_street`;
  add `_pot_odds` via P1.1.

### Marked SUPERSEDED.md (keep on disk, do not cite)

- `results/inference_head_ablation/ministral8b_l16_cot/` — superseded by
  P0.1 recon-pipeline rerun. Annotated.
- `results/mode_balanced_probe/ministral8b_l16/` — n=16, NaN. Annotated.

### Do NOT delete

We deliberately keep the superseded results on disk so:
1. Reviewers can verify the pipeline change wasn't post-hoc cherry-picking.
2. The regen-fidelity bug (Ministral 5% baseline parse_OK) is itself
   evidence of a real phenomenon worth one sentence in the limitations
   section.
3. The 16-pair Ministral mode-balanced cell is evidence of Ministral's
   non-CoT bucket-skew, which corroborates the consolidation gradient.

---

## GPU-hour budget

| Item    | GPU hours | Status |
|---------|-----------|--------|
| P0.1 inference recon (3 models)            | 3.0   | required |
| P0.2 Ministral sextet condition            | 0.5   | required |
| P0.3 Llama L=15 patching parallel set      | 1.0   | required |
| P0.4 continuation overwrite (3 models)     | 1.5 (overlaps P0.1) | required |
| P1.1 pot_odds_quartile (3 models)          | 0.5   | recommended |
| P1.2 Qwen L=22 components                  | 0.5   | recommended |
| P1.3 Llama L=15 negative-control ablation  | 0.5   | optional |
| P2.* (Tier 4 L=15, Qwen layer ablation)    | 1.0   | optional |
| **Total P0**                               | **4.5–5.0** | |
| **Total P0+P1**                            | **6.0–6.5** | |
| **Total all**                              | **~7.5** | |

---

## One-shot GPU runner

After disk increase:

```bash
cd xpoker2026Extension
git pull origin main
export HF_HOME=/workspace/huggingface HF_TOKEN=...

# P0.1 + P0.4 + P1.1 (continuation + inference + pot_odds)
FORCE_RERUN=1 CONTINUE_TOKENS=180 bash scripts/run_phase_q_audit_rerun.sh

# P0.3 (Llama L=15 patching parallel set)
FORCE_RERUN=1 bash scripts/run_phase_q_llama_l15_parallel.sh

# P0.2 (Ministral sextet)
MODEL=ministral PIPELINE=recon \
  FILTER_RECORDED_BUCKET=illegal_fold \
  bash scripts/run_inference_head_ablation.sh \
  --conditions baseline triplet extended control
```

After all three complete, re-read every `SUMMARY.md` and update the
cross-model claims tables in `AUDIT_FINDINGS.md` and `PHASE_Q.md`.

---

## What this changes about the writeup

1. **No more "L\* across all 3 models" elision.** Llama's L\* has TWO
   useful interpretations (L=14 compute; L=15 commit) and we report both.
2. **Head ablation results are model-specific in design**, not just in
   outcome. Ministral uses a sextet because its triplet is sub-circuit;
   Qwen's "triplet" is a surface-comparison artifact, not a circuit.
3. **The consolidation-gradient claim survives** but is now more nuanced:
   "the smallest head set that crosses the verb-flip threshold scales
   with model" — Llama 3 heads, Ministral 6 heads, Qwen >32 heads /
   distributed across multiple layers. That's a clean finding.
