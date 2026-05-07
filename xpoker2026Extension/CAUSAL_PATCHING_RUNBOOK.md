# Causal patching runbook

What was built locally, what needs to run on the GPU box, what to read when it
returns. This is the BlackboxNLP-grade follow-up to the §13 logit-lens
correlational finding.

## What this experiment tests

> §13 finding (correlational): `clean_CHECK_OR_CALL` decisions show
> late-layer (~27-29) revision from FOLD-leaning to CHECK; `illegal_fold`
> decisions stay 100% FOLD-committed throughout.
>
> §13 hypothesis (causal): the CHECK signal in the late layers of clean
> decisions is what *causes* the model to emit CHECK. If we patch a clean
> CHECK_OR_CALL decision's late-layer residual into an illegal_FOLD
> decision, the model should flip its predicted verb toward CHECK.

## Files added in this slice (all already in the repo)

### Core infrastructure (CPU writeable, GPU testable)

| File | Purpose | Where it runs |
|---|---|---|
| [`poker_env/interp/patching.py`](poker_env/interp/patching.py) | `HiddenStateCapture` + `HiddenStatePatch` forward hooks | imported by GPU code |
| [`poker_env/interp/forward_helpers.py`](poker_env/interp/forward_helpers.py) | Prompt reconstruction + single-forward helper + verb-position finder | mostly CPU-OK |
| [`experiments/causal_patching.py`](experiments/causal_patching.py) | Main CLI driver | **GPU** |
| [`experiments/verify_position_mapping.py`](experiments/verify_position_mapping.py) | Phase 1c CPU verification | CPU (needs network 1st time for tokenizer) |
| [`experiments/verify_prompt_reconstruction.py`](experiments/verify_prompt_reconstruction.py) | Phase 1b GPU verification | **GPU** |

### Driver scripts (GPU)

| File | Phase | Wall-clock |
|---|---|---|
| [`scripts/run_causal_patching_pilot.sh`](scripts/run_causal_patching_pilot.sh) | Phase 2 pilot (sparse layers 22-30) | ~50 min H100 |
| [`scripts/run_causal_patching_layer_sweep.sh`](scripts/run_causal_patching_layer_sweep.sh) | Phase 2.5 — fine-grained layer sweep | ~3 h H100 |
| [`scripts/run_causal_patching_full.sh`](scripts/run_causal_patching_full.sh) | Phase 3/4 | 30 min - 10 h depending on `SCOPE` env var |

## Local CPU validation (already done by me)

Everything that could be tested without a GPU has been:

- Prompt reconstruction (`PromptReconstructor`) reproduces the recorded
  `prompt_hash` exactly on 5/5 sampled records of Ministral s42 t=0.
- Action-verb char-offset finder works on FOLD / CHECK_OR_CALL / BET_OR_RAISE
  responses across Llama and Ministral.
- Bucketing logic (`classify_decision`) on Ministral s42 t=0 gives:
  `clean_CHECK_OR_CALL=29, clean_legal_fold=98, clean_bet_or_raise=7,
  illegal_fold=179, illegal_other=0, alias_unrecognized=0, json_failure=0`
  — matches the §13 numbers.
- All module imports clean, no lints.

## What you need to run on the GPU box

### Step 1: Pull the new code

```bash
ssh user@gpu-box
cd <repo>/xpoker2026Extension
git pull origin <branch>
```

The new files ride along with the existing logit-lens infrastructure — no
extra dependencies.

### Step 2: Pre-flight (REQUIRED before the pilot)

This is Phase 1b of the plan. **EXPERIMENT BLOCKED** if it doesn't pass 5/5.

```bash
python -m experiments.verify_prompt_reconstruction \
    --enriched-log logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    --n-samples 5 \
    --device cuda
```

Expected output: 5/5 top-1 matches. If <5/5 you'll see the offending records
with `[FAIL]` markers + top-5 alternatives — diagnose what changed before
proceeding (most likely: agent code was updated since the run, OR a different
tokenizer version was used).

If you want to also CPU-verify the position mapping (cheap, no GPU):

```bash
python -m experiments.verify_position_mapping \
    --enriched-log logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz \
    --n-samples 10
```

### Step 3: Pilot

```bash
bash scripts/run_causal_patching_pilot.sh
```

Defaults:
- Cell: Ministral s42 t=0 (where signal is strongest, 179 illegal_FOLDs)
- 10 sources × 30 targets × 5 layers (22, 24, 26, 28, 30) = 1,500 patched
  forwards
- Wall-clock: ~50 min on one H100 (unbatched)
- Plus ~5 min of pre-flights (controls + verifications)

Output:
```
results/causal_patching/ministral8b_t0_s42_pilot/
    summary.json     # machine-readable
    SUMMARY.md       # per-layer Δlogit(CHECK − FOLD) table
    by_pair.csv      # one row per (source, target, layer) with patched logits
```

### Step 4: Read the pilot result + decide

Open `SUMMARY.md`. The headline is the per-layer Δlogit table:

| Layer | n   | mean Δlogit(CHECK − FOLD) | top-1 flipped to CHECK-family |
|---:|---:|---:|---:|
| 22 | 300 | _baseline-ish_         | _low_  |
| 24 | 300 | _maybe rising_         | _low_  |
| 26 | 300 | _rising_               | _med_  |
| 28 | 300 | _peaking_              | _high_ |
| 30 | 300 | _saturating_           | _high_ |

Decision gate (from plan §C):

| Outcome | Criterion | Action |
|---|---|---|
| Strong success | mean_delta @ L=28 ≥ 3× value at L=22 AND L=28 abs > 1.0 nat | go Phase 3 (full-scope, scripts/run_causal_patching_full.sh) |
| Weak success | layer-specificity holds but L=28 abs < 1.0 nat | go Phase 3 with caveated framing |
| Null with confound | random-source control failed (mean abs > 0.5 nat) | halt, debug |
| Clean null | no layer specificity, controls fine | publish negative result for BlackboxNLP |

Also check:
- `controls.baseline_top1_match_rate` should be ≥ 0.95 (target verbs are
  reproduced from prompt+response_up_to_verb).
- `controls.self_patch_max_logit_drift` should be < 1e-2 (sanity for hooks).
- `controls.random_source_mean_delta` should be |x| < 0.5 nat (sanity for
  source-content specificity).

### Step 4b (added 2026-05-XX after first pilot saturated): layer sweep

The first pilot tested layers 22 24 26 28 30 and found 100% flip + ~+11 nat
shift AT EVERY LAYER (saturated). The verb-prediction info is already
encoded in the residual by L=22, so the sparse-late sweep can't pin the
deliberation depth.

To find where the patching effect EMERGES (= the deliberation circuit's
upper boundary), run a fine-grained sweep over every layer:

```bash
bash scripts/run_causal_patching_layer_sweep.sh
```

Defaults: every layer 0..35 of Ministral 8B, 10 sources × 15 targets, 5
random-source-per-layer null. ~5,400 main forwards + ~280 control = ~3 h on
H100. Output:

```
results/causal_patching/ministral8b_t0_s42_layer_sweep/
    SUMMARY.md       # per-layer table with main effect, random null,
                     # AND specificity-adjusted Δ (writeup-ready signal)
    summary.json     # controls.random_source_per_layer is the per-layer null
    by_pair.csv      # 5400 rows for plotting effect-vs-layer
```

What to look for in `SUMMARY.md` (read top-to-bottom):

| Layer | mean Δ | random null Δ | specificity-adjusted Δ | top-1 → CHECK |
|---:|---:|---:|---:|---:|
| 0   | _expect ~0_ | _~0_       | **~0**       | _low_ |
| 5   | _expect ~0_ | _~0_       | **~0**       | _low_ |
| ... | ...         | ...        | ...          | ... |
| L*  | rises       | ~0         | **rises**    | rises |
| ... | saturated   | ~0         | **~+11**     | 100% |

The layer L* where the **specificity-adjusted Δ** first crosses ~+1 nat is
the deliberation-circuit boundary. That's the BlackboxNLP plot.

If L* is, say, 18, then: "the model's verb decision crystallizes at layer
~18 in the residual stream; layers 18-22 host the deliberation circuit;
layers 22-35 are causally read-out." That's a publishable mechanistic claim.

Decision after the sweep:
- **L* well-defined (clean transition over 2-4 layers)**: Phase 3 with
  layers concentrated around L*, ~30 min on H100. Tight stats for paper.
- **L* unclear (gradual rise from L=0)**: indicates verb info is encoded
  very early (i.e., in the prompt processing itself, before CoT generation
  even begins). Different but still publishable claim.
- **No layer specificity even now**: hypothesis falsified at the residual-
  stream level; pivot to attention-pattern analysis.

### Step 5: Phase 3 (if pilot success gate passes)

```bash
SCOPE=subsampled bash scripts/run_causal_patching_full.sh
```

Subsampled = 29 sources × 30 targets × 7 layers = 6,090 forwards, ~30 min.
This is the recommended Phase-3 scope unless your pilot signal was small
enough that you need more pairs to reach significance.

For full-scope (29 × 179 × 7 = 36,337 forwards, ~10 h on H100):
```bash
SCOPE=full bash scripts/run_causal_patching_full.sh
```

For multi-model (Ministral + Llama) — Phase 4:
```bash
SCOPE=multi bash scripts/run_causal_patching_full.sh
```

### Step 6: Pull results back, write the BlackboxNLP paragraph

Once `results/causal_patching/<scope>/SUMMARY.md` is on the GPU box:

```bash
# on local machine
git pull origin <branch>
# read xpoker2026Extension/results/causal_patching/<scope>/SUMMARY.md
```

I'll then write a §14 in `updates.md` interpreting the per-layer effect
profile and append a new "Causal validation" section to
`results/tier1a_small_cot_logitlens/MECHANISTIC_FINDINGS.md` with the
single writeup-ready paragraph.

## Compute summary

| Step | Where | Time |
|---|---|---|
| Local build | done | ✓ |
| Pre-flights | GPU | ~5 min |
| Pilot | GPU | ~50 min |
| Phase 3 subsampled | GPU | ~30 min |
| Phase 3 full | GPU | ~10 h |
| Phase 4 multi | GPU | +~4-10 h |
| Writeup | local | ~1 h |

## Optimization (if you want to make full-scope cheaper)

The current implementation is unbatched (one forward per patched pair).
Batching K source-residuals into one target's forward at the patching layer
reduces forwards by K×. Stub:

```python
# Same target tokens repeated K times; source residuals stacked [K, hidden]
# Hook at layer L replaces residual[:, -1, :] = source_stack
```

A batched mode could land in `experiments/causal_patching.py` as
`--batch-size K` if the unbatched 30-min scope turns out to be the constant.
For pilot it's not worth the added complexity.
