# Runbook — rejection-proofing + novelty follow-ups (GPU box)

Everything here was implemented and CPU-validated locally (syntax + synthetic self-tests).
CPU analyses already ran and committed their reports; the GPU cells below produce the
artifacts those analyses and the writeup consume. Canonical layers: **Llama L14 (heads
[5,23,24]), Ministral L16, Qwen L23 (sufficiency) / L19 (necessity, heads [31,3,21,1,0])**.

Setup on the box:
```bash
cd xpoker2026Extension && git pull
export HF_HOME=/workspace/huggingface HF_TOKEN=...
```

---

## DONE already (CPU, committed — no GPU)
| Artifact | What it shows |
|---|---|
| `results/direction_probe_baselines/CONFOUND_PROJECTION.md` | decision direction is verb-aligned (cos +0.95–0.99) not bet-aligned → partial rebuttal of the crux |
| `results/causal_patching/qwen8b_t0_pooled_layer_sweep/SUFFICIENCY_CI.md` | target-clustered CIs; spec-adj range [+15.6,+25.1] not the point +18.3 |
| `results/inference_head_ablation/REGEN_DRIFT.md` | which necessity cells are reliable (Qwen clean_legal_fold 15% drift) vs not (Llama 73%) |
| `CLAIMS_AND_IDENTIFICATION.md` | reframed claims + identification-assumptions section (per arXiv 2605.08012) |

Re-run any: `python -m experiments.confound_projection_analysis` / `... .sufficiency_ci ...` / `... .regen_drift_audit ...`

---

## PHASE B — Tier 0 (rejection-proofing). ~3–4 GPU-h.

### B1 — Bet-matched probe (closes the crux). Per model. ~20 min each.
Recapture residuals tagged with bet_to_call/verb (+oracle/belief for C1), then the CPU probe.
```bash
# QWEN (repeat for llama L14, ministral L16)
python -m experiments.bet_matched_recapture \
  --enriched-log logs/cot_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz \
                 logs/cot_qwen8b_t0_s123_informative_v2_enriched.jsonl.gz \
                 logs/cot_qwen8b_t0_s456_informative_v2_enriched.jsonl.gz \
  --layer 23 --device cuda --dtype bfloat16 \
  --out results/direction_probe/qwen8b_l23/raw_residuals_tagged.npz
# CPU (can run on the box or locally):
python -m experiments.bet_matched_probe \
  --tagged results/direction_probe/qwen8b_l23/raw_residuals_tagged.npz \
  --out results/direction_probe_baselines/BET_MATCHED_qwen_l23.md
```
PASS = probes A (CALL@bet>0 vs legal_fold) and B (CHECK@bet=0 vs illegal_fold) and C
(bet-balanced) stay far above the permuted floor. (llama: `--layer 14`; ministral: `--layer 16`.)

### B2 — Bet-matched PATCHING (causal, bet held constant). Per model. ~20 min each.
```bash
# facing-a-bet regime: CALL@bet>0 source -> clean_legal_fold@bet>0 target (Qwen L23)
python -m experiments.causal_patching \
  --enriched-log logs/cot_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz \
                 logs/cot_qwen8b_t0_s123_informative_v2_enriched.jsonl.gz \
                 logs/cot_qwen8b_t0_s456_informative_v2_enriched.jsonl.gz \
  --source-bucket clean_check_or_call --source-bet-filter facing \
  --target-bucket clean_legal_fold   --target-bet-filter facing \
  --layers 23 --n-source 10 --n-target 30 --seed 42 \
  --device cuda --dtype bfloat16 \
  --out-dir results/causal_patching/qwen8b_betmatched_facing_l23
```
PASS = top-1→FOLD-side still flips under the CALL source with bet constant (sufficiency
survives bet-matching). Mirror with `--source-bet-filter nobet --target-bucket illegal_fold
--target-bet-filter nobet` if desired. (llama `--layers 14 15`; ministral `--layers 16`.)

### B3 — Ministral 3-seed (it was effectively single-seed). ~40 min.
Sufficiency, pooled 3 seeds (mirrors the qwen pooled cell):
```bash
python -m experiments.causal_patching \
  --enriched-log logs/cot_ministral8b_t0_s42_informative_v2_enriched.jsonl.gz \
                 logs/cot_ministral8b_t0_s123_informative_v2_enriched.jsonl.gz \
                 logs/cot_ministral8b_t0_s456_informative_v2_enriched.jsonl.gz \
  --source-bucket clean_check_or_call --target-bucket illegal_fold \
  --layers 16 --n-source 10 --n-target 30 --seed 42 \
  --device cuda --dtype bfloat16 \
  --out-dir results/causal_patching/ministral8b_t0_pooled_layer_sweep
```
Necessity, pooled 3 seeds (the §9 null should reproduce with balanced seeds):
```bash
python -m experiments.inference_head_ablation \
  --enriched-log logs/cot_ministral8b_t0_s42_informative_v2_enriched.jsonl.gz \
                 logs/cot_ministral8b_t0_s123_informative_v2_enriched.jsonl.gz \
                 logs/cot_ministral8b_t0_s456_informative_v2_enriched.jsonl.gz \
  --layer 16 --pipeline recon --filter-recorded-bucket illegal_fold \
  --conditions baseline triplet control extended \
  --device cuda --dtype bfloat16 \
  --out-dir results/inference_head_ablation/ministral8b_l16_recon_illegal_fold_3seed
# then re-check drift balance + significance (CPU):
python -m experiments.regen_drift_audit --glob 'results/inference_head_ablation/ministral8b_l16_recon_illegal_fold_3seed' --out /tmp/min3seed_drift.md
python -m experiments.necessity_significance --glob 'results/inference_head_ablation/ministral8b_l16_recon_illegal_fold_3seed' --out results/inference_head_ablation/SIGNIFICANCE_ministral_l16_3seed.md
```

### B4 — Qwen L19 same-depth random-head control (is L19 special vs ANY L19 heads?). ~30 min.
```bash
python -m experiments.inference_head_ablation \
  --enriched-log logs/cot_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz \
                 logs/cot_qwen8b_t0_s123_informative_v2_enriched.jsonl.gz \
                 logs/cot_qwen8b_t0_s456_informative_v2_enriched.jsonl.gz \
  --layer 19 --pipeline recon --filter-recorded-bucket clean_legal_fold \
  --conditions baseline \
  --head-sets 'top5:31 3 21 1 0' 'rand5a:rand:5:101' 'rand5b:rand:5:202' 'rand5c:rand:5:303' \
  --device cuda --dtype bfloat16 \
  --out-dir results/inference_head_ablation/qwen8b_l19_samedepth_control
# CPU: compare top5 vs the random draws
python -m experiments.necessity_significance \
  --glob 'results/inference_head_ablation/qwen8b_l19_samedepth_control' \
  --within-cell-control rand5a \
  --out results/inference_head_ablation/SIGNIFICANCE_qwen_l19_samedepth.md
```
PASS = `top5` flips significantly MORE than the random-5 draws (necessity is L19-head-specific,
not just "remove any 5 heads at a deep layer").

---

## PHASE C — Tier 1 (novelty). Depends on B1/B2 confirming the direction is real. ~3–4 GPU-h.

### C1 — Encode-vs-decode ("knows but mis-states"). Reuses B1's tagged npz (has oracle+belief). CPU after recapture.
```bash
python -m experiments.encode_vs_decode \
  --tagged results/direction_probe/qwen8b_l23/raw_residuals_tagged.npz \
  --out results/direction_probe_baselines/ENCODE_VS_DECODE_qwen_l23.md \
  --save-direction results/direction_probe/qwen8b_l23/steer_trash_direction.npz
```
HEADLINE = high oracle-posterior decodability R² (esp. trash mass) + high stated-belief JS ⇒
the agent ENCODES the Bayesian posterior it fails to STATE. Saves the steering direction for C2.

### C2 — Posterior steering (de-bias the decision). ~30 min.
```bash
python -m experiments.posterior_steering \
  --enriched-log logs/cot_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz \
                 logs/cot_qwen8b_t0_s123_informative_v2_enriched.jsonl.gz \
                 logs/cot_qwen8b_t0_s456_informative_v2_enriched.jsonl.gz \
  --layer 23 --direction results/direction_probe/qwen8b_l23/steer_trash_direction.npz \
  --alphas 0 2 4 8 --target-bucket clean_legal_fold --n-decisions 60 \
  --device cuda --dtype bfloat16 \
  --out-dir results/posterior_steering/qwen8b_l23
```
HEADLINE = a monotone CHECK−FOLD shift with alpha for the trash direction that EXCEEDS the
built-in random-direction control = a controllable, task-grounded de-biasing steer.

### C3 — Behavior at scale (circuit → win-rate). Reuses run_experiment via env vars. ~1–2 GPU-h per condition.
```bash
# baseline win-rate
python run_experiment.py --agent hf --hf-model qwen-8b --opponent threshold \
  --opponent-preset informative_v2 --hands 300 --seed 42 --elicit-beliefs \
  --out logs/scale_qwen_baseline.jsonl -v
# necessity at scale: ablate the L19 fold heads during play
CIRCUIT_ABLATE_LAYER=19 CIRCUIT_ABLATE_HEADS="31 3 21 1 0" \
python run_experiment.py --agent hf --hf-model qwen-8b --opponent threshold \
  --opponent-preset informative_v2 --hands 300 --seed 42 --elicit-beliefs \
  --out logs/scale_qwen_ablate_l19.jsonl -v
# de-bias at scale: steer toward the calibration direction during play
CIRCUIT_STEER_LAYER=23 CIRCUIT_STEER_NPZ=results/direction_probe/qwen8b_l23/steer_trash_direction.npz \
CIRCUIT_STEER_ALPHA=4 \
python run_experiment.py --agent hf --hf-model qwen-8b --opponent threshold \
  --opponent-preset informative_v2 --hands 300 --seed 42 --elicit-beliefs \
  --out logs/scale_qwen_steer.jsonl -v
```
Compare the win-rate / net-chips the driver prints across the three runs (and re-enrich +
compute_pce_distribution for calibration change). HEADLINE = the circuit intervention moves
real gameplay outcomes, not just verb logits.

---

## PHASE D — Tier 2 (depth, if reviewers ask). ~1 GPU-h.

### D1 — Qwen distributed-band SVD (distributed across heads but low-rank?).
```bash
python -m experiments.qwen_band_svd \
  --enriched-log logs/cot_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz \
                 logs/cot_qwen8b_t0_s123_informative_v2_enriched.jsonl.gz \
                 logs/cot_qwen8b_t0_s456_informative_v2_enriched.jsonl.gz \
  --layers 18 19 20 23 --bucket clean_check_or_call --n 150 \
  --direction results/direction_probe/qwen8b_l23/raw_residuals.npz \
  --device cuda --dtype bfloat16 --out results/causal_patching/qwen_band_svd.md
```

### D2 — 4th model family (real-axis vs 3-point check). Last; depends on pipeline stable.
Run the whole pipeline (`scripts/run_phase_q_all.sh`-style) on a 4th 8B (e.g. gemma-2-9b or
phi). If it slots onto the consolidation gradient, the gradient is an axis; if not, it's 3
idiosyncratic points — either way a stronger claim than now.

---

## After everything lands
- Re-finalize `CLAIMS_AND_IDENTIFICATION.md` (flip the ⏳ items to ✅/🟡 with the new numbers).
- Update the `AUDIT_FINDINGS.md` cross-model tables with bet-matched + Ministral-3-seed + B4 numbers.

---

## V2 BATCH — controls + reworks (after the v1 results came back)

The v1 results closed the bet confound (B1/B2 ✓) but flagged three things needing a second
batch (see `CLAIMS_AND_IDENTIFICATION.md` "RESULTS LANDED"). One script runs them in order:

```bash
cd xpoker2026Extension && git pull
export HF_HOME=/workspace/huggingface HF_TOKEN=...
bash scripts/run_followups_gpu_v2.sh            # all (~4–6 GPU-h)
PHASES="C1" bash scripts/run_followups_gpu_v2.sh  # just the make-or-break C1 control
```

- **C1-CTRL** (make-or-break for the encode-vs-decode headline): recapture at **L2**, decode the
  oracle, and compare to L*. `ENCODE_LAYER_COMPARE.md` verdict: COMPUTED (late ≫ early → the model
  builds the posterior, "knows-but-mis-states" holds) vs INPUT-PRESENCE (early ≈ late → downgrade
  to "linearly available + discarded by readout"). **Pull this back first.**
- **C2-DEC** steering decision readout reworked: L19 (compute) + L23 (commit), **±alpha**, on
  **illegal_fold (wrong folds)** + clean_legal_fold, vs the built-in random control. (v1 C2 was
  L23/clean_legal_fold only → null.)
- **C2-BEL** steering belief readout: does steering **reduce JS(belief, oracle)** (repair the
  readout)? Reuses `HFAgent.belief` under the hook; reports parse-rate to catch over-steer.
- **C3** behavior-at-scale **rerun with `--cot`** (v1 ran non-CoT → degenerate constant-loss agent;
  the env itself is correct) + safe steering (`CIRCUIT_STEER_LASTONLY=1`, alpha 2).

Pull back `results/direction_probe_baselines/ENCODE_*`, `results/posterior_steering/*`,
`logs/scale_qwen_cot_*.jsonl` and we post-process: read `ENCODE_LAYER_COMPARE.md` first (it sets
whether the headline is "knows" or the weaker claim), then the steering SUMMARYs (trash vs control),
then diff the three CoT win-rates.
