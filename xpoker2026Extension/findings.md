# Findings — skeptical audit of xpoker2026Extension

## Resolved non-issues
- **Date "2026-05-29" in docs** = UTC timezone, not fabrication. Local is PDT (-07).
  `git log` confirms commits 2b88d27/2119466/853e662 are from tonight (2026-05-28 PDT).
  Repo is ACTIVELY being worked on; working tree has uncommitted changes
  (AUDIT_FINDINGS.md, PHASE_Q.md, inference_head_ablation.py modified; new
  necessity_significance.py + 4 SIGNIFICANCE_*.md untracked).

## Open questions to verify
- [ ] Do SUMMARY.md headline numbers match the committed JSONL when re-aggregated?
- [ ] Per-seed (42/123/456) consistency — is pooling hiding a divergent seed?
- [ ] necessity_significance.py McNemar p-values reproduce? Is L8 a fair control?
- [ ] Is the L=14(Llama)/L=23(Qwen)/L=16(Ministral) layer choice post-hoc?
- [ ] Multiple-comparisons exposure: how many (layer × head-set × pool) cells tested
      before the "L19 significant" claim?
- [ ] Tier-4 distinct-seed regen: do the _distinctseed result dirs actually exist
      and match the AUDIT_FINDINGS table?
- [ ] Oracle/probe correctness (CardOnly / StrategyAware / mode-balanced probe).

## Verified claims (confirmed against data/code)
1. **necessity_significance.py reproduces EXACTLY** — re-ran Qwen clean_legal_fold
   glob; output byte-matches committed SIGNIFICANCE_qwen_clean_legal_fold.md.
   L19 whole-attn net +42.7pp, McNemar vs L8 control p=3.76e-05 ✓; L19 top-5
   p=1.02e-03 ✓; L23 saturation p=0.36 NS. McNemar exact impl is correct.
2. **3-way base-seed consistency (Qwen necessity) HOLDS.** "seed" field is a
   COMPOSITE (hand_offset*1000 + base_seed); base = seed%1000 ∈ {42,123,456},
   ~50 records each. L19 net per seed = +53.1/+40.0/+35.3pp (always peak);
   L19>L23 in all 3 seeds. Seed 123 noisiest (L18 ties L19, L23 elevated) but
   qualitative ordering robust → not a pooling artifact.

## Methodological notes / things to keep scrutinizing
- **L8 as the only "early control":** whole-attn ablation at L8 already gives
  +20.7pp (big generic "remove-attention→fold-less" floor). L19-vs-L8 conflates
  "L19 specifically" with "deeper layers matter more." Mitigant: L20/L23 are NOT
  elevated (non-monotonic in depth) + L19 top-5-head subset also beats control.
  Still, a same-depth or random-head control at L19 would be cleaner. FLAG.
- Couldn't retrieve OpenReview reviews (API returned 0 notes). Base-paper context
  is from README "Prior Results (Llama-70B N=1,084)" only. May ask user.

## More verified (significance backbone — ALL 4 reproduce)
3. **All 4 SIGNIFICANCE_*.md reproduce exactly** (scoped-glob data rows byte-identical):
   - Qwen illegal_fold n=24: EXACT MATCH (no cell sig vs control — small-n honest).
   - Llama L14: triplet ✓ sig necessary p=2.1e-5 vs head-control; L15 negctrl
     triplet +1.5pp NS (reproduced when I widened glob) → clean compute/saturation split.
   - Ministral L16: triplet & sextet ✗ REVERSED (control heads flip MORE, p=8.2e-4)
     → genuine NULL, no sparse circuit. Honest.
   => Necessity story (Llama sparse-L14 / Qwen distributed-L19 / Ministral null)
      is fully supported by reproducible McNemar stats. Not asserted — derived.

## Agent A — patching/ablation code audit (RECEIVED)
SUFFICIENCY IS SOUND:
- top-1→CHECK measured at the position that PREDICTS the verb; input truncated
  strictly before verb's 1st char (forward_helpers.py:301-302, 320-335). No leak.
- Patch = source residual at its own verb-predecessor pos → target verb-predecessor
  pos, one mid layer; verb token never present in either seq (patching.py:120-191).
  Genuinely "before commit." No trivial source-answer copy.
- AttnHeadZeroAblation zeros o_proj INPUT per-head slice (post-attn,pre-proj) — correct
  cut point; zeros last position each generate step.
CONCERNS (to verify myself):
- [A5 — MUST CHECK] `--source-residual-top1`/`--target-residual-top1` DROP items whose
  residual top-1 already disagrees → post-hoc selecting items that agree w/ hypothesis;
  could inflate "100%". Need to confirm whether HEADLINE sufficiency cells used these.
- [A1] spec-adj null averages n_random sources vs a SINGLE target → small spec-adj
  (Ministral Tier4 +0.16–0.70) untrustworthy; only large (Qwen +18–20) survive.
- [A6] self_patch tolerance relaxed 1e-4→1e-2 (causal_patching.py:26-28,640) to absorb
  bf16 nondeterminism. Necessity rests on T=0 generate() regen; Llama baseline regen
  drifts 56% (docs admit "unmeasurable"). => necessity only trustworthy as control-
  paired net deltas on LOW-baseline-drift cells (Qwen). Sufficiency unaffected.
- [A7] continuation runs UNPATCHED greedy decode (no KV cache) after a single patched
  verb forward (continuation_after_patch.py:102-119,316-325). "Coherent continuation
  after patch" partly = model re-reading its own non-patched prefix. Caveat claim #6.
- BIGGEST THREAT per A: necessity confounded by nondeterministic T=0 regen; sufficiency clean.

## SUFFICIENCY verified myself (Qwen forward, t0 pooled + 3 replicates)
- Headline cell qwen8b_t0_pooled_layer_sweep: UNFILTERED (no residual-top1). A5
  concern does NOT touch the headline; _filtered cells are separate nocot circuit-hunt.
- Clean monotonic saturation curve, n=240 pairs: top-1→CHECK 0→1.7→27.9→55.4→76.2→
  100%(L23); spec-adj +0.28→+18.3(L23). Beautiful.
- **3-seed sufficiency: top-1→CHECK=100% at L23 in ALL 3 seeds** (n=40/90/110).
  Categorical sufficiency ROBUST across replications. ✓✓
- BUT spec-adj magnitude noisy: +25.0/+15.6/+24.9 (seed42/123/456). seed123 null=16.70
  (others ~3.4) → its 9 targets sit near boundary. Pooled +18.3 is dragged down by
  seed123; report a CI/spread, not a point. [FLAG for writeup]
- **Pseudoreplication**: independent unit = target decision (24 pooled; 4/9/11 per seed),
  NOT 240 pairs. illegal_fold is rare. "n=240" overstates precision. [FLAG]

## Suspected issues (open)
- [FLAG-writeup] report per-seed spread/CI for spec-adj, not pooled point estimate.
- [FLAG-stats] pseudoreplication: cluster by target decision; effective n ~24.
- L8-only control for Qwen necessity (single early control).
- A1 spec-adj null uses single target → small-effect cells (Ministral) untrustworthy.
- A6/A7 necessity & continuation confounded by T=0 regen drift (sufficiency clean).
## Agent C — results consistency (RECEIVED): everything matches disk
- Sufficiency, reverse, BET, Tier-4 distinctseed (all 10 dirs), Qwen compute sweep
  (attn 92/78/39/1/8%), SUPERSEDED.md hygiene — ALL MATCH the AUDIT tables exactly.
- Qwen L19 seed split 42:69/123:50/456:55% (matches my check); Llama L14 triplet
  42:94/123:97/456:100% — balanced & consistent.
- **NEW [FLAG-seed]: Ministral cells are effectively SINGLE-SEED.** Headline Ministral
  sufficiency = single-seed s42 (n=15 targets). §9 necessity null pool = 78/80 seed-42
  (seeds 123/456 contribute 0 and 2). Numbers match §9 but "3-seed pooling" is nominal.
  => Ministral end of the consolidation gradient is weaker-evidenced than Qwen/Llama.
- No SUMMARY contradicts the gradient. Context-stratified "context-modulated" auto-verdict
  is disclosed-as-misleading in §5 (not hidden).
- NOT checked by C: mode-balanced cos (§8), continuation rows (§3/§4).

## ★★★ CRUX FINDING (Agent B + my verification): GAME-STATE CONFOUND ★★★
The CHECK-vs-FOLD "decision direction" is statistically indistinguishable from a
trivial observable game-state feature "am I facing a bet?" (bet_to_call>0).
Cross-model probe baselines (results/direction_probe_baselines/*.json):
| Model | verb probe | permuted floor | bet_to_call cross-task |
| Llama L14 | 0.988 | 0.523 | **0.988 (==verb)** |
| Qwen  L23 | 0.998 | 0.508 | **1.000 (>verb)** |
| Ministral L16 | 0.900 | **0.900 (==floor!)** | 0.921 |
- Llama/Qwen: "decision direction" ≈ "facing-a-bet direction" (same residuals decode
  both equally). Ministral probe is DEGENERATE (verb acc == permuted floor == majority
  base rate; a random direction at 0.946 BEATS it). Committed .md "reading guide" spins
  high cross-task acc as "information-rich residuals," NOT naming it a confound. [ISSUE]
- WHY: buckets are PARTLY DEFINITIONAL in bet_to_call. Confirmed empirically (Llama 3
  seeds): FOLD = 0 bet=0 / 289 bet>0 → **clean_legal_fold ⟺ bet>0 (100%)**;
  illegal_fold ⟺ bet=0 (fold illegal when check is free). check_or_call = 46% face bet.

### What the confound DOES vs DOESN'T undermine (my analysis):
- UNDERMINED: all PROBE/correlational claims — "decision direction exists" (it may be
  the bet-context direction), and the **+0.51 mode-stability cosine** (could be stability
  of the facing-a-bet direction). Ministral probe meaningless. THIS is the #1 paper risk.
- LIKELY ROBUST: the **illegal_fold causal sufficiency** (main headline). Target is
  definitionally bet=0; the model still tried to FOLD; patching a check-source flips to
  CHECK at 100% across 3 seeds. Sources are MIXED bet-context (46% face bet) yet flip is
  uniform → transferred signal is a DECISION (check/call), not bet-context (a facing-bet
  source would push toward FOLD if it were bet-context). Strong.
- VULNERABLE: clean_legal_fold / Tier-4 opponent-invariance — cross-regime (no-bet source
  → facing-bet target), so "+5–20 nats invariance" could partly be bet-context injection.

### KEY ADDITIONAL EXPERIMENT this implies:
**Bet-matched control**: restrict to facing-a-bet decisions (bet>0) and probe/patch
CALL vs FOLD with bet_to_call held constant. If direction/flip survives → decision rep
is real beyond game state. Directly answers the reviewer objection. (Mirror: bet=0
CHECK vs illegal-FOLD.) This is the single most important experiment to add.

## Environment
- CPU only (torch.cuda=False). numpy 2.4.1, scipy 1.17.0. NO sklearn, NO pandas.
