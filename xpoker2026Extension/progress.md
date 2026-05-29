# Progress log

## 2026-05-28
- Read README.md, REDO_PLAN.md, PHASE_Q.md, AUDIT_FINDINGS.md (extension).
- Confirmed CPU-only env; significance tests re-runnable, GPU/sklearn not.
- Resolved date concern (UTC vs PDT).
- Created task_plan.md / findings.md / progress.md.
- NEXT: fetch base paper + reviews; begin independent result verification.
- Verified all 4 SIGNIFICANCE_*.md reproduce exactly; Qwen necessity + sufficiency
  seed-consistent; sufficiency top-1=100% all 3 seeds (magnitude noisy).
- 3 audit agents returned: patching sufficiency SOUND; oracle math CORRECT; probe CV clean
  BUT decision-direction confounded with bet_to_call (verb probe acc == bet probe acc:
  Llama .988/.988, Qwen .998/1.000; Ministral probe degenerate == permuted floor).
- Verified confound myself: FOLD=100% facing-bet; illegal_fold sufficiency robust (within
  bet=0 regime), clean_legal_fold/Tier4 + all probe/cosine claims vulnerable.
- Found Ministral cells effectively single-seed (sufficiency s42 n=15; §9 null 78/80 s42).
- Valency: mapped competitive set (Emergent World Beliefs poker-GPT 2512.23722 = direct
  neighbor; PokerBench; GENSTRAT; universality 2410.06672; identification-assumptions
  position paper 2605.08012). See findings_literature.md.
- DELIVERED synthesis in chat. Could NOT get OpenReview reviews (API 0 notes) — flagged.

## 2026-05-28 (execution push — "do everything")
CPU (ran + committed reports):
- confound_projection_analysis.py -> CONFOUND_PROJECTION.md (direction verb-aligned, cos .95-.99)
- sufficiency_ci.py -> SUFFICIENCY_CI.md (top1=100% all seeds; spec-adj range [15.6,25.1])
- regen_drift_audit.py -> REGEN_DRIFT.md (Qwen clf 15% OK; Llama if 73% unreliable)
- bet_matched_probe.py (CPU analysis half) — self-test passes; sklearn installed
- CLAIMS_AND_IDENTIFICATION.md (reframe + identification assumptions per 2605.08012)
GPU scripts written + syntax/import-validated (NOT run here — no GPU):
- bet_matched_recapture.py (B1, also tags oracle+belief for C1)
- causal_patching.py: +--source/target-bet-filter (B2) + fixed pre-existing --help %% bug
- inference_head_ablation.py: +rand:K:SEED head-set syntax (B4)
- poker_env/interp/patching.py: +ActivationAdditionHook (steering; additive, all-positions)
- encode_vs_decode.py (C1; synthetic test passes) + posterior_steering.py (C2)
- run_experiment.py: +_maybe_attach_circuit_hook env-var injection (C3, behavior-at-scale)
- qwen_band_svd.py (D1)
- RUNBOOK_followups.md = turnkey GPU commands (Phases B/C/D)
Regression check: 106 poker_env+analysis tests pass; edited modules import clean.
