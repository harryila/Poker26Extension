# Task plan — skeptical audit + literature positioning of xpoker2026Extension

## Goal
Comb through `xpoker2026Extension` as a skeptical researcher/engineer:
1. Verify design + implementation make sense (no bugs, nothing odd).
2. Verify all committed RESULTS are internally consistent — especially anything
   run N times (3-seed pooling: 42/123/456) must agree across trials.
3. Confirm the headline claims (consolidation gradient; sufficiency/necessity/
   opponent-invariance/mode-stability) are actually supported by the data on disk,
   not just asserted in markdown.
4. Use Valency to map recent top-venue literature; find how to push this to a
   main-track paper + what additional experiments to run.

## Context (what this is)
- Base paper: OpenReview 7cRVi47XT0 — "Miscalibrated Beliefs in LLM Poker Agents".
- Extension = mech-interp: do poker LLMs have a causal "decision/verb direction"
  (FOLD / CHECK_OR_CALL / BET_OR_RAISE) in the residual stream?
- Methods: linear probes, activation/residual patching, component decomposition,
  attention-head ablation, continuation-after-patch, Tier-4 opponent-invariance.
- Headline: "circuit-consolidation gradient" Qwen-8B > Llama-8B > Ministral-8B.
- Models: Llama-3.1-8B, Ministral-8B, Qwen3-8B (all 8B). 3 seeds 42/123/456 pooled.
- Env here: CPU only (no CUDA). Can re-run stdlib/numpy analysis + significance,
  NOT GPU experiments and NOT sklearn probes (sklearn missing).

## Phases
- [x] Phase 0: Orientation — read README, PHASE_Q, REDO_PLAN, AUDIT_FINDINGS.
- [x] Phase 1: Base-paper context (README prior results). Reviews NOT retrievable
      (OpenReview API 0 notes) — may ask user.
- [x] Phase 2: Result verification — ALL 4 significance files reproduce exactly;
      Qwen necessity + sufficiency seed-consistent; sufficiency top-1=100% all 3 seeds.
- [x] Phase 3: Code audit (3 agents) — patching sufficiency SOUND (no leak), oracle
      math CORRECT, probe CV clean; confound found in probe construct.
- [x] Phase 4: Methodology — CRUX confound (decision dir ≈ "facing a bet"); Ministral
      probe degenerate + single-seed; pseudoreplication; spec-adj noisy; L8 control;
      GoF mostly disciplined (L19 survives Bonferroni). 
- [x] Phase 5: Literature research (Valency) — done; see findings_literature.md.
- [x] Phase 6: Synthesis — delivered in chat (audit verdict + positioning + experiments).

## AUDIT VERDICT (so far)
- Engineering/artifacts: unusually rigorous & honest. Everything I re-derived
  reproduced exactly; audit docs self-document their own confounds.
- STRONGEST claim = causal SUFFICIENCY on illegal_fold (robust to confound, seed-stable).
- WEAKEST = probe-based "belief/decision direction" + mode-stability cosine
  (confounded with bet_to_call) and Ministral cells (single-seed + degenerate probe).
- #1 fix = bet-matched control. Report spec-adj CIs not points. Re-balance Ministral seeds.

## Decisions / constraints
- Trust NOTHING in the .md until checked against committed JSONL/CSV or code.
- Distinguish "real finding" from "narrative imposed on 3 points."
- No GPU here → audit = artifacts + code + recomputable stats only.

## Status: Phase 0 done, starting Phase 1+2.
