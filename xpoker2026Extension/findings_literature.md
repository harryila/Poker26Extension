# Literature positioning (Valency) — for main-track framing

## ★ DIRECT NEIGHBORS / competitive set
- **2512.23722 Emergent World Beliefs: Transformers in Stochastic Games** (Dec 2025).
  Pretrains a GPT on Poker Hand History; nonlinear probes recover hand ranks + EQUITY +
  belief states correlating with theory. CLOSEST paper. DIFFERENTIATORS for us:
  (1) CAUSAL patching/ablation, not just decodability; (2) off-the-shelf INSTRUCTION-TUNED
  chat LLMs via prompting (deployment setting) vs bespoke pretrained GPT; (3) we find
  beliefs MISCALIBRATED/heuristic (base-rate neglect) vs "beliefs exist"; (4) cross-model
  gradient. NOTE: complementary — they show from-scratch poker-GPT forms beliefs; we show
  prompted LLMs have a causal decision direction but wrong beliefs. MUST cite + differentiate.
- **2403.15498 Emergent World Models...Chess-Playing LLMs** (Karvonen 2024). Othello-GPT→chess;
  linear probes + activation edits to board state; "player skill vector" improves win 2.6×.
  Canonical emergent-world-model lineage. We extend to IMPERFECT-INFO + miscalibration + causal verb direction.
- **2406.19501 Monitoring Latent World States: Propositional Probes** (Feng/Russell/Steinhardt 2024).
  "LLMs encode a FAITHFUL world model but DECODE it unfaithfully." Inverse framing to ours —
  engage directly: is poker miscalibration an encoding failure or a decoding/readout failure?
  (Our patching could test this — strong intellectual hook.)

## ★ SUPPORTING (cite to motivate the miscalibration premise)
- **2505.10571 On the Failure of Latent State Persistence in LLMs** (2026). LLMs violate
  probabilistic principles, "reactive post-hoc solvers not proactive planners." Strong support.
- **2504.13644 LLMs to Demonstrate Rational Probabilistic Beliefs** (2025). LLMs lack coherent
  probabilistic beliefs. Directly supports base-paper premise.
- **2508.19851 Tracking World States...Chess (state-based, model-agnostic)** (2025). Model-agnostic
  state-affordance eval — alternative to internal probes; relevant to robustness of claims.

## ★ METHODS references (cite for credibility; we mostly comply)
- **2309.16042 Towards Best Practices of Activation Patching** (Zhang & Nanda 2024). THE methods
  ref. Our spec-adj null control + metric choices align. Cite to preempt methodology reviewers.
- Steering-vector line our "decision direction" sits in: 2410.04962 (activation scaling),
  2503.00177 (sparse/SAE steering), 2604.24693 (contextual linear steering, 2026), 2508.16989
  (latent directions of reflection). Our direction = a task-grounded, causally-validated steering
  vector; position vs these (we add behavioral game-theoretic grounding + cross-model gradient).

## ★ CROSS-MODEL / UNIVERSALITY (frames the "consolidation gradient")
- **2410.06672 Towards Universality: Mechanistic Similarity Across LM Architectures** (2024).
  SAEs show most features similar across Transformer vs Mamba. Universality hypothesis. Our
  "consolidation GRADIENT" is the nuance: same SUFFICIENT direction, but necessity/locality
  DIFFERS by family (Llama sparse / Qwen distributed / Ministral flow-through). Position as
  "universality of the WHAT, divergence in the HOW/WHERE."
- **2310.08744 Circuit Component Reuse Across Tasks** (Merullo/Pavlick 2024); **2411.16105
  Adaptive Circuit Behavior** — circuits generalize across tasks. **2511.20273 Singular
  Vector-Based Interpretability** (2025) — head computations "more distributed than assumed"
  (supports the Qwen-distributed result; consider SVD-of-heads as a method upgrade).

## ★ BEHAVIORAL poker / imperfect-info game evals (the layer ABOVE us)
- **2501.08328 PokerBench** (Jan 2025) — 11k poker scenarios; SOTA LLMs underperform optimal
  play; FT helps. The play-quality benchmark. We study BELIEF/MECHANISM, not optimality — cite & contrast.
- **2605.23238 GENSTRAT** (May 2026) — procedurally generated 2-player zero-sum imperfect-info
  CARD games; 6-axis capability profiles (incl opponent modeling), 9 LLMs, 36k matches; behavioral.
  We provide the mechanistic substrate for these behavioral diagnostics. Very current — cite.
- **2407.04467 Are LLMs Strategic Decision Makers?** — systematic biases; CoT helps some/hurts others
  (relevant to our CoT-vs-noCoT mode analysis). **2412.13013**, **2603.19167**, **2503.02582** — strategic-reasoning evals.

## ★★ CAUSAL-VALIDITY / probing-confound (DIRECTLY about our crux)
- **2605.08012 Position: Mech Interp Must Disclose Identification Assumptions for Causal Claims**
  (May 2026). Audits 10 papers; "validation is NOT identification"; demands: state if causal,
  name identification strategy, enumerate + stress assumptions. ⇒ ADD an explicit
  "identification assumptions" section; the bet_to_call confound is the assumption to stress.
  Citing+complying puts us AHEAD of the rigor curve. KEY framing device.
- **2408.15510 How Reliable are Causal Probing Interventions?** (2024) — completeness vs
  selectivity tradeoff; nonlinear interventions more reliable. Method upgrade path for our probe.
- **2309.16042 Best Practices of Activation Patching** (Zhang&Nanda) — already noted; comply & cite.

## GAP / OPPORTUNITY (preliminary)
- The crowded part = "probe shows X is decodable." The UNDER-served part = CAUSAL + BEHAVIORAL
  (does the direction change game decisions) + CROSS-MODEL-FAMILY mechanism comparison +
  the encode-vs-decode question (Feng et al). Our causal sufficiency on a real decision is the edge.
- BUT the bet_to_call confound (findings.md crux) is exactly what a reviewer who knows
  2512.23722 / probing-confound literature will attack. Bet-matched control is non-negotiable.
