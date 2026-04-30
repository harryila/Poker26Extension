# Miscalibrated Beliefs in LLM Poker Agents

> **Do LLM poker agents maintain coherent Bayesian-style beliefs about hidden opponent hands, or do they rely on heuristics while outputting plausible-sounding distributions?**

A reproducible, instrumented **Fixed-Limit Texas Hold'em** environment built on [PokerKit](https://github.com/uoftcprg/pokerkit) for studying belief formation and updating in LLM agents. Supports open-weight models (Llama, Mistral, Qwen) via HuggingFace and closed models (GPT, Claude, Gemini) via API, with Chain-of-Thought prompting and mechanistic interpretability tooling.

---

## Overview

This repository studies **belief formation and updating**, not optimal poker play. Poker naturally requires reasoning under uncertainty about hidden information — making it ideal for diagnosing whether LLMs perform genuine probabilistic inference or rely on shallow heuristics.

The experiment pipeline:
1. An LLM agent plays heads-up poker and is prompted for explicit probability distributions over opponent hand categories
2. Two Bayesian oracle baselines compute ground-truth posteriors for comparison
3. Divergence metrics, calibration analysis, and mechanistic interpretability tools reveal *how* and *why* beliefs diverge from rationality

---

## Two Oracle Baselines

| Oracle | Computes | Ignores |
|--------|----------|---------|
| **CardOnly** | P(opp hand \| board + hero cards) — uniform over blocker-consistent hands | Betting history |
| **StrategyAware** | P(opp hand \| board + hero cards + **actions**) — Bayesian update with opponent model | Nothing observable |

The gap between these baselines captures the **information content of opponent actions**. If LLM beliefs land closer to CardOnly, the model ignores action signals. If closer to StrategyAware, it performs genuine Bayesian-style updating.

A separate **EquityOracle** computes win/tie/lose probabilities given true hole cards — privileged ground truth used only for logging.

---

## Supported Models

### Open Models (HuggingFace)

| Short Name | Model ID | System Role | Notes |
|:-----------|:---------|:-----------:|:------|
| `llama-8b` | `meta-llama/Llama-3.1-8B-Instruct` | Yes | Default dev model |
| `llama-70b` | `meta-llama/Llama-3.1-70B-Instruct` | Yes | Research model (96 GB+ VRAM) |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | No | System msg merged into user turn |
| `qwen-7b` | `Qwen/Qwen2.5-7B-Instruct` | Yes | — |

Any HuggingFace model ID can also be passed directly via `--hf-model`.

### Closed Models (API)

| Provider | Default Model | Logprobs | Env Var |
|:---------|:-------------|:--------:|:--------|
| **OpenAI** | `gpt-4o` | Yes (top-20) | `OPENAI_API_KEY` |
| **Anthropic** | `claude-sonnet-4-20250514` | No | `ANTHROPIC_API_KEY` |
| **Google** | `gemini-2.0-flash` | No | `GOOGLE_API_KEY` |

---

## Features

| Category | Capability |
|:---------|:-----------|
| **Belief elicitation** | 14 semantic hand-strength buckets (7/30 bucket schemes for ablation) |
| **Chain-of-Thought** | Separate token budgets, reasoning quality scoring |
| **Logit lens** | Layer-by-layer prediction evolution during generation |
| **Attention analysis** | Token category fractions (cards vs. actions vs. instructions) |
| **Linear probing** | Per-layer classifiers revealing internal vs. expressed beliefs |
| **Token attribution** | Integrated gradients via Captum for input saliency |
| **Oracle baselines** | CardOnly + StrategyAware with matched opponent model |
| **Full telemetry** | Parse success, coherence diagnostics, prompt hashes, reproducibility |

---

## Requirements

- **Python >= 3.11** (required by PokerKit)
- PokerKit >= 0.6.0, NumPy, SciPy, pandas, matplotlib
- For HF agents: `torch`, `transformers`, `accelerate`
- For API agents: `openai`, `anthropic`, and/or `google-genai`
- For mech interp: `captum`, `scikit-learn` (optional)

## Installation

```bash
git clone <repo-url>
cd Updated-miscalibrated-belief-llms

# Option A: setup script
./setup.sh                              # basic
./setup.sh --with-gpu                   # with CUDA PyTorch
./setup.sh --with-gpu --hf-token YOUR_TOKEN

# Option B: manual
python -m venv venv
source venv/bin/activate                # Linux/Mac
# venv\Scripts\activate                 # Windows
pip install -r requirements.txt
```

---

## Quick Start

### Baseline experiments (no GPU needed)

```bash
# Random vs. random — 100 hands
python run_experiment.py --agent random --hands 100 --seed 42 -v

# Threshold vs. call-station
python run_experiment.py --agent threshold --opponent call \
    --opponent-preset informative_v2 --hands 100 -v

# List available model short names
python run_experiment.py --list-models
```

### HuggingFace LLM experiments

```bash
# Llama 8B with belief elicitation
python run_experiment.py \
    --agent hf --hf-model llama-8b \
    --opponent threshold --opponent-preset informative_v2 \
    --hands 50 --seed 42 --elicit-beliefs -v

# Mistral 7B with Chain-of-Thought
python run_experiment.py \
    --agent hf --hf-model mistral-7b \
    --opponent threshold --opponent-preset informative_v2 \
    --hands 50 --seed 42 --elicit-beliefs --cot -v

# Qwen 7B with logit lens capture
python run_experiment.py \
    --agent hf --hf-model qwen-7b \
    --opponent threshold --opponent-preset informative_v2 \
    --hands 20 --seed 42 --elicit-beliefs --logit-lens -v

# 70B research run (multi-GPU)
python run_experiment.py \
    --agent hf --hf-model llama-70b \
    --opponent threshold --opponent-preset informative_v2 \
    --hands 1000 --seed 42 --temperature 0.0 --elicit-beliefs \
    --out logs/phase2_70b.jsonl -v
```

### Closed model experiments (API)

```bash
# GPT-4o (logprobs captured automatically)
python run_experiment.py \
    --agent api --api-provider openai --api-model gpt-4o \
    --opponent threshold --opponent-preset informative_v2 \
    --hands 50 --seed 42 --elicit-beliefs -v

# Claude with CoT
python run_experiment.py \
    --agent api --api-provider anthropic \
    --opponent threshold --opponent-preset informative_v2 \
    --hands 50 --seed 42 --elicit-beliefs --cot -v

# Gemini
python run_experiment.py \
    --agent api --api-provider google --api-model gemini-2.0-flash \
    --opponent threshold --opponent-preset informative_v2 \
    --hands 50 --seed 42 --elicit-beliefs -v
```

---

## Analysis Pipeline

### 1. Enrich logs with oracle posteriors

```bash
python -m analysis.build_dataset logs/experiment.jsonl logs/enriched.jsonl \
    --opponent informative_v2
```

### 2. Analyze beliefs

```bash
# JS/L1 divergence analysis
python -m analysis.analyze_beliefs logs/enriched.jsonl \
    --json-out results/combined_analysis.json

# PCE distribution with bootstrap CIs
python -m analysis.compute_pce_distribution logs/enriched.jsonl \
    --output-records results/pce_distribution.csv \
    --output-summary results/pce_summary.csv \
    --bootstrap 2000

# Update coherence (card reveals vs. actions)
python -m analysis.compute_update_coherence logs/enriched.jsonl \
    --output results/update_coherence.csv \
    --output-summary results/update_coherence_summary.json

# Paper figures
python -m analysis.plot_paper_figures \
    --pce-data results/pce_distribution.csv \
    --pce-summary results/pce_summary.csv \
    --update-data results/update_coherence.csv \
    --output-dir figures/
```

### 3. CoT and interpretability analysis

```bash
# CoT vs. direct prompting comparison
python -m analysis.analyze_cot logs/cot_enriched.jsonl logs/direct_enriched.jsonl \
    --json-out results/cot_comparison.json

# Logit lens (crystallization analysis)
python -m analysis.analyze_logit_lens logs/experiment_logit_lens.jsonl \
    --plot figures/entropy_curve.png \
    --json-out results/logit_lens.json

# Attention / attribution / probe analysis
python -m analysis.analyze_attention logs/experiment_attention.jsonl
python -m analysis.analyze_attribution logs/experiment.jsonl
python -m analysis.analyze_probes logs/experiment_hiddens.jsonl
```

---

## Command-Line Reference

### Core Flags

| Flag | Default | Description |
|:-----|:--------|:------------|
| `--agent` | `random` | Agent type: `random`, `call`, `threshold`, `hf`, `api` |
| `--opponent` | — | Opponent type (heads-up only) |
| `--opponent-preset` | `default` | Threshold preset: `default`, `informative_v2`, `tight_aggressive`, … |
| `--hands` | `100` | Number of hands to play |
| `--seed` | `42` | Base random seed |
| `--out` | `logs/experiment.jsonl` | Output JSONL path |
| `--elicit-beliefs` | off | Prompt agent for belief distributions |
| `--no-oracle` | off | Skip equity oracle (faster) |
| `-v` | off | Verbose progress output |

### HuggingFace Flags

| Flag | Default | Description |
|:-----|:--------|:------------|
| `--hf-model` | `llama-8b` | Model ID or short name |
| `--temperature` | `0.2` | Generation temperature (0 = greedy) |
| `--top-p` | `0.9` | Top-p sampling |
| `--belief-format` | `compact` | `compact` (list) or `full` (labeled dict) |
| `--cot` | off | Enable Chain-of-Thought prompting |
| `--logit-lens` | off | Capture per-layer hidden states |

### API Flags

| Flag | Default | Description |
|:-----|:--------|:------------|
| `--api-provider` | `openai` | `openai`, `anthropic`, or `google` |
| `--api-model` | provider default | Model name (e.g. `gpt-4o`, `gemini-2.0-flash`) |

---

## Project Structure

```
poker_env/                              Core poker environment
├── env.py                              PokerKitEnv — reset/step game loop (2–6 players)
├── actions.py                          Action/ActionType definitions
├── obs.py                              Observation dataclass, stable history serialization
├── deck.py                             DeterministicDeck for reproducibility
├── config.py                           Model registry, token budgets, bucket config
├── agents/
│   ├── base.py                         BaseAgent abstract interface
│   ├── random_agent.py                 Uniform random baseline
│   ├── call_agent.py                   Always check/call baseline
│   ├── threshold_agent.py              Hand-strength threshold bot
│   ├── hf_agent.py                     HuggingFace LLM agent (Llama, Mistral, Qwen)
│   ├── api_agent.py                    API agent (OpenAI, Anthropic, Google)
│   ├── prompts.py                      System messages and prompt templates
│   └── json_utils.py                   JSON extraction, CoT parsing
├── oracle/
│   └── win_prob.py                     EquityOracle — Monte Carlo equity computation
├── logging/
│   └── decision_logger.py              JSONL logger (config → decisions → summaries)
├── interp/                             Mechanistic interpretability
│   ├── logit_lens.py                   Per-layer entropy and top tokens
│   ├── attention.py                    Attention pattern extraction
│   ├── probing.py                      ProbeDataset + LinearProbe (sklearn)
│   └── attribution.py                  Captum integrated gradients
└── tests/
    ├── test_env.py                     Environment tests (basic + multi-way)
    ├── test_determinism.py             Reproducibility tests
    └── test_golden.py                  Known-outcome golden tests

analysis/                               Offline analysis pipeline
├── buckets.py                          14-bucket hand classification (+ 7/30 ablation)
├── opponent_model.py                   ParametricOpponent, hand-strength computation
├── posterior_oracle.py                 CardOnlyPosterior, StrategyAwarePosterior
├── belief_schema.py                    BeliefOutput validation and coherence checking
├── belief_utils.py                     Compact ↔ dict conversion, repair
├── projection.py                       Simplex projection (Duchi et al.)
├── prompts.py                          Belief prompt templates, bucket descriptions
├── build_dataset.py                    Enrich JSONL with oracle posteriors
├── analyze_beliefs.py                  JS/L1 divergence, action-conditioning analysis
├── compute_pce_distribution.py         Per-component error with bootstrap CIs
├── compute_update_coherence.py         Belief update dynamics (card vs. action)
├── plot_paper_figures.py               Matplotlib paper figure generation
├── analyze_cot.py                      CoT reasoning quality scoring
├── analyze_logit_lens.py               Crystallization layer, entropy curves
├── analyze_attention.py                Attention pattern analysis
├── analyze_attribution.py              Token attribution analysis
├── analyze_probes.py                   Linear probe analysis
├── metrics/
│   ├── calibration.py                  PCE, KL, JS, Brier, ECE
│   ├── coherence.py                    Probability axiom violation checks
│   ├── update_coherence.py             Bayesian update quality metrics
│   └── belief_action.py               Stated vs. implied belief divergence
├── implied_belief/
│   ├── q_value.py                      Monte Carlo Q-value estimation
│   └── inverse.py                      Inverse decision rule for implied beliefs
└── tests/
    ├── test_buckets.py
    ├── test_posteriors.py
    ├── test_metrics.py
    └── test_q_values.py

run_experiment.py                       Main CLI experiment driver
test_v2_features.py                     Integration tests (42 tests)
```

---

## Key Design Decisions

**Why heads-up only for belief experiments?**
Multi-way (3+ players) introduces multiple opponent posteriors, exponentially complex action-likelihood modeling, and side-pot complications. The codebase blocks `--elicit-beliefs` with >2 players unless `--i-know-what-im-doing` is passed.

**Why `informative_v2` opponent?**
With a `call` opponent, CardOnly and StrategyAware posteriors are nearly identical (JS ~ 0.015) — there's no signal to test whether the LLM uses betting history. The `informative_v2` preset achieves JS(CardOnly, StrategyAware) ~ 0.05–0.06 by making opponent actions highly correlated with hand strength.

**Why ThresholdAgent imports from analysis?**
`ThresholdAgent` imports `compute_hand_strength` from `analysis.opponent_model` so the bot's in-game behavior exactly matches the parametric model used in offline analysis. This ensures StrategyAware posteriors are valid.

**Why two belief formats?**
- **Compact**: `{"schema":"buckets_14_v1","probs":[p0,...,p13]}` — fewer tokens, lower parse-failure rate
- **Full**: `{"premium_pairs":0.05, "strong_pairs":0.08, ...}` — more readable, self-documenting

---

## Metrics Reference

| Metric | Measures | Range |
|:-------|:---------|:------|
| JS(LLM, CardOnly) | Distance to blocker-only baseline | [0, 0.83] |
| JS(LLM, StrategyAware) | Distance to Bayesian posterior | [0, 0.83] |
| JS(CardOnly, StrategyAware) | Oracle separation (test validity) | Should be > 0.05 |
| PCE | Per-component calibration error | [0, 1] |
| Coherence rate | Fraction passing probability axiom checks | [0, 1] |
| Update correlation | Correlation of LLM vs. oracle belief updates | [−1, 1] |
| Update magnitude ratio | LLM update size / oracle update size | [0, ∞) |
| Crystallization layer | Layer where logit-lens prediction stabilizes | [0, num_layers] |

---

## Prior Results (Llama 3.1 70B, N = 1,084)

| Metric | Value | Interpretation |
|:-------|:------|:---------------|
| JS(LLM, CardOnly) | 0.407 | Distance to combo-counting baseline |
| JS(LLM, StrategyAware) | 0.420 | Distance to Bayesian posterior |
| LLM closer to | CardOnly | Model ignores betting history |
| LLM "trash" mass | 16.9% | Should be ~66% — severe base-rate neglect |
| Update correlation | 0.056 | Nearly random update direction |
| Update magnitude ratio | 3–11× | Over-updating relative to oracle |

These results are from a single model. The current codebase enables systematic comparison across **Llama, Mistral, Qwen, GPT-4o, Claude, and Gemini** — with CoT and mechanistic interpretability to understand *why* beliefs are miscalibrated.

---

## Running Tests

```bash
# All 148 tests
python -m pytest -v

# By module
python -m pytest poker_env/tests/ -v       # 48 environment tests
python -m pytest analysis/tests/ -v         # 58 analysis tests
python -m pytest test_v2_features.py -v     # 42 integration tests
```

---

## Programmatic Usage

```python
from poker_env import PokerKitEnv, ActionType

env = PokerKitEnv(num_players=2)
obs = env.reset(seed=42)

done = False
while not done:
    action = next(a for a in env.legal_actions() if a.type == ActionType.CHECK_OR_CALL)
    obs, reward, done, info = env.step(action)

print(f"Final stacks: {info['final_stacks']}")
```

### With oracle posteriors

```python
from analysis import CardOnlyPosterior, StrategyAwarePosterior, ParametricOpponent

card_only = CardOnlyPosterior()
posterior = card_only.compute(hero_hole=["Ah", "Kd"], board=["Qc", "Jh", "2s"])

opponent = ParametricOpponent.from_preset("informative_v2")
strategy_aware = StrategyAwarePosterior(opponent)
posterior = strategy_aware.compute(
    hero_hole=["Ah", "Kd"],
    board=["Qc", "Jh", "2s"],
    opponent_actions=[{"action": "BET_OR_RAISE", "street": "PREFLOP"}],
)
```

---

## License

MIT
