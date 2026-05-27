# Poker Environment for LLM Belief Modeling Research

A reproducible, instrumented Fixed-Limit Texas Hold'em environment built on [PokerKit](https://github.com/uoftcprg/pokerkit) for research on belief modeling in LLM poker agents. Supports 2-6 players.

## Research Overview

> **Scope Note:** This repository studies **belief formation and updating**, not optimal poker play. Results should not be interpreted as claims about exploitability, equilibrium convergence, or policy optimality. The goal is to diagnose whether LLMs reason probabilistically about hidden information, not to build competitive poker agents.

This codebase supports research into **whether LLMs actually maintain coherent Bayesian beliefs** about hidden information. The core research question:

> *Do LLM poker agents internally maintain coherent probability distributions over opponent hands—or do they rely on surface-level heuristics while producing plausible-sounding but inconsistent belief statements?*

### Important: Two Different "Oracles"

This system uses two distinct oracle concepts—don't confuse them:

| Oracle | Module | Purpose | When Used |
|--------|--------|---------|-----------|
| **EquityOracle** | `poker_env/oracle/` | Win/tie/lose probability using **true opponent holes** (privileged ground truth) | During gameplay logging |
| **PosteriorOracle** | `analysis/posterior_oracle.py` | P(opponent_hand \| public_history) using Bayesian inference + opponent model | Offline analysis |

The **EquityOracle** answers: "Given I know everyone's cards, what's my win probability?"  
The **PosteriorOracle** answers: "Given only public information, what should I believe about opponent hands?"

### Heads-Up for Belief Experiments

**All belief/posterior experiments should use heads-up (2-player) games.**

While the environment supports 2-6 players, multi-way introduces:
- Multiple opponent posteriors to track
- Exponentially complex action-likelihood modeling
- Side pot complications

The codebase warns when using >2 players. Multi-way is "engineering-supported" but not recommended for the core research claims.

### The Research Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  1. Run Games   │───▶│  2. Log Data    │───▶│ 3. Enrich with  │───▶│  4. Compute     │
│  (Experiment)   │    │  (Decision Pts) │    │ PosteriorOracle │    │  Metrics        │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                                              │
        │                      ▼                                              ▼
        │              equity_given_true_hands                      ┌─────────────────┐
        │              (EquityOracle, optional)                     │  5. Analyze:    │
        │                                                           │  - PCE          │
        └───────────────────────────────────────────────────────────│  - Coherence    │
                                                                    │  - Belief-Action│
                                                                    └─────────────────┘
```

### Key Research Contributions Enabled

1. **Posterior Calibration Error (PCE)**: Quantify how far LLM beliefs diverge from Bayesian posteriors
2. **Coherence Metrics**: Measure probability axiom violations (negative probs, wrong sums)
3. **Update Coherence**: Track whether beliefs update Bayesianly with new information
4. **Belief-Action Divergence**: Compare stated beliefs to action-implied beliefs
5. **"Plays Well Despite Bad Beliefs"**: Identify when LLMs make good decisions with nonsensical beliefs

### Research Findings (January 2026)

This codebase was used to produce a complete research study. Key findings:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| N (valid beliefs) | 1,084 | Combined Phase 1A + Phase 2 data |
| JS(LLM, CardOnly) | 0.4067 | LLM distance to combo-counting baseline |
| JS(LLM, StrategyAware) | 0.4204 | LLM distance to Bayesian action-conditioned |
| **LLM closer to** | **CardOnly** | **By 0.0137 — LLM ignores betting history** |
| LLM trash mass | 16.89% | Should be ~66% (4x underestimate) |
| Update correlation | 0.056 | Nearly random — updates in wrong direction |

> **Headline:** Llama 3.1 70B shows weak directional sensitivity to betting actions, but severe base-rate neglect dominates—beliefs stay closer to CardOnly than StrategyAware. The model attempts to update beliefs but does so incorrectly: over-updating by 3-11x with near-zero correlation to oracle updates.

See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) for complete analysis.

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - project overview, API reference, and analysis system |
| [LLM_AGENT_README.md](LLM_AGENT_README.md) | HuggingFace LLM agent integration (Llama 3.1 8B/70B) |
| [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md) | Step-by-step research experiment guide with commands |
| [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) | Recorded findings from sanity check experiments |
| [DEFERRED_WORK.md](DEFERRED_WORK.md) | Deferred experiments and extensions (scope discipline) |

## Requirements

- **Python >= 3.11** (required by PokerKit)
- PokerKit >= 0.6.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0

## Installation

```bash
# Clone the repo
git clone https://github.com/harryila/poker2026.git
cd poker2026

# Run setup script (creates venv, installs dependencies)
./setup.sh

# Or manually:
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Run experiments

```bash
# 2 players (heads-up) - random vs random
python run_experiment.py --agent random --hands 100 --seed 42 -v

# 2 players - random vs call
python run_experiment.py --agent random --opponent call --hands 100 -v

# 4 players with mixed agents
python run_experiment.py --num-players 4 --agents random,call,random,call --hands 100 -v

# 6 players all random (fast, no oracle)
python run_experiment.py --num-players 6 --agent random --hands 100 --no-oracle -v
```

### Build analysis dataset (enrich logs with oracle posteriors)

```bash
# Add oracle posteriors to experiment logs
python -m analysis.build_dataset logs/experiment.jsonl logs/enriched.jsonl --opponent default
```

## Command Line Reference

### All Flags Explained

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `--num-players` | Number of players in the game (2-6) | `2` | `--num-players 4` |
| `--agent` | Default agent type for all players | `random` | `--agent call` |
| `--agents` | Comma-separated list of agent types for each player. Overrides `--agent`. | None | `--agents random,call,random,call` |
| `--opponent` | Agent type for player 1 only (heads-up shortcut). Only works with 2 players. | None | `--opponent threshold` |
| `--opponent-preset` | Preset for threshold opponent behavior. | `default` | `--opponent-preset informative` |
| `--hands` | Number of hands to play | `100` | `--hands 500` |
| `--seed` | Base random seed for reproducibility. Each hand uses `seed + hand_num * 1000`. | `42` | `--seed 12345` |
| `--out` | Output file path for JSONL decision logs | `logs/experiment.jsonl` | `--out results/run1.jsonl` |
| `--no-oracle` | Skip equity oracle computation. Equity oracle is ON by default. | Off (oracle computed) | `--no-oracle` |
| `--elicit-beliefs` | Call belief() on agents at each decision (for LLM belief analysis) | False | `--elicit-beliefs` |
| `--i-know-what-im-doing` | Allow multi-way belief experiments (bypasses heads-up enforcement) | False | `--i-know-what-im-doing` |
| `-v, --verbose` | Print progress every 10 hands and show summary at end | False | `-v` |

### Agent Types

| Type | Behavior |
|------|----------|
| `random` | Selects uniformly at random from legal actions |
| `call` | Always checks or calls (never folds, never raises) |
| `threshold` | Plays based on hand strength (raises strong, folds weak). Use for belief experiments. |

### Threshold Agent Presets

When using `--opponent threshold`, specify behavior with `--opponent-preset`:

| Preset | Description | Use Case |
|--------|-------------|----------|
| `informative_v2` | High action-hand correlation (canonical) | **Recommended for belief experiments** |
| `informative` | Legacy alias for `informative_v2` | Backwards compatibility |
| `default` | Balanced play | General testing |
| `tight_aggressive` | Selective but aggressive | Robustness checks |
| `loose_aggressive` | Plays many hands aggressively | Robustness checks |

> **Why `informative_v2` matters:** With a `call` opponent (always calls), CardOnly and StrategyAware posteriors are nearly identical (JS=0.015), making it impossible to test if LLM uses betting history. The `informative_v2` preset achieves JS(CardOnly, StrategyAware) ≈ 0.05-0.06, providing sufficient signal for belief experiments. See [EXPERIMENT_RESULTS.md - Appendix: Opponent Informativeness Validation](EXPERIMENT_RESULTS.md#appendix-opponent-informativeness-validation).

### Flag Combinations

**Setting agents for each player:**

```bash
# Method 1: Same agent for all players
--agent random                          # All players use random

# Method 2: Different agents per player (use --agents)
--agents random,call,random,call        # P0=random, P1=call, P2=random, P3=call

# Method 3: Heads-up shortcut (2 players only)
--agent random --opponent call          # P0=random, P1=call
```

**Controlling output:**

```bash
# Verbose mode shows progress
python run_experiment.py --hands 100 -v
# Output:
#   Completed 10/100 hands
#   Completed 20/100 hands
#   ...
#   === Experiment Summary (2 players) ===
#   Hands played: 100
#   Player 0 (Player0_random): +50.0 chips (+0.50/hand)
#   Player 1 (Player1_random): -50.0 chips (-0.50/hand)

# Custom output path
python run_experiment.py --out my_results/experiment_001.jsonl
```

**Speed vs accuracy tradeoff:**

```bash
# Full run with oracle (slower, computes win probabilities)
python run_experiment.py --hands 100

# Fast run without oracle (~10x faster)
python run_experiment.py --hands 100 --no-oracle
```

**Reproducibility:**

```bash
# Same seed = same hands = same results (with same agents)
python run_experiment.py --seed 42 --hands 100
python run_experiment.py --seed 42 --hands 100  # Identical results
```

### LLM Agent Flags

When using `--agent hf` (HuggingFace LLM agent):

| Flag | Default | Description |
|------|---------|-------------|
| `--hf-model` | `Llama-3.1-8B-Instruct` | HuggingFace model ID |
| `--temperature` | `0.2` | Generation temperature (0 = deterministic) |
| `--elicit-beliefs` | `False` | Prompt LLM for beliefs at each decision |
| `--belief-format` | `compact` | Belief output format (`compact` or `full`) |
| `--belief-max-new-tokens` | `384` | Max tokens for belief generation |

See [LLM_AGENT_README.md](LLM_AGENT_README.md) for complete HF agent documentation.

### Example Commands

```bash
# Basic 2-player test
python run_experiment.py --agent random --hands 100 --seed 42 -v

# Random vs call-station heads-up
python run_experiment.py --agent random --opponent call --hands 200 -v

# 4-player game with mixed strategies
python run_experiment.py --num-players 4 --agents random,call,random,call --hands 100 -v

# 6-player fast run (no oracle)
python run_experiment.py --num-players 6 --agent random --hands 500 --no-oracle -v

# Custom output location
python run_experiment.py --hands 100 --out results/test_run.jsonl -v

# Specific seed for reproducibility
python run_experiment.py --seed 12345 --hands 100 -v
```

### LLM Belief Experiment Commands (Actual Research)

These are the commands used in the actual research study:

```bash
# Phase 1A: Sanity check with informative opponent (50 hands)
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative_v2 \
    --hands 50 \
    --seed 42 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/sanity_70b_t0_s42_informative.jsonl \
    -v

# Phase 2: Scale-up with informative opponent (1000 hands target)
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative_v2 \
    --hands 1000 \
    --seed 42 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/phase2_70b_t0_s42_informative_v2.jsonl \
    -v
```

**Runtime note:** 70B model inference takes ~2 minutes per hand. A 1000-hand run takes ~33 hours. See [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md) for the full grid.

### Dataset Builder Command

After running experiments, enrich logs with oracle posteriors:

```bash
python -m analysis.build_dataset <input> <output> [--opponent <preset>]
```

| Argument | Description | Example |
|----------|-------------|---------|
| `input` | Path to raw experiment JSONL | `logs/raw.jsonl` |
| `output` | Path for enriched output JSONL | `logs/enriched.jsonl` |
| `--opponent` | Opponent model preset for strategy-aware oracle | `--opponent tight_aggressive` |

**Available opponent presets:** `default`, `tight_passive`, `tight_aggressive`, `loose_passive`, `loose_aggressive`

**Example:**

```bash
# Enrich with default opponent model
python -m analysis.build_dataset logs/experiment.jsonl logs/enriched.jsonl

# Enrich assuming tight-aggressive opponent
python -m analysis.build_dataset logs/experiment.jsonl logs/enriched_ta.jsonl --opponent tight_aggressive

# Compare posteriors under different opponent assumptions
python -m analysis.build_dataset logs/raw.jsonl logs/vs_tight.jsonl --opponent tight_passive
python -m analysis.build_dataset logs/raw.jsonl logs/vs_loose.jsonl --opponent loose_aggressive
```

## Use the Environment Programmatically

```python
from poker_env import PokerKitEnv, Action, ActionType

# Create 4-player environment
env = PokerKitEnv(
    num_players=4,
    stacks=(200, 200, 200, 200),
    blinds=(1, 2),
    small_bet=2,
    big_bet=4,
)

# Reset for a new hand
obs = env.reset(seed=42)

# Game loop
done = False
while not done:
    # Get legal actions
    legal_actions = env.legal_actions()
    print(f"Player {env.current_player()} to act: {[a.type.value for a in legal_actions]}")
    
    # Select an action (e.g., always check/call)
    action = next(a for a in legal_actions if a.type == ActionType.CHECK_OR_CALL)
    
    # Apply action
    obs, reward, done, info = env.step(action)

print(f"Hand complete. Final stacks: {info['final_stacks']}")
print(f"Deltas: {info['deltas']}")
```

### With explicit cards (for deterministic replay)

```python
# 2 players with explicit hole cards
obs = env.reset(
    seed=42,
    hole_cards=["AcAs", "KhKd"],  # Player 0, Player 1
    board="Jc3d5c4h9s",           # Predetermined board
)

# 4 players, some explicit, some random (None)
obs = env.reset(
    seed=42,
    hole_cards=["AcAs", "KhKd", None, "QsQc"],
)
```

## Project Structure

```
poker_env/
├── __init__.py          # Package exports
├── env.py               # PokerKitEnv - main environment class (2-6 players)
├── actions.py           # Action types and PokerKit mapping
├── obs.py               # Observation dataclass
├── deck.py              # Deterministic deck handling
├── agents/
│   ├── base.py          # BaseAgent interface
│   ├── random_agent.py  # Random action selection
│   └── call_agent.py    # Always check/call
├── oracle/
│   └── win_prob.py      # Monte Carlo win probability (multi-way)
├── logging/
│   └── decision_logger.py
└── tests/
    ├── test_env.py         # Basic + multi-way tests
    ├── test_determinism.py # Reproducibility tests
    └── test_golden.py      # Known-outcome tests
```

## Core API

### PokerKitEnv

```python
class PokerKitEnv:
    def __init__(
        self,
        num_players: int = 2,           # 2-6 players
        stacks: tuple[int, ...] = None, # Default: 200 each
        blinds: tuple[int, int] = (1, 2),
        small_bet: int = 2,
        big_bet: int = 4,
    ): ...
    
    def reset(self, seed, hole_cards=None, board=None) -> Obs
    def step(self, action) -> tuple[Obs, float, bool, dict]
    def current_player(self) -> int
    def legal_actions(self) -> list[Action]
    def get_obs(self, player_index) -> Obs
    def get_hidden_state() -> dict
```

### Action Types

| Action | Description |
|--------|-------------|
| `ActionType.FOLD` | Fold the hand, forfeit any chips in the pot |
| `ActionType.CHECK_OR_CALL` | Check (if no bet to call) or call (match the current bet) |
| `ActionType.BET_OR_RAISE` | Bet (if no bet yet) or raise (increase the current bet) |

### Observation Fields

| Field | Type | Description |
|-------|------|-------------|
| `hand_id` | `str` | Unique identifier for this hand |
| `seed` | `int` | Random seed used for this hand |
| `player_index` | `int` | Which player this observation is for |
| `num_players` | `int` | Total number of players in the game |
| `street` | `str` | Current betting round: `PREFLOP`, `FLOP`, `TURN`, or `RIVER` |
| `board` | `list[str]` | Community cards, e.g., `["Jc", "3d", "5c"]` |
| `hero_hole` | `list[str]` | This player's hole cards, e.g., `["Ac", "As"]` |
| `stacks` | `list[int]` | Current chip stacks for all players |
| `pot_total` | `int` | Total chips in the pot |
| `to_act` | `int` | Index of player who must act |
| `legal_actions` | `list[Action]` | Actions available to the player |
| `history` | `list[dict]` | Serialized action history |

## Oracle

The `WinProbOracle` computes ground-truth win/tie/lose probabilities against multiple opponents:

```python
from poker_env.oracle import WinProbOracle

oracle = WinProbOracle(num_samples=10000)

# Multi-way pot
probs = oracle.compute(
    hero_hole=["Ac", "As"],
    opponent_holes=[["Kh", "Kd"], ["Qc", "Qd"]],  # 2 opponents
    board=["Jc", "3d", "5c"],
)
# {"p_win": 0.75, "p_tie": 0.01, "p_lose": 0.24}
```

## Logging

Decision logs are written in JSONL format (one JSON object per line):

```json
{
  "hand_id": "abc123",
  "seed": 42,
  "decision_idx": 3,
  "player_to_act": 0,
  "street": "FLOP",
  "obs": {...},
  "hidden": {"player0_hole": ["Ac","As"], "player1_hole": ["Kh","Kd"], ...},
  "legal_actions": [...],
  "agent_belief": {...},
  "agent_action": "BET_OR_RAISE",
  "oracle_truth": {...}
}
```

## Running Tests

```bash
pytest poker_env/tests/ -v
```

## Creating Custom Agents

```python
from poker_env.agents import BaseAgent
from poker_env.actions import Action, ActionType
from poker_env.obs import Obs

class MyAgent(BaseAgent):
    def act(self, obs: Obs) -> Action:
        # Your logic here
        return obs.legal_actions[0]
    
    def belief(self, obs: Obs) -> dict | None:
        # Optional: return belief about game state
        return {
            "p_win": 0.5,
            "p_tie": 0.1,
            "p_lose": 0.4,
        }
```

---

# Analysis System

The `analysis/` package provides the infrastructure for computing oracle posteriors, eliciting beliefs from LLMs, and measuring calibration/coherence.

## End-to-End Research Workflow

### Step 1: Run Experiments

Generate decision-point logs by running games:

```bash
# Basic experiment
python run_experiment.py --hands 1000 --seed 42 --out logs/raw.jsonl -v

# With LLM agent (when implemented)
python run_experiment.py --agent llm --hands 500 --out logs/llm_run.jsonl
```

### Step 2: Enrich with Oracle Posteriors

Add Bayesian posteriors to each decision point:

```bash
python -m analysis.build_dataset logs/raw.jsonl logs/enriched.jsonl --opponent default
```

**Opponent model presets:**

| Preset | Aggression | Fold Threshold | Bluff Freq | Description |
|--------|------------|----------------|------------|-------------|
| `default` | 0.4 | 0.3 | 0.1 | Balanced player |
| `tight_passive` | 0.2 | 0.4 | 0.02 | Folds often, rarely raises |
| `tight_aggressive` | 0.6 | 0.4 | 0.08 | Selective but aggressive |
| `loose_passive` | 0.2 | 0.2 | 0.05 | Plays many hands, just calls |
| `loose_aggressive` | 0.6 | 0.2 | 0.15 | Plays many hands, raises often |
| `informative_v2` | 0.85 | 0.55 | 0.02 | **Canonical** - max oracle separation |
| `informative` | 0.85 | 0.55 | 0.02 | Legacy alias for `informative_v2` |

> **Preset Versioning:** `informative` and `informative_v2` have identical parameters. Use `informative_v2` for new experiments. Phase 1A logs used `informative` (valid, same params).

### Step 3: Run Full Analysis

Analyze LLM beliefs against oracle posteriors:

```bash
# Single file
python -m analysis.analyze_beliefs logs/enriched.jsonl

# Multiple files (aggregated)
python -m analysis.analyze_beliefs logs/run1_enriched.jsonl logs/run2_enriched.jsonl

# Save JSON report
python -m analysis.analyze_beliefs logs/enriched.jsonl --json-out results.json
```

**Output includes:**

| Section | Description |
|---------|-------------|
| Validity Audit | Raw output stats (prob_sum, negatives, all-zeros) |
| JS Distance Analysis | LLM vs CardOnly vs StrategyAware (normalized) |
| L1 Distance Analysis | Scale-sensitive vs shape-only error separation |
| Action-Conditioning | Does LLM shift beliefs after opponent aggression? |
| Summary Table | Paper-ready metrics |

**Key metrics:**

| Metric | Interpretation |
|--------|----------------|
| JS(LLM, CardOnly) | Distance to combo-counting baseline |
| JS(LLM, StrategyAware) | Distance to Bayesian posterior |
| JS(CardOnly, StrategyAware) | Oracle separation (test validity, should be >0.05) |
| LLM closer to | CardOnly = ignores history, StrategyAware = uses history |
| L1 (clipped, unnorm) | Scale-sensitive distance (captures sum inflation) |
| L1 (normalized) | Shape-only distance (isolates distributional error) |
| Strong-shift ratio | LLM response to aggression / Oracle response |

### Step 4: Paper-Ready Analysis (Phase 3)

For publication-grade metrics with bootstrap confidence intervals:

```bash
# PCE Distribution with Bootstrap CIs
python -m analysis.compute_pce_distribution \
    logs/enriched1.jsonl logs/enriched2.jsonl \
    --output-records results/pce_distribution.csv \
    --output-summary results/pce_summary.csv \
    --bootstrap 2000

# Update Coherence (separates card reveals from opponent actions)
python -m analysis.compute_update_coherence \
    logs/enriched1.jsonl logs/enriched2.jsonl \
    --output results/update_coherence.csv \
    --output-summary results/update_coherence_summary.json

# Generate Paper Figures
python -m analysis.plot_paper_figures \
    --pce-data results/pce_distribution.csv \
    --pce-summary results/pce_summary.csv \
    --update-data results/update_coherence.csv \
    --output-dir figures/
```

**Phase 3 output files:**

| File | Description |
|------|-------------|
| `results/pce_distribution.csv` | Per-record PCE with slicing variables (N records) |
| `results/pce_summary.csv` | Aggregated stats with bootstrap CIs by group |
| `results/update_coherence.csv` | Per-update magnitude, correlation, direction |
| `results/update_coherence_summary.json` | Summary by CARD_REVEAL vs ACTION |
| `figures/fig1_pce_cdf.png` | CDF of JS distances (main result figure) |
| `figures/fig2_baserate_neglect.png` | Trash mass comparison bar chart |
| `figures/fig3_update_scatter.png` | Update magnitude scatter (2 panels) |
| `figures/fig4_street_stability.png` | JS distance by street |

### Step 4: Compute Metrics (Programmatic)

```python
from analysis.build_dataset import load_analysis_dataset, extract_beliefs_and_oracles
from analysis.metrics import compute_pce, compute_coherence_summary

# Load enriched data
dataset = load_analysis_dataset("logs/enriched.jsonl")
agent_beliefs, card_only, strategy_aware = extract_beliefs_and_oracles(dataset)

# Compute calibration
pce_result = compute_pce(agent_beliefs, strategy_aware, method="js")
print(f"PCE (JS divergence): {pce_result['pce']:.4f}")

# Compute coherence
coherence = compute_coherence_summary(agent_beliefs)
print(f"Coherence rate: {coherence['coherence_rate']:.1%}")
```

## Analysis Components

### Hand Buckets (with Ablation Hooks)

Instead of tracking all 1,326 possible opponent hands, we group them into semantic buckets.

**Multiple schemes available for ablation studies:**

| Scheme | Buckets | Use Case |
|--------|---------|----------|
| `default` / `buckets_14` | 14 | Recommended for main results |
| `coarse` / `buckets_7` | 7 | Quick analysis, robustness check |
| `fine` / `buckets_30` | 30 | Detailed analysis, sensitivity check |

```python
from analysis.buckets import get_bucket_scheme

scheme = get_bucket_scheme("default")  # or "coarse", "fine"
print(scheme.bucket_names)  # List of bucket names
print(scheme.description)   # Human-readable description
```

**Default 14-bucket scheme:**

| Bucket | Hands | Example |
|--------|-------|---------|
| `premium_pairs` | AA, KK, QQ | Monsters |
| `strong_pairs` | JJ, TT | Very strong |
| `medium_pairs` | 99-66 | Playable pairs |
| `small_pairs` | 55-22 | Set mining |
| `premium_broadway` | AKs, AKo, AQs | Top unpaired |
| `strong_broadway` | AQo, AJs, KQs, ATs | Strong unpaired |
| `medium_broadway` | KQo, KJs, QJs, etc. | Playable broadway |
| `suited_aces` | A9s-A2s | Flush potential |
| `suited_connectors` | T9s-54s | Straight+flush draws |
| `suited_gappers` | J9s, T8s, etc. | One-gap suited |
| `offsuit_connectors` | T9o-65o | Straight potential |
| `weak_broadway` | KTo, QTo, etc. | Marginal broadway |
| `speculative_suited` | Small suited cards | Draw hands |
| `trash` | Everything else | Weak hands |

```python
from analysis import hand_to_bucket, get_bucket_prior

# Map a hand to its bucket
bucket = hand_to_bucket(["Ah", "As"])  # "premium_pairs"

# Get prior probability distribution (uniform over valid hands)
prior = get_bucket_prior(blockers=["Kh", "Kd"])  # Accounts for blockers
```

### Posterior Oracles

Two Bayesian oracles compute P(opponent_hand | public_info):

#### CardOnlyPosterior (Oracle A)

Uniform distribution over hands consistent with blockers. **Ignores betting history.**

```python
from analysis import CardOnlyPosterior

oracle = CardOnlyPosterior()
posterior = oracle.compute(
    hero_hole=["Ah", "Kd"],
    board=["Qc", "Jh", "2s"],
)
# {"premium_pairs": 0.05, "strong_pairs": 0.08, ...}
```

**Use for:** Baseline comparison. Measures how much action history should help.

#### StrategyAwarePosterior (Oracle B)

**Main oracle.** Incorporates betting history via an opponent model:

```
P(hand | history) ∝ P(hand) × ∏ P(action_t | hand, state_t)
```

```python
from analysis import StrategyAwarePosterior, ParametricOpponent

opponent_model = ParametricOpponent.from_preset("tight_aggressive")
oracle = StrategyAwarePosterior(opponent_model)

posterior = oracle.compute(
    hero_hole=["Ah", "Kd"],
    board=["Qc", "Jh", "2s"],
    opponent_actions=[
        {"action": "BET_OR_RAISE", "street": "PREFLOP"},
        {"action": "BET_OR_RAISE", "street": "FLOP"},
    ],
)
# Premium hands now more likely due to aggressive actions
```

**Key insight:** There is no "true" posterior without an opponent model. The posterior is Bayes-optimal *relative to* the assumed opponent behavior. This is scientifically honest and enables robustness analysis.

### Robustness to Opponent Assumptions

Since posteriors depend on opponent model, always test robustness:

```python
from analysis import StrategyAwarePosterior, ParametricOpponent

# Test same decision under different opponent assumptions
presets = ["tight_passive", "tight_aggressive", "loose_passive", "loose_aggressive"]

posteriors = {}
for preset in presets:
    model = ParametricOpponent.from_preset(preset)
    oracle = StrategyAwarePosterior(model)
    posteriors[preset] = oracle.compute(hero_hole, board, opponent_actions)

# Compare: Are conclusions stable?
from analysis.metrics import compute_js_divergence
for p1 in presets:
    for p2 in presets:
        if p1 < p2:
            js = compute_js_divergence(posteriors[p1], posteriors[p2])
            print(f"{p1} vs {p2}: JS = {js:.4f}")
```

**Report in paper:** "Results shown with default opponent model; robustness analysis across 5 presets in Appendix."

### Opponent Models

```python
from analysis import ParametricOpponent, ActionType, PublicState

# Create from preset
opponent = ParametricOpponent.from_preset("tight_aggressive")

# Or customize
opponent = ParametricOpponent(
    aggression=0.6,      # P(raise | should_continue)
    fold_threshold=0.4,  # Fold if hand_strength < threshold
    bluff_freq=0.08,     # P(raise | weak hand)
)

# Query action probabilities
state = PublicState(
    street="FLOP",
    board=["Qh", "7d", "2c"],
    pot=10,
    bet_to_call=2,
    num_raises_this_street=0,
    history=[],
)

p_raise = opponent.action_prob(
    hand=("Ah", "As"),
    state=state,
    action=ActionType.BET_RAISE,
)  # High probability with aces
```

### Belief Elicitation

Prompt templates for extracting structured beliefs from LLMs:

```python
from analysis.prompts import format_belief_prompt

prompt = format_belief_prompt(
    hero_hole=["Ah", "Kd"],
    board=["Qc", "Jh", "2s"],
    pot=20,
    street="FLOP",
    history=[...],
    template="default",  # or "simple", "reasoning"
)
# Returns a prompt asking LLM for probability distribution over buckets
```

### Prompt Uniformity Testing

**Critical for validity:** If beliefs vary wildly with prompt phrasing, you're measuring the prompt, not the model.

```python
from analysis.uniformity import (
    generate_prompt_variants,
    test_uniformity,
    compute_uniformity_summary,
)

# Generate 5 equivalent prompts for same game state
variants = generate_prompt_variants(
    hero_hole=["Ah", "Kd"],
    board=["Qc", "Jh", "2s"],
    pot=20,
    street="FLOP",
    history=[...],
)

# Test uniformity (requires LLM query function)
def query_llm(prompt: str) -> dict[str, float]:
    # Your LLM call here
    return belief_dict

result = test_uniformity(
    game_state={"hero_hole": ["Ah", "Kd"], "board": ["Qc", "Jh", "2s"], ...},
    llm_query_fn=query_llm,
)

print(f"JS mean across variants: {result.js_mean:.4f}")
print(f"JS stddev: {result.js_stddev:.4f}")
# Good uniformity: js_mean < 0.1
# Acceptable: js_mean < 0.2
```

**Prompt variants generated:**
- `default`: Standard structured prompt
- `simple`: Minimal prompt  
- `reasoning`: Chain-of-thought prompt
- `direct`: Direct question format
- `bayesian`: Bayesian framing

**Always log `prompt_template_id` and `prompt_hash`** to enable uniformity analysis.

### Simplex Projection

LLMs often output incoherent probabilities. Project them to valid distributions:

```python
from analysis import project_to_simplex, repair_belief

# Fix invalid distribution
raw_belief = {"premium_pairs": 0.3, "strong_pairs": -0.1, "trash": 0.5}  # Invalid!
fixed, info = repair_belief(raw_belief)
# fixed: valid distribution on simplex
# info: {"was_valid": False, "repair_distance_l2": 0.15, ...}
```

## Metrics Reference

### Posterior Calibration Error (PCE)

Measures average divergence between LLM beliefs and oracle posteriors:

```python
from analysis.metrics import compute_pce

result = compute_pce(
    llm_beliefs=llm_beliefs,      # List of belief dicts
    oracle_beliefs=oracle_beliefs, # List of oracle dicts
    method="kl",  # "kl", "js", "l2", "tv"
)
# result["pce"] = average divergence
# result["per_sample"] = list of per-decision divergences
```

| Method | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| `kl` | KL(oracle \|\| llm) | [0, ∞) | Information lost using LLM instead of oracle |
| `js` | JS(oracle, llm) | [0, 0.69] | Symmetric divergence |
| `l2` | √Σ(p-q)² | [0, √2] | Euclidean distance |
| `tv` | 0.5×Σ\|p-q\| | [0, 1] | Total variation |

### Coherence Metrics

Check probability axiom violations:

```python
from analysis.metrics import check_coherence, compute_coherence_summary

# Single belief
result = check_coherence(belief)
# result["is_coherent"] = True/False
# result["sum_violation"] = |sum - 1|
# result["negative_mass"] = sum of negative probs
# result["repair_distance"] = L2 distance to valid distribution

# Aggregate over many beliefs
summary = compute_coherence_summary(all_beliefs)
# summary["coherence_rate"] = fraction that pass all checks
# summary["avg_repair_distance"] = how much repair is needed on average
```

### Update Coherence

Measure whether LLM beliefs update Bayesianly:

```python
from analysis.metrics import compute_update_agreement

result = compute_update_agreement(
    llm_beliefs_over_time=[b1, b2, b3],    # LLM beliefs at t=1,2,3
    oracle_beliefs_over_time=[o1, o2, o3], # Oracle beliefs at t=1,2,3
)
# result["update_correlation"] = correlation of belief changes
# result["avg_direction_agreement"] = do they update same direction?
# result["update_magnitude_ratio"] = LLM update size / oracle update size
```

### Belief-Action Divergence

Compare stated beliefs to action-implied beliefs:

```python
from analysis.metrics import compute_belief_action_divergence, compute_decision_regret
from analysis.implied_belief import MonteCarloQValue, infer_implied_belief

# Compute Q-values for each action given opponent bucket
qv = MonteCarloQValue(num_rollouts=500)
q_by_bucket = qv.compute_q_values_by_bucket(
    hero_hole=["Ah", "Kd"],
    board=["Qc", "Jh", "2s"],
    pot=10,
    bet_to_call=2,
    street="FLOP",
)

# Infer what belief would make the chosen action optimal
implied = infer_implied_belief(
    chosen_action=ActionType.BET_RAISE,
    q_values=q_by_bucket,
    method="softmax",
)

# Compare stated vs implied
divergence = compute_belief_action_divergence(stated_belief, implied)

# Compute regret under oracle belief
regret = compute_decision_regret(
    chosen_action=ActionType.BET_RAISE,
    oracle_belief=oracle_posterior,
    q_values=q_by_bucket,
)
```

### "Plays Well Despite Bad Beliefs" Analysis

The key phenomenon this research investigates:

```python
from analysis.metrics.belief_action import identify_plays_well_despite_bad_beliefs

result = identify_plays_well_despite_bad_beliefs(
    stated_beliefs=llm_beliefs,
    oracle_beliefs=oracle_posteriors,
    chosen_actions=actions,
    q_values_list=all_q_values,
    belief_threshold=0.3,  # JS divergence for "bad beliefs"
    regret_threshold=0.5,  # Regret for "good play"
)

# result["plays_well_despite_bad_beliefs_rate"] = THE KEY METRIC
# result["good_play_given_bad_beliefs"] = P(good play | bad beliefs)
# result["good_play_given_good_beliefs"] = P(good play | good beliefs)
```

## Configurable Variables Summary

### Experiment Configuration

| Variable | CLI Flag | Default | Description |
|----------|----------|---------|-------------|
| Number of players | `--num-players` | 2 | 2-6 players (heads-up recommended for belief research) |
| Number of hands | `--hands` | 100 | Sample size |
| Random seed | `--seed` | 42 | For reproducibility |
| Agent types | `--agent`, `--agents` | random | Agent behavior |
| Compute oracle | `--no-oracle` | False (oracle ON) | Skip equity computation for speed |

### Dataset Builder Configuration

| Variable | CLI Flag | Default | Description |
|----------|----------|---------|-------------|
| Opponent model | `--opponent` | default | Which behavioral model for posteriors |

### Analysis Configuration (in code)

| Variable | Where | Options | Description |
|----------|-------|---------|-------------|
| Divergence method | `compute_pce()` | kl, js, l2, tv | How to measure belief mismatch |
| Opponent preset | `ParametricOpponent` | 5 presets | Assumed opponent behavior |
| Implied belief method | `infer_implied_belief()` | softmax, optimal_set | How to invert decisions |
| Q-value rollouts | `MonteCarloQValue` | int | Accuracy vs speed tradeoff |
| Belief threshold | Metrics | float | What counts as "bad" belief |
| Regret threshold | Metrics | float | What counts as "good" play |

### Why Change These Variables?

**Opponent model preset:**
- Different assumptions yield different "ground truth" posteriors
- Robustness: Do conclusions hold across opponent models?
- Matching: If you know opponent plays tight, use tight model

**Divergence method:**
- KL: Standard for information theory, but asymmetric
- JS: Symmetric, bounded—better for comparing across conditions
- L2: Simple, interpretable as distance
- TV: Interpretable as "probability mass in wrong place"

**Number of players:**
- 2 (heads-up): Simplest, cleanest for belief modeling
- 3-6: Multi-way adds complexity (multiple opponent posteriors)

**Seed:**
- Fixed seed = exact reproducibility
- Different seeds = robustness across card distributions

## Project Structure (Full)

```
poker_env/                    # Core poker environment
├── __init__.py
├── env.py                    # PokerKitEnv (2-6 players)
├── actions.py                # Action types
├── obs.py                    # Observation dataclass
├── deck.py                   # Deterministic dealing
├── config.py                 # Centralized model configuration
├── agents/
│   ├── base.py               # BaseAgent interface
│   ├── random_agent.py
│   ├── call_agent.py
│   ├── threshold_agent.py    # Hand-strength based (for belief experiments)
│   └── hf_agent.py           # HuggingFace LLM agent
├── oracle/
│   └── win_prob.py           # EquityOracle (ground truth equity)
├── logging/
│   └── decision_logger.py    # JSONL logging
└── tests/

analysis/                     # Belief analysis system
├── __init__.py               # Main exports
├── buckets.py                # Bucket schemes (14/7/30) + ablation hooks
├── opponent_model.py         # ParametricOpponent with presets
├── posterior_oracle.py       # CardOnlyPosterior, StrategyAwarePosterior
├── belief_schema.py          # BeliefOutput dataclass
├── belief_utils.py           # Belief conversion utilities
├── projection.py             # Simplex projection
├── prompts.py                # LLM prompt templates
├── uniformity.py             # Prompt invariance testing
├── build_dataset.py          # CLI for enriching logs with oracles
├── analyze_beliefs.py        # CLI for JS/L1 analysis + action-conditioning
├── compute_pce_distribution.py  # PCE by street/action with bootstrap CIs
├── compute_update_coherence.py  # Card vs action update separation
├── plot_paper_figures.py     # Generate paper-ready figures
├── metrics/
│   ├── calibration.py        # PCE, KL, JS, Brier, ECE
│   ├── coherence.py          # Axiom violation checks
│   ├── update_coherence.py   # Bayesian update quality
│   └── belief_action.py      # Stated vs implied beliefs
├── implied_belief/
│   ├── q_value.py            # Monte Carlo Q estimation
│   └── inverse.py            # Inverse decision rule
└── tests/
    ├── test_buckets.py
    ├── test_posteriors.py
    ├── test_metrics.py
    └── test_q_values.py

run_experiment.py             # Main experiment script

logs/                         # Experiment data (generated)
├── sanity_*.jsonl            # Phase 1A raw logs
├── *_enriched.jsonl          # Logs with oracle posteriors
├── phase2_*.jsonl            # Phase 2 scale-up logs
└── *.json                    # Analysis summaries

results/                      # Analysis outputs (generated)
├── pce_distribution.csv      # Per-record PCE metrics
├── pce_summary.csv           # Aggregated with bootstrap CIs
├── update_coherence.csv      # Per-update metrics
├── update_coherence_summary.json
└── combined_analysis.json    # Full analysis with L1 metrics

figures/                      # Paper figures (generated)
├── fig1_pce_cdf.png/pdf      # Main result: JS distance CDF
├── fig2_baserate_neglect.png/pdf  # Trash mass comparison
├── fig3_update_scatter.png/pdf    # Update magnitude scatter
└── fig4_street_stability.png/pdf  # Effect by street
```

## Hand Manifests (Reproducibility)

Every hand can be exported as a manifest for exact replay:

```python
from poker_env import PokerKitEnv

env = PokerKitEnv()
obs = env.reset(seed=42)

# Play some actions...
env.step(action1)
env.step(action2)

# Export manifest
manifest = env.export_manifest()
print(manifest.to_json())
# {
#   "seed": 42,
#   "deck_order": ["Ah", "Kd", ...],
#   "hole_cards": ["AhKd", "QcQs"],
#   "board_cards": ["Jc", "3d", "5c", "8h", "2s"],
#   "actions": [{"player": 0, "action": "BET_OR_RAISE"}, ...],
#   "env_config": {...},
#   "env_config_hash": "abc123..."
# }

# Replay exact same hand
env2 = PokerKitEnv()
env2.reset_from_manifest(manifest)
```

**Manifests enable:**
- Exact reproduction of any hand
- Debugging specific decision points
- Sharing reproducible examples for papers

## Running All Tests

```bash
# All tests (106 total)
pytest -v

# Just poker environment
pytest poker_env/tests/ -v

# Just analysis
pytest analysis/tests/ -v
```

---

## Complete Research Workflow (What We Actually Did)

This section documents the complete workflow from initial setup to paper-ready figures.

### Phase 1: Environment Setup

```bash
# 1. Clone and setup
git clone https://github.com/harryila/poker2026.git
cd poker2026
./setup.sh --with-gpu --hf-token YOUR_TOKEN

# 2. Verify GPU (NVIDIA GH200 480GB used)
nvidia-smi

# 3. Activate environment
source venv/bin/activate
```

### Phase 2: Initial Discovery

**Original plan:** Use `call` opponent, expect LLM to be closer to StrategyAware.

**What happened:**
```bash
# First attempt with call opponent (INCONCLUSIVE)
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent call \
    --hands 50 \
    --seed 42 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/sanity_70b_t0_s42.jsonl \
    -v

# Result: JS(CardOnly, StrategyAware) = 0.0147 — too low to test anything!
# The call opponent provides no betting signal.
```

**Solution:** Created `ThresholdAgent` with `informative_v2` preset to achieve JS(CardOnly, StrategyAware) ≈ 0.05-0.06.

### Phase 3: Phase 1A Sanity Grid (4 runs)

```bash
# Run grid: 2 temps × 2 seeds × 50 hands
for temp in 0.0 0.2; do
    for seed in 42 123; do
        python run_experiment.py \
            --agent hf \
            --hf-model meta-llama/Llama-3.1-70B-Instruct \
            --opponent threshold \
            --opponent-preset informative_v2 \
            --hands 50 \
            --seed $seed \
            --temperature $temp \
            --elicit-beliefs \
            --out logs/sanity_70b_t${temp/./}_s${seed}_informative.jsonl \
            -v
    done
done

# Enrich with oracle posteriors
for f in logs/sanity_70b_*_informative.jsonl; do
    python -m analysis.build_dataset $f ${f%.jsonl}_enriched.jsonl --opponent informative_v2
done

# Analyze Phase 1A
python -m analysis.analyze_beliefs logs/*_informative_enriched.jsonl \
    --json-out logs/phase1a_complete.json
```

**Phase 1A Result (N=246):** LLM closer to CardOnly by 0.0170. Negative result — LLM ignores betting history.

### Phase 4: Phase 2 Scale-Up (Partial)

**Original plan:** 6 runs × 1000 hands = ~200 GPU hours.

**What we did:** Stopped after 2 runs (12+ hours) when results replicated Phase 1A.

```bash
# Phase 2 runs (in tmux for persistence)
tmux new -s phase2

python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative_v2 \
    --hands 1000 \
    --seed 42 \
    --temperature 0.0 \
    --elicit-beliefs \
    --out logs/phase2_70b_t0_s42_informative_v2.jsonl \
    -v
# Completed 366 of 1000 hands before stopping

python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent threshold \
    --opponent-preset informative_v2 \
    --hands 1000 \
    --seed 42 \
    --temperature 0.2 \
    --elicit-beliefs \
    --out logs/phase2_70b_t02_s42_informative_v2.jsonl \
    -v
# Completed 326 of 1000 hands before stopping
```

**Phase 2 Result (N=838):** Virtually identical to Phase 1A. Effect replicated with 3.4x more data.

### Phase 5: Phase 3 Paper Analysis

```bash
# Combine all enriched data (Phase 1A + Phase 2)
ALL_FILES="logs/sanity_70b_t0_s42_informative_enriched.jsonl \
logs/sanity_70b_t0_s123_informative_enriched.jsonl \
logs/sanity_70b_t02_s42_informative_enriched.jsonl \
logs/sanity_70b_t02_s123_informative_enriched.jsonl \
logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl \
logs/phase2_70b_t02_s42_informative_v2_enriched.jsonl"

# Full analysis with L1 metrics
python -m analysis.analyze_beliefs $ALL_FILES \
    --json-out results/combined_analysis.json

# PCE Distribution with Bootstrap CIs
python -m analysis.compute_pce_distribution $ALL_FILES \
    --output-records results/pce_distribution.csv \
    --output-summary results/pce_summary.csv \
    --bootstrap 2000

# Update Coherence Analysis
python -m analysis.compute_update_coherence $ALL_FILES \
    --output results/update_coherence.csv \
    --output-summary results/update_coherence_summary.json

# Generate Paper Figures
python -m analysis.plot_paper_figures \
    --pce-data results/pce_distribution.csv \
    --pce-summary results/pce_summary.csv \
    --update-data results/update_coherence.csv \
    --output-dir figures/
```

### Final Output Files

| File | Size | Description |
|------|------|-------------|
| `results/pce_distribution.csv` | 250 KB | 1,084 per-record PCE values |
| `results/pce_summary.csv` | 3 KB | Aggregated with bootstrap CIs |
| `results/update_coherence.csv` | 57 KB | 318 update metrics |
| `figures/fig1_pce_cdf.png` | 180 KB | Main result figure |
| `figures/fig2_baserate_neglect.png` | 155 KB | Trash mass comparison |
| `figures/fig3_update_scatter.png` | 410 KB | Update diagnosis (2 panels) |
| `figures/fig4_street_stability.png` | 185 KB | Effect by street |

---

## What We Learned (Research Lessons)

1. **Opponent choice is critical:** A `call` opponent provides no signal for testing action-conditioning. Always validate JS(CardOnly, StrategyAware) > 0.05 before running experiments.

2. **Model size matters:** Llama 3.1 8B outputs degenerate beliefs (100% trash) for every game state. Belief elicitation requires models above a capability threshold.

3. **Don't trust surface coherence:** The 8B model had "perfect" prob_sum=1.0 while outputting nonsense. Always compare to Bayesian baselines.

4. **Negative results are publishable:** "LLM ignores betting history" is a valid finding. Scale up to characterize the failure mode, don't stop.

5. **Analysis > Data:** After replicating the effect, deeper analysis (L1 metrics, update coherence) yields more insight than more data.

6. **Parse rate ~50% is acceptable:** LLM belief elicitation will have failures. Filter and proceed; sufficient valid samples remain.

---

## License

MIT
