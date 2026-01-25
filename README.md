# Poker Environment for LLM Belief Modeling Research

A reproducible, instrumented Fixed-Limit Texas Hold'em environment built on [PokerKit](https://github.com/uoftcprg/pokerkit) for research on belief modeling in LLM poker agents. Supports 2-6 players.

## Requirements

- **Python >= 3.11** (required by PokerKit)
- PokerKit >= 0.6.0

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

## Command Line Reference

### All Flags Explained

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `--num-players` | Number of players in the game (2-6) | `2` | `--num-players 4` |
| `--agent` | Default agent type for all players | `random` | `--agent call` |
| `--agents` | Comma-separated list of agent types for each player. Overrides `--agent`. | None | `--agents random,call,random,call` |
| `--opponent` | Agent type for player 1 only (heads-up shortcut). Only works with 2 players. | None | `--opponent call` |
| `--hands` | Number of hands to play | `100` | `--hands 500` |
| `--seed` | Base random seed for reproducibility. Each hand uses `seed + hand_num * 1000`. | `42` | `--seed 12345` |
| `--out` | Output file path for JSONL decision logs | `logs/experiment.jsonl` | `--out results/run1.jsonl` |
| `--no-oracle` | Skip oracle (win probability) computation. Makes runs ~10x faster. | False | `--no-oracle` |
| `-v, --verbose` | Print progress every 10 hands and show summary at end | False | `-v` |

### Agent Types

| Type | Behavior |
|------|----------|
| `random` | Selects uniformly at random from legal actions |
| `call` | Always checks or calls (never folds, never raises) |

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

## License

MIT
