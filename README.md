# Poker Environment for LLM Belief Modeling Research

A reproducible, instrumented Heads-Up Fixed-Limit Texas Hold'em environment built on [PokerKit](https://github.com/uoftcprg/pokerkit) for research on belief modeling in LLM poker agents.

## Requirements

- **Python >= 3.11** (required by PokerKit)
- PokerKit >= 0.6.0

## Installation

```bash
# Create virtual environment with Python 3.11+
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run an experiment

```bash
python run_experiment.py --agent random --opponent call --hands 100 --seed 42 --out logs/test.jsonl -v
```

### Use the environment programmatically

```python
from poker_env import PokerKitEnv, Action, ActionType

# Create environment
env = PokerKitEnv(
    stacks=(200, 200),
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
```

### With explicit cards (for deterministic replay)

```python
obs = env.reset(
    seed=42,
    hero_hole="AcAs",      # Player 0 gets pocket aces
    villain_hole="KhKd",   # Player 1 gets pocket kings
    board="Jc3d5c4h9s",    # Predetermined board
)
```

## Project Structure

```
poker_env/
├── __init__.py          # Package exports
├── env.py               # PokerKitEnv - main environment class
├── actions.py           # Action types and PokerKit mapping
├── obs.py               # Observation dataclass
├── deck.py              # Deterministic deck handling
├── agents/
│   ├── base.py          # BaseAgent interface
│   ├── random_agent.py  # Random action selection
│   └── call_agent.py    # Always check/call
├── oracle/
│   └── win_prob.py      # Monte Carlo win probability
├── logging/
│   └── decision_logger.py  # JSONL decision logging
└── tests/
    ├── test_env.py         # Basic environment tests
    ├── test_determinism.py # Reproducibility tests
    └── test_golden.py      # Known-outcome tests
```

## Core API

### PokerKitEnv

```python
class PokerKitEnv:
    def reset(self, seed, hero_hole=None, villain_hole=None, board=None) -> Obs
    def step(self, action) -> tuple[Obs, float, bool, dict]
    def current_player(self) -> int
    def legal_actions(self) -> list[Action]
    def get_obs(self, player_index) -> Obs
    def get_hidden_state(self) -> dict
```

### Action Types

- `ActionType.FOLD` - Fold the hand
- `ActionType.CHECK_OR_CALL` - Check (if no bet) or call (if facing bet)
- `ActionType.BET_OR_RAISE` - Bet (if no bet) or raise (if facing bet)

### Observation

```python
@dataclass
class Obs:
    hand_id: str           # Unique hand identifier
    seed: int              # Random seed
    player_index: int      # Which player this obs is for
    street: str            # PREFLOP, FLOP, TURN, RIVER
    board: list[str]       # Community cards
    hero_hole: list[str]   # Player's hole cards
    stacks: list[int]      # Current stack sizes
    pot_total: int         # Total pot
    to_act: int            # Player index to act
    legal_actions: list    # Legal actions
    history: list[dict]    # Action history
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
  "hidden": {"player0_hole": ["Ac","As"], "player1_hole": ["Kh","Kd"]},
  "legal_actions": ["CHECK_OR_CALL", "BET_OR_RAISE"],
  "agent_belief": null,
  "agent_action": "BET_OR_RAISE",
  "oracle_truth": {"p_win": 0.82, "p_tie": 0.01, "p_lose": 0.17}
}
```

## Oracle

The `WinProbOracle` computes ground-truth win/tie/lose probabilities:

```python
from poker_env.oracle import WinProbOracle

oracle = WinProbOracle(num_samples=10000)
probs = oracle.compute(
    hero_hole=["Ac", "As"],
    villain_hole=["Kh", "Kd"],
    board=["Jc", "3d", "5c"],
)
# {"p_win": 0.82, "p_tie": 0.01, "p_lose": 0.17}
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
