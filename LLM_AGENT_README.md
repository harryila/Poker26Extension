# LLM Agent Integration

This document describes the HuggingFace-based LLM agent integration for running poker experiments with Llama 3.1 models.

## Configuration

**All defaults are centralized in `poker_env/config.py`** - this is the single source of truth for model configuration.

```python
# poker_env/config.py

DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"      # Safe default for dev/CI
RESEARCH_MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"   # For research runs

DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_ACTION_MAX_TOKENS = 64       # Actions need few tokens
DEFAULT_BELIEF_MAX_TOKENS = 384      # Beliefs need more
DEFAULT_BELIEF_FORMAT = "compact"    # Fixed-order list format
BELIEF_SCHEMA_ID = "buckets_14_v1"   # Versioned schema
```

## Overview

The `HFAgent` class enables running poker experiments with large language models, featuring:

- **Strict JSON output enforcement** via Llama chat templates
- **Deterministic fallback** (CHECK_OR_CALL/FOLD) on parse failure
- **Full telemetry** for parse success, coherence diagnostics, reproducibility
- **Separate token budgets** for action vs belief generation
- **Compact belief format** (fixed-order list) for reliable parsing
- **Truncation telemetry** for history analysis
- **Centralized configuration** in `poker_env/config.py`

## Requirements

### Hardware

| Model | VRAM Required | Recommended GPU |
|-------|---------------|-----------------|
| Llama 3.1 8B (default) | ~16 GB | A100-40GB, A6000 |
| Llama 3.1 70B (research) | ~80 GB | GH200, H100, multi-GPU |

### Software
```bash
# Core dependencies (in requirements.txt)
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
huggingface_hub>=0.19.0
```

### HuggingFace Access
1. Create account at https://huggingface.co
2. Request access to `meta-llama/Llama-3.1-8B-Instruct` (and 70B for research)
3. Create access token at https://huggingface.co/settings/tokens
4. Login: `huggingface-cli login` or via setup.sh

## Installation

```bash
# Run setup with GPU support and HF login
./setup.sh --with-gpu --hf-token YOUR_HF_TOKEN

# Or manually:
source venv/bin/activate
pip install torch transformers accelerate
huggingface-cli login
```

## Usage

### Development/CI (8B model, default)

```bash
# Basic run with 8B model (uses config defaults)
python run_experiment.py --agent hf --opponent call --hands 10 --no-oracle -v
```

### Research Run (70B model)

```bash
# Use 70B model for research
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent call \
    --hands 100 \
    --elicit-beliefs \
    -v
```

### Full Configuration

```bash
python run_experiment.py \
    --agent hf \
    --hf-model meta-llama/Llama-3.1-70B-Instruct \
    --opponent call \
    --hands 100 \
    --seed 42 \
    --temperature 0.2 \
    --top-p 0.9 \
    --action-max-new-tokens 64 \
    --belief-max-new-tokens 384 \
    --belief-format compact \
    --elicit-beliefs \
    --randomize-probe-order \
    --out logs/research_run.jsonl \
    -v
```

## Command Line Flags

### HF Agent Options

| Flag | Default | Description |
|------|---------|-------------|
| `--agent hf` | - | Use HuggingFace LLM agent |
| `--hf-model` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID |
| `--temperature` | `0.2` | Generation temperature (0 = deterministic) |
| `--top-p` | `0.9` | Top-p sampling parameter |
| `--action-max-new-tokens` | `64` | Max tokens for action generation |
| `--belief-max-new-tokens` | `384` | Max tokens for belief generation |
| `--max-input-tokens` | `2048` | Max input context length |
| `--belief-format` | `compact` | Belief output format ("compact" or "full") |
| `--elicit-beliefs` | `False` | Call `belief()` at each decision |
| `--randomize-probe-order` | `False` | Randomize action/belief probe order |
| `--i-know-what-im-doing` | `False` | Allow multi-way belief experiments (bypasses heads-up enforcement) |

### Heads-Up Enforcement for Belief Experiments

**Important:** When `--elicit-beliefs` is enabled with `--num-players > 2`, the system will block execution and show a warning:

```
WARNING: Multi-way belief experiments are not recommended!

Belief elicitation with >2 players introduces:
- Multiple opponent posteriors to track
- Exponentially complex action-likelihood modeling
- Side pot complications

For valid research results, use heads-up (2 players) for belief experiments.
To proceed anyway, add: --i-know-what-im-doing
```

This protects against accidentally contaminating research data with multi-way games.

### Temperature Guidelines

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| `0` | Deterministic (greedy) | Reproducible experiments |
| `0.2` | Low variance | Recommended default |
| `0.7` | Higher creativity | Exploration studies |

## Belief Formats

### Compact Format (Default)

Uses a fixed-order list with embedded schema for self-description:

```json
{"schema": "buckets_14_v1", "probs": [0.05, 0.08, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.08, 0.06, 0.05, 0.08, 0.07, 0.05]}
```

**Bucket order (indices 0-13):**

| Index | Bucket Name | Description |
|-------|-------------|-------------|
| 0 | premium_pairs | AA, KK, QQ |
| 1 | strong_pairs | JJ, TT |
| 2 | medium_pairs | 99-66 |
| 3 | small_pairs | 55-22 |
| 4 | premium_broadway | AKs, AKo, AQs |
| 5 | strong_broadway | AQo, AJs, KQs, ATs |
| 6 | medium_broadway | KQo, KJs, QJs, etc. |
| 7 | suited_aces | A9s-A2s |
| 8 | suited_connectors | T9s-54s |
| 9 | suited_gappers | J9s, T8s, etc. |
| 10 | offsuit_connectors | T9o-65o |
| 11 | weak_broadway | KTo, QTo, etc. |
| 12 | speculative_suited | Small suited cards |
| 13 | trash | Everything else |

### Full Format (for debugging)

Uses labeled dict:

```json
{"premium_pairs": 0.05, "strong_pairs": 0.08, "medium_pairs": 0.12, ...}
```

## Output Format

### Log Structure

Each decision point logs:

```json
{
  "hand_id": "abc123",
  "decision_idx": 3,
  "player_to_act": 0,
  "agent_action": "BET_OR_RAISE",
  "agent_belief": {"premium_pairs": 0.05, "trash": 0.30, ...},
  "action_metadata": {
    "parse_success": true,
    "fallback_used": false,
    "raw_response": "{\"action\": \"BET_OR_RAISE\"}",
    "action_chosen": "BET_OR_RAISE",
    "prompt_hash": "a1b2c3d4e5f6g7h8",
    "prompt_template_id": "action_strict_v1",
    "history_events_before": 15,
    "history_events_after": 15,
    "was_truncated": false
  },
  "belief_metadata": {
    "parse_success": true,
    "prob_sum": 1.0,
    "prob_min": 0.02,
    "negative_count": 0,
    "prompt_hash": "f9e8d7c6b5a43210",
    "prompt_template_id": "belief_compact_v1",
    "belief_format": "compact",
    "belief_schema_id": "buckets_14_v1",
    "history_events_before": 15,
    "history_events_after": 15,
    "was_truncated": false,
    "repair_distance_l1": 0.0,
    "repair_distance_l2": 0.0
  }
}
```

### Telemetry Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `action_parse_rate` | % of action responses parsed | >95% |
| `belief_parse_rate` | % of belief responses parsed | >80% |
| `fallback_rate` | % using deterministic fallback | <10% |
| `coherence_rate` | % of beliefs with valid probabilities | >70% |
| `avg_prob_sum` | Mean sum of belief probabilities | ~1.0 |
| `avg_repair_distance_l1` | Mean L1 distance to simplex | ~0.0 |
| `avg_repair_distance_l2` | Mean L2 distance to simplex | ~0.0 |

## Standardized Prompts

### Action System Message

```
You are a poker player. Choose your action.
Return ONLY valid JSON. No other text.
Format: {"action": "FOLD" | "CHECK_OR_CALL" | "BET_OR_RAISE"}
```

### Belief System Message (Compact)

```
You are analyzing opponent hand ranges in poker. Return ONLY valid JSON. No other text.
Output format: {"schema":"buckets_14_v1","probs":[p0,p1,...,p13]}
Constraints: each pi must be a decimal between 0 and 1, non-negative, and the 14 values must sum to 1.0 exactly (adjust the last value to fix rounding). Use at most 3 decimal places.
```

## Implementation Details

### do_sample Rule

The agent applies the standard HuggingFace rule:

```python
if self.temperature == 0:
    gen_kwargs["do_sample"] = False  # Deterministic
else:
    gen_kwargs["do_sample"] = True
    gen_kwargs["temperature"] = self.temperature
    gen_kwargs["top_p"] = self.top_p
```

### Fallback Behavior

On parse failure:
1. If `CHECK_OR_CALL` is legal → use it
2. Otherwise → `FOLD`

### History Truncation

To stay within token budget:
```python
# Keep first 5 (blinds/deals) + last (max_events - 5)
def _truncate_history(history, max_events=50):
    if len(history) <= max_events:
        return history
    return history[:5] + history[-(max_events - 5):]
```

Truncation is logged: `history_events_before`, `history_events_after`, `was_truncated`

### Belief Conversion Utilities

```python
from analysis.belief_utils import compact_to_dict, dict_to_compact, repair_belief

# Convert compact to labeled dict
belief, metadata = compact_to_dict([0.05, 0.08, ...])

# Convert dict to compact (with clipping/renormalization)
probs, metadata = dict_to_compact({"premium_pairs": 0.05, ...})

# Repair invalid distribution
repaired, metadata = repair_belief(invalid_belief)
```

## Files

| File | Description |
|------|-------------|
| `poker_env/config.py` | **Central configuration** - edit model defaults here |
| `poker_env/agents/hf_agent.py` | HFAgent class |
| `analysis/prompts.py` | Prompt templates |
| `analysis/belief_utils.py` | Belief conversion utilities |
| `run_experiment.py` | CLI interface |

## Smoke Tests

After implementation, run these tests:

### Test 1: Actions Only
```bash
python run_experiment.py --agent hf --opponent call --hands 5 --no-oracle -v
```
**Expect:** `action_parse_rate ~100%`, `fallback_rate ~0%`

### Test 2: Beliefs with Compact Format
```bash
python run_experiment.py --agent hf --opponent call --hands 5 \
  --elicit-beliefs --belief-format compact --belief-max-new-tokens 384 --no-oracle -v
```
**Expect:** `belief_parse_rate >80%`

### Test 3: Deterministic Mode
```bash
python run_experiment.py --agent hf --opponent call --hands 5 \
  --elicit-beliefs --temperature 0 --no-oracle -v
```
**Expect:** Parse rates stable, outputs consistent

### Test 4: Verify Schema in Logs
```bash
grep "buckets_14_v1" logs/experiment.jsonl | head -3
```
**Check:** Every belief record includes `"schema":"buckets_14_v1"`

## API Reference

### HFAgent

```python
from poker_env.agents import HFAgent

agent = HFAgent(
    model_id=None,                    # Uses config.DEFAULT_MODEL_ID
    temperature=None,                  # Uses config.DEFAULT_TEMPERATURE
    top_p=None,                       # Uses config.DEFAULT_TOP_P
    action_max_new_tokens=None,       # Uses config.DEFAULT_ACTION_MAX_TOKENS
    belief_max_new_tokens=None,       # Uses config.DEFAULT_BELIEF_MAX_TOKENS
    max_input_tokens=None,            # Uses config.DEFAULT_MAX_INPUT_TOKENS
    belief_format=None,               # Uses config.DEFAULT_BELIEF_FORMAT
    name="HFAgent",
)

# Get action
action = agent.act(obs)

# Get action with metadata (includes truncation info)
action, metadata = agent.act_with_metadata(obs)

# Get belief
belief = agent.belief(obs)

# Get belief with metadata (includes truncation info)
belief, metadata = agent.belief_with_metadata(obs)
```

### Metadata Classes

```python
@dataclass
class ActionMetadata:
    parse_success: bool
    fallback_used: bool
    raw_response: str
    action_chosen: str
    prompt_hash: str
    prompt_template_id: str
    history_events_before: int
    history_events_after: int
    was_truncated: bool

@dataclass
class BeliefMetadata:
    parse_success: bool
    raw_response: str
    prob_sum: float | None
    prob_min: float | None
    negative_count: int
    prompt_hash: str
    prompt_template_id: str
    belief_format: str
    belief_schema_id: str
    history_events_before: int
    history_events_after: int
    was_truncated: bool
    repair_distance_l1: float | None  # L1 distance to simplex (~0 for valid)
    repair_distance_l2: float | None  # L2 distance to simplex (~0 for valid)
```
