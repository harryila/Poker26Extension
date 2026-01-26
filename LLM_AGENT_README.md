# LLM Agent Integration (Llama 3.1 8B)

This document describes the HuggingFace-based LLM agent integration for running poker experiments with Llama 3.1 8B Instruct.

## Overview

The `HFAgent` class enables running poker experiments with large language models, featuring:

- **Strict JSON output enforcement** via Llama chat templates
- **Deterministic fallback** (CHECK_OR_CALL/FOLD) on parse failure
- **Full telemetry** for parse success, coherence diagnostics, reproducibility
- **Configurable generation** (temperature, token limits)
- **Belief elicitation** with bucket-level probability distributions

## Requirements

### Hardware
- **GPU**: 40GB+ VRAM recommended (A100, A6000)
- Llama 3.1 8B in bfloat16 uses ~16GB VRAM
- Leave headroom for KV cache during generation

### Software
```bash
# Core dependencies (in requirements.txt)
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
```

### HuggingFace Access
1. Create account at https://huggingface.co
2. Request access to `meta-llama/Llama-3.1-8B-Instruct`
3. Create access token at https://huggingface.co/settings/tokens
4. Login: `huggingface-cli login` or via Python (see setup.sh)

## Installation

```bash
# Run setup (includes HF login prompt)
./setup.sh

# Or manually:
source venv/bin/activate
pip install torch transformers accelerate
python -c "from huggingface_hub import login; login()"
```

## Usage

### Basic HF Agent Run

```bash
# HF agent vs random opponent, 5 hands
python run_experiment.py \
    --agent hf \
    --opponent random \
    --hands 5 \
    --seed 42 \
    --out logs/hf_run.jsonl \
    --no-oracle \
    -v
```

### With Belief Elicitation

```bash
# Elicit beliefs at each decision point
python run_experiment.py \
    --agent hf \
    --opponent call \
    --hands 10 \
    --elicit-beliefs \
    --out logs/hf_beliefs.jsonl \
    -v
```

### Full Research Configuration

```bash
python run_experiment.py \
    --agent hf \
    --opponent call \
    --hands 100 \
    --seed 42 \
    --elicit-beliefs \
    --temperature 0.2 \
    --max-new-tokens 256 \
    --max-input-tokens 2048 \
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
| `--max-new-tokens` | `128` | Max tokens to generate per response |
| `--max-input-tokens` | `2048` | Max input context length |
| `--elicit-beliefs` | `False` | Call `belief()` at each decision |
| `--randomize-probe-order` | `False` | Randomize action/belief probe order |

### Temperature Guidelines

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| `0` | Deterministic (greedy) | Reproducible experiments |
| `0.2` | Low variance | Recommended default |
| `0.7` | Higher creativity | Exploration studies |
| `1.0` | Full sampling | Maximum diversity |

### Token Limits

| Parameter | Recommendation | Notes |
|-----------|----------------|-------|
| `--max-new-tokens` | 128 for actions, 256+ for beliefs | Belief JSON needs more tokens |
| `--max-input-tokens` | 2048 | Truncates history if exceeded |

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
    "prompt_template_id": "action_strict_v1"
  },
  "belief_metadata": {
    "parse_success": true,
    "prob_sum": 1.02,
    "prob_min": 0.0,
    "negative_count": 0,
    "prompt_hash": "f9e8d7c6b5a43210",
    "prompt_template_id": "belief_strict_v1"
  },
  "prompt_template_id": "action_strict_v1",
  "prompt_hash": "a1b2c3d4e5f6g7h8",
  "probe_order": "action_first"
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

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      run_experiment.py                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ PokerKitEnv │───▶│   HFAgent   │───▶│ DecisionLogger  │  │
│  │             │    │             │    │                 │  │
│  │ - reset()   │    │ - act()     │    │ - log_decision()│  │
│  │ - step()    │    │ - belief()  │    │ - end_hand()    │  │
│  │ - get_obs() │    │             │    │                 │  │
│  └─────────────┘    └──────┬──────┘    └─────────────────┘  │
│                            │                                 │
│                     ┌──────▼──────┐                         │
│                     │ Llama 3.1   │                         │
│                     │ 8B Instruct │                         │
│                     │             │                         │
│                     │ bfloat16    │                         │
│                     │ device_map  │                         │
│                     └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### JSON Extraction (Robust)

The agent extracts JSON from potentially chatty responses:

```python
def _extract_json(text: str) -> dict | None:
    # 1. Strip code fences (```json ... ```)
    # 2. Find all {...} blocks
    # 3. Try each, shortest first (cleaner)
    # 4. Handle trailing commas
```

### Fallback Behavior

On parse failure:
1. If `CHECK_OR_CALL` is legal → use it
2. Otherwise → `FOLD`

This ensures experiments never crash and provides interpretable fallback.

### History Truncation

To stay within token budget:
```python
def _truncate_history(history, max_events=50):
    # Keep first 5 (blinds/deals) + last (max_events - 5)
```

### Prompt Hashing

Every prompt is hashed (SHA256, 16 chars) for reproducibility:
```python
prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
```

## Integration Test Results

Test run: HF agent vs call agent, 3 hands

```
=== Experiment Summary (2 players) ===
Hands played: 3
Config hash: 770d229a252c882d
Player 0 (Player0_hf): +X.0 chips
Player 1 (Player1_call): -X.0 chips
```

### Observed Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Action parse rate | 100% | Model returns clean JSON |
| Belief parse rate | ~50% | Needs `--max-new-tokens 256+` |
| Fallback rate | 0% | No fallbacks needed |
| Model load time | ~8s | First run, cached after |
| Inference time | ~0.5-1s/decision | With A100 GPU |

## Troubleshooting

### "CUDA out of memory"

Reduce batch size or use quantization:
```python
# In hf_agent.py, load with 8-bit quantization:
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
)
```

### "Token indices out of bounds"

Increase `--max-input-tokens` or the model will truncate important context.

### Low belief parse rate

Increase `--max-new-tokens 256` or `512` for full belief output.

### "Permission denied (publickey)" when pushing

The SSH key isn't set up. Either:
1. Add the generated SSH key to GitHub settings
2. Use HTTPS: `git remote set-url origin https://github.com/...`

## Files Modified/Created

| File | Status | Description |
|------|--------|-------------|
| `poker_env/agents/hf_agent.py` | **New** | HFAgent class |
| `poker_env/agents/prompts.py` | **New** | Action/belief prompts |
| `poker_env/agents/__init__.py` | Modified | Conditional HFAgent import |
| `poker_env/logging/decision_logger.py` | Modified | Added metadata fields |
| `run_experiment.py` | Modified | Added HF CLI flags |
| `requirements.txt` | Modified | Added torch, transformers |
| `setup.sh` | Modified | Added HF login |

## Next Steps

### Immediate
1. Run larger experiments (100+ hands)
2. Tune `--max-new-tokens` for reliable belief parsing
3. Analyze parse rates and coherence metrics

### Research Pipeline
1. **Generate data**: Run HF agent vs various opponents
2. **Enrich with oracles**: `python -m analysis.build_dataset`
3. **Compute metrics**: PCE, coherence, belief-action divergence
4. **Analyze**: "Plays well despite bad beliefs" phenomenon

### Model Variations
- Try different temperatures for variance analysis
- Compare Llama 3.1 8B vs 70B (with quantization)
- Test prompt uniformity across variants

## API Reference

### HFAgent

```python
from poker_env.agents import HFAgent

agent = HFAgent(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.2,
    max_new_tokens=128,
    max_input_tokens=2048,
    max_history_events=50,
    name="HFAgent",
)

# Get action
action = agent.act(obs)

# Get action with metadata
action, metadata = agent.act_with_metadata(obs)

# Get belief
belief = agent.belief(obs)

# Get belief with metadata
belief, metadata = agent.belief_with_metadata(obs)

# Access last metadata
agent.get_last_action_metadata()
agent.get_last_belief_metadata()
```

### ActionMetadata

```python
@dataclass
class ActionMetadata:
    parse_success: bool
    fallback_used: bool
    raw_response: str
    action_chosen: str
    prompt_hash: str
    prompt_template_id: str
```

### BeliefMetadata

```python
@dataclass
class BeliefMetadata:
    parse_success: bool
    raw_response: str
    prob_sum: float | None
    prob_min: float | None
    negative_count: int
    prompt_hash: str
    prompt_template_id: str
```
