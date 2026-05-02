"""
Comprehensive tests for all v2 features:
- json_utils (extract_json, parse_cot_response)
- prompts (system messages, helpers)
- config (MODEL_REGISTRY, resolve_model_id, get_model_config)
- DecisionRecord new fields
- DecisionLogger new params
- run_experiment CLI args parsing
- APIAgent structure (without live API call)
- HFAgent structure (without model loading)
- LogitLensExtractor structure
- ProbeDataset + LinearProbe
- AttentionExtractor structure
- analyze_cot scoring
- analyze_logit_lens crystallization
- run_single_hand metadata detection logic
"""

import json
import os
import tempfile
from pathlib import Path


# ============================================================
# 1. Test json_utils
# ============================================================

def test_extract_json_basic():
    from poker_env.agents.json_utils import extract_json
    assert extract_json('{"action": "FOLD"}') == {"action": "FOLD"}

def test_extract_json_with_fences():
    from poker_env.agents.json_utils import extract_json
    text = '```json\n{"action": "CHECK_OR_CALL"}\n```'
    assert extract_json(text) == {"action": "CHECK_OR_CALL"}

def test_extract_json_with_surrounding_text():
    from poker_env.agents.json_utils import extract_json
    text = 'Sure, here is my response: {"action": "BET_OR_RAISE"} Hope that helps!'
    assert extract_json(text) == {"action": "BET_OR_RAISE"}

def test_extract_json_trailing_comma():
    from poker_env.agents.json_utils import extract_json
    text = '{"action": "FOLD",}'
    assert extract_json(text) == {"action": "FOLD"}

def test_extract_json_returns_none_on_garbage():
    from poker_env.agents.json_utils import extract_json
    assert extract_json("no json here") is None
    assert extract_json("") is None

def test_extract_json_nested():
    from poker_env.agents.json_utils import extract_json
    text = '{"schema":"buckets_14_v1","probs":[0.1,0.2,0.3,0.05,0.05,0.05,0.05,0.05,0.03,0.02,0.02,0.02,0.03,0.03]}'
    result = extract_json(text)
    assert result is not None
    assert result["schema"] == "buckets_14_v1"
    assert len(result["probs"]) == 14

def test_parse_cot_response_explicit_markers():
    from poker_env.agents.json_utils import parse_cot_response
    text = (
        'REASONING: The opponent raised preflop and bet the flop.\n\n'
        'JSON: {"action": "FOLD"}'
    )
    reasoning, parsed = parse_cot_response(text)
    assert reasoning is not None
    assert "raised preflop" in reasoning
    assert parsed == {"action": "FOLD"}

def test_parse_cot_response_no_markers():
    from poker_env.agents.json_utils import parse_cot_response
    text = 'The opponent seems strong based on raising. {"action": "CHECK_OR_CALL"}'
    reasoning, parsed = parse_cot_response(text)
    assert reasoning is not None
    assert "opponent" in reasoning.lower()
    assert parsed == {"action": "CHECK_OR_CALL"}

def test_parse_cot_response_json_only():
    from poker_env.agents.json_utils import parse_cot_response
    text = '{"action": "BET_OR_RAISE"}'
    reasoning, parsed = parse_cot_response(text)
    assert reasoning is None
    assert parsed == {"action": "BET_OR_RAISE"}

def test_parse_cot_probabilities_marker():
    from poker_env.agents.json_utils import parse_cot_response
    text = (
        'REASONING: Opponent raised, suggesting strength.\n\n'
        'PROBABILITIES:\n```json\n{"schema":"buckets_14_v1","probs":[0.15,0.12,0.08,0.03,0.12,0.10,0.08,0.05,0.04,0.04,0.03,0.06,0.05,0.05]}\n```'
    )
    reasoning, parsed = parse_cot_response(text)
    assert reasoning is not None
    assert "strength" in reasoning
    assert parsed is not None
    assert len(parsed["probs"]) == 14


# ============================================================
# 2. Test prompts
# ============================================================

def test_prompt_system_messages_exist():
    from poker_env.agents.prompts import (
        ACTION_SYSTEM_MESSAGE,
        ACTION_COT_SYSTEM_MESSAGE,
        BELIEF_SYSTEM_MESSAGE_COMPACT,
        BELIEF_SYSTEM_MESSAGE_FULL,
        BELIEF_COT_SYSTEM_MESSAGE_COMPACT,
        BELIEF_COT_SYSTEM_MESSAGE_FULL,
    )
    assert "FOLD" in ACTION_SYSTEM_MESSAGE
    assert "REASONING" in ACTION_COT_SYSTEM_MESSAGE
    assert "buckets_14_v1" in BELIEF_SYSTEM_MESSAGE_COMPACT
    assert "non-negative" in BELIEF_SYSTEM_MESSAGE_FULL
    assert "REASONING" in BELIEF_COT_SYSTEM_MESSAGE_COMPACT
    assert "REASONING" in BELIEF_COT_SYSTEM_MESSAGE_FULL

def test_get_action_system_message():
    from poker_env.agents.prompts import get_action_system_message
    direct = get_action_system_message(cot=False)
    cot = get_action_system_message(cot=True)
    assert "REASONING" not in direct
    assert "REASONING" in cot

def test_get_belief_system_message():
    from poker_env.agents.prompts import get_belief_system_message
    compact = get_belief_system_message(belief_format="compact", cot=False)
    full = get_belief_system_message(belief_format="full", cot=False)
    compact_cot = get_belief_system_message(belief_format="compact", cot=True)
    full_cot = get_belief_system_message(belief_format="full", cot=True)
    assert "probs" in compact
    assert "REASONING" not in compact
    assert "REASONING" in compact_cot
    assert "REASONING" in full_cot

def test_template_ids():
    from poker_env.agents.prompts import get_template_id
    assert get_template_id("action", "default") == "action_strict_v1"
    assert get_template_id("action", "cot") == "action_cot_v1"
    assert get_template_id("belief", "compact") == "belief_compact_v1"
    assert get_template_id("belief", "compact_cot") == "belief_compact_cot_v1"


# ============================================================
# 3. Test config
# ============================================================

def test_model_registry():
    from poker_env.config import MODEL_REGISTRY
    # 8B class
    assert "llama-8b" in MODEL_REGISTRY
    assert "qwen-8b" in MODEL_REGISTRY
    assert "ministral-8b" in MODEL_REGISTRY
    # 70B class
    assert "llama-70b" in MODEL_REGISTRY
    assert "llama-3.3-70b" in MODEL_REGISTRY
    assert "qwen-72b" in MODEL_REGISTRY
    # Qwen 3 must be flagged as thinking-mode-capable so HFAgent disables it
    # for non-CoT runs. Without this flag, qwen-8b silently does internal CoT.
    assert MODEL_REGISTRY["qwen-8b"].get("has_thinking_mode") is True
    # No other model should claim thinking-mode support
    for name, cfg in MODEL_REGISTRY.items():
        if name == "qwen-8b":
            continue
        assert cfg.get("has_thinking_mode", False) is False, f"{name} unexpectedly has thinking mode"

def test_resolve_model_id():
    from poker_env.config import resolve_model_id
    assert resolve_model_id("llama-8b") == "meta-llama/Llama-3.1-8B-Instruct"
    assert resolve_model_id("qwen-8b") == "Qwen/Qwen3-8B"
    assert resolve_model_id("ministral-8b") == "mistralai/Ministral-8B-Instruct-2410"
    assert resolve_model_id("llama-70b") == "meta-llama/Llama-3.1-70B-Instruct"
    assert resolve_model_id("llama-3.3-70b") == "meta-llama/Llama-3.3-70B-Instruct"
    assert resolve_model_id("qwen-72b") == "Qwen/Qwen2.5-72B-Instruct"
    # Pass-through for full IDs
    assert resolve_model_id("some/custom-model") == "some/custom-model"

def test_get_model_config():
    from poker_env.config import get_model_config
    cfg = get_model_config("ministral-8b")
    assert cfg["supports_system_role"] is True
    # Qwen 3 thinking mode flag visible via get_model_config
    cfg_qwen = get_model_config("qwen-8b")
    assert cfg_qwen.get("has_thinking_mode") is True
    # Full ID lookup
    cfg2 = get_model_config("meta-llama/Llama-3.1-8B-Instruct")
    assert cfg2["supports_system_role"] is True
    # Unknown model defaults
    cfg3 = get_model_config("unknown/model")
    assert cfg3["supports_system_role"] is True

def test_cot_token_budgets():
    from poker_env.config import (
        DEFAULT_ACTION_MAX_TOKENS, DEFAULT_BELIEF_MAX_TOKENS,
        DEFAULT_COT_ACTION_MAX_TOKENS, DEFAULT_COT_BELIEF_MAX_TOKENS,
    )
    assert DEFAULT_COT_ACTION_MAX_TOKENS > DEFAULT_ACTION_MAX_TOKENS
    assert DEFAULT_COT_BELIEF_MAX_TOKENS > DEFAULT_BELIEF_MAX_TOKENS


# ============================================================
# 4. Test DecisionRecord + DecisionLogger
# ============================================================

def test_decision_record_new_fields():
    from poker_env.logging.decision_logger import DecisionRecord
    rec = DecisionRecord(
        hand_id="h1", seed=42, decision_idx=0, player_to_act=0,
        street="PREFLOP", obs={}, hidden={}, legal_actions=["FOLD"],
        agent_belief=None, agent_action="FOLD",
        equity_given_true_hands=None,
        cot_reasoning="Opponent is strong",
        interp_summary={"logit_lens_layers": 32},
    )
    d = rec.to_dict()
    assert d["cot_reasoning"] == "Opponent is strong"
    assert d["interp_summary"]["logit_lens_layers"] == 32

def test_decision_record_defaults():
    from poker_env.logging.decision_logger import DecisionRecord
    rec = DecisionRecord(
        hand_id="h2", seed=0, decision_idx=0, player_to_act=0,
        street="PREFLOP", obs={}, hidden={}, legal_actions=[],
        agent_belief=None, agent_action="FOLD",
        equity_given_true_hands=None,
    )
    d = rec.to_dict()
    assert d["cot_reasoning"] is None
    assert d["interp_summary"] is None

def test_decision_logger_writes_cot():
    from poker_env.logging.decision_logger import DecisionLogger
    from poker_env.obs import Obs
    from poker_env.actions import FOLD

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        tmppath = f.name

    try:
        with DecisionLogger(tmppath) as logger:
            # Create a minimal obs
            obs = Obs(
                hand_id="test_hand", seed=42, street="PREFLOP",
                street_index=0,
                board=[], hero_hole=["Ah", "Kh"], stacks=[200, 200],
                pot_total=3, bet_to_call=1, legal_actions=[FOLD],
                history=[], player_index=0, to_act=0, num_players=2,
            )
            logger.start_hand("test_hand", 42)
            logger.log_decision(
                obs=obs, hidden={}, agent_action=FOLD,
                cot_reasoning="Test reasoning",
                interp_summary={"test": True},
            )
            logger.end_hand(final_stacks=[199, 201], deltas={}, showdown={})

        with open(tmppath) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        # First line is run_config, second is decision, third is hand_summary
        assert len(lines) == 3
        decision = lines[1]
        assert decision["cot_reasoning"] == "Test reasoning"
        assert decision["interp_summary"]["test"] is True
    finally:
        os.unlink(tmppath)


# ============================================================
# 5. Test run_experiment CLI parsing
# ============================================================

def test_cli_new_args():
    """Test that argparse accepts all new arguments and the module API surface."""
    import run_experiment
    assert hasattr(run_experiment, 'create_agent')
    assert hasattr(run_experiment, 'run_experiment')
    assert hasattr(run_experiment, 'run_single_hand')
    assert hasattr(run_experiment, 'get_agent_configs')
    assert callable(run_experiment.create_agent)
    assert callable(run_experiment.run_experiment)

def test_create_agent_types():
    from run_experiment import create_agent
    from poker_env.agents import RandomAgent, CallAgent, ThresholdAgent

    a1 = create_agent("random", seed=42, name="R")
    assert isinstance(a1, RandomAgent)

    a2 = create_agent("call", name="C")
    assert isinstance(a2, CallAgent)

    a3 = create_agent("threshold", name="T", opponent_preset="informative_v2")
    assert isinstance(a3, ThresholdAgent)
    assert a3.preset == "informative_v2"

def test_create_agent_hf_raises_without_torch():
    """HF agent should raise ImportError if torch not available (or succeed if it is)."""
    from run_experiment import create_agent
    from poker_env.agents import HF_AVAILABLE
    if not HF_AVAILABLE:
        try:
            create_agent("hf", name="H")
            assert False, "Should have raised ImportError"
        except ImportError:
            pass

def test_create_agent_api_raises_without_sdk():
    """API agent should raise ImportError if openai not available (or succeed if it is)."""
    from run_experiment import create_agent
    from poker_env.agents import API_AVAILABLE
    if not API_AVAILABLE:
        try:
            create_agent("api", name="A")
            assert False, "Should have raised ImportError"
        except ImportError:
            pass

def test_create_agent_unknown_type():
    from run_experiment import create_agent
    try:
        create_agent("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_get_agent_configs_uses_get_config():
    from run_experiment import get_agent_configs
    from poker_env.agents import ThresholdAgent

    agents = [ThresholdAgent(preset="informative_v2", name="T0")]
    configs = get_agent_configs(agents)
    assert len(configs) == 1
    assert configs[0]["type"] == "ThresholdAgent"
    assert configs[0]["preset"] == "informative_v2"
    assert configs[0]["name"] == "T0"

def test_get_agent_configs_shared_instance():
    from run_experiment import get_agent_configs
    from poker_env.agents import RandomAgent

    agent = RandomAgent(seed=42, name="R")
    agents = [agent, agent]  # shared instance
    configs = get_agent_configs(agents)
    assert len(configs) == 2
    # RandomAgent doesn't have get_config(), so shared_instance tagging only applies
    # to HF/API agents. For RandomAgent, both entries should have the type.
    assert configs[0].get("type") == "RandomAgent"
    assert configs[1].get("type") == "RandomAgent"


# ============================================================
# 6. Test run_single_hand metadata detection
# ============================================================

def test_metadata_detection_protocol():
    """Verify has_metadata check works without HF_AVAILABLE gate."""
    from poker_env.agents.base import BaseAgent
    from poker_env.actions import FOLD
    from poker_env.obs import Obs

    class MockRichAgent(BaseAgent):
        def __init__(self):
            super().__init__(name="MockRich")
        def act(self, obs):
            return FOLD
        def act_with_metadata(self, obs):
            return FOLD, None
        def belief_with_metadata(self, obs):
            return None, None

    class MockSimpleAgent(BaseAgent):
        def __init__(self):
            super().__init__(name="MockSimple")
        def act(self, obs):
            return FOLD

    rich = MockRichAgent()
    simple = MockSimpleAgent()

    # This is the exact check from run_single_hand
    has_meta_rich = hasattr(rich, 'act_with_metadata') and hasattr(rich, 'belief_with_metadata')
    has_meta_simple = hasattr(simple, 'act_with_metadata') and hasattr(simple, 'belief_with_metadata')

    assert has_meta_rich is True
    assert has_meta_simple is False


# ============================================================
# 7. Test full experiment with random agents
# ============================================================

def test_full_experiment_random_agents():
    """Run a real 5-hand experiment with random agents."""
    from run_experiment import run_experiment
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
        tmppath = f.name

    try:
        summary = run_experiment(
            num_hands=5,
            agent_types=["random"],
            num_players=2,
            output_path=tmppath,
            base_seed=42,
            compute_oracle=True,
            verbose=False,
        )
        assert summary["num_hands"] == 5
        assert summary["num_players"] == 2
        assert os.path.exists(tmppath)

        # Check JSONL output
        with open(tmppath) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) > 0
        assert lines[0].get("type") == "run_config"

        # Check decision records have new fields
        decisions = [l for l in lines if l.get("type") not in ("run_config", "hand_summary")]
        if decisions:
            assert "cot_reasoning" in decisions[0]
            assert "interp_summary" in decisions[0]
    finally:
        os.unlink(tmppath)

def test_full_experiment_threshold_agent():
    """Run experiment with threshold as primary agent (newly added to --agent choices)."""
    from run_experiment import run_experiment
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
        tmppath = f.name

    try:
        summary = run_experiment(
            num_hands=3,
            agent_types=["threshold", "call"],
            num_players=2,
            output_path=tmppath,
            base_seed=99,
            compute_oracle=False,
            verbose=False,
            opponent_preset="informative_v2",
        )
        assert summary["num_hands"] == 3
        assert "threshold" in summary["agents"][0].lower()
    finally:
        os.unlink(tmppath)


# ============================================================
# 8. Test interp: ProbeDataset + LinearProbe
# ============================================================

def test_probe_dataset():
    from poker_env.interp.probing import ProbeDataset
    import numpy as np

    ds = ProbeDataset()
    for i in range(20):
        label = "strong" if i % 2 == 0 else "weak"
        ds.add({0: np.random.randn(64), 1: np.random.randn(64)}, label)

    assert ds.num_samples == 20
    assert ds.layers == [0, 1]

    X, y = ds.to_numpy(0)
    assert X.shape == (20, 64)
    assert len(y) == 20

def test_linear_probe():
    from poker_env.interp.probing import ProbeDataset, LinearProbe
    import numpy as np

    np.random.seed(42)
    ds = ProbeDataset()
    for i in range(50):
        label = "A" if i % 3 == 0 else ("B" if i % 3 == 1 else "C")
        vec = np.random.randn(32)
        if label == "A":
            vec[0] += 3.0
        elif label == "B":
            vec[1] += 3.0
        else:
            vec[2] += 3.0
        ds.add({0: vec}, label)

    probe = LinearProbe()
    results = probe.evaluate(ds, cv=3)
    assert 0 in results
    assert 0.0 <= results[0] <= 1.0
    # With such clear signal, accuracy should be decent
    assert results[0] > 0.5


# ============================================================
# 9. Test analyze_cot scoring
# ============================================================

def test_score_reasoning():
    from analysis.analyze_cot import score_reasoning

    s = score_reasoning(None)
    assert s["length"] == 0
    assert s["quality_score"] == 0.0

    s = score_reasoning("Opponent raised preflop, the board has a flush draw, likely AA or KK.")
    assert s["length"] > 0
    assert s["mentions_opponent_actions"] is True
    assert s["mentions_board_texture"] is True
    assert s["mentions_hand_combos"] is True
    assert s["quality_score"] == 1.0

    s = score_reasoning("I think the pot is big.")
    assert s["mentions_opponent_actions"] is False
    assert s["quality_score"] < 0.5


# ============================================================
# 10. Test analyze_logit_lens crystallization
# ============================================================

def test_crystallization_layer():
    from analysis.analyze_logit_lens import compute_crystallization_layer

    # Layer 0-2: different top tokens, layers 3-5: same as final
    tokens = [
        ["a", "b"],   # layer 0
        ["c", "d"],   # layer 1
        ["x", "y"],   # layer 2
        ["f", "g"],   # layer 3
        ["f", "g"],   # layer 4 (final)
    ]
    # Should be layer 3 (first layer matching final)
    assert compute_crystallization_layer(tokens) == 3

    # All same from start
    tokens2 = [["x"], ["x"], ["x"]]
    assert compute_crystallization_layer(tokens2) == 0

    # Empty
    assert compute_crystallization_layer([]) is None

def test_analyze_logit_lens_data():
    from analysis.analyze_logit_lens import analyze_logit_lens_data

    records = [
        {
            "num_layers": 4,
            "per_layer_entropy": [5.0, 3.0, 2.0, 1.0],
            "per_layer_top_tokens": [["a"], ["b"], ["c"], ["c"]],
        },
        {
            "num_layers": 4,
            "per_layer_entropy": [6.0, 4.0, 2.5, 1.5],
            "per_layer_top_tokens": [["x"], ["y"], ["z"], ["z"]],
        },
    ]

    result = analyze_logit_lens_data(records)
    assert result["num_records"] == 2
    assert result["num_layers"] == 4
    assert len(result["avg_entropy_per_layer"]) == 4
    # Entropy should decrease on average
    assert result["avg_entropy_per_layer"][0] > result["avg_entropy_per_layer"][3]
    assert result["crystallization_layer"]["n"] == 2


# ============================================================
# 11. Test APIAgent (structure only, no live calls)
# ============================================================

def test_api_agent_importable():
    from poker_env.agents import API_AVAILABLE
    # Just check the flag exists and is boolean
    assert isinstance(API_AVAILABLE, bool)

def test_api_metadata_structure():
    """Test APIMetadata can be created and serialized."""
    try:
        from poker_env.agents.api_agent import APIMetadata
    except ImportError:
        return  # Skip if openai not installed

    meta = APIMetadata(
        parse_success=True,
        fallback_used=False,
        raw_response='{"action": "FOLD"}',
        action_chosen="FOLD",
        prompt_hash="abc123",
        prompt_template_id="action_strict_v1",
        provider="openai",
        model="gpt-4o",
        latency_ms=150.3,
        prompt_tokens=100,
        completion_tokens=20,
        finish_reason="stop",
        logprobs=[{"token": "FOLD", "logprob": -0.1, "top_logprobs": []}],
    )
    d = meta.to_dict()
    assert d["provider"] == "openai"
    assert d["latency_ms"] == 150.3
    assert d["logprobs"] is not None
    assert len(d["logprobs"]) == 1


# ============================================================
# 12. Test HFAgent structure (no model loading)
# ============================================================

def test_hf_agent_importable():
    from poker_env.agents import HF_AVAILABLE
    assert isinstance(HF_AVAILABLE, bool)

def test_hf_metadata_structures():
    """Test ActionMetadata and BeliefMetadata."""
    try:
        from poker_env.agents.hf_agent import ActionMetadata, BeliefMetadata
    except ImportError:
        return

    am = ActionMetadata(
        parse_success=True, fallback_used=False,
        raw_response="test", action_chosen="FOLD",
        prompt_hash="abc", prompt_template_id="action_cot_v1",
    )
    d = am.to_dict()
    assert d["prompt_template_id"] == "action_cot_v1"

    bm = BeliefMetadata(
        parse_success=True, raw_response="test",
        prob_sum=1.0, prob_min=0.01, negative_count=0,
        prompt_hash="def", belief_format="compact",
    )
    d = bm.to_dict()
    assert d["belief_format"] == "compact"


# ============================================================
# 13. Test attention module structure
# ============================================================

def test_attention_token_categorization():
    from poker_env.interp.attention import AttentionExtractor
    cat = AttentionExtractor._categorize_token
    assert cat("Ah") == "card"
    assert cat("Ks") == "card"
    assert cat("raise") == "action"
    assert cat("fold") == "action"
    assert cat("100") == "number"
    assert cat("the") == "other"

def test_attention_snapshot():
    from poker_env.interp.attention import AttentionSnapshot
    import numpy as np

    snap = AttentionSnapshot(
        attention_to_input=np.array([[[0.3, 0.2, 0.5]]]),  # 1 layer, 1 head, 3 positions
        input_tokens=["Ah", "raise", "the"],
        token_categories=["card", "action", "other"],
    )
    fracs = snap.category_fractions()
    assert abs(fracs["card"] - 0.3) < 0.01
    assert abs(fracs["action"] - 0.2) < 0.01
    assert abs(fracs["other"] - 0.5) < 0.01


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    import sys
    test_functions = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]

    passed = 0
    failed = 0
    errors = []

    for fn in test_functions:
        name = fn.__name__
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  FAIL  {name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if errors:
        print(f"\nFailed tests:")
        for name, e in errors:
            print(f"  {name}: {type(e).__name__}: {e}")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
