"""JSONL decision logging for experiment replay and analysis."""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from poker_env.obs import Obs
from poker_env.actions import Action


@dataclass
class DecisionRecord:
    """
    Record of a single decision point for logging.

    Contains all information needed to replay and analyze
    a decision point in a poker hand.
    """

    hand_id: str
    seed: int
    decision_idx: int
    player_to_act: int
    street: str
    obs: dict
    hidden: dict
    legal_actions: list[str]
    agent_belief: Optional[dict]
    agent_action: str
    equity_given_true_hands: Optional[dict]

    # LLM agent metadata (populated by HFAgent or APIAgent)
    action_metadata: Optional[dict] = None
    belief_metadata: Optional[dict] = None

    # Reproducibility fields
    prompt_template_id: Optional[str] = None
    prompt_hash: Optional[str] = None
    probe_order: Optional[str] = None

    # Chain-of-Thought reasoning (extracted text, separate from raw_response)
    cot_reasoning: Optional[str] = None

    # Interpretability summary (lightweight inline; heavy data in sidecar files)
    interp_summary: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class HandSummary:
    """Summary record for a completed hand."""

    hand_id: str
    seed: int
    num_decisions: int
    final_stacks: list[int]
    deltas: dict
    showdown: dict


@dataclass
class RunConfig:
    """Configuration for a run, used for reproducibility tracking."""

    env_config_hash: str
    agent_configs: list[dict]
    prompt_version: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionLogger:
    """
    Logger that writes decision records to JSONL files.

    Records one JSON object per line for each decision point,
    plus a summary record at the end of each hand.
    Supports 2-6 players.

    Includes config hashes for paper-quality reproducibility.
    """

    def __init__(
        self,
        output_path: str,
        env_config_hash: str = "",
        agent_configs: list[dict] | None = None,
        prompt_version: str | None = None,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.run_config = RunConfig(
            env_config_hash=env_config_hash,
            agent_configs=agent_configs or [],
            prompt_version=prompt_version,
        )

        self._current_hand_id: Optional[str] = None
        self._current_seed: int = 0
        self._decision_idx: int = 0
        self._records: list[DecisionRecord] = []
        self._file = None

    @property
    def decision_idx(self) -> int:
        return self._decision_idx

    def __enter__(self):
        if self._file is not None:
            self._file.close()
        self._file = open(self.output_path, "a")
        self._write_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            self._file = None

    def _write_header(self):
        header = {
            "type": "run_config",
            **self.run_config.to_dict(),
        }
        self._write_record(header)

    def start_hand(self, hand_id: str, seed: int) -> None:
        self._current_hand_id = hand_id
        self._current_seed = seed
        self._decision_idx = 0
        self._records = []

    def log_decision(
        self,
        obs: Obs,
        hidden: dict,
        agent_action: Action,
        agent_belief: Optional[dict] = None,
        equity_given_true_hands: Optional[dict] = None,
        action_metadata: Optional[dict] = None,
        belief_metadata: Optional[dict] = None,
        prompt_template_id: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        probe_order: Optional[str] = None,
        cot_reasoning: Optional[str] = None,
        interp_summary: Optional[dict] = None,
    ) -> None:
        record = DecisionRecord(
            hand_id=obs.hand_id,
            seed=obs.seed,
            decision_idx=self._decision_idx,
            player_to_act=obs.to_act,
            street=obs.street,
            obs=obs.to_dict(),
            hidden=hidden,
            legal_actions=[a.type.value for a in obs.legal_actions],
            agent_belief=agent_belief,
            agent_action=agent_action.type.value,
            equity_given_true_hands=equity_given_true_hands,
            action_metadata=action_metadata,
            belief_metadata=belief_metadata,
            prompt_template_id=prompt_template_id,
            prompt_hash=prompt_hash,
            probe_order=probe_order,
            cot_reasoning=cot_reasoning,
            interp_summary=interp_summary,
        )

        self._records.append(record)
        self._write_record(record.to_dict())
        self._decision_idx += 1

    def end_hand(
        self,
        final_stacks: list[int],
        deltas: dict,
        showdown: dict,
    ) -> None:
        summary = HandSummary(
            hand_id=self._current_hand_id or "",
            seed=self._current_seed,
            num_decisions=self._decision_idx,
            final_stacks=final_stacks,
            deltas=deltas,
            showdown=showdown,
        )

        self._write_record({
            "type": "hand_summary",
            **asdict(summary),
        })

        self._current_hand_id = None
        self._decision_idx = 0
        self._records = []

    def _write_record(self, record: dict) -> None:
        if self._file is None:
            import warnings
            warnings.warn(
                "DecisionLogger._write_record called but no file is open. "
                "Use 'with DecisionLogger(...) as logger:' to ensure records are written.",
                UserWarning,
                stacklevel=2,
            )
            return
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()


def load_decisions(jsonl_path: str) -> list[dict]:
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_hand_summaries(jsonl_path: str) -> list[dict]:
    summaries = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("type") == "hand_summary":
                    summaries.append(record)
    return summaries


def load_run_config(jsonl_path: str) -> Optional[dict]:
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("type") == "run_config":
                    return record
    return None
