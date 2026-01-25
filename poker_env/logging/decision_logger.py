"""JSONL decision logging for experiment replay and analysis."""

import json
from pathlib import Path
from typing import Optional, Any
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
    oracle_truth: Optional[dict]

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
    player0_delta: float
    player1_delta: float
    showdown: dict


class DecisionLogger:
    """
    Logger that writes decision records to JSONL files.

    Records one JSON object per line for each decision point,
    plus a summary record at the end of each hand.
    """

    def __init__(self, output_path: str):
        """
        Initialize the logger.

        Args:
            output_path: Path to output JSONL file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Current hand state
        self._current_hand_id: Optional[str] = None
        self._decision_idx: int = 0
        self._records: list[DecisionRecord] = []

        # Open file in append mode
        self._file = None

    def __enter__(self):
        """Context manager entry."""
        self._file = open(self.output_path, "a")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file:
            self._file.close()
            self._file = None

    def start_hand(self, hand_id: str, seed: int) -> None:
        """
        Start logging a new hand.

        Args:
            hand_id: Unique identifier for the hand
            seed: Random seed used for the hand
        """
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
        oracle_truth: Optional[dict] = None,
    ) -> None:
        """
        Log a decision point.

        Args:
            obs: Observation at the decision point
            hidden: Hidden state (opponent hole cards, etc.)
            agent_action: Action selected by the agent
            agent_belief: Optional belief dict from agent
            oracle_truth: Optional ground truth from oracle
        """
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
            oracle_truth=oracle_truth,
        )

        self._records.append(record)
        self._write_record(record.to_dict())
        self._decision_idx += 1

    def end_hand(
        self,
        final_stacks: list[int],
        player0_delta: float,
        player1_delta: float,
        showdown: dict,
    ) -> None:
        """
        Log hand completion summary.

        Args:
            final_stacks: Final stack sizes
            player0_delta: Stack change for player 0
            player1_delta: Stack change for player 1
            showdown: Showdown information (hole cards, board)
        """
        summary = HandSummary(
            hand_id=self._current_hand_id or "",
            seed=self._current_seed,
            num_decisions=self._decision_idx,
            final_stacks=final_stacks,
            player0_delta=player0_delta,
            player1_delta=player1_delta,
            showdown=showdown,
        )

        self._write_record({
            "type": "hand_summary",
            **asdict(summary),
        })

        # Reset state
        self._current_hand_id = None
        self._decision_idx = 0
        self._records = []

    def _write_record(self, record: dict) -> None:
        """Write a single record to the JSONL file."""
        if self._file:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()


def load_decisions(jsonl_path: str) -> list[dict]:
    """
    Load decision records from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        List of decision record dictionaries
    """
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_hand_summaries(jsonl_path: str) -> list[dict]:
    """
    Load only hand summary records from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        List of hand summary dictionaries
    """
    summaries = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("type") == "hand_summary":
                    summaries.append(record)
    return summaries
