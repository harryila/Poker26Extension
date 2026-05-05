"""
Pot-odds and EV-based action quality analysis.

Reviewer feedback on poker26 was that calling an action "clearly correct"
based purely on an equity threshold is incomplete; the right benchmark is
expected value (EV) given the betting structure (pot odds). This module
operationalizes that critique:

For each LLM decision in an enriched JSONL log, we compute:

  1. **Pot odds** = bet_to_call / (pot + bet_to_call), the equity required
     to break even on a CHECK_OR_CALL.
  2. **True equity** from the logged `equity_given_true_hands` field
     (Monte Carlo equity vs the actual villain hand the env knows).
  3. **EV per legal action** under three opposing distributions:
       - Truth: villain's actual hole cards (single-hand rollout).
       - StrategyAware oracle: per-bucket posterior from `oracle_strategy_aware`.
       - Agent belief: the LLM's stated bucket distribution.
     Each EV is computed via `analysis.implied_belief.q_value.MonteCarloQValue`.
  4. **Action quality flags**:
       - `equity_threshold_correct`: did the agent's call/fold match the simple
         "equity vs pot odds" rule?
       - `ev_truth_optimal`: did the agent pick the EV-best action under the
         actual villain hand?
       - `ev_belief_optimal`: did the agent pick the EV-best action consistent
         with its OWN stated belief (internal consistency)?
       - `ev_oracle_optimal`: did the agent pick the EV-best action under the
         strategy-aware oracle posterior?
       - `ev_regret_*`: gap between chosen action's EV and best action's EV.

Outputs a CSV row per decision plus an aggregate summary JSON.

Usage:
    python -m analysis.pot_odds_analysis \\
        --input logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl \\
        --output results/pot_odds.csv \\
        --num-rollouts 100 \\
        --samples-per-bucket 5
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Optional

from analysis.implied_belief.q_value import MonteCarloQValue
from analysis.opponent_model import ActionType, ParametricOpponent


# ============================================================================
# Pot odds / break-even
# ============================================================================

def compute_pot_odds(pot: float, bet_to_call: float) -> float:
    """Required equity to break even on CHECK_OR_CALL.

    Returns 0.0 when there is no bet to face (a check is free).
    """
    if bet_to_call is None or bet_to_call <= 0:
        return 0.0
    denom = float(pot) + float(bet_to_call)
    if denom <= 0:
        return 0.0
    return float(bet_to_call) / denom


def equity_threshold_action(equity: float, pot_odds: float) -> str:
    """The simple 'call-if-equity > pot odds' decision rule."""
    if equity is None or pot_odds is None:
        return "UNKNOWN"
    return "CHECK_OR_CALL" if equity >= pot_odds else "FOLD"


# ============================================================================
# EV under villain distributions
# ============================================================================

def _action_str_to_type(action_str: str) -> Optional[ActionType]:
    if action_str == "FOLD":
        return ActionType.FOLD
    if action_str == "CHECK_OR_CALL":
        return ActionType.CHECK_CALL
    if action_str == "BET_OR_RAISE":
        return ActionType.BET_RAISE
    return None


def _action_type_to_str(action_type: ActionType) -> str:
    return {
        ActionType.FOLD: "FOLD",
        ActionType.CHECK_CALL: "CHECK_OR_CALL",
        ActionType.BET_RAISE: "BET_OR_RAISE",
    }[action_type]


def ev_under_villain_hand(
    q_estimator: MonteCarloQValue,
    hero_hole: list[str],
    villain_hole: tuple[str, str],
    board: list[str],
    pot: int,
    bet_to_call: int,
    street: str,
    legal_actions: list[ActionType],
) -> dict[ActionType, float]:
    """EV per legal action when the villain's exact hand is known."""
    return q_estimator.compute_q_values(
        hero_hole=list(hero_hole),
        villain_hole=tuple(villain_hole),
        board=list(board),
        pot=int(pot),
        bet_to_call=int(bet_to_call),
        street=str(street),
        legal_actions=legal_actions,
    )


def ev_under_bucket_distribution(
    q_estimator: MonteCarloQValue,
    hero_hole: list[str],
    distribution: dict[str, float],
    board: list[str],
    pot: int,
    bet_to_call: int,
    street: str,
    legal_actions: list[ActionType],
    samples_per_bucket: int,
) -> dict[ActionType, float]:
    """EV per legal action marginalized over a bucket distribution.

    Uses `compute_q_values_by_bucket` then takes a probability-weighted
    expectation.
    """
    if not distribution:
        return {a: 0.0 for a in legal_actions}

    q_by_bucket = q_estimator.compute_q_values_by_bucket(
        hero_hole=list(hero_hole),
        board=list(board),
        pot=int(pot),
        bet_to_call=int(bet_to_call),
        street=str(street),
        legal_actions=legal_actions,
        samples_per_bucket=samples_per_bucket,
    )

    total_prob = sum(p for p in distribution.values() if p and p > 0)
    if total_prob <= 0:
        return {a: 0.0 for a in legal_actions}

    ev: dict[ActionType, float] = {a: 0.0 for a in legal_actions}
    for bucket, prob in distribution.items():
        if not prob or prob <= 0:
            continue
        weight = prob / total_prob
        bucket_qs = q_by_bucket.get(bucket, {})
        for a in legal_actions:
            ev[a] += weight * bucket_qs.get(a, 0.0)
    return ev


def best_action(ev_by_action: dict[ActionType, float]) -> tuple[Optional[ActionType], float]:
    """Argmax over a possibly empty EV dict."""
    if not ev_by_action:
        return None, 0.0
    best = max(ev_by_action, key=lambda a: ev_by_action[a])
    return best, ev_by_action[best]


# ============================================================================
# Decision-level analysis
# ============================================================================

def analyze_decision(
    record: dict,
    q_estimator: MonteCarloQValue,
    samples_per_bucket: int,
    skip_belief_ev: bool = False,
    skip_oracle_ev: bool = False,
) -> Optional[dict]:
    """Compute pot-odds + EV metrics for a single decision record.

    Returns None if the record is malformed or missing required fields.
    """
    obs = record.get("obs") or {}
    hero_hole = obs.get("hero_hole") or []
    if len(hero_hole) != 2:
        return None

    board = obs.get("board") or []
    pot = obs.get("pot_total", 0) or 0
    bet_to_call = obs.get("bet_to_call", 0) or 0
    street = obs.get("street") or "PREFLOP"
    legal_action_strs = record.get("legal_actions") or []
    legal_actions = [t for t in (_action_str_to_type(s) for s in legal_action_strs) if t is not None]
    if not legal_actions:
        return None

    chosen_str = record.get("agent_action") or ""
    chosen_type = _action_str_to_type(chosen_str)

    pot_odds = compute_pot_odds(pot, bet_to_call)

    # Pull true villain hole if present (heads-up assumption: the only "playerN_hole" key not equal to hero).
    hidden = record.get("hidden") or {}
    hero_idx = record.get("player_to_act", obs.get("player_index", 0))
    villain_hole: Optional[tuple[str, str]] = None
    for key, value in hidden.items():
        if not key.endswith("_hole"):
            continue
        if key == f"player{hero_idx}_hole":
            continue
        if isinstance(value, list) and len(value) == 2:
            villain_hole = (value[0], value[1])
            break

    # True equity (Monte Carlo equity vs the real villain hand) from the enriched log.
    # The env's EquityOracle uses keys equity_win / equity_tie / equity_lose;
    # accept the older win_prob / tie_prob naming as well for forward-compat.
    equity_truth: Optional[float] = None
    eq_dict = record.get("equity_given_true_hands") or {}
    if isinstance(eq_dict, dict):
        win = eq_dict.get("equity_win", eq_dict.get("win_prob"))
        tie = eq_dict.get("equity_tie", eq_dict.get("tie_prob"))
        if win is not None:
            equity_truth = float(win) + 0.5 * float(tie or 0.0)

    # EV under truth (only possible when we know the villain hand).
    ev_truth: dict[ActionType, float] = {}
    if villain_hole is not None:
        try:
            ev_truth = ev_under_villain_hand(
                q_estimator=q_estimator,
                hero_hole=hero_hole,
                villain_hole=villain_hole,
                board=board,
                pot=pot,
                bet_to_call=bet_to_call,
                street=street,
                legal_actions=legal_actions,
            )
        except Exception:
            ev_truth = {}

    # EV under the strategy-aware oracle posterior over buckets.
    ev_oracle: dict[ActionType, float] = {}
    oracle_dist = record.get("oracle_strategy_aware") or {}
    if not skip_oracle_ev and oracle_dist:
        try:
            ev_oracle = ev_under_bucket_distribution(
                q_estimator=q_estimator,
                hero_hole=hero_hole,
                distribution=oracle_dist,
                board=board,
                pot=pot,
                bet_to_call=bet_to_call,
                street=street,
                legal_actions=legal_actions,
                samples_per_bucket=samples_per_bucket,
            )
        except Exception:
            ev_oracle = {}

    # EV under the agent's stated belief.
    ev_belief: dict[ActionType, float] = {}
    belief = record.get("agent_belief") or {}
    if not skip_belief_ev and belief:
        try:
            ev_belief = ev_under_bucket_distribution(
                q_estimator=q_estimator,
                hero_hole=hero_hole,
                distribution=belief,
                board=board,
                pot=pot,
                bet_to_call=bet_to_call,
                street=street,
                legal_actions=legal_actions,
                samples_per_bucket=samples_per_bucket,
            )
        except Exception:
            ev_belief = {}

    truth_best, truth_best_ev = best_action(ev_truth)
    oracle_best, oracle_best_ev = best_action(ev_oracle)
    belief_best, belief_best_ev = best_action(ev_belief)

    chosen_ev_truth = ev_truth.get(chosen_type) if chosen_type is not None else None
    chosen_ev_oracle = ev_oracle.get(chosen_type) if chosen_type is not None else None
    chosen_ev_belief = ev_belief.get(chosen_type) if chosen_type is not None else None

    threshold_call = equity_threshold_action(equity_truth, pot_odds) if equity_truth is not None else None
    threshold_correct: Optional[bool] = None
    if threshold_call is not None and chosen_str:
        # The simple rule only resolves between FOLD and CHECK_OR_CALL.
        # For BET_OR_RAISE we cannot say from this rule alone.
        if chosen_str in ("FOLD", "CHECK_OR_CALL"):
            threshold_correct = (threshold_call == chosen_str)

    return {
        "hand_id": record.get("hand_id"),
        "decision_idx": record.get("decision_idx"),
        "street": street,
        "player_to_act": hero_idx,
        "pot": pot,
        "bet_to_call": bet_to_call,
        "pot_odds": round(pot_odds, 6),
        "equity_truth": round(equity_truth, 6) if equity_truth is not None else None,
        "agent_action": chosen_str,
        "legal_actions": ",".join(legal_action_strs),
        "threshold_call": threshold_call,
        "equity_threshold_correct": threshold_correct,
        "ev_truth_chosen": _round_optional(chosen_ev_truth),
        "ev_truth_best": _round_optional(truth_best_ev) if truth_best is not None else None,
        "ev_truth_best_action": _action_type_to_str(truth_best) if truth_best is not None else None,
        "ev_truth_optimal": (truth_best is not None and chosen_type is not None and truth_best == chosen_type),
        "ev_truth_regret": _round_optional(
            (truth_best_ev - chosen_ev_truth) if (truth_best is not None and chosen_ev_truth is not None) else None
        ),
        "ev_oracle_chosen": _round_optional(chosen_ev_oracle),
        "ev_oracle_best": _round_optional(oracle_best_ev) if oracle_best is not None else None,
        "ev_oracle_best_action": _action_type_to_str(oracle_best) if oracle_best is not None else None,
        "ev_oracle_optimal": (oracle_best is not None and chosen_type is not None and oracle_best == chosen_type),
        "ev_oracle_regret": _round_optional(
            (oracle_best_ev - chosen_ev_oracle) if (oracle_best is not None and chosen_ev_oracle is not None) else None
        ),
        "ev_belief_chosen": _round_optional(chosen_ev_belief),
        "ev_belief_best": _round_optional(belief_best_ev) if belief_best is not None else None,
        "ev_belief_best_action": _action_type_to_str(belief_best) if belief_best is not None else None,
        "ev_belief_optimal": (belief_best is not None and chosen_type is not None and belief_best == chosen_type),
        "ev_belief_regret": _round_optional(
            (belief_best_ev - chosen_ev_belief) if (belief_best is not None and chosen_ev_belief is not None) else None
        ),
    }


def _round_optional(x: Optional[float], ndigits: int = 4) -> Optional[float]:
    return round(float(x), ndigits) if x is not None else None


# ============================================================================
# I/O
# ============================================================================

def _open_log(path: str):
    """Open a .jsonl or .jsonl.gz transparently as a text iterator."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_decision_records(input_path: str):
    """Yield decision records from an enriched JSONL (.jsonl or .jsonl.gz),
    skipping config/summary lines."""
    with _open_log(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") in ("run_config", "hand_summary"):
                continue
            # Only consider rows that look like LLM decision points.
            if record.get("agent_belief") is None and record.get("action_metadata") is None:
                # Still allow purely action-based records, but require a chosen action.
                if not record.get("agent_action"):
                    continue
            yield record


CSV_FIELDS = [
    "hand_id", "decision_idx", "street", "player_to_act",
    "pot", "bet_to_call", "pot_odds",
    "equity_truth", "agent_action", "legal_actions",
    "threshold_call", "equity_threshold_correct",
    "ev_truth_chosen", "ev_truth_best", "ev_truth_best_action",
    "ev_truth_optimal", "ev_truth_regret",
    "ev_oracle_chosen", "ev_oracle_best", "ev_oracle_best_action",
    "ev_oracle_optimal", "ev_oracle_regret",
    "ev_belief_chosen", "ev_belief_best", "ev_belief_best_action",
    "ev_belief_optimal", "ev_belief_regret",
]


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-decision rows into headline numbers."""
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def _safe_mean(vals: list[float]) -> Optional[float]:
        clean = [v for v in vals if v is not None]
        return sum(clean) / len(clean) if clean else None

    def _frac_true(key: str) -> Optional[float]:
        clean = [bool(r[key]) for r in rows if r.get(key) is not None]
        return sum(clean) / len(clean) if clean else None

    summary = {
        "n": n,
        "mean_pot_odds": _safe_mean([r["pot_odds"] for r in rows]),
        "mean_equity_truth": _safe_mean([r["equity_truth"] for r in rows]),
        "frac_equity_threshold_correct": _frac_true("equity_threshold_correct"),
        "frac_ev_truth_optimal": _frac_true("ev_truth_optimal"),
        "frac_ev_oracle_optimal": _frac_true("ev_oracle_optimal"),
        "frac_ev_belief_optimal": _frac_true("ev_belief_optimal"),
        "mean_ev_truth_regret": _safe_mean([r["ev_truth_regret"] for r in rows]),
        "mean_ev_oracle_regret": _safe_mean([r["ev_oracle_regret"] for r in rows]),
        "mean_ev_belief_regret": _safe_mean([r["ev_belief_regret"] for r in rows]),
    }

    # Disagreement between simple equity rule and EV-under-truth (the reviewer's exact concern).
    threshold_only = [
        (r["threshold_call"], r["ev_truth_best_action"])
        for r in rows
        if r.get("threshold_call") in ("FOLD", "CHECK_OR_CALL")
        and r.get("ev_truth_best_action") is not None
    ]
    if threshold_only:
        agree = sum(1 for tc, eb in threshold_only if tc == eb)
        summary["frac_threshold_agrees_with_ev_truth"] = agree / len(threshold_only)
        summary["n_threshold_vs_ev_compared"] = len(threshold_only)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compute pot-odds and EV-based action quality from enriched poker logs.",
    )
    parser.add_argument("--input", required=True, help="Enriched JSONL log path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--summary-out", default=None,
        help="Optional JSON path for the aggregate summary (default: <output>.summary.json)",
    )
    parser.add_argument(
        "--num-rollouts", type=int, default=100,
        help="Monte Carlo rollouts per Q-value (default: 100). Higher = slower, less noisy.",
    )
    parser.add_argument(
        "--samples-per-bucket", type=int, default=5,
        help="Villain hands sampled per bucket for distribution-based EV (default: 5).",
    )
    parser.add_argument(
        "--opponent-preset", default="default",
        choices=["default", "tight_passive", "tight_aggressive",
                 "loose_passive", "loose_aggressive", "informative", "informative_v2"],
        help="Opponent model preset for villain response simulation (default: default)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for rollouts (default: 42)")
    parser.add_argument("--small-bet", type=int, default=2, help="Small-bet size (default: 2)")
    parser.add_argument("--big-bet", type=int, default=4, help="Big-bet size (default: 4)")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after analyzing this many decisions (debug aid).",
    )
    parser.add_argument(
        "--skip-belief-ev", action="store_true",
        help="Skip EV-under-belief (faster; loses internal-consistency metric).",
    )
    parser.add_argument(
        "--skip-oracle-ev", action="store_true",
        help="Skip EV-under-strategy-aware-oracle (faster).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    opponent_model = ParametricOpponent.from_preset(args.opponent_preset)
    q_estimator = MonteCarloQValue(
        opponent_model=opponent_model,
        num_rollouts=args.num_rollouts,
        seed=args.seed,
        small_bet=args.small_bet,
        big_bet=args.big_bet,
    )

    rows: list[dict] = []
    n_seen = 0
    n_skipped = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for record in iter_decision_records(args.input):
            n_seen += 1
            row = analyze_decision(
                record,
                q_estimator=q_estimator,
                samples_per_bucket=args.samples_per_bucket,
                skip_belief_ev=args.skip_belief_ev,
                skip_oracle_ev=args.skip_oracle_ev,
            )
            if row is None:
                n_skipped += 1
                continue
            writer.writerow(row)
            rows.append(row)

            if args.verbose and len(rows) % 50 == 0:
                print(f"  analyzed {len(rows)} decisions...")

            if args.limit is not None and len(rows) >= args.limit:
                break

    summary = summarize(rows)
    summary["input"] = args.input
    summary["output"] = str(out_path)
    summary["n_seen"] = n_seen
    summary["n_skipped"] = n_skipped
    summary["num_rollouts"] = args.num_rollouts
    summary["samples_per_bucket"] = args.samples_per_bucket
    summary["opponent_preset"] = args.opponent_preset

    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if args.verbose or args.limit:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(f"Analyzed {summary['n']} decisions -> {out_path}")
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
