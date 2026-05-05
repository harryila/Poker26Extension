# Action-metadata re-attribution (v2)

This report re-categorizes action-metadata failures from CoT logs using
the alias-aware logic in `poker_env/agents/json_utils.py::normalize_action_str`
and a finer-grained split of "why fallback fired".

Columns:

* `n` — total decisions in the cell.
* `parse_v1%` — original `parse_success` rate (json AND recognized AND legal,
   pre-alias).
* `parse_v2%` — same, but using ACTION_ALIASES (CHECK -> CHECK_OR_CALL, etc.).
* `Δ recovered%` — points recovered purely by alias normalization
   (i.e. the in-code fix in `json_utils.py`). This is the upper bound
   of what the alias fix would have rescued in already-collected data.
* `JSON-fail%` — `parse_v2=False` because no valid JSON object was returned.
* `Alias-unk%` — `parse_v2=False` because the model emitted a string we
   still don't recognize even after normalization (NOT fixable in code;
   add another alias if it shows up frequently).
* `Illegal%` — `parse_v2=False` because the model picked a recognized
   action that wasn't in `legal_actions` (e.g. FOLD when bet_to_call=0).
   This is a *model behavior* observation, NOT a parser bug.


## Per-cell breakdown

| Log | n | parse_v1% | parse_v2% | Δ recovered% | JSON-fail% | Alias-unk% | Illegal% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `phase2_70b_t02_s42_informative_v2_enriched.jsonl` | 2345 |  45.2 |  57.8 |  12.6 |  42.2 |   0.0 |   0.0 |
| `phase2_70b_t0_s42_informative_v2_enriched.jsonl` | 2546 |  45.7 |  56.8 |  11.0 |  43.2 |   0.0 |   0.0 |
| `sanity_70b_t02_s123_informative_enriched.jsonl` | 333 |  47.7 |  55.9 |   8.1 |  44.1 |   0.0 |   0.0 |
| `sanity_70b_t02_s42_informative_enriched.jsonl` | 331 |  43.8 |  56.5 |  12.7 |  43.5 |   0.0 |   0.0 |
| `sanity_70b_t0_s123_informative_enriched.jsonl` | 353 |  47.0 |  56.1 |   9.1 |  43.9 |   0.0 |   0.0 |
| `sanity_70b_t0_s42_informative_enriched.jsonl` | 384 |  43.8 |  57.6 |  13.8 |  42.4 |   0.0 |   0.0 |
| `sanity_8b_t0_s42_enriched.jsonl` | 450 |  44.4 |  44.4 |   0.0 |  55.6 |   0.0 |   0.0 |

## Failure-mode detail (top illegal-action contexts and unknown aliases)

### `phase2_70b_t02_s42_informative_v2_enriched.jsonl`
Unknown action strings (after normalization):
- `CALL_OR_RAISE` — **1**

