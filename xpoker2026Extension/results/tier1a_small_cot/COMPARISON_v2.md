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
| `cot_llama8b_t02_s123_informative_v2_enriched.jsonl` | 528 |  55.5 |  55.5 |   0.0 |  40.9 |   0.0 |   3.6 |
| `cot_llama8b_t02_s42_informative_v2_enriched.jsonl` | 646 |  53.1 |  53.1 |   0.0 |  41.5 |   0.0 |   5.4 |
| `cot_llama8b_t02_s456_informative_v2_enriched.jsonl` | 542 |  56.5 |  56.5 |   0.0 |  39.5 |   0.0 |   4.1 |
| `cot_llama8b_t0_s123_informative_v2_enriched.jsonl` | 596 |  53.2 |  53.2 |   0.0 |  41.1 |   0.0 |   5.7 |
| `cot_llama8b_t0_s42_informative_v2_enriched.jsonl` | 610 |  54.8 |  54.8 |   0.0 |  42.6 |   0.0 |   2.6 |
| `cot_llama8b_t0_s456_informative_v2_enriched.jsonl` | 524 |  56.9 |  56.9 |   0.0 |  39.7 |   0.0 |   3.4 |
| `cot_ministral8b_t02_s123_informative_v2_enriched.jsonl` | 206 |  49.5 |  49.5 |   0.0 |  49.5 |   0.0 |   1.0 |
| `cot_ministral8b_t02_s42_informative_v2_enriched.jsonl` | 519 |  23.5 |  28.7 |   5.2 |  40.5 |   0.0 |  30.8 |
| `cot_ministral8b_t02_s456_informative_v2_enriched.jsonl` | 221 |  48.4 |  48.9 |   0.5 |  48.4 |   0.0 |   2.7 |
| `cot_ministral8b_t0_s123_informative_v2_enriched.jsonl` | 203 |  49.8 |  49.8 |   0.0 |  49.8 |   0.0 |   0.5 |
| `cot_ministral8b_t0_s42_informative_v2_enriched.jsonl` | 525 |  21.7 |  25.5 |   3.8 |  40.4 |   0.0 |  34.1 |
| `cot_ministral8b_t0_s456_informative_v2_enriched.jsonl` | 209 |  49.3 |  49.3 |   0.0 |  49.3 |   0.0 |   1.4 |
| `cot_qwen8b_t02_s123_informative_v2_enriched.jsonl` | 612 |  54.2 |  54.2 |   0.0 |  43.6 |   0.0 |   2.1 |
| `cot_qwen8b_t02_s42_informative_v2_enriched.jsonl` | 804 |  56.8 |  56.8 |   0.0 |  42.3 |   0.0 |   0.9 |
| `cot_qwen8b_t02_s456_informative_v2_enriched.jsonl` | 568 |  56.3 |  56.3 |   0.0 |  41.7 |   0.0 |   1.9 |
| `cot_qwen8b_t0_s123_informative_v2_enriched.jsonl` | 603 |  56.4 |  56.4 |   0.0 |  42.0 |   0.0 |   1.7 |
| `cot_qwen8b_t0_s42_informative_v2_enriched.jsonl` | 796 |  56.4 |  56.4 |   0.0 |  42.8 |   0.0 |   0.8 |
| `cot_qwen8b_t0_s456_informative_v2_enriched.jsonl` | 616 |  56.0 |  56.0 |   0.0 |  42.0 |   0.0 |   1.9 |

## Failure-mode detail (top illegal-action contexts and unknown aliases)

### `cot_llama8b_t02_s123_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **19**

### `cot_llama8b_t02_s42_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **35**

### `cot_llama8b_t02_s456_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **22**

### `cot_llama8b_t0_s123_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **34**

### `cot_llama8b_t0_s42_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **16**

### `cot_llama8b_t0_s456_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **18**

### `cot_ministral8b_t02_s123_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **2**

### `cot_ministral8b_t02_s42_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **160**

### `cot_ministral8b_t02_s456_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **6**

### `cot_ministral8b_t0_s123_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **1**

### `cot_ministral8b_t0_s42_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **179**

### `cot_ministral8b_t0_s456_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **3**

### `cot_qwen8b_t02_s123_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **13**

### `cot_qwen8b_t02_s42_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **6**
- `BET_OR_RAISE not in ['CHECK_OR_CALL', 'FOLD']` — **1**

### `cot_qwen8b_t02_s456_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **10**
- `BET_OR_RAISE not in ['CHECK_OR_CALL', 'FOLD']` — **1**

### `cot_qwen8b_t0_s123_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **9**
- `BET_OR_RAISE not in ['CHECK_OR_CALL', 'FOLD']` — **1**

### `cot_qwen8b_t0_s42_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **4**
- `BET_OR_RAISE not in ['CHECK_OR_CALL', 'FOLD']` — **2**

### `cot_qwen8b_t0_s456_informative_v2_enriched.jsonl`
Top illegal-in-context attempts:
- `FOLD not in ['BET_OR_RAISE', 'CHECK_OR_CALL']` — **11**
- `BET_OR_RAISE not in ['CHECK_OR_CALL', 'FOLD']` — **1**

## CoT vs baseline (parse_v2%)

| Cell | baseline n | baseline parse_v2% | CoT n | CoT parse_v2% | Δ (CoT − baseline) |
|---|---:|---:|---:|---:|---:|
| `llama8b_t02_s123_informative_v2` | 1374 |  54.4 | 528 |  55.5 | +1.1 |
| `llama8b_t02_s42_informative_v2` | 1624 |  55.3 | 646 |  53.1 | -2.2 |
| `llama8b_t02_s456_informative_v2` | 1382 |  55.8 | 542 |  56.5 | +0.7 |
| `llama8b_t0_s123_informative_v2` | 1444 |  54.1 | 596 |  53.2 | -0.9 |
| `llama8b_t0_s42_informative_v2` | 1695 |  55.2 | 610 |  54.8 | -0.4 |
| `llama8b_t0_s456_informative_v2` | 1470 |  55.5 | 524 |  56.9 | +1.4 |
| `ministral8b_t02_s123_informative_v2` | 200 |  50.0 | 206 |  49.5 | -0.5 |
| `ministral8b_t02_s42_informative_v2` | 429 |  53.4 | 519 |  28.7 | -24.7 |
| `ministral8b_t02_s456_informative_v2` | 200 |  50.0 | 221 |  48.9 | -1.1 |
| `ministral8b_t0_s123_informative_v2` | 200 |  50.0 | 203 |  49.8 | -0.2 |
| `ministral8b_t0_s42_informative_v2` | 415 |  51.8 | 525 |  25.5 | -26.3 |
| `ministral8b_t0_s456_informative_v2` | 200 |  50.0 | 209 |  49.3 | -0.7 |
| `qwen8b_t02_s123_informative_v2` | 651 |  57.3 | 612 |  54.2 | -3.0 |
| `qwen8b_t02_s42_informative_v2` | 490 |  59.2 | 804 |  56.8 | -2.3 |
| `qwen8b_t02_s456_informative_v2` | 664 |  58.1 | 568 |  56.3 | -1.8 |
| `qwen8b_t0_s123_informative_v2` | 620 |  57.4 | 603 |  56.4 | -1.0 |
| `qwen8b_t0_s42_informative_v2` | 489 |  59.1 | 796 |  56.4 | -2.7 |
| `qwen8b_t0_s456_informative_v2` | 657 |  58.0 | 616 |  56.0 | -2.0 |
