# Action-verb token audit

For every logit-lens sidecar, scans the FINAL-layer top-1 token at
the JSON action-verb position. Reports how many records map to a known
action family (FOLD/CHECK/CALL/BET/RAISE) and how many don't.

Uncovered tokens at the verb position are CANDIDATES for new
aliases. If the count is non-trivial (>1% of records in a cell),
consider adding the token's first piece to ACTION_TOKEN_GROUPS or
ACTION_SUBWORD_PREFIXES in `analyze_logit_lens_by_failure_mode.py`.

## Cross-cell summary

| Cell | Records | With verb pos | Covered | Uncovered | Coverage % |
|---|---:|---:|---:|---:|---:|
| `llama8b_t02_s123` | 317 | 317 | 317 | 0 | 100.0% |
| `llama8b_t02_s42` | 355 | 355 | 355 | 0 | 100.0% |
| `llama8b_t02_s456` | 311 | 311 | 311 | 0 | 100.0% |
| `llama8b_t0_s123` | 351 | 351 | 351 | 0 | 100.0% |
| `llama8b_t0_s42` | 350 | 350 | 350 | 0 | 100.0% |
| `llama8b_t0_s456` | 316 | 316 | 316 | 0 | 100.0% |
| `ministral8b_t02_s123` | 106 | 106 | 106 | 0 | 100.0% |
| `ministral8b_t02_s42` | 309 | 308 | 308 | 0 | 100.0% |
| `ministral8b_t02_s456` | 108 | 108 | 108 | 0 | 100.0% |
| `ministral8b_t0_s123` | 102 | 102 | 102 | 0 | 100.0% |
| `ministral8b_t0_s42` | 313 | 313 | 313 | 0 | 100.0% |
| `ministral8b_t0_s456` | 106 | 106 | 106 | 0 | 100.0% |
| `qwen8b_t02_s123` | 394 | 394 | 394 | 0 | 100.0% |
| `qwen8b_t02_s42` | 460 | 460 | 460 | 0 | 100.0% |
| `qwen8b_t02_s456` | 330 | 330 | 330 | 0 | 100.0% |
| `qwen8b_t0_s123` | 350 | 350 | 350 | 0 | 100.0% |
| `qwen8b_t0_s42` | 455 | 455 | 455 | 0 | 100.0% |
| `qwen8b_t0_s456` | 357 | 357 | 357 | 0 | 100.0% |

## Aggregate uncovered tokens (across all cells)

_No uncovered tokens found. Coverage is complete._

## Per-cell uncovered detail

### `llama8b_t02_s123`

_All verb-position tokens covered._

### `llama8b_t02_s42`

_All verb-position tokens covered._

### `llama8b_t02_s456`

_All verb-position tokens covered._

### `llama8b_t0_s123`

_All verb-position tokens covered._

### `llama8b_t0_s42`

_All verb-position tokens covered._

### `llama8b_t0_s456`

_All verb-position tokens covered._

### `ministral8b_t02_s123`

_All verb-position tokens covered._

### `ministral8b_t02_s42`

_All verb-position tokens covered._

### `ministral8b_t02_s456`

_All verb-position tokens covered._

### `ministral8b_t0_s123`

_All verb-position tokens covered._

### `ministral8b_t0_s42`

_All verb-position tokens covered._

### `ministral8b_t0_s456`

_All verb-position tokens covered._

### `qwen8b_t02_s123`

_All verb-position tokens covered._

### `qwen8b_t02_s42`

_All verb-position tokens covered._

### `qwen8b_t02_s456`

_All verb-position tokens covered._

### `qwen8b_t0_s123`

_All verb-position tokens covered._

### `qwen8b_t0_s42`

_All verb-position tokens covered._

### `qwen8b_t0_s456`

_All verb-position tokens covered._

