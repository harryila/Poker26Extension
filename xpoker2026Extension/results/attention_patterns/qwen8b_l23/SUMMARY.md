# Attention-pattern analysis at dominant heads

- Model: `Qwen/Qwen3-8B`
- Layer: **23**
- Heads analyzed: [26, 30, 2]
- Buckets: ['clean_check_or_call', 'clean_legal_fold', 'illegal_fold']
- Per-bucket sample size cap: 50
- Top-K positions per (decision, head): 8

## Head 26

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 50 | 4.218 | 0.121 | |
| `clean_legal_fold` | 50 | 4.323 | 0.078 | |
| `illegal_fold` | 24 | 4.355 | 0.043 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | '.' ×68 | '.' ×82 | '.' ×25 |
| 2 | '<\|im_start\|>' ×50 | '<\|im_start\|>' ×50 | '<\|im_start\|>' ×24 |
| 3 | '.\n\n' ×49 | '.\n\n' ×50 | '.\n\n' ×24 |
| 4 | ' check' ×33 | ')\n\n' ×39 | ' folding' ×21 |
| 5 | ',' ×26 | 'JSON' ×28 | ' "' ×17 |
| 6 | ' Checking' ×25 | ' "' ×23 | "']\n\n" ×14 |
| 7 | '0' ×24 | ' folding' ×21 | ' called' ×13 |
| 8 | ' hand' ×12 | ' fold' ×20 | '":' ×11 |
| 9 | ' "' ×11 | ' losses' ×20 | '\n' ×10 |
| 10 | '":' ×10 | ' Folding' ×8 | ')\n\n' ×7 |
| 11 | ' calling' ×9 | ' option' ×8 | 'JSON' ×6 |
| 12 | ' checking' ×8 | ').' ×7 | ' losses' ×6 |
| 13 | ' call' ×8 | ',' ×6 | ' fold' ×3 |
| 14 | ' or' ×8 | ' river' ×4 | ' option' ×3 |
| 15 | ')\n\n' ×6 | ' choice' ×4 | ' river' ×2 |
| 16 | "']\n\n" ×5 | '_CHO' ×3 | ',' ×1 |
| 17 | ' option' ×4 | ':' ×3 | ' choice' ×1 |
| 18 | ').' ×4 | '":' ×3 | ' CALL' ×1 |
| 19 | ' raised' ×4 | ' checked' ×3 | ').' ×1 |
| 20 | ' losses' ×4 | ' so' ×3 | ' decision' ×1 |
| 21 | 'JSON' ×4 | ' here' ×2 | ')\n' ×1 |
| 22 | ' river' ×3 | ')\n' ×2 | — |
| 23 | '\n' ×3 | "']\n\n" ×2 | — |
| 24 | ' is' ×2 | ' hand' ×1 | — |
| 25 | ' risk' ×2 | ' raise' ×1 | — |
| 26 | ' yet' ×2 | ' checking' ×1 | — |
| 27 | ' pair' ×2 | ' strong' ×1 | — |
| 28 | ' choice' ×1 | ' dry' ×1 | — |
| 29 | ' the' ×1 | ' Checking' ×1 | — |
| 30 | ' CALL' ×1 | ' again' ×1 | — |
| 31 | ' fold' ×1 | '\n' ×1 | — |
| 32 | ' weak' ×1 | ' bet' ×1 | — |
| 33 | ' develops' ×1 | — | — |
| 34 | '\n\n' ×1 | — | — |
| 35 | ' better' ×1 | — | — |
| 36 | ' folding' ×1 | — | — |
| 37 | 'action' ×1 | — | — |
| 38 | ':' ×1 | — | — |
| 39 | ' strong' ×1 | — | — |
| 40 | ' raise' ×1 | — | — |

## Head 30

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 50 | 2.828 | 0.356 | |
| `clean_legal_fold` | 50 | 2.522 | 0.287 | |
| `illegal_fold` | 24 | 2.613 | 0.220 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | '<\|im_start\|>' ×50 | '.' ×65 | '.' ×34 |
| 2 | ' or' ×35 | '<\|im_start\|>' ×50 | '<\|im_start\|>' ×24 |
| 3 | ' check' ×33 | 'OLD' ×47 | ' "' ×24 |
| 4 | '.' ×32 | ' "' ×43 | ' folding' ×21 |
| 5 | ',' ×31 | ' to' ×34 | '":' ×18 |
| 6 | '_OR' ×29 | ' folding' ×23 | '.\n\n' ×17 |
| 7 | ' Checking' ×27 | ' fold' ×20 | ' to' ×15 |
| 8 | ' is' ×22 | ',' ×20 | ',' ×12 |
| 9 | ' to' ×12 | ' option' ×13 | 'OLD' ×8 |
| 10 | ' decent' ×12 | '.\n\n' ×12 | ' fold' ×4 |
| 11 | "'," ×12 | ' losses' ×9 | ' option' ×3 |
| 12 | ' "' ×11 | ' Folding' ×8 | ' it' ×2 |
| 13 | ' calling' ×10 | '":' ×7 | ' river' ×2 |
| 14 | ' checking' ×9 | ' it' ×6 | ' choice' ×1 |
| 15 | '.\n\n' ×9 | ' I' ×6 | ').' ×1 |
| 16 | ' call' ×9 | ' is' ×6 | ' It' ×1 |
| 17 | ' option' ×7 | ' choice' ×4 | ' so' ×1 |
| 18 | ' checked' ×5 | ' avoid' ×3 | ' decision' ×1 |
| 19 | ').' ×4 | ' The' ×3 | ' I' ×1 |
| 20 | '\n' ×4 | ' so' ×3 | ' doesn' ×1 |
| 21 | ' and' ×3 | ' safest' ×2 | 'CHECK' ×1 |
| 22 | ' no' ×3 | ' and' ×2 | — |
| 23 | '2' ×3 | ' opponent' ×2 | — |
| 24 | ' ' ×2 | 'ASON' ×2 | — |
| 25 | '":' ×2 | ' consider' ×2 | — |
| 26 | '4' ×2 | ' here' ×1 | — |
| 27 | ' avoid' ×2 | ' call' ×1 | — |
| 28 | ' but' ×2 | '4' ×1 | — |
| 29 | ' again' ×2 | ').' ×1 | — |
| 30 | 'CHECK' ×2 | ' mistake' ×1 | — |
| 31 | ' river' ×1 | ' calling' ×1 | — |
| 32 | ' not' ×1 | ' might' ×1 | — |
| 33 | ' choice' ×1 | ' weak' ×1 | — |
| 34 | ' betting' ×1 | — | — |
| 35 | ' take' ×1 | — | — |
| 36 | ' weak' ×1 | — | — |
| 37 | ' fold' ×1 | — | — |
| 38 | ' The' ×1 | — | — |
| 39 | ' should' ×1 | — | — |
| 40 | 'ISE' ×1 | — | — |

## Head 2

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 50 | 2.237 | 0.111 | |
| `clean_legal_fold` | 50 | 2.373 | 0.081 | |
| `illegal_fold` | 24 | 2.344 | 0.045 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | '":' ×50 | '":' ×50 | '":' ×24 |
| 2 | 'action' ×50 | 'action' ×50 | 'action' ×24 |
| 3 | ':' ×50 | 'JSON' ×50 | ' "' ×24 |
| 4 | ' "' ×50 | ' "' ×50 | '<\|im_start\|>' ×24 |
| 5 | '.\n\n' ×50 | '<\|im_start\|>' ×50 | '.\n\n' ×24 |
| 6 | 'JSON' ×50 | '.\n\n' ×50 | ':' ×24 |
| 7 | '<\|im_start\|>' ×50 | ':' ×50 | 'JSON' ×24 |
| 8 | ' {"' ×46 | ' {"' ×48 | ' {"' ×23 |
| 9 | ' don' ×1 | '.' ×1 | ',' ×1 |
| 10 | ' situation' ×1 | ' don' ×1 | — |
| 11 | ' option' ×1 | — | — |
| 12 | ' to' ×1 | — | — |
| 13 | — | — | — |
| 14 | — | — | — |
| 15 | — | — | — |
| 16 | — | — | — |
| 17 | — | — | — |
| 18 | — | — | — |
| 19 | — | — | — |
| 20 | — | — | — |
| 21 | — | — | — |
| 22 | — | — | — |
| 23 | — | — | — |
| 24 | — | — | — |
| 25 | — | — | — |
| 26 | — | — | — |
| 27 | — | — | — |
| 28 | — | — | — |
| 29 | — | — | — |
| 30 | — | — | — |
| 31 | — | — | — |
| 32 | — | — | — |
| 33 | — | — | — |
| 34 | — | — | — |
| 35 | — | — | — |
| 36 | — | — | — |
| 37 | — | — | — |
| 38 | — | — | — |
| 39 | — | — | — |
| 40 | — | — | — |

## Interpretation guide

- **If the top attended tokens differ markedly between `clean_check_or_call` and `illegal_fold`**: the head is reading different context for different verb decisions — strong evidence that the head is doing decision-relevant computation, not just attending to format/structural tokens.
- **If the top tokens are mostly format tokens (e.g., `:`, newlines, `"`, prompt-section labels)**: the head is doing structural attention, not content-based decision-making. The decision signal would have to come from somewhere else (other heads, MLP, residual flow-through).
- **Mean entropy comparison**: heads with low entropy (~1-3 nats) on a long sequence (1000+ tokens) are sharply focused. If entropy differs systematically across buckets, the head's focus *itself* depends on which decision is being made.
- **Same top tokens across buckets but different ranks/weights**: the head looks at the same context but weighs it differently — consistent with a 'soft router' that emphasizes one feature (e.g., 'Bet to call:' line) for CHECK decisions and another (e.g., 'Stack:') for FOLD decisions.
