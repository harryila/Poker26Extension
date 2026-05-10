# Attention-pattern analysis at dominant heads

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Layer: **14**
- Heads analyzed: [5, 23, 24]
- Buckets: ['clean_check_or_call', 'clean_legal_fold', 'illegal_fold']
- Per-bucket sample size cap: 50
- Top-K positions per (decision, head): 8

## Head 5

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 50 | 2.389 | 0.544 | |
| `clean_legal_fold` | 50 | 2.347 | 0.541 | |
| `illegal_fold` | 50 | 2.197 | 0.476 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | ' to' ×45 | '.' ×90 | '.' ×68 |
| 2 | '":' ×39 | '<\|begin_of_text\|>' ×37 | ' folding' ×43 |
| 3 | '.' ×34 | ' losses' ×30 | ' consider' ×39 |
| 4 | ',' ×33 | ' cautious' ×25 | '<\|begin_of_text\|>' ×33 |
| 5 | ' calling' ×32 | ' folding' ×25 | ' cautious' ×32 |
| 6 | ' consider' ×21 | ',' ×20 | ' and' ×29 |
| 7 | ' and' ×15 | ' to' ×19 | ' losses' ×25 |
| 8 | ' checking' ×14 | ' option' ×19 | ' to' ×22 |
| 9 | ' cautious' ×14 | ' is' ×19 | ',' ×22 |
| 10 | ' should' ×13 | ' consider' ×18 | ' minimize' ×18 |
| 11 | ' call' ×12 | ' so' ×16 | ' so' ×16 |
| 12 | '.\n\n' ×10 | ' minimize' ×15 | ' should' ×13 |
| 13 | ' see' ×10 | ' Folding' ×12 | ' option' ×7 |
| 14 | '<\|begin_of_text\|>' ×9 | ' it' ×9 | ' is' ×7 |
| 15 | ' option' ×8 | ' and' ×8 | '":' ×5 |
| 16 | ' reacts' ×7 | ' fold' ×7 | ' Folding' ×3 |
| 17 | ' or' ×7 | '":' ×5 | ' likely' ×3 |
| 18 | ' but' ×5 | '.\n\n' ×4 | ' it' ×2 |
| 19 | ' information' ×5 | ' bet' ×2 | ' opponent' ×2 |
| 20 | ' called' ×5 | ' safest' ×2 | ' bet' ×2 |
| 21 | ' is' ×4 | ' hand' ×2 | ' raised' ×1 |
| 22 | ' it' ×4 | ' than' ×2 | ' than' ×1 |
| 23 | ' prepared' ×3 | ' should' ×2 | ' hand' ×1 |
| 24 | ' further' ×3 | ' conservative' ×2 | ' Given' ×1 |
| 25 | ' Checking' ×3 | ' action' ×1 | 'ISE' ×1 |
| 26 | ' act' ×2 | ' aggressive' ×1 | ' be' ×1 |
| 27 | ' try' ×2 | ' betting' ×1 | ' action' ×1 |
| 28 | ' possible' ×2 | ' likely' ×1 | ' call' ×1 |
| 29 | ' card' ×2 | ' calling' ×1 | '.\n\n' ×1 |
| 30 | ' so' ×2 | ' but' ×1 | — |
| 31 | ' raised' ×2 | ' call' ×1 | — |
| 32 | ' Betting' ×2 | ' with' ×1 | — |
| 33 | ' good' ×2 | ' loss' ×1 | — |
| 34 | ' way' ×2 | 'wise' ×1 | — |
| 35 | ' raising' ×2 | — | — |
| 36 | ' draw' ×2 | — | — |
| 37 | ' check' ×1 | — | — |
| 38 | ' action' ×1 | — | — |
| 39 | ' protect' ×1 | — | — |
| 40 | ' get' ×1 | — | — |

## Head 23

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 50 | 2.411 | 0.717 | |
| `clean_legal_fold` | 50 | 2.609 | 0.585 | |
| `illegal_fold` | 50 | 2.932 | 0.417 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | ' to' ×50 | '<\|begin_of_text\|>' ×50 | '_OR' ×56 |
| 2 | '<\|begin_of_text\|>' ×48 | 'OLD' ×49 | '<\|begin_of_text\|>' ×50 |
| 3 | '_OR' ×37 | '.\n\n' ×43 | ' folding' ×43 |
| 4 | '":' ×34 | ' "' ×39 | '_CALL' ×43 |
| 5 | ' calling' ×32 | 'F' ×28 | ' "' ×41 |
| 6 | ' call' ×31 | ' folding' ×25 | ' call' ×33 |
| 7 | '_CALL' ×23 | ' to' ×20 | "'," ×30 |
| 8 | ' "' ×20 | ' is' ×19 | '.\n\n' ×23 |
| 9 | '.\n\n' ×18 | ' option' ×19 | ' to' ×22 |
| 10 | ' checking' ×14 | "'," ×16 | ' {"' ×14 |
| 11 | ' or' ×13 | ' {"' ×12 | ' minimize' ×11 |
| 12 | ',' ×12 | ' call' ×12 | ' should' ×8 |
| 13 | ' is' ×10 | ' Folding' ×12 | ' is' ×7 |
| 14 | "'," ×10 | ' minimize' ×11 | ' option' ×7 |
| 15 | ' option' ×5 | ' fold' ×7 | ' bet' ×4 |
| 16 | ' Checking' ×5 | ' or' ×7 | ' Folding' ×3 |
| 17 | ' {"' ×5 | '_OR' ×6 | ' the' ×2 |
| 18 | ' should' ×4 | '.' ×5 | ' cautious' ×1 |
| 19 | '.' ×4 | ' calling' ×5 | ' CALL' ×1 |
| 20 | ' might' ×3 | ' should' ×3 | "']\n\n" ×1 |
| 21 | ' would' ×3 | ' the' ×3 | — |
| 22 | ' checked' ×2 | '_CALL' ×2 | — |
| 23 | ' consider' ×2 | ' continuing' ×1 | — |
| 24 | ' betting' ×2 | ' raise' ×1 | — |
| 25 | ' way' ×2 | "']\n\n" ×1 | — |
| 26 | ' raising' ×2 | ' bet' ×1 | — |
| 27 | ' check' ×1 | ',' ×1 | — |
| 28 | ' while' ×1 | ' conservative' ×1 | — |
| 29 | ' folding' ×1 | ' cautious' ×1 | — |
| 30 | ' but' ×1 | — | — |
| 31 | ' bet' ×1 | — | — |
| 32 | 'OLD' ×1 | — | — |
| 33 | ' cautious' ×1 | — | — |
| 34 | ' it' ×1 | — | — |
| 35 | ' fold' ×1 | — | — |
| 36 | — | — | — |
| 37 | — | — | — |
| 38 | — | — | — |
| 39 | — | — | — |
| 40 | — | — | — |

## Head 24

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 50 | 3.287 | 0.251 | |
| `clean_legal_fold` | 50 | 3.529 | 0.141 | |
| `illegal_fold` | 50 | 3.620 | 0.095 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | '.' ×109 | '.' ×89 | '.' ×89 |
| 2 | '<\|begin_of_text\|>' ×49 | '.\n\n' ×50 | '<\|begin_of_text\|>' ×50 |
| 3 | '.\n\n' ×44 | '<\|begin_of_text\|>' ×50 | '.\n\n' ×50 |
| 4 | '":' ×42 | '":' ×33 | ' folding' ×43 |
| 5 | ' calling' ×24 | ' folding' ×25 | ' cautious' ×30 |
| 6 | ',' ×21 | ' losses' ×25 | ' I' ×30 |
| 7 | ' However' ×18 | ' I' ×21 | ' losses' ×22 |
| 8 | ' I' ×12 | ' cautious' ×17 | '":' ×20 |
| 9 | ' cautious' ×11 | ' option' ×17 | ' it' ×19 |
| 10 | ' call' ×9 | ' Folding' ×12 | ',' ×7 |
| 11 | ' hand' ×8 | ' {"' ×9 | ' option' ×7 |
| 12 | ' checking' ×7 | 'action' ×8 | ' hand' ×6 |
| 13 | ' option' ×6 | ' it' ×7 | ' However' ×5 |
| 14 | ' \n\n' ×6 | ' fold' ×7 | ' potential' ×4 |
| 15 | ' {"' ×4 | "']\n\n" ×6 | ' Folding' ×3 |
| 16 | 'action' ×4 | ',' ×5 | ' {"' ×3 |
| 17 | ' but' ×3 | ' hand' ×3 | ' minimize' ×2 |
| 18 | ' me' ×3 | ' me' ×3 | ' weakness' ×2 |
| 19 | ' act' ×2 | ' doesn' ×3 | ' amount' ×1 |
| 20 | ' action' ×2 | ' However' ×2 | ' mine' ×1 |
| 21 | ' "' ×2 | ' worth' ×1 | '-su' ×1 |
| 22 | ' further' ×2 | ' one' ×1 | ' me' ×1 |
| 23 | ' and' ×2 | ' minimize' ×1 | 'action' ×1 |
| 24 | ' check' ×1 | ' with' ×1 | ' does' ×1 |
| 25 | ' weak' ×1 | ' potential' ×1 | ' beat' ×1 |
| 26 | ' information' ×1 | ' loss' ×1 | ' call' ×1 |
| 27 | ' protect' ×1 | 'wise' ×1 | — |
| 28 | '0' ×1 | ' don' ×1 | — |
| 29 | ' don' ×1 | — | — |
| 30 | ' yet' ×1 | — | — |
| 31 | ' raising' ×1 | — | — |
| 32 | ' loss' ×1 | — | — |
| 33 | ' way' ×1 | — | — |
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
