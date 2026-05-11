# Attention-pattern analysis at dominant heads

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Layer: **14**
- Heads analyzed: [5, 23, 24]
- Buckets: ['clean_check_or_call', 'clean_legal_fold', 'illegal_fold']
- Per-bucket sample size cap: 200
- Top-K positions per (decision, head): 8

## Head 5

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 200 | 2.290 | 0.595 | |
| `clean_legal_fold` | 200 | 2.362 | 0.498 | |
| `illegal_fold` | 68 | 2.189 | 0.478 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | ' to' ×196 | '.' ×328 | '.' ×98 |
| 2 | ',' ×167 | ' losses' ×131 | ' folding' ×56 |
| 3 | '":' ×143 | ' folding' ×131 | ' consider' ×52 |
| 4 | '.' ×137 | ' cautious' ×113 | '<\|begin_of_text\|>' ×43 |
| 5 | ' calling' ×117 | '<\|begin_of_text\|>' ×108 | ' cautious' ×41 |
| 6 | ' consider' ×88 | ',' ×95 | ' and' ×38 |
| 7 | ' and' ×60 | ' consider' ×92 | ' losses' ×35 |
| 8 | ' should' ×57 | ' to' ×91 | ',' ×31 |
| 9 | ' call' ×54 | ' is' ×63 | ' to' ×29 |
| 10 | ' checking' ×49 | ' and' ×62 | ' minimize' ×27 |
| 11 | ' cautious' ×45 | ' option' ×59 | ' so' ×19 |
| 12 | '<\|begin_of_text\|>' ×34 | ' minimize' ×57 | ' should' ×18 |
| 13 | ' or' ×34 | ' so' ×53 | ' is' ×10 |
| 14 | ' option' ×33 | ' it' ×39 | ' option' ×8 |
| 15 | ' see' ×28 | ' Folding' ×34 | '":' ×7 |
| 16 | '.\n\n' ×27 | '":' ×27 | ' Folding' ×6 |
| 17 | ' information' ×22 | ' should' ×14 | ' it' ×3 |
| 18 | ' called' ×22 | '.\n\n' ×11 | ' likely' ×3 |
| 19 | ' but' ×19 | ' fold' ×10 | ' opponent' ×2 |
| 20 | ' reacts' ×19 | ' calling' ×10 | ' bet' ×2 |
| 21 | ' check' ×17 | ' bet' ×7 | ' than' ×2 |
| 22 | ' folding' ×12 | ' I' ×6 | '.\n\n' ×2 |
| 23 | ' draw' ×11 | ' safest' ×5 | ' choice' ×2 |
| 24 | ' betting' ×10 | ' hand' ×5 | ' raised' ×1 |
| 25 | ' card' ×9 | ' than' ×4 | ' hand' ×1 |
| 26 | ' I' ×9 | ' likely' ×4 | ' Given' ×1 |
| 27 | ' Checking' ×7 | ' conservative' ×4 | 'ISE' ×1 |
| 28 | ' so' ×7 | ' choice' ×4 | ' be' ×1 |
| 29 | ' good' ×7 | ' loss' ×3 | ' action' ×1 |
| 30 | ' prepared' ×6 | ' expensive' ×3 | ' call' ×1 |
| 31 | ' is' ×6 | ' aggressive' ×2 | ' conservative' ×1 |
| 32 | ' possible' ×6 | ' with' ×2 | ' they' ×1 |
| 33 | ' it' ×6 | ' pot' ×2 | ' best' ×1 |
| 34 | ' better' ×6 | ' possible' ×2 | — |
| 35 | ' bet' ×6 | ' action' ×1 | — |
| 36 | ' way' ×6 | ' betting' ×1 | — |
| 37 | ' react' ×6 | ' but' ×1 | — |
| 38 | ' raising' ×6 | ' call' ×1 | — |
| 39 | "'ll" ×6 | 'wise' ×1 | — |
| 40 | ' reaction' ×5 | ' best' ×1 | — |

## Head 23

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 200 | 2.340 | 0.727 | |
| `clean_legal_fold` | 200 | 2.626 | 0.591 | |
| `illegal_fold` | 68 | 2.924 | 0.424 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | ' to' ×210 | '<\|begin_of_text\|>' ×200 | '_OR' ×75 |
| 2 | '<\|begin_of_text\|>' ×193 | 'OLD' ×196 | '<\|begin_of_text\|>' ×68 |
| 3 | '_OR' ×155 | '.\n\n' ×158 | '_CALL' ×58 |
| 4 | ' call' ×127 | ' "' ×158 | ' folding' ×56 |
| 5 | '":' ×124 | ' folding' ×131 | ' "' ×55 |
| 6 | ' calling' ×118 | 'F' ×126 | ' call' ×45 |
| 7 | '_CALL' ×95 | ' to' ×95 | "'," ×41 |
| 8 | ' "' ×88 | ' minimize' ×74 | '.\n\n' ×33 |
| 9 | '.\n\n' ×78 | ' is' ×63 | ' to' ×30 |
| 10 | ' checking' ×48 | ' option' ×60 | ' {"' ×20 |
| 11 | ' or' ×45 | ' {"' ×48 | ' minimize' ×12 |
| 12 | "'," ×37 | "'," ×39 | ' should' ×10 |
| 13 | ' is' ×36 | ' call' ×35 | ' is' ×10 |
| 14 | ',' ×36 | ' Folding' ×34 | ' option' ×8 |
| 15 | ' option' ×27 | '_OR' ×23 | ' Folding' ×6 |
| 16 | ' check' ×19 | ' or' ×23 | ' bet' ×4 |
| 17 | ' should' ×18 | ' calling' ×21 | ' the' ×3 |
| 18 | '.' ×17 | ' should' ×19 | ' CALL' ×2 |
| 19 | ' {"' ×16 | '.' ×19 | ' choice' ×2 |
| 20 | ' and' ×12 | ' the' ×13 | ' calling' ×2 |
| 21 | ' Checking' ×9 | ' fold' ×10 | ' cautious' ×1 |
| 22 | ' folding' ×9 | ' cautious' ×10 | "']\n\n" ×1 |
| 23 | ' might' ×7 | '_CALL' ×5 | ' or' ×1 |
| 24 | ' way' ×6 | ' bet' ×4 | '":' ×1 |
| 25 | ' CALL' ×6 | ',' ×4 | — |
| 26 | ' consider' ×5 | ' choice' ×4 | — |
| 27 | ' but' ×5 | " '" ×4 | — |
| 28 | ' betting' ×4 | ' CALL' ×3 | — |
| 29 | ' bet' ×4 | '":' ×3 | — |
| 30 | 'OLD' ×4 | ' might' ×3 | — |
| 31 | ' fold' ×4 | ' raise' ×2 | — |
| 32 | ' checked' ×3 | ' would' ×2 | — |
| 33 | ' would' ×3 | ' checking' ×2 | — |
| 34 | ' raising' ×3 | ' continuing' ×1 | — |
| 35 | ' bluff' ×3 | "']\n\n" ×1 | — |
| 36 | ' cautious' ×2 | ' conservative' ×1 | — |
| 37 | ' good' ×2 | ' consider' ×1 | — |
| 38 | ' called' ×2 | ' pot' ×1 | — |
| 39 | 'F' ×2 | ' and' ×1 | — |
| 40 | ' raise' ×2 | ' Betting' ×1 | — |

## Head 24

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 200 | 3.260 | 0.234 | |
| `clean_legal_fold` | 200 | 3.525 | 0.132 | |
| `illegal_fold` | 68 | 3.602 | 0.101 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | '.' ×421 | '.' ×328 | '.' ×126 |
| 2 | '<\|begin_of_text\|>' ×197 | '<\|begin_of_text\|>' ×200 | '<\|begin_of_text\|>' ×68 |
| 3 | '.\n\n' ×181 | '.\n\n' ×199 | '.\n\n' ×68 |
| 4 | '":' ×161 | ' folding' ×130 | ' folding' ×56 |
| 5 | ' calling' ×95 | '":' ×125 | ' cautious' ×39 |
| 6 | ' However' ×77 | ' losses' ×120 | ' I' ×38 |
| 7 | ',' ×77 | ' cautious' ×94 | ' losses' ×32 |
| 8 | ' I' ×59 | ' I' ×70 | '":' ×29 |
| 9 | ' call' ×43 | ' option' ×57 | ' it' ×22 |
| 10 | ' cautious' ×32 | ',' ×46 | ' option' ×10 |
| 11 | 'action' ×32 | ' Folding' ×30 | ',' ×9 |
| 12 | ' option' ×30 | ' {"' ×29 | ' However' ×8 |
| 13 | ' checking' ×28 | ' it' ×27 | ' Folding' ×6 |
| 14 | ' hand' ×23 | 'action' ×20 | ' hand' ×6 |
| 15 | ' {"' ×21 | ' However' ×18 | ' minimize' ×5 |
| 16 | ' \n\n' ×19 | ' minimize' ×12 | ' potential' ×4 |
| 17 | ' check' ×11 | ' fold' ×10 | ' {"' ×3 |
| 18 | ' but' ×10 | ' doesn' ×10 | ' weakness' ×3 |
| 19 | '0' ×9 | ' hand' ×9 | ' me' ×2 |
| 20 | ' and' ×8 | ' me' ×9 | ' choice' ×2 |
| 21 | ' me' ×4 | ' potential' ×9 | ' amount' ×1 |
| 22 | ' reaction' ×4 | ' don' ×9 | ' mine' ×1 |
| 23 | ' it' ×4 | "']\n\n" ×8 | '-su' ×1 |
| 24 | ' "' ×3 | ' loss' ×3 | 'action' ×1 |
| 25 | ' protect' ×3 | ' expensive' ×3 | ' does' ×1 |
| 26 | ' further' ×3 | ' worth' ×2 | ' beat' ×1 |
| 27 | ' loss' ×3 | ' choice' ×2 | ' call' ×1 |
| 28 | ' way' ×3 | 'assistant' ×2 | ' worth' ×1 |
| 29 | ' draw' ×3 | ' not' ×2 | — |
| 30 | ' act' ×2 | ' calling' ×2 | — |
| 31 | ' action' ×2 | ' field' ×2 | — |
| 32 | ' don' ×2 | ' one' ×1 | — |
| 33 | ' yet' ×2 | ' with' ×1 | — |
| 34 | ' raising' ×2 | 'wise' ×1 | — |
| 35 | ' Checking' ×2 | ' mine' ×1 | — |
| 36 | ' potential' ×2 | ' \n\n' ×1 | — |
| 37 | ' better' ×2 | ' safest' ×1 | — |
| 38 | ' small' ×2 | ' and' ×1 | — |
| 39 | ' weak' ×1 | ':' ×1 | — |
| 40 | ' information' ×1 | ' odds' ×1 | — |

## Interpretation guide

- **If the top attended tokens differ markedly between `clean_check_or_call` and `illegal_fold`**: the head is reading different context for different verb decisions — strong evidence that the head is doing decision-relevant computation, not just attending to format/structural tokens.
- **If the top tokens are mostly format tokens (e.g., `:`, newlines, `"`, prompt-section labels)**: the head is doing structural attention, not content-based decision-making. The decision signal would have to come from somewhere else (other heads, MLP, residual flow-through).
- **Mean entropy comparison**: heads with low entropy (~1-3 nats) on a long sequence (1000+ tokens) are sharply focused. If entropy differs systematically across buckets, the head's focus *itself* depends on which decision is being made.
- **Same top tokens across buckets but different ranks/weights**: the head looks at the same context but weighs it differently — consistent with a 'soft router' that emphasizes one feature (e.g., 'Bet to call:' line) for CHECK decisions and another (e.g., 'Stack:') for FOLD decisions.
