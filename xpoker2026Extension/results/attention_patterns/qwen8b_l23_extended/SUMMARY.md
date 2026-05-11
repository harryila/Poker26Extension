# Attention-pattern analysis at dominant heads

- Model: `Qwen/Qwen3-8B`
- Layer: **23**
- Heads analyzed: [26, 30, 28]
- Buckets: ['clean_check_or_call', 'clean_legal_fold', 'illegal_fold']
- Per-bucket sample size cap: 200
- Top-K positions per (decision, head): 8

## Head 26

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 200 | 4.228 | 0.120 | |
| `clean_legal_fold` | 200 | 4.249 | 0.133 | |
| `illegal_fold` | 24 | 4.355 | 0.043 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | '.' ×253 | '.' ×302 | '.' ×25 |
| 2 | '<\|im_start\|>' ×200 | '<\|im_start\|>' ×200 | '<\|im_start\|>' ×24 |
| 3 | '.\n\n' ×197 | '.\n\n' ×191 | '.\n\n' ×24 |
| 4 | ' check' ×120 | ')\n\n' ×162 | ' folding' ×21 |
| 5 | '0' ×104 | 'JSON' ×91 | ' "' ×17 |
| 6 | ' Checking' ×94 | ' "' ×82 | "']\n\n" ×14 |
| 7 | ',' ×91 | ' losses' ×79 | ' called' ×13 |
| 8 | ' checking' ×54 | ' fold' ×72 | '":' ×11 |
| 9 | ' hand' ×44 | ' folding' ×66 | '\n' ×10 |
| 10 | ' option' ×42 | ' Folding' ×60 | ')\n\n' ×7 |
| 11 | ' calling' ×42 | ' raise' ×45 | 'JSON' ×6 |
| 12 | ' "' ×34 | '4' ×41 | ' losses' ×6 |
| 13 | 'JSON' ×33 | ')\n' ×24 | ' fold' ×3 |
| 14 | '":' ×26 | ' weak' ×22 | ' option' ×3 |
| 15 | ' or' ×24 | ' option' ×21 | ' river' ×2 |
| 16 | ' call' ×23 | ').' ×18 | ',' ×1 |
| 17 | "']\n\n" ×22 | ',' ×16 | ' choice' ×1 |
| 18 | ')\n\n' ×18 | ' choice' ×14 | ' CALL' ×1 |
| 19 | ' losses' ×15 | '":' ×13 | ').' ×1 |
| 20 | ' risk' ×12 | ' river' ×13 | ' decision' ×1 |
| 21 | ').' ×11 | ':' ×8 | ')\n' ×1 |
| 22 | ' raised' ×11 | "']\n\n" ×6 | — |
| 23 | ' river' ×10 | ' so' ×6 | — |
| 24 | ' is' ×9 | ' again' ×6 | — |
| 25 | ' weak' ×9 | ' checking' ×5 | — |
| 26 | '\n' ×8 | '_CHO' ×4 | — |
| 27 | ' raise' ×7 | ' checked' ×4 | — |
| 28 | '\n\n' ×6 | ' Checking' ×4 | — |
| 29 | ' pair' ×6 | ' here' ×3 | — |
| 30 | ' betting' ×6 | ' bet' ×3 | — |
| 31 | ' yet' ×5 | 'lop' ×3 | — |
| 32 | ' called' ×5 | ' twice' ×2 | — |
| 33 | ' and' ×5 | ' turn' ×2 | — |
| 34 | ' choice' ×4 | ' hand' ×1 | — |
| 35 | ' the' ×4 | ' strong' ×1 | — |
| 36 | 'action' ×4 | ' dry' ×1 | — |
| 37 | ' strong' ×4 | '\n' ×1 | — |
| 38 | ' safe' ×4 | ' situation' ×1 | — |
| 39 | ' streets' ×4 | ' don' ×1 | — |
| 40 | ' improving' ×4 | ')' ×1 | — |

## Head 30

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 200 | 2.781 | 0.348 | |
| `clean_legal_fold` | 200 | 2.621 | 0.337 | |
| `illegal_fold` | 24 | 2.613 | 0.220 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | '<\|im_start\|>' ×200 | '.' ×293 | '.' ×34 |
| 2 | ' or' ×144 | '<\|im_start\|>' ×200 | '<\|im_start\|>' ×24 |
| 3 | ' check' ×123 | ' "' ×182 | ' "' ×24 |
| 4 | '.' ×122 | 'OLD' ×136 | ' folding' ×21 |
| 5 | '_OR' ×121 | ' to' ×122 | '":' ×18 |
| 6 | ',' ×116 | ',' ×91 | '.\n\n' ×17 |
| 7 | ' Checking' ×102 | ' fold' ×73 | ' to' ×15 |
| 8 | ' is' ×93 | ' folding' ×68 | ',' ×12 |
| 9 | ' option' ×60 | ' Folding' ×60 | 'OLD' ×8 |
| 10 | ' checking' ×54 | ' The' ×46 | ' fold' ×4 |
| 11 | ' calling' ×51 | '":' ×39 | ' option' ×3 |
| 12 | "'," ×49 | ' option' ×37 | ' it' ×2 |
| 13 | ' decent' ×48 | '.\n\n' ×35 | ' river' ×2 |
| 14 | ' "' ×37 | ' losses' ×30 | ' choice' ×1 |
| 15 | ' to' ×33 | ' it' ×28 | ').' ×1 |
| 16 | ' call' ×29 | '4' ×21 | ' It' ×1 |
| 17 | '.\n\n' ×26 | ' opponent' ×19 | ' so' ×1 |
| 18 | 'CHECK' ×20 | ' choice' ×18 | ' decision' ×1 |
| 19 | ' and' ×13 | ' avoid' ×14 | ' I' ×1 |
| 20 | '2' ×12 | ' is' ×13 | ' doesn' ×1 |
| 21 | '\n' ×11 | ' consider' ×12 | 'CHECK' ×1 |
| 22 | ' checked' ×11 | ' I' ×9 | — |
| 23 | '":' ×9 | ' so' ×9 | — |
| 24 | ' betting' ×9 | 'ASON' ×5 | — |
| 25 | ').' ×8 | ').' ×5 | — |
| 26 | ' take' ×8 | ' and' ×4 | — |
| 27 | "']\n\n" ×7 | ' weak' ×4 | — |
| 28 | ' but' ×6 | ' safest' ×3 | — |
| 29 | ' no' ×5 | ' calling' ×3 | — |
| 30 | ' again' ×5 | ' don' ×3 | — |
| 31 | '0' ×5 | ' here' ×2 | — |
| 32 | '_CALL' ×5 | ' mistake' ×2 | — |
| 33 | '4' ×4 | '2' ×2 | — |
| 34 | ' choice' ×4 | ' doesn' ×2 | — |
| 35 | ' ' ×3 | ' It' ×2 | — |
| 36 | 'ISE' ×3 | ' call' ×1 | — |
| 37 | ' safest' ×3 | ' might' ×1 | — |
| 38 | '-fold' ×3 | ' little' ×1 | — |
| 39 | 'ET' ×3 | ' Checking' ×1 | — |
| 40 | ' bet' ×3 | '_CHO' ×1 | — |

## Head 28

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 200 | 3.075 | 0.304 | |
| `clean_legal_fold` | 200 | 3.467 | 0.225 | |
| `illegal_fold` | 24 | 3.359 | 0.156 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | '_OR' ×383 | 'OLD' ×448 | 'OLD' ×26 |
| 2 | '<\|im_start\|>' ×200 | '"' ×214 | '<\|im_start\|>' ×24 |
| 3 | '"' ×157 | ' "' ×207 | ' "' ×24 |
| 4 | ' or' ×140 | '<\|im_start\|>' ×200 | '.\n\n' ×23 |
| 5 | 'CHECK' ×133 | '.\n\n' ×107 | ' folding' ×21 |
| 6 | ' check' ×113 | 'CHECK' ×83 | '"' ×20 |
| 7 | ' "' ×108 | ' fold' ×73 | '_OR' ×19 |
| 8 | "'," ×79 | ' folding' ×66 | 'CHECK' ×17 |
| 9 | '_CALL' ×66 | ' Folding' ×60 | '":' ×9 |
| 10 | ' calling' ×47 | '":' ×40 | ' fold' ×4 |
| 11 | ' Checking' ×31 | '_OR' ×32 | ' option' ×2 |
| 12 | 'ISE' ×29 | '_CALL' ×27 | ' it' ×1 |
| 13 | ' call' ×23 | 'ISE' ×9 | ' choice' ×1 |
| 14 | ' checking' ×19 | ' option' ×6 | ' doesn' ×1 |
| 15 | 'ET' ×14 | ' choice' ×5 | — |
| 16 | ' is' ×12 | ' is' ×5 | — |
| 17 | ' betting' ×9 | ' to' ×4 | — |
| 18 | '.\n\n' ×8 | ' calling' ×3 | — |
| 19 | ' and' ×5 | ' don' ×3 | — |
| 20 | '":' ×5 | ' or' ×2 | — |
| 21 | '2' ×4 | 'F' ×1 | — |
| 22 | ' not' ×3 | ' mistake' ×1 | — |
| 23 | '-fold' ×2 | '.' ×1 | — |
| 24 | " '" ×1 | ' and' ×1 | — |
| 25 | 'OLD' ×1 | ' would' ×1 | — |
| 26 | ' fold' ×1 | ' losses' ×1 | — |
| 27 | ' ' ×1 | — | — |
| 28 | ' to' ×1 | — | — |
| 29 | ' again' ×1 | — | — |
| 30 | ' raise' ×1 | — | — |
| 31 | '\n' ×1 | — | — |
| 32 | '.' ×1 | — | — |
| 33 | ' option' ×1 | — | — |
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
