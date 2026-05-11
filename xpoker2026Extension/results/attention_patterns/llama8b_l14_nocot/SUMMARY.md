# Attention-pattern analysis at dominant heads

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Layer: **14**
- Heads analyzed: [5, 23, 24]
- Buckets: ['clean_check_or_call', 'clean_legal_fold', 'clean_bet_or_raise']
- Per-bucket sample size cap: 200
- Top-K positions per (decision, head): 8

## Head 5

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 200 | 2.458 | 0.143 | |
| `clean_legal_fold` | 80 | 2.300 | 0.080 | |
| `clean_bet_or_raise` | 200 | 2.532 | 0.277 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `clean_bet_or_raise` |
|---:|---|---|---|
| 1 | 'ISE' ×222 | '<\|start_header_id\|>' ×128 | 'ISE' ×214 |
| 2 | '<\|start_header_id\|>' ×209 | '<\|begin_of_text\|>' ×80 | '<\|start_header_id\|>' ×208 |
| 3 | '<\|begin_of_text\|>' ×200 | "']\n\n" ×80 | '<\|begin_of_text\|>' ×200 |
| 4 | "']\n\n" ×200 | "'," ×57 | "']\n\n" ×166 |
| 5 | 'CHECK' ×193 | ' CALL' ×57 | ' CALL' ×162 |
| 6 | ' RA' ×154 | 'B' ×56 | ' BET' ×121 |
| 7 | ' CALL' ×81 | ')\n\n' ×49 | 'B' ×116 |
| 8 | 'B' ×79 | 'CHECK' ×32 | "'," ×98 |
| 9 | '_CALL' ×67 | '_CALL' ×22 | 'CHECK' ×86 |
| 10 | 'ET' ×56 | 'OLD' ×21 | 'ET' ×75 |
| 11 | ' Opp' ×46 | ' RA' ×20 | ')\n\n' ×69 |
| 12 | 'OLD' ×27 | ' Opp' ×12 | ')\n' ×30 |
| 13 | "'," ×25 | 'ISE' ×8 | '_RA' ×23 |
| 14 | ')\n\n' ×13 | ' (' ×7 | ' Bet' ×18 |
| 15 | ' BET' ×10 | '_RA' ×5 | ' (' ×8 |
| 16 | '0' ×8 | ' CHECK' ×4 | '0' ×3 |
| 17 | ' (' ×3 | '\n' ×2 | '4' ×1 |
| 18 | '_RA' ×3 | — | '_CALL' ×1 |
| 19 | ')\n' ×3 | — | '7' ×1 |
| 20 | '\n' ×1 | — | — |
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

## Head 23

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 200 | 3.518 | 0.097 | |
| `clean_legal_fold` | 80 | 3.651 | 0.172 | |
| `clean_bet_or_raise` | 200 | 3.496 | 0.189 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `clean_bet_or_raise` |
|---:|---|---|---|
| 1 | '_OR' ×253 | '<\|begin_of_text\|>' ×80 | 'ISE' ×233 |
| 2 | '<\|begin_of_text\|>' ×200 | 'OLD' ×80 | '_OR' ×209 |
| 3 | "']\n\n" ×200 | '_OR' ×80 | '<\|begin_of_text\|>' ×200 |
| 4 | "'," ×180 | "']\n\n" ×80 | ' call' ×142 |
| 5 | 'OLD' ×144 | '<\|start_header_id\|>' ×67 | "'," ×141 |
| 6 | ' call' ×99 | " '" ×59 | "']\n\n" ×132 |
| 7 | " '" ×95 | '\n' ×57 | " '" ×117 |
| 8 | '<\|start_header_id\|>' ×90 | ' CALL' ×33 | 'OLD' ×98 |
| 9 | 'CHECK' ×90 | "'," ×23 | 'CHECK' ×81 |
| 10 | '_CALL' ×88 | ' CHECK' ×21 | '_CALL' ×69 |
| 11 | 'F' ×76 | 'F' ×20 | '<\|start_header_id\|>' ×62 |
| 12 | 'ISE' ×40 | '_CALL' ×14 | ' CALL' ×42 |
| 13 | ' CALL' ×15 | ' call' ×12 | ')\n' ×30 |
| 14 | '\n' ×13 | 'B' ×6 | 'ET' ×21 |
| 15 | '"}' ×6 | ' "' ×4 | ' "' ×16 |
| 16 | ' CHECK' ×4 | '"}' ×3 | '"}' ×6 |
| 17 | 'B' ×3 | '<\|eot_id\|>' ×1 | '<\|eot_id\|>' ×1 |
| 18 | ' "' ×3 | — | — |
| 19 | '<\|eot_id\|>' ×1 | — | — |
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

## Head 24

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 200 | 3.757 | 0.047 | |
| `clean_legal_fold` | 80 | 3.747 | 0.045 | |
| `clean_bet_or_raise` | 200 | 3.691 | 0.066 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `clean_bet_or_raise` |
|---:|---|---|---|
| 1 | '<\|begin_of_text\|>' ×200 | 'assistant' ×80 | '<\|begin_of_text\|>' ×200 |
| 2 | '\n\n' ×200 | '<\|begin_of_text\|>' ×80 | 'B' ×199 |
| 3 | 'assistant' ×198 | '\n\n' ×80 | 'assistant' ×177 |
| 4 | '<\|start_header_id\|>' ×183 | ' RA' ×75 | '<\|start_header_id\|>' ×166 |
| 5 | 'Choose' ×130 | '<\|start_header_id\|>' ×72 | ')\n\n' ×163 |
| 6 | "']\n\n" ×123 | 'Choose' ×67 | '\n\n' ×133 |
| 7 | ')\n\n' ×121 | 'B' ×57 | '\n' ×109 |
| 8 | '\n' ×90 | '\n' ×53 | 'Choose' ×103 |
| 9 | ' Opp' ×83 | "']\n\n" ×23 | "']\n\n" ×92 |
| 10 | 'B' ×81 | ' Opp' ×23 | '":' ×62 |
| 11 | ' RA' ×74 | ')\n\n' ×18 | ' RA' ×51 |
| 12 | '":' ×62 | ' You' ×5 | ')\n' ×34 |
| 13 | '_RA' ×52 | '_CHO' ×4 | '0' ×33 |
| 14 | ' You' ×2 | ' your' ×3 | '_CHO' ×32 |
| 15 | '_CHO' ×1 | — | '_RA' ×31 |
| 16 | — | — | ' your' ×11 |
| 17 | — | — | ' Opp' ×3 |
| 18 | — | — | '4' ×1 |
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
