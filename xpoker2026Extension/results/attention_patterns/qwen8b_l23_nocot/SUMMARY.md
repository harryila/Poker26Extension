# Attention-pattern analysis at dominant heads

- Model: `Qwen/Qwen3-8B`
- Layer: **23**
- Heads analyzed: [26, 30, 28]
- Buckets: ['clean_check_or_call', 'clean_legal_fold', 'clean_bet_or_raise']
- Per-bucket sample size cap: 200
- Top-K positions per (decision, head): 8

## Head 26

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 89 | 3.488 | 0.024 | |
| `clean_legal_fold` | 100 | 3.567 | 0.077 | |
| `clean_bet_or_raise` | 100 | 3.648 | 0.064 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `clean_bet_or_raise` |
|---:|---|---|---|
| 1 | '\n' ×93 | ' "' ×190 | '\n' ×285 |
| 2 | '<\|im_start\|>' ×89 | ')\n' ×111 | '<\|im_start\|>' ×100 |
| 3 | ')\n\n' ×89 | '<\|im_start\|>' ×100 | "']\n\n" ×100 |
| 4 | 'lop' ×89 | ')\n\n' ×100 | '0' ×98 |
| 5 | "']\n\n" ×89 | '\n' ×100 | ' "' ×92 |
| 6 | ')\n' ×89 | "'," ×92 | '\n\n' ×89 |
| 7 | '0' ×89 | ' CHECK' ×37 | ')\n\n' ×11 |
| 8 | 'P' ×78 | "']\n\n" ×24 | 'lop' ×11 |
| 9 | 'CHECK' ×6 | '-' ×22 | ')\n' ×11 |
| 10 | ' "' ×1 | 'lop' ×11 | 'CHECK' ×3 |
| 11 | — | '4' ×10 | — |
| 12 | — | '":' ×3 | — |
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

## Head 30

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 89 | 2.993 | 0.031 | |
| `clean_legal_fold` | 100 | 1.871 | 0.083 | |
| `clean_bet_or_raise` | 100 | 2.454 | 0.197 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `clean_bet_or_raise` |
|---:|---|---|---|
| 1 | 'CHECK' ×149 | '<\|im_start\|>' ×111 | '<\|im_start\|>' ×109 |
| 2 | '<\|im_start\|>' ×90 | ' "' ×111 | 'CHECK' ×101 |
| 3 | 'B' ×89 | 'OLD' ×100 | ' "' ×100 |
| 4 | ' "' ×89 | ')\n\n' ×100 | '_RA' ×98 |
| 5 | '_CHO' ×89 | '4' ×89 | 'ET' ×90 |
| 6 | '_OR' ×79 | 'ISE' ×88 | ' CHECK' ×89 |
| 7 | '0' ×49 | ' Opp' ×78 | 'ISE' ×89 |
| 8 | '_CALL' ×44 | '_RA' ×57 | '\n\n' ×71 |
| 9 | '_RA' ×16 | "'," ×55 | 'B' ×21 |
| 10 | 'ET' ×11 | '6' ×11 | '\n' ×19 |
| 11 | '\n' ×5 | — | '_CHO' ×8 |
| 12 | '2' ×1 | — | '4' ×1 |
| 13 | '5' ×1 | — | '7' ×1 |
| 14 | — | — | ' J' ×1 |
| 15 | — | — | '9' ×1 |
| 16 | — | — | ' ' ×1 |
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

## Head 28

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 89 | 2.681 | 0.045 | |
| `clean_legal_fold` | 100 | 3.265 | 0.051 | |
| `clean_bet_or_raise` | 100 | 3.145 | 0.103 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `clean_bet_or_raise` |
|---:|---|---|---|
| 1 | '_OR' ×178 | 'ISE' ×183 | '_OR' ×163 |
| 2 | 'CHECK' ×178 | ' "' ×111 | '_CALL' ×137 |
| 3 | '_CALL' ×177 | '<\|im_start\|>' ×100 | 'CHECK' ×111 |
| 4 | '"' ×89 | '4' ×87 | '"' ×100 |
| 5 | '<\|im_start\|>' ×83 | '"' ×85 | '<\|im_start\|>' ×97 |
| 6 | ' "' ×7 | 'CHECK' ×78 | ' "' ×95 |
| 7 | — | '_CALL' ×78 | 'ISE' ×89 |
| 8 | — | 'OLD' ×49 | 'ET' ×8 |
| 9 | — | '6' ×11 | — |
| 10 | — | 'ET' ×11 | — |
| 11 | — | '_OR' ×7 | — |
| 12 | — | — | — |
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
