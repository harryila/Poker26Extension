# Attention-pattern analysis at dominant heads

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Layer: **16**
- Heads analyzed: [22, 9, 15]
- Buckets: ['clean_check_or_call', 'clean_legal_fold', 'illegal_fold']
- Per-bucket sample size cap: 200
- Top-K positions per (decision, head): 8

## Head 22

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 33 | 3.101 | 0.203 | |
| `clean_legal_fold` | 200 | 3.296 | 0.209 | |
| `illegal_fold` | 183 | 3.258 | 0.228 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | '.' ×42 | '.' ×356 | '.' ×282 |
| 2 | '.\n\n' ×33 | '.\n\n' ×187 | ' to' ×181 |
| 3 | ' see' ×27 | ' option' ×177 | '.\n\n' ×169 |
| 4 | ' and' ×22 | '<s>' ×108 | ' fold' ×125 |
| 5 | ' check' ×19 | 'olding' ×101 | ' your' ×104 |
| 6 | ',' ×13 | ' F' ×100 | ',' ×90 |
| 7 | ' to' ×12 | ',' ×88 | '<s>' ×70 |
| 8 | ' best' ×9 | ' folding' ×76 | ' best' ×49 |
| 9 | ' Checking' ×8 | ' is' ×62 | ' option' ×45 |
| 10 | ' option' ×8 | '":' ×47 | ' It' ×40 |
| 11 | ' hand' ×8 | ' your' ×38 | 'olding' ×34 |
| 12 | ' bet' ×7 | ' to' ×31 | ' F' ×34 |
| 13 | ' calling' ×5 | ' chips' ×31 | ' you' ×32 |
| 14 | ' does' ×5 | ' best' ×30 | ' it' ×30 |
| 15 | ' brings' ×4 | 'O' ×30 | ' should' ×25 |
| 16 | ' again' ×4 | ' fold' ×20 | ' "' ×25 |
| 17 | ' bets' ×4 | ' raise' ×16 | ' so' ×19 |
| 18 | ' betting' ×3 | ' "' ×15 | ' is' ×16 |
| 19 | ' do' ×3 | ' I' ×15 | 'O' ×15 |
| 20 | ' your' ×3 | ' and' ×13 | ' folding' ×11 |
| 21 | ' call' ×3 | ' so' ×13 | ' and' ×11 |
| 22 | ' the' ×2 | ' it' ×12 | ' avoid' ×10 |
| 23 | ' safer' ×2 | ' size' ×6 | ' hand' ×8 |
| 24 | ' turn' ×2 | ' should' ×5 | ' called' ×6 |
| 25 | ' worth' ×2 | ' enough' ×5 | ' action' ×6 |
| 26 | ' before' ×1 | '),' ×4 | ' I' ×5 |
| 27 | ' risky' ×1 | ' of' ×3 | ' bet' ×4 |
| 28 | ' continue' ×1 | ' you' ×2 | ' enough' ×3 |
| 29 | ' folding' ×1 | ' amount' ×2 | "'s" ×3 |
| 30 | ' is' ×1 | ').' ×1 | ' money' ×2 |
| 31 | '<s>' ×1 | ' action' ×1 | ' want' ×2 |
| 32 | ' a' ×1 | 'd' ×1 | ').' ×2 |
| 33 | ' considering' ×1 | ' hand' ×1 | ' now' ×1 |
| 34 | ' avoid' ×1 | ' potential' ×1 | '":' ×1 |
| 35 | ' should' ×1 | ' fl' ×1 | '),' ×1 |
| 36 | ' afford' ×1 | ' money' ×1 | ' raise' ×1 |
| 37 | ' consider' ×1 | — | ' river' ×1 |
| 38 | ' play' ×1 | — | ' f' ×1 |
| 39 | ' reasonable' ×1 | — | — |
| 40 | — | — | — |

## Head 9

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 33 | 2.717 | 0.072 | |
| `clean_legal_fold` | 200 | 2.778 | 0.075 | |
| `illegal_fold` | 183 | 2.830 | 0.070 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | 'JSON' ×33 | 'JSON' ×200 | 'JSON' ×183 |
| 2 | '<s>' ×33 | 'action' ×200 | 'action' ×183 |
| 3 | '.\n\n' ×33 | '<s>' ×200 | '<s>' ×183 |
| 4 | 'action' ×33 | ':' ×200 | ' {"' ×183 |
| 5 | ' {"' ×33 | ' {"' ×200 | ':' ×183 |
| 6 | ':' ×33 | ' "' ×200 | '.\n\n' ×183 |
| 7 | ' "' ×33 | '.\n\n' ×200 | ' "' ×183 |
| 8 | '":' ×32 | '":' ×200 | '":' ×182 |
| 9 | '.' ×1 | — | ' Therefore' ×1 |
| 10 | — | — | — |
| 11 | — | — | — |
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

## Head 15

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 33 | 4.039 | 0.100 | |
| `clean_legal_fold` | 200 | 4.241 | 0.098 | |
| `illegal_fold` | 183 | 4.128 | 0.138 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | '.' ×52 | '.' ×405 | '.' ×320 |
| 2 | '<s>' ×33 | '<s>' ×200 | '<s>' ×183 |
| 3 | '.\n\n' ×33 | '.\n\n' ×200 | '.\n\n' ×183 |
| 4 | 'action' ×31 | 'action' ×197 | 'action' ×170 |
| 5 | ' "' ×25 | ' "' ×196 | ' "' ×169 |
| 6 | ' check' ×19 | 'olding' ×101 | ' {"' ×149 |
| 7 | ' and' ×18 | ' {"' ×90 | ' fold' ×125 |
| 8 | ' Checking' ×8 | 'JSON' ×80 | 'JSON' ×62 |
| 9 | ' turn' ×7 | ' folding' ×76 | 'olding' ×34 |
| 10 | ' brings' ×4 | ' fold' ×20 | ' weak' ×17 |
| 11 | ' again' ×4 | ' weak' ×15 | ' small' ×13 |
| 12 | ' bets' ×4 | ' strong' ×5 | ' folding' ×11 |
| 13 | ' do' ×3 | ' pre' ×4 | ' pair' ×7 |
| 14 | ' calling' ×3 | ' calling' ×2 | ' check' ×5 |
| 15 | ' call' ×3 | ',' ×2 | ' F' ×4 |
| 16 | ' is' ×2 | ' hand' ×1 | ',' ×4 |
| 17 | 'op' ×2 | ' betting' ×1 | ' calling' ×3 |
| 18 | ' better' ×2 | ' small' ×1 | ' raise' ×2 |
| 19 | ' {"' ×2 | ' raise' ×1 | ' hand' ×1 |
| 20 | ' board' ×1 | ' pair' ×1 | ' betting' ×1 |
| 21 | ' does' ×1 | ' F' ×1 | ' strong' ×1 |
| 22 | ' bet' ×1 | ' Checking' ×1 | — |
| 23 | ' folding' ×1 | — | — |
| 24 | ' or' ×1 | — | — |
| 25 | ' raise' ×1 | — | — |
| 26 | ' cards' ×1 | — | — |
| 27 | '2' ×1 | — | — |
| 28 | '6' ×1 | — | — |
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
