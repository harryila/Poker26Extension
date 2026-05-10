# Attention-pattern analysis at dominant heads

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Layer: **16**
- Heads analyzed: [22, 9, 15]
- Buckets: ['clean_check_or_call', 'clean_legal_fold', 'illegal_fold']
- Per-bucket sample size cap: 50
- Top-K positions per (decision, head): 8

## Head 22

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 33 | 3.101 | 0.203 | |
| `clean_legal_fold` | 50 | 3.141 | 0.189 | |
| `illegal_fold` | 50 | 3.245 | 0.240 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | '.' ×42 | '.' ×63 | '.' ×77 |
| 2 | '.\n\n' ×33 | '.\n\n' ×50 | ' to' ×46 |
| 3 | ' see' ×27 | '<s>' ×39 | '.\n\n' ×45 |
| 4 | ' and' ×22 | ' option' ×38 | ' fold' ×34 |
| 5 | ' check' ×19 | ',' ×26 | ' your' ×27 |
| 6 | ',' ×13 | ' is' ×26 | ',' ×24 |
| 7 | ' to' ×12 | ' folding' ×25 | '<s>' ×16 |
| 8 | ' best' ×9 | '":' ×23 | ' option' ×13 |
| 9 | ' Checking' ×8 | ' your' ×18 | ' best' ×13 |
| 10 | ' option' ×8 | ' best' ×18 | ' you' ×12 |
| 11 | ' hand' ×8 | ' to' ×13 | ' "' ×11 |
| 12 | ' bet' ×7 | 'olding' ×13 | 'olding' ×11 |
| 13 | ' calling' ×5 | ' F' ×13 | ' F' ×11 |
| 14 | ' does' ×5 | ' fold' ×11 | ' it' ×9 |
| 15 | ' brings' ×4 | ' it' ×8 | ' should' ×9 |
| 16 | ' again' ×4 | ' "' ×7 | ' It' ×8 |
| 17 | ' bets' ×4 | ' should' ×3 | ' is' ×6 |
| 18 | ' betting' ×3 | ' and' ×3 | 'O' ×4 |
| 19 | ' do' ×3 | ' of' ×1 | ' called' ×3 |
| 20 | ' your' ×3 | ' you' ×1 | ' and' ×3 |
| 21 | ' call' ×3 | ' so' ×1 | ' action' ×3 |
| 22 | ' the' ×2 | — | ' avoid' ×3 |
| 23 | ' safer' ×2 | — | ' folding' ×2 |
| 24 | ' turn' ×2 | — | ' enough' ×2 |
| 25 | ' worth' ×2 | — | ' so' ×2 |
| 26 | ' before' ×1 | — | ' hand' ×2 |
| 27 | ' risky' ×1 | — | "'s" ×1 |
| 28 | ' continue' ×1 | — | ' money' ×1 |
| 29 | ' folding' ×1 | — | ' bet' ×1 |
| 30 | ' is' ×1 | — | ' I' ×1 |
| 31 | '<s>' ×1 | — | — |
| 32 | ' a' ×1 | — | — |
| 33 | ' considering' ×1 | — | — |
| 34 | ' avoid' ×1 | — | — |
| 35 | ' should' ×1 | — | — |
| 36 | ' afford' ×1 | — | — |
| 37 | ' consider' ×1 | — | — |
| 38 | ' play' ×1 | — | — |
| 39 | ' reasonable' ×1 | — | — |
| 40 | — | — | — |

## Head 9

### Mean attention entropy at the last position (nats)

| Bucket | n | mean entropy | std | (low = focused, high = diffuse) |
|---|---:|---:|---:|---|
| `clean_check_or_call` | 33 | 2.717 | 0.072 | |
| `clean_legal_fold` | 50 | 2.727 | 0.049 | |
| `illegal_fold` | 50 | 2.822 | 0.072 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | 'JSON' ×33 | 'JSON' ×50 | 'JSON' ×50 |
| 2 | '<s>' ×33 | 'action' ×50 | 'action' ×50 |
| 3 | '.\n\n' ×33 | '<s>' ×50 | '<s>' ×50 |
| 4 | 'action' ×33 | ':' ×50 | ' {"' ×50 |
| 5 | ' {"' ×33 | ' {"' ×50 | ':' ×50 |
| 6 | ':' ×33 | ' "' ×50 | '.\n\n' ×50 |
| 7 | ' "' ×33 | '.\n\n' ×50 | ' "' ×50 |
| 8 | '":' ×32 | '":' ×50 | '":' ×50 |
| 9 | '.' ×1 | — | — |
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
| `clean_legal_fold` | 50 | 4.201 | 0.101 | |
| `illegal_fold` | 50 | 4.112 | 0.139 | |

### Top-40-most-frequently-attended token strings

(Aggregated across decisions: count = number of times the token string appeared in the top-8 attended positions of any decision, summed.)

| Rank | `clean_check_or_call` | `clean_legal_fold` | `illegal_fold` |
|---:|---|---|---|
| 1 | '.' ×52 | '.' ×85 | '.' ×87 |
| 2 | '<s>' ×33 | '<s>' ×50 | '<s>' ×50 |
| 3 | '.\n\n' ×33 | '.\n\n' ×50 | '.\n\n' ×50 |
| 4 | 'action' ×31 | 'action' ×49 | 'action' ×47 |
| 5 | ' "' ×25 | ' "' ×49 | ' "' ×47 |
| 6 | ' check' ×19 | 'JSON' ×38 | ' {"' ×39 |
| 7 | ' and' ×18 | ' {"' ×26 | ' fold' ×34 |
| 8 | ' Checking' ×8 | ' folding' ×25 | 'JSON' ×18 |
| 9 | ' turn' ×7 | 'olding' ×13 | 'olding' ×11 |
| 10 | ' brings' ×4 | ' fold' ×11 | ' weak' ×4 |
| 11 | ' again' ×4 | ' calling' ×1 | ' small' ×4 |
| 12 | ' bets' ×4 | ' strong' ×1 | ' folding' ×2 |
| 13 | ' do' ×3 | ' hand' ×1 | ' pair' ×2 |
| 14 | ' calling' ×3 | ',' ×1 | ',' ×2 |
| 15 | ' call' ×3 | — | ' F' ×1 |
| 16 | ' is' ×2 | — | ' calling' ×1 |
| 17 | 'op' ×2 | — | ' check' ×1 |
| 18 | ' better' ×2 | — | — |
| 19 | ' {"' ×2 | — | — |
| 20 | ' board' ×1 | — | — |
| 21 | ' does' ×1 | — | — |
| 22 | ' bet' ×1 | — | — |
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
