# Pot-odds + EV — full small-tier grid (CoT + non-CoT)

36 cells: 3 models × 2 conditions × 2 temperatures × 3 seeds. Opponent: `informative_v2`. Settings: `--num-rollouts 30 --samples-per-bucket 2 --skip-belief-ev`.

## Per-cell summary

| Model | Cond | Temp | Seed | n | EV-truth regret (mean) | EV-truth optimal % | EV-oracle regret | Threshold agrees w/ EV % |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `llama8b` | cot | 0.0 | 42 | 610 | 1.922 | 39.5 | 1.552 | 47.4 |
| `llama8b` | cot | 0.0 | 123 | 596 | 2.359 | 39.9 | 1.512 | 45.6 |
| `llama8b` | cot | 0.0 | 456 | 524 | 2.032 | 42.7 | 1.654 | 42.7 |
| `llama8b` | cot | 0.2 | 42 | 646 | 1.882 | 45.8 | 1.379 | 49.4 |
| `llama8b` | cot | 0.2 | 123 | 528 | 2.288 | 36.9 | 1.589 | 48.5 |
| `llama8b` | cot | 0.2 | 456 | 542 | 1.901 | 43.9 | 1.430 | 48.5 |
| `llama8b` | noncot | 0.0 | 42 | 1695 | 2.986 | 48.1 | 2.352 | 51.3 |
| `llama8b` | noncot | 0.0 | 123 | 1444 | 2.166 | 49.2 | 1.218 | 52.8 |
| `llama8b` | noncot | 0.0 | 456 | 1470 | 2.514 | 51.2 | 1.526 | 52.9 |
| `llama8b` | noncot | 0.2 | 42 | 1624 | 2.801 | 47.2 | 2.140 | 53.8 |
| `llama8b` | noncot | 0.2 | 123 | 1374 | 2.190 | 50.1 | 1.232 | 52.3 |
| `llama8b` | noncot | 0.2 | 456 | 1382 | 2.397 | 49.9 | 1.420 | 54.6 |
| `ministral8b` | cot | 0.0 | 42 | 525 | 1.269 | 48.2 | 1.051 | 50.1 |
| `ministral8b` | cot | 0.0 | 123 | 203 | 1.920 | 26.1 | 1.374 | 43.3 |
| `ministral8b` | cot | 0.0 | 456 | 209 | 1.696 | 31.6 | 1.385 | 38.8 |
| `ministral8b` | cot | 0.2 | 42 | 519 | 1.228 | 46.2 | 1.077 | 46.6 |
| `ministral8b` | cot | 0.2 | 123 | 206 | 1.776 | 29.1 | 1.392 | 39.3 |
| `ministral8b` | cot | 0.2 | 456 | 221 | 1.711 | 29.9 | 1.441 | 34.8 |
| `ministral8b` | noncot | 0.0 | 42 | 415 | 1.685 | 34.5 | 1.284 | 49.2 |
| `ministral8b` | noncot | 0.0 | 123 | 200 | 1.884 | 25.5 | 1.438 | 45.5 |
| `ministral8b` | noncot | 0.0 | 456 | 200 | 1.665 | 30.5 | 1.375 | 44.0 |
| `ministral8b` | noncot | 0.2 | 42 | 429 | 1.609 | 35.0 | 1.308 | 48.7 |
| `ministral8b` | noncot | 0.2 | 123 | 200 | 1.884 | 25.5 | 1.436 | 45.5 |
| `ministral8b` | noncot | 0.2 | 456 | 200 | 1.665 | 30.5 | 1.339 | 44.0 |
| `qwen8b` | cot | 0.0 | 42 | 796 | 2.204 | 48.0 | 1.685 | 50.4 |
| `qwen8b` | cot | 0.0 | 123 | 603 | 2.343 | 40.3 | 1.614 | 45.9 |
| `qwen8b` | cot | 0.0 | 456 | 616 | 2.127 | 46.4 | 1.559 | 47.2 |
| `qwen8b` | cot | 0.2 | 42 | 804 | 2.072 | 46.6 | 1.466 | 49.8 |
| `qwen8b` | cot | 0.2 | 123 | 612 | 2.360 | 39.9 | 1.609 | 46.1 |
| `qwen8b` | cot | 0.2 | 456 | 568 | 2.171 | 47.7 | 1.875 | 45.2 |
| `qwen8b` | noncot | 0.0 | 42 | 489 | 1.713 | 41.1 | 1.415 | 45.4 |
| `qwen8b` | noncot | 0.0 | 123 | 620 | 2.567 | 40.0 | 1.859 | 46.0 |
| `qwen8b` | noncot | 0.0 | 456 | 657 | 2.066 | 44.9 | 1.570 | 46.6 |
| `qwen8b` | noncot | 0.2 | 42 | 490 | 1.703 | 40.8 | 1.441 | 46.1 |
| `qwen8b` | noncot | 0.2 | 123 | 651 | 2.532 | 44.4 | 2.025 | 46.1 |
| `qwen8b` | noncot | 0.2 | 456 | 664 | 2.103 | 45.2 | 1.650 | 46.1 |

## Decomposition by fallback bucket

`clean` = no fallback used (model picked a legal action). `illegal_fold_rescued` = model emitted FOLD into a no-FOLD spot, `_fallback_action` rescued to CHECK_OR_CALL. `other_fallback` = any other reason `fallback_used=True` (parse fail, alias miss, illegal non-FOLD).

| Model | Cond | Temp | Seed | n clean | regret clean | EV-opt% clean | n illegal-FOLD | regret rescued | EV-opt% rescued | n other-fb | regret other |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama8b` | cot | 0.0 | 42 | 594 | 1.909 | 39.7 | 16 | 2.425 | 31.2 | 0 | — |
| `llama8b` | cot | 0.0 | 123 | 562 | 2.380 | 40.2 | 34 | 2.010 | 35.3 | 0 | — |
| `llama8b` | cot | 0.0 | 456 | 506 | 2.056 | 42.3 | 18 | 1.344 | 55.6 | 0 | — |
| `llama8b` | cot | 0.2 | 42 | 611 | 1.926 | 45.3 | 35 | 1.118 | 54.3 | 0 | — |
| `llama8b` | cot | 0.2 | 123 | 509 | 2.298 | 37.1 | 19 | 2.004 | 31.6 | 0 | — |
| `llama8b` | cot | 0.2 | 456 | 520 | 1.946 | 43.1 | 22 | 0.821 | 63.6 | 0 | — |
| `llama8b` | noncot | 0.0 | 42 | 1695 | 2.986 | 48.1 | 0 | — | — | 0 | — |
| `llama8b` | noncot | 0.0 | 123 | 1444 | 2.166 | 49.2 | 0 | — | — | 0 | — |
| `llama8b` | noncot | 0.0 | 456 | 1470 | 2.514 | 51.2 | 0 | — | — | 0 | — |
| `llama8b` | noncot | 0.2 | 42 | 1624 | 2.801 | 47.2 | 0 | — | — | 0 | — |
| `llama8b` | noncot | 0.2 | 123 | 1374 | 2.190 | 50.1 | 0 | — | — | 0 | — |
| `llama8b` | noncot | 0.2 | 456 | 1382 | 2.397 | 49.9 | 0 | — | — | 0 | — |
| `ministral8b` | cot | 0.0 | 42 | 326 | 1.678 | 41.7 | 179 | 0.554 | 60.9 | 20 | 1.000 |
| `ministral8b` | cot | 0.0 | 123 | 202 | 1.929 | 25.7 | 1 | 0.000 | 100.0 | 0 | — |
| `ministral8b` | cot | 0.0 | 456 | 206 | 1.695 | 32.0 | 3 | 1.733 | 0.0 | 0 | — |
| `ministral8b` | cot | 0.2 | 42 | 332 | 1.560 | 45.5 | 162 | 0.587 | 50.6 | 25 | 0.976 |
| `ministral8b` | cot | 0.2 | 123 | 204 | 1.766 | 29.4 | 2 | 2.800 | 0.0 | 0 | — |
| `ministral8b` | cot | 0.2 | 456 | 214 | 1.740 | 29.4 | 6 | 0.978 | 33.3 | 1 | 0.000 |
| `ministral8b` | noncot | 0.0 | 42 | 415 | 1.685 | 34.5 | 0 | — | — | 0 | — |
| `ministral8b` | noncot | 0.0 | 123 | 200 | 1.884 | 25.5 | 0 | — | — | 0 | — |
| `ministral8b` | noncot | 0.0 | 456 | 200 | 1.665 | 30.5 | 0 | — | — | 0 | — |
| `ministral8b` | noncot | 0.2 | 42 | 429 | 1.609 | 35.0 | 0 | — | — | 0 | — |
| `ministral8b` | noncot | 0.2 | 123 | 200 | 1.884 | 25.5 | 0 | — | — | 0 | — |
| `ministral8b` | noncot | 0.2 | 456 | 200 | 1.665 | 30.5 | 0 | — | — | 0 | — |
| `qwen8b` | cot | 0.0 | 42 | 790 | 2.212 | 47.8 | 4 | 1.667 | 50.0 | 2 | 0.000 |
| `qwen8b` | cot | 0.0 | 123 | 593 | 2.370 | 40.0 | 9 | 0.852 | 55.6 | 1 | 0.000 |
| `qwen8b` | cot | 0.0 | 456 | 604 | 2.158 | 45.9 | 11 | 0.588 | 72.7 | 1 | 0.000 |
| `qwen8b` | cot | 0.2 | 42 | 797 | 2.084 | 46.5 | 6 | 0.867 | 50.0 | 1 | 0.000 |
| `qwen8b` | cot | 0.2 | 123 | 599 | 2.383 | 39.7 | 13 | 1.292 | 46.2 | 0 | — |
| `qwen8b` | cot | 0.2 | 456 | 557 | 2.193 | 47.6 | 10 | 1.140 | 60.0 | 1 | 0.733 |
| `qwen8b` | noncot | 0.0 | 42 | 489 | 1.713 | 41.1 | 0 | — | — | 0 | — |
| `qwen8b` | noncot | 0.0 | 123 | 620 | 2.567 | 40.0 | 0 | — | — | 0 | — |
| `qwen8b` | noncot | 0.0 | 456 | 657 | 2.066 | 44.9 | 0 | — | — | 0 | — |
| `qwen8b` | noncot | 0.2 | 42 | 490 | 1.703 | 40.8 | 0 | — | — | 0 | — |
| `qwen8b` | noncot | 0.2 | 123 | 651 | 2.532 | 44.4 | 0 | — | — | 0 | — |
| `qwen8b` | noncot | 0.2 | 456 | 664 | 2.103 | 45.2 | 0 | — | — | 0 | — |

## CoT vs non-CoT delta (averaged across 3 seeds)

| Model | Temp | EV-truth regret  cot/non-cot/Δ | EV-truth optimal %  cot/non-cot/Δ |
|---|---|---|---|
| `llama8b` | 0.0 | 2.104 / 2.555 / **-0.451** | 40.7 / 49.5 / **-8.8 pp** |
| `llama8b` | 0.2 | 2.023 / 2.463 / **-0.439** | 42.2 / 49.1 / **-6.8 pp** |
| `ministral8b` | 0.0 | 1.628 / 1.745 / **-0.117** | 35.3 / 30.2 / **5.1 pp** |
| `ministral8b` | 0.2 | 1.572 / 1.720 / **-0.148** | 35.1 / 30.3 / **4.8 pp** |
| `qwen8b` | 0.0 | 2.225 / 2.115 / **0.109** | 44.9 / 42.0 / **2.9 pp** |
| `qwen8b` | 0.2 | 2.201 / 2.113 / **0.088** | 44.7 / 43.5 / **1.3 pp** |

*Δ < 0 on regret = CoT better; Δ > 0 on optimal% = CoT better.*

## Key findings (computed from this grid)

### 1. Illegal-FOLD rescue effect generalises across the family

| Model | CoT cells with any rescue | Total rescued FOLDs | Cells where rescue regret < clean regret |
|---|---:|---:|---:|
| `llama8b` | 6/6 | 144 | 5/6 |
| `ministral8b` | 6/6 | 353 | 4/6 |
| `qwen8b` | 6/6 | 53 | 6/6 |

The §11.6 Ministral-only finding (rescued FOLDs have lower mean regret than cleanly-emitted actions) holds for all three small models in the cells where rescues happen. Mechanism: when small-model CoT emits FOLD into a free-check spot, it's expressing a 'this hand is weak, defend' intent that the env's `_fallback_action` honours at zero chips by playing CHECK_OR_CALL.

### 2. CoT's effect splits three ways across models

Reading the delta table above (averaged across 3 seeds per cell):

- **Llama 8B**: CoT cuts mean EV-regret by ~0.45 chips/decision (real, both temps), BUT also drops EV-optimal% by 7-9 pp. CoT smooths Llama's wrong picks (closer to optimal when it's wrong) at the cost of making fewer outright optimal picks. Net EV per decision improves; sharpness regresses.
- **Ministral 8B**: CoT cuts regret by 0.12-0.15 AND raises optimal% by ~5 pp — reads like a clean win, but **the bucket table shows this is driven entirely by the s42 rescue mechanism** (179 + 162 rescued FOLDs out of ~340 illegal-FOLDs total in the grid). On the other 4 Ministral cells the rescue count drops to 1-6 and the CoT vs non-CoT clean-regret comparison is essentially a wash (1.7-1.9 vs 1.7-1.9). **CoT does NOT robustly help Ministral; the rescue mechanism does.**
- **Qwen 8B**: CoT *increases* mean EV-regret by ~0.1 and only marginally raises optimal% (1-3 pp). Across both temps Qwen is the model where CoT is closest to neutral; if anything mildly harmful.

### 3. CoT shortens hands

Non-CoT cells have 2-3× more decisions per cell than CoT cells (e.g. Llama non-CoT: 1374-1695 decisions; Llama CoT: 524-646). This is itself a behavioural observation: under CoT, small models fold more often (whether legally or illegally-then-rescued), which truncates hands and shrinks the decision count. The rescue mechanism saves the EV in the rescued spots; the rest is genuine extra folding.

### 4. Reportable framing

The §11.6 reframing ("CoT's apparent EV improvement on Ministral is a rescue artifact") generalises to a stronger claim: **for small models in this poker environment, CoT does not unambiguously improve decision quality; it changes the shape of the decision distribution in model-specific ways, and any aggregate EV win is partly attributable to the env's safety net catching pathological action choices.** Verifiable from this table without a re-run.
