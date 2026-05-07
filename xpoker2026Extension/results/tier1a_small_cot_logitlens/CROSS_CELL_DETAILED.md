# Cross-cell detailed logit-lens analysis (per emitted action)

Generalizes the §13 Ministral-s42-only finding to ALL 18 cells. For each
cell × emitted action bucket, reports the action-group crystallization
layer at the action-verb position. The per-bucket per-layer mix tables
for each cell follow.

## Per-model aggregate (weighted-mean crystallization layer)

Weighted by per-bucket n across that model's 6 cells. Only buckets
with non-zero records contribute. Lower = the model 'decided' earlier;
higher = late-layer revision.

| Model | clean CHECK_OR_CALL | clean LEGAL FOLD | clean BET_OR_RAISE | illegal FOLD |
|---|---:|---:|---:|---:|
| `llama8b` | 25.4 (n=1080) | 20.9 (n=579) | 24.1 (n=198) | 20.7 (n=143) |
| `ministral8b` | 29.6 (n=83) | 23.8 (n=596) | 27.0 (n=15) | 22.7 (n=350) |
| `qwen8b` | 30.6 (n=1149) | 29.8 (n=522) | 29.6 (n=626) | 30.4 (n=42) |

**Interpretation guide.** If illegal-FOLD crystallizes EARLIER than
clean CHECK_OR_CALL by 2+ layers, that's the §13 'baseline FOLD pull
with late-layer revision in CHECK decisions only' pattern, generalized
across this model's 6 cells.

## Per-cell summary (action-group crystallization layer)

| Cell | n CHECK_OR_CALL | crys CHECK_OR_CALL | n LEGAL_FOLD | crys LEGAL_FOLD | n illegal_FOLD | crys illegal_FOLD | Δ (illegal − clean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llama8b_t02_s123` | 167 | 25.5 | 98 | 20.8 | 28 | 21.0 | -4.4 |
| `llama8b_t02_s42` | 175 | 25.9 | 95 | 20.9 | 26 | 20.4 | -5.5 |
| `llama8b_t02_s456` | 181 | 24.9 | 97 | 20.9 | 21 | 20.6 | -4.3 |
| `llama8b_t0_s123` | 198 | 24.8 | 94 | 20.8 | 34 | 20.9 | -4.0 |
| `llama8b_t0_s42` | 173 | 26.6 | 95 | 20.9 | 16 | 20.4 | -6.1 |
| `llama8b_t0_s456` | 186 | 24.7 | 100 | 20.9 | 18 | 20.9 | -3.8 |
| `ministral8b_t02_s123` | 4 | 31.8 | 100 | 23.9 | 2 | 23.0 | -8.8 |
| `ministral8b_t02_s42` | 42 | 29.2 | 98 | 23.6 | 161 | 22.7 | -6.5 |
| `ministral8b_t02_s456` | 4 | 34.0 | 100 | 23.9 | 4 | 23.0 | -11.0 |
| `ministral8b_t0_s123` | 1 | 29.0 | 100 | 23.9 | 1 | 22.0 | -7.0 |
| `ministral8b_t0_s42` | 29 | 28.8 | 98 | 23.7 | 179 | 22.7 | -6.2 |
| `ministral8b_t0_s456` | 3 | 34.0 | 100 | 23.9 | 3 | 22.3 | -11.7 |
| `qwen8b_t02_s123` | 210 | 30.3 | 83 | 29.6 | 8 | 30.5 | +0.2 |
| `qwen8b_t02_s42` | 229 | 30.8 | 84 | 30.1 | 4 | 29.8 | -1.1 |
| `qwen8b_t02_s456` | 150 | 30.5 | 94 | 29.6 | 6 | 30.3 | -0.2 |
| `qwen8b_t0_s123` | 180 | 30.5 | 84 | 29.6 | 9 | 30.3 | -0.2 |
| `qwen8b_t0_s42` | 215 | 31.0 | 87 | 29.9 | 4 | 30.8 | -0.3 |
| `qwen8b_t0_s456` | 165 | 30.4 | 90 | 29.6 | 11 | 30.4 | +0.1 |

## Per-cell late-layer trajectory (final 12 layers)

Compact view: only the final 12 layers, only the buckets with n≥3.
Format per cell: per-layer mapped action group at the verb position,
showing fractions ≥0.05.

### `llama8b_t02_s123`

_32 layers; showing 12 latest (L=20–31)_

**clean CHECK_OR_CALL** (n=167)
  - crystallization layer: mean=25.49, median=22, range=[22, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L20: OTHER0.99
    - L21: FOLD0.07 OTHER0.93
    - L22: FOLD0.09 CHECK0.88
    - L23: FOLD0.07 CHECK0.93
    - L24: FOLD0.13 CHECK0.87
    - L25: FOLD0.08 CHECK0.92
    - L26: FOLD0.07 CHECK0.54 CALL0.39
    - L27: FOLD0.08 CHECK0.71 CALL0.21
    - L28: FOLD0.08 CHECK0.59 CALL0.34
    - L29: FOLD0.08 CHECK0.76 CALL0.14
    - L30: CHECK0.96
    - L31: CHECK1.00

**clean LEGAL FOLD** (n=98)
  - crystallization layer: mean=20.82, median=21.0, range=[18, 21]
  - late-layer mix (action-group fractions at verb pos):
    - L20: FOLD0.06 OTHER0.94
    - L21: FOLD1.00
    - L22: FOLD1.00
    - L23: FOLD1.00
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00

**illegal FOLD (rescued)** (n=28)
  - crystallization layer: mean=21.04, median=21.0, range=[18, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L20: FOLD0.11 OTHER0.89
    - L21: FOLD1.00
    - L22: FOLD1.00
    - L23: FOLD1.00
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD0.96


### `llama8b_t02_s42`

_32 layers; showing 12 latest (L=20–31)_

**clean CHECK_OR_CALL** (n=175)
  - crystallization layer: mean=25.89, median=27, range=[22, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L20: OTHER0.98
    - L21: FOLD0.06 OTHER0.94
    - L22: FOLD0.09 CHECK0.90
    - L23: FOLD0.05 CHECK0.93
    - L24: FOLD0.13 CHECK0.82
    - L25: FOLD0.07 CHECK0.90
    - L26: FOLD0.07 CHECK0.51 CALL0.42
    - L27: FOLD0.09 CHECK0.59 CALL0.31
    - L28: FOLD0.08 CHECK0.54 CALL0.38
    - L29: FOLD0.08 CHECK0.65 CALL0.26
    - L30: CHECK0.98
    - L31: CHECK1.00

**clean LEGAL FOLD** (n=95)
  - crystallization layer: mean=20.88, median=21, range=[18, 21]
  - late-layer mix (action-group fractions at verb pos):
    - L20: OTHER0.96
    - L21: FOLD1.00
    - L22: FOLD1.00
    - L23: FOLD1.00
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00

**illegal FOLD (rescued)** (n=26)
  - crystallization layer: mean=20.38, median=21.0, range=[18, 21]
  - late-layer mix (action-group fractions at verb pos):
    - L20: FOLD0.35 OTHER0.65
    - L21: FOLD1.00
    - L22: FOLD1.00
    - L23: FOLD1.00
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00


### `llama8b_t02_s456`

_32 layers; showing 12 latest (L=20–31)_

**clean CHECK_OR_CALL** (n=181)
  - crystallization layer: mean=24.94, median=22, range=[22, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L20: OTHER1.00
    - L21: OTHER0.96
    - L22: FOLD0.09 CHECK0.87
    - L23: CHECK0.96
    - L24: FOLD0.14 CHECK0.86
    - L25: FOLD0.07 CHECK0.93
    - L26: FOLD0.07 CHECK0.62 CALL0.31
    - L27: FOLD0.09 CHECK0.75 CALL0.15
    - L28: FOLD0.08 CHECK0.66 CALL0.26
    - L29: FOLD0.08 CHECK0.81 CALL0.10
    - L30: CHECK0.97
    - L31: CHECK0.99

**clean LEGAL FOLD** (n=97)
  - crystallization layer: mean=20.95, median=21, range=[18, 24]
  - late-layer mix (action-group fractions at verb pos):
    - L20: OTHER0.96
    - L21: FOLD0.99
    - L22: FOLD1.00
    - L23: FOLD0.98
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00

**illegal FOLD (rescued)** (n=21)
  - crystallization layer: mean=20.62, median=21, range=[18, 21]
  - late-layer mix (action-group fractions at verb pos):
    - L20: FOLD0.14 OTHER0.86
    - L21: FOLD1.00
    - L22: FOLD1.00
    - L23: FOLD1.00
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00


### `llama8b_t0_s123`

_32 layers; showing 12 latest (L=20–31)_

**clean CHECK_OR_CALL** (n=198)
  - crystallization layer: mean=24.84, median=22.0, range=[22, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L20: OTHER0.99
    - L21: OTHER0.96
    - L22: FOLD0.11 CHECK0.86
    - L23: CHECK0.95
    - L24: FOLD0.13 CHECK0.87
    - L25: FOLD0.06 CHECK0.94
    - L26: FOLD0.09 CHECK0.62 CALL0.29
    - L27: FOLD0.11 CHECK0.78 CALL0.11
    - L28: FOLD0.09 CHECK0.68 CALL0.23
    - L29: FOLD0.09 CHECK0.88
    - L30: CHECK0.97
    - L31: CHECK1.00

**clean LEGAL FOLD** (n=94)
  - crystallization layer: mean=20.84, median=21.0, range=[18, 21]
  - late-layer mix (action-group fractions at verb pos):
    - L20: FOLD0.05 OTHER0.95
    - L21: FOLD1.00
    - L22: FOLD1.00
    - L23: FOLD1.00
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00

**illegal FOLD (rescued)** (n=34)
  - crystallization layer: mean=20.88, median=21.0, range=[18, 21]
  - late-layer mix (action-group fractions at verb pos):
    - L20: FOLD0.06 OTHER0.94
    - L21: FOLD1.00
    - L22: FOLD1.00
    - L23: FOLD1.00
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00


### `llama8b_t0_s42`

_32 layers; showing 12 latest (L=20–31)_

**clean CHECK_OR_CALL** (n=173)
  - crystallization layer: mean=26.55, median=29, range=[22, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L20: OTHER0.99
    - L21: OTHER0.96
    - L22: FOLD0.06 CHECK0.90
    - L23: CHECK0.95
    - L24: FOLD0.11 CHECK0.87
    - L25: CHECK0.93
    - L26: CHECK0.42 CALL0.54
    - L27: CHECK0.54 CALL0.42
    - L28: CHECK0.48 CALL0.48
    - L29: CHECK0.57 CALL0.36
    - L30: CHECK0.98
    - L31: CHECK1.00

**clean LEGAL FOLD** (n=95)
  - crystallization layer: mean=20.91, median=21, range=[18, 21]
  - late-layer mix (action-group fractions at verb pos):
    - L20: OTHER0.97
    - L21: FOLD1.00
    - L22: FOLD1.00
    - L23: FOLD1.00
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00

**illegal FOLD (rescued)** (n=16)
  - crystallization layer: mean=20.44, median=21.0, range=[18, 21]
  - late-layer mix (action-group fractions at verb pos):
    - L20: FOLD0.19 OTHER0.81
    - L21: FOLD1.00
    - L22: FOLD1.00
    - L23: FOLD1.00
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00


### `llama8b_t0_s456`

_32 layers; showing 12 latest (L=20–31)_

**clean CHECK_OR_CALL** (n=186)
  - crystallization layer: mean=24.68, median=22.0, range=[22, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L20: OTHER1.00
    - L21: OTHER0.98
    - L22: FOLD0.09 CHECK0.90
    - L23: CHECK0.96
    - L24: FOLD0.14 CHECK0.86
    - L25: CHECK0.96
    - L26: FOLD0.08 CHECK0.63 CALL0.29
    - L27: FOLD0.08 CHECK0.80 CALL0.12
    - L28: FOLD0.08 CHECK0.70 CALL0.22
    - L29: FOLD0.07 CHECK0.89
    - L30: CHECK0.98
    - L31: CHECK1.00

**clean LEGAL FOLD** (n=100)
  - crystallization layer: mean=20.86, median=21.0, range=[18, 22]
  - late-layer mix (action-group fractions at verb pos):
    - L20: FOLD0.05 OTHER0.95
    - L21: FOLD0.99
    - L22: FOLD1.00
    - L23: FOLD1.00
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00

**illegal FOLD (rescued)** (n=18)
  - crystallization layer: mean=20.89, median=21.0, range=[19, 21]
  - late-layer mix (action-group fractions at verb pos):
    - L20: FOLD0.06 OTHER0.94
    - L21: FOLD1.00
    - L22: FOLD1.00
    - L23: FOLD1.00
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00


### `ministral8b_t02_s123`

_36 layers; showing 12 latest (L=24–35)_

**clean CHECK_OR_CALL** (n=4)
  - crystallization layer: mean=31.75, median=32.5, range=[28, 34]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD0.75 CHECK0.25
    - L29: CHECK0.25 CALL0.75
    - L30: CHECK0.25 CALL0.75
    - L31: CHECK0.75 CALL0.25
    - L32: CHECK0.75 CALL0.25
    - L33: CHECK0.50 CALL0.50
    - L34: CHECK1.00
    - L35: CHECK1.00

**clean LEGAL FOLD** (n=100)
  - crystallization layer: mean=23.91, median=24.0, range=[22, 24]
  - late-layer mix (action-group fractions at verb pos):
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00


### `ministral8b_t02_s42`

_36 layers; showing 12 latest (L=24–35)_

**clean CHECK_OR_CALL** (n=42)
  - crystallization layer: mean=29.2, median=29, range=[27, 35]
  - late-layer mix (action-group fractions at verb pos):
    - L24: FOLD0.15 OTHER0.85
    - L25: FOLD0.85 OTHER0.12
    - L26: FOLD0.98
    - L27: FOLD0.78 CHECK0.17
    - L28: FOLD0.61 CHECK0.32
    - L29: FOLD0.07 CHECK0.90
    - L30: FOLD0.10 CHECK0.88
    - L31: CHECK0.93
    - L32: FOLD0.07 CHECK0.90
    - L33: CHECK0.95
    - L34: CHECK0.95
    - L35: CHECK0.98

**clean LEGAL FOLD** (n=98)
  - crystallization layer: mean=23.62, median=24.0, range=[22, 24]
  - late-layer mix (action-group fractions at verb pos):
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00

**illegal FOLD (rescued)** (n=161)
  - crystallization layer: mean=22.73, median=23, range=[22, 23]
  - late-layer mix (action-group fractions at verb pos):
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00


### `ministral8b_t02_s456`

_36 layers; showing 12 latest (L=24–35)_

**clean CHECK_OR_CALL** (n=4)
  - crystallization layer: mean=34.0, median=34.0, range=[34, 34]
  - late-layer mix (action-group fractions at verb pos):
    - L24: FOLD0.25 OTHER0.75
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: CALL1.00
    - L30: CALL1.00
    - L31: CALL1.00
    - L32: CALL1.00
    - L33: CALL1.00
    - L34: CHECK1.00
    - L35: CHECK1.00

**clean LEGAL FOLD** (n=100)
  - crystallization layer: mean=23.88, median=24.0, range=[22, 24]
  - late-layer mix (action-group fractions at verb pos):
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00

**illegal FOLD (rescued)** (n=4)
  - crystallization layer: mean=23.0, median=23.0, range=[22, 24]
  - late-layer mix (action-group fractions at verb pos):
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00


### `ministral8b_t0_s123`

_36 layers; showing 12 latest (L=24–35)_

**clean LEGAL FOLD** (n=100)
  - crystallization layer: mean=23.92, median=24.0, range=[23, 24]
  - late-layer mix (action-group fractions at verb pos):
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00


### `ministral8b_t0_s42`

_36 layers; showing 12 latest (L=24–35)_

**clean CHECK_OR_CALL** (n=29)
  - crystallization layer: mean=28.83, median=29, range=[27, 34]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: FOLD0.86 OTHER0.14
    - L26: FOLD0.97
    - L27: FOLD0.76 CHECK0.21
    - L28: FOLD0.48 CHECK0.48
    - L29: CHECK0.90 CALL0.10
    - L30: CHECK0.90 CALL0.10
    - L31: CHECK0.90 CALL0.10
    - L32: CHECK0.90 CALL0.10
    - L33: CHECK0.90 CALL0.10
    - L34: CHECK1.00
    - L35: CHECK1.00

**clean LEGAL FOLD** (n=98)
  - crystallization layer: mean=23.72, median=24.0, range=[22, 24]
  - late-layer mix (action-group fractions at verb pos):
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00

**illegal FOLD (rescued)** (n=179)
  - crystallization layer: mean=22.67, median=23, range=[22, 23]
  - late-layer mix (action-group fractions at verb pos):
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00


### `ministral8b_t0_s456`

_36 layers; showing 12 latest (L=24–35)_

**clean CHECK_OR_CALL** (n=3)
  - crystallization layer: mean=34.0, median=34, range=[34, 34]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: CALL1.00
    - L30: CALL1.00
    - L31: CHECK0.33 CALL0.67
    - L32: CHECK0.33 CALL0.67
    - L33: CALL1.00
    - L34: CHECK1.00
    - L35: CHECK1.00

**clean LEGAL FOLD** (n=100)
  - crystallization layer: mean=23.85, median=24.0, range=[23, 24]
  - late-layer mix (action-group fractions at verb pos):
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00

**illegal FOLD (rescued)** (n=3)
  - crystallization layer: mean=22.33, median=22, range=[22, 23]
  - late-layer mix (action-group fractions at verb pos):
    - L24: FOLD1.00
    - L25: FOLD1.00
    - L26: FOLD1.00
    - L27: FOLD1.00
    - L28: FOLD1.00
    - L29: FOLD1.00
    - L30: FOLD1.00
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00


### `qwen8b_t02_s123`

_36 layers; showing 12 latest (L=24–35)_

**clean CHECK_OR_CALL** (n=210)
  - crystallization layer: mean=30.34, median=30.0, range=[29, 35]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: CHECK0.28 OTHER0.68
    - L30: CHECK0.78 OTHER0.17
    - L31: CHECK0.77 OTHER0.23
    - L32: CHECK1.00
    - L33: CHECK1.00
    - L34: CHECK1.00
    - L35: CHECK1.00

**clean LEGAL FOLD** (n=83)
  - crystallization layer: mean=29.65, median=30, range=[29, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: FOLD0.48 OTHER0.52
    - L30: FOLD0.87 OTHER0.13
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00

**illegal FOLD (rescued)** (n=8)
  - crystallization layer: mean=30.5, median=30.0, range=[30, 34]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: OTHER1.00
    - L30: FOLD0.88 OTHER0.12
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD0.88 CHECK0.12
    - L34: FOLD1.00
    - L35: FOLD1.00


### `qwen8b_t02_s42`

_36 layers; showing 12 latest (L=24–35)_

**clean CHECK_OR_CALL** (n=229)
  - crystallization layer: mean=30.84, median=31, range=[29, 35]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: CHECK0.11 OTHER0.84
    - L30: CHECK0.59 OTHER0.35
    - L31: CHECK0.69 OTHER0.31
    - L32: CHECK1.00
    - L33: CHECK1.00
    - L34: CHECK0.98
    - L35: CHECK1.00

**clean LEGAL FOLD** (n=84)
  - crystallization layer: mean=30.07, median=30.0, range=[29, 34]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: FOLD0.20 OTHER0.80
    - L30: FOLD0.86 OTHER0.14
    - L31: FOLD1.00
    - L32: FOLD0.98
    - L33: FOLD0.98
    - L34: FOLD1.00
    - L35: FOLD1.00

**illegal FOLD (rescued)** (n=4)
  - crystallization layer: mean=29.75, median=30.0, range=[29, 30]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: FOLD0.25 OTHER0.75
    - L30: FOLD1.00
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00


### `qwen8b_t02_s456`

_36 layers; showing 12 latest (L=24–35)_

**clean CHECK_OR_CALL** (n=150)
  - crystallization layer: mean=30.51, median=30.0, range=[29, 35]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: CHECK0.23 BET0.07 OTHER0.69
    - L30: CHECK0.73 BET0.06 OTHER0.19
    - L31: CHECK0.78 OTHER0.21
    - L32: CHECK1.00
    - L33: CHECK1.00
    - L34: CHECK0.96
    - L35: CHECK1.00

**clean LEGAL FOLD** (n=94)
  - crystallization layer: mean=29.64, median=30.0, range=[29, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: FOLD0.49 OTHER0.51
    - L30: FOLD0.88 OTHER0.12
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00

**illegal FOLD (rescued)** (n=6)
  - crystallization layer: mean=30.33, median=30.0, range=[30, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: OTHER1.00
    - L30: FOLD0.67 OTHER0.33
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00


### `qwen8b_t0_s123`

_36 layers; showing 12 latest (L=24–35)_

**clean CHECK_OR_CALL** (n=180)
  - crystallization layer: mean=30.48, median=30.0, range=[29, 35]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: CHECK0.25 OTHER0.71
    - L30: CHECK0.73 OTHER0.22
    - L31: CHECK0.73 OTHER0.26
    - L32: CHECK1.00
    - L33: CHECK1.00
    - L34: CHECK0.98
    - L35: CHECK1.00

**clean LEGAL FOLD** (n=84)
  - crystallization layer: mean=29.65, median=30.0, range=[29, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: FOLD0.39 OTHER0.61
    - L30: FOLD0.95
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00

**illegal FOLD (rescued)** (n=9)
  - crystallization layer: mean=30.33, median=30, range=[30, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: OTHER1.00
    - L30: FOLD0.67 OTHER0.33
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00


### `qwen8b_t0_s42`

_36 layers; showing 12 latest (L=24–35)_

**clean CHECK_OR_CALL** (n=215)
  - crystallization layer: mean=31.04, median=31, range=[29, 35]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: CHECK0.09 BET0.06 OTHER0.85
    - L30: CHECK0.53 BET0.06 OTHER0.40
    - L31: CHECK0.69 OTHER0.31
    - L32: CHECK1.00
    - L33: CHECK1.00
    - L34: CHECK0.94 BET0.05
    - L35: CHECK1.00

**clean LEGAL FOLD** (n=87)
  - crystallization layer: mean=29.93, median=30, range=[29, 33]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: FOLD0.23 OTHER0.77
    - L30: FOLD0.90 OTHER0.10
    - L31: FOLD1.00
    - L32: FOLD0.98
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00

**illegal FOLD (rescued)** (n=4)
  - crystallization layer: mean=30.75, median=31.0, range=[30, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: OTHER1.00
    - L30: FOLD0.25 OTHER0.75
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00


### `qwen8b_t0_s456`

_36 layers; showing 12 latest (L=24–35)_

**clean CHECK_OR_CALL** (n=165)
  - crystallization layer: mean=30.39, median=30, range=[29, 35]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: CHECK0.25 OTHER0.67
    - L30: CHECK0.77 OTHER0.15
    - L31: CHECK0.80 OTHER0.19
    - L32: CHECK1.00
    - L33: CHECK1.00
    - L34: CHECK0.96
    - L35: CHECK1.00

**clean LEGAL FOLD** (n=90)
  - crystallization layer: mean=29.62, median=30.0, range=[29, 31]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: FOLD0.47 OTHER0.53
    - L30: FOLD0.91 OTHER0.09
    - L31: FOLD1.00
    - L32: FOLD1.00
    - L33: FOLD1.00
    - L34: FOLD1.00
    - L35: FOLD1.00

**illegal FOLD (rescued)** (n=11)
  - crystallization layer: mean=30.45, median=30, range=[30, 34]
  - late-layer mix (action-group fractions at verb pos):
    - L24: OTHER1.00
    - L25: OTHER1.00
    - L26: OTHER1.00
    - L27: OTHER1.00
    - L28: OTHER1.00
    - L29: OTHER1.00
    - L30: FOLD0.82 OTHER0.18
    - L31: FOLD1.00
    - L32: FOLD0.91 CHECK0.09
    - L33: FOLD0.91 CHECK0.09
    - L34: FOLD1.00
    - L35: FOLD1.00

