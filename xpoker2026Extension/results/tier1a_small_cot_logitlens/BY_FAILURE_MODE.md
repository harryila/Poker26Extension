# Logit-lens × failure-mode breakdown

Per-bucket per-layer mapped action group at the FINAL generated
position (i.e. the token where the model commits to an action in
the JSON payload). Fractions sum across decisions in the bucket.

Read like this: the FOLD column at layer L tells you what fraction
of decisions in that bucket had top-1 token in the FOLD family at
layer L's projection. If FOLD is 1.00 from layer 0, the model is
FOLD-committed top-to-bottom. If FOLD only crosses 0.5 in the late
layers while CHECK/CALL dominates early, that's the verbalization-
failure signature.

---

## llama8b_t02_s123_informative_v2_logitlens

**Stats:** enriched_decisions_seen=538, sidecar_records_loaded=317, joined=317, unmatched=221

### llama8b_t02_s123_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **289**
- N joined to logit-lens sidecar: 289
- Num layers: 32
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 9.39 |
| 1 | 1.00 | 9.26 |
| 2 | 1.00 | 9.27 |
| 3 | 1.00 | 9.34 |
| 4 | 1.00 | 9.26 |
| 5 | 1.00 | 9.33 |
| 6 | 1.00 | 9.33 |
| 7 | 1.00 | 9.34 |
| 8 | 1.00 | 9.35 |
| 9 | 1.00 | 9.41 |
| 10 | 1.00 | 9.46 |
| 11 | 1.00 | 9.47 |
| 12 | 1.00 | 9.50 |
| 13 | 1.00 | 9.33 |
| 14 | 1.00 | 9.35 |
| 15 | 1.00 | 9.28 |
| 16 | 1.00 | 9.21 |
| 17 | 1.00 | 8.80 |
| 18 | 1.00 | 8.38 |
| 19 | 1.00 | 7.94 |
| 20 | 1.00 | 6.95 |
| 21 | 1.00 | 6.47 |
| 22 | 1.00 | 5.96 |
| 23 | 1.00 | 5.16 |
| 24 | 1.00 | 4.47 |
| 25 | 1.00 | 3.64 |
| 26 | 1.00 | 3.40 |
| 27 | 1.00 | 2.93 |
| 28 | 1.00 | 2.58 |
| 29 | 1.00 | 2.09 |
| 30 | 1.00 | 1.50 |
| 31 | 1.00 | 1.04 |

### llama8b_t02_s123_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **28**
- N joined to logit-lens sidecar: 28
- Num layers: 32
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 9.36 |
| 1 | 1.00 | 9.25 |
| 2 | 1.00 | 9.24 |
| 3 | 1.00 | 9.32 |
| 4 | 1.00 | 9.22 |
| 5 | 1.00 | 9.33 |
| 6 | 1.00 | 9.33 |
| 7 | 1.00 | 9.33 |
| 8 | 1.00 | 9.26 |
| 9 | 1.00 | 9.36 |
| 10 | 1.00 | 9.43 |
| 11 | 1.00 | 9.44 |
| 12 | 1.00 | 9.47 |
| 13 | 1.00 | 9.30 |
| 14 | 1.00 | 9.33 |
| 15 | 1.00 | 9.26 |
| 16 | 1.00 | 9.14 |
| 17 | 1.00 | 8.77 |
| 18 | 1.00 | 8.33 |
| 19 | 1.00 | 7.96 |
| 20 | 1.00 | 6.99 |
| 21 | 1.00 | 6.61 |
| 22 | 1.00 | 6.02 |
| 23 | 1.00 | 5.34 |
| 24 | 1.00 | 4.59 |
| 25 | 1.00 | 3.90 |
| 26 | 1.00 | 3.66 |
| 27 | 1.00 | 3.13 |
| 28 | 1.00 | 2.63 |
| 29 | 1.00 | 2.15 |
| 30 | 1.00 | 1.56 |
| 31 | 1.00 | 1.08 |

### llama8b_t02_s123_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### llama8b_t02_s123_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### llama8b_t02_s123_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **221**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## llama8b_t02_s42_informative_v2_logitlens

**Stats:** enriched_decisions_seen=609, sidecar_records_loaded=355, joined=355, unmatched=254

### llama8b_t02_s42_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **329**
- N joined to logit-lens sidecar: 329
- Num layers: 32
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 9.36 |
| 1 | 1.00 | 9.23 |
| 2 | 1.00 | 9.25 |
| 3 | 1.00 | 9.31 |
| 4 | 1.00 | 9.23 |
| 5 | 1.00 | 9.31 |
| 6 | 1.00 | 9.32 |
| 7 | 1.00 | 9.34 |
| 8 | 1.00 | 9.36 |
| 9 | 1.00 | 9.40 |
| 10 | 1.00 | 9.45 |
| 11 | 1.00 | 9.45 |
| 12 | 1.00 | 9.49 |
| 13 | 1.00 | 9.32 |
| 14 | 1.00 | 9.32 |
| 15 | 1.00 | 9.26 |
| 16 | 1.00 | 9.17 |
| 17 | 1.00 | 8.80 |
| 18 | 1.00 | 8.37 |
| 19 | 1.00 | 8.02 |
| 20 | 1.00 | 6.95 |
| 21 | 1.00 | 6.39 |
| 22 | 1.00 | 5.89 |
| 23 | 1.00 | 5.19 |
| 24 | 1.00 | 4.51 |
| 25 | 1.00 | 3.73 |
| 26 | 1.00 | 3.46 |
| 27 | 1.00 | 3.00 |
| 28 | 1.00 | 2.62 |
| 29 | 1.00 | 2.15 |
| 30 | 1.00 | 1.53 |
| 31 | 1.00 | 1.06 |

### llama8b_t02_s42_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **26**
- N joined to logit-lens sidecar: 26
- Num layers: 32
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 9.37 |
| 1 | 1.00 | 9.25 |
| 2 | 1.00 | 9.26 |
| 3 | 1.00 | 9.32 |
| 4 | 1.00 | 9.23 |
| 5 | 1.00 | 9.36 |
| 6 | 1.00 | 9.36 |
| 7 | 1.00 | 9.34 |
| 8 | 1.00 | 9.28 |
| 9 | 1.00 | 9.40 |
| 10 | 1.00 | 9.46 |
| 11 | 1.00 | 9.47 |
| 12 | 1.00 | 9.50 |
| 13 | 1.00 | 9.32 |
| 14 | 1.00 | 9.34 |
| 15 | 1.00 | 9.25 |
| 16 | 1.00 | 9.15 |
| 17 | 1.00 | 8.80 |
| 18 | 1.00 | 8.38 |
| 19 | 1.00 | 8.05 |
| 20 | 1.00 | 7.04 |
| 21 | 1.00 | 6.52 |
| 22 | 1.00 | 5.93 |
| 23 | 1.00 | 5.27 |
| 24 | 1.00 | 4.54 |
| 25 | 1.00 | 3.84 |
| 26 | 1.00 | 3.59 |
| 27 | 1.00 | 3.08 |
| 28 | 1.00 | 2.63 |
| 29 | 1.00 | 2.14 |
| 30 | 1.00 | 1.53 |
| 31 | 1.00 | 1.06 |

### llama8b_t02_s42_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### llama8b_t02_s42_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### llama8b_t02_s42_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **254**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## llama8b_t02_s456_informative_v2_logitlens

**Stats:** enriched_decisions_seen=519, sidecar_records_loaded=311, joined=311, unmatched=208

### llama8b_t02_s456_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **290**
- N joined to logit-lens sidecar: 290
- Num layers: 32
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 9.38 |
| 1 | 1.00 | 9.26 |
| 2 | 1.00 | 9.27 |
| 3 | 1.00 | 9.34 |
| 4 | 1.00 | 9.25 |
| 5 | 1.00 | 9.34 |
| 6 | 1.00 | 9.34 |
| 7 | 1.00 | 9.35 |
| 8 | 1.00 | 9.36 |
| 9 | 1.00 | 9.41 |
| 10 | 1.00 | 9.46 |
| 11 | 1.00 | 9.47 |
| 12 | 1.00 | 9.51 |
| 13 | 1.00 | 9.34 |
| 14 | 1.00 | 9.36 |
| 15 | 1.00 | 9.29 |
| 16 | 1.00 | 9.21 |
| 17 | 1.00 | 8.80 |
| 18 | 1.00 | 8.38 |
| 19 | 1.00 | 7.95 |
| 20 | 1.00 | 6.94 |
| 21 | 1.00 | 6.47 |
| 22 | 1.00 | 5.98 |
| 23 | 1.00 | 5.15 |
| 24 | 1.00 | 4.44 |
| 25 | 1.00 | 3.60 |
| 26 | 1.00 | 3.36 |
| 27 | 1.00 | 2.88 |
| 28 | 1.00 | 2.55 |
| 29 | 1.00 | 2.06 |
| 30 | 1.00 | 1.48 |
| 31 | 1.00 | 1.03 |

### llama8b_t02_s456_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **21**
- N joined to logit-lens sidecar: 21
- Num layers: 32
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 9.35 |
| 1 | 1.00 | 9.22 |
| 2 | 1.00 | 9.22 |
| 3 | 1.00 | 9.30 |
| 4 | 1.00 | 9.21 |
| 5 | 1.00 | 9.32 |
| 6 | 1.00 | 9.32 |
| 7 | 1.00 | 9.30 |
| 8 | 1.00 | 9.24 |
| 9 | 1.00 | 9.33 |
| 10 | 1.00 | 9.40 |
| 11 | 1.00 | 9.41 |
| 12 | 1.00 | 9.43 |
| 13 | 1.00 | 9.28 |
| 14 | 1.00 | 9.30 |
| 15 | 1.00 | 9.23 |
| 16 | 1.00 | 9.13 |
| 17 | 1.00 | 8.77 |
| 18 | 1.00 | 8.34 |
| 19 | 1.00 | 8.04 |
| 20 | 1.00 | 7.08 |
| 21 | 1.00 | 6.58 |
| 22 | 1.00 | 5.96 |
| 23 | 1.00 | 5.26 |
| 24 | 1.00 | 4.50 |
| 25 | 1.00 | 3.81 |
| 26 | 1.00 | 3.57 |
| 27 | 1.00 | 3.06 |
| 28 | 1.00 | 2.61 |
| 29 | 1.00 | 2.16 |
| 30 | 1.00 | 1.57 |
| 31 | 1.00 | 1.09 |

### llama8b_t02_s456_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### llama8b_t02_s456_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### llama8b_t02_s456_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **208**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## llama8b_t0_s123_informative_v2_logitlens

**Stats:** enriched_decisions_seen=596, sidecar_records_loaded=351, joined=351, unmatched=245

### llama8b_t0_s123_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **317**
- N joined to logit-lens sidecar: 317
- Num layers: 32
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 9.38 |
| 1 | 1.00 | 9.26 |
| 2 | 1.00 | 9.27 |
| 3 | 1.00 | 9.34 |
| 4 | 1.00 | 9.25 |
| 5 | 1.00 | 9.33 |
| 6 | 1.00 | 9.33 |
| 7 | 1.00 | 9.35 |
| 8 | 1.00 | 9.36 |
| 9 | 1.00 | 9.41 |
| 10 | 1.00 | 9.46 |
| 11 | 1.00 | 9.47 |
| 12 | 1.00 | 9.50 |
| 13 | 1.00 | 9.34 |
| 14 | 1.00 | 9.35 |
| 15 | 1.00 | 9.28 |
| 16 | 1.00 | 9.20 |
| 17 | 1.00 | 8.79 |
| 18 | 1.00 | 8.39 |
| 19 | 1.00 | 7.96 |
| 20 | 1.00 | 6.93 |
| 21 | 1.00 | 6.46 |
| 22 | 1.00 | 5.95 |
| 23 | 1.00 | 5.15 |
| 24 | 1.00 | 4.49 |
| 25 | 1.00 | 3.66 |
| 26 | 1.00 | 3.42 |
| 27 | 1.00 | 2.94 |
| 28 | 1.00 | 2.58 |
| 29 | 1.00 | 2.09 |
| 30 | 1.00 | 1.50 |
| 31 | 1.00 | 1.05 |

### llama8b_t0_s123_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **34**
- N joined to logit-lens sidecar: 34
- Num layers: 32
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 9.37 |
| 1 | 1.00 | 9.24 |
| 2 | 1.00 | 9.24 |
| 3 | 1.00 | 9.30 |
| 4 | 1.00 | 9.19 |
| 5 | 1.00 | 9.33 |
| 6 | 1.00 | 9.33 |
| 7 | 1.00 | 9.32 |
| 8 | 1.00 | 9.25 |
| 9 | 1.00 | 9.35 |
| 10 | 1.00 | 9.43 |
| 11 | 1.00 | 9.45 |
| 12 | 1.00 | 9.46 |
| 13 | 1.00 | 9.30 |
| 14 | 1.00 | 9.33 |
| 15 | 1.00 | 9.24 |
| 16 | 1.00 | 9.14 |
| 17 | 1.00 | 8.74 |
| 18 | 1.00 | 8.32 |
| 19 | 1.00 | 7.96 |
| 20 | 1.00 | 7.03 |
| 21 | 1.00 | 6.67 |
| 22 | 1.00 | 6.11 |
| 23 | 1.00 | 5.41 |
| 24 | 1.00 | 4.69 |
| 25 | 1.00 | 4.00 |
| 26 | 1.00 | 3.71 |
| 27 | 1.00 | 3.16 |
| 28 | 1.00 | 2.67 |
| 29 | 1.00 | 2.13 |
| 30 | 1.00 | 1.52 |
| 31 | 1.00 | 1.05 |

### llama8b_t0_s123_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### llama8b_t0_s123_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### llama8b_t0_s123_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **245**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## llama8b_t0_s42_informative_v2_logitlens

**Stats:** enriched_decisions_seen=610, sidecar_records_loaded=350, joined=350, unmatched=260

### llama8b_t0_s42_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **334**
- N joined to logit-lens sidecar: 334
- Num layers: 32
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 9.35 |
| 1 | 1.00 | 9.23 |
| 2 | 1.00 | 9.24 |
| 3 | 1.00 | 9.30 |
| 4 | 1.00 | 9.23 |
| 5 | 1.00 | 9.30 |
| 6 | 1.00 | 9.32 |
| 7 | 1.00 | 9.34 |
| 8 | 1.00 | 9.36 |
| 9 | 1.00 | 9.40 |
| 10 | 1.00 | 9.45 |
| 11 | 1.00 | 9.45 |
| 12 | 1.00 | 9.48 |
| 13 | 1.00 | 9.31 |
| 14 | 1.00 | 9.32 |
| 15 | 1.00 | 9.26 |
| 16 | 1.00 | 9.17 |
| 17 | 1.00 | 8.78 |
| 18 | 1.00 | 8.34 |
| 19 | 1.00 | 7.97 |
| 20 | 1.00 | 6.85 |
| 21 | 1.00 | 6.31 |
| 22 | 1.00 | 5.82 |
| 23 | 1.00 | 5.14 |
| 24 | 1.00 | 4.46 |
| 25 | 1.00 | 3.66 |
| 26 | 1.00 | 3.41 |
| 27 | 1.00 | 2.97 |
| 28 | 1.00 | 2.62 |
| 29 | 1.00 | 2.15 |
| 30 | 1.00 | 1.53 |
| 31 | 1.00 | 1.07 |

### llama8b_t0_s42_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **16**
- N joined to logit-lens sidecar: 16
- Num layers: 32
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 9.36 |
| 1 | 1.00 | 9.25 |
| 2 | 1.00 | 9.25 |
| 3 | 1.00 | 9.31 |
| 4 | 1.00 | 9.22 |
| 5 | 1.00 | 9.31 |
| 6 | 1.00 | 9.30 |
| 7 | 1.00 | 9.30 |
| 8 | 1.00 | 9.24 |
| 9 | 1.00 | 9.35 |
| 10 | 1.00 | 9.42 |
| 11 | 1.00 | 9.43 |
| 12 | 1.00 | 9.44 |
| 13 | 1.00 | 9.29 |
| 14 | 1.00 | 9.31 |
| 15 | 1.00 | 9.25 |
| 16 | 1.00 | 9.13 |
| 17 | 1.00 | 8.72 |
| 18 | 1.00 | 8.30 |
| 19 | 1.00 | 7.95 |
| 20 | 1.00 | 6.98 |
| 21 | 1.00 | 6.57 |
| 22 | 1.00 | 5.87 |
| 23 | 1.00 | 5.20 |
| 24 | 1.00 | 4.53 |
| 25 | 1.00 | 3.86 |
| 26 | 1.00 | 3.61 |
| 27 | 1.00 | 3.09 |
| 28 | 1.00 | 2.58 |
| 29 | 1.00 | 2.14 |
| 30 | 1.00 | 1.51 |
| 31 | 1.00 | 1.06 |

### llama8b_t0_s42_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### llama8b_t0_s42_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### llama8b_t0_s42_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **260**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## llama8b_t0_s456_informative_v2_logitlens

**Stats:** enriched_decisions_seen=524, sidecar_records_loaded=316, joined=316, unmatched=208

### llama8b_t0_s456_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **298**
- N joined to logit-lens sidecar: 298
- Num layers: 32
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 9.38 |
| 1 | 1.00 | 9.25 |
| 2 | 1.00 | 9.28 |
| 3 | 1.00 | 9.35 |
| 4 | 1.00 | 9.25 |
| 5 | 1.00 | 9.34 |
| 6 | 1.00 | 9.34 |
| 7 | 1.00 | 9.35 |
| 8 | 1.00 | 9.36 |
| 9 | 1.00 | 9.41 |
| 10 | 1.00 | 9.46 |
| 11 | 1.00 | 9.47 |
| 12 | 1.00 | 9.51 |
| 13 | 1.00 | 9.34 |
| 14 | 1.00 | 9.36 |
| 15 | 1.00 | 9.29 |
| 16 | 1.00 | 9.21 |
| 17 | 1.00 | 8.79 |
| 18 | 1.00 | 8.38 |
| 19 | 1.00 | 7.96 |
| 20 | 1.00 | 6.90 |
| 21 | 1.00 | 6.45 |
| 22 | 1.00 | 5.96 |
| 23 | 1.00 | 5.13 |
| 24 | 1.00 | 4.46 |
| 25 | 1.00 | 3.62 |
| 26 | 1.00 | 3.37 |
| 27 | 1.00 | 2.89 |
| 28 | 1.00 | 2.55 |
| 29 | 1.00 | 2.07 |
| 30 | 1.00 | 1.50 |
| 31 | 1.00 | 1.05 |

### llama8b_t0_s456_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **18**
- N joined to logit-lens sidecar: 18
- Num layers: 32
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 9.39 |
| 1 | 1.00 | 9.28 |
| 2 | 1.00 | 9.27 |
| 3 | 1.00 | 9.35 |
| 4 | 1.00 | 9.26 |
| 5 | 1.00 | 9.36 |
| 6 | 1.00 | 9.37 |
| 7 | 1.00 | 9.33 |
| 8 | 1.00 | 9.28 |
| 9 | 1.00 | 9.37 |
| 10 | 1.00 | 9.45 |
| 11 | 1.00 | 9.44 |
| 12 | 1.00 | 9.47 |
| 13 | 1.00 | 9.32 |
| 14 | 1.00 | 9.34 |
| 15 | 1.00 | 9.24 |
| 16 | 1.00 | 9.14 |
| 17 | 1.00 | 8.76 |
| 18 | 1.00 | 8.28 |
| 19 | 1.00 | 7.94 |
| 20 | 1.00 | 6.97 |
| 21 | 1.00 | 6.59 |
| 22 | 1.00 | 5.99 |
| 23 | 1.00 | 5.31 |
| 24 | 1.00 | 4.57 |
| 25 | 1.00 | 3.86 |
| 26 | 1.00 | 3.58 |
| 27 | 1.00 | 3.03 |
| 28 | 1.00 | 2.60 |
| 29 | 1.00 | 2.09 |
| 30 | 1.00 | 1.54 |
| 31 | 1.00 | 1.05 |

### llama8b_t0_s456_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### llama8b_t0_s456_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### llama8b_t0_s456_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **208**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## qwen8b_t02_s123_informative_v2_logitlens

**Stats:** enriched_decisions_seen=683, sidecar_records_loaded=394, joined=394, unmatched=289

### qwen8b_t02_s123_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **385**
- N joined to logit-lens sidecar: 385
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.28 |
| 1 | 1.00 | 6.02 |
| 2 | 1.00 | 6.31 |
| 3 | 1.00 | 6.50 |
| 4 | 1.00 | 7.21 |
| 5 | 1.00 | 7.27 |
| 6 | 1.00 | 7.25 |
| 7 | 1.00 | 7.42 |
| 8 | 1.00 | 7.09 |
| 9 | 1.00 | 6.69 |
| 10 | 1.00 | 6.60 |
| 11 | 1.00 | 6.59 |
| 12 | 1.00 | 6.59 |
| 13 | 1.00 | 6.43 |
| 14 | 1.00 | 6.44 |
| 15 | 1.00 | 6.55 |
| 16 | 1.00 | 6.50 |
| 17 | 1.00 | 6.39 |
| 18 | 1.00 | 6.30 |
| 19 | 1.00 | 5.99 |
| 20 | 1.00 | 5.52 |
| 21 | 1.00 | 5.20 |
| 22 | 1.00 | 4.02 |
| 23 | 1.00 | 2.99 |
| 24 | 1.00 | 3.11 |
| 25 | 1.00 | 2.72 |
| 26 | 1.00 | 2.69 |
| 27 | 1.00 | 1.89 |
| 28 | 1.00 | 1.54 |
| 29 | 1.00 | 1.30 |
| 30 | 1.00 | 1.24 |
| 31 | 1.00 | 0.93 |
| 32 | 1.00 | 0.74 |
| 33 | 1.00 | 0.66 |
| 34 | 1.00 | 0.85 |
| 35 | 1.00 | 0.32 |

### qwen8b_t02_s123_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **8**
- N joined to logit-lens sidecar: 8
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.36 |
| 1 | 1.00 | 6.10 |
| 2 | 1.00 | 6.38 |
| 3 | 1.00 | 6.55 |
| 4 | 1.00 | 7.25 |
| 5 | 1.00 | 7.46 |
| 6 | 1.00 | 7.49 |
| 7 | 1.00 | 7.59 |
| 8 | 1.00 | 7.33 |
| 9 | 1.00 | 6.86 |
| 10 | 1.00 | 6.72 |
| 11 | 1.00 | 6.74 |
| 12 | 1.00 | 6.83 |
| 13 | 1.00 | 6.67 |
| 14 | 1.00 | 6.76 |
| 15 | 1.00 | 6.80 |
| 16 | 1.00 | 6.74 |
| 17 | 1.00 | 6.54 |
| 18 | 1.00 | 6.37 |
| 19 | 1.00 | 6.13 |
| 20 | 1.00 | 5.55 |
| 21 | 1.00 | 5.17 |
| 22 | 1.00 | 3.99 |
| 23 | 1.00 | 3.07 |
| 24 | 1.00 | 3.27 |
| 25 | 1.00 | 2.74 |
| 26 | 1.00 | 2.68 |
| 27 | 1.00 | 1.84 |
| 28 | 1.00 | 1.42 |
| 29 | 1.00 | 1.41 |
| 30 | 1.00 | 1.36 |
| 31 | 1.00 | 1.01 |
| 32 | 1.00 | 0.82 |
| 33 | 1.00 | 0.69 |
| 34 | 1.00 | 0.84 |
| 35 | 1.00 | 0.31 |

### qwen8b_t02_s123_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **1**
- N joined to logit-lens sidecar: 1
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.37 |
| 1 | 1.00 | 6.27 |
| 2 | 1.00 | 6.53 |
| 3 | 1.00 | 6.87 |
| 4 | 1.00 | 7.68 |
| 5 | 1.00 | 7.63 |
| 6 | 1.00 | 7.73 |
| 7 | 1.00 | 7.86 |
| 8 | 1.00 | 7.49 |
| 9 | 1.00 | 6.90 |
| 10 | 1.00 | 6.82 |
| 11 | 1.00 | 6.89 |
| 12 | 1.00 | 6.79 |
| 13 | 1.00 | 6.54 |
| 14 | 1.00 | 6.53 |
| 15 | 1.00 | 6.71 |
| 16 | 1.00 | 6.61 |
| 17 | 1.00 | 6.57 |
| 18 | 1.00 | 6.52 |
| 19 | 1.00 | 6.21 |
| 20 | 1.00 | 5.80 |
| 21 | 1.00 | 5.52 |
| 22 | 1.00 | 4.18 |
| 23 | 1.00 | 3.39 |
| 24 | 1.00 | 3.44 |
| 25 | 1.00 | 3.08 |
| 26 | 1.00 | 3.09 |
| 27 | 1.00 | 1.95 |
| 28 | 1.00 | 1.68 |
| 29 | 1.00 | 1.37 |
| 30 | 1.00 | 1.31 |
| 31 | 1.00 | 1.01 |
| 32 | 1.00 | 0.70 |
| 33 | 1.00 | 0.62 |
| 34 | 1.00 | 0.71 |
| 35 | 1.00 | 0.32 |

### qwen8b_t02_s123_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### qwen8b_t02_s123_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **289**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## qwen8b_t02_s42_informative_v2_logitlens

**Stats:** enriched_decisions_seen=801, sidecar_records_loaded=460, joined=460, unmatched=341

### qwen8b_t02_s42_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **455**
- N joined to logit-lens sidecar: 455
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.21 |
| 1 | 1.00 | 5.97 |
| 2 | 1.00 | 6.27 |
| 3 | 1.00 | 6.45 |
| 4 | 1.00 | 7.21 |
| 5 | 1.00 | 7.27 |
| 6 | 1.00 | 7.23 |
| 7 | 1.00 | 7.39 |
| 8 | 1.00 | 7.02 |
| 9 | 1.00 | 6.68 |
| 10 | 1.00 | 6.59 |
| 11 | 1.00 | 6.58 |
| 12 | 1.00 | 6.57 |
| 13 | 1.00 | 6.41 |
| 14 | 1.00 | 6.43 |
| 15 | 1.00 | 6.57 |
| 16 | 1.00 | 6.52 |
| 17 | 1.00 | 6.42 |
| 18 | 1.00 | 6.30 |
| 19 | 1.00 | 5.99 |
| 20 | 1.00 | 5.50 |
| 21 | 1.00 | 5.19 |
| 22 | 1.00 | 3.98 |
| 23 | 1.00 | 2.97 |
| 24 | 1.00 | 3.09 |
| 25 | 1.00 | 2.71 |
| 26 | 1.00 | 2.68 |
| 27 | 1.00 | 1.91 |
| 28 | 1.00 | 1.55 |
| 29 | 1.00 | 1.30 |
| 30 | 1.00 | 1.23 |
| 31 | 1.00 | 0.92 |
| 32 | 1.00 | 0.74 |
| 33 | 1.00 | 0.65 |
| 34 | 1.00 | 0.83 |
| 35 | 1.00 | 0.30 |

### qwen8b_t02_s42_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **4**
- N joined to logit-lens sidecar: 4
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.39 |
| 1 | 1.00 | 6.14 |
| 2 | 1.00 | 6.32 |
| 3 | 1.00 | 6.52 |
| 4 | 1.00 | 7.37 |
| 5 | 1.00 | 7.52 |
| 6 | 1.00 | 7.59 |
| 7 | 1.00 | 7.64 |
| 8 | 1.00 | 7.33 |
| 9 | 1.00 | 6.90 |
| 10 | 1.00 | 6.64 |
| 11 | 1.00 | 6.68 |
| 12 | 1.00 | 6.77 |
| 13 | 1.00 | 6.62 |
| 14 | 1.00 | 6.69 |
| 15 | 1.00 | 6.75 |
| 16 | 1.00 | 6.67 |
| 17 | 1.00 | 6.53 |
| 18 | 1.00 | 6.34 |
| 19 | 1.00 | 6.04 |
| 20 | 1.00 | 5.53 |
| 21 | 1.00 | 5.20 |
| 22 | 1.00 | 4.05 |
| 23 | 1.00 | 3.09 |
| 24 | 1.00 | 3.29 |
| 25 | 1.00 | 2.90 |
| 26 | 1.00 | 2.75 |
| 27 | 1.00 | 1.82 |
| 28 | 1.00 | 1.46 |
| 29 | 1.00 | 1.40 |
| 30 | 1.00 | 1.39 |
| 31 | 1.00 | 1.05 |
| 32 | 1.00 | 0.89 |
| 33 | 1.00 | 0.74 |
| 34 | 1.00 | 0.89 |
| 35 | 1.00 | 0.35 |

### qwen8b_t02_s42_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **1**
- N joined to logit-lens sidecar: 1
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.36 |
| 1 | 1.00 | 6.13 |
| 2 | 1.00 | 6.42 |
| 3 | 1.00 | 6.58 |
| 4 | 1.00 | 7.31 |
| 5 | 1.00 | 7.46 |
| 6 | 1.00 | 7.51 |
| 7 | 1.00 | 7.57 |
| 8 | 1.00 | 7.21 |
| 9 | 1.00 | 6.66 |
| 10 | 1.00 | 6.60 |
| 11 | 1.00 | 6.68 |
| 12 | 1.00 | 6.63 |
| 13 | 1.00 | 6.63 |
| 14 | 1.00 | 6.68 |
| 15 | 1.00 | 6.83 |
| 16 | 1.00 | 6.67 |
| 17 | 1.00 | 6.56 |
| 18 | 1.00 | 6.43 |
| 19 | 1.00 | 6.26 |
| 20 | 1.00 | 5.96 |
| 21 | 1.00 | 5.84 |
| 22 | 1.00 | 4.68 |
| 23 | 1.00 | 3.64 |
| 24 | 1.00 | 3.92 |
| 25 | 1.00 | 3.46 |
| 26 | 1.00 | 3.23 |
| 27 | 1.00 | 2.19 |
| 28 | 1.00 | 1.90 |
| 29 | 1.00 | 1.62 |
| 30 | 1.00 | 1.54 |
| 31 | 1.00 | 1.17 |
| 32 | 1.00 | 0.91 |
| 33 | 1.00 | 0.66 |
| 34 | 1.00 | 0.83 |
| 35 | 1.00 | 0.36 |

### qwen8b_t02_s42_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### qwen8b_t02_s42_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **341**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## qwen8b_t02_s456_informative_v2_logitlens

**Stats:** enriched_decisions_seen=570, sidecar_records_loaded=330, joined=330, unmatched=240

### qwen8b_t02_s456_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **323**
- N joined to logit-lens sidecar: 323
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.30 |
| 1 | 1.00 | 6.04 |
| 2 | 1.00 | 6.34 |
| 3 | 1.00 | 6.54 |
| 4 | 1.00 | 7.25 |
| 5 | 1.00 | 7.30 |
| 6 | 1.00 | 7.29 |
| 7 | 1.00 | 7.45 |
| 8 | 1.00 | 7.12 |
| 9 | 1.00 | 6.72 |
| 10 | 1.00 | 6.62 |
| 11 | 1.00 | 6.60 |
| 12 | 1.00 | 6.59 |
| 13 | 1.00 | 6.44 |
| 14 | 1.00 | 6.46 |
| 15 | 1.00 | 6.56 |
| 16 | 1.00 | 6.50 |
| 17 | 1.00 | 6.40 |
| 18 | 1.00 | 6.30 |
| 19 | 1.00 | 6.00 |
| 20 | 1.00 | 5.53 |
| 21 | 1.00 | 5.24 |
| 22 | 1.00 | 4.04 |
| 23 | 1.00 | 2.99 |
| 24 | 1.00 | 3.09 |
| 25 | 1.00 | 2.72 |
| 26 | 1.00 | 2.69 |
| 27 | 1.00 | 1.91 |
| 28 | 1.00 | 1.55 |
| 29 | 1.00 | 1.31 |
| 30 | 1.00 | 1.24 |
| 31 | 1.00 | 0.93 |
| 32 | 1.00 | 0.73 |
| 33 | 1.00 | 0.65 |
| 34 | 1.00 | 0.85 |
| 35 | 1.00 | 0.33 |

### qwen8b_t02_s456_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **6**
- N joined to logit-lens sidecar: 6
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.34 |
| 1 | 1.00 | 6.07 |
| 2 | 1.00 | 6.37 |
| 3 | 1.00 | 6.59 |
| 4 | 1.00 | 7.33 |
| 5 | 1.00 | 7.49 |
| 6 | 1.00 | 7.50 |
| 7 | 1.00 | 7.64 |
| 8 | 1.00 | 7.34 |
| 9 | 1.00 | 6.85 |
| 10 | 1.00 | 6.68 |
| 11 | 1.00 | 6.78 |
| 12 | 1.00 | 6.83 |
| 13 | 1.00 | 6.68 |
| 14 | 1.00 | 6.72 |
| 15 | 1.00 | 6.79 |
| 16 | 1.00 | 6.70 |
| 17 | 1.00 | 6.52 |
| 18 | 1.00 | 6.42 |
| 19 | 1.00 | 6.18 |
| 20 | 1.00 | 5.63 |
| 21 | 1.00 | 5.29 |
| 22 | 1.00 | 4.07 |
| 23 | 1.00 | 3.08 |
| 24 | 1.00 | 3.24 |
| 25 | 1.00 | 2.78 |
| 26 | 1.00 | 2.65 |
| 27 | 1.00 | 1.82 |
| 28 | 1.00 | 1.45 |
| 29 | 1.00 | 1.39 |
| 30 | 1.00 | 1.33 |
| 31 | 1.00 | 0.96 |
| 32 | 1.00 | 0.80 |
| 33 | 1.00 | 0.65 |
| 34 | 1.00 | 0.81 |
| 35 | 1.00 | 0.33 |

### qwen8b_t02_s456_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **1**
- N joined to logit-lens sidecar: 1
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.35 |
| 1 | 1.00 | 6.05 |
| 2 | 1.00 | 6.34 |
| 3 | 1.00 | 6.38 |
| 4 | 1.00 | 7.06 |
| 5 | 1.00 | 7.09 |
| 6 | 1.00 | 7.09 |
| 7 | 1.00 | 7.28 |
| 8 | 1.00 | 6.87 |
| 9 | 1.00 | 6.39 |
| 10 | 1.00 | 6.16 |
| 11 | 1.00 | 6.25 |
| 12 | 1.00 | 6.25 |
| 13 | 1.00 | 6.22 |
| 14 | 1.00 | 6.31 |
| 15 | 1.00 | 6.39 |
| 16 | 1.00 | 6.30 |
| 17 | 1.00 | 6.28 |
| 18 | 1.00 | 6.27 |
| 19 | 1.00 | 6.01 |
| 20 | 1.00 | 5.75 |
| 21 | 1.00 | 5.36 |
| 22 | 1.00 | 4.34 |
| 23 | 1.00 | 3.14 |
| 24 | 1.00 | 3.32 |
| 25 | 1.00 | 2.97 |
| 26 | 1.00 | 2.73 |
| 27 | 1.00 | 2.08 |
| 28 | 1.00 | 1.71 |
| 29 | 1.00 | 1.36 |
| 30 | 1.00 | 1.32 |
| 31 | 1.00 | 0.92 |
| 32 | 1.00 | 0.76 |
| 33 | 1.00 | 0.66 |
| 34 | 1.00 | 0.98 |
| 35 | 1.00 | 0.50 |

### qwen8b_t02_s456_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### qwen8b_t02_s456_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **240**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## qwen8b_t0_s123_informative_v2_logitlens

**Stats:** enriched_decisions_seen=603, sidecar_records_loaded=350, joined=350, unmatched=253

### qwen8b_t0_s123_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **340**
- N joined to logit-lens sidecar: 340
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.30 |
| 1 | 1.00 | 6.04 |
| 2 | 1.00 | 6.34 |
| 3 | 1.00 | 6.53 |
| 4 | 1.00 | 7.23 |
| 5 | 1.00 | 7.30 |
| 6 | 1.00 | 7.28 |
| 7 | 1.00 | 7.44 |
| 8 | 1.00 | 7.11 |
| 9 | 1.00 | 6.70 |
| 10 | 1.00 | 6.61 |
| 11 | 1.00 | 6.60 |
| 12 | 1.00 | 6.60 |
| 13 | 1.00 | 6.44 |
| 14 | 1.00 | 6.46 |
| 15 | 1.00 | 6.58 |
| 16 | 1.00 | 6.52 |
| 17 | 1.00 | 6.41 |
| 18 | 1.00 | 6.31 |
| 19 | 1.00 | 6.00 |
| 20 | 1.00 | 5.53 |
| 21 | 1.00 | 5.22 |
| 22 | 1.00 | 4.03 |
| 23 | 1.00 | 2.99 |
| 24 | 1.00 | 3.09 |
| 25 | 1.00 | 2.71 |
| 26 | 1.00 | 2.69 |
| 27 | 1.00 | 1.89 |
| 28 | 1.00 | 1.54 |
| 29 | 1.00 | 1.30 |
| 30 | 1.00 | 1.25 |
| 31 | 1.00 | 0.93 |
| 32 | 1.00 | 0.73 |
| 33 | 1.00 | 0.66 |
| 34 | 1.00 | 0.85 |
| 35 | 1.00 | 0.32 |

### qwen8b_t0_s123_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **9**
- N joined to logit-lens sidecar: 9
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.35 |
| 1 | 1.00 | 6.08 |
| 2 | 1.00 | 6.38 |
| 3 | 1.00 | 6.50 |
| 4 | 1.00 | 7.23 |
| 5 | 1.00 | 7.45 |
| 6 | 1.00 | 7.48 |
| 7 | 1.00 | 7.56 |
| 8 | 1.00 | 7.28 |
| 9 | 1.00 | 6.81 |
| 10 | 1.00 | 6.67 |
| 11 | 1.00 | 6.75 |
| 12 | 1.00 | 6.82 |
| 13 | 1.00 | 6.65 |
| 14 | 1.00 | 6.75 |
| 15 | 1.00 | 6.76 |
| 16 | 1.00 | 6.68 |
| 17 | 1.00 | 6.51 |
| 18 | 1.00 | 6.35 |
| 19 | 1.00 | 6.09 |
| 20 | 1.00 | 5.51 |
| 21 | 1.00 | 5.13 |
| 22 | 1.00 | 3.92 |
| 23 | 1.00 | 2.99 |
| 24 | 1.00 | 3.15 |
| 25 | 1.00 | 2.66 |
| 26 | 1.00 | 2.59 |
| 27 | 1.00 | 1.77 |
| 28 | 1.00 | 1.35 |
| 29 | 1.00 | 1.36 |
| 30 | 1.00 | 1.29 |
| 31 | 1.00 | 0.97 |
| 32 | 1.00 | 0.79 |
| 33 | 1.00 | 0.68 |
| 34 | 1.00 | 0.82 |
| 35 | 1.00 | 0.31 |

### qwen8b_t0_s123_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **1**
- N joined to logit-lens sidecar: 1
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.39 |
| 1 | 1.00 | 6.13 |
| 2 | 1.00 | 6.38 |
| 3 | 1.00 | 6.46 |
| 4 | 1.00 | 7.24 |
| 5 | 1.00 | 7.30 |
| 6 | 1.00 | 7.41 |
| 7 | 1.00 | 7.45 |
| 8 | 1.00 | 7.00 |
| 9 | 1.00 | 6.39 |
| 10 | 1.00 | 6.25 |
| 11 | 1.00 | 6.32 |
| 12 | 1.00 | 6.43 |
| 13 | 1.00 | 6.42 |
| 14 | 1.00 | 6.39 |
| 15 | 1.00 | 6.68 |
| 16 | 1.00 | 6.36 |
| 17 | 1.00 | 6.35 |
| 18 | 1.00 | 6.40 |
| 19 | 1.00 | 6.15 |
| 20 | 1.00 | 5.71 |
| 21 | 1.00 | 5.54 |
| 22 | 1.00 | 4.02 |
| 23 | 1.00 | 3.11 |
| 24 | 1.00 | 3.21 |
| 25 | 1.00 | 2.88 |
| 26 | 1.00 | 2.95 |
| 27 | 1.00 | 1.88 |
| 28 | 1.00 | 1.61 |
| 29 | 1.00 | 1.32 |
| 30 | 1.00 | 1.23 |
| 31 | 1.00 | 0.95 |
| 32 | 1.00 | 0.67 |
| 33 | 1.00 | 0.46 |
| 34 | 1.00 | 0.60 |
| 35 | 1.00 | 0.27 |

### qwen8b_t0_s123_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### qwen8b_t0_s123_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **253**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## qwen8b_t0_s42_informative_v2_logitlens

**Stats:** enriched_decisions_seen=796, sidecar_records_loaded=455, joined=455, unmatched=341

### qwen8b_t0_s42_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **449**
- N joined to logit-lens sidecar: 449
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.21 |
| 1 | 1.00 | 5.96 |
| 2 | 1.00 | 6.27 |
| 3 | 1.00 | 6.45 |
| 4 | 1.00 | 7.21 |
| 5 | 1.00 | 7.28 |
| 6 | 1.00 | 7.24 |
| 7 | 1.00 | 7.39 |
| 8 | 1.00 | 7.01 |
| 9 | 1.00 | 6.68 |
| 10 | 1.00 | 6.57 |
| 11 | 1.00 | 6.57 |
| 12 | 1.00 | 6.57 |
| 13 | 1.00 | 6.41 |
| 14 | 1.00 | 6.44 |
| 15 | 1.00 | 6.57 |
| 16 | 1.00 | 6.52 |
| 17 | 1.00 | 6.42 |
| 18 | 1.00 | 6.29 |
| 19 | 1.00 | 5.99 |
| 20 | 1.00 | 5.50 |
| 21 | 1.00 | 5.18 |
| 22 | 1.00 | 3.97 |
| 23 | 1.00 | 2.96 |
| 24 | 1.00 | 3.08 |
| 25 | 1.00 | 2.69 |
| 26 | 1.00 | 2.67 |
| 27 | 1.00 | 1.90 |
| 28 | 1.00 | 1.54 |
| 29 | 1.00 | 1.28 |
| 30 | 1.00 | 1.22 |
| 31 | 1.00 | 0.91 |
| 32 | 1.00 | 0.73 |
| 33 | 1.00 | 0.64 |
| 34 | 1.00 | 0.82 |
| 35 | 1.00 | 0.30 |

### qwen8b_t0_s42_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **4**
- N joined to logit-lens sidecar: 4
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.31 |
| 1 | 1.00 | 5.96 |
| 2 | 1.00 | 6.25 |
| 3 | 1.00 | 6.55 |
| 4 | 1.00 | 7.31 |
| 5 | 1.00 | 7.39 |
| 6 | 1.00 | 7.42 |
| 7 | 1.00 | 7.56 |
| 8 | 1.00 | 7.30 |
| 9 | 1.00 | 6.77 |
| 10 | 1.00 | 6.63 |
| 11 | 1.00 | 6.76 |
| 12 | 1.00 | 6.74 |
| 13 | 1.00 | 6.60 |
| 14 | 1.00 | 6.64 |
| 15 | 1.00 | 6.71 |
| 16 | 1.00 | 6.58 |
| 17 | 1.00 | 6.43 |
| 18 | 1.00 | 6.36 |
| 19 | 1.00 | 6.00 |
| 20 | 1.00 | 5.52 |
| 21 | 1.00 | 5.17 |
| 22 | 1.00 | 4.01 |
| 23 | 1.00 | 3.04 |
| 24 | 1.00 | 3.17 |
| 25 | 1.00 | 2.71 |
| 26 | 1.00 | 2.59 |
| 27 | 1.00 | 1.82 |
| 28 | 1.00 | 1.41 |
| 29 | 1.00 | 1.32 |
| 30 | 1.00 | 1.30 |
| 31 | 1.00 | 0.99 |
| 32 | 1.00 | 0.85 |
| 33 | 1.00 | 0.68 |
| 34 | 1.00 | 0.87 |
| 35 | 1.00 | 0.37 |

### qwen8b_t0_s42_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **2**
- N joined to logit-lens sidecar: 2
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.36 |
| 1 | 1.00 | 6.21 |
| 2 | 1.00 | 6.49 |
| 3 | 1.00 | 6.70 |
| 4 | 1.00 | 7.42 |
| 5 | 1.00 | 7.51 |
| 6 | 1.00 | 7.54 |
| 7 | 1.00 | 7.68 |
| 8 | 1.00 | 7.32 |
| 9 | 1.00 | 6.72 |
| 10 | 1.00 | 6.65 |
| 11 | 1.00 | 6.69 |
| 12 | 1.00 | 6.68 |
| 13 | 1.00 | 6.72 |
| 14 | 1.00 | 6.74 |
| 15 | 1.00 | 6.90 |
| 16 | 1.00 | 6.75 |
| 17 | 1.00 | 6.67 |
| 18 | 1.00 | 6.54 |
| 19 | 1.00 | 6.29 |
| 20 | 1.00 | 5.90 |
| 21 | 1.00 | 5.59 |
| 22 | 1.00 | 4.42 |
| 23 | 1.00 | 3.46 |
| 24 | 1.00 | 3.57 |
| 25 | 1.00 | 3.17 |
| 26 | 1.00 | 3.00 |
| 27 | 1.00 | 2.06 |
| 28 | 1.00 | 1.70 |
| 29 | 1.00 | 1.34 |
| 30 | 1.00 | 1.32 |
| 31 | 1.00 | 0.99 |
| 32 | 1.00 | 0.78 |
| 33 | 1.00 | 0.58 |
| 34 | 1.00 | 0.75 |
| 35 | 1.00 | 0.28 |

### qwen8b_t0_s42_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### qwen8b_t0_s42_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **341**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## qwen8b_t0_s456_informative_v2_logitlens

**Stats:** enriched_decisions_seen=616, sidecar_records_loaded=357, joined=357, unmatched=259

### qwen8b_t0_s456_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **345**
- N joined to logit-lens sidecar: 345
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.31 |
| 1 | 1.00 | 6.06 |
| 2 | 1.00 | 6.36 |
| 3 | 1.00 | 6.55 |
| 4 | 1.00 | 7.24 |
| 5 | 1.00 | 7.30 |
| 6 | 1.00 | 7.30 |
| 7 | 1.00 | 7.45 |
| 8 | 1.00 | 7.11 |
| 9 | 1.00 | 6.70 |
| 10 | 1.00 | 6.61 |
| 11 | 1.00 | 6.60 |
| 12 | 1.00 | 6.59 |
| 13 | 1.00 | 6.45 |
| 14 | 1.00 | 6.46 |
| 15 | 1.00 | 6.57 |
| 16 | 1.00 | 6.51 |
| 17 | 1.00 | 6.41 |
| 18 | 1.00 | 6.31 |
| 19 | 1.00 | 6.01 |
| 20 | 1.00 | 5.54 |
| 21 | 1.00 | 5.24 |
| 22 | 1.00 | 4.03 |
| 23 | 1.00 | 3.01 |
| 24 | 1.00 | 3.10 |
| 25 | 1.00 | 2.73 |
| 26 | 1.00 | 2.70 |
| 27 | 1.00 | 1.90 |
| 28 | 1.00 | 1.56 |
| 29 | 1.00 | 1.31 |
| 30 | 1.00 | 1.25 |
| 31 | 1.00 | 0.94 |
| 32 | 1.00 | 0.74 |
| 33 | 1.00 | 0.66 |
| 34 | 1.00 | 0.85 |
| 35 | 1.00 | 0.33 |

### qwen8b_t0_s456_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **11**
- N joined to logit-lens sidecar: 11
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.41 |
| 1 | 1.00 | 6.13 |
| 2 | 1.00 | 6.40 |
| 3 | 1.00 | 6.65 |
| 4 | 1.00 | 7.38 |
| 5 | 1.00 | 7.59 |
| 6 | 1.00 | 7.66 |
| 7 | 1.00 | 7.74 |
| 8 | 1.00 | 7.50 |
| 9 | 1.00 | 6.97 |
| 10 | 1.00 | 6.77 |
| 11 | 1.00 | 6.83 |
| 12 | 1.00 | 6.91 |
| 13 | 1.00 | 6.75 |
| 14 | 1.00 | 6.83 |
| 15 | 1.00 | 6.89 |
| 16 | 1.00 | 6.81 |
| 17 | 1.00 | 6.61 |
| 18 | 1.00 | 6.50 |
| 19 | 1.00 | 6.21 |
| 20 | 1.00 | 5.62 |
| 21 | 1.00 | 5.26 |
| 22 | 1.00 | 4.05 |
| 23 | 1.00 | 3.13 |
| 24 | 1.00 | 3.28 |
| 25 | 1.00 | 2.82 |
| 26 | 1.00 | 2.71 |
| 27 | 1.00 | 1.82 |
| 28 | 1.00 | 1.46 |
| 29 | 1.00 | 1.38 |
| 30 | 1.00 | 1.39 |
| 31 | 1.00 | 1.01 |
| 32 | 1.00 | 0.83 |
| 33 | 1.00 | 0.71 |
| 34 | 1.00 | 0.85 |
| 35 | 1.00 | 0.33 |

### qwen8b_t0_s456_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **1**
- N joined to logit-lens sidecar: 1
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 6.33 |
| 1 | 1.00 | 5.93 |
| 2 | 1.00 | 6.17 |
| 3 | 1.00 | 6.36 |
| 4 | 1.00 | 7.27 |
| 5 | 1.00 | 7.33 |
| 6 | 1.00 | 7.32 |
| 7 | 1.00 | 7.48 |
| 8 | 1.00 | 7.13 |
| 9 | 1.00 | 6.64 |
| 10 | 1.00 | 6.43 |
| 11 | 1.00 | 6.38 |
| 12 | 1.00 | 6.46 |
| 13 | 1.00 | 6.33 |
| 14 | 1.00 | 6.30 |
| 15 | 1.00 | 6.35 |
| 16 | 1.00 | 6.25 |
| 17 | 1.00 | 6.17 |
| 18 | 1.00 | 6.26 |
| 19 | 1.00 | 5.92 |
| 20 | 1.00 | 5.46 |
| 21 | 1.00 | 5.21 |
| 22 | 1.00 | 4.01 |
| 23 | 1.00 | 3.14 |
| 24 | 1.00 | 3.17 |
| 25 | 1.00 | 2.65 |
| 26 | 1.00 | 2.46 |
| 27 | 1.00 | 1.79 |
| 28 | 1.00 | 1.59 |
| 29 | 1.00 | 1.37 |
| 30 | 1.00 | 1.20 |
| 31 | 1.00 | 0.95 |
| 32 | 1.00 | 0.71 |
| 33 | 1.00 | 0.62 |
| 34 | 1.00 | 0.88 |
| 35 | 1.00 | 0.30 |

### qwen8b_t0_s456_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### qwen8b_t0_s456_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **259**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## ministral8b_t02_s123_informative_v2_logitlens

**Stats:** enriched_decisions_seen=209, sidecar_records_loaded=106, joined=106, unmatched=103

### ministral8b_t02_s123_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **104**
- N joined to logit-lens sidecar: 104
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 10.54 |
| 1 | 1.00 | 9.97 |
| 2 | 1.00 | 10.27 |
| 3 | 1.00 | 10.21 |
| 4 | 1.00 | 10.30 |
| 5 | 1.00 | 10.27 |
| 6 | 1.00 | 10.35 |
| 7 | 1.00 | 10.14 |
| 8 | 1.00 | 10.12 |
| 9 | 1.00 | 10.08 |
| 10 | 1.00 | 10.31 |
| 11 | 1.00 | 10.44 |
| 12 | 1.00 | 10.50 |
| 13 | 1.00 | 10.45 |
| 14 | 1.00 | 10.42 |
| 15 | 1.00 | 10.10 |
| 16 | 1.00 | 10.12 |
| 17 | 1.00 | 9.99 |
| 18 | 1.00 | 9.68 |
| 19 | 1.00 | 9.27 |
| 20 | 1.00 | 8.79 |
| 21 | 1.00 | 8.24 |
| 22 | 1.00 | 8.40 |
| 23 | 1.00 | 7.90 |
| 24 | 1.00 | 7.55 |
| 25 | 1.00 | 6.69 |
| 26 | 1.00 | 6.30 |
| 27 | 1.00 | 6.10 |
| 28 | 1.00 | 5.55 |
| 29 | 1.00 | 4.96 |
| 30 | 1.00 | 4.68 |
| 31 | 1.00 | 3.69 |
| 32 | 1.00 | 2.87 |
| 33 | 1.00 | 2.60 |
| 34 | 1.00 | 1.21 |
| 35 | 1.00 | 0.96 |

### ministral8b_t02_s123_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **2**
- N joined to logit-lens sidecar: 2
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 10.49 |
| 1 | 1.00 | 9.90 |
| 2 | 1.00 | 10.25 |
| 3 | 1.00 | 10.27 |
| 4 | 1.00 | 10.32 |
| 5 | 1.00 | 10.32 |
| 6 | 1.00 | 10.34 |
| 7 | 1.00 | 10.18 |
| 8 | 1.00 | 10.23 |
| 9 | 1.00 | 10.21 |
| 10 | 1.00 | 10.37 |
| 11 | 1.00 | 10.52 |
| 12 | 1.00 | 10.52 |
| 13 | 1.00 | 10.43 |
| 14 | 1.00 | 10.41 |
| 15 | 1.00 | 10.19 |
| 16 | 1.00 | 10.14 |
| 17 | 1.00 | 10.05 |
| 18 | 1.00 | 9.74 |
| 19 | 1.00 | 9.46 |
| 20 | 1.00 | 8.91 |
| 21 | 1.00 | 8.36 |
| 22 | 1.00 | 8.62 |
| 23 | 1.00 | 8.11 |
| 24 | 1.00 | 7.88 |
| 25 | 1.00 | 7.15 |
| 26 | 1.00 | 6.66 |
| 27 | 1.00 | 6.35 |
| 28 | 1.00 | 5.54 |
| 29 | 1.00 | 4.86 |
| 30 | 1.00 | 4.73 |
| 31 | 1.00 | 3.84 |
| 32 | 1.00 | 3.28 |
| 33 | 1.00 | 2.56 |
| 34 | 1.00 | 1.47 |
| 35 | 1.00 | 1.19 |

### ministral8b_t02_s123_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### ministral8b_t02_s123_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### ministral8b_t02_s123_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **103**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## ministral8b_t02_s42_informative_v2_logitlens

**Stats:** enriched_decisions_seen=519, sidecar_records_loaded=309, joined=309, unmatched=210

### ministral8b_t02_s42_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **148**
- N joined to logit-lens sidecar: 148
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 10.46 |
| 1 | 1.00 | 9.86 |
| 2 | 1.00 | 10.17 |
| 3 | 1.00 | 10.11 |
| 4 | 1.00 | 10.23 |
| 5 | 1.00 | 10.16 |
| 6 | 1.00 | 10.29 |
| 7 | 1.00 | 10.20 |
| 8 | 1.00 | 10.16 |
| 9 | 1.00 | 10.10 |
| 10 | 1.00 | 10.33 |
| 11 | 1.00 | 10.47 |
| 12 | 1.00 | 10.49 |
| 13 | 1.00 | 10.43 |
| 14 | 1.00 | 10.46 |
| 15 | 1.00 | 10.23 |
| 16 | 1.00 | 10.19 |
| 17 | 1.00 | 10.06 |
| 18 | 1.00 | 9.79 |
| 19 | 1.00 | 9.45 |
| 20 | 1.00 | 8.91 |
| 21 | 1.00 | 8.43 |
| 22 | 1.00 | 8.57 |
| 23 | 1.00 | 8.22 |
| 24 | 1.00 | 7.85 |
| 25 | 1.00 | 7.18 |
| 26 | 1.00 | 6.71 |
| 27 | 1.00 | 6.46 |
| 28 | 1.00 | 5.92 |
| 29 | 1.00 | 5.10 |
| 30 | 1.00 | 4.89 |
| 31 | 1.00 | 3.98 |
| 32 | 1.00 | 3.32 |
| 33 | 1.00 | 2.72 |
| 34 | 1.00 | 1.46 |
| 35 | 1.00 | 1.21 |

### ministral8b_t02_s42_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **161**
- N joined to logit-lens sidecar: 161
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 10.49 |
| 1 | 1.00 | 9.93 |
| 2 | 1.00 | 10.24 |
| 3 | 1.00 | 10.20 |
| 4 | 1.00 | 10.36 |
| 5 | 1.00 | 10.27 |
| 6 | 1.00 | 10.35 |
| 7 | 1.00 | 10.22 |
| 8 | 1.00 | 10.24 |
| 9 | 1.00 | 10.20 |
| 10 | 1.00 | 10.41 |
| 11 | 1.00 | 10.48 |
| 12 | 1.00 | 10.53 |
| 13 | 1.00 | 10.46 |
| 14 | 1.00 | 10.44 |
| 15 | 1.00 | 10.10 |
| 16 | 1.00 | 10.11 |
| 17 | 1.00 | 9.91 |
| 18 | 1.00 | 9.60 |
| 19 | 1.00 | 9.18 |
| 20 | 1.00 | 8.56 |
| 21 | 1.00 | 8.11 |
| 22 | 1.00 | 8.30 |
| 23 | 1.00 | 7.95 |
| 24 | 1.00 | 7.53 |
| 25 | 1.00 | 6.77 |
| 26 | 1.00 | 6.35 |
| 27 | 1.00 | 6.14 |
| 28 | 1.00 | 5.56 |
| 29 | 1.00 | 4.99 |
| 30 | 1.00 | 4.77 |
| 31 | 1.00 | 3.78 |
| 32 | 1.00 | 3.06 |
| 33 | 1.00 | 2.59 |
| 34 | 1.00 | 1.41 |
| 35 | 1.00 | 1.19 |

### ministral8b_t02_s42_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### ministral8b_t02_s42_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### ministral8b_t02_s42_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **210**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## ministral8b_t02_s456_informative_v2_logitlens

**Stats:** enriched_decisions_seen=212, sidecar_records_loaded=108, joined=108, unmatched=104

### ministral8b_t02_s456_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **104**
- N joined to logit-lens sidecar: 104
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 10.54 |
| 1 | 1.00 | 9.97 |
| 2 | 1.00 | 10.26 |
| 3 | 1.00 | 10.21 |
| 4 | 1.00 | 10.31 |
| 5 | 1.00 | 10.28 |
| 6 | 1.00 | 10.35 |
| 7 | 1.00 | 10.15 |
| 8 | 1.00 | 10.13 |
| 9 | 1.00 | 10.09 |
| 10 | 1.00 | 10.32 |
| 11 | 1.00 | 10.45 |
| 12 | 1.00 | 10.51 |
| 13 | 1.00 | 10.45 |
| 14 | 1.00 | 10.42 |
| 15 | 1.00 | 10.11 |
| 16 | 1.00 | 10.12 |
| 17 | 1.00 | 9.99 |
| 18 | 1.00 | 9.68 |
| 19 | 1.00 | 9.27 |
| 20 | 1.00 | 8.80 |
| 21 | 1.00 | 8.25 |
| 22 | 1.00 | 8.40 |
| 23 | 1.00 | 7.92 |
| 24 | 1.00 | 7.56 |
| 25 | 1.00 | 6.71 |
| 26 | 1.00 | 6.31 |
| 27 | 1.00 | 6.10 |
| 28 | 1.00 | 5.55 |
| 29 | 1.00 | 4.95 |
| 30 | 1.00 | 4.65 |
| 31 | 1.00 | 3.67 |
| 32 | 1.00 | 2.86 |
| 33 | 1.00 | 2.57 |
| 34 | 1.00 | 1.21 |
| 35 | 1.00 | 0.96 |

### ministral8b_t02_s456_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **4**
- N joined to logit-lens sidecar: 4
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 10.48 |
| 1 | 1.00 | 9.88 |
| 2 | 1.00 | 10.22 |
| 3 | 1.00 | 10.22 |
| 4 | 1.00 | 10.32 |
| 5 | 1.00 | 10.31 |
| 6 | 1.00 | 10.34 |
| 7 | 1.00 | 10.24 |
| 8 | 1.00 | 10.23 |
| 9 | 1.00 | 10.20 |
| 10 | 1.00 | 10.38 |
| 11 | 1.00 | 10.51 |
| 12 | 1.00 | 10.51 |
| 13 | 1.00 | 10.47 |
| 14 | 1.00 | 10.46 |
| 15 | 1.00 | 10.26 |
| 16 | 1.00 | 10.24 |
| 17 | 1.00 | 10.15 |
| 18 | 1.00 | 9.85 |
| 19 | 1.00 | 9.58 |
| 20 | 1.00 | 9.08 |
| 21 | 1.00 | 8.58 |
| 22 | 1.00 | 8.74 |
| 23 | 1.00 | 8.37 |
| 24 | 1.00 | 8.01 |
| 25 | 1.00 | 7.39 |
| 26 | 1.00 | 6.78 |
| 27 | 1.00 | 6.43 |
| 28 | 1.00 | 5.72 |
| 29 | 1.00 | 4.99 |
| 30 | 1.00 | 4.84 |
| 31 | 1.00 | 4.11 |
| 32 | 1.00 | 3.43 |
| 33 | 1.00 | 2.81 |
| 34 | 1.00 | 1.56 |
| 35 | 1.00 | 1.25 |

### ministral8b_t02_s456_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### ministral8b_t02_s456_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### ministral8b_t02_s456_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **104**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## ministral8b_t0_s123_informative_v2_logitlens

**Stats:** enriched_decisions_seen=203, sidecar_records_loaded=102, joined=102, unmatched=101

### ministral8b_t0_s123_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **101**
- N joined to logit-lens sidecar: 101
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 10.53 |
| 1 | 1.00 | 9.95 |
| 2 | 1.00 | 10.26 |
| 3 | 1.00 | 10.21 |
| 4 | 1.00 | 10.31 |
| 5 | 1.00 | 10.25 |
| 6 | 1.00 | 10.34 |
| 7 | 1.00 | 10.14 |
| 8 | 1.00 | 10.12 |
| 9 | 1.00 | 10.09 |
| 10 | 1.00 | 10.30 |
| 11 | 1.00 | 10.44 |
| 12 | 1.00 | 10.49 |
| 13 | 1.00 | 10.44 |
| 14 | 1.00 | 10.41 |
| 15 | 1.00 | 10.11 |
| 16 | 1.00 | 10.11 |
| 17 | 1.00 | 9.96 |
| 18 | 1.00 | 9.67 |
| 19 | 1.00 | 9.26 |
| 20 | 1.00 | 8.75 |
| 21 | 1.00 | 8.16 |
| 22 | 1.00 | 8.34 |
| 23 | 1.00 | 7.80 |
| 24 | 1.00 | 7.48 |
| 25 | 1.00 | 6.61 |
| 26 | 1.00 | 6.20 |
| 27 | 1.00 | 6.02 |
| 28 | 1.00 | 5.43 |
| 29 | 1.00 | 4.80 |
| 30 | 1.00 | 4.53 |
| 31 | 1.00 | 3.62 |
| 32 | 1.00 | 2.82 |
| 33 | 1.00 | 2.63 |
| 34 | 1.00 | 1.31 |
| 35 | 1.00 | 1.02 |

### ministral8b_t0_s123_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **1**
- N joined to logit-lens sidecar: 1
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 10.55 |
| 1 | 1.00 | 9.96 |
| 2 | 1.00 | 10.32 |
| 3 | 1.00 | 10.40 |
| 4 | 1.00 | 10.39 |
| 5 | 1.00 | 10.34 |
| 6 | 1.00 | 10.35 |
| 7 | 1.00 | 10.25 |
| 8 | 1.00 | 10.27 |
| 9 | 1.00 | 10.22 |
| 10 | 1.00 | 10.44 |
| 11 | 1.00 | 10.55 |
| 12 | 1.00 | 10.56 |
| 13 | 1.00 | 10.48 |
| 14 | 1.00 | 10.42 |
| 15 | 1.00 | 10.30 |
| 16 | 1.00 | 10.33 |
| 17 | 1.00 | 10.26 |
| 18 | 1.00 | 10.01 |
| 19 | 1.00 | 9.77 |
| 20 | 1.00 | 9.33 |
| 21 | 1.00 | 8.68 |
| 22 | 1.00 | 9.07 |
| 23 | 1.00 | 8.67 |
| 24 | 1.00 | 8.34 |
| 25 | 1.00 | 7.70 |
| 26 | 1.00 | 7.26 |
| 27 | 1.00 | 7.04 |
| 28 | 1.00 | 6.18 |
| 29 | 1.00 | 5.69 |
| 30 | 1.00 | 5.45 |
| 31 | 1.00 | 4.71 |
| 32 | 1.00 | 3.78 |
| 33 | 1.00 | 3.09 |
| 34 | 1.00 | 1.43 |
| 35 | 1.00 | 1.11 |

### ministral8b_t0_s123_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### ministral8b_t0_s123_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### ministral8b_t0_s123_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **101**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## ministral8b_t0_s42_informative_v2_logitlens

**Stats:** enriched_decisions_seen=525, sidecar_records_loaded=313, joined=313, unmatched=212

### ministral8b_t0_s42_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **134**
- N joined to logit-lens sidecar: 134
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 10.45 |
| 1 | 1.00 | 9.85 |
| 2 | 1.00 | 10.16 |
| 3 | 1.00 | 10.10 |
| 4 | 1.00 | 10.23 |
| 5 | 1.00 | 10.14 |
| 6 | 1.00 | 10.28 |
| 7 | 1.00 | 10.20 |
| 8 | 1.00 | 10.17 |
| 9 | 1.00 | 10.12 |
| 10 | 1.00 | 10.33 |
| 11 | 1.00 | 10.47 |
| 12 | 1.00 | 10.48 |
| 13 | 1.00 | 10.41 |
| 14 | 1.00 | 10.44 |
| 15 | 1.00 | 10.22 |
| 16 | 1.00 | 10.19 |
| 17 | 1.00 | 10.07 |
| 18 | 1.00 | 9.81 |
| 19 | 1.00 | 9.47 |
| 20 | 1.00 | 8.97 |
| 21 | 1.00 | 8.48 |
| 22 | 1.00 | 8.63 |
| 23 | 1.00 | 8.31 |
| 24 | 1.00 | 7.92 |
| 25 | 1.00 | 7.24 |
| 26 | 1.00 | 6.80 |
| 27 | 1.00 | 6.54 |
| 28 | 1.00 | 6.02 |
| 29 | 1.00 | 5.19 |
| 30 | 1.00 | 4.97 |
| 31 | 1.00 | 4.07 |
| 32 | 1.00 | 3.38 |
| 33 | 1.00 | 2.79 |
| 34 | 1.00 | 1.46 |
| 35 | 1.00 | 1.20 |

### ministral8b_t0_s42_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **179**
- N joined to logit-lens sidecar: 179
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 10.50 |
| 1 | 1.00 | 9.92 |
| 2 | 1.00 | 10.24 |
| 3 | 1.00 | 10.21 |
| 4 | 1.00 | 10.36 |
| 5 | 1.00 | 10.27 |
| 6 | 1.00 | 10.35 |
| 7 | 1.00 | 10.21 |
| 8 | 1.00 | 10.23 |
| 9 | 1.00 | 10.19 |
| 10 | 1.00 | 10.41 |
| 11 | 1.00 | 10.48 |
| 12 | 1.00 | 10.53 |
| 13 | 1.00 | 10.47 |
| 14 | 1.00 | 10.45 |
| 15 | 1.00 | 10.11 |
| 16 | 1.00 | 10.13 |
| 17 | 1.00 | 9.93 |
| 18 | 1.00 | 9.61 |
| 19 | 1.00 | 9.21 |
| 20 | 1.00 | 8.58 |
| 21 | 1.00 | 8.13 |
| 22 | 1.00 | 8.32 |
| 23 | 1.00 | 7.99 |
| 24 | 1.00 | 7.57 |
| 25 | 1.00 | 6.78 |
| 26 | 1.00 | 6.35 |
| 27 | 1.00 | 6.14 |
| 28 | 1.00 | 5.59 |
| 29 | 1.00 | 5.04 |
| 30 | 1.00 | 4.84 |
| 31 | 1.00 | 3.81 |
| 32 | 1.00 | 3.08 |
| 33 | 1.00 | 2.61 |
| 34 | 1.00 | 1.40 |
| 35 | 1.00 | 1.18 |

### ministral8b_t0_s42_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### ministral8b_t0_s42_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### ministral8b_t0_s42_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **212**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

---

## ministral8b_t0_s456_informative_v2_logitlens

**Stats:** enriched_decisions_seen=209, sidecar_records_loaded=106, joined=106, unmatched=103

### ministral8b_t0_s456_informative_v2_logitlens — bucket: `clean`

- N decisions in bucket: **103**
- N joined to logit-lens sidecar: 103
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 10.53 |
| 1 | 1.00 | 9.94 |
| 2 | 1.00 | 10.26 |
| 3 | 1.00 | 10.21 |
| 4 | 1.00 | 10.31 |
| 5 | 1.00 | 10.25 |
| 6 | 1.00 | 10.34 |
| 7 | 1.00 | 10.15 |
| 8 | 1.00 | 10.13 |
| 9 | 1.00 | 10.10 |
| 10 | 1.00 | 10.31 |
| 11 | 1.00 | 10.44 |
| 12 | 1.00 | 10.49 |
| 13 | 1.00 | 10.43 |
| 14 | 1.00 | 10.42 |
| 15 | 1.00 | 10.12 |
| 16 | 1.00 | 10.12 |
| 17 | 1.00 | 9.98 |
| 18 | 1.00 | 9.68 |
| 19 | 1.00 | 9.27 |
| 20 | 1.00 | 8.77 |
| 21 | 1.00 | 8.18 |
| 22 | 1.00 | 8.37 |
| 23 | 1.00 | 7.84 |
| 24 | 1.00 | 7.53 |
| 25 | 1.00 | 6.66 |
| 26 | 1.00 | 6.24 |
| 27 | 1.00 | 6.06 |
| 28 | 1.00 | 5.47 |
| 29 | 1.00 | 4.84 |
| 30 | 1.00 | 4.57 |
| 31 | 1.00 | 3.67 |
| 32 | 1.00 | 2.88 |
| 33 | 1.00 | 2.64 |
| 34 | 1.00 | 1.32 |
| 35 | 1.00 | 1.03 |

### ministral8b_t0_s456_informative_v2_logitlens — bucket: `illegal_fold`

- N decisions in bucket: **3**
- N joined to logit-lens sidecar: 3
- Num layers: 36
- Crystallization layer: n/a (no records joined)

| layer | OTHER | entropy |
| --- | --- | --- |
| 0 | 1.00 | 10.48 |
| 1 | 1.00 | 9.77 |
| 2 | 1.00 | 10.24 |
| 3 | 1.00 | 10.31 |
| 4 | 1.00 | 10.35 |
| 5 | 1.00 | 10.28 |
| 6 | 1.00 | 10.31 |
| 7 | 1.00 | 10.17 |
| 8 | 1.00 | 10.18 |
| 9 | 1.00 | 10.12 |
| 10 | 1.00 | 10.36 |
| 11 | 1.00 | 10.48 |
| 12 | 1.00 | 10.51 |
| 13 | 1.00 | 10.46 |
| 14 | 1.00 | 10.42 |
| 15 | 1.00 | 10.16 |
| 16 | 1.00 | 10.16 |
| 17 | 1.00 | 10.06 |
| 18 | 1.00 | 9.73 |
| 19 | 1.00 | 9.38 |
| 20 | 1.00 | 8.85 |
| 21 | 1.00 | 8.40 |
| 22 | 1.00 | 8.62 |
| 23 | 1.00 | 8.30 |
| 24 | 1.00 | 8.02 |
| 25 | 1.00 | 7.37 |
| 26 | 1.00 | 6.80 |
| 27 | 1.00 | 6.54 |
| 28 | 1.00 | 5.74 |
| 29 | 1.00 | 5.09 |
| 30 | 1.00 | 4.89 |
| 31 | 1.00 | 4.01 |
| 32 | 1.00 | 3.32 |
| 33 | 1.00 | 2.69 |
| 34 | 1.00 | 1.47 |
| 35 | 1.00 | 1.21 |

### ministral8b_t0_s456_informative_v2_logitlens — bucket: `illegal_other`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### ministral8b_t0_s456_informative_v2_logitlens — bucket: `alias_unrecognized`

- N decisions in bucket: **0**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

### ministral8b_t0_s456_informative_v2_logitlens — bucket: `json_failure`

- N decisions in bucket: **103**
- N joined to logit-lens sidecar: 0
- Num layers: 0
- Crystallization layer: n/a (no records joined)

_No logit-lens records joined — nothing to report._

