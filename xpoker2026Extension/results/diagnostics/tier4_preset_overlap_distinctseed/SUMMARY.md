# Tier 4 opponent-preset duplication diagnostic

Cross-preset overlap of recorded `prompt_hash` sets (decision records).
**`identical=YES` means two presets are the same data labeled twice** — 
the opponent's action sequence collapsed because it shares the `seed=43` 
RNG stream and the presets' policies are too similar to diverge. See 
`AUDIT_FINDINGS.md` §12.

## llama-8b

- distinct prompt_hashes per preset: default=247, informative_v2=449, tight_aggressive=322, loose_aggressive=168, loose_passive=98

| preset A | preset B | |A| | |B| | A∩B | A∪B | Jaccard | identical? |
|---|---|---:|---:|---:|---:|---:|:--:|
| default | informative_v2 | 247 | 449 | 69 | 627 | 0.11 | no |
| default | tight_aggressive | 247 | 322 | 58 | 511 | 0.11 | no |
| default | loose_aggressive | 247 | 168 | 6 | 409 | 0.01 | no |
| default | loose_passive | 247 | 98 | 0 | 345 | 0.00 | no |
| informative_v2 | tight_aggressive | 449 | 322 | 266 | 505 | 0.53 | no |
| informative_v2 | loose_aggressive | 449 | 168 | 21 | 596 | 0.04 | no |
| informative_v2 | loose_passive | 449 | 98 | 0 | 547 | 0.00 | no |
| tight_aggressive | loose_aggressive | 322 | 168 | 49 | 441 | 0.11 | no |
| tight_aggressive | loose_passive | 322 | 98 | 20 | 400 | 0.05 | no |
| loose_aggressive | loose_passive | 168 | 98 | 89 | 177 | 0.50 | no |

- No fully-collapsed pairs; 5 distinct distributions.

## qwen-8b

- distinct prompt_hashes per preset: default=251, informative_v2=168, tight_aggressive=163, loose_aggressive=143, loose_passive=142

| preset A | preset B | |A| | |B| | A∩B | A∪B | Jaccard | identical? |
|---|---|---:|---:|---:|---:|---:|:--:|
| default | informative_v2 | 251 | 168 | 90 | 329 | 0.27 | no |
| default | tight_aggressive | 251 | 163 | 72 | 342 | 0.21 | no |
| default | loose_aggressive | 251 | 143 | 8 | 386 | 0.02 | no |
| default | loose_passive | 251 | 142 | 0 | 393 | 0.00 | no |
| informative_v2 | tight_aggressive | 168 | 163 | 134 | 197 | 0.68 | no |
| informative_v2 | loose_aggressive | 168 | 143 | 13 | 298 | 0.04 | no |
| informative_v2 | loose_passive | 168 | 142 | 0 | 310 | 0.00 | no |
| tight_aggressive | loose_aggressive | 163 | 143 | 42 | 264 | 0.16 | no |
| tight_aggressive | loose_passive | 163 | 142 | 29 | 276 | 0.11 | no |
| loose_aggressive | loose_passive | 143 | 142 | 130 | 155 | 0.84 | no |

- No fully-collapsed pairs; 5 distinct distributions.

## ministral-8b

- distinct prompt_hashes per preset: default=49, informative_v2=49, tight_aggressive=61, loose_aggressive=102, loose_passive=105

| preset A | preset B | |A| | |B| | A∩B | A∪B | Jaccard | identical? |
|---|---|---:|---:|---:|---:|---:|:--:|
| default | informative_v2 | 49 | 49 | 49 | 49 | 1.00 | **YES** |
| default | tight_aggressive | 49 | 61 | 40 | 70 | 0.57 | no |
| default | loose_aggressive | 49 | 102 | 4 | 147 | 0.03 | no |
| default | loose_passive | 49 | 105 | 0 | 154 | 0.00 | no |
| informative_v2 | tight_aggressive | 49 | 61 | 40 | 70 | 0.57 | no |
| informative_v2 | loose_aggressive | 49 | 102 | 4 | 147 | 0.03 | no |
| informative_v2 | loose_passive | 49 | 105 | 0 | 154 | 0.00 | no |
| tight_aggressive | loose_aggressive | 61 | 102 | 24 | 139 | 0.17 | no |
| tight_aggressive | loose_passive | 61 | 105 | 21 | 145 | 0.14 | no |
| loose_aggressive | loose_passive | 102 | 105 | 96 | 111 | 0.86 | no |

- **Collapsed groups: default≡informative_v2 ⇒ 4 distinct distribution(s), NOT 5.** Report each collapsed group as a single cell.

