# Tier 4 opponent-preset duplication diagnostic

Cross-preset overlap of recorded `prompt_hash` sets (decision records).
**`identical=YES` means two presets are the same data labeled twice** — 
the opponent's action sequence collapsed because it shares the `seed=43` 
RNG stream and the presets' policies are too similar to diverge. See 
`AUDIT_FINDINGS.md` §12.

## llama-8b

- distinct prompt_hashes per preset: default=282, informative_v2=468, tight_aggressive=370, loose_aggressive=370, loose_passive=199

| preset A | preset B | |A| | |B| | A∩B | A∪B | Jaccard | identical? |
|---|---|---:|---:|---:|---:|---:|:--:|
| default | informative_v2 | 282 | 468 | 125 | 625 | 0.20 | no |
| default | tight_aggressive | 282 | 370 | 162 | 490 | 0.33 | no |
| default | loose_aggressive | 282 | 370 | 162 | 490 | 0.33 | no |
| default | loose_passive | 282 | 199 | 85 | 396 | 0.21 | no |
| informative_v2 | tight_aggressive | 468 | 370 | 280 | 558 | 0.50 | no |
| informative_v2 | loose_aggressive | 468 | 370 | 280 | 558 | 0.50 | no |
| informative_v2 | loose_passive | 468 | 199 | 49 | 618 | 0.08 | no |
| tight_aggressive | loose_aggressive | 370 | 370 | 370 | 370 | 1.00 | **YES** |
| tight_aggressive | loose_passive | 370 | 199 | 49 | 520 | 0.09 | no |
| loose_aggressive | loose_passive | 370 | 199 | 49 | 520 | 0.09 | no |

- **Collapsed groups: tight_aggressive≡loose_aggressive ⇒ 4 distinct distribution(s), NOT 5.** Report each collapsed group as a single cell.

## qwen-8b

- distinct prompt_hashes per preset: default=153, informative_v2=142, tight_aggressive=142, loose_aggressive=142, loose_passive=199

| preset A | preset B | |A| | |B| | A∩B | A∪B | Jaccard | identical? |
|---|---|---:|---:|---:|---:|---:|:--:|
| default | informative_v2 | 153 | 142 | 135 | 160 | 0.84 | no |
| default | tight_aggressive | 153 | 142 | 135 | 160 | 0.84 | no |
| default | loose_aggressive | 153 | 142 | 135 | 160 | 0.84 | no |
| default | loose_passive | 153 | 199 | 108 | 244 | 0.44 | no |
| informative_v2 | tight_aggressive | 142 | 142 | 142 | 142 | 1.00 | **YES** |
| informative_v2 | loose_aggressive | 142 | 142 | 142 | 142 | 1.00 | **YES** |
| informative_v2 | loose_passive | 142 | 199 | 92 | 249 | 0.37 | no |
| tight_aggressive | loose_aggressive | 142 | 142 | 142 | 142 | 1.00 | **YES** |
| tight_aggressive | loose_passive | 142 | 199 | 92 | 249 | 0.37 | no |
| loose_aggressive | loose_passive | 142 | 199 | 92 | 249 | 0.37 | no |

- **Collapsed groups: informative_v2≡tight_aggressive≡loose_aggressive ⇒ 3 distinct distribution(s), NOT 5.** Report each collapsed group as a single cell.

## ministral-8b

- distinct prompt_hashes per preset: default=134, informative_v2=107, tight_aggressive=93, loose_aggressive=107, loose_passive=199

| preset A | preset B | |A| | |B| | A∩B | A∪B | Jaccard | identical? |
|---|---|---:|---:|---:|---:|---:|:--:|
| default | informative_v2 | 134 | 107 | 97 | 144 | 0.67 | no |
| default | tight_aggressive | 134 | 93 | 84 | 143 | 0.59 | no |
| default | loose_aggressive | 134 | 107 | 97 | 144 | 0.67 | no |
| default | loose_passive | 134 | 199 | 90 | 243 | 0.37 | no |
| informative_v2 | tight_aggressive | 107 | 93 | 93 | 107 | 0.87 | no |
| informative_v2 | loose_aggressive | 107 | 107 | 107 | 107 | 1.00 | **YES** |
| informative_v2 | loose_passive | 107 | 199 | 58 | 248 | 0.23 | no |
| tight_aggressive | loose_aggressive | 93 | 107 | 93 | 107 | 0.87 | no |
| tight_aggressive | loose_passive | 93 | 199 | 51 | 241 | 0.21 | no |
| loose_aggressive | loose_passive | 107 | 199 | 58 | 248 | 0.23 | no |

- **Collapsed groups: informative_v2≡loose_aggressive ⇒ 4 distinct distribution(s), NOT 5.** Report each collapsed group as a single cell.

