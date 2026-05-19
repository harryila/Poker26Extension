# Mode-balanced direction probe (CoT vs non-CoT, hand-matched)

- Model: `Qwen/Qwen3-8B`
- Layer: **23**
- CoT log:    `logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
- non-CoT log: `logs/scaled_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz`
- Matched (hand_id × decision_idx) pairs attempted: 110
- Captured: CoT 110, non-CoT 110
- Hidden dim: 4096

## Per-mode probe accuracy (5-fold CV)

- CoT: **1.000 ± 0.000**
- non-CoT: **1.000 ± 0.000**

## Cosines on matched data

- **cos(w_CoT, w_nonCoT) = +0.5069**
- cos(centroid_CoT, centroid_nonCoT) = +0.5983

## Cross-projection (mean CHECK − FOLD when projecting onto the OTHER mode's direction)

- non-CoT residuals projected onto w_CoT: **+43.952** ✅
- CoT residuals projected onto w_nonCoT: **+37.848** ✅

## Reading guide

Compare to the unmatched cosines from `direction_cosine_compare/` (Phase M §18b: Llama 0.27, Qwen 0.34). If the matched cosine is **substantially higher** (≥0.6), the §18b non-identity was primarily a data-distribution-shift artifact — directions are much closer than they appeared. If matched cosine is **similar** (0.2–0.4), the direction tilt is real even with hand population controlled — the model represents the verb decision along mode-specific axes regardless of which hands you sample.
