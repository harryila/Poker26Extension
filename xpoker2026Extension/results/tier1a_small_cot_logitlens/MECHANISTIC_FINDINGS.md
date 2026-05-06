# Mechanistic findings — Tier 1A.small CoT + logit-lens (18 cells)

**Run:** 18 cells (3 models × 3 seeds × 2 temps × 100 hands), `--cot --capture-logprobs --logit-lens`. Wall-clock ~26 h on Blackwell.
**Aggregator:** `analysis/analyze_logit_lens_by_failure_mode.py` joins each per-decision logit-lens sidecar to the enriched log on `(hand_id, decision_idx)`, anchors the action-verb token via the JSON `' "..."}' ` payload (see `_find_action_position`), and reports per-bucket × per-layer mapped action group at that token.

---

## 1. The headline (cross-cell)

For each cell, the mean **crystallization layer** (the earliest layer from which the action-group prediction at the action-verb position is stable through to the final layer):

| Cell | n clean | n illegal_fold | crys clean | crys illegal_fold | Δ (illegal − clean) | layers |
|---|---:|---:|---:|---:|---:|---:|
| `llama8b_t0_s42`        | 334 |  16 | 24.4 | 20.4 | **−4.0** | 32 |
| `llama8b_t0_s123`       | 317 |  34 | 23.6 | 20.9 | **−2.7** | 32 |
| `llama8b_t0_s456`       | 298 |  18 | 23.4 | 20.9 | **−2.5** | 32 |
| `llama8b_t02_s42`       | 329 |  26 | 24.1 | 20.4 | **−3.7** | 32 |
| `llama8b_t02_s123`      | 289 |  28 | 23.8 | 21.0 | **−2.8** | 32 |
| `llama8b_t02_s456`      | 290 |  21 | 23.6 | 20.6 | **−2.9** | 32 |
| `ministral8b_t0_s42`    | 134 | 179 | 24.9 | 22.7 | **−2.3** | 36 |
| `ministral8b_t0_s123`   | 101 |   1 | 24.0 | 22.0 | −2.0 | 36 |
| `ministral8b_t0_s456`   | 103 |   3 | 24.1 | 22.3 | −1.8 | 36 |
| `ministral8b_t02_s42`   | 148 | 161 | 25.4 | 22.7 | **−2.7** | 36 |
| `ministral8b_t02_s123`  | 104 |   2 | 24.2 | 23.0 | −1.2 | 36 |
| `ministral8b_t02_s456`  | 104 |   4 | 24.3 | 23.0 | −1.3 | 36 |
| `qwen8b_t0_s42`         | 449 |   4 | 30.4 | 30.8 | +0.3 | 36 |
| `qwen8b_t0_s123`        | 340 |   9 | 30.1 | 30.3 | +0.3 | 36 |
| `qwen8b_t0_s456`        | 345 |  11 | 29.9 | 30.4 | +0.5 | 36 |
| `qwen8b_t02_s42`        | 455 |   4 | 30.4 | 29.8 | −0.6 | 36 |
| `qwen8b_t02_s123`       | 385 |   8 | 30.0 | 30.5 | +0.5 | 36 |
| `qwen8b_t02_s456`       | 323 |   6 | 30.0 | 30.3 | +0.3 | 36 |

### Per-model summary

| Model | Δ range across 6 cells | Direction | What it means |
|---|---|---|---|
| **Llama 8B**     | −2.5 to **−4.0** | illegal-FOLD crystallizes EARLIER | strong, consistent effect |
| **Ministral 8B** | −1.2 to **−2.7** | illegal-FOLD crystallizes EARLIER | consistent, smaller magnitude |
| **Qwen 8B**      | −0.6 to +0.5     | no significant difference | noisy at small n; basically null |

For Llama and Ministral, **illegal-FOLD decisions become FOLD-committed in the residual stream 2–4 layers earlier than clean decisions.** Qwen shows no such effect.

---

## 2. The deeper story — split clean by emitted action

The "clean" bucket pools three different emitted actions (legal FOLD, CHECK_OR_CALL, BET_OR_RAISE). Splitting it reveals what's actually happening.

### Ministral 8B, t=0, s=42 (the cell with the most illegal FOLDs: n=179)

| Bucket | n | Crystallization layer (action-group axis) |
|---|---:|---:|
| `clean_CHECK_OR_CALL` |  29 | **28.8** |
| `clean_BET_OR_RAISE`  |   7 | 25.7 |
| `clean_LEGAL_FOLD`    |  98 | 23.7 |
| `illegal_FOLD`        | 179 | **22.7** |

### Per-layer action mix at the action-verb position (final 16 layers)

```
  L | clean_CHECK_OR_CALL          | clean_LEGAL_FOLD             | illegal_FOLD
----+------------------------------+------------------------------+------------------------------
 20 | OTHER 1.00                   | OTHER 1.00                   | OTHER 1.00
 21 | OTHER 1.00                   | OTHER 1.00                   | OTHER 1.00
 22 | OTHER 1.00                   | FOLD 0.02  OTHER 0.98        | FOLD 0.33  OTHER 0.67
 23 | OTHER 1.00                   | FOLD 0.26  OTHER 0.74        | FOLD 1.00
 24 | OTHER 1.00                   | FOLD 1.00                    | FOLD 1.00
 25 | FOLD  0.86  OTHER 0.14       | FOLD 1.00                    | FOLD 1.00
 26 | FOLD  0.97  OTHER 0.00       | FOLD 1.00                    | FOLD 1.00
 27 | FOLD  0.76  CHECK 0.21       | FOLD 1.00                    | FOLD 1.00
 28 | FOLD  0.48  CHECK 0.48       | FOLD 1.00                    | FOLD 1.00
 29 | FOLD  0.00  CHECK 0.90       | FOLD 1.00                    | FOLD 1.00
 30 | FOLD  0.00  CHECK 0.90       | FOLD 1.00                    | FOLD 1.00
 31 | FOLD  0.00  CHECK 0.90       | FOLD 1.00                    | FOLD 1.00
 ...
 35 | FOLD  0.00  CHECK 1.00       | FOLD 1.00                    | FOLD 1.00
```

### What this shows

1. **All decisions start neutral** at the action-verb position (layers 0–21 = 100% OTHER, residual stream not yet vocab-aligned).
2. **By layer 22–23, a FOLD-leaning signal emerges** in the residual stream for *every* bucket — including decisions that will eventually emit CHECK_OR_CALL.
3. **Legal-FOLD decisions** lock in at FOLD by layer 24 and never revise.
4. **Illegal-FOLD decisions** lock in **one layer earlier** (layer 23) and are even more confidently FOLD (100% at layer 23 vs 26% for legal FOLDs at the same layer). The model is *more certain* of the wrong FOLD than of correct FOLDs.
5. **CHECK_OR_CALL decisions** are the only bucket that **late-layer-revises** — they start FOLD (86% at layer 25), then the CHECK signal climbs from 0% → 21% → 48% → 90% across layers 27–29, fully overtaking by layer 35.

---

## 3. The mechanistic claim

Small-model (8B-class) CoT in this poker setting exhibits a **baseline FOLD pull** in the mid-to-late residual stream (~layer 22+ for Llama and Ministral). What distinguishes the three observed action outcomes is *whether* the late-layer deliberation circuit overrides that pull:

- **CHECK_OR_CALL** → late-layer revision succeeds; CHECK signal overtakes FOLD in the final ~6–8 layers.
- **legal FOLD** → late-layer revision doesn't fire (or doesn't need to); FOLD locks in early and the action is contextually legal.
- **illegal FOLD** → late-layer revision fails to fire even harder than for legal FOLDs (FOLD locks in *earlier* and *more confidently*); the action is contextually illegal but the env's `_fallback_action` rescues it to CHECK_OR_CALL.

This **is the opposite of the verbalization-failure hypothesis** that motivated the run. We're not seeing "early/mid layers say CHECK, last layer flips to FOLD". We're seeing "early-mid layers commit to FOLD, and CHECK only emerges via late-layer revision in some decisions". The illegal-FOLD pathology is therefore a **failure of late-layer deliberation**, not a verbalization-stage glitch.

This is consistent with — and a mechanistic explanation for — the §12c full-grid pot-odds finding that "CoT's apparent EV improvement on Ministral is a rescue artifact": when the deliberation circuit doesn't fire, the model commits to FOLD; the env safety net catches those FOLDs in free-check spots and converts them to the EV-optimal CHECK_OR_CALL.

---

## 4. Why Qwen is different

Qwen 8B's illegal-FOLD count is 4–11 per cell (vs Llama's 16–34 and Ministral's 1–179). At those sample sizes the crystallization comparison is noisy; the observed Δ ≈ 0 is consistent with both "Qwen's deliberation circuit is more uniform" and "we don't have enough illegal FOLDs to detect the effect". Per the full-grid pot-odds (§12c), Qwen is the model that benefits *least* from CoT — slightly worse aggregate EV under CoT than direct prompting — so a more uniform layer structure (less reliance on a brittle late-layer override) is consistent with that behavioral profile too.

---

## 5. Caveats & open follow-ups

- **Token anchor heuristic:** the action-verb position is found by walking back from the JSON `'"}'` close brace to the value-opening `' "'` quote. Validated by inspection on Llama / Qwen / Ministral; subword splits like `'F'+'OLD'` or `'B'+'ET'+'_OR'+'_RA'+'ISE'` are handled correctly.
- **`json_failure` bucket has 0 joined records** — by construction, no parseable action JSON means no anchor point. Correct behavior, not a bug.
- **`alias_unrecognized` bucket has 0 records** — confirms the §11 alias fix landed cleanly; no model emits unrecognized verb forms in this run.
- **Hidden-state sidecars (`_hiddens.jsonl`) are present** but not consumed by this analysis. They could enable layer-level intervention experiments (e.g. patch layer-22 hidden state from a CHECK trajectory into a FOLD trajectory and see if the model emits CHECK).
- **The "late-layer revision" finding could be tested causally** with a small attention-pattern study at layers 25–29 of Ministral on s42 — what attends to what to flip the residual from FOLD-leaning to CHECK-leaning?

---

## 6. Files

- `BY_FAILURE_MODE.md` — full per-cell × per-bucket × per-layer action-mix tables (18 cells, 5 buckets, 32–36 layers).
- `by_failure_mode_<cell>.json` — JSON form of the same per-cell analysis.
- `logitlens_<cell>.json` — descriptive (per-layer entropy, raw-token-axis crystallization).
- `entropy_<cell>.png` — per-layer entropy curve plot.
- Raw inputs (gitignored): `logs/cot_<cell>_logitlens_logit_lens.jsonl.gz` (sidecars), `logs/cot_<cell>_logitlens_enriched.jsonl.gz` (enriched logs), `logs/cot_<cell>_logitlens_hiddens.jsonl` (hidden states).
