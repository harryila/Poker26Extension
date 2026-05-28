# SUPERSEDED — do not cite

This cell was confounded by HFAgent re-inference fidelity. **Do not include
its numbers in the writeup.**

## What's wrong

`baseline_rows.jsonl` shows that of the 80 records with
`recorded_bucket == "illegal_fold"`, only **4** produce a parseable JSON
under HFAgent re-inference at T=0 (5%). The remaining 76 fall back to
HFAgent's default action (`CHECK_OR_CALL`) while `_reparse_one(raw_response)`
still classifies the broken raw text as `illegal_fold` because it finds
"FOLD" anywhere in the truncated output. That makes
`replay_bucket="illegal_fold"` essentially a measure of whether HFAgent's
fallback fired, **not** whether the model emitted a FOLD verb.

The headline figures in the legacy SUMMARY:

```
baseline:  38.0% illegal_FOLD
triplet:   42.5% illegal_FOLD   (+4.5 pp)
control:   34.5% illegal_FOLD   (-3.5 pp)
```

are therefore **not interpretable as head-necessity evidence** — they
mostly reflect HFAgent regen-fidelity drift between the recorded log and
re-inference (likely chat-template, dtype, or history-truncation drift; we
did not isolate the root cause).

## Replacement

The correct cell uses the recon pipeline (PromptReconstructor + raw
`model.generate(do_sample=False)`, identical to
`continuation_after_patch.regenerate_ablated`) and filters to recorded
illegal_fold targets:

```
results/inference_head_ablation/ministral8b_l16_recon_illegal_fold/  (after rerun)
```

Produced by:

```bash
MODEL=ministral PIPELINE=recon FILTER_RECORDED_BUCKET=illegal_fold \
  N_DECISIONS=80 bash scripts/run_inference_head_ablation.sh
```

Cross-check: this should agree with the
`regenerate_ablated` flip rate in
`results/continuation_after_patch/ministral8b_l16/SUMMARY.md` (≈4/25 ⇒ 16%
on the same target pool).

## Why we keep the directory

The legacy data documents the regen-fidelity bug; reviewers may ask why we
swapped pipelines. The `*_rows.jsonl` files are the evidence.

See `AUDIT_FINDINGS.md` §2 for full discussion.
