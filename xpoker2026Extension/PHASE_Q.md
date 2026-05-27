# Phase Q — seven follow-up experiments

Implements the pre-submission GPU suite discussed in the planning thread.

## Quick start (GPU box)

```bash
cd xpoker2026Extension
bash scripts/run_phase_q_all.sh
```

**After first run (disk fix + code fixes):** re-run only broken/missing cells:

```bash
export HF_HOME=/workspace/huggingface HF_TOKEN=...
FORCE_RERUN=1 CONTINUE_TOKENS=180 bash scripts/run_phase_q_gpu_followup.sh
```

This runs reverse + BET patching (Llama/Qwen if missing), then **re-runs**
continuation + context-stratified for all `MODELS` (default: ministral llama qwen).
Skips inference ablation and tier4 (already valid or redundant).

Or run individually:

| # | Experiment | Script |
|---|------------|--------|
| 1 | Inference head ablation × behavior | `bash scripts/run_inference_head_ablation.sh` |
| 2 | Continuation after patch | `bash scripts/run_continuation_after_patch.sh` |
| 3 | Reverse FOLD→CHECK patching | `bash scripts/run_causal_patching_reverse_full.sh` |
| 4 | BET→illegal_FOLD patching | `bash scripts/run_causal_patching_bet_to_illegal_fold.sh` |
| 5 | Tier 4 preset patching | `bash scripts/run_tier4_patching.sh` |
| 6 | Mode-balanced probe (+ seed verify) | `bash scripts/run_seed_matched_nocot_verify.sh` |
| 7 | Context-stratified patching | `bash scripts/run_context_stratified_patching.sh` |

## New Python modules

- `poker_env/interp/generation_ablation.py` — `AttnHeadZeroAblation` hook for `generate()`
- `experiments/inference_head_ablation.py` — behavioral illegal_FOLD rate under ablation
- `experiments/continuation_after_patch.py` — qualitative full-response comparison
- `experiments/context_stratified_patching.py` — stratified patching (`--stratify-by street` default)

## HFAgent change

`_generate()` accepts optional `agent._extra_generation_hooks` (list of objects with
`attach()` / `detach()`), used by inference ablation.

## Notes

- **Item 6** does not require 10h re-inference: non-CoT `scaled_*` logs already use
  the same seeds as CoT; pairing is `(seed, decision_idx)` (§22j-bis).
- **Llama Tier 4** is skipped by default (`RUN_LLAMA_TIER4=0`) due to chat-template
  `Today Date:` drift (§22m.2).
- **Reverse / verb matrix** from Phase L remain valid; scripts here add named
  `*_reverse_fold_to_check_*` and `*_bet_to_illegal_fold_*` dirs for citation.
- **Inference ablation** defaults to `POOLED=1` (3 seeds). Ablation zeros heads at
  the last position on **every** `generate()` step (full CoT+JSON), not verb-only.
- **Continuation** loops all `n_source` residuals (`source_idx` in JSONL).
  `CONTINUE_TOKENS=80` default; use `180` for paper figure.
- **BET→illegal_FOLD**: run `analyze_patching_top1_groups` after patching for
  `patched_top1_group` → BET_RAISE rates (`SUMMARY_bet_to_illegal_fold_top1.md`).
- **Context-stratified**: default `STRATIFY_BY=street` (pot-odds quartiles collapse
  when most CHECK spots have `bet_to_call=0`). Use `pot_odds_quartile` only on
  facing-bet subsample.
- **Continuation**: patch continuation is scored on **assistant response suffix only**
  (not full chat decode); prompt-leak strings → `broken_json`.
- **Inference ablation (exp 1)**: negative result is real — heads are **redundant**,
  not necessary; do not re-run unless you change head set.
