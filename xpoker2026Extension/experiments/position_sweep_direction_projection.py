"""
Position-sweep decision-direction projection (D1).

Given a learned verb-direction at L* (from `decision_direction_probe.py`),
capture residuals at L* at MULTIPLE POSITIONS throughout the FULL input
(prompt + complete recorded raw_response) and project each onto the
direction. Output: a position-vs-projection trajectory per decision,
aggregated by bucket.

Question answered:
    "Where in the model's RESPONSE does the verb decision become
     committed in residual space at L*?"

Compute-then-commit predicts: projection should be near zero (undecided)
during prompt processing, then sharply diverge near the verb-emission
position, then stay committed through the rest of the response.

Procedure:
  1. Load the saved direction probe (`raw_residuals.npz`) for a
     (model, L*) cell. The weight vector w defines the verb direction
     (positive = CHECK, negative = FOLD).
  2. For each sampled decision, build the FULL input string =
     prompt + raw_response (NOT truncated at the verb position).
  3. Run a single forward pass with hooks at L* capturing the residual
     at EVERY position.
  4. Project each position's residual onto w. Record (position, projection,
     token).
  5. Align positions to the verb-emission offset (so position 0 = the
     last input position used by the patching experiments).
  6. Aggregate per bucket: mean ± std projection at each relative
     position; output a CSV + a SUMMARY.md with key trajectory points.

This is much cheaper than position-flexible patching:
    — One forward per decision (no patching).
    — One hook at L* (not all layers).
    — All positions captured in the same forward.
Total: ~100 decisions × 50 ms each = ~5 seconds per bucket on H100, plus
model load. Roughly 5-10 minutes per model.

Usage::

    python -m experiments.position_sweep_direction_projection \\
        --enriched-log logs/cot_llama8b_t0_s42_*.jsonl.gz \\
                       logs/cot_llama8b_t0_s123_*.jsonl.gz \\
                       logs/cot_llama8b_t0_s456_*.jsonl.gz \\
        --layer 14 \\
        --probe-npz results/direction_probe/llama8b_l14/raw_residuals.npz \\
        --max-decisions-per-bucket 50 \\
        --positions-relative -200 -100 -50 -20 -10 -5 -2 -1 0 1 2 5 10 20 50 \\
        --out-dir results/position_sweep/llama8b_l14 \\
        --device cuda --dtype bfloat16

Outputs:
  - SUMMARY.md: per-bucket mean projection at each relative position.
  - by_decision.csv: one row per (decision, position) with raw projection.
  - summary.json: structured aggregate.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.recategorize_action_metadata import _reparse_one  # noqa: E402
from poker_env.interp.forward_helpers import (  # noqa: E402
    PromptReconstructor,
    find_action_verb_response_offset,
)
from poker_env.interp.patching import HiddenStateCapture  # noqa: E402
from experiments.causal_patching import (  # noqa: E402
    BUCKET_NAMES,
    classify_decision,
    _iter_decisions,
    _load_agent_config,
)


class PerLayerAllPosCapture:
    """Captures the FULL [seq, hidden] residual at one layer (the only
    layer we need for projection). Replaces HiddenStateCapture which only
    captures the last position.
    """

    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        from poker_env.interp.patching import _get_layers, _layer_output_residual
        self._layers = _get_layers(model)
        self._layer_output_residual = _layer_output_residual
        self._hook = None
        self._captured = None

    def attach(self):
        layer = self._layers[self.layer_idx]
        self._captured = None

        def hook(module, input, output):
            res = self._layer_output_residual(output)
            # res: [batch, seq, hidden]; keep all positions of batch 0.
            self._captured = res[0].detach().to("cpu")

        self._hook = layer.register_forward_hook(hook)

    def detach(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *exc):
        self.detach()


def main():
    parser = argparse.ArgumentParser(
        description="Position-sweep direction projection (D1)."
    )
    parser.add_argument("--enriched-log", required=True, nargs="+")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--probe-npz", required=True,
                        help="Path to a `raw_residuals.npz` produced by "
                             "`decision_direction_probe.py` — used to read "
                             "the weight vector w.")
    parser.add_argument("--max-decisions-per-bucket", type=int, default=50)
    parser.add_argument(
        "--positions-relative",
        type=int, nargs="+",
        default=[-300, -200, -100, -50, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50, 100],
        help="Positions to report, RELATIVE to the verb-emission position. "
             "0 = the position whose lm_head output IS the verb prediction. "
             "Negative = earlier in the response (or in the prompt for very "
             "negative offsets).",
    )
    parser.add_argument("--buckets", nargs="+",
                        default=["clean_check_or_call",
                                 "clean_legal_fold",
                                 "illegal_fold"],
                        choices=BUCKET_NAMES)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enriched_logs = list(args.enriched_log)

    # Load direction.
    import numpy as np
    print(f"[init] loading direction probe weights from {args.probe_npz} ...")
    npz = np.load(args.probe_npz)
    w = npz["weight_vec"].astype(np.float32)
    w_norm = float(np.linalg.norm(w))
    print(f"[init] direction loaded: shape={w.shape}, ||w||={w_norm:.4f}")

    agent_config = _load_agent_config(enriched_logs[0])
    model_id = args.model_id or agent_config["model_id"]
    if len(enriched_logs) > 1:
        for extra in enriched_logs[1:]:
            other_cfg = _load_agent_config(extra)
            if other_cfg.get("model_id") != model_id:
                print(f"[abort] model mismatch", file=sys.stderr)
                sys.exit(2)
    print(f"[init] model_id={model_id}, layer={args.layer}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[init] loading model on {args.device} ({args.dtype}) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=args.device,
    )
    model.eval()
    print(f"[init] model loaded in {time.time() - t0:.1f}s")

    if w.shape[0] != model.config.hidden_size:
        print(f"[abort] direction dim {w.shape[0]} != model hidden_size "
              f"{model.config.hidden_size}", file=sys.stderr)
        sys.exit(2)

    recon = PromptReconstructor(tokenizer, agent_config)

    # Bucket scan.
    by_bucket: dict[str, list[dict]] = {b: [] for b in BUCKET_NAMES}
    for log_path in enriched_logs:
        for rec in _iter_decisions(log_path):
            am = rec.get("action_metadata")
            if am is None or not am.get("raw_response"):
                continue
            rec.setdefault("_source_log", log_path)
            by_bucket[classify_decision(rec)].append(rec)
    for b in args.buckets:
        print(f"  {b:<22}: {len(by_bucket[b])}")

    # Capture per decision.
    by_pair_path = out_dir / "by_decision.csv"
    by_pair_f = open(by_pair_path, "w", newline="")
    writer = csv.writer(by_pair_f)
    writer.writerow([
        "bucket", "hand_id", "decision_idx",
        "verb_abs_pos", "rel_pos", "abs_pos",
        "projection", "token",
    ])

    # Aggregate per (bucket, rel_pos): list of projections.
    agg: dict[tuple[str, int], list[float]] = {}
    for b in args.buckets:
        for r in args.positions_relative:
            agg[(b, r)] = []

    print(f"\n[capture] up to {args.max_decisions_per_bucket} decisions per bucket "
          f"at L={args.layer}, projecting onto direction ...")

    cap = PerLayerAllPosCapture(model, args.layer)

    for bucket in args.buckets:
        recs = by_bucket[bucket]
        if not recs:
            print(f"  [{bucket}] empty — skipping")
            continue
        n = 0
        t_b = time.time()
        for rec in recs:
            if n >= args.max_decisions_per_bucket:
                break
            am = rec.get("action_metadata") or {}
            raw = am.get("raw_response")
            if not raw:
                continue
            prompt = recon.build(rec)
            full_text = prompt + raw
            # Find the verb position in the FULL input.
            verb_offset_info = find_action_verb_response_offset(raw, tokenizer)
            if verb_offset_info is None:
                continue
            verb_char_offset_in_response, _ = verb_offset_info
            text_up_to_verb = prompt + raw[:verb_char_offset_in_response]
            ids_up_to_verb = tokenizer(
                text_up_to_verb, add_special_tokens=False
            )["input_ids"]
            verb_abs_pos = len(ids_up_to_verb) - 1   # last position whose
                                                     # logits = verb prediction.

            # Tokenize the full input (prompt + raw_response).
            enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(args.device)
            seq_len = int(input_ids.shape[1])

            with cap, torch.no_grad():
                model(input_ids=input_ids)
            res = cap._captured  # [seq, hidden] CPU tensor
            if res is None:
                continue
            # Convert to numpy float32 for the dot product.
            res_np = res.to(torch.float32).numpy()
            # Project each position onto w (signed, normalized).
            proj_all = (res_np @ w) / (w_norm + 1e-9)  # [seq]

            ids_list = input_ids[0].detach().to("cpu").tolist()

            for r in args.positions_relative:
                abs_p = verb_abs_pos + r
                if 0 <= abs_p < seq_len:
                    p = float(proj_all[abs_p])
                    tok = tokenizer.decode([ids_list[abs_p]]) if 0 <= abs_p < len(ids_list) else "<oob>"
                    agg[(bucket, r)].append(p)
                    writer.writerow([
                        bucket, rec.get("hand_id", "?"),
                        rec.get("decision_idx", "?"),
                        verb_abs_pos, r, abs_p,
                        f"{p:.4f}", tok,
                    ])
            n += 1
            if n % 10 == 0:
                rate = n / max(time.time() - t_b, 1e-3)
                print(f"  [{bucket}] {n} captured ({rate:.2f}/s)")
        print(f"  [{bucket}] DONE: {n} decisions ({time.time()-t_b:.0f}s)")

    by_pair_f.close()

    # Summary.
    summary = {
        "model_id": model_id,
        "layer": args.layer,
        "probe_npz": args.probe_npz,
        "buckets": args.buckets,
        "positions_relative": args.positions_relative,
        "per_bucket_per_position": {},
    }
    for bucket in args.buckets:
        summary["per_bucket_per_position"][bucket] = {}
        for r in args.positions_relative:
            vals = agg[(bucket, r)]
            if vals:
                summary["per_bucket_per_position"][bucket][str(r)] = {
                    "n": len(vals),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                }
            else:
                summary["per_bucket_per_position"][bucket][str(r)] = None
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # SUMMARY.md
    md = []
    md.append("# Position-sweep direction projection")
    md.append("")
    md.append(f"- Model: `{model_id}`")
    md.append(f"- Layer: **{args.layer}**")
    md.append(f"- Direction probe weights: `{args.probe_npz}`")
    md.append(f"- Buckets: {args.buckets}")
    md.append(f"- Per-bucket sample cap: {args.max_decisions_per_bucket}")
    md.append("")
    md.append(
        "Sign convention: positive = CHECK side (signed projection onto "
        "the probe's weight vector, normalized by ||w||). Position is "
        "RELATIVE to the verb-emission position (0 = the last input "
        "position whose lm_head output predicts the verb)."
    )
    md.append("")
    md.append("## Mean projection at each relative position")
    md.append("")
    header = "| rel_pos | " + " | ".join(f"`{b}` mean ± std (n)" for b in args.buckets) + " |"
    sep = "|---:|" + "|".join(["---"] * len(args.buckets)) + "|"
    md.append(header)
    md.append(sep)
    for r in args.positions_relative:
        cells = []
        for b in args.buckets:
            entry = summary["per_bucket_per_position"][b].get(str(r))
            if entry is None:
                cells.append("—")
            else:
                cells.append(f"{entry['mean']:+.2f} ± {entry['std']:.2f} ({entry['n']})")
        md.append(f"| {r:+d} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("## Reading guide")
    md.append("")
    md.append(
        "- **`rel_pos = 0` is the verb-emission position** — by construction this is "
        "where the patching experiments measure the decision. The mean "
        "projection here should match the per-bucket projections from the "
        "direction probe (within sampling noise).")
    md.append(
        "- **`rel_pos < 0` (earlier in response)**: where in the response trace does "
        "the decision crystallize? Compute-then-commit predicts: projection should "
        "be near zero (undecided) for very negative offsets (early in the response, "
        "still reasoning), then diverge sharply approaching the verb position.")
    md.append(
        "- **`rel_pos > 0` (after the verb)**: how does the residual evolve after "
        "the model has emitted the verb? If the projection stays at the bucket's "
        "verb-position level, the decision is consistently encoded throughout. If "
        "it relaxes back toward zero, the model's residual moves on quickly.")
    md.append(
        "- **Cross-bucket comparison at very negative offsets**: if the buckets are "
        "INDISTINGUISHABLE there, the prompt itself doesn't pre-determine the verb "
        "(no prompt-level shortcut). If they ARE distinguishable, the prompt context "
        "alone partially predicts the decision (and the residual is doing less work "
        "than we thought).")
    md.append(
        "- **Variance comparison**: `illegal_fold` is expected to have higher "
        "variance at the verb position (consistent with §17a's higher attention entropy).")
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[done] wrote {out_dir / 'SUMMARY.md'}")
    print(f"[done] wrote {out_dir / 'summary.json'}")
    print(f"[done] wrote {by_pair_path}")


if __name__ == "__main__":
    main()
