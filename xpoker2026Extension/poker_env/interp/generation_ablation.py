"""
Attention-head ablation hooks active during ``model.generate()``.

Zeros the o_proj input slices for selected heads at the **last sequence
position** on every forward pass while the hook is attached. Used for
behavioral necessity tests (illegal_FOLD rate under live CoT generation)
and continuation-after-patch experiments.
"""

from __future__ import annotations

import torch

from poker_env.interp.patching import (
    HiddenStatePatchAttnHeadSubset,
    get_head_geometry,
)


class AttnHeadZeroAblation:
    """Zero selected attention heads at the last token position at one layer.

    Wraps ``HiddenStatePatchAttnHeadSubset`` with an all-zero per-head source.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_idx: int,
        head_indices: list[int],
    ):
        num_heads, head_dim = get_head_geometry(model)
        for h in head_indices:
            if not (0 <= h < num_heads):
                raise ValueError(
                    f"head_idx={h} out of range [0, {num_heads})"
                )
        zero = torch.zeros((num_heads, head_dim), dtype=torch.float32)
        self._patch = HiddenStatePatchAttnHeadSubset(
            model, layer_idx, zero, head_indices
        )
        self.layer_idx = layer_idx
        self.head_indices = list(head_indices)

    def attach(self) -> None:
        self._patch.attach()

    def detach(self) -> None:
        self._patch.detach()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *exc):
        self.detach()


__all__ = ["AttnHeadZeroAblation"]
