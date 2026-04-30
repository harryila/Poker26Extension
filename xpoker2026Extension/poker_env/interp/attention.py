"""
Attention pattern analysis for poker LLM agents.

Extracts attention weights during generation and categorizes which
input tokens (cards, actions, instructions) the model attends to
when producing action/belief output tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore


# Token categories for aggregation
TOKEN_CATEGORIES = {
    "card": re.compile(
        r'^[2-9TJQKA][cdhs♣♦♥♠]$|^(10|[2-9]|[TJQKA])\s*(of\s+)?(clubs?|diamonds?|hearts?|spades?)$',
        re.IGNORECASE,
    ),
    "action": re.compile(
        r'fold|call|check|raise|bet|raised|called|checked',
        re.IGNORECASE,
    ),
    "number": re.compile(r'^\d+\.?\d*$'),
}


@dataclass
class AttentionSnapshot:
    """Attention data for one decision point."""
    # Per-layer, per-head attention from last generated token to all input tokens
    # Shape reference: [num_layers, num_heads, input_seq_len]
    attention_to_input: np.ndarray | None = None
    input_tokens: list[str] = field(default_factory=list)
    token_categories: list[str] = field(default_factory=list)

    def category_fractions(self) -> dict[str, float]:
        """Fraction of total attention going to each token category."""
        if self.attention_to_input is None or len(self.token_categories) == 0:
            return {}

        # Only average over layers that have non-zero attention (populated by hooks)
        layer_sums = self.attention_to_input.sum(axis=(1, 2))
        populated_mask = layer_sums > 0
        if not populated_mask.any():
            return {}
        avg_attn = self.attention_to_input[populated_mask].mean(axis=(0, 1))

        category_sums: dict[str, float] = {}
        for i, cat in enumerate(self.token_categories):
            category_sums[cat] = category_sums.get(cat, 0.0) + float(avg_attn[i])

        total = sum(category_sums.values())
        if total > 0:
            return {k: v / total for k, v in category_sums.items()}
        return category_sums

    def to_dict(self) -> dict:
        fracs = self.category_fractions()
        return {
            "num_input_tokens": len(self.input_tokens),
            "category_fractions": fracs,
        }


class AttentionExtractor:
    """
    Captures attention weights during a forward pass / generation.

    Usage::

        extractor = AttentionExtractor(model, tokenizer)
        extractor.attach_hooks()
        outputs = model.generate(**inputs)
        snapshot = extractor.collect(input_ids)
        extractor.detach_hooks()
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self._hooks: list = []
        self._attention_weights: dict[int, list[torch.Tensor]] = {}

    def attach_hooks(self) -> None:
        self._attention_weights.clear()
        layers = self._get_layers()
        for idx, layer in enumerate(layers):
            attn_module = self._get_attn_submodule(layer)
            if attn_module is not None:
                hook = attn_module.register_forward_hook(self._make_hook(idx))
                self._hooks.append(hook)

    def detach_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._attention_weights.clear()

    def collect(self, input_ids: torch.Tensor | None = None) -> AttentionSnapshot:
        """
        Collect attention data captured during generation.

        Args:
            input_ids: the input token IDs used for generation (for token decoding)
        """
        if not self._attention_weights:
            return AttentionSnapshot()

        # Decode input tokens for category labeling
        input_tokens: list[str] = []
        token_categories: list[str] = []
        if input_ids is not None:
            ids = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
            input_tokens = [self.tokenizer.decode([tid]) for tid in ids]
            token_categories = [self._categorize_token(t) for t in input_tokens]

        # Build attention array: take the LAST captured attention per layer
        num_layers = max(self._attention_weights.keys()) + 1
        token_len = len(input_tokens) if input_tokens else None

        # Determine effective length from attention tensors
        attn_key_len = None
        for layer_idx in range(num_layers):
            attn_list = self._attention_weights.get(layer_idx, [])
            if attn_list and attn_list[-1].dim() == 4:
                attn_key_len = attn_list[-1].shape[-1]
                break

        if token_len is not None and attn_key_len is not None:
            input_len = min(token_len, attn_key_len)
        elif attn_key_len is not None:
            input_len = attn_key_len
        else:
            input_len = token_len

        # Truncate token lists to match effective length
        if input_len is not None and input_tokens and len(input_tokens) > input_len:
            input_tokens = input_tokens[:input_len]
            token_categories = token_categories[:input_len]

        attention_slices = []
        for layer_idx in range(num_layers):
            attn_list = self._attention_weights.get(layer_idx, [])
            if attn_list:
                last_attn = attn_list[-1]
                if last_attn.dim() == 4:
                    if input_len is None:
                        input_len = last_attn.shape[-1]
                    attn_to_input = last_attn[0, :, -1, :input_len].detach().cpu().numpy()
                    attention_slices.append((layer_idx, attn_to_input))

        if attention_slices and input_len:
            num_heads = attention_slices[0][1].shape[0]
            full_array = np.zeros((num_layers, num_heads, input_len))
            for layer_idx, attn_slice in attention_slices:
                full_array[layer_idx] = attn_slice
            attention_to_input = full_array
        else:
            attention_to_input = None

        self._attention_weights.clear()
        return AttentionSnapshot(
            attention_to_input=attention_to_input,
            input_tokens=input_tokens,
            token_categories=token_categories,
        )

    # ------------------------------------------------------------------

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # Many attention implementations return (attn_output, attn_weights, ...)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1]
                if layer_idx not in self._attention_weights:
                    self._attention_weights[layer_idx] = []
                self._attention_weights[layer_idx].append(attn_weights.detach().cpu())
        return hook_fn

    @staticmethod
    def _categorize_token(token: str) -> str:
        stripped = token.strip()
        for category, pattern in TOKEN_CATEGORIES.items():
            if pattern.search(stripped):
                return category
        return "other"

    def _get_layers(self) -> nn.ModuleList:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise AttributeError(f"Cannot find layers on {type(self.model).__name__}")

    @staticmethod
    def _get_attn_submodule(layer: nn.Module) -> nn.Module | None:
        """Find the self-attention sub-module inside a transformer layer."""
        for name in ("self_attn", "attention"):
            if hasattr(layer, name):
                return getattr(layer, name)
        return None
