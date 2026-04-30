"""
Token attribution via input saliency (integrated gradients).

Uses the captum library to compute which input tokens most influence
the model's output distribution for action/belief tokens.

Answers: are card tokens and action tokens the top contributors,
or is the model over-relying on instructions/formatting?
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from captum.attr import LayerIntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


class AttributionAnalyzer:
    """
    Compute input-token attributions for LLM poker agent decisions.

    Uses Layer Integrated Gradients on the embedding layer to attribute
    the predicted output logit to each input token.

    Usage::

        analyzer = AttributionAnalyzer(model, tokenizer)
        result = analyzer.attribute(input_ids, target_token_id)
        # result: dict with per-token attribution scores
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for AttributionAnalyzer")
        if not CAPTUM_AVAILABLE:
            raise ImportError("captum is required: pip install captum")

        self.model = model
        self.tokenizer = tokenizer
        self._embedding_layer = self._find_embedding_layer()

        # Wrap model forward for captum (input_embeds -> logits)
        def forward_fn(input_embeds: torch.Tensor) -> torch.Tensor:
            attention_mask = torch.ones(
                input_embeds.shape[:2], dtype=torch.long, device=input_embeds.device
            )
            outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
            )
            return outputs.logits[:, -1, :]  # logits for last position

        self._lig = LayerIntegratedGradients(forward_fn, self._embedding_layer)

    def attribute(
        self,
        input_ids: torch.Tensor,
        target_token_id: int,
        n_steps: int = 50,
    ) -> dict:
        """
        Compute attribution scores for each input token.

        Args:
            input_ids: [1, seq_len] tensor of input token IDs
            target_token_id: the output token ID to attribute toward
            n_steps: number of interpolation steps for integrated gradients

        Returns:
            dict with:
                tokens: list[str]
                attributions: list[float] (per-token attribution magnitude)
                top_5: list of (token, score) tuples
        """
        input_ids = input_ids.to(self.model.device)

        # Get embeddings as input (captum needs differentiable input)
        input_embeds = self._embedding_layer(input_ids)

        # Baseline: zero embeddings
        baseline = torch.zeros_like(input_embeds)

        attrs = self._lig.attribute(
            input_embeds,
            baselines=baseline,
            target=target_token_id,
            n_steps=n_steps,
        )

        # Sum across embedding dim -> per-token scalar
        attr_scores = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        attr_magnitudes = np.abs(attr_scores)

        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        # Top-5 by magnitude
        top_indices = attr_magnitudes.argsort()[-5:][::-1]
        top_5 = [(tokens[i], float(attr_magnitudes[i])) for i in top_indices]

        return {
            "tokens": tokens,
            "attributions": attr_magnitudes.tolist(),
            "top_5": top_5,
        }

    def attribute_to_categories(
        self,
        input_ids: torch.Tensor,
        target_token_id: int,
        token_categories: list[str],
        n_steps: int = 50,
    ) -> dict[str, float]:
        """
        Compute attribution aggregated by token category.

        Args:
            input_ids: [1, seq_len]
            target_token_id: output token to attribute toward
            token_categories: per-token category labels (from AttentionExtractor)
            n_steps: IG interpolation steps

        Returns:
            dict mapping category -> fraction of total attribution
        """
        result = self.attribute(input_ids, target_token_id, n_steps)
        attrs = result["attributions"]

        category_sums: dict[str, float] = {}
        for score, cat in zip(attrs, token_categories):
            category_sums[cat] = category_sums.get(cat, 0.0) + score

        total = sum(category_sums.values())
        if total > 0:
            return {k: v / total for k, v in category_sums.items()}
        return category_sums

    def _find_embedding_layer(self) -> nn.Module:
        """Find the token embedding layer."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        raise AttributeError(f"Cannot find embedding layer on {type(self.model).__name__}")
