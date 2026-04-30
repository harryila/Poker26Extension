"""
Logit lens: project hidden states at each transformer layer through the
unembedding matrix to see how predictions evolve layer-by-layer.

Attaches forward hooks *before* model.generate() so hidden states are
captured during the existing generation -- no redundant forward pass.

Compatible with Llama, Mistral, Qwen (all use model.model.layers + model.lm_head).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class LogitLensExtractor:
    """
    Captures per-layer hidden states during generation and projects them
    through the model's unembedding head to produce per-layer logits.

    Usage::

        extractor = LogitLensExtractor(model, tokenizer)
        extractor.attach_hooks()
        outputs = model.generate(**inputs)
        data = extractor.collect()   # dict with per-layer info
        extractor.detach_hooks()
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        # layer_idx -> list of hidden-state tensors (one per forward call / generated token)
        self._layer_hidden_states: dict[int, list[torch.Tensor]] = {}
        self._num_layers = self._count_layers()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def attach_hooks(self) -> None:
        """Register forward hooks on every transformer layer."""
        self._clear_buffers()
        layers = self._get_layers()
        for idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def detach_hooks(self) -> None:
        """Remove all hooks and free GPU memory."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def collect(self) -> dict:
        """
        Collect captured data and return a summary dict.

        Returns dict with:
            num_layers: int
            num_positions: int (generated tokens captured)
            per_layer_entropy: list[float]  (mean entropy over positions, per layer)
            per_layer_top_token: list[list[str]]  (top-1 token per position, per layer)
        """
        if not self._layer_hidden_states:
            return {
                "num_layers": 0,
                "num_positions": 0,
                "per_layer_entropy": [],
                "per_layer_top_tokens": [],
            }

        num_positions = len(next(iter(self._layer_hidden_states.values())))
        per_layer_entropy: list[float] = []
        per_layer_top_tokens: list[list[str]] = []

        norm = self._get_final_norm()
        lm_head = self._get_lm_head()

        for layer_idx in range(self._num_layers):
            hidden_list = self._layer_hidden_states.get(layer_idx, [])
            entropies = []
            top_tokens = []

            for h in hidden_list:
                # h is [batch, seq_len, hidden_dim]; take last position
                # h was moved to CPU in hook; move to model device for norm/lm_head
                last_hidden = h[0, -1, :].unsqueeze(0)
                target_device = next(lm_head.parameters()).device
                last_hidden = last_hidden.to(target_device)
                if norm is not None:
                    last_hidden = norm(last_hidden)
                logits = lm_head(last_hidden)  # [1, vocab_size]
                probs = torch.softmax(logits[0].float(), dim=-1)
                entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
                entropies.append(entropy)
                top_id = probs.argmax().item()
                top_tokens.append(self.tokenizer.decode([top_id]))

            per_layer_entropy.append(float(np.mean(entropies)) if entropies else 0.0)
            per_layer_top_tokens.append(top_tokens)

        self._clear_buffers()

        return {
            "num_layers": self._num_layers,
            "num_positions": num_positions,
            "per_layer_entropy": per_layer_entropy,
            "per_layer_top_tokens": per_layer_top_tokens,
        }

    # ------------------------------------------------------------------
    # Sidecar I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save_sidecar(data: dict, path: str | Path, hand_id: str, decision_idx: int) -> None:
        """Append a logit-lens record to a JSONL sidecar file."""
        record = {
            "hand_id": hand_id,
            "decision_idx": decision_idx,
            **data,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def load_sidecar(path: str | Path) -> list[dict]:
        """Load all logit-lens records from a sidecar file."""
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output is usually a tuple; first element is the hidden state tensor
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            if layer_idx not in self._layer_hidden_states:
                self._layer_hidden_states[layer_idx] = []
            self._layer_hidden_states[layer_idx].append(hidden.detach().cpu())
        return hook_fn

    def _clear_buffers(self) -> None:
        self._layer_hidden_states.clear()

    def _get_layers(self) -> nn.ModuleList:
        """Get the transformer layers. Works for Llama/Mistral/Qwen."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise AttributeError(
            f"Cannot find transformer layers on {type(self.model).__name__}. "
            "Expected model.model.layers (Llama/Mistral/Qwen architecture)."
        )

    def _get_final_norm(self) -> nn.Module | None:
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        return None

    def _get_lm_head(self) -> nn.Module:
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        raise AttributeError(f"Cannot find lm_head on {type(self.model).__name__}.")

    def _count_layers(self) -> int:
        try:
            return len(self._get_layers())
        except AttributeError:
            return 0
