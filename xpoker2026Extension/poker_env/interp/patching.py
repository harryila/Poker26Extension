"""
Activation patching for the late-layer-deliberation hypothesis.

Companion to ``logit_lens.py``. Two cooperating classes:

  - ``HiddenStateCapture``: forward-hooks every transformer layer and saves the
    residual at the LAST input position (the position whose lm_head projection
    becomes the next-token prediction). One forward pass populates all layers.

  - ``HiddenStatePatch``: forward-hook on ONE transformer layer that *replaces*
    the residual at the LAST input position with a previously captured tensor.
    Used in target-side forward passes to inject a source decision's late-layer
    state into a different decision and observe the resulting logits.

Why "last input position":
  In a HuggingFace ``model(input_ids=...)`` forward pass on a fixed input
  (no generation loop), the next-token logits are derived from the residual at
  position ``input_ids.shape[1] - 1``. Patching that position is equivalent to
  asking "what verb would the model emit next, given this layer-L state?".

Compatibility:
  Llama / Mistral / Qwen all expose ``model.model.layers`` with each layer
  output being a tuple whose first element is the ``[batch, seq_len,
  hidden_dim]`` residual stream. Same architecture assumption as logit_lens.py.

Determinism:
  Captured tensors are detached and moved to CPU. For patching, the saved
  tensor is moved back to the target layer's device just before injection.
  Patching modifies the layer output in-place via ``.clone()`` then index
  assignment, so subsequent layers see the patched value transparently.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers — shared with logit_lens.py but redeclared to avoid cross-imports
# ---------------------------------------------------------------------------


def _get_layers(model: nn.Module) -> nn.ModuleList:
    """Return the transformer layers list (Llama/Mistral/Qwen)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError(
        f"Cannot find transformer layers on {type(model).__name__}; "
        "expected model.model.layers (Llama/Mistral/Qwen architecture)."
    )


def _layer_output_residual(output) -> torch.Tensor:
    """A transformer layer's output is either a Tensor or a tuple whose first
    element is the residual stream tensor."""
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_layer_output_residual(output, new_residual: torch.Tensor):
    """Return an output object identical to ``output`` except with the residual
    stream replaced by ``new_residual``. Preserves tuple structure."""
    if isinstance(output, tuple):
        return (new_residual,) + output[1:]
    return new_residual


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


class HiddenStateCapture:
    """Captures the residual at the LAST input position for every layer.

    Single forward pass populates all layers in one shot. After ``collect()``,
    ``per_layer_last_pos[layer_idx]`` is a 1-D CPU tensor of shape
    ``[hidden_dim]``.

    Usage::

        cap = HiddenStateCapture(model)
        cap.attach_hooks()
        with torch.no_grad():
            model(input_ids=tokens)
        states = cap.collect()
        cap.detach_hooks()
        # states["per_layer_last_pos"][layer_idx]  -> 1D tensor
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._layers = _get_layers(model)
        self._num_layers = len(self._layers)
        self._hooks: list = []
        self._captures: dict[int, torch.Tensor] = {}

    def attach_hooks(self) -> None:
        self._captures.clear()
        for idx, layer in enumerate(self._layers):
            self._hooks.append(layer.register_forward_hook(self._make_hook(idx)))

    def detach_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def collect(self) -> dict:
        """Return ``{"num_layers": N, "per_layer_last_pos": {layer_idx: 1D
        CPU tensor}}``. Empty if no forward pass has run since attach."""
        return {
            "num_layers": self._num_layers,
            "per_layer_last_pos": dict(self._captures),
        }

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            residual = _layer_output_residual(output)
            # residual: [batch, seq_len, hidden_dim]; we want last position.
            last_pos = residual[0, -1, :].detach().to("cpu")
            self._captures[layer_idx] = last_pos
        return hook


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------


class HiddenStatePatch:
    """Forward-hook that replaces the residual at the LAST input position at
    ONE layer with a provided tensor.

    Usage::

        patch = HiddenStatePatch(model, layer_idx=28, source_residual=src_state)
        patch.attach()
        with torch.no_grad():
            out = model(input_ids=target_tokens)
        patch.detach()
        logits_at_last_pos = out.logits[0, -1, :]    # [vocab]
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        source_residual: torch.Tensor,
    ):
        layers = _get_layers(model)
        if not (0 <= layer_idx < len(layers)):
            raise ValueError(
                f"layer_idx={layer_idx} out of range [0, {len(layers)})"
            )
        self.model = model
        self.layer_idx = layer_idx
        self.layer = layers[layer_idx]
        self.source_residual = source_residual
        self._hook = None

    def attach(self) -> None:
        self._hook = self.layer.register_forward_hook(self._hook_fn)

    def detach(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *exc):
        self.detach()

    def _hook_fn(self, module, input, output):
        residual = _layer_output_residual(output)
        # residual: [batch, seq_len, hidden_dim]
        # source_residual: [hidden_dim]
        target_device = residual.device
        target_dtype = residual.dtype
        src = self.source_residual.to(device=target_device, dtype=target_dtype)
        # Clone so we don't mutate the original tensor (which may be returned
        # from earlier in the layer module's own bookkeeping).
        new_residual = residual.clone()
        new_residual[0, -1, :] = src
        return _replace_layer_output_residual(output, new_residual)


# ---------------------------------------------------------------------------
# Multi-layer patch (for layer-sweep ablations in one forward pass — optional;
# the experiment driver uses one HiddenStatePatch per layer to keep the
# scoring per-layer-clean, but exposing this for completeness)
# ---------------------------------------------------------------------------


class HiddenStateMultiPatch:
    """Patch the LAST input position residual at multiple layers at once.

    Useful for "patch every late layer simultaneously" controls, NOT used by
    the main per-layer driver.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_to_residual: dict[int, torch.Tensor],
    ):
        layers = _get_layers(model)
        for layer_idx in layer_to_residual:
            if not (0 <= layer_idx < len(layers)):
                raise ValueError(
                    f"layer_idx={layer_idx} out of range [0, {len(layers)})"
                )
        self.model = model
        self.layer_to_residual = layer_to_residual
        self._hooks: list = []
        self._layers_list = layers

    def attach(self) -> None:
        for layer_idx, src in self.layer_to_residual.items():
            patcher = HiddenStatePatch(self.model, layer_idx, src)
            self._hooks.append(self._layers_list[layer_idx].register_forward_hook(
                patcher._hook_fn
            ))

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *exc):
        self.detach()


__all__ = [
    "HiddenStateCapture",
    "HiddenStatePatch",
    "HiddenStateMultiPatch",
]
