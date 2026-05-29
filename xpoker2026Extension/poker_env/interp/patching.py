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


class ActivationAdditionHook:
    """Forward-hook that ADDS ``alpha * vec`` to a layer's residual (steering).

    Unlike HiddenStatePatch (which REPLACES the last position), this ADDS a fixed
    direction. By default it steers EVERY position so the steer persists across all
    tokens emitted by ``model.generate()`` (set ``last_only=True`` to steer only the
    last input position, e.g. for a single-forward logit readout).

    Exposes ``attach()``/``detach()`` (and is a context manager), so it plugs into both
    the single-forward path (``with hook:``) and generation via
    ``attached_hooks([hook])`` / ``agent._extra_generation_hooks`` — identical wiring to
    AttnHeadZeroAblation.

    Usage::

        steer = ActivationAdditionHook(model, layer_idx=23, vec=direction, alpha=6.0)
        with steer:
            out = model.generate(input_ids=ids, max_new_tokens=200, do_sample=False)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        vec: torch.Tensor,
        alpha: float = 1.0,
        last_only: bool = False,
    ):
        layers = _get_layers(model)
        if not (0 <= layer_idx < len(layers)):
            raise ValueError(f"layer_idx={layer_idx} out of range [0, {len(layers)})")
        self.model = model
        self.layer_idx = layer_idx
        self.layer = layers[layer_idx]
        self.vec = vec
        self.alpha = float(alpha)
        self.last_only = bool(last_only)
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
        v = self.vec.to(device=residual.device, dtype=residual.dtype)
        new_residual = residual.clone()
        if self.last_only:
            new_residual[0, -1, :] = new_residual[0, -1, :] + self.alpha * v
        else:
            new_residual[0, :, :] = new_residual[0, :, :] + self.alpha * v
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


# ---------------------------------------------------------------------------
# Component-level capture / patch (B1: per-sublayer + per-head)
# ---------------------------------------------------------------------------
#
# These primitives let us answer "WHICH sublayer (attention vs MLP) and WHICH
# attention heads at layer L mediate the verb-decision effect?" rather than
# the layer-as-a-whole question that HiddenStatePatch (above) addresses.
#
# Llama / Mistral / Qwen all use the standard transformer block layout:
#     def forward(x):
#         h   = x + self_attn(norm1(x))     # attention sublayer contribution
#         out = h + mlp(norm2(h))           # MLP sublayer contribution
#         return out
#
# So the layer's output residual = x + attn_contribution + mlp_contribution.
# We capture / patch the two contributions separately by hooking on the
# `self_attn` and `mlp` submodules.
#
# For per-head, we hook on `self_attn.o_proj` — a Linear whose INPUT is the
# pre-projection per-head concatenation of shape [batch, seq, num_heads *
# head_dim]. We can capture that input (reshaped per-head) and patch
# specific head slices via a forward_pre_hook that returns a modified input
# tuple. After o_proj projects the modified input, the rest of the layer
# (residual addition + MLP) sees the modified attention output.
#
# Two architectural assumptions, both verified on Llama-3.1-8B-Instruct,
# Ministral-8B-Instruct-2410, and Qwen3-8B:
#   - layer.self_attn exists and its forward returns the projected attention
#     output (shape [batch, seq, hidden_dim])
#   - layer.self_attn.o_proj exists, takes [batch, seq, num_heads * head_dim]
#     input, returns [batch, seq, hidden_dim]
# These are stable across the GQA / MHA distinction because o_proj's input
# is always the "Q-aligned" per-head representation regardless of how many
# KV heads are used.


def _attn_module(model: nn.Module, layer_idx: int):
    return _get_layers(model)[layer_idx].self_attn


def _mlp_module(model: nn.Module, layer_idx: int):
    return _get_layers(model)[layer_idx].mlp


def _o_proj_module(model: nn.Module, layer_idx: int):
    return _get_layers(model)[layer_idx].self_attn.o_proj


def get_head_geometry(model) -> tuple[int, int]:
    """Return ``(num_heads, head_dim)`` for the QUERY heads. Uses
    ``model.config.num_attention_heads`` and ``head_dim`` (or derives the
    latter from ``hidden_size // num_attention_heads`` for older configs)."""
    num_heads = int(model.config.num_attention_heads)
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = int(model.config.hidden_size) // num_heads
    return num_heads, int(head_dim)


class HiddenStateCaptureMulti:
    """One-pass capture of (residual, attn_contribution, mlp_contribution,
    per_head_pre_oproj) at the last input position for every layer.

    Use this in component-level experiments: a single source forward
    populates everything you need to patch any sublayer or any head at any
    layer downstream.

    Attributes after ``collect()`` (all values are 1-D or 2-D CPU tensors):
      - per_layer_residual[L]:        shape [hidden_dim]
      - per_layer_attn_out[L]:        shape [hidden_dim]
      - per_layer_mlp_out[L]:         shape [hidden_dim]
      - per_layer_attn_per_head[L]:   shape [num_heads, head_dim]
                                       (the o_proj input, reshaped)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._layers = _get_layers(model)
        self._num_layers = len(self._layers)
        self._num_heads, self._head_dim = get_head_geometry(model)
        self._hooks: list = []
        self._residual: dict[int, torch.Tensor] = {}
        self._attn_out: dict[int, torch.Tensor] = {}
        self._mlp_out: dict[int, torch.Tensor] = {}
        self._attn_per_head: dict[int, torch.Tensor] = {}

    def attach_hooks(self) -> None:
        self._residual.clear()
        self._attn_out.clear()
        self._mlp_out.clear()
        self._attn_per_head.clear()
        for idx, layer in enumerate(self._layers):
            self._hooks.append(
                layer.register_forward_hook(self._make_residual_hook(idx))
            )
            self._hooks.append(
                layer.self_attn.register_forward_hook(self._make_attn_hook(idx))
            )
            self._hooks.append(
                layer.mlp.register_forward_hook(self._make_mlp_hook(idx))
            )
            self._hooks.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    self._make_oproj_pre_hook(idx)
                )
            )

    def detach_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def collect(self) -> dict:
        return {
            "num_layers": self._num_layers,
            "num_heads": self._num_heads,
            "head_dim": self._head_dim,
            "per_layer_residual": dict(self._residual),
            "per_layer_attn_out": dict(self._attn_out),
            "per_layer_mlp_out": dict(self._mlp_out),
            "per_layer_attn_per_head": dict(self._attn_per_head),
        }

    def _make_residual_hook(self, layer_idx: int):
        def hook(module, input, output):
            res = _layer_output_residual(output)
            self._residual[layer_idx] = res[0, -1, :].detach().to("cpu")
        return hook

    def _make_attn_hook(self, layer_idx: int):
        def hook(module, input, output):
            # self_attn output: usually a Tensor [batch, seq, hidden]; some
            # HF versions return a tuple (output, attn_weights, ...).
            out = output[0] if isinstance(output, tuple) else output
            self._attn_out[layer_idx] = out[0, -1, :].detach().to("cpu")
        return hook

    def _make_mlp_hook(self, layer_idx: int):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self._mlp_out[layer_idx] = out[0, -1, :].detach().to("cpu")
        return hook

    def _make_oproj_pre_hook(self, layer_idx: int):
        # forward_pre_hook signature: (module, args) where args is a tuple.
        # First positional arg is the input tensor.
        nh, hd = self._num_heads, self._head_dim
        def hook(module, args):
            x = args[0]                                # [batch, seq, nh*hd]
            last_pos = x[0, -1, :].detach().to("cpu")  # [nh*hd]
            self._attn_per_head[layer_idx] = last_pos.view(nh, hd)
            return None  # don't modify
        return hook


class HiddenStatePatchAttnOnly:
    """Replace ONLY the attention sublayer's output at the last position at
    one layer. The MLP downstream (and the layer's residual addition) sees
    the modified attention contribution; everything else is unchanged.

    Usage::

        patch = HiddenStatePatchAttnOnly(model, layer_idx=14,
                                         source_attn_out=src_attn_state)
        with patch:
            out = model(input_ids=tgt_ids)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        source_attn_out: torch.Tensor,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.attn_module = _attn_module(model, layer_idx)
        self.source_attn_out = source_attn_out
        self._hook = None

    def attach(self) -> None:
        self._hook = self.attn_module.register_forward_hook(self._hook_fn)

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
        res = _layer_output_residual(output)
        target_device = res.device
        target_dtype = res.dtype
        src = self.source_attn_out.to(device=target_device, dtype=target_dtype)
        new_res = res.clone()
        new_res[0, -1, :] = src
        return _replace_layer_output_residual(output, new_res)


class HiddenStatePatchMLPOnly:
    """Replace ONLY the MLP sublayer's output at the last position at one
    layer. The layer's residual addition sees the modified MLP contribution;
    the attention contribution is untouched.

    Usage::

        patch = HiddenStatePatchMLPOnly(model, layer_idx=14,
                                        source_mlp_out=src_mlp_state)
        with patch:
            out = model(input_ids=tgt_ids)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        source_mlp_out: torch.Tensor,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.mlp_module = _mlp_module(model, layer_idx)
        self.source_mlp_out = source_mlp_out
        self._hook = None

    def attach(self) -> None:
        self._hook = self.mlp_module.register_forward_hook(self._hook_fn)

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
        # MLP output is a tensor [batch, seq, hidden].
        out = output[0] if isinstance(output, tuple) else output
        target_device = out.device
        target_dtype = out.dtype
        src = self.source_mlp_out.to(device=target_device, dtype=target_dtype)
        new_out = out.clone()
        new_out[0, -1, :] = src
        if isinstance(output, tuple):
            return (new_out,) + output[1:]
        return new_out


class HiddenStatePatchAttnHeadSubset:
    """Replace a SUBSET of attention heads at the last position at one layer
    by modifying the o_proj input. Heads not in ``head_indices`` pass through
    unchanged.

    ``source_per_head_residual`` is shape ``[num_heads, head_dim]`` (matches
    ``HiddenStateCaptureMulti``'s ``per_layer_attn_per_head[layer_idx]``).
    ``head_indices`` is the list of head indices to replace.

    Usage::

        # Replace heads 7, 12, 19 only:
        patch = HiddenStatePatchAttnHeadSubset(
            model, layer_idx=14,
            source_per_head_residual=src["per_layer_attn_per_head"][14],
            head_indices=[7, 12, 19],
        )
        with patch:
            out = model(input_ids=tgt_ids)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        source_per_head_residual: torch.Tensor,
        head_indices: Iterable[int],
    ):
        num_heads, head_dim = get_head_geometry(model)
        if source_per_head_residual.shape != (num_heads, head_dim):
            raise ValueError(
                f"source_per_head_residual shape {tuple(source_per_head_residual.shape)} "
                f"!= expected ({num_heads}, {head_dim})"
            )
        self.model = model
        self.layer_idx = layer_idx
        self.o_proj = _o_proj_module(model, layer_idx)
        self.source_per_head = source_per_head_residual
        self.head_indices = list(head_indices)
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._hook = None

    def attach(self) -> None:
        self._hook = self.o_proj.register_forward_pre_hook(self._hook_fn)

    def detach(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *exc):
        self.detach()

    def _hook_fn(self, module, args):
        x = args[0]                            # [batch, seq, nh*hd]
        target_device = x.device
        target_dtype = x.dtype
        src = self.source_per_head.to(device=target_device, dtype=target_dtype)
        # Per-head view into x, last position.
        new_x = x.clone()
        # Reshape last position only: [nh*hd] -> [nh, hd], replace, flatten.
        last = new_x[0, -1, :].view(self._num_heads, self._head_dim).clone()
        for h_idx in self.head_indices:
            if not (0 <= h_idx < self._num_heads):
                raise ValueError(
                    f"head_idx={h_idx} out of range [0, {self._num_heads})"
                )
            last[h_idx, :] = src[h_idx, :]
        new_x[0, -1, :] = last.view(-1)
        return (new_x,) + args[1:]


__all__ = [
    "HiddenStateCapture",
    "HiddenStatePatch",
    "HiddenStateMultiPatch",
    "HiddenStateCaptureMulti",
    "HiddenStatePatchAttnOnly",
    "HiddenStatePatchMLPOnly",
    "HiddenStatePatchAttnHeadSubset",
    "get_head_geometry",
]
