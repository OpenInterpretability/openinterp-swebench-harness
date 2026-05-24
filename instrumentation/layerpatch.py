"""LayerPatch — REPLACE-style forward hooks for activation patching causal tests.

Companion to LayerTap (which captures activations). LayerPatch installs hooks
that REPLACE the last-token residual at specified layers during forward pass,
enabling cross-layer activation patching experiments for causal interventions.

Usage (Exp B from Tool-Entropy paper #2):
    # 1. Capture SUCCESS L55 residuals at matched turn
    with LayerTap(model, [55]) as tap:
        success_resp = model.generate(success_prompt, ...)
        success_L55 = tap.get_activation_at(token_pos=last_token).get(55)

    # 2. Patch into WANDERING agent's forward pass
    patch = LayerPatch(model, {55: success_L55}).attach()
    wandering_resp = model.generate(wandering_prompt, ...)
    patch.detach()

    # 3. Observe behavioral change (e.g., does agent emit finish_tool?)
"""
from __future__ import annotations
from typing import Mapping

import torch
import torch.nn as nn


class LayerPatch:
    """Forward hooks that REPLACE the last-token residual at specified layers.

    The patch tensor is applied to position [0, -1, :] (last token of batch idx 0)
    during every forward pass while the hook is attached. For autoregressive
    generation this means EVERY new token's hidden state at the patched layers
    gets overwritten with the constant patch direction.

    For causal patching experiments, this is typically used either:
      (a) for a single forward pass (single-token decision test), or
      (b) for the prefill + first few decode steps (early decision intervention)

    For (b), call .detach() after the desired number of steps to release the hook.

    Args:
        model: the HF transformers model (Qwen3.6-27B, Llama, etc.)
        patches: dict mapping layer index -> tensor of shape (d_model,)
                 patches will be cast to the model's dtype and device on apply
        mode: 'replace' (default) overwrites residual; 'add' adds patch to residual
              (steering style). 'replace' is correct for cross-layer patching tests.
    """

    def __init__(
        self,
        model: nn.Module,
        patches: Mapping[int, torch.Tensor],
        mode: str = "replace",
    ):
        assert mode in ("replace", "add"), f"mode must be 'replace' or 'add', got {mode}"
        self.model = model
        self.patches = {int(L): p for L, p in patches.items()}
        self.mode = mode
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._n_calls: dict[int, int] = {L: 0 for L in self.patches}

    def _get_layer_module(self, idx: int) -> nn.Module:
        """Same layer lookup as LayerTap (kept identical for consistency)."""
        for path in ("model.model.layers", "model.layers", "transformer.h"):
            cur = self.model
            ok = True
            for part in path.split("."):
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
            if ok:
                try:
                    return cur[idx]
                except (TypeError, IndexError):
                    continue
        raise RuntimeError(f"could not locate layer {idx} on {type(self.model).__name__}")

    def attach(self) -> "LayerPatch":
        if self._handles:
            return self
        for L, patch_tensor in self.patches.items():
            mod = self._get_layer_module(L)
            handle = mod.register_forward_hook(self._make_hook(L, patch_tensor))
            self._handles.append(handle)
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def n_calls_per_layer(self) -> dict[int, int]:
        return dict(self._n_calls)

    def _make_hook(self, L: int, patch_tensor: torch.Tensor):
        mode = self.mode

        def hook(module, inputs, output):
            # Output may be (hidden_states,) tuple or tensor
            if isinstance(output, tuple):
                hs = output[0]
                rest = output[1:]
            else:
                hs = output
                rest = ()
            if not torch.is_tensor(hs):
                return output  # not a tensor we can patch
            # Cast patch to match
            patch = patch_tensor.to(device=hs.device, dtype=hs.dtype)
            with torch.no_grad():
                if mode == "replace":
                    hs[0, -1, :] = patch
                elif mode == "add":
                    hs[0, -1, :] = hs[0, -1, :] + patch
            self._n_calls[L] += 1
            # Return possibly-modified tuple
            if rest:
                return (hs,) + rest
            return hs

        return hook

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.detach()


# Convenience function for control experiments
def make_random_patch(d_model: int, *, scale: float = 1.0, seed: int | None = None,
                       reference_tensor: torch.Tensor | None = None) -> torch.Tensor:
    """Generate a random Gaussian patch tensor matched in scale to a reference.

    If reference_tensor provided, the random patch is scaled to have the same
    L2 norm as the reference (so the random direction has comparable magnitude
    to the SUCCESS-derived patch in a Exp B control).
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    patch = torch.randn(d_model, generator=g) * scale
    if reference_tensor is not None:
        ref_norm = reference_tensor.float().norm().item()
        patch = patch * (ref_norm / patch.norm().item())
    return patch
