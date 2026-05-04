from __future__ import annotations
from typing import Iterable

import torch
import torch.nn as nn


class LayerTap:
    """Forward hooks on selected transformer layers that buffer the last-token activation
    of every forward pass during a generation. Keeps activations on CPU in bf16.

    During autoregressive generation:
      - Forward pass at step k captures hs[0, -1, :] from each tapped layer.
      - For prompt-prefill, this is the activation at the last prompt token.
      - For decoding step j (j-th generated token), this is the activation that produced
        token j. So buffer[j] is the activation that decided generated token j.

    Usage:
        tap = LayerTap(model, layers=[11, 23, 31, 43, 55]).attach()
        tap.reset()
        # ... model.generate(...) ...
        snap = tap.get_activation_at(token_pos)  # {layer: tensor(d_model,)} for that gen step
        tap.detach()
    """

    def __init__(self, model: nn.Module, layers: Iterable[int]):
        self.model = model
        self.layers = tuple(int(L) for L in layers)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._buffers: dict[int, list[torch.Tensor]] = {L: [] for L in self.layers}

    def _get_layer_module(self, idx: int) -> nn.Module:
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

    def attach(self) -> "LayerTap":
        if self._handles:
            return self
        for L in self.layers:
            mod = self._get_layer_module(L)
            handle = mod.register_forward_hook(self._make_hook(L))
            self._handles.append(handle)
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, L: int):
        buf = self._buffers[L]
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hs):
                return
            with torch.no_grad():
                last = hs[0, -1, :].detach().to(dtype=torch.bfloat16, device="cpu")
            buf.append(last)
        return hook

    def reset(self) -> None:
        for L in self.layers:
            self._buffers[L].clear()

    def n_steps(self) -> int:
        if not self._buffers:
            return 0
        return min(len(self._buffers[L]) for L in self.layers)

    def get_activation_at(self, token_pos: int) -> dict[int, torch.Tensor]:
        n = self.n_steps()
        if token_pos < 0 or token_pos >= n:
            raise IndexError(f"token_pos {token_pos} out of range [0, {n})")
        return {L: self._buffers[L][token_pos].clone() for L in self.layers}

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.detach()
