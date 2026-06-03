"""decision-locator — find and steer the layer where a tool-calling agent commits a decision.

Method behind the WANDERING arc's first positive (paper #6, "The Lever Is Late"): for a long-horizon
agent, the control surface of a discrete action decision (e.g. emit ``finish`` vs keep working) is NOT
the mid-layer representation that *predicts* it — it is a LATE action-commitment block. This tool
generalizes the three primitives that localized and steered that decision, on any open-weight HF model:

  1. locate()       — logit-lens gap to a target action token across layers  (where is it readable?)
  2. sweep_patch()  — activation-patch a donor state per layer                (where is it writable?)
  3. steer_generate() — patch the commitment layer at the decision position, then decode freely
                        (does steering produce a REAL action emission, not just a probability bump?)

Model-agnostic: resolves decoder layers / final norm for common HF architectures (Qwen, Llama, GPT-2,
GPT-NeoX, Mistral, ...). Pure PyTorch + transformers, no other deps. See README.md.
"""
from __future__ import annotations
import math
import torch

_LAYER_PATHS = (
    "model.language_model.layers", "language_model.layers",
    "model.model.layers", "model.layers", "transformer.h", "gpt_neox.layers",
)
_NORM_PATHS = (
    "model.language_model.norm", "model.model.norm", "model.norm",
    "transformer.ln_f", "gpt_neox.final_layer_norm",
)


def _dig(root, dotted):
    cur = root
    for p in dotted.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur


def _resolve_layers(model):
    for path in _LAYER_PATHS:
        mod = _dig(model, path)
        if mod is not None:
            return list(mod)
    raise RuntimeError(
        "Could not auto-locate decoder layers for this model; pass layer_modules=list(...) explicitly."
    )


def _resolve_norm(model):
    for path in _NORM_PATHS:
        mod = _dig(model, path)
        if mod is not None and hasattr(mod, "weight"):
            return mod.weight.detach().float()
    return None


class DecisionLocator:
    """Wraps an HF causal LM + tokenizer with decision-localization / steering primitives.

    A *decision point* is a prompt whose final token leaves the model poised to choose an action
    (for a tool-calling agent: the prompt ends right before the function-name token). You supply the
    target action's token id and the alternative action token ids; P(target) is the softmax over just
    those action tokens at the last position — the agent's internal vote for that action.
    """

    def __init__(self, model, tokenizer, layer_modules=None):
        self.model = model.eval()
        self.tok = tokenizer
        self.layers = layer_modules if layer_modules is not None else _resolve_layers(model)
        self.n_layers = len(self.layers)
        self.norm_w = _resolve_norm(model)
        self.WU = model.get_output_embeddings().weight.detach()
        self.device = next(model.parameters()).device

    # -- probability of the target action over the action set ------------------------------------
    def p_target(self, logits_row, target_id, alt_ids):
        ids = [target_id] + list(alt_ids)
        p = torch.softmax(logits_row[ids].float(), -1)
        return float(p[0])

    # -- capture last-position residual at chosen layers -----------------------------------------
    def _capture(self, input_ids, layers):
        caps, handles = {}, []
        def mk(i):
            def h(m, inp, out):
                caps[i] = (out[0] if isinstance(out, tuple) else out)[:, -1:, :].detach()
            return h
        for i in layers:
            handles.append(self.layers[i].register_forward_hook(mk(i)))
        try:
            with torch.no_grad():
                out = self.model(input_ids=input_ids.to(self.device), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return out.logits, caps

    # -- 1. LOCATE: logit-lens gap to target across layers ---------------------------------------
    def logit_lens(self, resid, target_id, alt_ids):
        """Project a residual through the final norm + unembedding; return (target - mean(alts))."""
        x = resid.float()
        x = x / (x.norm() / math.sqrt(x.numel()) + 1e-6)            # RMSNorm-style scale
        if self.norm_w is not None:
            x = x * self.norm_w
        lg = x @ self.WU.float().T
        return float(lg[target_id] - sum(lg[a] for a in alt_ids) / len(alt_ids))

    def locate(self, input_ids, target_id, alt_ids, layers=None):
        """Where does the action decision become readable? Returns {layer: logit-lens gap}."""
        layers = list(range(self.n_layers)) if layers is None else layers
        _, caps = self._capture(input_ids, layers)
        return {L: self.logit_lens(caps[L][0, 0], target_id, alt_ids) for L in layers}

    # -- donor states (for patching) -------------------------------------------------------------
    def donor_state(self, input_ids_list, layer):
        """Mean last-position residual at `layer` over a set of donor prompts (e.g. SUCCESS runs).

        Tip (paper #6): a *task-matched* single donor steers far better than a class mean. Pass a
        one-element list of a same-task donor instead of the full class for a stronger lever.
        """
        acc = []
        for ids in input_ids_list:
            _, caps = self._capture(ids, [layer])
            acc.append(caps[layer][0, 0].float())
        return torch.stack(acc).mean(0)

    def _patch_hook(self, donor, prefill_only):
        def h(m, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if prefill_only and hs.shape[1] == 1:      # skip incremental decode steps
                return out
            hs = hs.clone()
            hs[:, -1, :] = donor.to(hs.dtype)
            return (hs, *out[1:]) if isinstance(out, tuple) else hs
        return h

    # -- 2. SWEEP_PATCH: where is the decision writable? -----------------------------------------
    def patch_delta(self, input_ids, layer, donor, target_id, alt_ids):
        """ΔP(target) from replacing the last-position residual at `layer` with `donor`."""
        base_lg, _ = self._capture(input_ids, [])
        base = self.p_target(base_lg[0, -1], target_id, alt_ids)
        hh = self.layers[layer].register_forward_hook(self._patch_hook(donor, prefill_only=False))
        try:
            with torch.no_grad():
                lg = self.model(input_ids=input_ids.to(self.device), use_cache=False).logits
        finally:
            hh.remove()
        return self.p_target(lg[0, -1], target_id, alt_ids) - base

    def sweep_patch(self, input_ids, donor_by_layer, layers, target_id, alt_ids):
        """ΔP(target) per layer when injecting donor_by_layer[L]. The layer where this jumps (and a
        control donor does not) is the commitment layer — the lever."""
        return {L: self.patch_delta(input_ids, L, donor_by_layer[L], target_id, alt_ids) for L in layers}

    # -- 3. STEER_GENERATE: does the lever produce a real emission? ------------------------------
    def steer_generate(self, input_ids, layer=None, donor=None, max_new_tokens=24, **gen_kwargs):
        """Patch `layer` at the decision position only (prefill), then greedily decode. Returns the
        continuation string — check whether it actually emits the target action (the high bar; a
        one-token probability bump does not imply a real emission)."""
        am = torch.ones_like(input_ids)
        hh = None
        if layer is not None and donor is not None:
            hh = self.layers[layer].register_forward_hook(self._patch_hook(donor, prefill_only=True))
        try:
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=input_ids.to(self.device), attention_mask=am.to(self.device),
                    max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=(self.tok.eos_token_id or self.tok.pad_token_id), **gen_kwargs,
                )
        finally:
            if hh is not None:
                hh.remove()
        return self.tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=False)


def commitment_layer(gap_pos, gap_neg=None, layers=None):
    """Convenience: given per-layer logit-lens gaps for a positive class (and optional negative/null),
    return the layer with the largest positive separation — the candidate commitment layer."""
    layers = sorted(gap_pos) if layers is None else layers
    def sep(L):
        return gap_pos[L] - (gap_neg[L] if gap_neg else 0.0)
    return max(layers, key=sep)
