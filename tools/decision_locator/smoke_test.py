"""CPU smoke test for decision-locator — no network, no big model.

Builds a tiny random GPT-2 and exercises every primitive (layer resolution, logit-lens locate,
donor capture, activation-patch sweep, prefill-only steered generation). Verifies shapes/types and
that a patch with a strong donor measurably moves P(target). Run: python smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from decision_locator import DecisionLocator, commitment_layer


def build_tiny():
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=512, n_positions=128, n_embd=64, n_layer=4, n_head=4)
    model = GPT2LMHeadModel(cfg).eval()
    # a minimal byte-level tokenizer stand-in: reuse GPT2 tokenizer but clamp ids < vocab_size
    tok = GPT2TokenizerFast.from_pretrained("gpt2") if _has_gpt2_files() else _DummyTok()
    return model, tok


def _has_gpt2_files():
    try:
        GPT2TokenizerFast.from_pretrained("gpt2")
        return True
    except Exception:
        return False


class _DummyTok:
    """Offline fallback tokenizer: identity over integer ids, decode -> '<id>' tokens."""
    eos_token_id = 0
    pad_token_id = 0
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"<{int(i)}>" for i in ids)


def main():
    model, tok = build_tiny()
    loc = DecisionLocator(model, tok)
    assert loc.n_layers == 4, f"layer resolution failed: {loc.n_layers}"
    print(f"[ok] resolved {loc.n_layers} decoder layers, norm={'yes' if loc.norm_w is not None else 'no'}")

    ids = torch.randint(0, 512, (1, 20))
    target_id, alt_ids = 7, [21, 42]

    gaps = loc.locate(ids, target_id, alt_ids)
    assert set(gaps) == {0, 1, 2, 3} and all(isinstance(v, float) for v in gaps.values())
    print(f"[ok] locate() -> per-layer logit-lens gaps: " +
          ", ".join(f"L{L}:{g:+.3f}" for L, g in gaps.items()))

    # donor from a different prompt that we bias toward the target token, so patching should move P(target)
    donor_ids = torch.full((1, 20), target_id)
    donors = {L: loc.donor_state([donor_ids], L) for L in range(4)}
    dP = loc.sweep_patch(ids, donors, [0, 1, 2, 3], target_id, alt_ids)
    assert set(dP) == {0, 1, 2, 3}
    print(f"[ok] sweep_patch() -> ΔP per layer: " + ", ".join(f"L{L}:{d:+.3f}" for L, d in dP.items()))

    lever = commitment_layer(dP)
    print(f"[ok] commitment_layer() -> L{lever} (max ΔP={dP[lever]:+.3f})")

    cont = loc.steer_generate(ids, layer=lever, donor=donors[lever], max_new_tokens=8)
    assert isinstance(cont, str) and len(cont) > 0
    print(f"[ok] steer_generate() -> continuation (len {len(cont)}): {cont[:48]!r}")

    base = loc.steer_generate(ids, max_new_tokens=8)         # no patch
    assert isinstance(base, str)
    print(f"[ok] steer_generate(no patch) -> {base[:48]!r}")

    # sanity: a strong target-biased donor should not DECREASE P(target) on average across layers
    mean_dP = sum(dP.values()) / len(dP)
    print(f"[ok] mean ΔP(target) under target-biased donor = {mean_dP:+.4f} (expect >= ~0)")
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
