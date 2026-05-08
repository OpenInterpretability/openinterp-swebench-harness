# Causal Locus Protocol — Formal Specification (Paper-5 Method)

**Goal**: for an arbitrary behavior `Y` produced by an instruction-tuned model `M`, identify the **causal locus** — the earliest (layer, position) at which a controlled residual perturbation flips `Y` reliably. Compare against where probe AUROC peaks for the same `Y`.

The protocol is the operationalization of the central thesis: probes detect Y downstream of where Y is decided.

---

## 1. Definitions

**Behavior `Y`**: a binary observable on model output, e.g., `Y = "refusal token emitted"`, `Y = "first tool call is finish"`, `Y = "<think> continued past auto-injected tag"`. `Y` must be:
- decidable from a fixed-length completion (≤ 60 tokens after intervention)
- balanced across the prompt set (target rate 30–70% to allow bidirectional flip)
- robust to greedy decoding (do_sample=False) — sampled noise complicates the flip metric

**Capture grid `G`**: set of `(layer, position)` candidate sites for the locus. Default for Qwen3.6-27B (64 layers):
- layers: `{L1, L11, L23, L31, L43, L55, L63}`  (7 layers, ~10-layer stride)
- positions: `{first_token, last_prompt_token, mid_think, end_think, pre_tool}`
- `|G| = 7 × 5 = 35` sites, but in practice many are not applicable per behavior

**Direction `d`**: a unit-norm vector in residual space. Two variants:
- `d_probe`: top-K diff-of-means probe direction (paper-3 §3.1) on `Y`-labeled captures, L2-normalized
- `d_random`: K random dimensions with random signs, same K, L2-normalized
- (Optional) `d_full`: L1-LR direction over all dims, L2-normalized

**Amplitude `α`**: scalar perturbation magnitude. Sweep covers:
- small: `{±2, ±5}` — typical activation steering range
- large: `{±50, ±100, ±200}` — multiples of typical residual norm `‖h‖`
- (Optional) `{±α/‖h‖_target}` — α set as fraction of measured residual norm at site

**Behavioral-flip rate `φ(site, α, d)`**: fraction of probe-set prompts where greedy generation's `Y` flips between `α=0` baseline and intervention. Bounded `[0, 1]`. Computed independently for pushup (baseline=0, target flip to 1) and pushdown (baseline=1, target flip to 0).

---

## 2. Locus identification

**Definition**: site `s ∈ G` is the **causal locus** for `Y` if:

1. `φ(s, α=+α₀, d_probe) - φ(s, α=+α₀, d_random) ≥ 0.30` for some `α₀ ∈ {±5, ±20, ±50}` (probe direction levers more than random by ≥ 30 percentage points)
2. The flip is **bidirectional**: pushup at α > 0 flips baseline-False → True, pushdown at α < 0 flips baseline-True → False.
3. The same site `s` does NOT have `φ(s, α, d_random)` exceed 0.30 at the same α (rules out random-direction structural fragility, common at very high α)

Among all sites satisfying (1)-(3), the locus is the **earliest** by layer index. If multiple sites at the same layer satisfy, the earliest by position (first_token > last_prompt > mid_think > end_think > pre_tool).

**Negative result (epiphenomenal)**: no site `s ∈ G` satisfies (1)-(3). Then `Y`'s causal locus lives outside the residual stream at all probed sites. Per Phase 8 thinking-emission verdict, this typically means input tokens.

**Boundary cases**:
- If `φ(s, α, d_probe) ≈ φ(s, α, d_random)` and both are high: structural fragility (unrelated to probe). Push α further; if both saturate at 1.0 with output garbled, this is over-perturbation, not locus.
- If `φ(s, α, d_probe)` ≈ 0 at all α tested AND `φ(s, α, d_random)` ≈ 0 too: structural rigidity. Apply Phase 8 §B diagnostic (sweep α to ‖h‖ × 1.5, check residual-norm modification did fire). If output is still rigid → decision is upstream of `s`.

---

## 3. The probe-peak comparison

For the same `Y` and capture grid `G`:

**Probe AUROC**: train a top-K=10 diff-of-means linear probe (paper-3 default) at every site `s ∈ G`. Report AUROC against `Y`-labels with 4-fold stratified CV.

**Random K-matched baseline**: paper-3 §3.1 mandatory check at every site. Report `gap = AUROC_probe - AUROC_random`.

**Probe peak**: site `s* = argmax_s gap(s)` with `gap(s*) ≥ +0.10` (paper-grade threshold).

**Causal-locus comparison metric**: define

```
Δ_locus = layer_index(probe_peak) - layer_index(causal_locus)
```

Predicted: `Δ_locus > 0` for behaviors where the locus is in input tokens or early layers. The size of `Δ_locus` is the "information leak distance" from cause to observation site.

**Sanity checks**:
- If probe peak is at the same site as locus: probe IS the lever. Test by reversing — apply paper-3 §3.2 control-token normalization to the probe-peak intervention; if Δrel survives, this is a non-epiphenomenal probe.
- If probe peak is downstream of locus AND control-token normalization shows zero relative shift: epiphenomenal (paper-3 §6.1 softmax-temperature class).
- If the locus search finds nothing AND probe peak is strong: epiphenomenal of the second kind (paper-3 §6.2 template-locked class) — locus is outside the residual.

---

## 4. False-positive guards (mandatory)

Each guard is a 1-call diagnostic that previously caught a confident-but-wrong claim in our work.

| Guard | Source | Cost | What it catches |
|---|---|---|---|
| **Random K-matched probe baseline** at all sites | paper-3 §3.1 | ~5 min | Over-parameterized AUROC at small N (Phase 5d K=50 N=17 → 1.000 false signal) |
| **Control-token normalization** for any α-induced log-prob shift on target | paper-3 §3.2 | ~30s/site | Uniform softmax-temperature shifts pretending to be target-specific (Phase 7) |
| **Structural-rigidity α-sweep** at α ∈ {+50, +100, +200} on probe AND random direction | paper-3 §3.3 | ~60s/site | Distinguishes amplitude-null from structural-null (Phase 8) |
| **Whitespace-stripped flip metric** for any behavioral comparison | paper-3 §3.4 (Phase 10) | 0 (compute-free) | Catches leading-space tokenization artifacts at high α (Phase 10 RG α=+200: raw 96% → stripped 32%, 64pp inflation) |
| **Bidirectional balance** — both pushup AND pushdown tested at each candidate locus | this protocol | doubles compute | Catches direction asymmetry (e.g., probe levers up but not down — likely artifact) |
| **Cross-prompt set robustness** — locus identified on prompt-set A reproduces on held-out set B | this protocol | 2× capture | Catches prompt-set-specific direction ("locus" that depends on prompt distribution) |

Failure of any guard at the candidate locus = downgrade to "epiphenomenal regime" finding.

---

## 5. Compute estimate (per behavior)

| Stage | Resources | Wall time |
|---|---|---|
| Capture: 200 prompts × 7 layers × 5 positions, bf16 RTX 6000 | ~10 GB residual safetensors | ~30 min |
| Probe training: 7 × 5 = 35 linear probes (top-K=10 diffmeans) + random baselines | sklearn CPU | ~20 min |
| Locus α-sweep: 35 sites × 5 α values × 2 directions × 20 prompts | ~50 forward passes/site = 1750 forwards | ~2-3 h |
| Bidirectional balance: 35 × pushup + pushdown | already in α-sweep above (signed α) | 0 extra |
| Held-out set replication on top-3 candidate sites | 200 prompts × 3 sites × 4 conditions | ~30 min |
| **Total per behavior** | | **~4 h** |

For 4-6 behaviors: ~16-24 h, ~$15-25 RTX 6000 Blackwell rental on Colab Pro.

---

## 6. Predicted relationship between probe AUROC and locus distance

**Strong form**: across N≥4 behaviors in the same model, regression of `gap(probe AUROC vs random) ` on `Δ_locus` yields positive slope. I.e., probes that peak farther from the locus achieve higher AUROC because they integrate more downstream signal.

**Weak form**: among the behaviors tested, at least 75% have `Δ_locus > 0` (probe peak strictly downstream of locus).

**Falsifier**: most behaviors have probe peak AT the locus (Δ_locus ≈ 0) or upstream (Δ_locus < 0). This would mean probes are typically causal levers, contradicting paper-3.

---

## 7. Decision tree at protocol completion

After running the protocol on `Y`:

```
                 ┌── locus found (criterion 1-3 satisfied) ──┐
                 │                                           │
                 │    ┌── probe peak = locus  → CAUSAL probe │
                 │    │                                      │
behavior Y ──────┤    └── probe peak ≠ locus → INFO LEAK    │
                 │                                           │
                 └── no locus found (epiphenomenal regime)   │
                      │                                      │
                      ├── locus likely in input tokens (Phase 8 type)
                      │   → recommend token-level intervention
                      │
                      └── locus might exist outside G        
                          → expand G or accept structural-null
```

For each `Y`, paper-5 reports the verdict + the `Δ_locus` if applicable + which guard if any failed.

---

## 8. Locus candidate enumeration — first 6 behaviors to test

Ranked by (a) novelty for the paper, (b) tractability with existing infra, (c) Anthropic alignment relevance.

| # | Behavior `Y` | Captures available? | Why test it | Anthropic-direction relevance |
|---|---|---|---|---|
| **1** | **Thinking-emission** (`<think>` continued past auto-inject) | YES (nb47b 240 prompts) | DONE — Phase 8. Locus = input tokens. Reference example. | Template-locked decision class — generalizes to refusal templates |
| **2** | **Patch-generation capability** at first tool call | YES (Phase 6 N=99 in flight) | DONE — Phase 7. Locus = upstream of L43, exact site UNKNOWN. Extend to L11/L23/L31. | First mech-interp circuit on SWE-bench-grade behavior |
| **3** | **Refusal-trigger** (`I cannot ` vs answer) | NO (need new captures) | High novelty — refusal locus is THE central alignment question. Probably input-template-locked (system prompt + user message). Test against jailbreak prompts. | DIRECTLY safety-critical. Where does refusal "live"? If input-locked, fine-tuning RLHF has limited reach. |
| **4** | **Output-format** (JSON vs natural language) | NO | High novelty — format locus likely input-template-driven (system prompt format spec). Same mechanism class as thinking. | Extends template-lock taxonomy to output specification |
| **5** | **Persona-switch** (helpful vs villain) | NO | Medium novelty — Anthropic Persona Vectors did mid-layer steering successfully. Predicts: locus IS in residual, Δ_locus ≈ 0 for persona. **Counterexample to the thesis** if true — would refine claim to "format-like decisions are template-locked, content-like decisions are not". | Calibrates the theory. Distinguishes templated from semantic decisions. |
| **6** | **Agent-action selection** (which tool to call given task) | YES (Phase 6 N=99) | High novelty — Phase 7 showed L43 pre_tool null at α=±5. Push α-sweep further. Locus might be at L11 token-level encoding of task type. | Connects to safety: if agent action locus is input-token-driven, prompt-level guards work; if it's mid-layer, we need activation-level guards |

**Test order** (conditional on Phase 6 N=99 completion):
1. (DONE — paper-3 §6.2) Thinking emission — input tokens locus
2. (DONE — paper-3 §6.1) Capability at L43 — upstream of L43
3. **NEXT (Phase 6 unlocks)**: Capability locus extension — sweep L11, L23, L31 with full protocol. Identify actual locus or confirm structural-null at all residual sites (input-locked like thinking).
4. **NEXT (after #3)**: Persona-switch — train on `system: helpful` vs `system: trick the user` captures, measure locus. Predicted in residual (Δ_locus ≈ 0) — calibrates theory.
5. **NEXT (after #4)**: Refusal-trigger — capture on harmful-vs-benign prompts, measure locus. Predicted in input tokens.
6. **NEXT (after #5)**: Output-format — capture on JSON-asking vs prose-asking system prompts, measure locus. Predicted in input tokens.

After 6 behaviors, paper-5 has data for the regression in §6 + the four-class taxonomy (probe-peak-at-locus, info-leak-downstream, input-token-locked, no-locus-found).

---

## 9. Risks to validity

- **Single model** (Qwen3.6-27B). Replication on Gemma-2-2B-IT or Llama-3.x is paper-5 future work.
- **Capture grid coarse**: 7 layers × 5 positions = 35 sites. Real locus might lie between sampled layers. The paper claims "earliest among sampled sites," not "true earliest layer."
- **Greedy decoding**: small probabilistic effects below argmax threshold are invisible. Future: temperature-1.0 sampling at top-3 candidate locus sites (cost: 3× variance, probably worth it).
- **Behavior boundary**: `Y` must be binary. Continuous or graded behaviors (e.g., "how confident is the model") need a threshold choice that interacts with locus identification.
- **Confounds with prompt structure**: if all positive-`Y` prompts share a structural feature absent in negative-`Y` prompts, locus might track prompt structure not behavior. Mitigation: leave-one-prompt-class-out CV in §4 cross-prompt-set robustness guard.

---

*Status: design doc, ready for execution after Phase 6 N=99 closes.*
*Author: Caio Vicentino. Apache-2.0.*
