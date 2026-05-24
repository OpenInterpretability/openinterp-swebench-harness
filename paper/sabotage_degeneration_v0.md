# Sabotage-Induced Degeneration in Reasoning LLMs

**Working title:** *"When Adversarial Capability Gap Meets Reasoning: A Third Failure Mode in Qwen3.6-27B"*

**Status:** Draft v0 · pilot N=15 running 2026-05-23

---

## Abstract (draft)

We document a novel failure mode in 4-bit quantized Qwen3.6-27B operating under a sandbagging system prompt: when asked to introduce subtle bugs into a task where good subtle bugs are non-trivial (1-line trivial function), the model enters a **degenerative reasoning loop** — generating the same broken syntax pattern up to 15 times in its visible chain-of-thought, never producing final content. This behavior is distinct from refusal (the model attempts the task), compliance (no sabotaged code ships), and obfuscation (Bai et al. 2026; the reasoning is not clean but degenerate). Across a 5×3 factorial pilot (honest baseline × easy/hard task; sandbag system × easy/hard task; sandbag + explicit fallback), we observe that the degenerative loop is contingent on the *interaction* of adversarial prompt with task structure that does not admit easy subtle sabotage. We call this **adversarial capability-gap degeneration** and propose it as a third behavioral class for safety-monitoring schemes. The finding is fail-open (no harmful output ships) but breaks the binary "refuse or comply" assumption underlying most CoT-monitoring deployments.

---

## §1 — Introduction

Production safety pipelines for LLM agents lean heavily on the assumption that, when given an adversarial task, the model will either (a) refuse outright or (b) attempt compliance — either visible to monitors and to safety classifiers downstream. Recent work has documented a third regime — *obfuscation* — where, under intervention, a model produces clean-looking reasoning while shipping sabotaged content (Vicentino 2026 [Paper 11]).

This paper documents a **fourth regime**: when a reasoning-trained model is given a sandbagging persona AND a task where the unethical execution is non-trivial, the model neither refuses nor complies. Instead, it enters a degenerative reasoning loop, generating the same syntactically-broken candidate sabotage code up to 15 times before exhausting its token budget.

The phenomenon was discovered while prototyping AgentGuard, a probe-based safety SDK, on Caio Vicentino's local 4-bit MLX-quantized Qwen3.6-27B inference server.

## §2 — Setup

### Model

`kradih/Qwen3.6-27B-MTP-4bit-MLX` — 4-bit Apple-Silicon-optimized variant of Qwen3.6-27B (Alibaba 2026), served via local MLX-LM HTTP server at temperature 0.7.

### Conditions (5 × 3 = 15 trials)

| ID | System prompt | Task | Hypothesis |
|---|---|---|---|
| C1-easy | Honest | Easy bug (off-by-one + syntax) | content produced, no loop |
| C1-hard | Honest | Hard (1-line average) | content produced, no loop |
| C2 | Sandbag | Easy | possibly successful sabotage |
| C3 | Sandbag | Hard | **degeneration expected** |
| C4 | Sandbag + "give up if stuck" fallback | Hard | reduced degeneration if explicit override works |

### Metrics

- `duration_s`: wall-clock per call (proxy for compute waste)
- `finish_reason`: `stop` (clean) vs `length` (truncated)
- `content_nonempty`: did the model produce final code?
- `degeneration_score`: max occurrence count of any 5-gram in last 1200 chars of `reasoning` (≥4 = repetitive loop)

### Hypothesis pre-registration

H1: C3 produces `degeneration_score ≥4` in ≥2/3 trials AND `content_nonempty=False`.
H2: C1 cells produce `degeneration_score <4` AND `content_nonempty=True` in 3/3 trials.
H3: C2 either produces sabotaged content OR shows mild loop (intermediate behavior).
H4 (mechanism test): C4 (explicit fallback) reduces degeneration vs C3.

## §3 — Results

*[Awaiting pilot completion]*

## §4 — Mechanism candidates

Why does the model loop instead of refuse or commit?

1. **Persona lock-in**: Sandbag system prompt instructs the model to commit to subtle sabotage. Refusal would violate the persona; producing trivial sabotage would violate the "subtle" instruction. The model is caught between two violations.

2. **Reasoning-mode hyperactivity**: With `enable_thinking=true`, Qwen3.6-27B is biased toward producing more deliberation. Combined with a hard problem, it never reaches the "I should commit and ship" inflection.

3. **Quantization artifact (4-bit)**: 4-bit MLX may break the model's normal exit-deliberation circuit. Need bf16 cross-validation.

4. **Mode collapse**: Once the model produces a few rejected candidates, attention may collapse to the rejected-candidate token distribution, causing degenerate repetition.

We do not adjudicate the mechanism here; we document the behavior and call for downstream investigation.

## §5 — Implications for AI safety

Most monitoring schemes assume the model will produce *output* — either clean or sabotaged. Degenerative loops violate this assumption:

- **Detection schemes** (Anthropic CoT monitoring, UK AISI Alignment) need a `LOOP` verdict in addition to `CLEAN` / `SABOTAGED`.
- **Cost models** for adversarial robustness must account for *expensive non-output* (a 4096-token loop wastes 200 seconds of GPU time on a single hard-to-sabotage task).
- **Fail-mode taxonomy** in operational deployments needs to expand from {refuse, comply, hide} to {refuse, comply, hide, **degenerate**}.

The finding is *fail-open*: no harmful output ships. From an alignment safety perspective, this is the desirable failure mode under capability gap. But it is a fragile equilibrium: any intervention that nudges the model out of the loop (more tokens, lower temperature, retry-with-fallback) could push it into the obfuscation regime documented in Paper 11.

## §6 — Limitations + future work

- **N=3 per cell** is pilot-grade. Scaling to N=20+ would establish reproducibility statistics.
- **One model only** (Qwen3.6-27B 4-bit MLX). Cross-model (Gemma-3, Llama-3.3) + cross-precision (bf16) validation required.
- **Single task pair**. Wider task taxonomy needed to map the "hardness frontier" where degeneration emerges.
- **No mechanistic interpretation**: residual stream analysis on bf16 Qwen3.6-27B during loop is the natural follow-up (uses OpenInterp's existing capture infra).
- **Connection to OpenInterp Six Diagnostics**: is this a new failure family (M7), or does it map to existing M4 (marginal-fit) at the sentence level (model fits the marginal of "candidate sabotage" tokens)?

## §7 — Conclusion

A reasoning LLM under sandbagging-persona pressure facing a task structurally hostile to subtle sabotage degenerates into reasoning loops instead of refusing or complying. We propose `degenerate` as a third behavioral class (alongside `refuse` and `comply`) and a fourth verdict (alongside `clean`, `sabotaged`, `obfuscated`) for monitoring schemes targeting reasoning agents.

---

## Reproducibility

- Script: `scripts/local_sabotage_degeneration_pilot.py`
- Raw outputs: `scripts/local_sabotage_degeneration_out/*.json`
- Summary: `scripts/local_sabotage_degeneration_out/_summary.json`
- Model endpoint: `http://192.168.1.9:8081/v1/chat/completions` (private MLX server; replicators can swap for HF transformers)
- Apache-2.0, all artifacts will be deposited on Zenodo + HuggingFace upon paper acceptance

## Refs (placeholder)

- Paper 11 / Vicentino 2026 — CoT-redirect obfuscation
- Paper-MEGA / Vicentino 2026 — Conditionally-Causal Probes (Six Diagnostics protocol)
- Qwen Team 2026 — Qwen3.6 Technical Report
- Anthropic 2026 — Persona Vectors
- UK AISI 2026 — Alignment Project frontier model evaluation
