# `agent-probe-guard` — SDK Plan

Drop-in pre-flight gate for code agents. Reads internal activations at the first reasoning token, predicts whether the agent's trace will succeed, returns a verdict before any tool call is spent.

**Status**: pending Phase 6 validation (gate at AUROC ≥ 0.85, N=99, permutation null p < 0.01). If validated, ship v0.1 within 7 days post-Phase-6c.

---

## 1. Product positioning

| Layer | What it does | Ours? |
|---|---|---|
| Agent loop | tool-call → observation → next-call | external (Cursor / Aider / OpenHands) |
| **Pre-flight gate** | **decide skip/proceed/escalate at first think token** | **`agent-probe-guard`** |
| Model | Qwen3.6-27B (initial release), Qwen3-coder, future others | external |

The SDK is **a one-function decision layer** that sits between the user's prompt and the agent's first action. It is not an agent and does not modify model weights.

**One-line value prop**: *"Read the model's internal capability signal at think onset. Skip the runs it knows it can't solve, before any tool budget is spent."*

---

## 2. Public API (v0.1)

```python
from agent_probe_guard import ProbeGuard

guard = ProbeGuard.load("openinterp/agent-probe-guard-qwen36-27b-swebench-v1")
v = guard.gate(model, tokenizer, problem_statement, repo_context=...)

# v.score          -> float in [0, 1], calibrated probability of solve
# v.decision       -> "proceed" | "escalate" | "skip"
# v.layer          -> int (e.g., 43)
# v.position       -> "think_start"
# v.threshold      -> 0.5 (configurable)
# v.latency_ms     -> ~200 (single forward up to first <think>)
# v.metadata       -> {probe_version, model_match_ok, ...}

if v.decision == "proceed":
    run_local_agent(...)
elif v.decision == "escalate":
    run_claude_sonnet_or_opus(...)  # user provides
else:
    return None  # skip: tell the user this is below model capability
```

**Decision policy** (configurable):

```python
guard = ProbeGuard.load(...).with_policy(
    skip_below=0.20,        # below this: refuse outright
    escalate_below=0.50,    # between: hand off to bigger model
    # above 0.50: proceed local
)
```

Three policies shipped: `conservative` (skip_below=0.10, escalate_below=0.40), `balanced` (default), `aggressive` (skip_below=0.30, escalate_below=0.60).

---

## 3. Implementation

### 3.1 Repo layout

```
agent-probe-guard/
├── src/agent_probe_guard/
│   ├── __init__.py
│   ├── probe.py          # joblib LR + StandardScaler + top-K feature index
│   ├── hook.py           # forward hook to capture L43/L55 think_start activation
│   ├── runtime.py        # ProbeGuard class + Verdict dataclass + gate()
│   ├── policy.py         # decision policies
│   ├── tokens.py         # find first <think> token id (Qwen3.6 chat-template helper)
│   └── cli.py            # `probe-guard wrap <cmd>` (wraps subprocess agent)
├── examples/
│   ├── basic_usage.py
│   ├── openhands_integration.py
│   └── aider_integration.py
├── tests/
│   ├── test_probe.py     # joblib round-trip, scaler reproducibility
│   ├── test_hook.py      # hook captures correct token + layer
│   └── test_e2e.py       # mock model + known activation → expected score
├── pyproject.toml
├── README.md
└── LICENSE (Apache-2.0)
```

### 3.2 Core flow

```python
def gate(self, model, tokenizer, problem: str, system: str | None = None) -> Verdict:
    # 1) Build prompt as the agent would: system + problem + "<think>"
    prompt = self._build_prompt(tokenizer, system, problem)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 2) Install hook on target layer (default L43)
    captured = {}
    def hook(_mod, _in, out):
        captured["act"] = out[0][:, -1, :].detach().cpu().float().numpy()
    h = model.model.layers[self.layer].register_forward_hook(hook)

    # 3) One forward pass — no generation. We only need the activation at the
    #    last input token (which by construction is the first <think> token).
    try:
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        h.remove()

    # 4) Apply probe: select top-K features, scale, LR predict_proba
    act = captured["act"][0]                           # (d_model,)
    sel = act[self.feature_idx]                         # (top_k,)
    sel = self.scaler.transform(sel.reshape(1, -1))     # standardize
    score = float(self.clf.predict_proba(sel)[0, 1])

    return Verdict(
        score=score,
        decision=self.policy.decide(score),
        layer=self.layer, position=self.position,
        threshold=self.policy.threshold,
        latency_ms=...,
        metadata={"probe_version": self.version, ...},
    )
```

**Why this works fast**: only ONE forward pass, NO autoregressive generation, NO KV cache. Stops at first think token (~50-200 input tokens). On Qwen3.6-27B Pro Blackwell: ~150-300ms. On A100 80GB: ~250-450ms. On 8-bit single 24GB: ~600ms.

### 3.3 Probe artifact format

Published as a single HF dataset repo `openinterp/agent-probe-guard-qwen36-27b-swebench-v1` containing:

```
probe_v1.joblib              # sklearn Pipeline(StandardScaler, LR)
feature_idx_v1.npy           # top-50 indices into the 5120-dim residual stream
config.json                  # layer=43, position=think_start, model="Qwen/Qwen3.6-27B"
calibration.json             # threshold candidates + their precision/recall on N=99
permutation_null.json        # null distribution for sanity at runtime
README.md
```

Loader does SHA256 verification + `model.config._name_or_path` match check. If user runs the probe on a different model, `Verdict.metadata.model_match_ok=False` and a warning fires.

### 3.4 Hook safety

- Hook removed in `finally` block (no leak even on exception)
- `use_cache=False` prevents KV-cache contamination of subsequent agent calls
- `register_forward_hook` (not pre-hook) so we capture the OUTPUT of layer 43
- `detach().cpu()` immediately to free GPU memory

### 3.5 CLI wrapper

```bash
probe-guard wrap --model Qwen/Qwen3.6-27B --probe v1 --policy balanced -- \
  aider --message "fix the failing test in tests/test_foo.py"
```

CLI:
1. Spawns subprocess for the wrapped agent command
2. Intercepts stdin/stdout
3. On first user prompt → runs gate
4. If decision=skip: prints reason, exits 1
5. If decision=escalate: prints handoff message, exits 2
6. If decision=proceed: passes prompt to subprocess unchanged

---

## 4. Phase plan

### Phase A — v0.1 ship (post Phase 6c, target +5 days)

**Day 1** (post-Phase-6c results):
- Confirm AUROC ≥ 0.85 + permutation p < 0.01 + random-feature baseline gap > 0.15
- Lock probe weights, calibration, layer choice
- Upload to HF as `openinterp/agent-probe-guard-qwen36-27b-swebench-v1`

**Day 2-3**: SDK core (probe.py, hook.py, runtime.py, policy.py)
- Unit tests with mock model + frozen activations
- E2E test on real Qwen3.6-27B with 5 held-out problems (assert score correlates with verdict)

**Day 4**: CLI wrapper + 2 example integrations (OpenHands, Aider)

**Day 5**: README + landing page snippet at openinterp.org/products/agent-probe-guard
- PyPI release `agent-probe-guard==0.1.0`
- HF artifact public
- X/Twitter post on @0xCVYH (research-series style: "we trained a probe that reads if the agent will solve before it tries")

### Phase B — Cross-domain (week 2-3)

- Retrain probe on **HumanEval** + **LiveCodeBench** + **MBPP** with TRUE labels (no Docker, just unit-test exec)
- Decide: **single universal probe** (all domains in training) vs **per-domain probes** (3 separate artifacts)
- Publish both, let users choose

### Phase C — Cross-model (week 4-6)

For each of {Qwen3-coder-32B, Llama-4-Code-70B, DeepSeek-V3.5-Coder, GPT-OSS-120B}:
1. Run Phase 1 protocol: 20 stratified SWE-bench Pro problems
2. Train probe at L_model × 0.6 think_start (proportional layer pick)
3. Validate at N=50
4. Publish artifact `openinterp/agent-probe-guard-{model}-v1`

Goal: 4-5 model-specific artifacts in the HF org. SDK auto-loads the right one based on `model.config._name_or_path`.

### Phase D — Production integrations (month 2+)

- **OpenHands** plugin: drop-in middleware in their config
- **Aider** PR: add `--probe-guard` flag
- **Cursor** extension (long-term): show probe score in editor before agent runs

---

## 5. Telemetry + iteration

Optional opt-in telemetry (env: `AGENT_PROBE_GUARD_TELEMETRY=1`):

```jsonl
{"ts": "...", "probe_version": "v1", "model": "Qwen3.6-27B", "score": 0.42, "decision": "escalate", "actual_outcome_if_known": null}
```

User can later upload `actual_outcome_if_known` = solve/fail to feed back into a continuous calibration dataset.

This becomes the dataset for v0.2 (which has 100× more labeled traces than the original N=99 we shipped on).

---

## 6. Differentiation vs adjacent work

| Tool | Layer | Decision input | Latency |
|---|---|---|---|
| **agent-probe-guard** | mech interp probe | internal activation | ~200ms |
| Self-consistency / vote | output sampling | n samples → majority | n × full trace |
| Verifier model | secondary LLM | full trace + tests | full trace + judge |
| Confidence prompt | model self-report | "are you sure?" | one extra turn |
| Tool-error early stop | execution trace | failures observed | first few tool calls |

We're the only one that fires **before any compute is spent on the trace**. All other approaches need at least one full attempt (or sample) first.

---

## 7. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Probe overfit to N=99 traces | Cross-benchmark Phase B before claiming generality. Always ship calibration.json with held-out numbers. |
| Threshold too aggressive → unhappy users | Default policy = `balanced` (skip_below=0.20). Conservative as opt-in. |
| Model version drift (Qwen3.6 → Qwen3.7) | Probe artifact pins exact `_name_or_path`. Loader warns if mismatch. v0.2 retrains. |
| User runs wrong layer | Probe metadata pins `layer=43`. Loader extracts from artifact, not user-config. |
| Hook leak on crash | `try/finally` mandatory. Tested with synthetic exceptions. |
| Closed-source agents (Cursor) can't import | CLI wrapper covers this — wraps subprocess. |
| Probe direction tested adversarially fails | Honest disclosure: probe is *correlative*, not *causal* (Phase 3 steering counter-experiment). README flags this. |

---

## 8. What this does NOT promise

- **Does not improve the agent.** Detection only.
- **Does not generalize to non-coding tasks** until Phase B/C.
- **Does not replace evals.** Cheap gating layer, not a guarantee.
- **Does not work on closed models** (no hook access). API-only models out of scope until provider exposes activations.
- **Does not fix bad prompts.** Garbage in → low score, but not because the model can't, just because the prompt is unparseable.

---

## 9. Success metric for v0.1

| Metric | Target | How measured |
|---|---|---|
| Probe AUROC on N=99 holdout | ≥ 0.85 | Phase 6c |
| Permutation null p-value | < 0.01 | Phase 6c |
| Random-feature gap | ≥ 0.15 AUROC | Phase 6c |
| Latency on Qwen3.6-27B | < 500ms p50 | benchmark.py in tests/ |
| Time to integrate (3 lines) | < 10 min | user study with 3 OSS contributors |
| First external user shipping it | within 30 days | OpenHands or Aider PR |

If first 3 fail → no v0.1 ship. Reframe paper as honest negative.
If first 3 pass and last 3 fail → revisit positioning (maybe internal tool, not public SDK).

---

## 10. Open questions to resolve at Phase-6c verdict time

1. **Threshold calibration source**: should `balanced` threshold come from the N=99 Youden's J, or from a precision-fixed criterion (e.g., precision ≥ 0.90 → recall = X)?
2. **Single layer (L43) or ensemble (L43 + L55)**: ensemble adds 2× forward cost for marginal AUROC gain. Probably ship single-layer.
3. **Probe head**: keep sklearn LR (interpretable) vs torch.nn.Linear (faster on GPU)? Latency check at v0.1 — if LR is < 5ms it's irrelevant.
4. **Pricing / monetization**: free SDK, paid hosted endpoint? Or fully OSS? Default: fully OSS Apache-2.0 + hosted as a future option if traction warrants.
