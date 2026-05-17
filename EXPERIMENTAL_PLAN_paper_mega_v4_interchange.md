# Paper-MEGA v4 — Interchange-Intervention Experimental Plan

**Goal:** Validate or falsify the five operational constraints (C1–C5) of paper-MEGA
*Conditionally-Causal Probes* via Geiger-style interchange interventions. Closes
the §4.7.1 "future work" gap in v3.4 and promotes the framework from empirical
regularity (correlative observation across 11 probes) to causal-abstraction
claim (each probe direction is shown to either preserve or break interchange
validity, predicted in advance by its C1–C5 configuration).

**Estimated total compute:** 10–20 GPU-hr H100 ≈ $25–55 on Runpod/Modal.
**Estimated wall clock:** 2–3 sessions over 1 week.
**Authoring repo:** `OpenInterpretability/openinterp-swebench-harness`.
**Target output:** Paper-MEGA v4 (§4.7 reframed from conjecture to result) + HF
dataset extension + verify script extension (target 35–40 PASS).

---

## 1. Theoretical motivation

The five-axis operational-constraints framework (paper-MEGA §4) is empirically
motivated by 11 probe verdicts on Qwen3.6-27B but theoretically incomplete:
**we have not shown that any single probe direction stands in a formal
causal-abstraction relation to the high-level behavioral variable it predicts.**
The α-sweep + control-token + onset-timing diagnostics catch *negative* claims
(distinguish R4 / R5 / amplitude-null from genuine lever) but cannot validate
*positive* abstraction claims. A high-AUROC + lever-positive probe (R1, R2, R3)
could still be "approximately correlated" with the high-level variable without
being its abstraction.

The interchange-intervention protocol (Geiger et al. 2021; 2023; 2025) is the
standard operationalization of causal abstraction. Two variables stand in
causal-abstraction relation iff intervening on the low-level variable (the
probe direction) reproduces the behavioral effect of intervening on the
high-level variable. The protocol:

```
Run P+ → output Y+ (baseline)
Run P− → output Y− (counterfactual baseline)

Interchange test of probe direction v at (layer L, position p):
  h_orig    = residual from P+ at (L, p)
  h_counter = residual from P− at (L, p)
  α_orig    = ⟨h_orig, v⟩
  α_counter = ⟨h_counter, v⟩
  h_new     = h_orig + (α_counter − α_orig) · v
  Continue forward from h_new → output Y_interchange

If Y_interchange ≈ Y− → probe direction CARRIES the abstraction
If Y_interchange ≈ Y+ → probe direction is independent of behavior
If Y_interchange random → probe direction is noise (R4/R5)
```

**Key methodological strength:** the intervention magnitude is *data-driven*
(`α_counter − α_orig` is what would naturally occur), not a hyperparameter to
be swept. Negative results are therefore interpretable: failure to shift Y
under interchange means the direction does not encode the high-level variable
abstractly, regardless of how predictive it is.

This is the strongest single experiment that can be run on paper-MEGA's
framework, and it has not been done in any of papers 3/5/6/7/8.

---

## 2. Experimental matrix (11 probes × counterfactual pairs)

| # | Probe | Regime | Abstracted variable | P+ source | P− source | N pairs | Prediction |
|---|---|---|---|---|---|---|---|
| 1 | ST_L31_gen | R1 | thinking-fraction | hard GSM8K (Phase 2A long-think baseline) | easy GSM8K matched | 25 | ✅ PASS — Y+ length shifts toward Y− length |
| 2 | ST_L11_gen | R4 (neg control for C1) | same as #1 | same | same | 25 | ❌ NULL — wrong layer |
| 3 | ST_L55_gen | R4 (neg control for C1) | same as #1 | same | same | 25 | ❌ NULL — wrong layer |
| 4 | RG_L55_mid_think | R2 | reasoning-quality | high-quality CoT (validated good) | low-quality CoT (validated bad) | 50 | ✅ PASS — quality shifts |
| 5 | Cap_L23_pre_tool | R3 | tool-success | SWE Pro N=99 successes | matched fails | 11 | ✅ PASS |
| 6 | Cap_L31_pre_tool | R3 | tool-success | same | same | 11 | ✅ PASS |
| 7 | Cap_L43_turn_end | R3 | tool-success | same | same | 11 | ✅ PASS |
| 8 | Cap_L55_pre_tool | R3 (sat-coupled) | tool-success | same | same | 11 | ✅ PASS (HE+MBPP); FLIP-DIRECTION (CF) |
| 9 | CoT_L55_mid_think | R4 (neg control for C5) | thinking-emission | `enable_thinking=True` | `enable_thinking=False` | 25 | ❌ NULL — template-locked |
| 10 | SWE_L43_pre_tool | R5 (neg control) | tool-finish | same as Cap | same as Cap | 11 | ❌ NULL — softmax-temp epi |
| 11 | FG_L31_pre_tool | R5 (neg control) | fabrication | fabricated answer (high prob) | factual answer (high prob) | 25 | ❌ NULL — pure epi |

**6 positive predictions + 5 negative controls.** Positives validate; negatives
confirm specificity. All 11 use the SAME interchange protocol and SAME random-
direction matched control; only the probe direction `v` differs.

---

## 3. Per-constraint factorial (R1 as deep case study)

ST_L31_gen (probe #1) is the strongest R1 case and serves as the deep dive.
With ST_L31_gen held as the *target probe*, vary each of C1–C5 independently:

### C1 (Layer) — spatial constraint

Same probe direction (trained at L31), apply interchange at **5 different layers**:

| Layer | Expected | Reasoning |
|---|---|---|
| L11 | NULL | Wrong layer (already known inert under α-sweep) |
| L23 | NULL | Wrong layer |
| **L31** | **PASS** | Trained layer; predicted abstraction valid here |
| L43 | NULL | Wrong layer |
| L55 | NULL | Wrong layer (already known inert under α-sweep) |

Conditions: 5; pairs: 25; total runs: 5 × 25 × 2 = 250 forward passes.

### C2 (Trajectory) — temporal constraint

Apply interchange at **3 onset points**:

| Onset | Expected | Reasoning |
|---|---|---|
| **Token 1 (immediate)** | **PASS** | KV-cache hasn't accumulated; abstraction holds |
| Decode step 50 | DEGRADED (3/10) | Partial KV-cache lock-in |
| Decode step 200 | NULL (0/10) | Full KV-cache lock-in |

Conditions: 3; pairs: 25; total: 3 × 25 × 2 = 150 forward passes.

### C3 (Magnitude) — α scaling

Interchange uses natural `α_counter − α_orig` as base magnitude. Vary scaling factor:

| Scale | Expected | Reasoning |
|---|---|---|
| 0.5× | PARTIAL | Sub-threshold |
| **1.0× (natural)** | **PASS** | Data-driven natural magnitude |
| 2.0× | DEGENERATE | OOD residual, behavior degrades into random-direction equivalent |

Conditions: 3; pairs: 25; total: 3 × 25 × 2 = 150 forward passes.

### C4 (Direction / saturation) — pair-direction asymmetry

Run pair both ways:

| Direction | Expected | Reasoning |
|---|---|---|
| P+ (long) → P− (short) | PASS | Pushdown along saturation headroom |
| P− (short) → P+ (long) | WEAKER OR NULL | Pushup against saturation |

Conditions: 2; pairs: 25; total: 2 × 25 × 2 = 100 forward passes.

### C5 (Decision locus) — structural control

Use CoT_L55_mid_think (template-locked) probe instead of ST_L31_gen as
*negative control* on identical pairs. Interchange should NULL out regardless
of layer/onset/magnitude.

Conditions: 1 (full re-run with different probe); pairs: 25; total: 1 × 25 × 2 = 50.

### Factorial summary

| Block | Conditions | Pairs | Forward passes |
|---|---|---|---|
| C1 factorial | 5 | 25 | 250 |
| C2 factorial | 3 | 25 | 150 |
| C3 factorial | 3 | 25 | 150 |
| C4 factorial | 2 | 25 | 100 |
| C5 negative control | 1 | 25 | 50 |
| **R1 deep dive** | **14** | **25** | **700** |

R2, R3 (×4), R4 (×3 neg), R5 (×2 neg) each get **one default condition** (C1+C2+C3+C4 all at predicted-pass setting): 6 probes × 25-50 pairs × 2 ≈ 400 additional forward passes.

**Grand total: ~1100 forward passes.**

---

## 4. Compute budget (Qwen3.6-27B inference, H100 80GB)

Per forward pass with 256–512 token generation:
- Single H100 80GB: 10–20 tok/s → 30–60 sec/gen
- Single A100 40GB (fallback): 5–10 tok/s → 1–2 min/gen

| Block | Forward passes | H100 wall clock | Cost ($2.50/hr) |
|---|---|---|---|
| R1 deep factorial | 700 | 6–12 hr | $15–30 |
| R2 default | 100 | 1–2 hr | $3–5 |
| R3 × 4 sites | 88 | 1–2 hr | $3–5 |
| R4 negative × 3 | 150 | 1.5–3 hr | $4–8 |
| R5 negative × 2 | 100 | 1–2 hr | $3–5 |
| Random-direction matched controls (50% overhead) | ~570 | 5–10 hr | $13–25 |
| **Total** | **~1700** | **~15–30 hr** | **~$40–80** |

Realistic budget: **$50 + 2 wall-clock days running.**

This matches the paper-MEGA §4.7.1 estimate of "~36 GPU-hr" (now refined upward
to ~20–30 hr for the canonical version + matched random controls). The estimate
of "~36 GPU-hr" was within bracket but slightly optimistic; honest revised
upper bound is 30 hr.

---

## 5. Counterfactual pair construction (per probe)

### #1, #2, #3 (subjective-time probes)

**P+ source:** 14 GSM8K prompts from Phase 2A where probe induced shortening
(probe_shortens=True in `phase2a_aggregate_stats.json`). These are the cases
where the baseline thinks LONG.

**P− source:** matched GSM8K easy problems where baseline thinks SHORT.
Source: existing GSM8K paired difficulty examples; OR Caio can hand-pick 14
short-thinking GSM8K problems matched on length/topic.

**Need to construct:** 14 paired (long-think, short-think) tuples. Pre-existing
data partially available; ~30 min to finalize pairs.

### #4 (RG_L55_mid_think)

**P+ source:** validated high-quality reasoning chain (Claude-as-judge labeled "good")
**P− source:** validated low-quality chain (Claude-as-judge labeled "bad")

**Pair sources:**
- Option A: same problem, both chains pre-generated with different temperatures + filtered by judge
- Option B: GSM8K problems where Qwen3.6 produced correct vs incorrect (Phase 11/Phase 10 RG_L55_mid_think captures have this)

**Recommended:** Option B, uses existing `phase10_fg_rg_causality/rg_full.json`
captures. ~50 paired (good_chain, bad_chain) tuples available.

### #5–8 (capability probes)

**P+ source:** 11 SWE-bench Pro N=99 cases where Qwen3.6 produced a patch that PASSED
**P− source:** 11 matched fails (same repo or similar difficulty)

**Pair source:** `swebench_v6_phase6/selected_iids.json` + `phase6_n99_verdict.json`
already classify success/fail. Need pairing logic: match each success to a
fail in the same repo (when possible) or by problem length.

### #9 (CoT_L55 template control)

**P+ source:** prompts with `enable_thinking=True`
**P− source:** same prompts with `enable_thinking=False`

**Pair source:** trivial — 25 GSM8K prompts × 2 chat-template settings. Direct.

### #10 (SWE_L43 epi control)

Same pair source as Cap probes (success vs fail SWE patches).

### #11 (FG_L31 epi control)

**P+ source:** fabricated answers (model confidently wrong)
**P− source:** factual answers (model correct)

**Pair source:** existing fabrication-detection probe training set (Phase 4
N=54 captures); pair each fabricated case with a factual one matched by domain.

---

## 6. Technical implementation

### Stack
- **Model:** Qwen3.6-27B with `transformers` from main (per `reference_qwen36_transformers_main.md`)
- **LAYERS shim:** `LAYERS = model.model.layers if hasattr(model.model, 'layers') else model.model.language_model.layers`
- **GPU:** Single H100 80GB on Modal/Runpod; A100 40GB fallback
- **Forward hooks:** standard PyTorch register_forward_hook on `LAYERS[L]`

### Capture phase (per pair, per layer)

```python
def capture_residual(model, prompt_p_minus, layer, position_token_idx):
    """Run P− forward, capture residual at (layer, position) into storage."""
    capture = {}
    def hook(module, inp, out):
        # out is (residual, attn_weights, ...) tuple usually
        capture['h'] = out[0][:, position_token_idx, :].detach().clone()
    h = LAYERS[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=tokenize(prompt_p_minus))
    finally:
        h.remove()
    return capture['h']  # shape (d_model,)
```

### Interchange phase (per pair, per layer)

```python
def interchange_generate(model, prompt_p_plus, layer, position_token_idx,
                          probe_direction, h_counter, scale=1.0):
    """Generate from P+ but interchange projection onto probe at (L, pos)."""
    v = probe_direction / probe_direction.norm()
    alpha_counter = (h_counter @ v).item()

    def hook(module, inp, out):
        h = out[0]
        if h.shape[1] > position_token_idx:
            h_orig = h[:, position_token_idx, :]
            alpha_orig = (h_orig @ v).item()
            delta = scale * (alpha_counter - alpha_orig) * v
            h[:, position_token_idx, :] = h_orig + delta
        return (h,) + out[1:]

    handle = LAYERS[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            output = model.generate(
                tokenize(prompt_p_plus),
                max_new_tokens=512,
                do_sample=False,  # deterministic for fair interchange
                temperature=1.0,
            )
    finally:
        handle.remove()
    return output
```

### Matched random-direction control

Same interchange protocol but `v` replaced with a unit-norm Gaussian random
vector resampled per pair. Distinguishes "probe abstraction" from
"any-direction perturbation".

### Measurement utilities

- **ST_L31 (thinking-length):** count tokens until `</think>` closing tag in generation
- **RG_L55 (quality):** Claude-as-judge with 0–5 scale on generated reasoning
- **Cap (patch-success):** run patch through SWE-bench Pro eval harness
- **CoT_L55 (thinking-emission):** binary — did `<think>` block appear?
- **SWE_L43 / FG_L31 (tool-finish / fabrication):** log-prob shift of target token + behavioral flip

---

## 7. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Counterfactual pair construction is hard for some probes (esp. RG) | High | Use Option B for RG (existing captures); for other probes, fallback to weaker pairs is acceptable |
| Residual capture position misalignment (P+ vs P− have different token counts) | Med | Align by *role* (e.g., last `<think>` token, not absolute index) |
| Matched random-direction control accidentally produces shifts | Low | Average over 3 random directions per pair; report mean ± std |
| Behavioral measurement noisy (esp. RG quality) | Med | Use Claude-as-judge with N≥3 judges or majority voting |
| All R1/R2/R3 fail (framework falsified) | ~30% | This is also a paper — frame as honest-negative methodology contribution |
| All R4/R5 unexpectedly pass | <10% | Re-examine protocol specificity; tighten controls |
| Compute overrun (>$80) | Low | Stop after R1 full + R2-R5 default; defer C2/C3 factorial if needed |
| Qwen3.6-27B inference instability on Modal/Runpod | Med | Use proven setup from Phase 2A (already validated env) |

---

## 8. Decision tree for outcomes

```
After running R1 + R2 + R3 (positive predictions):

  All 3 PASS interchange:
    └─ Continue with R4 + R5 negative controls
        ├─ All R4/R5 NULL:
        │   └─ ✅ CLEAN FRAMEWORK VALIDATION
        │       → Paper-MEGA v4 with §4.7 reframed as validated
        │       → arXiv preprint + TMLR submission
        │       → STRONG Anthropic Fellows signal
        └─ R4/R5 unexpectedly pass:
            └─ ⚠️ PROTOCOL ISSUE
                → Investigate which negative control passed and why
                → Likely reveals interesting mechanism (potential new paper)

  Mixed (some R1/R2/R3 pass, some fail):
    └─ Refine framework into subclasses
        ├─ ⚠️ HONEST PARTIAL VALIDATION
        │   → Paper-MEGA v4 reports which constraints validated, which didn't
        │   → Bundle with cross-model experiment before final submission

  All R1/R2/R3 FAIL:
    └─ 🔴 FRAMEWORK FALSIFICATION
        → Framework as causal-abstraction theory is wrong
        → All 11 probes are correlative-only, not abstracting
        → Reframe as HONEST-NEGATIVE PAPER:
           "Linear probes that lever under α-sweep STILL fail interchange tests
           — what we thought was causality is sophisticated correlation"
        → This is potentially the biggest paper of the series
        → ALSO strong Anthropic Fellows signal (honest-negative discipline)
```

All four outcomes are publishable. This is the key sign that the experiment
is worth running.

---

## 9. Success criteria

**Minimum viable result (must produce):**
- Interchange protocol implemented and validated against known α-sweep behavior
- R1 (ST_L31_gen) tested with at least the C1 factorial (5 layers)
- R4 negative control (ST_L11 or ST_L55) tested at same conditions
- One of {R2, R3} tested at default conditions
- Results saved to Drive + uploaded to HF dataset

**Target result (paper-MEGA v4):**
- All 11 probes tested at default conditions
- R1 full C1+C2 factorial
- All results in extended HF dataset
- Verify script extended to cover interchange checks (target 35-40 PASS)
- §4.7 rewritten as validated/falsified result, no longer future work

**Stretch result (paper standalone):**
- R1 full C1+C2+C3+C4+C5 factorial
- R2 with Claude-as-judge quality measurement
- R3 with all 4 sites + cross-distribution validation (HE+MBPP and CF)
- Standalone paper: "Interchange Validation of Operationally-Constrained Probe Causality"

---

## 10. Notebook skeleton outline (`nb_paper_mega_interchange.ipynb`)

```
Cell 1: Setup
  - !pip install git+https://github.com/huggingface/transformers.git
  - !pip install accelerate scipy
  - imports + DRIVE path constants + LAYERS shim definition

Cell 2: Load model + tokenizer
  - Qwen3.6-27B in bf16
  - device_map="auto"
  - verify LAYERS works

Cell 3: Load probe weights
  - From HF dataset caiovicentino1/openinterp-paper-mega-conditionally-causal
  - Or from local agent_probe_guard_eval/ + phase11_capability_locus/
  - Build dict: probe_name → (layer, position_token_role, direction_vector)

Cell 4: Counterfactual pair construction
  - Function per-probe-family
  - For ST: load GSM8K paired difficulty
  - For RG: load phase10 RG captures
  - For Cap: load phase6 SWE successes/fails + matching logic
  - For CoT: trivial template toggle
  - Returns dict: probe_name → list of (P+, P−) prompt pairs

Cell 5: Residual capture utility
  - Function capture_residual(model, prompt, layer, position) → tensor

Cell 6: Interchange forward hook + generation
  - Function interchange_generate(model, prompt, layer, position, v, h_counter, scale=1.0)

Cell 7: Behavioral measurement utilities
  - measure_thinking_length(text) → int
  - measure_reasoning_quality(text, judge=claude) → 0-5
  - measure_patch_success(patch, instance) → bool
  - measure_thinking_emission(text) → bool

Cell 8: R1 ST_L31_gen — C1 layer factorial
  - For each of 5 layers, for each of 25 pairs:
    - Capture h_counter from P− at this layer
    - Run interchange on P+ at this layer
    - Run matched random-direction control
  - Save results dict to Drive

Cell 9: R1 ST_L31_gen — C2 trajectory factorial
  - For each of 3 onset positions, for each of 25 pairs:
    - Apply interchange only from onset position forward
  - Save

Cell 10: R1 — C3 magnitude scaling, C4 pair-direction asymmetry, C5 control
  - Each as separate sub-block; save per-condition

Cell 11: R2 RG_L55 default run
  - 50 pairs × interchange + matched random
  - Save

Cell 12: R3 × 4 sites default run
  - For each of 4 capability probes
  - 11 pairs × interchange + matched random
  - Save

Cell 13: R4 + R5 negative controls
  - 5 probes × 25 pairs × interchange + matched random
  - Save

Cell 14: Aggregation + verdict
  - Per probe: compute behavioral_shift_toward_P_minus
  - Compare to random-direction baseline
  - Statistical test (paired Wilcoxon)
  - Classify each as PASS / NULL / FAIL
  - Cross-reference with paper-MEGA Table 2 predictions

Cell 15: Save aggregated verdicts + upload to HF
  - Append to caiovicentino1/openinterp-paper-mega-conditionally-causal
  - Update Appendix A claims

Cell 16: Visualization
  - Per-probe behavioral shift vs random baseline
  - C1 factorial heatmap (layers × shift magnitude)
  - C2 onset decay curve
  - Decision-flowchart annotated with interchange results
```

Estimated lines: ~700–900.
Estimated execution time: 15–30 GPU-hr split across 2–3 sessions.

---

## 11. Sequencing recommendation

**Phase A (this week, no compute):**
1. Build counterfactual pair source files (ST pairs from GSM8K, RG pairs from
   existing captures, Cap pairs from N=99 verdict). ~2 hr.
2. Write the notebook skeleton (Cells 1–16). ~3 hr.
3. Dry-run on CPU with toy data to validate logic. ~1 hr.

**Phase B (next week, compute available):**
4. Minimum viable run: R1 C1 factorial only (5–10 GPU-hr). ~$15-25.
5. Inspect results: does ST_L31 PASS at L31 while ST_L11/L55 NULL?
6. If YES → continue with R2/R3/R4/R5 at default conditions (5–10 more hr).
7. If NO → stop and reframe (this IS the negative-result paper).

**Phase C (week 3, contingent on B):**
8. Full C2 + C3 + C4 + C5 factorial. 10–15 GPU-hr additional.
9. Write up results + update paper-MEGA to v4.
10. Extend verify_paper_mega_claims.py to cover interchange checks.
11. Re-upload to HF + redeploy paper.

**Phase D (week 4):**
12. arXiv preprint + LinkedIn / Twitter / LessWrong announcement.
13. Share with Lindsey (if he responded) / submit Anthropic Fellows
    application reference.

---

## 12. What this experiment does NOT do

- Does not test cross-model generalization (future paper)
- Does not test mixture-of-experts or multimodal models
- Does not formalize the framework in causal-abstraction calculus
  notation (Geiger 2025 formalism); we test empirically only
- Does not address whether the probe-direction is the "minimal sufficient"
  direction or just "a sufficient" one (would require concept-bottleneck-style
  ablations)
- Does not produce a new linear-probe SDK (existing agent-probe-guard
  v0.1 sufficient)

---

## 13. Notes for execution

- **Determinism:** Use `do_sample=False, temperature=1.0` for all generations
  to make interchange comparison fair.
- **Resume on disconnect:** Save intermediate results every 5 pairs;
  Modal/Runpod sessions can drop.
- **Per-pair tracking:** Log instance IDs explicitly so partial runs can be
  resumed without re-doing completed pairs.
- **Memory:** Capturing residuals for all 11 probes × 50 pairs requires
  ~1 GB storage. Negligible.
- **Hook safety:** Always `try/finally` register_forward_hook with `.remove()`
  to avoid hook leakage poisoning subsequent calls.

---

## 14. Reference dependencies

- `transformers` from main (per `reference_qwen36_transformers_main.md`)
- LAYERS shim (per same memory)
- Drive mount path: `~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/`
  (per `reference_google_drive_mount_path.md`)
- HF auth as `caiovicentino1`
- Existing probe weights in `agent_probe_guard_eval/` + `phase11_capability_locus/`
- Existing pair sources:
  - `subjective_time_phase2a/phase2a_aggregate_stats.json` (ST pairs)
  - `phase10_fg_rg_causality/rg_full.json` (RG quality pairs)
  - `swebench_v6_phase6/{selected_iids.json,phase6_n99_verdict.json}` (Cap pairs)

---

## 15. Status

- **2026-05-17:** Plan drafted (this document)
- **Next:** Construct counterfactual pair files (Phase A step 1)
- **Then:** Write notebook skeleton (Phase A step 2)
- **Then:** Wait for compute window + execute Phase B

This plan is a working document. Update it as Phase A reveals issues with pair
construction or notebook logic.
