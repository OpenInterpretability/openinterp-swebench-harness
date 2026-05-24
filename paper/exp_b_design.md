# Exp B Design — Cross-Layer Activation Patching Causal Test

**Paper #2 contribution**: causal evidence for the mid-layer-to-edge alignment hypothesis from Tool-Entropy paper §13.

**Status (2026-05-24)**:
- LayerPatch class implemented (`instrumentation/layerpatch.py`)
- Design doc + skeleton complete
- **Determinism check FAILED** — only 5/20 trajectories ≥80% replayable (`scripts/exp_b_determinism_check.py`)
- **PIVOT to Plan B** (always-on hook from turn 0) — see §"Updated execution plan" below
- Awaits: Phase 6b Docker eval completion (running in background) + compute environment setup

## Updated execution plan (Plan B post-determinism check)

After analytical audit of WANDERING tool_calls (`scripts/exp_b_determinism_check.py`), confirmed
17.5% are DANGEROUS commands (pip, pytest, python, git, etc.) with potential non-determinism risks.
Only 25% of trajectories cross ≥80% replayable threshold. **Plan A (surgical replay + patch at T)
is engineering-heavy with substantial determinism risk.**

**Plan B (always-on hook from turn 0)**:
1. Capture mean L55 pre_tool residual from matched SUCCESS donors (offline, no model run)
2. Install LayerPatch on L55 with SUCCESS-derived residual, `mode='replace'`
3. Run FRESH agent inference for WANDERING instance with hook attached throughout
4. Observe outcome: did `finish_tool` emit within max_turns?

**Causal claim tested by Plan B**: "Persistent injection of SUCCESS-derived L55 direction biases
WANDERING agents toward `finish_tool` emission." (Slightly different from Plan A's "patching at
lock-in turn flips outcome" but equally publishable as causal evidence for edge-layer alignment.)

Controls updated for Plan B:
- **C1 random direction** (unchanged) — tests if ANY persistent perturbation flips
- **C2 LOCKED donor** (unchanged) — tests SUCCESS-specificity
- **C3 L43 mid-layer** (unchanged) — tests edge-layer-specificity
- **C4 amplitude sweep** (replaces C4 "pre-lock turn" from Plan A) — sweep α at L55 in `mode='add'`
  to test if magnitude matters

No determinism risk in Plan B since each run is fresh and deterministic given seed.

## Causal hypothesis

From Tool-Entropy paper:
> "WANDERING agents have consolidated a mid-layer verdict (model 'knows' it's done), but edge circuits (L11 surface processing, L55 output planning) that translate verdict → `finish_tool` action haven't aligned. Decision-to-action circuit desynchronized."

## Direct causal test

If the hypothesis is true: patching L55 (output planning) residual from a SUCCESS agent (who DID emit `finish_tool`) into a WANDERING agent (who DID NOT) at lock-in turn should cause the WANDERING agent to emit `finish_tool`.

## Experimental design

### Primary intervention

For each of 20 WANDERING trajectories:

1. **Identify lock-in turn T**: earliest turn where probe stays > 0.70 stable until end. From `inflection_results.json` per-trajectory `score_trajectory`.
2. **Find matched SUCCESS donor**: same repo + similar task complexity. Compute mean L55 pre_tool residual across matched SUCCESS trajectories at turn T (or nearest aligned token).
3. **Replay sandbox state**: deterministically execute turns 0..T-1 of WANDERING trajectory (replay captured `tool_calls` against fresh sandbox at `base_commit`).
4. **Install LayerPatch hook** at L55 with SUCCESS-derived residual, `mode='replace'`.
5. **Resume agent inference from turn T**: model generates new response at turn T with patched L55, sees the bash output, continues normally.
6. **Continue until** `finish_tool` emitted OR `max_turns` reached.
7. **Measure**: binary outcome — did `finish_tool` emit?

### Falsification controls

| Control | Patch source | Expected if hypothesis TRUE | Expected if mechanism wrong |
|---|---|---|---|
| **C1 — Random direction** | Random L55 direction (matched norm) | NO flip | Same flip rate as primary → patch is non-specific |
| **C2 — LOCKED donor** | L55 from LOCKED agent (who also failed) | NO flip | Flip rate similar → not SUCCESS-specific signal |
| **C3 — Mid-layer (L43)** | L43 mid-layer SUCCESS direction | NO flip if edge-specific | Flip → mechanism is general layer-count effect, not edge-specific |
| **C4 — Pre-lock turn** | Patch at turn T-10 (before consolidation) | LOWER flip rate (no verdict yet) | Same flip → not consolidation-dependent |

Each control: 20 paired runs. McNemar's test on primary vs each control.

### Statistical power

- 20 WANDERING (paired design)
- McNemar's exact for binary outcome
- Detectable effect: 30 percentage point difference at α=0.05, β=0.80, requires ~17 discordant pairs → feasible at N=20

### Decision matrix

| Outcome | Primary flip rate | Control rate (best) | Interpretation |
|---|---|---|---|
| **Strong** | ≥40% | ≤10% | Mechanism CONFIRMED — edge-layer alignment failure |
| **Weak** | 20-40% | 10-20% | Partial — L55 contributes but not sole cause |
| **Null** | <20% | similar | Mechanism REFUTED — patches don't restore action emission |
| **Confound** | High | Also high | Any L55 perturbation flips → non-specific effect |

Any non-null outcome is a publishable finding. Even REFUTED is informative: would force reframe of paper #1's mechanism claim.

## Compute requirements

| Component | Resource | Cost |
|---|---|---|
| Qwen3.6-27B inference (full precision) | H100 80GB or A100 80GB | Required |
| Per-trajectory replay | ~1 min sandbox setup | — |
| Per-trajectory inference | ~25 turns × 5sec = 125s | — |
| Primary + 4 controls × 20 traj | ~5 × 20 × 3min = 5h | — |
| Model loading | ~5min × 1 session | — |
| **Total wall time** | ~5-6h compute | — |
| **Recommended env** | vast.ai H100 @ $2/h × 6h | **~$12** |
| **Alternative** | Caio's local Mac (MLX 4-bit, different hook API, +2 days dev) | $0 |

## Engineering tasks (when ready to execute)

1. **`scripts/exp_b_cross_layer_patching.py`** — main orchestration script:
   - Load WANDERING / SUCCESS / LOCKED IIDs + lock_in turns from `inflection_results.json`
   - For each pair: replay sandbox state up to T, patch L55, run from T
   - Log: finish_tool emission, turns used, final patch
   - Save: per-trajectory JSON + aggregate stats

2. **Sandbox state replay logic** — extend `agent/loop.py` to support `replay_until(turn_idx)`:
   - Execute captured `tool_calls[0..turn_idx-1]` directly via bash session
   - Verify each step's `tool_results` matches captured (sanity check determinism)
   - Skip if any divergence (flag for manual review)

3. **Hook activation in agent loop** — modify `runner.py` to accept optional `LayerPatch` instance:
   - Install patch before turn T inference
   - Detach after inference if "single-turn intervention" mode
   - Or keep attached for "persistent intervention" mode

4. **Matched-donor selection** — `scripts/exp_b_donor_match.py`:
   - Group SUCCESS / LOCKED by repo
   - Within repo, capture L55 residuals at turn T from each donor
   - Compute mean → donor residual

## Reviewer-targeting

The 4 controls (C1-C4) are critical — they prevent the common reviewer critique:
> "Any perturbation at any layer might flip behavior — your finding could be non-specific."

By showing:
- C1 (random) doesn't flip → patch direction matters
- C2 (LOCKED) doesn't flip → SUCCESS-specific signal matters
- C3 (L43) doesn't flip → edge-layer matters specifically
- C4 (pre-lock) doesn't flip → mid-layer consolidation timing matters

We pre-emptively close the most likely rejection vectors.

## Open questions for falsification

1. **Determinism risk**: if sandbox state at turn T is not byte-identical between original run and replay, the agent may see different state → different decision unrelated to the patch. Mitigation: verify `tool_results` match captured at each replay step; abort on divergence.

2. **Token alignment**: L55 patch is applied at `[0, -1, :]` (last token). The "lock-in turn" is defined at `pre_tool` position. Need to verify patch is applied at the SAME logical position as the captured donor residual.

3. **Patch persistence**: does the patched state propagate forward in KV cache? Or does new generation overwrite it? Likely the latter — the patch only affects the immediate forward pass. For sustained intervention need patch to persist across decode steps.

4. **Hook collision with KV cache reuse**: HF transformers caches K/V for efficiency. If our hook replaces the hidden state, the cached K/V from PREVIOUS tokens still reflects unpatched. Need to verify this doesn't violate causal claim.

These are normal mech-interp engineering challenges. Caio's prior Paper-MEGA work tackled similar issues — apply same Conditionally-Causal Probes 5-constraints framework here.

## Timeline (post-Phase 6b completion)

| Day | Task |
|---|---|
| 1 | Implement sandbox replay logic + verify determinism on 1 trajectory |
| 2 | Implement matched-donor selection + capture donor L55 residuals |
| 3 | Pilot Exp B on 5 WANDERING + 5 SUCCESS pairs (primary only) |
| 4 | Full Exp B on 20 + 4 controls |
| 5 | Analysis + writeup |

Total: ~1 week active work + compute.
