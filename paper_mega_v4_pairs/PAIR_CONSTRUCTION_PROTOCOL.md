# Paper-MEGA v4 — Pair Construction Protocol (Phase A.1)

Counterfactual (P+, P−) pair files for the 11-probe interchange-intervention
experiment. Built 2026-05-17. Status legend: ✅ ready, ⚠️ proxy/limited, 🔴 blocked.

## File inventory

| Probe family | File | Pairs | Status | Notes |
|---|---|---|---|---|
| ST_L31_gen, ST_L11_gen, ST_L55_gen | `st_pairs.json` | 15 P+ only | ⚠️ partial | P− needs Phase B0 measurement pass on easy GSM8K |
| RG_L55_mid_think | `rg_l55_pairs.json` | 25 (length-proxy) | ⚠️ weak proxy | Length ≠ quality; needs Claude-as-judge in v4.1 |
| Cap_L23, L31, L43, L55 + SWE_L43 | `cap_pairs.json` | 10 (emit-proxy) | ⚠️ proxy | Emit-vs-no-emit, not pass-vs-fail; needs SWE-bench Pro eval for true labels |
| CoT_L55_mid_think | `cot_l55_pairs.json` | 25 | ✅ ready | Trivial template toggle |
| FG_L31_pre_tool | `fg_pairs.json` | — | 🔴 blocked | Phase 4 N=54 captures not located on Drive; FabricationGuard product data lives in cli repo (separate) |

## Per-probe construction details

### Probes #1-3 (ST_L31/L11/L55 — subjective-time) ⚠️

**Status:** P+ ready (15 long-think prompts from Phase 2A), P− pending.

**P+ source:** `subjective_time_phase2a/phase2a_steering_results.json` — filtered to
prompts where baseline `thinking_len >= 300` (long-think). Yields 15 candidates
(matches 14 used in Fisher test + 1 outlier).

**P− construction (Phase B0):**
- Input: GSM8K test split (~1319 problems)
- Filter to 1-step arithmetic (single operation, single number)
- Run Qwen3.6-27B baseline on 50 candidates, keep ~25 where `thinking_len < 200`
- Compute: ~30 min H100, ~$1.50
- Output: `phase_b0_easy_gsm8k_pairs.json` populated

**Use in experiment:**
- Probe #1 (R1): apply interchange at L31 → expect PASS (thinking shortens)
- Probes #2, #3 (R4 neg): apply interchange at L11 / L55 → expect NULL

### Probe #4 (RG_L55_mid_think — reasoning quality) ⚠️

**Status:** Length-proxy pairs built. Quality labels need full re-generation.

**Why correctness-label FAILED for v4 minimum:** `phase10/rg_full.json`
`baseline_text` field stores only the FIRST ~130-180 chars of generation
(truncated at the residual capture point for L55 mid_think). Each entry stops
mid-thinking before the model emits any final answer. Naive answer-extraction
via regex grabs intermediate numbers from problem-setup text (e.g. "ratio 7:11"
yields "7", not the actual computed answer).

**This is a data-truncation artifact, NOT a Qwen3.6-27B capability failure.**
The model would easily score 60-80%+ on GSM8K with or without thinking; the
Phase 10 captures simply don't preserve the full generation for behavioral eval.

**Built (weak length-proxy):** Sorted 50 entries by `baseline_text` length, top 25
vs bottom 25. Length here correlates with how much the model committed to setup
before being captured — weak signal for "reasoning thoroughness" but real signal
for "model engagement", which is closer to what RG probe predicts anyway.

**v4.1 upgrade path:**
1. Re-run baseline generation with full max_new_tokens=512 on the same 50 GSM8K
   prompts (~15min H100, $0.75) → full chains with final answers
2. Label correctness vs gold_answer (expect 60-80% correct, real split available)
3. Optionally Claude-as-judge for quality 0-5 on top of correctness
4. Rebuild pairs as N=20+ paper-grade good vs bad

### Probes #5-8 + #10 (Cap × 4 sites + SWE_L43) ⚠️

**Status:** Emit-proxy pairs built. True patch-success blocked.

**Why proxy:** `phase6_results.json` records `patch_n_bytes` per instance (89/99 produced
patches) but does NOT record patch-pass status. True patch-pass requires running each
patch through SWE-bench Pro test harness (~$50 + Docker setup).

**Built (proxy):** P+ = 10 instances with `patch_n_bytes > 0`, P− = 10 with `patch_n_bytes = 0`.
Only 10 P− available (89 produced patches, 10 didn't). N=10 per pair.

**v4.1 upgrade path:**
1. Run SWE-bench Pro eval on all 89 patches → success/fail label per iid
2. Rebuild pairs as N=11 success × N=88 fail (matched by repo)

### Probe #9 (CoT_L55_mid_think — template control) ✅

**Status:** Ready. Trivial.

**Construction:** Same 25 GSM8K questions × chat-template flag `enable_thinking={True, False}`.
Pairs `pair_id` linked.

**Expected:** NULL across all conditions (template-locked decision is in input tokens,
not residual stream).

### Probe #11 (FG_L31_pre_tool — fabrication) 🔴

**Status:** Blocked. Source not located on Drive.

**What's needed:** Pairs of (fabricated_answer_prompt, factual_answer_prompt) where
Qwen3.6-27B confidently produces wrong vs correct answers.

**Where to find:**
1. FabricationGuard product training set (lived in `cli` repo, possibly HF dataset under
   `caiovicentino1/fabrication-guard-*` — check)
2. notebook 28 captures (referenced in MEMORY.md `project_fabricationguard_pivot_results.md`)
3. HaluEval / SimpleQA datasets with Qwen3.6-27B re-labeled

**v4.1 unblock path:** ~1 hour to locate source + 30 min to build pair file.

## Phase B0 — pre-experiment measurement protocol

Required before Phase B (compute):

1. **ST P− construction** (~30 min, $1.50):
   - Load GSM8K test split
   - Filter to short solutions (1-2 reasoning steps)
   - Run Qwen3.6-27B baseline, measure `thinking_len`
   - Keep 25 with `thinking_len < 200`
   - Save as `phase_b0_easy_gsm8k.json`

2. **(optional v4.1) RG re-labeling** (~30 min Claude API):
   - For each of 50 RG chains, prompt Claude with chain + ask quality 0-5
   - Build pairs from N=12 high (4-5) vs N=12 low (0-1)

3. **(optional v4.1) Cap patch-success** (~$50 + 2-3 hrs):
   - SWE-bench Pro Docker setup
   - Run all 89 patches through test harness
   - Per-iid pass/fail label

## What's good enough for paper-MEGA v4 minimum?

**Phase B with current proxies will validate the FRAMEWORK qualitatively:**

- ✅ ST probes (#1-3): once P− built in B0, full C1 factorial works cleanly
- ⚠️ RG (#4): length-proxy will give a directionally-valid signal but won't be paper-grade
- ⚠️ Cap (#5-8): emit-proxy will give R3 pushdown signal but not the cleanest claim
- ✅ CoT (#9): negative control trivially clean
- ⚠️ SWE_L43 (#10): emit-proxy; OK for negative control
- 🔴 FG (#11): omit from v4 minimum; include in v4.1

**Strategy:** ship v4 minimum with explicit "weak proxy" disclosures in Appendix.
Run v4.1 with proper labels for final submission.

## Summary table for paper-MEGA v4 §3.6 (new)

| Probe | Pair source quality | v4 conclusion strength |
|---|---|---|
| ST_L31 (R1 target) | Phase 2A + B0 | STRONG (clean P+/P− pairs) |
| ST_L11/L55 (R4 controls) | same | STRONG |
| RG_L55 (R2) | length proxy | WEAK — flag in paper |
| Cap × 4 (R3) | emit proxy | MODERATE — flag |
| CoT_L55 (R4 control) | template toggle | STRONG |
| SWE_L43 (R5 control) | emit proxy | MODERATE — flag |
| FG_L31 (R5 control) | DEFER to v4.1 | — |

**Net:** R1 case (ST_L31) + R4 controls + R5-CoT control are paper-grade.
R2/R3/R5-SWE add corroboration with weaker proxies. v4 minimum = honest
mixed-strength result; v4.1 = full paper-grade.
