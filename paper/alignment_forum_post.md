# Tool-Entropy Collapse: A Cross-Architecture Signature of Agent WANDERING Failure

*A practical safety-monitor for production LLM agents, derived from 99 Qwen3.6-27B SWE-bench Pro trajectories and validated across Llama-70b + GPT-5.*

**TL;DR**

- Probe-based monitoring of LLM agents has a quantifiable **34% blind spot**: the WANDERING sub-class where the probe says "success" but the agent never emits the completion action and hits its turn budget.
- I tested six detector designs across three signal channels (text, residual cross-layer, action entropy). The most promising candidate is **tool-use entropy collapse** — WANDERING agents repeat the same 2-3 tools in their final 10 turns while varying the natural-language wording. The loop lives in **action space, not text space**.
- The W/S median entropy ratio is **0.41 in Qwen3.6-27B and identically 0.41 in Llama-70b**, 0.71 in GPT-5 router (same direction). Cross-architecture invariance within SWE-bench-style code-execution tasks.
- Cross-task validation on METR MALT (15+ task families) is **null** — scoping the claim to multi-turn code-execution agents with rich action spaces, not universally task-invariant.
- A combined detector (v1 text post-hoc ∪ v5 tool-entropy) achieves **70% WANDERING recall at 5% SUCCESS false-positive rate**, suitable for production Tier-3 autonomous-termination deployment with per-customer FP-budget agreement.

**Why I'm posting this here**

The Alignment Forum has the audience that should care most: safety-monitoring practitioners, Anthropic/METR/Apollo researchers, and people building agent eval infrastructure. The paper's central pivot is from "probes detect deception" (Apollo, sleeper-agents) to "probes have a blind spot that text and action signals can close together." That extension feels natural to this community.

**The blind spot, mechanistically**

Linear probe (top-10 diff-means at L43 pre_tool, 5-fold CV) on 99 Qwen3.6-27B SWE-bench Pro trajectories: CV AUROC 0.934, but among failures 34% are "WANDERING" — probe stays >0.70 with median final score 1.000, 95% produce patches, but the agent never emits `finish_tool`. Two manually-inspected examples:

- *"All tests pass. Let me verify the final state of the modified files."* (qutebrowser-2dd8966, turn 49, patch present, probe = 1.00)
- *"The tests are failing due to the lack of a display server (Xvfb) — this is an infrastructure limitation, not a code issue. The code changes have been successfully made and verified."* (qutebrowser-9b71c1ea, turn 49, patch present, probe = 1.00)

The model has internally consolidated a "done" verdict but the surface-to-action circuit hasn't fired. This is a natural-trajectory analog of intervention-induced obfuscation in my prior Paper 11 (CoT-redirect): in both cases, internal belief and external action diverge — here without any intervention.

**Six detector designs (3 channels)**

I tested detectors across text, residual cross-layer disagreement, and action entropy. Honest negatives matter here:

| Detector | Channel | Recall | FP | Verdict |
|---|---|---|---|---|
| v1 post-hoc text | Text | 35% | **0%** | Tier 1 forensics ✓ |
| v2 naive early-warning | Text | 30% | 15% | FP unacceptable |
| v3 persistence hypothesis | Text | 10% | 5% | **Refuted in opposite direction** (SUCCESS persists more, p = 0.92) |
| v4 cross-layer disagreement | Residual | 65% | 30% | Tier 2 advisory ✓ |
| **v5 tool-entropy collapse** | **Action** | **55%** | **5%** | **Tier 3 ✓ — best Tier-3 candidate in this work** |
| v6 residual stability | Residual | 15% | 5% | Null at Tier 3 |

The interesting story is v3 → v5. I hypothesized that WANDERING agents are "stuck in a verification loop" — and looked for it in text persistence (consecutive turns of "tests pass" / "code is correct" phrases). That refuted in the opposite direction: SUCCESS agents verbalize completion more persistently. But the same hypothesis was true at the **action level**: WANDERING agents repeat the same TOOLS (test → view → test → view) while varying the wording. The loop exists, but at the wrong observable for v3.

**Cross-architecture validation (3 of 4 confirm)**

Public agent trajectory datasets let me check this on different models:

| Model | Lab | n (W/S) | W/S ratio | p |
|---|---|---|---|---|
| Qwen3.6-27B | Alibaba | 20 / 40 | **0.41** | 1.0×10⁻⁶ |
| Llama-70b | Meta | 1,358 / 488 | **0.41** | < 10⁻¹⁵ |
| GPT-5 router | OpenAI | 137 / 664 | **0.71** | 8.9×10⁻³⁵ |
| Claude 3.7 Sonnet | Anthropic | 1,116 / 1,813 | (N/A) | dataset unsuitable* |

*Claude data from SWE-bench/SWE-smith-trajectories is SFT-curated; success and failure classes follow identical tool sequences (`bash → editor_str_replace → bash → editor_view → submit`), so entropy can't differentiate. Not a signal failure — a data-quality non-finding. Live Claude failed-eval traces are needed.

The ratio invariance (W/S = 0.41 in Qwen and Llama, exact match) was the unexpected finding. GPT-5 is weaker (0.71) but same direction.

**Cross-task validation (honest null)**

METR MALT dataset has 15+ task families (password_check, wordle, hackthebox, gaia, ai_rd_*, etc.) with explicit `gives_up` labels. v5 does NOT discriminate WANDERING from SUCCESS in MALT: W = 1.459 vs S = 1.449, p = 0.81. Four candidate explanations:

1. MALT `gives_up` is a manual label (≠ behavioral WANDERING)
2. n = 9 underpowered
3. MALT trajectories too short (median 4 unique tools vs Llama 17)
4. Restricted XML scaffold (`<bash>`/`<python>`/`<submit>`) limits action diversity

This **scopes** the universal claim: tool-entropy collapse is a robust WANDERING signature for multi-turn code-execution agent tasks with rich action spaces (validated cross-architecture on SWE-bench across Qwen, Llama, GPT). It does NOT generalize to short multi-task agent benchmarks with restricted action vocabularies, OR to alternative failure modes that differ from budget-exhaustion WANDERING.

**Three deployment tiers, all shippable**

| Tier | Use case | Detector | Recall | FP | Lead |
|---|---|---|---|---|---|
| 1 (forensics) | log analysis | v1 alone | 35% | 0% | 0 |
| 2 (advisory) | live escalation | v1 ∪ v4 | 80% | 30% | 15 turns |
| **3 (autonomous)** | **terminate agent** | **v1 ∪ v5** | **70%** | **5%** | post-hoc |

Tier 3 deployment in production requires explicit per-customer FP-budget agreement. At scale of millions of trajectories, 5% FP rate ≈ thousands of terminated successes. The detector design lets callers tighten the threshold for stricter precision: v5 at `tool_entropy < 0.45` gives 50% recall at 2.5% FP; at `< 0.40`, 25% recall at 2.5% FP.

**What I'm NOT claiming**

- The mid-layer ablation evidence for "edge-layer specific" mechanism is correlational; it's also consistent with a layer-count effect on the disagreement metric. I hedge between these in §5.4.1.
- The causal direction (does collapsed tool-entropy *cause* WANDERING, or co-occur?) is untested. Steering interventions on L11/L55 to test the mid-layer-to-edge alignment hypothesis are explicitly future work.
- The behavioral sub-classification on cross-model datasets uses exit-status proxies, not the probe-based WANDERING definition. Acknowledged in limitations.
- **This is empirical work scoped to one task family (SWE-bench-style code-execution) with one new signal candidate, not a paradigm-shifting result.** The cross-architecture ratio match (W/S ≈ 0.41 in Qwen AND Llama) is the most striking pattern and merits independent replication on additional models before treating as a "discovery". The Tier-3 numbers (70% × 5% FP) come from N=20 WANDERING trajectories on the Qwen primary dataset; at production scale of millions of trajectories, 5% FP corresponds to thousands of terminated successes, and the threshold needs explicit per-customer calibration before deployment.

**What this opens up**

A practical safety monitor for agent infrastructure that works on any agent framework logging per-turn actions — no model access, no probes, no residuals. Operational integration into AgentGuard-style SaaS:

- Tier 1 forensics dashboards: "your N agents had M trajectories with WANDERING signature"
- Tier 2 advisory escalation: "agent at risk — extend budget or human review?"
- Tier 3 autonomous: "kill agent at 5% FP cost, save 95% of compute on WANDERING-prone runs"

Open questions I'd value feedback on:

1. The mid-layer-to-edge alignment failure interpretation — is this consistent with what others have seen in deception-probe work? (Apollo, Anthropic sleeper-agents.)
2. The four candidate explanations for the MALT null — which feels most likely, and what task design would let us isolate the right one?
3. The 25% residual blind spot at Tier 3 (6/20 WANDERING not caught by v1 ∪ v5 even at 5% FP) — anyone seen analogous failure modes that text + action signals jointly miss?

**Links**

- **Zenodo DOI**: [10.5281/zenodo.20368807](https://doi.org/10.5281/zenodo.20368807) (paper PDF + LaTeX source + 5 figures, CC-BY-4.0)
- **GitHub**: [github.com/OpenInterpretability/openinterp-swebench-harness](https://github.com/OpenInterpretability/openinterp-swebench-harness) (all per-trajectory output JSONs, reproducibility scripts, Apache-2.0)
- **GitHub Release**: [paper-v0.8](https://github.com/OpenInterpretability/openinterp-swebench-harness/releases/tag/paper-v0.8) (versioned PDF + figures)
- Tool-entropy analysis scripts: `scripts/v5_tool_entropy.py`, `scripts/v5_cross_model_*.py`, `scripts/v5_cross_task_malt.py`

**Citation**:
```
Vicentino, C. (2026). Tool-Entropy Collapse: A Cross-Architecture Signature
of Agent WANDERING Failure. Zenodo. https://doi.org/10.5281/zenodo.20368807
```

I'm Caio Vicentino, working independently on agent safety monitoring under the OpenInterpretability umbrella. Background: 7 years in DeFi/crypto, transitioned to mechanistic interpretability in 2026. ORCID 0009-0003-4331-6259. Comments and pushback welcome.
