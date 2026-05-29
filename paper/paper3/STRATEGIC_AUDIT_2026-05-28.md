I have all the material I need from the digest and synthesis agents. This is a synthesis task — no further investigation required. Writing the report directly.

# OpenInterp Stack Audit — Final Strategic Synthesis
**For: Caio Vicentino | Date: 2026-05-28 | Author: Final Synthesis Auditor**

---

## 1. EXECUTIVE VERDICT

OpenInterp is a paper-grade research engine bolted to a zero-traction distribution layer, and the gap between those two facts is the entire story. The core science is real and reproducible — Paper #1's headline detectors (v5: 55%/5%, v1∪v5: 70%/5%) reproduce to machine precision with genuine cross-architecture validation (Llama-70B p<1e-15, GPT-5 p=8.9e-35) — but it is invisible (0 GitHub stars, 0 HF downloads, 26 Zenodo views), undermined by a single self-documented bug in its own flagship public artifact, and diluted by strategic scatter across two incompatible markets (quant vs. agent-interp) and a vaporware crypto pivot. **The one biggest insight: you do not have a quality problem, you have a focus-and-shippability problem — and the highest-EV move is not new research, it's converting work already 95% done (Paper #3 draft, Paper #2 fixes, the v5 bug, the Inspect PR) into landed, cited, merged artifacts before the agent-interp niche closes in H2 2026.** Stop shipping volume; start landing things.

---

## 2. STACK HEALTH (brutal)

**SOLID (defend these):**
- **openinterp-swebench-harness** — the genuine crown jewel. Clean LayerTap/CaptureBuffer/LayerPatch instrumentation, hash-based seeding, real error handling. This is the asset no one else has.
- **Paper #1 core metrics** — fully verifiable to machine precision. The cross-arch validation is the moat's empirical spine.
- **mechreward** — production-grade SAE+probe training, Stage Gates 1-3 passed (ρ=0.540→76%→83% GSM8K). Polishing only; core research done.
- **Methodological rigor** — 6 honest walk-backs via random-feature/shuffled-source/control-token baselines. This is rare and is your reputational capital.

**FRAGILE (fixable, but actively bleeding credibility):**
- **inspect-tool-entropy-collapse** — contains a CRITICAL bug on your highest-distribution public asset (UK AISI PR #1716). `tool_entropy_last_n` conflates "no tools" and "single repeated tool" both to entropy=0.0; the test at `test_detectors.py:63` asserts `fired is False` while the live code now FIRES — **the published test asserts the opposite of what the code does.** A maintainer running pytest finds a contradiction. This violates Paper #1 §6's own canonical WANDERING claim and misses 10-15% of real cases.
- **Paper #2** — stuck at v0.2 with 5 enumerated CRITICAL fixes unapplied (McNemar p should be exact 1.00 not 0.65; title "Refute"→"Refine"; missing Wilson CIs = workshop desk-reject trigger; no figure; buried stable-WANDERING subgroup). Blocked on write-up, not analysis.
- **Paper #3** — experiments complete 19+ days, ZERO draft text. Pure staleness risk in a closing niche.
- **CI/reproducibility** — 5/7 repos have no CI; all deps float (`torch>=2.4`, `transformers>=5.0`). This is the exact mechanism that caused the documented 12→0 silent-refusal artifact. For a lab whose *only* solo-beats-institution lever is reproducibility, this is self-sabotage.
- **HF presence** — 97 assets, near-zero engagement, zero Spaces, sparse cards. "Try it in 2 minutes" is currently false.

**VAPORWARE (stop pretending these are real):**
- **AgentGuard / FabricationGuard / ReasonGuard / ProbePack SaaS** — zero customer calls, zero demo, zero revenue, no API spec/latency/cost model. The "2 orders of magnitude crypto traction" claim rests on polymarket-mcp's 519★ — **a repo Caio did not author.** Citing this in grants is a credibility landmine.
- **openinterp PyPI** — `upload_trace`/`steer`/`score`/`circuit` all `NotImplementedError(Q2)`; Atlas returns 4 hardcoded features. Looks like vaporware to the few who try it.
- **Web Laboratory/Academy/InterpScore** — coming-soon shells behind roadmap.

---

## 3. THE MOAT

**The single defensible position: cross-architecture-validated mechanistic *monitorability evals* for agent failure — activation-level prediction of WANDERING before the agent acts.** The evidence triad is unique and survives scrutiny:
1. **Reproducible signal** — Paper #1 metrics match to machine precision with real cross-arch validation.
2. **Orthogonal to all incumbents** — all 5 monitoring vendors (Maxim, LangSmith, Arize, LangFuse, Agent 365) are behavioral/telemetry; NONE do activation-level pre-action prediction. The niche is "expanding but NOT yet crowded" (4-5 papers in 2026).
3. **Uncontested wedge** — agent-failure mechanistics is nascent; you bridge internals→agent outcomes where everyone else does post-hoc analysis or environment-side fixes.

Everything else is commoditized or a liability: SAE training (Goodfire $1.25B, Qwen-Scope, SAELens 200+), steering (now a documented jailbreak multiplier — Externalities 57-80%), feature cataloging (Neuronpedia owns it). The arXiv ban means you can't win on citation velocity, so you win on the orthogonal axis: **reproducibility** (field baseline 3% complete eval cards; >95% Croissant/OSF/Zenodo completeness is a credible solo-beats-institution lever).

**THE SINGLE HIGHEST-EV BET: Land Inspect Evals PR #1716 as a merged, reference-grade UK AISI eval before Q3 2026.** UK AISI distribution (200+ evals, adopted by Anthropic/DeepMind/Grok) > any blog post, any Zenodo DOI, any tweet. It is your single highest-distribution credential and it is *stalled on two fixable things*: the v5 bug and an undisclosed arXiv blocker. Unblock both this week.

---

## 4. RESEARCH AGENDA (prioritized — compound L11, not L55)

The thread that compounds is **L11 edge-layer drift as the WANDERING locus.** L55 has two nulls; L11 has three correlational confirmations (Paper #3 stability selection sel-freq 0.95 surviving L43 drop; Phase 4 run-to-run-stability AUC 0.725; the 1-of-5 pilot rescue at α≈1.15). Do not start anything new until the in-flight L11 work lands.

1. **Ship Paper #3 (Multi-Channel Signatures) to Zenodo + NeurIPS MI Workshop.** `findings.md` is complete with stats, confusion matrix, walk-back table, and the verified E2 thinking-length control (partial ρ=-0.289, p=0.0037). Only LaTeX narrative is missing. It independently confirms L11 and *refines Paper #2's nulls to "wrong layer," not "no mechanism"* — which is what makes the nulls publishable. Zero-draft state is pure execution risk in a closing niche.
2. **Run the gated L11 N=20 SUCCESS-donor injection (Task 241)** at pilot-best α≈1.15 with the norm-matched random control already in `exp_l11_donors.safetensors`. Pre-register rescue-rate + McNemar vs. no-hook baseline. A positive converts signature→causal locus, hits the funder causal-steering bottleneck (Schmidt, Coefficient) directly, and is unscoopable (needs your custom harness + donor residuals). **Pre-register honestly: if it nulls too, frame as "three convergent nulls + one localized positive trend." Do not let a possible null block Paper #3.**
3. **Port the full WANDERING phenotype (tool_diversity, L11_drift, L43_cosine) to existing Llama-70B and GPT-5 traces.** Paper #1 only cross-validated the v5 detector, not the full phenotype. Traces already exist — analysis-only cost. Every audit flags single-model fragility; this is the cheapest test of whether the moat survives off-Qwen.
4. **Apply Paper #2's 5 fixes and ship v0.3 — but hold final framing until L11 N=20 lands**, so the nulls read as correct localization, not failure.

**Do NOT** start new SAE-architecture work or crypto research now. That fragments the one compounding asset. Freeze.

---

## 5. TOOLS TO BUILD (named, with users)

Ordered by differentiation-per-effort:

1. **Inspect-native Monitorability Eval Pack** (lowest effort, highest fit) — finish PR #1716 + add an offline-checkpoint "monitorability eval" submodule wrapping v1/v5. *Users:* AISI/METR/labs running offline checkpoint risk-scoring under the NIST Q4 + EU Aug-2026 compliance tailwind (52% offline-eval adoption gap).
2. **SAE Causality Gauntlet** — one CLI command (`openinterp gauntlet`) running random-feature baseline + control-token-norm + α-sweep on any probe/SAE direction, emitting a Croissant/eval-card verdict. The Sanity-Checks paper made random-feature baselines mandatory (AUROC 0.87 vs 0.90) yet they're absent from SAELens/TransformerLens/nnterp. You already run this internally (6 walk-backs prove it). *Users:* every solo researcher facing ICML/NeurIPS reproducibility review.
3. **WANDERING-Atlas HF Space** — interactive SWE-bench trajectory replay with live per-turn detector firing + L11/L43 signals. The digest flags ZERO Spaces as THE adoption blocker (HF top 0.01% capture 49.6% of downloads via discoverability). Cheapest credibility/adoption fix available. *Users:* cold traffic, grant reviewers, the curious.
4. **AgentTrajectoryLens** (higher effort) — extract per-turn activation probing + the InterchangeTest primitive (currently manual Colab) from the harness as a library, with Llama/Gemma adapters. Captures the sub-market before Anthropic extends SAEs to agent reasoning.

**Do NOT build** net-new SAE-training tooling — SAELens/Qwen-Scope own that. Integrate them as backends, never compete.

---

## 6. SKILLS/ONBOARDING TO ADD (concrete repo artifacts)

The adoption blocker is that every artifact assumes you are Caio on Qwen3.6-27B. Concrete fixes, in order:

1. **Lift the Causality Protocol out of the MCP skill** (`openinterp-mcp/skills/causality-protocol/SKILL.md`) into a standalone `/docs/causality-gauntlet.md` decision-tree (the 5-verdict table + thresholds gap>0.1 / flip>10% / Δrel already exist) PLUS a copy-paste `causality_gauntlet.py`. Cite Sanity-Checks-for-SAEs. Publish on GitHub + Zenodo. *This is your only field-differentiated standard, currently locked where no external researcher will ever invoke it.*
2. **Publish 3-5 canonical probes** (L11 tool-entropy, L55, FabricationGuard, one documented honest-negative epiphenomenal probe) to `openinterp-community/probes` HF repo with `manifest.json` + `direction.safetensors`. Wire `list_probes` to query HF (currently returns empty on fresh session → "bring your own agent" has no working example). The honest-negative probe operationalizes the anti-Goodhart manifesto claim.
3. **One "first verdict in 30 min" Colab** using a *small public model (Gemma-2-2B or Qwen2.5-3B), NOT Qwen3.6-27B* — removes compute + model-lock barriers. Link from web `/start`.
4. **`notebooks/INDEX.md`** for the 71 notebooks (zero index today): one line each — purpose, paper target, compute tier, load-bearing vs. exploratory, + a "reproduce Paper #1 claim" ordered path.
5. **Restore CI** from the existing `*-saved-workflows/` dirs into `openinterp-swebench-harness` and `inspect-tool-entropy-collapse`; pin deps (`torch==`, `transformers==` + commit hash for transformers-main).

Ship all WITH the H-PROBE-EPIPHENOMENAL and run-stability caveats, or external reproduction fails loudly and damages credibility worse than zero adoption.

---

## 7. BLINDSPOTS (adversarial — what you are NOT seeing)

- **You mistake shipping volume for progress.** 11-12 papers, 321 memory files, 97 HF assets, 9 repos — and 0 stars, 0 downloads, 26 views. The portfolio's breadth is the *symptom*, not the achievement. One landed Inspect PR beats five more drafts.
- **The crypto pivot is a trap you've rationalized as a strategy.** Zero customer calls. The 519★ "traction signal" is misattributed to a repo you didn't write. You declared B+C primary on 2026-05-23, yet 70% of your actual outreach is still Anthropic/academic. Your behavior reveals your real priority; your memory contradicts it.
- **You cannot support causal-locus claims yet.** Three nulls + an N=5 mixed pilot + run-unstable WANDERING (temperature=1.0 stochasticity: Phase 6 0/20 vs independent no-hook re-run 7/20 vs fresh determinism check 0/5, all on RTX 6000 — no H100) is not causal evidence. If you over-claim causality before the clean N=20 test, you destroy the *one* moat — and a rival publishing the run-instability first makes Paper #1's N=20 look fragile.
- **Three competing Zenodo DOIs for Paper #1** (20368601, 20368807, 20368600 appear across digest) split your citation graph and contradict the reproducibility-leader positioning that is your entire arXiv workaround. Collapse to one canonical concept DOI.
- **MALT p-value discrepancy** (abstract 0.81 vs JSON ~0.69) is an unexplained published inconsistency a replicator will surface. Reconcile or issue an erratum before someone else finds it.
- **Bus-factor-1.** Single author, 0 forks, 65/68 runs lack verdicts, 178GB unindexed. A 3-month pause makes the org read as abandoned.

---

## 8. 90-DAY ROADMAP (week-by-week)

**Weeks 1-2 (now–Jun 11) — UNBLOCK + the one credential window:**
- **Today:** Fix the v5 entropy=0.0 conflation bug (guard on `len(names)>0`); flip the stale `test_detectors.py:63` assertion to expect FIRE; add dataset.py context-managers. Cut v0.3.1 with a Table-1 caveat.
- **This week:** Resolve PR #1716's arXiv blocker — openly disclose the ban to @celiawaggoner, offer Zenodo concept DOI + 4 ICML reviews + ORCID as the reproducibility credential.
- **By May 30 EOD:** Send the day-14 Jack Lindsey follow-up (window closes), then drop per protocol. Deploy the drafted-but-unsent Apollo/AISI institutional emails.
- **By Jun 12:** Submit ONE polished paper (Paper #1 resubmit OR Paper #3) to ICML MI Workshop — non-archival, accepts Zenodo + code-first, sidesteps the arXiv ban. Highest-credibility credential reachable in 90 days.
- Apply Paper #2's 5 fixes; reconcile the MALT p-value.

**Weeks 3-4 — convert dead assets to traction:**
- Draft and ship Paper #3 to Zenodo (findings complete; only narrative missing).
- Deploy WANDERING-Atlas + FabricationGuard HF Spaces; write complete model cards for top 5-7 assets; collapse Paper #1 to one DOI.
- Start 1 blog post/week on openinterp.org/blog (LessWrong substitute) — methodology lessons + the 6 honest walk-backs.

**Weeks 5-8 — reproducibility infra + validate-or-kill crypto:**
- Restore CI + pin deps across the 5 repos; publish 3-5 canonical probes; ship the Causality Gauntlet + reference card.
- Run the gated L11 N=20 causal test (disclose hardware, validate 5-10 iids on a 2nd GPU).
- Submit ONE grant (Schmidt or Coefficient — they fund activation-monitoring exactly) framed as offline monitorability evals + reproducibility infra, NOT as AgentGuard revenue.
- Run 3 Tier-1 AgentGuard ICP calls (Will Papper verified email first). **No call in 2 weeks → strip AgentGuard from all grant narratives.**

**Weeks 9-12 — generalize + consolidate:**
- Cross-model phenotype replication (Llama-70B/GPT-5 traces); land the Inspect PR merge; build the experiment manifest + backfill 65 verdicts; consolidate scattered side-papers into one portfolio narrative.

**Sequencing rule:** ICML submission is non-negotiable #1. The v5 fix + PR unblock are the gating credibility moves. Defer Paper #2 fixes if Weeks 1-2 collide.

---

## 9. WHAT CAIO SHOULD DO MONDAY MORNING

**Fix the v5 `tool_entropy_last_n` bug and ship inspect-tool-entropy-collapse v0.3.1 — before touching anything else.** Guard the firing condition on `len(names)>0` so a single repeated tool (entropy=0.0) FIRES per Paper #1 §6, separate from "no tools" (no fire); flip the `test_detectors.py:63` assertion and delete the TODO; add a test asserting single-tool fires at threshold=0.5. Then post the patch to PR #1716 with a one-line note disclosing the arXiv ban and offering the Zenodo DOI + ICML reviews as the reproducibility credential.

This is a one-to-two-hour fix on your single highest-distribution asset, where the published test currently asserts the *opposite* of what the code does — a contradiction any reviewer running pytest will hit. It unblocks the highest-EV bet (the merged UK AISI eval), protects the reproducible-metrics core that is your *only* real moat, and converts a credibility landmine into a credibility signal. Everything else this quarter compounds off a landed PR; nothing compounds off a broken one.