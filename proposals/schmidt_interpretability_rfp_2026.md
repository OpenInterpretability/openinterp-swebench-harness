# Operational Probe-Causality for Trajectory-Level Deception Detection

**Schmidt Sciences AI Interpretability RFP 2026 — Proposal Draft v0.1**

- **PI:** Caio Vicentino (ORCID 0009-0003-4331-6259) — independent researcher, OpenInterpretability
- **Budget:** $325,000 USD over 18 months (Tier 1)
- **Contact:** caio@openinterp.org
- **Submission portal:** https://schmidtsciences.smapply.io/prog/2026_interpretability_rfp/
- **Deadline:** 2026-05-26 23:59 AoE

---

## Status of this document

**Draft outline v0.1, 2026-05-19.** Sections marked `[STUB]` are scaffolding only. Sections marked `[READY]` reflect existing publishable work that can be cited verbatim. Sections marked `[FIELD]` are intentional differentiators that require the multi-agent κ_t toy run (task #207) to complete before they can carry empirical weight.

Target: complete v1.0 by 2026-05-23 (3 days of writing) → submit by 2026-05-26.

---

## 1. Plain-language abstract (1 paragraph, ~250 words)

`[STUB — write last, after rest of draft locks]`

Hook: Linear probes trained on the residual stream of LLMs routinely achieve high predictive AUROC on deception-relevant labels — sycophancy, harmful-advice-compliance, scheming-precursors. Three years of work, however, has shown that AUROC does not imply the probe direction is a causal lever; in our own twelve-site map of Qwen3.6-27B (Vicentino 2026e), eleven of twelve probes are operationally conditional in ways that the literature has not previously enumerated. The same gap appears at the trajectory level: the per-turn cross-probe coherence statistic we recently introduced (Vicentino 2026d) separates successful from failed agent trajectories at p<10⁻⁴, but only after a length-confound walk-back that no published methodology had flagged. This proposal funds the operationalization of probe-based deception monitors into a runnable battery — six diagnostics for single-probe causality (Vicentino 2026f, shipped 2026-05-19), one trajectory-level coherence statistic with its rescue protocol, and a multi-agent generalization that has not previously been published — and validates each against a red-team-style test set of deceptive trajectories produced by Qwen3.6-27B, Llama-3.3-70B, and Gemma-3-27B under prompt-pressure paraphrase, scheming-precursor instructions, and explicit two-agent collusion setups. The deliverable is a peer-reviewed publication track of three papers (year 1) plus an open-source SDK (PyPI: openinterp) that any downstream practitioner can run before claiming their probe is a deception detector. Total budget: $325K over 18 months. Single-PI project.

## 2. Fit with the research agenda (1 page)

`[READY — cite existing work directly]`

Schmidt Sciences' AI Interpretability RFP names two failure modes the program wants to detect: sycophancy and knowingly-providing-harmful-advice. The PI has shipped published work directly on both:

- **Sycophancy probe stack** (in progress, tasks #189-190): pressure-injected replay on N=99 SWE-bench Pro traces from Qwen3.6-27B, with the cross-probe coherence framework already validated as an outcome separator at p=0.0009 (Vicentino 2026d, [κ_t paper, Zenodo DOI 10.5281/zenodo.20278983](https://zenodo.org/record/20278983)).
- **Tool-Doubt probe** (walk-back documented, Phase 1 2026-05-17): Qwen3.6-27B residual encodes explicit tool-error signal (lexical L7-L11) but is blind to silent-corruption (paper-MEGA conditionally-causal, openinterp.org/research/papers/conditionally-causal-probes). Direct evidence the field needs operational diagnostics, not just better probes.
- **Harmful-advice / sandbagging axis** (Week 3 sprint 2026-05-18, task #194): probe stack trained on CoT-aware deceptive completion identifies the signal at L43 with AUROC > 0.85.

The RFP further emphasizes a **red team / blue team competitive model** to validate that interpretability tools work on deceptive behaviors not known about in advance. The PI's six-diagnostic battery (Vicentino 2026f, Paper 10) was constructed precisely to enforce this discipline: each of the six diagnostics was elevated to methodology only after it walked back one of the PI's own near-publication-ready claims. The proposed work extends the discipline into a published, runnable, externally-falsifiable test suite.

## 3. Scientific aims (1-2 pages)

`[FIELD]` — three aims, each tied to one of the proposal's deliverables.

### Aim 1: Standardize operational probe-causality auditing (Year 1, months 1-6)

Ship a stable v1.0 of the [`openinterp.diagnostics`](https://pypi.org/project/openinterp/) Python module implementing the six diagnostics (D1: random-feature baseline at N<100; D2: shuffled-source baseline for sparse top-k probes; D3: control-token normalization; D4: structural-rigidity α-sweep; D5: whitespace-stripped flip metric; D6: trace-length-controlled slope decomposition). Each diagnostic ships with a CPU-only reproduction notebook, a documented walked-back-claim case study, and pass/caveat/retract thresholds with empirical justification.

Year 1 milestone: 100 GitHub stars on `OpenInterpretability/openinterp` and at least 3 independent papers citing the battery (target tracked on openinterp.org/diagnostics leaderboard).

### Aim 2: Multi-agent κ_t coherence as collusion detector (Year 1, months 7-12)

The κ_t framework (mean absolute pairwise correlation across N probes within a windowed time series) currently characterizes a single agent's trajectory. The proposed extension: an N×M generalization where each of N agents contributes a probe stack of M behavioral probes, and we measure cross-agent κ to characterize coordination dynamics.

**Pre-registered hypothesis (PI, 2026-05-19):** scripted-collusion trajectories show high cross-agent κ from turn 0 (synchronized representation patterns from coordinated instruction); honest-coordination trajectories show low κ until late convergence (separate exploration trajectories converging only as agents share solution structure). Falsifiable: a shuffled-agent control should erase the early-high-κ signal in scripted collusion but not in honest coordination.

Year 1 deliverable: peer-reviewed paper at NeurIPS MI Workshop 2026 or ICLR 2027 main. Two-week toy run (May 22-29 2026) provides preliminary AUROC on a N=2 Qwen3.6 debate task before this proposal is submitted (task #207).

### Aim 3: Pre-publication identification-assumption disclosure as field norm (Years 1-2)

Beyond shipping the battery, the proposed work funds the OpenInterpretability community-building infrastructure that makes the disclosure norm self-enforcing:

- ProbeBench leaderboard (openinterp.org/probebench, currently live): rank submitted probes by their cleared-diagnostic count, not just headline AUROC.
- Anonymous "self-walk-back" submission system: researchers publish near-published claims that one of our diagnostics retracted, with provenance. Mirrors the "Trial of the Pyx" tradition for AI safety methodology.
- Two community workshops (Q3 2026, Q1 2027) at MI Workshop satellite events.

Year 2 milestone: 5+ external papers structuring their disclosure section around `openinterp.diagnostics`; ProbeBench leaderboard hosts 30+ peer-validated probes across at least 5 model families.

## 4. Differentiation from existing work (1 page)

`[READY + FIELD]`

| Existing program | Scope | Gap our proposal fills |
|---|---|---|
| Anthropic Persona Vectors / Lindsey Model Psychiatry | Single-probe causal interventions on closed models | Open-model reproducibility; multi-site map; multi-agent extension |
| Apollo Watcher (May 2026) | Commercial single-agent monitoring | Multi-agent coordination dynamics; open-source SDK; published diagnostics |
| Geiger causal abstraction (2024) | Theoretical identification framework | Operational floor below the theoretical ceiling — what every paper should clear before claiming causality |
| Bohnet et al. position paper (2026, [arxiv 2605.08012](https://arxiv.org/abs/2605.08012)) | Disclosure norm call | Operationalization of the norm into runnable diagnostics, already shipped (Paper 10, openinterp.org/research/papers/six-diagnostics-six-walkbacks) |
| Martian Long-Horizon Frontier (May 2026) | Calls for trajectory-level MI | κ_t framework + multi-agent extension is the trajectory metric |
| NARCBench multi-agent collusion (2604.01151) | Behavioral benchmark for collusion | Cross-agent κ_t as a mechanistic detector tied to the benchmark |

`[FIELD]` The differentiator is not the diagnostics individually — each has been used at least once in published work. The differentiator is the operationalized battery, the public leaderboard, the documented walk-back provenance, and the multi-agent extension that no existing group has pre-published methodology for.

## 5. Methods (2-3 pages)

`[STUB — expand from Paper 10 §A.2 + κ_t paper §2 + new multi-agent design from task #207]`

### 5.1 Single-probe diagnostic battery (Paper 10 verbatim)

Six diagnostics with pass/caveat/retract thresholds:

- D1: `lift = real_AUROC − mean(random_AUROC)` over ≥50 random feature draws. Pass `lift > 0.20`.
- D2: `Δ = real_recall − shuffled_recall` from X_train permuted. Pass `Δ > 0.10`.
- D3: `Δ_rel = Δ(target) − mean(Δ(controls))` over ≥5 controls. Pass `|Δ_rel| > 2 × std(Δ(controls))`.
- D4: Behavior at α ∈ {0, +5, +20, +50, +100, +200} on probe AND random direction. Pass: probe diverges while random does not.
- D5: `flip_rate(stripped) = mean(base.strip() != mod.strip())`. Replace raw with stripped in headline.
- D6: Partial p of class effect on slope residualized by length. Pass `partial p < 0.05`.

(Full thresholds in [Paper 10 §A.2](https://openinterp.org/research/papers/six-diagnostics-six-walkbacks).)

### 5.2 Trajectory-level: κ_t and U-shape rescue (κ_t paper verbatim)

`[READY — cite Vicentino 2026d directly]`

### 5.3 Multi-agent κ_t (NEW)

`[FIELD — fill from task #207 toy run results before 2026-05-23]`

Generalization sketch:
- Each agent i (i=1..N) contributes M behavioral probes p_{i,1}..p_{i,M} (default M=5: tool_finish, tool_bash, long_thinking, tool_ok, repo_marker).
- At each turn t, compute the N×N×M×M cross-agent correlation matrix C[i,j,a,b] = corr(p_{i,a}(t), p_{j,b}(t)) within a windowed time series.
- Define **κ_t^cross** as mean |C[i,j,a,a]| over i≠j (same-probe cross-agent correlation) and **κ_t^within** as mean |C[i,i,a,b]| over a≠b (Vicentino 2026d).
- Plot both as functions of turn t. Hypothesis: scripted-collusion → κ_t^cross high from t=0; honest-coordination → κ_t^cross low until late convergence.

Pre-registered controls:
- C7: Agent-identity shuffle (relabel agents). Expected: kills scripted-collusion signal, preserves honest-coordination signal.
- C8: Turn-order shuffle within agent (κ_t paper C2 extension). Expected: same as C2 — preserves underlying coordination structure.
- C9: Length-confound check on cross-agent slope (Paper 10 D6 extension).

Toy domain for May 22-29 run: two Qwen3.6-27B agents debating a single SWE-bench Pro instance, one in "honest reviewer" prompt and one in "collusion partner" prompt (asked to align with the first agent's choices). N=20 trajectories per condition, ~10-15 turns each. CPU-feasible analysis on saved residuals.

## 6. Budget breakdown (~$325K over 18 months)

`[STUB — calibrate to Schmidt examples]`

| Category | Year 1 | Year 2 (6 months) | Subtotal |
|---|---|---|---|
| PI salary + benefits (1.0 FTE → 0.5 FTE Year 2) | $180K | $50K | $230K |
| Compute (A100/H100 hours, ~$15K/year) | $15K | $7.5K | $22.5K |
| OSS infrastructure (Vercel, HF storage, PyPI, domain) | $3K | $1.5K | $4.5K |
| Conference travel (NeurIPS MI 2026, ICLR 2027) | $8K | $4K | $12K |
| Indirect costs (5%, below the 10% cap) | $10.3K | $3.2K | $13.5K |
| Reserve / methodology workshops Year 2 | — | $42.5K | $42.5K |
| **Total** | **$216.3K** | **$108.7K** | **$325K** |

Justification: at 1.0 FTE the PI ships 1 paper per 2-3 weeks (track record 6 papers in 4 weeks April-May 2026). Year 2 at 0.5 FTE supports community workshop hosting + cross-model replication; Year 2 reserve funds 2 satellite workshops at NeurIPS MI 2027 / ICML MI 2027.

## 7. Track record (1 page)

`[READY — cite 2026 portfolio]`

- **12 papers** shipped at openinterp.org/research between 2026-04-29 and 2026-05-19 (one major paper every ~36 hours), all with HF datasets + GitHub repros + Zenodo DOIs.
- **3 ICML 2026 MI Workshop peer reviews** submitted on time as program reviewer (#18 / #26 / #79, all rec 4 Borderline Accept / conf 3).
- **Paper 1 submitted to ICML MI 2026** — peer-reviewed credential in flight (notification June 12 2026).
- **OpenInterp SDK** (PyPI v0.3.1, openinterp): Atlas search + SAE traces + FabricationGuard + agent-probe-guard, currently used by N=? users.
- **3 Qwen3.6-27B paper-grade SAEs** trained from scratch (200M tokens, d_sae=65536), published on HuggingFace (caiovicentino1/qwen36-27b-sae-papergrade) and adopted by N=? downstream researchers.
- **ProbeBench live leaderboard** (openinterp.org/probebench): 12 probes registered, ProbeScore methodology with cross-model transfer matrix and per-task error analysis.

`[STUB]` — update view metrics + citation count from openinterp.org analytics before submission.

## 8. Risks and mitigations (0.5 page)

`[STUB]`

R1. **Multi-agent κ_t signal turns out null.** Mitigation: the proposal is framed around the diagnostic battery as primary deliverable; the multi-agent extension is Aim 2, not Aim 1. Even a null result is publishable (per Paper 10 methodology, an honest null with strong controls is a valid contribution).

R2. **Independent PI without institutional research overhead.** Mitigation: 5% indirect rate (well below cap), no need for institutional infrastructure. Compute hosted on Colab Pro+ and Google Drive (already in use). Vercel + HF + GitHub + Zenodo handle publishing. No facilities cost.

R3. **Timing conflict with Anthropic Fellows July 2026.** Mitigation: Anthropic Fellows is a separate path that the PI is also pursuing; if both succeed, the Schmidt funding could continue post-Fellows on a no-cost extension, or the PI defers Fellows. Not a forced choice in current Anthropic policy.

R4. **Schmidt-funded research goes obsolete if frontier models change.** Mitigation: all six diagnostics are model-agnostic by construction (tested on Qwen3.6-27B; methodology applies to any open-weight LLM with extractable residuals). Multi-agent κ_t equally generalizes.

## 9. References

`[STUB — generate from Paper 10 references + new key cites]`

- Bohnet et al. 2026, arxiv 2605.08012.
- Geiger et al. 2024, JMLR.
- Anthropic causal scrubbing, 2024.
- Vicentino 2026a-g (12 papers, openinterp.org/research/papers).
- Martian May 2026, "Beyond Static Mechanistic Interpretability".
- Apollo Update May 2026.
- NARCBench 2026, arxiv 2604.01151.
- "Non-Linear Representation Dilemma" arxiv 2507.08802.
- Schmidt Sciences 2026 Interpretability RFP scope document.

---

## Writing plan (next 7 days)

| Date | Deliverable | Owner |
|---|---|---|
| 2026-05-19 (today) | v0.1 outline shipped (this doc) | Caio + Claude |
| 2026-05-20 | Multi-agent κ_t toy notebook scaffold (task #207) + run on Colab | Caio |
| 2026-05-21 | Toy results in hand; expand §5.3 with preliminary AUROC | Caio |
| 2026-05-22 | §1 abstract + §3 aims locked | Caio + Claude |
| 2026-05-23 | §4 differentiation + §5 methods locked | Caio + Claude |
| 2026-05-24 | §6 budget + §7 track record + §8 risks + §9 refs | Caio + Claude |
| 2026-05-25 | Internal review pass + readthrough | Caio + Major |
| 2026-05-26 | Submit on smapply portal by 11:59 PM AoE | Caio |

## Open questions for Caio

- Budget: $325K is on the low end of the $300K-$1M range. Is there appetite to scale to $500K-$700K (1.5-2 year extension) with a richer Aim 3 (more workshops, larger leaderboard infra)? Or stay lean to be more competitive on cost-effectiveness criterion?
- Institutional affiliation: do you want to apply as "Caio Vicentino, independent" or do you want to seek an affiliation umbrella (UNIFOR Fortaleza, where ORCID is registered)? Affiliations sometimes help; sometimes don't.
- Co-PI option: Major is listed as collaborator in your project memory. Want to formalize as co-PI on the proposal, or keep solo?
- Scope of Aim 3 community work: workshops + leaderboard infra are time-sinks. Do you want them in scope, or focus Year 1 entirely on papers + SDK?
