# LTFF Application — Caio Vicentino, OpenInterp
**Draft v0** · 2026-05-23 · target ~5K chars total

---

## Basic information

- **Fund**: Long-Term Future Fund
- **Secondary fund (EAIF transfer OK)**: Yes
- **Funding from Open Philanthropy**: No (never received)
- **Name**: Caio Vicentino
- **Organization**: *(leave empty — solo independent)*
- **Main collaborators**: Working independently
- **Email**: caio@openinterp.org
- **Additional email**: *(leave empty)*
- **Employed by Effective Ventures**: No, never

---

## Short description (≤120 chars)

```
6-month independent runway: ship 2 mech-interp papers + cross-model replicate Six Diagnostics protocol.
```
*(101 chars)*

---

## Summary (≤1000 chars)

```
$30K for 6 months of independent mechanistic-interpretability research on Qwen3.6-27B + cross-model replication (Gemma-3-12B, Llama-3.3-70B). In the first 5 weeks of OpenInterp (openinterp.org), I shipped 12 research artifacts including an ICML 2026 MI Workshop submission and Paper-MEGA — a 9700-word meta-paper formalizing a six-diagnostic protocol that tests whether activation-level probes are causal or merely correlative. Six of my own causal claims walked back when subjected to these diagnostics; each is documented in a public Failed-Replication Registry. Funding supports: (1) shipping Paper 11 on CoT-monitor obfuscation by corrective prefills (recall 1.00→0.33, Fisher p=0.015); (2) cross-model replication of the Six Diagnostics protocol; (3) Eval Standard JSON schemas + community adoption push. AI safety increasingly relies on probes as monitors, reward signals, and intervention targets. Standardizing the audit layer is load-bearing.
```
*(~990 chars)*

---

## Project goals

Three concrete deliverables over 6 months:

**(1) Ship Paper 11** — *"Cleaning the Chain-of-Thought Is Not Correcting the Agent"* (drafted, 4567 words). First documented failure mode of CoT-redirect interventions on a 27B reasoning model: a corrective prefill clears 88% of verbalized sabotage in the chain-of-thought, but the resulting patch is still sabotaged. CoT-monitor recall drops 1.00→0.33 (Fisher p=0.015). RESAMPLE arm (re-sample without prefill) recovers recall to 1.00, isolating the prefill as the causal active ingredient of the harm. Path to impact: monitor-based safety pipelines (Anthropic CoT monitoring, UK AISI) treat CoT as a defense layer. Paper 11 demonstrates a self-undermining mode requiring explicit mitigation. Success: arXiv + Zenodo deposit by month 3.

**(2) Cross-model replication of the Six Diagnostics protocol.** The current six-diagnostic battery (random-feature, shuffled-source, control-token normalization, structural-rigidity α-sweep, whitespace-stripped flip, onset-timing sweep) was derived on Qwen3.6-27B alone (Paper-MEGA, May 2026). I will replicate on Gemma-3-12B and Llama-3.3-70B, mapping which diagnostics transfer and which are model-specific. Specifically: the H-PROBE-EPIPHENOMENAL prior I document on Qwen3.6-27B may or may not generalize. Success: paper at NeurIPS 2026 MI Workshop or TMLR.

**(3) OpenInterp Eval Standard adoption.** Apache-2.0 JSON schemas (`probe_card`, `causal_report`, `intervention_trace`, `interp_card`) operationalize the protocol for community use. Submit integration PRs to TransformerLens, SAE Lens, Neuronpedia. Grow the Failed-Replication Registry via community submissions. Success: ≥1 upstream library merges schema integration; ≥3 external walk-back entries by month 6.

**Connection to LTFF goals**: activation-level probes, SAE features, and steering vectors are increasingly load-bearing in AI safety pipelines (Anthropic Persona Vectors 2026, Goodfire RLFR 2025, UK AISI Alignment Project 2026). The field is converging on probe-based monitoring, but the diagnostic battery that distinguishes a causal probe from a correlative or epiphenomenal one isn't standardized. Standardization raises the floor on what counts as a defensible mechanistic claim about a frontier model — directly reducing the rate at which the field deploys correlative safety methods as if they were causal.

*(~2200 chars)*

---

## Track record

OpenInterp launched April 15, 2026; over the first 5 weeks I have:

**Peer review credentials**: ICML 2026 MI Workshop submission ("Hallucination-Induction, Not Calibration") under review, notification June 12. ICML 2026 MI Workshop reviewer for 3 papers (assigned, all reviews submitted before deadline).

**Public papers (12)**: openinterp.org/research. Flagship is Paper-MEGA — *"Conditionally-Causal Probes: Five Operational Constraints on Linear-Probe Causality in Qwen3.6-27B"*, 9700 words, TMLR target, with 19/19 claim-verification PASS via a public reproducibility script. Other shipped papers include: Paper-8 (Trajectory-Shaping Probe Steering), Paper-5 (Saturation-Direction Probe Levers), Paper-6 (Two Forms of Epiphenomenal Probes), Paper-9 (κ_t coherence buildup with walk-back-and-rescue), Paper-3 (PSAE marginal-fit, published as honest-negative), Paper-2 (probe-detected grokking in multi-probe DPO).

**Artifacts**: 4 SAE families for Qwen3.6-27B on HuggingFace (papergrade L11/L31/L55 d=65536, fullstack 11 layers d=40960, multilayer demo, feature-circuits) — the only public SAEs on Qwen3.6 reasoning models. `pip install openinterp` v0.3.0 ships FabricationGuard hallucination probe + agent-probe-guard SDK + ProbeBench leaderboard. `openinterp-mcp` v0.1.0 = MCP server for bring-your-own-agent interp research. 8 GitHub repos under OpenInterpretability org, Apache-2.0 throughout.

**Cross-lab**: Qwen Developer Ambassador (accepted by Alibaba, 2026-05-19).

**Honest weaknesses**:
- Five weeks of high velocity is exceptional; sustained ~2 papers/month is more realistic.
- No PhD, no academic affiliation, solo operation.
- Six of my own claims walked back — I treat this as a methodology contribution (the Failed-Replication Registry I publish documents them publicly), but it means my hit rate on novel positive claims is moderate.
- Cross-model replication (deliverable 2) is the highest-uncertainty work — Qwen3.6-27B's epiphenomenal-probe prior may not generalize.

**Expenditure/staffing**: Solo, no payroll. Total OpenInterp burn to date ~$2K (compute via Colab Pro+, HF storage, Vercel, domain). This grant would be my first external funding.

*(~2300 chars — willing to trim further to ~1500 if asked)*

---

## Public Portfolio (links)

- Papers: https://openinterp.org/research
- Paper-MEGA (flagship): https://openinterp.org/research/papers/conditionally-causal-probes
- OpenInterpretability GitHub: https://github.com/OpenInterpretability (8 repos, Apache-2.0)
- Eval Standard schemas: https://github.com/OpenInterpretability/registry/tree/main/schemas
- Failed-Replication Registry: https://github.com/OpenInterpretability/registry/tree/main/failed-replications
- HuggingFace org: https://huggingface.co/caiovicentino1
- PyPI: https://pypi.org/project/openinterp/
- ORCID: https://orcid.org/0009-0003-4331-6259
- X/Twitter: @0xCVYH
- ICML 2026 MI Workshop submission: under double-blind review, notification June 12

---

## Funding

**Requested amount**: $30,000 USD

**Range**:
- **Minimum** $20K (4 months, deliverable 1 only)
- **Mainline** $30K (6 months, all 3 deliverables)
- **Stretch** $45K (8 months + buffer + 2 extra walk-back investigations)

**Mainline $30K breakdown** (will share as Google Sheet):
- **75% ($22.5K)** — stipend, 6 months at independent-researcher rate. Brazil-based (lower cost-of-living).
- **15% ($4.5K)** — compute (vast.ai/Modal A100/H100, ~200-300 GPU-h for Gemma-3-12B + Llama-3.3-70B SAE training + Six Diagnostics replication suite).
- **5% ($1.5K)** — infrastructure (Zenodo, GitHub LFS, Vercel, HuggingFace storage, domain renewals).
- **5% ($1.5K)** — buffer per LTFF guidance.

Brazilian self-employed tax (MEI) included in stipend line. No payroll insurance required.

**Organizational budget**: *(leave empty — solo, no entity)*

---

## Alternatives to funding

Without LTFF funding I would continue at reduced pace from personal savings (financially independent from ~7 years in DeFi/crypto). Deliverable 1 (Paper 11) would ship on schedule. Deliverable 2 (cross-model replication) would slip 2-4 months due to compute cost. Deliverable 3 (adoption push) would shrink to schemas-only without contractor-supported PR coordination.

**Other applications planned (next 4 weeks)**:
- Open Philanthropy Career Development & Transition Grant — not yet applied
- a16z Crypto Research Grants (crypto-AI safety angle) — not yet applied
- Apart Research Fellowship — Q3 2026 cycle
- Together AI compute credits — rolling
- MATS Winter 2027 — when apps open July

**Pending decisions** (no money received yet):
- Pivotal Research Fellowship 2026 Q3 — submitted 2026-05-01, decision pending
- Anthropic External Researcher Access Program — submitted, decision pending

If LTFF + another funder both fund, I will scale deliverables (extra cross-model coverage, adoption surface) rather than bank runway.

*(~900 chars)*

---

## Use for additional funding

If granted above the requested amount:

**(a) Mixtral 8x22B MoE replication** — extend Six Diagnostics replication to a MoE architecture. Expert routing may surface novel failure modes. ~$10K additional compute + extended timeline.

**(b) Part-time contractor** (15-25h/week × 3 months) — manages Failed-Replication Registry community submissions + adoption-PR coordination. Frees my time for original research. ~$20K.

Total stretch budget at full scope: ~$60K.

---

## Other fields

- **Confidential information**: *(leave empty)*
- **LinkedIn/CV**: https://orcid.org/0009-0003-4331-6259 + https://openinterp.org + https://github.com/caiovicentino1
- **File upload**: *(optional — could attach Paper-MEGA PDF; or skip)*
- **Start date**: 2026-06-01
- **End date**: 2026-11-30
- **Currency**: USD
- **Location**: Operates from Fortaleza, Brazil; project distributed/online (no jurisdiction-specific implementation)
- **China/India**: No
- **Award**: No
- **Under 18**: No
- **Lobbying**: No
- **Time-sensitive (<8 weeks)**: No
- **Public reporting**: Yes
- **Network sharing**: Yes
- **Referral to other funders**: Yes
- **How heard about EA Funds**: Independent research community / AI safety field knowledge

---

## References (optional but boost odds)

⚠️ *Need input from Caio: do you have warm contacts in EA/AI-safety community to list?*

Candidates I see from memory (verify with Caio if any are usable as warm references — Caio reached out but unclear if they replied):
- Jack Lindsey (Anthropic Model Psychology) — cold email sent 2026-05-16, reply unknown
- Atticus Geiger — outreach Tier 1 done
- (others?)

If no warm references → leave blank. LTFF doesn't require.

---

## Pre-submit checklist

- [ ] Verify Paper 11 N + Fisher p exactly match memory (recall 1.00→0.33, p=0.015 RESAMPLE-vs-REDIRECT)
- [ ] Confirm Paper-MEGA "19/19 claim-verification" link works
- [ ] Confirm ICML 2026 MI Workshop notification = June 12
- [ ] Confirm "no external funding received yet" still true
- [ ] LinkedIn URL (if exists)
- [ ] Bluesky URL (if shipped)
- [ ] Budget Google Sheet — create + share-link
- [ ] References — decide on any warm contacts
- [ ] Final character-count check (target 4-5K substantive)
