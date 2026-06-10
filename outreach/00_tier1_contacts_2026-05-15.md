# Tier 1 Outreach Contact List — 2026-05-15

> **Scope decided 2026-05-15**: Tier 1 (~10 high-value targets, Anthropic researchers excluded). Targeted research engagement, not mass outreach. Rationale: Caio is actively applying for Anthropic Fellows; cold-emailing Anthropic interp team during application = manoeuvre risk. Out-of-Anthropic targets give methodology feedback + potential ML-community reference candidates without that conflict.
>
> **Cadence**: 1 message per day across 10 days. NOT mass-CC. Individual messages.
> **Tone**: cite their specific work + your specific result that engages with it. Ask for 15 min input, not endorsement. Never mention Fellows app explicitly.
> **No invented emails**: where personal email not memory-verified or known public, look up on lab page / arXiv author info / X bio. Bounce = bad signal.

---

## Status legend

- 🟢 **In flight** — first message sent, awaiting response
- 🟡 **Drafted, not sent** — message file exists, ready to dispatch
- ⚪ **Not contacted** — needs message draft + email lookup

---

## 1. Marius Hobbhahn — Apollo Research

- **Status**: 🟢 X DM sent 2026-05-09 to @MariusHobbhahn. Awaiting response (see `project_outreach_apollo_aisi_log.md`).
- **Apollo institutional email**: `research@apolloresearch.ai` (memory-verified, draft `03_email_apollo_research.txt` pending dispatch)
- **Why him**: Apollo's deception probes + eval-awareness work is closest methodological neighbor to FabricationGuard + ProbeBench. Already in 1st-contact channel.
- **Next action**: Wait for DM response. If silent by 2026-05-23 (14d), send the institutional `research@apolloresearch.ai` follow-up (draft already exists). DO NOT pile on with multiple messages.

## 2. Geoffrey Irving — UK AI Safety Institute (AISI)

- **Status**: 🟢 X DM sent 2026-05-09 to @geoffreyirving. Awaiting response.
- **AISI institutional email**: `research@aisi.gov.uk` (memory-verified, draft `04_email_aisi_research.txt` pending dispatch)
- **Why him**: AISI's reproducibility + methodology focus aligns with the 4-sanity-check stack in paper-5. Direct institutional fit.
- **Next action**: Same as Apollo — wait for DM, send institutional follow-up by 2026-05-23 if no response.

## 3. Neel Nanda — Google DeepMind (verify current affiliation)

- **Status**: ⚪ Not contacted.
- **Email lookup**: His personal site (neelnanda.io) usually lists contact. Also active on X (@NeelNanda5) and his Substack.
- **⚠️ VERIFY**: Per memory, my last-known affiliation is GDM. If he moved to Anthropic, **skip entirely** (Anthropic-during-Fellows rule).
- **Memory link**: Caio's crosscoder paper cites Minder/Dumas/Juang/Chughtai/**Nanda** NeurIPS 2025 (arxiv:2504.02922) — direct prior art relationship.
- **Why him**: MATS director, central node in MI network, actively surfaces independent researchers. Highest single-contact value if affiliation cleared.
- **Hook**: "Independent researcher, ICML 2026 MI Workshop submission. Your crosscoder Latent Scaling paper (Minder et al. 2025) is prior art for my Pearson_CE crosscoder analysis on Gemma-2-2B base/IT (median cosine 0.965 vs causal Pearson 0.616 across shared features). Would value 15min input."

## 4. Sam Marks — ❌ SKIP (Anthropic 2024-25, verified 2026-05-15)

- **Status**: 🔴 **REMOVED FROM LIST 2026-05-15.** OpenReview profile shows "Researcher, Anthropic 2024–2025" as most-recent affiliation. Google Scholar still lists northeastern.edu email but career history is Anthropic. Per anti-mistake #1 (Anthropic-during-Fellows-app rule), skip entirely for the duration of the Fellows app review window.
- **Revisit**: post-Fellows decision (likely H2 2026) if outcome allows.

## 5. Nora Belrose — EleutherAI (verify current affiliation)

- **Status**: ⚪ Not contacted. Not memory-verified.
- **Email lookup**: EleutherAI public contact channels or her X (@norabelrose).
- **Why her**: Tuned-lens lineage; outspoken on probe failure modes; lineage matches paper-5's "structural fragility" class (class 3 of the 5-class taxonomy).
- **Hook**: "Your tuned-lens work informed how I framed layer-wise probe causality in paper-5. Class 3 of my 5-class taxonomy (structural fragility) describes layers where random direction matches probe direction's behavioral effect — essentially a tuned-lens-adjacent observation. Would value your read."

## 6. Atticus Geiger — Goodfire (verified 2026-05-15 via atticusg.github.io)

- **Status**: 🟢 Email SENT 2026-05-16. Awaiting response. (Email: `atticus@goodfire.ai`, verified on atticusg.github.io 2026-05-15.)
- **Affiliation update**: Moved from Stanford/Pr(Ai)²R to Goodfire (independent applied-interpretability startup). NOT Anthropic. Safe to contact during Fellows window.
- **Why him**: Causal abstraction framework (JMLR 2025) is the theoretical lens for distinguishing detection from causation in probes. Direct fit with paper-6 "Two Forms of Epiphenomenal Probes." Goodfire's applied-interp infrastructure also adjacent to OpenInterp's positioning.
- **Hook**: "Your causal abstraction framework helped me formalize when a probe levers vs detects. Paper Two Forms of Epiphenomenal Probes documents 2 distinct mechanisms (softmax-temperature artifact at L43; chat-template lock at L55) where causal-abstraction predictions fail empirically. Curious if these fit your formalism."
- **Draft file**: `05_email_atticus_geiger.txt`

## 7. Stephen Casper — MIT (verify current affiliation)

- **Status**: ⚪ Not contacted. Not memory-verified.
- **Email lookup**: MIT page / X (@StephenLCasper).
- **Why him**: Diagnostic suites for interpretability; published critiques of probe methodology; ProbeBench v0.0.2 anti-Goodhart axis design responds to exactly his concerns.
- **Hook**: "Your work on diagnostic suites for interpretability is precisely the gap I tried to fill with ProbeBench v0.0.2 (8-axis anti-Goodhart ProbeScore, including a goodhart_resistance axis where FabricationGuard scores 0.32 and ReasonGuard 0.30 — honest disclosure). Would value your critique."

## 8. Buck Shlegeris — Redwood Research

- **Status**: ⚪ Not contacted. Not memory-verified.
- **Email lookup**: Redwood Research page / X (@bshlgrs).
- **Why him**: AI control + monitoring; agent-probe-guard SDK's detect-only design is downstream of control thinking.
- **Hook**: "Built agent-probe-guard SDK for silent coding-agent failure detection on Qwen3.6-27B (L55 thinking AUROC 0.848, L43 capability 0.830). Explicitly shipped as detect-only after Phase 7-8 verdicts found probe direction is epiphenomenal under steering — your control framing influenced the decision to not ship 'boost mode'. Would value input."

## 9. Joseph Bloom — Neuronpedia / Decode Research

- **Status**: ⚪ Not contacted directly (but Caio is in Neuronpedia Slack #introductions per memory; Yehuda Atzmon VC inbound came from there).
- **Email lookup**: Neuronpedia public contact or DM on Slack.
- **Memory link**: Caio has Neuronpedia API key (`reference_neuronpedia_api_key.md`) and submitted SAEs to NP queue 2026-04-17.
- **Why him**: SAE community hub. Already-existing low-friction channel through Neuronpedia. Probably easiest first response.
- **Hook**: "Joseph, Caio from Neuronpedia Slack (also submitted Qwen3.5-4B + Gemma-4-E4B SAEs to NP queue Apr 2026). Shipped qwen36-27b-sae-papergrade (200M tokens, d_sae=65536, ve 0.71-0.84, first public SAE for Qwen3.6 reasoning). Would value your feedback on the InterpScore eval framework."

## 10. Esben Kran — Apart Research

- **Status**: ⚪ Not contacted directly.
- **Email lookup**: Apart Research site / X (@apartresearch / @esbenkc).
- **Memory link**: Caio's Apart Fellowship application is pending (task #99). This is an additional, parallel channel.
- **Why him**: Apart Research mentor network specifically targets independent → safety transitions, which is exactly Caio's trajectory.
- **Hook**: "Apart Fellowship applicant. Shipped MI corpus in 2026: ICML 2026 MI Workshop paper submission, 3 self-published preprints (saturation-direction, NLA two-tier, epiphenomenal probes), MCP server, 16+ experiment datasets, ProbeBench leaderboard. Would value strategic Apart input on next 6 months of independent work."

---

## Anti-mistakes — re-read before sending any email

1. **Verify affiliation before sending.** Names can move quarterly. If anyone listed above moved to Anthropic since my knowledge cutoff (Jan 2026), **skip entirely** for the duration of the Fellows app review.
2. **One per day**, not all at once. 10-day rollout.
3. **Each message individual**, never mass-CC or BCC.
4. **No "respond by [date]" urgency.** They owe you nothing.
5. **Don't mention Fellows app** in first message. If conversation deepens (2-3 exchanges), you can mention it naturally.
6. **No invented emails.** If lookup fails for someone, leave them for last and try a different channel (X DM, Slack, conference). Bounce = bad signal.
7. **Update `project_outreach_apollo_aisi_log.md`** memory file every time you send or receive a message — same turn, never deferred.

## Expected ROI (honest)

- 3-5 substantive responses
- 1-2 real ongoing conversations
- 0-1 potential reference candidate in 4-6 weeks

If that lands, this list paid for itself.

---

## Files in this directory

| File | Purpose | Status |
|---|---|---|
| `00_tier1_contacts_2026-05-15.md` | THIS file. Strategic index of 10 targets. | Active |
| `01_dm_marius_apollo.txt` | Marius DM v1 (short, sent) | Sent 2026-05-09 |
| `02_dm_geoffrey_aisi.txt` | Geoffrey DM v1 (short, sent) | Sent 2026-05-09 |
| `03_email_apollo_research.txt` | Apollo institutional email draft | Pending dispatch |
| `04_email_aisi_research.txt` | AISI institutional email draft | Pending dispatch |

To add: per-contact draft files `05_email_*.txt` for contacts 3-10 as they get drafted. Same one-per-file pattern.
