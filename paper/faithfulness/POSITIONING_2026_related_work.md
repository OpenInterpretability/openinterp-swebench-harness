# Positioning / Related-Work dossier — the 2026 frontier vs our arc (2026-06-19)

Reusable for: future versions of *The Late Channel* (20752896) and *Located, Not Secured* (20764858), the
flagship "white-box vs black-box monitoring" paper, and any peer-review submission (BlackboxNLP / ICML MI / ICLR).
Built from a deep read of each source (3 agents, fetched + cross-checked the arXiv full text). **Scoop verdict:
CLEAN — all three predate our 18-Jun Late Channel, so they are prior art we must cite, but none pre-empts a
load-bearing claim.**

---

## 1. Young 2026 — "Why Models Know But Don't Say" (arXiv:2603.26410, submitted 27 Mar 2026)
- **Theirs:** purely *behavioral / text*. 12 open-weight reasoning models, MMLU/GPQA + misleading hints,
  keyword-match whether the hint is acknowledged in thinking-text vs answer-text. Headline: in **55.4%** of
  hint-influenced cases the thinking tokens mention the hint but the answer omits it ("thinking-only divergence");
  per-model 19.6% (Qwen3.5-27B) → 94.7% (Step-3.5-Flash). No interventions, no layers, no causality. Their own
  caveat: "Thinking tokens are a second text-generation channel, not a direct readout of latent computation."
- **Overlap:** only the slogan "know but don't say." Their axis = *text-channel verbalization divergence*; ours =
  *layer-depth causal localization* + *conditional-on-answer-change*. They cannot speak to performative-vs-causal
  (they restrict to influenced cases, no causal measure).
- **Survives novel for us:** SAE causal-patch (decode-reencode), conditional faithfulness (44/60 performative),
  late-layer consolidation (logit-lens), agent-action transfer — all untouched.
- **Cite as:** concurrent text-channel faithfulness; differentiate on causality + depth.
- `\bibitem{young2026}` Young. Why Models Know But Don't Say: CoT Faithfulness Divergence Between Thinking Tokens and Answers. 2026. arXiv:2603.26410.

## 2. Ye, Loffgren, Kotadia, Wong 2026 — "Mechanistic Evidence for Faithfulness Decay" (arXiv:2602.11201; OpenReview wVj7dB7waI; Feb 2026)
- **Theirs:** a "Reasoning Horizon" at **70–85% of CHAIN LENGTH** (token/step index), beyond which steps have
  "negligible or negative causal influence." Methods: counterfactual single-step corruption → NLDD (logit decay),
  RSA + TAS (geometry, *single layer = 50% depth*), linear probes for task variables. Models: DeepSeek-Coder-6.7B,
  Llama-3.1-8B, Gemma-2-9B on GSM8K / Dyck-n / PrOntoQA. Striking result: Gemma 99% acc with **negative NLDD
  (−52.5)** = "anti-faithful" — corrupting the CoT *raises* confidence.
- **Axis = ORTHOGONAL.** Theirs = chain-position (token index); ours = network depth (layer). They explicitly
  defer depth: "TAS and RSA use a single layer (50% depth)… comprehensive analysis may reveal depth-dependent
  effects." That deferred gap is exactly what *The Late Channel* fills.
- **Convergence to flag (not a clash):** their "anti-faithful / high-acc but CoT-disconnected" independently
  corroborates our "performative when the model already knew."
- **Lexical-collision warning:** their "late" = late *CoT tokens* (degraded); our "late" = late *layers*
  (consolidated). Disambiguate in one sentence anywhere we say "late."
- `\bibitem{ye2026}` Ye, Loffgren, Kotadia, Wong. Mechanistic Evidence for Faithfulness Decay in Chain-of-Thought Reasoning. 2026. arXiv:2602.11201.

## 3. Kramár, Engels, Wang, Chughtai, Shah, Nanda, Conmy (Google DeepMind) 2026 — "Building Production-Ready Probes for Gemini" (arXiv:2601.11516, Jan 2026) — **the field-alignment linchpin**
- **Theirs:** a deployed probe cascade. Linear/attention probes on a single mid layer (+ new **MultiMax** head-max
  probe for the short→long-context shift), cascading to a prompted LLM only on ~8% ambiguous traffic. Match/beat
  Gemini 2.5 Flash/Pro as a classifier at **>10,000× lower cost**, ~1/50 inference cost overall. **Input-level
  only**, **cyber-offensive** domain. "Informed the successful deployment of misuse mitigation probes in
  user-facing Gemini."
- **Deployment fact (confirmed):** probe-first cascades now in production at **both Anthropic and GDM**. Lab
  contrast (stated in the Jan-2026 AI-Safety-Frontier review, NOT verbatim in-paper, so attribute to the review):
  Anthropic all-layers / exchange-level / CBRN; GDM single-mid-layer / inputs-only / cyber.
- **The two open questions they NAME — and we answer:**
  1. *Qualitative advantage of white-box over black-box beyond cost?* They justify probes purely on cost, never
     claim the probe catches what text misses. → We answer **both directions**: white-box WINS (Late Channel:
     internals causally carry answer+action, beat CoT-text monitors) AND has a disjoint blind spot (felt≠granted:
     internal authz probe passes 21/21 realistic over-reaches an external task-grounded check catches).
  2. *Probe evasion under adversarial optimization?* They explicitly do NOT handle it: "our techniques do not
     significantly reduce the success rate of adaptive adversarial attacks; … extremely challenging, if not
     impossible." → We supply the measured failure: the late action-brake **collapses 0→1.0 ASR** under an
     adaptive white-box attack on the known locus. Governing rule = AUDIT a fixed model (their regime), don't
     DEFEND against an optimizer.
- **This is our tightest field hook:** their deployed system is the production validation that motivates our
  agenda, and our arc answers the two questions they leave open.
- `\bibitem{kramar2026}` Kramár, Engels, Wang, Chughtai, Shah, Nanda, Conmy (Google DeepMind). Building Production-Ready Probes for Gemini. 2026. arXiv:2601.11516.

---

## Where each citation landed (done 2026-06-19)
- **Late Channel** `late_channel.tex` — related-work paragraph: added Young + Ye as concurrent faithfulness on
  *other axes* (text-channel; chain-position), orthogonal to our layer-depth; Ye's anti-faithful corroborates our
  performative case. (`goodfire2026`/Reasoning-Theater already cited for performative reasoning — did NOT add a
  separate "Boppana" to avoid mis-attribution.) Recompiles 7pp, 0 undefined.
- **Located, Not Secured** `located_not_secured.tex` — §Implication: added Kramár as the deployed non-adversarial
  audit regime that leaves open the two questions our Limits 1–2 and Limit 4 answer. Recompiles 5pp, 0 undefined.

## Open follow-up for the flagship paper (white-box vs black-box)
The Kramár framing is the paper's spine: *"Production probe monitoring is deployed and cost-justified; we ask the
qualitative question they leave open — when does reading internals catch what black-box monitoring misses, and
when is it blind — on a reasoning agent, with the head-to-head they did not run."* Bring forward felt≠granted
(white-box blind) + Late Channel (white-box wins) + COLLAPSE (white-box adversarially fragile) as the three
anchor results, then run the missing scaled action-level head-to-head (Tier-B was underpowered at 5 positives).
