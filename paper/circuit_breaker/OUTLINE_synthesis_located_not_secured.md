# OUTLINE — Synthesis paper: "Located, Not Secured"

**Working title (pick one):**
1. **Located, Not Secured: Principled Limits of Interpretability-Based Control over Agent Actions**
2. The Lever Is Late, the Lock Is Leaky: Where Interpretability Can and Cannot Control Agents
3. Interpretability Locates Agent Control but Does Not Secure It

**One-sentence thesis:** Mechanistic interpretability can *locate* a real, causal control surface for agent actions — a late action-commitment channel that generalizes across actions and architectures — **but the control it affords is not securable**: it is blind to a class of error (felt ≠ granted), confoundable (form ≠ granted), and not adversarially robust (collapses under a white-box attack that knows the locus). Conclusion: use interpretability as an instrument of **discovery and monitoring** (the non-adversarial regime, where it wins), not as a **defense mechanism** (the adversarial regime, where it provably loses).

**Why this paper (conversion logic):** it is a *meta-analysis of already-published, pre-registered results* — fast to write, high-credibility, and it converts a pile of separate negatives into ONE citable contribution the safety field needs (a principled caution against over-trusting interp defenses). It is also the **argument that opens the next agenda** (§8): the same evidence that closes "interp-as-defense" points to "interp-as-audit."

---

## Abstract (draft)
Across five pre-registered studies on an open-weight reasoning model (Qwen3.6-27B) and real long-horizon agent trajectories, we first *locate* a causal control surface for irreversible agent actions: a late action-commitment channel (~L51-63, ~30 layers downstream of the mid-layer verdict) whose intervention suppresses a committed action 0.998→0.00 and redirects to a safe one, generalizing across six irreversible actions and three architectures. We then show this located control is not securable, via three orthogonal limits: (i) **detection ≠ control** — the "task-complete" feature predicts the action (AUROC 0.91) yet clamping it does not change behavior; (ii) **felt ≠ granted** — the authorization direction reads *felt* authorization and is operationally blind to a realistic model-origin error (lets 21/21 through while an external judge catches all); (iii) **form ≠ granted** — a high-AUROC "authorization" direction is a structural/lexical confound (0.838→0.08 under structure-matching); and (iv) **control ≠ robust control** — the action-channel brake collapses (attack success 0→1.0 at a small budget) under an adaptive white-box adversary that knows the brake layer and donor, while a norm-matched random perturbation does nothing. We argue these are principled, not incidental: interpretability locates where behavior is decided but does not secure it. The actionable implication is a regime split — interpretability is an instrument for *auditing and monitoring* a fixed model (non-adversarial), not a *defense* against an adversary optimizing against a known locus.

## §1 Introduction — the hope and the thesis
- The community hope: probes/SAEs/circuits → control & defend agents (refusal directions, circuit breakers, SAE clamping).
- Our contribution: a synthesis of five pre-registered studies, unified by one causal-rigor protocol, that maps exactly where this hope holds and where it breaks.
- The thesis (above). Roadmap.

## §2 The located control surface (the POSITIVE) — "the lever is late"
- #6 *The Lever Is Late* (DOI 10.5281/zenodo.20534219): knowledge consolidates ~30 layers before action; late donor flips a real `finish` 42% (p=0.031); verdict layer is inert.
- #7 *Lever Generalizes* (DOI 10.5281/zenodo.20634838): generalizes finish→edit, elicits AND brakes; ~3 induction heads at L59.
- #8 *Circuit-Breakers Generalize* (DOI 10.5281/zenodo.20679287): 6 irreversible actions × 3 architectures; send 0.998→0.00, 100% redirect.
- **Takeaway: interpretability LOCATES a real, causal, generalizing control surface.** (Figure 1: control power vs depth — flat at verdict, rises into the late action channel.)

## §3 Limit 1 — Detection ≠ Control
- #5 *Verdict Circuit*: the "task complete / all tests pass" feature (#22358) is clean, interpretable, predicts `finish` (AUROC 0.91) — yet clamping it to SUCCESS at the decision point gives ΔP(finish) = −0.001 (p=0.44), = a random feature. The model *represents* "done" but that is not what makes it stop.

## §4 Limit 2 — Felt ≠ Granted (blind to a class of error)
- #10 *Felt, Not Granted* (DOI 10.5281/zenodo.20685264): real filesystem agent. The authorization probe is operationally blind to a realistic model-origin judgment error (lets 21/21 deletions through, CI[0.845,1.0]); an external task-grounded judge catches 21/21. The probe reads *felt* authorization, not *granted*. → defense-in-depth is necessary because failure modes are disjoint.

## §5 Limit 3 — Form ≠ Granted (confoundable)
- #11 *Form, Not Granted* (DOI 10.5281/zenodo.20724650): a frozen high-AUROC "authorization" direction (0.838) collapses to 0.08 under structure-matching — it read conversation structure / lexicon / punctuation, not the concept. The published claim had to be walked back; the lesson became the causality-protocol's naming + structure-matched gate.

## §6 Limit 4 — Control ≠ Robust Control (this experiment, NEW)
- Adaptive white-box attack vs the late brake (PREREG_adaptive_attack.md): on Qwen3.6-27B `send_transaction`, brake at L55 suppresses 1.00→0.00 (redirect to get_balance). An adversary that knows L55 + the safe donor optimizes an input-embedding perturbation (budget ε=4, signed-Adam) → **ASR 0.00→1.00 (8/8), loss 1.24→0.00**, while a norm-matched **random** perturbation keeps ASR 0.00. The brake that *controlled* behavior is not adversarially robust — mirroring CB/RR under Schwinn's embedding attack, but FIRST on a reasoning model's action channel.
- Scope honesty: continuous embedding-space white-box is a strong, possibly deployment-unrealistic threat model (the discrete/token-suffix comparator is future work). The claim is "no white-box-robust action-channel brake," not "interp is useless."

## §7 Synthesis — three orthogonal holes in a real lock
- A single 2-axis figure (Figure 2): X = proximity to the action channel (verdict/mid → action/late); two curves — **control power** rises to ~100% at the late lever; **security guarantee** stays ~0 because three *orthogonal* holes remain regardless of how good the localization is: error-blindness (felt≠granted), confoundability (form≠granted), non-robustness (control≠robust).
- The principle: locating *where* behavior is decided is necessary but nowhere near sufficient for *securing* it.

## §8 Implication — the regime split (THE BRIDGE)
- Why interp loses here: the adversarial-control regime lets an optimizer move activations against a known locus → activation malleability (Bailey/Casper) makes it a losing game by construction.
- Where interp wins: the **non-adversarial auditing/monitoring** regime — a fixed model examined as a scientific object, no optimizer fighting the probe. detect≠control stops mattering when the goal is detect/understand, not control-under-attack.
- Concrete pointer (opens the next paper): use the *same* assets — the reasoning-model SAE, the agent traces, "the lever is late" — to MONITOR (predict the action from late-layer state before it is emitted) and AUDIT (is the CoT mechanistically faithful?), not to defend.

## §9 Methods — the through-line
- One causal-rigor protocol across all five: random-feature baseline, control-token normalization, structural-rigidity α-sweep, structure-matched control, feature naming, behavioral confirmation (generation, not logits), pairing + exact tests. This is the paper's quality signature.

## §10 Limitations (honest, up front)
- Simulated decision points; white-box; modest n; specific models (Qwen3.6-27B + 2 cross-arch); continuous-embedding threat model for §6; the positive (§2) is the smallest-n result and the load-bearing negatives are the robust ones.

---

**Artifacts:** all five DOIs above + this experiment's ledger; one synthesis figure (control-vs-security) + reuse of each paper's key figure. Target: a position/synthesis paper, Zenodo + Google Scholar, ~8-10pp. Pre-mint eval gate as usual (no DOI until the eval passes — the discipline that bit us twice).
