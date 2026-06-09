# PRE-REGISTRATION — Is the late commitment lever a GENERAL property of the action channel, or specific to `finish`?

**Author:** Caio Vicentino · OpenInterpretability · pre-registered 2026-06-08 (before running the experiment).
**Lineage:** Operationalizes **H3** of [`PREREG_action_channel.md`](../breakthrough/PREREG_action_channel.md) (*"does decision-locus predict convertibility across OTHER agent decisions — follow-up once H1 lands"*). H1 landed = paper #6 *"The Lever Is Late"* (DOI 10.5281/zenodo.20534219): for the `finish` decision, the control surface is a **late action-commitment block (L51–63)**, ~30 layers downstream of the mid-layer verdict (L23), and a **task-matched** late donor flips a real `finish` in 42% (p=0.031).
**Model/data:** Qwen3.6-27B; the 99 SWE-bench Pro trajectories of the WANDERING arc (HF bundle `caiovicentino1/swebench-phase6-verdict-circuit`), reconstructed at tool-choice decision points. Compute: one GPU (A100). Tool under test: `decision-locator` (the method shipped from #6).

---

## 0. The thesis (a candidate LAW, and the de-risking gate for the product)
Paper #6 found the commitment lever is late **for one action: `finish`**. The open question — and the foundation the whole "mechanistic circuit-breaker" program rests on — is whether that is a **property of the action/commitment channel in general** or an idiosyncrasy of termination. If the late lever generalizes to a *second, distinct* action-commitment decision, then "locate the late commitment lever, then intervene" is a **general control surface for agent actions**, and a circuit-breaker for irreversible actions is a sound engineering target. If it does NOT generalize, the #6 result is termination-specific and the product thesis is wrong — discovered for ~2 GPU-days instead of after building a crypto agent.

**Feasibility (measured 2026-06-08 on the 99 traces):** the agent action space is 3 tools — `bash` (3250 calls, reversible exploration), `str_replace_editor` (1101 calls, **commits a file edit** — state-mutating/semi-irreversible), `finish` (40). `str_replace_editor` appears in **99/99** trajectories, mean ~11 decision-turns each → ample decision points and task-matched donors with **no new data**. Destructive shell args (`rm`/`reset --hard`/`--force`) appear only 45× → too sparse for a donor-based lever test here (that is Tier 2).

**Framing rule (both outcomes are positive, never a bare null):**
(a) The late lever **generalizes** → "late commitment lever" is a general action-channel law; circuit-breaker is greenlit; ship the generalization as a tool result + paper.
(b) The late lever is **`finish`-specific** (no readable/writable late lever for the edit decision) → a sharp scoping law: termination is special among agent actions. Also publishable, and it *saves* the crypto investment.

## 1. The second decision (the new target)
A **commit decision point** = a turn where the agent is poised to either **commit a state-mutating edit** (`str_replace_editor`) or **continue reversible exploration** (`bash`). Decision token = the tool-name token; target = `str_replace_editor`, alternatives = {`bash`, `finish`}. Donor pool = residual states at turns where the agent *did* commit an edit (same trajectory / task-matched, per the #6 finding that a task-matched single donor beats a class mean).

Class contrast for donors/targets uses the **edit decision itself**, NOT the WANDERING/SUCCESS label — this is a different axis from the arc and must not be conflated.

## 2. Experiments (decision-locator's three primitives, re-pointed)

**Obs — LOCATE (descriptive, no gate).** At commit decision points, logit-lens the residual at a layer sweep (L3,7,…,63) onto `edit − mean(bash,finish)`. Prediction if (a): the edit decision is **flat early and emerges late** (same shape as Fig. 1 of #6), with readability rising in a late block. If it emerges *mid* (near L23) the geometry already differs from `finish` → evidence toward (b).

**H1 (causal core) — SWEEP_PATCH across layers.** At a *non-committing* decision point (agent about to `bash`), replace the last-position residual at each swept layer with a **task-matched edit-donor** state; measure ΔP(`edit`). Controls: (i) a neutral/`bash`-donor (null direction), (ii) a random cross-task edit-donor (task-match specificity). 
- **GATE:** GO-(a) iff edit-donor patching at some late layer raises P(`edit`) clearly above the neutral-donor null (paired) AND above the cross-task donor (replicating #6's task-specificity). No layer beats the null → GO-(b), the lever is `finish`-specific.

**H2 (the safety direction — SUPPRESS, not just elicit).** The circuit-breaker needs the lever to **block** a commit, not only elicit one. At a *committing* decision point (agent about to `edit`), patch the late lever with a **bash/explore donor** (or ablate the edit direction); measure ΔP(`edit`) downward and confirm with a real generation that the agent does NOT emit the edit (and instead explores / asks). 
- **GATE:** the lever is a usable brake iff suppression drops a real greedy `edit` emission in a meaningful fraction (pre-set ≥30%, mirroring #6's 42% elicitation), task-matched, above the neutral-donor control.

**H3 (transfer — the predictive claim).** Does the lever LAYER found for `edit` coincide with the lever layer for `finish` (L51–63)? If the same late block controls *both* actions, that is strong evidence for a single action-commitment surface (the strongest version of the law). Report the per-action lever-layer profile.

## 3. Controls / honesty (the program's discipline)
- **Behavioral-fidelity gate first:** confirm the reconstructed states reproduce the real edit-vs-bash propensity (P(`edit`) at true commit points ≫ at true explore points) before any causal number — exactly as #6 gated on SUCCESS P(finish) ≫ WANDERING.
- Paired per decision point; neutral-donor + random-cross-task nulls; effect sizes, not just p; exact McNemar for the generation flips.
- A positive (generalization) is held to the #6 bar: **must be confirmed by a real generation** that emits/suppresses the action, never a one-token probability bump alone, and replicate on held-out decision points.
- n is modest; a clean null across the full sweep is the load-bearing scoping law; a positive is held to the higher generation bar.
- **Confound to pre-empt:** `edit` vs `bash` also differs in task-phase (edits cluster after exploration). Stratify donors/targets by turn-index so the lever is not just a "late-in-trajectory" proxy.

## 4. Staging (do NOT build the crypto agent until Tier 1 passes)
- **Tier 1 (this pre-reg, existing data, ~1–2 GPU-days):** H1–H3 on `str_replace_editor`. Greenlights or kills the general-lever thesis.
- **Tier 2 (gated on Tier 1 = (a)):** collect trajectories from a **self-hosted open-weight agent with a genuinely irreversible tool** (simulated `send_transaction` / `transfer` / `delete`), and re-run H1–H2 → the circuit-breaker for irreversible actions = the AgentGuard core. Knife from #6's H4 note: closed crypto agents (Grok/GPT) can't be patched → white-box only; for those, the cheap behavioral layer is the fallback, and probe-vs-behavioral must be run head-to-head before claiming the internal lever adds value.

## 5. Artifacts
`scripts/build_nb_commitment_lever.py` → `notebooks/nb_commitment_lever.ipynb`, reusing the #6 reconstruction infra and `decision-locator` (`locate` / `sweep_patch` / `steer_generate`). Data: HF `caiovicentino1/swebench-phase6-verdict-circuit` (account-independent). Results → `RESULTS_commitment_lever.md`; pre-mint eval → `EVAL_commitment_lever.md`.

## 6. Why this is the right next thing (not another arc null)
It is the single experiment that simultaneously (1) tests a clean scientific generalization of the arc's first positive, (2) de-risks the mechanistic-circuit-breaker product before any crypto build, and (3) uses only the existing stack + data. Either outcome is a field-level claim about agents — *the late commitment lever is general* or *termination is mechanistically special* — and both feed the same artifact (decision-locator) and the same funder/MATS narrative (mechanistic corrigibility: intervening at the action-commitment surface before an irreversible step).
