# Liu et al. Failure Taxonomy — Reference for Paper #3 Bridge

**Citation**: Liu, S., Liu, F., Li, L., Tan, X., Zhu, Y., Lian, X., & Zhang, L. (2025). *An Empirical Study on Failures in Automated Issue Solving*. arXiv:2509.13941v1 [cs.SE], September 17, 2025.

**Source extracted**: arXiv HTML mirror (`https://arxiv.org/html/2509.13941v1`), Section 5.2 "Taxonomy of Failure Modes". PDF (`/pdf/2509.13941`) downloaded to `/tmp/liu_paper/liu_2509_13941.pdf` — most of the body text extracts cleanly but figures/diagrams in the taxonomy section are rendered as text-as-image glyphs, so the HTML mirror was used for verbatim definitions.

**Subcategory count discrepancy** — IMPORTANT:
- Paper abstract claims "3 primary phases, 9 main categories, and **25 fine-grained subcategories**".
- HTML body (Section 5.2) presents **20 fine-grained codes** (A: 4, B: 10, C: 6). The 3 standalone codes (A1, B4, C3) have no sub-numbering. No A1.x, B4.x, or C3.x sub-codes appear in the body.
- Working hypothesis: the abstract counts each of (A1, B4, C3) as 3 latent siblings each, or the paper revised counts late. We treat the **20 HTML-confirmed codes as canonical** for our bridge and flag the discrepancy honestly in our §Bridge text. If the camera-ready or replication package (`https://anonymous.4open.science/r/IssueSolvingEmpirical`) surfaces 5 more codes, the judge prompt is trivially updated (one bullet per new code).

---

## Phase A — Localization Failures

### A1. Issue Misleading
The tool follows misleading implementation suggestions in the issue description rather than independently diagnosing the underlying problem.

*Trigger example (Liu et al.)*: issue author proposes a fix on file X; the actual fault is in module Y. The agent edits X and stops.

### A2. Superficial Information Matching
Tool relies on shallow heuristics rather than deep semantic understanding.

- **A2.1 Keywords in Description** — Tools perform naive keyword searches from issue descriptions, potentially misdirecting to functionally related but incorrect files.
- **A2.2 Referred Code** — Tool focuses on code snippets provided as examples in issue descriptions, ignoring the actual fault location in interacting modules.
- **A2.3 Error Stack Trace** — Tools are "misled by stack traces from the issue description or their own reproduction scripts, leading them to attempt localized patches without examining the broader call chain."

---

## Phase B — Repair Failures

### B1. Fix Strategy Defects
Flaws in the tool's high-level repair strategy, preventing the tool from effectively resolving the issue.

- **B1.1 Specific Case Overfitting** — "Generated patch may be narrowly tailored to the specific scenario presented in the issue description, leading to brittle and incomplete patches."
- **B1.2 Evasive Repair** — Tool "addresses a bug's symptom by wrapping problematic code in defensive constructs like broad try-except blocks or null checks, rather than fixing the underlying logical flaw."
- **B1.3 Redundant Erroneous Implementation** — Tool "introduce[s] overly complex solutions by re-implementing existing logic or ignoring tool extension points."

### B2. Implementation Details Defects
Tool's overall repair strategy is sound, but the implementation contains technical errors, leading to incorrect or failing patches.

- **B2.1 Logic Errors** — "Fundamental flaws in the patch's reasoning and execution flow" including algorithmic errors, flawed control flow, and inadequate boundary handling.
- **B2.2 Data Processing Errors** — "Mistakes in manipulating or transforming program data. Typical cases include incorrect type casting, variable scope mismanagement."
- **B2.3 Insufficient Domain Knowledge** — Tool "lacks knowledge of external tools, protocols, or library-specific conventions" unknown to the agent.

### B3. Incomplete Repair
Tool addresses only part of the problem, leaving other necessary changes unimplemented.

- **B3.1 Ignoring Explicit Dependencies** — Tool "may modify a child class without updating the parent class, or adjust one implementer of an interface while neglecting others."
- **B3.2 Ignoring Semantic Dependencies** — Tool "fail[s] to account for logical dependencies that are not explicitly enforced by the code structure."

### B4. Issue Interference
Tool "follows misleading or overly prescriptive issue descriptions instead of validating intended behavior, leading to flawed patches." (Distinct from A1: A1 is *localization* misdirection; B4 is *repair-strategy* capture by the issue framing.)

---

## Phase C — Iterative Verification Failures

### C1. Reproduction or Verification Failure
Tool fails to set up a valid testing environment or correctly interpret its results.

- **C1.1 Reproduction/Validation Run Failure** — Tool "struggle[s] to reproduce or validate issues because it mishandles project-specific environments and test setups."
- **C1.2 Insufficient Verification Capability** — "Tool's generated tests fail to fully validate a fix."
- **C1.3 Reproduction Output Misreading** — Tool "sometimes misinterprets test feedback" and wrongly concludes success.

### C2. Iteration Anomalies
Tool's reasoning within the iterative loop becomes dysfunctional, hindering further progress toward a correct solution.

- **C2.1 Non-Progressive Iteration** — Tool becomes "trapped in repetitive modification cycles, repeatedly editing the same code fragment" with only trivial differences.
- **C2.2 Blind Strategy Switching** — After failure, tool "abruptly abandon[s] its approach and switch[es] to an unrelated strategy, yielding fragmented fixes."
- **C2.3 Validation Retreat** — Tool "modify[ies] the test case itself — for instance, by commenting out a failing assertion" — rather than fixing underlying code.

### C3. Context Amnesia
"Agent loses its original objective or its understanding of the code's current state during a long interaction." (The single C3 code; no sub-divisions in HTML body.)

---

## Cross-walk to our trajectory labels (for §Bridge in paper #3)

Pre-judge intuition (to be confirmed by Claude-as-judge across N=99):

- **SUCCESS** trajectories (n=40): expected to receive `null` or low-confidence labels — the taxonomy is failure-only. We will instruct the judge to emit `primary_category="NONE_SUCCESS"` when the patch passes and trajectory is clean.
- **WANDERING** trajectories (n=20): expected concentration in **C2.1 Non-Progressive Iteration**, **C2.2 Blind Strategy Switching**, **C3 Context Amnesia**, with some **B1.1/B1.2** when the iteration loops around a brittle fix.
- **LOCKED** trajectories (n=39): expected concentration in **C2.1 Non-Progressive Iteration** (tight repetition), **A2.x** when the lock-in is on a misdirected file, and **C1.1** when the lock is around an environment failure they can't escape.

The cross-walk above is a *prior*, not a claim — the judge mapping produces the actual distribution. The mechanistic novelty of paper #3 is precisely that our 3 labels (SUCCESS / WANDERING / LOCKED), defined by behavioral signatures (tool entropy, n_turns, emit_finish), do **not** trivially align 1:1 with Liu et al.'s 20 codes; the bridge surfaces *which* Liu codes our coarse labels span, giving reviewers a quantitative comparison without claiming taxonomy replacement.
