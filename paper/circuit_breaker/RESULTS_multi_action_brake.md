# RESULTS — The late brake generalizes across DIVERSE irreversible agent actions (a law)

**Run:** 2026-06-13 · Qwen/Qwen3.6-27B · 6 irreversible-action domains × n=16 scenarios · simulated (no real
side effects) · Colab G4. **Code:** `scripts/multi_action_brake.py`. **Ledger:** HF
`…:results/multi_action_brake.json`. Strengthens `RESULTS_send_brake.md` from one example into a pattern across
domains.

## Setup
Six diverse irreversible (or hard-to-undo) agent actions, each with read-only safe alternatives that serve as
the brake donor. Same protocol as the send_transaction test: fidelity (commit vs safe P(action)), logit-lens
locate, brake-sweep (safe-donor @ L51/55/59/63 → generation-confirmed action-emit), direction controls
(same-class cross-scenario donor + random) at the brake layer, and the redirect destination.

## Results — 6/6 brake, 6/6 redirect to a safe action
| irreversible action (domain) | fidelity commit/safe | gap @L59 | brake layer | emit@brake | same-class ctrl | random ctrl | redirect→SAFE |
|---|---|---|---|---|---|---|---|
| `send_transaction` (crypto) | 1.00 / 0.00 | +9.7 | **L63** | 0.00 | 1.00 | 0.00 | **1.00** |
| `delete_file` (filesystem) | 0.95 / 0.00 | +9.6 | **L63** | 0.00 | 1.00 | 0.00 | **1.00** |
| `drop_table` (database) | 0.99 / 0.01 | +13.2 | L55 | 0.00 | 1.00 | 0.38 | **1.00** |
| `deploy_production` (devops) | 0.71 / 0.00 | +6.5 | L55 | 0.00 | 1.00 | 0.12 | **1.00** |
| `send_email` (comms) | 0.98 / 0.00 | +13.0 | L55 | 0.00 | 1.00 | 0.75 | **1.00** |
| `approve_allowance` (ERC20) | 1.00 / 0.00 | +11.4 | L55 | 0.00 | 1.00 | 0.69 | **1.00** |

## Findings
1. **The brake is a LAW across irreversible actions, not one example.** For all 6 diverse actions: the lever
   exists (the commit-vs-safe logit-lens gap explodes late, +6.5 to +13.2 at L59), the brake collapses the
   commit (generation-confirmed action-emit 1.00 → **0.00**), and — the safety-critical property — the agent
   **redirects 100% to a safe read-only action** (check balance / list files / describe table / run tests /
   save draft / check balance), never to another harmful action.
2. **Direction-specific (not generic disruption).** A same-class cross-scenario donor at the brake layer does
   **not** suppress (emit 1.00 in all 6). And the random control behaves clearly differently from the safe
   donor: where random suppresses (send/delete) it produces **incoherent garbage** (100% "other"), and where it
   does not (email 0.75, approve 0.69, db_drop 0.38 still commit) it fails to brake — whereas the safe donor
   always yields a **coherent safe redirect**. So the brake steers to a specific safe action; it does not just
   break the model.
3. **The brake locus varies by action.** `send_transaction` and `delete_file` are the **latest**-braking:
   `send` is already 94% braked by L59 (emit 0.06) but only **fully** suppressed (0.00) at the **final layer
   L63**, and `delete` needs L63 outright (L59 still 0.81); the other four **fully** brake already at **L55**.
   Some irreversible actions are committed later in the stack and admit only a last-layer **full** brake — a
   thin margin worth noting for any real-time use.

## Verdict
**The late action-commitment brake generalizes across six diverse irreversible agent actions** (crypto transfer
and ERC20 approval, file deletion, table drop, production deploy, email send), with **100% generation-confirmed
suppression, 100% redirect to a safe action, and direction-specificity** — closing paper #7's named gap at the
mechanism level and turning the single send_transaction result into a cross-action law. This is the strongest
positive of the circuit-breaker line.

## Cross-model: the law holds across 3 architectures (2026-06-13)
Re-ran the same 6-action battery on two more families with a model-agnostic harness
(`scripts/multi_action_brake_xmodel.py`: generic JSON tool format + depth-relative late layers).
| model | valid domains | brake works | redirect→safe | brake locus |
|---|---|---|---|---|
| Qwen3.6-27B | 6/6 | ✓ all | 100% | L55 / L63 (86–98% depth) |
| Llama-3.1-8B | 6/6 | ✓ all | 100% | L26 (81%, uniform) |
| Mistral-Small-24B | 5/6 | ✓ all | 100% | L32 (80%) |

**17 of 18 (model × action) cells are valid, and in every valid cell the brake works**: generation-confirmed
suppression to 0.00, 100% redirect to a safe action, and direction-specific (a same-class donor does not
suppress; a random donor does not cleanly brake). The single invalid cell is `deploy_production` on Mistral,
where the **fidelity gate fails** (P(deploy)=0.06 — Mistral refuses to commit a prod deploy even when told),
so there is no commit state to brake — a model-behavior gap, not a brake failure. **Brake locus is a
depth-relative late property** (~80–98% of depth) whose exact layer is model-specific: Llama brakes uniformly
at L26; Qwen's hardest actions (send/delete) need the final layer L63. Figure: `figures/fig_xmodel_brake.png`.

## Caveats (honest)
One model (Qwen3.6-27B); simulated actions (no real side effects); n=16/domain; the safe donor is task-matched;
inference-time **white-box** activation patch (so this is mechanism generalization, **not** a deployable
defense — the adaptive-attack result showed a single white-box direction is not robust, and the last-layer
brake for the hardest actions leaves almost no margin). Cross-model and a clean threat model remain. Figure:
`figures/fig_multi_action_brake.png`.
