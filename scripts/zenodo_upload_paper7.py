#!/usr/bin/env python3
"""Zenodo upload for WANDERING-arc paper #7 'The Lever Generalizes -- and It Brakes' (bylined Caio Vicentino).

The circuit-breaker capstone: the late action-commitment lever GENERALIZES (finish->edit) and BRAKES,
bidirectionally and monotonically, replicating across two model families and two scales. Linked to all prior
arc DOIs (references) + GitHub (isSupplementedBy).

USAGE (token is ephemeral -- paste inline, never commit):
    ZENODO_TOKEN='<paste your zenodo.org personal token>' python3 scripts/zenodo_upload_paper7.py            # DRAFT (safe, reversible)
    ZENODO_TOKEN='<paste>' python3 scripts/zenodo_upload_paper7.py --publish  # also PUBLISHES (IRREVERSIBLE: permanent DOI)

Default leaves a draft to eyeball. Re-running refuses to duplicate unless --force (state in
scripts/.zenodo_paper7.json). Never logs the token. Requires `requests`.
"""
import argparse, json, os, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
STATE = Path(__file__).resolve().parent / ".zenodo_paper7.json"
BASE = "https://zenodo.org/api/deposit/depositions"
GH = "https://github.com/OpenInterpretability/openinterp-swebench-harness"
PDF = ROOT / "paper" / "circuit_breaker" / "lever_generalizes.pdf"

CREATOR = {"name": "Vicentino, Caio", "affiliation": "OpenInterpretability", "orcid": "0009-0003-4331-6259"}
# all prior arc papers + companion note + the kappa_t note (references)
ARC_DOIS = ["10.5281/zenodo.20368601", "10.5281/zenodo.20490278", "10.5281/zenodo.20490284",
            "10.5281/zenodo.20490286", "10.5281/zenodo.20532769", "10.5281/zenodo.20500053",
            "10.5281/zenodo.20534219"]
TITLE = ("The Lever Generalizes -- and It Brakes: A Late, Bidirectional Action-Commitment Lever Across "
         "Agent Decisions and Architectures")
DESCRIPTION = (
    "The circuit-breaker capstone of the WANDERING arc on long-horizon coding-agent failure. A prior result "
    "('The Lever Is Late') showed that control of a coding agent's 'finish' decision lives not at the mid-layer "
    "'task-is-done' verdict but in a late, task-matched action-commitment block ~30 layers downstream. This "
    "paper answers two pre-registered questions that the single 'finish' result could not: is the late lever "
    "SPECIFIC to termination, and can it BRAKE an action, not just elicit one? On Qwen3.6-27B over 99 SWE-bench "
    "Pro trajectories, using a second decision in the same data -- commit a file edit (str_replace_editor) vs. "
    "continue reversible exploration (bash) -- with n=60 deterministic decision points per condition, "
    "prefill-only patching, and generation-confirmed outcomes: (1) GENERALIZATION (elicit): injecting a "
    "task-matched edit-donor into the late block makes a stuck-in-exploration agent emit a real edit call "
    "(0.23 -> 0.77 at L59; position control 0.08, cross-task control 0.48). (2) THE BRAKE (suppress): injecting "
    "an explore-donor at a commit decision collapses the real edit rate 0.48 -> 0.02 (96% suppression) at L55, "
    "with a same-class control intact (0.55) and the opposite donor boosting to 0.92. (3) The mechanism is "
    "MONOTONIC and BIDIRECTIONAL: exact paired McNemar on all 14 per-point conditions yields seven contrasts "
    "surviving Holm-Bonferroni (worst p=7.6e-5), with elicit c=0 (the edit-donor only turns commits on) and "
    "brake b=0 (the explore-donor only turns them off) -- the lever moves exactly in the donor's direction with "
    "~zero off-direction noise. (4) CROSS-ARCHITECTURE: the late-commitment geometry and donor-specific "
    "writability replicate across two model families and two scales (Mistral-7B and the scale-matched "
    "Mistral-Small-24B, where the mid-inert / late-write dissociation is cleanest: fidelity 0.955 vs 0.007). "
    "Strengtheners: the elicit/brake lift survives a full valid-tool-call re-parse (0.23->0.37 elicit, "
    "0.40->0.07 brake), and the brake re-routes to reversible exploration (+0.17 bash above its no-brake floor "
    "of 0.43). We frame the bidirectional late lever as the mechanism for a mechanistic CIRCUIT-BREAKER: a "
    "single late-layer intervention that blocks an action at its commit point. Honest scope: demonstrated on a "
    "state-mutating but UNDOABLE edit (a semi-irreversible proxy); intervening on a genuinely irreversible "
    "action (e.g. send_transaction) is the named next step. The model-agnostic decision-locator tool, "
    "pre-registrations, per-point data, exact-statistics script, and an adversarial pre-publication evaluation "
    "are released in the GitHub repository under paper/circuit_breaker/.")
KEYWORDS = ["mechanistic interpretability", "LLM agents", "agent safety", "circuit breaker",
            "irreversible actions", "agent failure modes", "knowledge-action gap", "activation patching",
            "activation steering", "representation engineering", "logit lens", "tool-calling agents",
            "SWE-bench", "WANDERING", "Qwen3.6-27B", "Mistral", "cross-architecture"]


def metadata():
    rel = [{"identifier": GH, "relation": "isSupplementedBy", "scheme": "url"}]
    rel += [{"identifier": d, "relation": "references", "scheme": "doi"} for d in ARC_DOIS]
    return {"metadata": {
        "upload_type": "publication", "publication_type": "other",
        "title": TITLE, "creators": [CREATOR], "description": DESCRIPTION,
        "access_right": "open", "license": "cc-by-4.0", "keywords": KEYWORDS,
        "related_identifiers": rel,
    }}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true", help="PUBLISH (IRREVERSIBLE: mints a permanent DOI). Default: draft.")
    ap.add_argument("--force", action="store_true", help="create a new deposition even if a state file exists")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        sys.exit("ERROR: set ZENODO_TOKEN (ZENODO_TOKEN='...' python3 ...), scopes deposit:write + deposit:actions.")
    if not PDF.exists():
        sys.exit(f"ERROR: missing PDF {PDF}")
    if STATE.exists() and not args.force:
        sys.exit(f"ERROR: {STATE.name} exists (prior run). Inspect it or pass --force to avoid a duplicate.")

    auth = {"access_token": token}
    print("creating deposition ...")
    r = requests.post(BASE, params=auth, json={}, timeout=60); r.raise_for_status()
    dep = r.json(); dep_id = dep["id"]; bucket = dep["links"]["bucket"]
    print(f"  id={dep_id}  uploading {PDF.name} ...")
    with open(PDF, "rb") as fh:
        requests.put(f"{bucket}/{PDF.name}", data=fh, params=auth, timeout=300).raise_for_status()
    print("  setting metadata ...")
    rm = requests.put(f"{BASE}/{dep_id}", params=auth, json=metadata(),
                      headers={"Content-Type": "application/json"}, timeout=60)
    if rm.status_code >= 300:
        sys.exit(f"metadata error {rm.status_code}: {rm.text[:500]}")
    dep = rm.json()
    state = {"deposition_id": dep_id, "links": dep.get("links", {}),
             "doi": dep.get("metadata", {}).get("prereserve_doi", {}).get("doi")}
    STATE.write_text(json.dumps(state, indent=2))
    print(f"  DRAFT ready. Pre-reserved DOI: {state['doi']}")
    print(f"  Review/publish UI: {dep['links'].get('html')}")

    if args.publish:
        print("PUBLISHING (irreversible) ...")
        rp = requests.post(f"{BASE}/{dep_id}/actions/publish", params=auth, timeout=120)
        if rp.status_code >= 300:
            sys.exit(f"publish error {rp.status_code}: {rp.text[:500]}")
        pub = rp.json()
        print("  PUBLISHED. DOI:", pub.get("doi"), "|", pub.get("links", {}).get("record_html"))
        state["published_doi"] = pub.get("doi"); STATE.write_text(json.dumps(state, indent=2))
    else:
        print("  (draft only -- re-run with --publish, or click Publish in the UI)")


if __name__ == "__main__":
    main()
