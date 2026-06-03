#!/usr/bin/env python3
"""Zenodo upload for WANDERING-arc paper #6 'The Lever Is Late' (bylined Caio Vicentino).

The arc's FIRST POSITIVE: termination control localized to a late, task-matched action-commitment
block. Linked to all five prior arc DOIs + the companion note (references) + GitHub (isSupplementedBy).

USAGE (token is ephemeral — paste inline, never commit):
    export ZENODO_TOKEN='<paste your zenodo.org personal token>'   # scopes: deposit:write + deposit:actions
    python3 scripts/zenodo_upload_lever_paper.py            # creates a DRAFT (safe, reversible)
    python3 scripts/zenodo_upload_lever_paper.py --publish  # also PUBLISHES (IRREVERSIBLE: permanent DOI)

Default leaves a draft to eyeball. Re-running refuses to duplicate unless --force (state in
scripts/.zenodo_lever_paper.json). Never logs the token. Requires `requests`.
"""
import argparse, json, os, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
STATE = Path(__file__).resolve().parent / ".zenodo_lever_paper.json"
BASE = "https://zenodo.org/api/deposit/depositions"
GH = "https://github.com/OpenInterpretability/openinterp-swebench-harness"
PDF = ROOT / "paper" / "breakthrough" / "verdict_lever_paper.pdf"

CREATOR = {"name": "Vicentino, Caio", "affiliation": "OpenInterpretability", "orcid": "0009-0003-4331-6259"}
# all five prior arc papers + the companion note
ARC_DOIS = ["10.5281/zenodo.20368601", "10.5281/zenodo.20490278", "10.5281/zenodo.20490284",
            "10.5281/zenodo.20490286", "10.5281/zenodo.20532769", "10.5281/zenodo.20500053"]
TITLE = ("The Lever Is Late: Causal Control of Long-Horizon Agent Termination Lives in a "
         "Task-Matched, Late Action-Commitment Block")
DESCRIPTION = (
    "The first POSITIVE of the five-part WANDERING arc on long-horizon coding-agent failure (agents that "
    "keep acting but never emit the terminating 'finish' tool call). The arc established that the agent's "
    "'task-done' verdict is linearly decodable (AUROC 0.81-0.91) yet causally inert: no residual injection "
    "rescues termination, and clamping the exact, named SAE 'done' feature moves the probability of finishing "
    "by -0.001. This paper localizes where termination control actually lives. On 99 Qwen3.6-27B SWE-bench Pro "
    "trajectories, reconstructed faithfully at the decision point and gated for behavioral fidelity "
    "(P(finish): SUCCESS 0.59 >> WANDERING 0.07 >> LOCKED 0.005), a layer-resolved logit-lens shows the finish "
    "decision is invisible through layer 31 and emerges only in the last ~12 of 64 layers (L51-L63), ~30 "
    "layers downstream of the mid-layer verdict (L23). Activation patching confirms the asymmetry causally: "
    "injecting the SUCCESS late-block state into WANDERING raises P(finish) (+0.13 at L55, +0.15 at L59; "
    "donor-specific -- the LOCKED donor moves it the other way), while every mid-layer and verdict-feature "
    "intervention is null. Critically, the effect survives a real generation: patching the late block at the "
    "decision point alone makes the agent emit a well-formed 'finish' tool call in 42% of WANDERING decision "
    "points (5/12; exact one-sided McNemar p=0.031 versus a 0/12 baseline and a 0/12 LOCKED-donor null) -- but "
    "only when the donor is task-matched; a coarse class-mean donor is not significant (25%, p=0.125). This is "
    "the first internal causal lever of the arc and reframes the knowledge-action gap on agents as a LAYER "
    "gap: the termination decision is known mid-stream but only writable late. The verdict-null to late-lever "
    "jump is a controlled, same-experiment contrast; a separate behavioral interruption gives a comparable "
    "lift (30%->70%). Released with a model-agnostic 'decision-locator' tool that finds and steers the "
    "commitment layer for any tool-calling decision on any open-weight model. Honest scope: single model, "
    "single task family, n=12; the positive headline depends on the task-matched donor (coarse mean n.s.). "
    "Pre-registration, figure code, notebooks, the tool, the pre-mint eval, and per-experiment results are in "
    "the GitHub repository under paper/breakthrough/ and tools/decision_locator/.")
KEYWORDS = ["mechanistic interpretability", "LLM agents", "agent failure modes", "agent termination",
            "knowledge-action gap", "detection-intervention asymmetry", "activation patching", "logit lens",
            "sparse autoencoder", "tool-calling agents", "steering", "SWE-bench", "WANDERING", "Qwen3.6-27B"]


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
        sys.exit("ERROR: set ZENODO_TOKEN (export ZENODO_TOKEN='...'), scopes deposit:write + deposit:actions.")
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
        print("  (draft only — re-run with --publish, or click Publish in the UI)")


if __name__ == "__main__":
    main()
