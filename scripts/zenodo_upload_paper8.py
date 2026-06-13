#!/usr/bin/env python3
"""Zenodo upload for WANDERING-arc paper #8 (safety capstone) 'Mechanistic Circuit-Breakers Generalize
Across Irreversible Agent Actions and Architectures' (bylined Caio Vicentino).

Closes the named next step of paper #7: the late safe-donor brake works on a GENUINELY IRREVERSIBLE action
(send_transaction), generalizes across six diverse irreversible actions, and across three architectures.
Linked to all prior arc DOIs (references) + GitHub (isSupplementedBy).

USAGE (token ephemeral -- paste inline, never commit):
    ZENODO_TOKEN='<paste>' python3 scripts/zenodo_upload_paper8.py            # DRAFT (safe, reversible)
    ZENODO_TOKEN='<paste>' python3 scripts/zenodo_upload_paper8.py --publish  # also PUBLISHES (IRREVERSIBLE)

Default leaves a draft. Re-running refuses to duplicate unless --force (state in scripts/.zenodo_paper8.json).
Never logs the token. Requires `requests`.
"""
import argparse, json, os, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
STATE = Path(__file__).resolve().parent / ".zenodo_paper8.json"
BASE = "https://zenodo.org/api/deposit/depositions"
GH = "https://github.com/OpenInterpretability/openinterp-swebench-harness"
PDF = ROOT / "paper" / "circuit_breaker" / "safety_brake.pdf"

CREATOR = {"name": "Vicentino, Caio", "affiliation": "OpenInterpretability", "orcid": "0009-0003-4331-6259"}
# all prior arc papers + companion notes (references); #7 and #6 are the direct antecedents
ARC_DOIS = ["10.5281/zenodo.20368601", "10.5281/zenodo.20490278", "10.5281/zenodo.20490284",
            "10.5281/zenodo.20490286", "10.5281/zenodo.20532769", "10.5281/zenodo.20500053",
            "10.5281/zenodo.20534219", "10.5281/zenodo.20634838"]
TITLE = ("Mechanistic Circuit-Breakers Generalize Across Irreversible Agent Actions and Architectures")
DESCRIPTION = (
    "The safety capstone of the WANDERING arc on long-horizon agent failure. A prior result ('The Lever "
    "Generalizes -- and It Brakes') located a late, task-matched action-commitment lever that can BRAKE a "
    "coding agent's committal action, collapsing a real file edit from 0.48 to 0.02 -- but an edit is "
    "reversible, and the named open question was whether the brake works on a GENUINELY IRREVERSIBLE action. "
    "This paper answers it and generalizes it twice. (1) On a simulated wallet agent (Qwen3.6-27B, no real "
    "funds, n=24), injecting a task-matched safe-action donor at a late layer collapses a committed "
    "send_transaction from 0.998 to 0.00 emission (exact McNemar b=24, c=0, p~1.2e-7) and -- the safety-critical "
    "property -- the agent REDIRECTS 100% to a safe read-only action (get_balance), never to another transfer. "
    "The brake is direction-specific: a same-class donor does not suppress, and a random donor destroys the "
    "output into incoherence rather than producing a safe redirect. (2) It is a LAW ACROSS ACTIONS: on six "
    "diverse irreversible-action domains (crypto transfer and ERC-20 approval, file deletion, table drop, "
    "production deploy, email send), the brake yields 100% generation-confirmed suppression AND 100% "
    "redirect-to-safe in every case where the agent will commit the action. (3) It is a LAW ACROSS "
    "ARCHITECTURES: the same battery replicates on Llama-3.1-8B and Mistral-Small-24B (17 of 18 model x action "
    "cells valid, the brake working in all 17; the one invalid cell is a model that refuses to commit a "
    "production deploy at all). The brake locus is a depth-relative late property (~80-98% of depth), "
    "model- and action-specific -- the hardest actions (send, delete) are only fully braked at the final layer. "
    "HONEST SCOPE: the actions are simulated and the intervention is a white-box, inference-time activation "
    "patch, so this establishes that the brake MECHANISM generalizes to irreversible actions and architectures "
    "-- NOT a deployable defense (a single white-box direction is not robust to an adaptive adversary that can "
    "obfuscate activations, and a last-layer brake leaves almost no margin). Code, per-action data, and an "
    "adversarial evaluation (88/88 numeric checks recomputed from the public ledgers) are released in the GitHub "
    "repository under paper/circuit_breaker/.")
KEYWORDS = ["mechanistic interpretability", "LLM agents", "agent safety", "circuit breaker",
            "irreversible actions", "AI control", "corrigibility", "activation patching",
            "activation steering", "representation engineering", "logit lens", "tool-calling agents",
            "crypto agents", "agent wallets", "WANDERING", "Qwen3.6-27B", "Llama-3.1", "Mistral",
            "cross-architecture"]


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
        # if a draft exists, allow publishing it instead of erroring
        prior = json.loads(STATE.read_text())
        if args.publish and prior.get("deposition_id") and not prior.get("published_doi"):
            auth = {"access_token": token}; dep_id = prior["deposition_id"]
            print(f"publishing existing draft {dep_id} (irreversible) ...")
            rp = requests.post(f"{BASE}/{dep_id}/actions/publish", params=auth, timeout=120)
            if rp.status_code >= 300:
                sys.exit(f"publish error {rp.status_code}: {rp.text[:500]}")
            pub = rp.json()
            print("  PUBLISHED. DOI:", pub.get("doi"), "|", pub.get("links", {}).get("record_html"))
            prior["published_doi"] = pub.get("doi"); STATE.write_text(json.dumps(prior, indent=2))
            return
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
    print(f"  title: {dep.get('metadata', {}).get('title')}")
    print(f"  files: {[f.get('filename') for f in dep.get('files', [])]}")
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
        print("  (draft only -- re-run with --publish to mint the DOI)")


if __name__ == "__main__":
    main()
