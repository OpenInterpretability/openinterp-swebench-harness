#!/usr/bin/env python3
"""Zenodo upload for the WANDERING-arc companion note (bylined Caio Vicentino).

'No Better Than Behavioral' — the context-rot pre-registered negative. Linked to all four
arc DOIs (references) + GitHub (isSupplementedBy).

USAGE (token is ephemeral — paste inline, never commit):
    export ZENODO_TOKEN='<paste your zenodo.org personal token>'   # scopes: deposit:write + deposit:actions
    python3 scripts/zenodo_upload_context_rot_note.py            # creates a DRAFT (safe, reversible)
    python3 scripts/zenodo_upload_context_rot_note.py --publish  # also PUBLISHES (IRREVERSIBLE: permanent DOI)

Default leaves a draft on zenodo.org to eyeball before you click Publish. Re-running refuses
to duplicate unless --force (remembers the deposition id in scripts/.zenodo_context_rot_note.json).
Never logs the token. Requires `requests`.
"""
import argparse, json, os, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
STATE = Path(__file__).resolve().parent / ".zenodo_context_rot_note.json"
BASE = "https://zenodo.org/api/deposit/depositions"
GH = "https://github.com/OpenInterpretability/openinterp-swebench-harness"
PDF = ROOT / "paper" / "context_rot" / "no_better_than_behavioral.pdf"

CREATOR = {"name": "Vicentino, Caio", "affiliation": "OpenInterpretability", "orcid": "0009-0003-4331-6259"}
ARC_DOIS = ["10.5281/zenodo.20368601", "10.5281/zenodo.20490278",
            "10.5281/zenodo.20490284", "10.5281/zenodo.20490286"]
TITLE = ("No Better Than Behavioral: A Residual Velocity-Freezing Fingerprint Predicts Agent "
         "WANDERING No Better Than the Cheap Tool-Entropy Detector")
DESCRIPTION = (
    "Companion note to the four-paper WANDERING arc, reporting a pre-registered NEGATIVE. "
    "Motivated by the context-rot literature (long-context degradation is representational, not "
    "retrieval; arXiv:2510.05381), we ask whether the residual stream carries an earlier or better "
    "detector of long-horizon agent WANDERING than the cheap probe-free tool-entropy signal -- does "
    "the geometry rot before the behavior does? On the same 99 Qwen3.6-27B SWE-bench Pro trajectories "
    "(CPU re-analysis, no new compute): Stage 1 (raw residual geometry, no SAE) finds a real but weak "
    "fingerprint, representational velocity-freezing -- WANDERING trajectories settle toward an attractor "
    "sooner (smaller early per-turn state change), directionally consistent across all five layers "
    "(4/5 raw p<0.05, length-controlled), with one mid-network layer (L31) clearing a pre-registered "
    "trend-and-divergence conjunction (p=0.015), but nothing surviving multiple-comparison correction "
    "over the 4x5 metric-layer grid. Stage 2 (the decisive predictive test) shows the fingerprint adds "
    "nothing: early velocity at L31 reaches AUROC 0.695, statistically indistinguishable from the fair "
    "early behavioral baseline (tool_entropy_first10, 0.688; paired bootstrap delta=+0.008, 95% CI "
    "[-0.170,+0.211]), and clearly below the deployed late detector (0.888); as a sharp alarm at <=5% "
    "false-positive it catches only 1-3 of 20 WANDERING and never fires earlier than the deployed "
    "detector. The residual fingerprint of context rot is real but downstream-redundant -- it carries "
    "no predictive information beyond the cheap behavioral signal, strengthening the arc: for this "
    "failure mode, watching the cheap behavior is as good as or better than reading the residual stream. "
    "This is a statement about prediction and redundancy, not causation. Pre-registration, both stage "
    "results, and analysis code are in the GitHub repository under paper/context_rot/.")
KEYWORDS = ["mechanistic interpretability", "LLM agents", "agent failure modes", "context rot",
            "honest negative", "pre-registration", "residual stream geometry", "tool-entropy",
            "SWE-bench", "WANDERING", "Qwen3.6-27B"]


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
