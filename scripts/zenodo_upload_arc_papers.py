#!/usr/bin/env python3
"""Turnkey Zenodo upload for the 3 WANDERING-arc papers (bylined Caio Vicentino).

Companion paper #1 (Tool-Entropy Collapse) is already on Zenodo:
DOI 10.5281/zenodo.20368601. This ships #2, #3, #4.

USAGE (token is ephemeral — paste inline, never commit):
    export ZENODO_TOKEN='<paste your zenodo.org personal token>'
    python3 scripts/zenodo_upload_arc_papers.py            # creates 3 DRAFTS (safe, reversible)
    python3 scripts/zenodo_upload_arc_papers.py --publish  # also PUBLISHES (irreversible: permanent DOIs)

Default leaves drafts on zenodo.org so you can eyeball them and click Publish
yourself. Re-running refuses to create duplicates unless --force (it remembers
deposition ids in scripts/.zenodo_arc_deposits.json).

Never logs the token. Requires `requests`.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
STATE = Path(__file__).resolve().parent / ".zenodo_arc_deposits.json"
BASE = "https://zenodo.org/api/deposit/depositions"

GH = "https://github.com/OpenInterpretability/openinterp-swebench-harness"
TOOLENTROPY_DOI = "10.5281/zenodo.20368601"

CREATOR = {
    "name": "Vicentino, Caio",
    "affiliation": "OpenInterpretability",
    "orcid": "0009-0003-4331-6259",
}
COMMON_KEYWORDS = [
    "mechanistic interpretability",
    "LLM agents",
    "agent failure modes",
    "SWE-bench",
    "activation steering",
    "tool-entropy",
    "WANDERING",
]

PAPERS = [
    {
        "key": "paper2",
        "pdf": ROOT / "paper" / "paper2" / "two_honest_nulls.pdf",
        "title": ("Causal Localization of Agent WANDERING to Edge-Layer L11: "
                  "The Right Locus Is Still Not a Rescue Lever"),
        "description": (
            "Three causal tests of the Tool-Entropy WANDERING mechanism hypothesis "
            "(mid-layer verdict consolidated, edge-layer alignment fails) on Qwen3.6-27B "
            "SWE-bench Pro trajectories. All three are null on rescuing WANDERING: a "
            "forced-finish counterfactual (Fisher p=0.71), an always-on L55 SUCCESS-donor "
            "hook (Fisher p=1.00), and a norm-matched L11 alpha-sweep at the strongest "
            "detector locus (paired McNemar p=0.73). The one robust effect is the opposite "
            "of a rescue: high-magnitude L11 injection destabilizes the model into invalid "
            "tool calls (12/20). A load-bearing methodological finding: WANDERING is not "
            "run-stable at temperature 1.0 (the same instances flip finish 7/20 with no "
            "intervention), so every intervention test must be paired and the unpaired 0/20 "
            "baseline manufactures a false positive. Companion to Tool-Entropy Collapse "
            "(DOI 10.5281/zenodo.20368601)."),
    },
    {
        "key": "paper3",
        "pdf": ROOT / "paper" / "paper3" / "multichannel_wandering.pdf",
        "title": ("Multi-Channel Mechanistic Signatures of Agent WANDERING: "
                  "Classification, Causal Localization, and Behavior-Legible Response to Intervention"),
        "description": (
            "Mechanistic characterization of agent WANDERING on N=99 Qwen3.6-27B SWE-bench "
            "Pro trajectories. 60 multi-channel features (text, tool-use, per-layer residual, "
            "temporal) classify SUCCESS/LOCKED/WANDERING at macro-F1 0.636 (p=0.001), after a "
            "transparent walk-back from a leaky 0.987 baseline. Stability selection "
            "independently recovers a mid-to-edge mechanism (LOCKED->L43, WANDERING->L11), and "
            "an LLM-judge bridge to a human taxonomy places ~60% of both LOCKED and WANDERING "
            "into one text category, matching a mechanistically weak boundary (p=0.035). "
            "Finally, the residual signature does not predict which agents flip to finish under "
            "a companion L11 injection run (LOO-AUC 0.619), but tool-entropy collapse depth does "
            "(AUC 0.768): response to intervention is residual-blind but behavior-legible. "
            "Companion to Tool-Entropy Collapse (DOI 10.5281/zenodo.20368601)."),
    },
    {
        "key": "paper4",
        "pdf": ROOT / "paper" / "paper4" / "modality_matters.pdf",
        "title": ("Modality Matters: A Transient Behavioral Interruption Rescues Agent "
                  "WANDERING Where Residual Steering Does Not"),
        "description": (
            "On the same 20 WANDERING Qwen3.6-27B SWE-bench Pro trajectories where residual "
            "steering fails three times, a transient behavioral interruption -- one fresh user "
            "turn at a live tool-entropy collapse point -- roughly doubles the rate at which "
            "agents finalize (30% -> 70%, paired McNemar p=0.021), while a residual L11 "
            "injection stays inert (p=0.63). The lever is the interruption itself, not its "
            "content: a content-neutral message rescues as well as a re-plan (p=1.0). "
            "SWE-bench Pro Docker evaluation indicates the rescued finalizations are real fixes "
            "and suggests the interruption also raises solve-rate (~23% -> 50%, cross-session, "
            "p=0.062). For long-horizon agents the predictive signal lives in the residual "
            "stream but the causal lever lives in behavior. Completes a four-paper arc "
            "(detect -> localize -> residual fails -> behavioral works). Companion to "
            "Tool-Entropy Collapse (DOI 10.5281/zenodo.20368601)."),
    },
]


def metadata(p):
    return {"metadata": {
        "upload_type": "publication",
        "publication_type": "preprint",
        "title": p["title"],
        "creators": [CREATOR],
        "description": p["description"],
        "access_right": "open",
        "license": "cc-by-4.0",
        "keywords": COMMON_KEYWORDS,
        "related_identifiers": [
            {"identifier": GH, "relation": "isSupplementedBy", "scheme": "url"},
            {"identifier": TOOLENTROPY_DOI, "relation": "isContinuationOf", "scheme": "doi"},
        ],
    }}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true",
                    help="PUBLISH after upload (IRREVERSIBLE: mints permanent DOIs). Default: draft only.")
    ap.add_argument("--force", action="store_true",
                    help="create new depositions even if a prior state file exists")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        sys.exit("ERROR: set ZENODO_TOKEN (export ZENODO_TOKEN='...'). Get it at "
                 "https://zenodo.org/account/settings/applications/tokens/new/ "
                 "with scopes deposit:write + deposit:actions.")

    if STATE.exists() and not args.force:
        sys.exit(f"ERROR: {STATE.name} already exists (prior run). Re-running would create "
                 f"DUPLICATE drafts. Inspect it, or pass --force to override.")

    for p in PAPERS:
        if not p["pdf"].exists():
            sys.exit(f"ERROR: missing PDF {p['pdf']}")

    auth = {"access_token": token}
    results = []
    for p in PAPERS:
        print(f"\n[{p['key']}] creating deposition ...")
        r = requests.post(BASE, params=auth, json={}, timeout=60)
        r.raise_for_status()
        dep = r.json()
        dep_id = dep["id"]
        bucket = dep["links"]["bucket"]

        print(f"  id={dep_id}  uploading {p['pdf'].name} ...")
        with open(p["pdf"], "rb") as fh:
            ru = requests.put(f"{bucket}/{p['pdf'].name}", data=fh, params=auth, timeout=300)
        ru.raise_for_status()

        print("  setting metadata ...")
        rm = requests.put(f"{BASE}/{dep_id}", params=auth,
                          json=metadata(p),
                          headers={"Content-Type": "application/json"}, timeout=60)
        if rm.status_code >= 400:
            sys.exit(f"  metadata error {rm.status_code}: {rm.text[:500]}")

        entry = {"key": p["key"], "id": dep_id,
                 "draft_html": dep["links"].get("html"),
                 "title": p["title"], "published": False, "doi": None}

        if args.publish:
            print("  PUBLISHING (irreversible) ...")
            rp = requests.post(f"{BASE}/{dep_id}/actions/publish", params=auth, timeout=120)
            rp.raise_for_status()
            pub = rp.json()
            entry["published"] = True
            entry["doi"] = pub.get("doi") or pub.get("metadata", {}).get("doi")
            entry["record_html"] = pub["links"].get("record_html") or pub["links"].get("latest_html")
            print(f"  PUBLISHED  DOI={entry['doi']}")
        else:
            print(f"  DRAFT ready: {entry['draft_html']}")
        results.append(entry)

    STATE.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} deposition records to {STATE}")
    print("\n=== SUMMARY ===")
    for e in results:
        if e["published"]:
            print(f"  {e['key']}: PUBLISHED  DOI {e['doi']}  {e.get('record_html','')}")
        else:
            print(f"  {e['key']}: DRAFT  {e['draft_html']}  (review, then click Publish)")
    if not args.publish:
        print("\nDrafts are NOT public yet. Review each on zenodo.org and Publish, or re-run "
              "with --publish (irreversible).")


if __name__ == "__main__":
    main()
