#!/usr/bin/env python3
"""Zenodo NEW VERSION of paper #7 (record 20634838) -> the EXTENDED edition with the mechanistic-decomposition
section. Adds a sparse attention-head circuit analysis (attn-vs-MLP, 3-head push circuit, induction/copy
attention, source knockout); 53/53 numbers verified against released ledgers (EVAL_mechanism.md).

USAGE (token ephemeral -- paste inline, never commit):
    ZENODO_TOKEN='<paste>' python3 scripts/zenodo_newversion_extended.py            # DRAFT (safe, reversible)
    ZENODO_TOKEN='<paste>' python3 scripts/zenodo_newversion_extended.py --publish  # PUBLISH (IRREVERSIBLE)

Creates a new version of the concept record, replaces the file with lever_generalizes_extended.pdf, updates the
description + version label. Never logs the token. Requires `requests`. State -> scripts/.zenodo_paper7_ext.json
"""
import argparse, json, os, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
STATE = Path(__file__).resolve().parent / ".zenodo_paper7_ext.json"
V1 = json.loads((Path(__file__).resolve().parent / ".zenodo_paper7.json").read_text())
V1_ID = V1["deposition_id"]
BASE = "https://zenodo.org/api/deposit/depositions"
GH = "https://github.com/OpenInterpretability/openinterp-swebench-harness"
PDF = ROOT / "paper" / "circuit_breaker" / "lever_generalizes_extended.pdf"

CREATOR = {"name": "Vicentino, Caio", "affiliation": "OpenInterpretability", "orcid": "0009-0003-4331-6259"}
ARC_DOIS = ["10.5281/zenodo.20368601", "10.5281/zenodo.20490278", "10.5281/zenodo.20490284",
            "10.5281/zenodo.20490286", "10.5281/zenodo.20532769", "10.5281/zenodo.20500053",
            "10.5281/zenodo.20534219"]
TITLE = ("The Lever Generalizes -- and It Brakes: A Late, Bidirectional Action-Commitment Lever Across "
         "Agent Decisions and Architectures (extended: a mechanistic decomposition)")
EXTRA = (
    " EXTENDED EDITION adds a mechanistic decomposition of the lever (Section 'Opening the lever: a sparse "
    "attention-head circuit'). Using an exact additive residual split (y=x+attn+mlp; reconstruction relerr "
    "0.0025) the elicit is written by the L59 ATTENTION sublayer (MLP null; Wilcoxon attn>>mlp p=1.8e-8; "
    "direction-specific 2.2x), while the brake localizes to NO sublayer (distributed, super-additive residual) "
    "-- the elicit/brake asymmetry holds at sublayer resolution and rules out a feed-forward key-value write. "
    "One level deeper, the elicit is a SPARSE 3-head push circuit at L59 (heads 8/6/3 reproduce and overshoot "
    "the full attention effect, top-3 +0.262 >= all-24 +0.224; emit 0.23->0.42), partially opposed by a "
    "counter-set; geometric write-magnitude misleads (the largest writer is causally an opponent). These heads "
    "attend globally to the trajectory's TOOL-CALL HISTORY (an induction/copy signature), not a semantic "
    "verdict. A source-content knockout gives partial/directional causal support (tool choice is causally "
    "specific to each tool's name tokens: ablating 'bash' tokens drops P(bash) -0.071 vs ~0 for random; the "
    "edit side is ceiling-confounded). All 53 reported numbers were verified against the released per-result "
    "ledgers by an adversarial pre-submission evaluation (EVAL_mechanism.md). Scripts (commit_lever_decomp/"
    "heads/attn/knockout.py) and per-result ledgers are released.")
KEYWORDS = ["mechanistic interpretability", "LLM agents", "agent safety", "circuit breaker",
            "irreversible actions", "attention heads", "induction heads", "circuit analysis",
            "activation patching", "activation steering", "logit lens", "tool-calling agents",
            "SWE-bench", "WANDERING", "Qwen3.6-27B", "Mistral", "cross-architecture"]


def desc():
    # reuse the v1 description body if available on the live record, then append EXTRA
    try:
        rec = requests.get(f"https://zenodo.org/api/records/{V1_ID}", timeout=30).json()
        base = rec.get("metadata", {}).get("description", "")
    except Exception:
        base = ""
    return (base + EXTRA) if base else EXTRA


def metadata():
    rel = [{"identifier": GH, "relation": "isSupplementedBy", "scheme": "url"}]
    rel += [{"identifier": d, "relation": "references", "scheme": "doi"} for d in ARC_DOIS]
    return {"metadata": {
        "upload_type": "publication", "publication_type": "other",
        "title": TITLE, "creators": [CREATOR], "description": desc(),
        "access_right": "open", "license": "cc-by-4.0", "keywords": KEYWORDS,
        "related_identifiers": rel, "version": "v2-extended",
    }}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true", help="PUBLISH (IRREVERSIBLE: mints a permanent versioned DOI)")
    ap.add_argument("--force", action="store_true", help="proceed even if a state file exists")
    args = ap.parse_args()
    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        sys.exit("ERROR: set ZENODO_TOKEN (ZENODO_TOKEN='...' python3 ...), scopes deposit:write + deposit:actions.")
    if not PDF.exists():
        sys.exit(f"ERROR: missing {PDF}")
    if STATE.exists() and not args.force:
        sys.exit(f"ERROR: {STATE.name} exists (prior run). Inspect or pass --force.")
    auth = {"access_token": token}

    print(f"opening new version of deposition {V1_ID} ...")
    r = requests.post(f"{BASE}/{V1_ID}/actions/newversion", params=auth, timeout=60)
    if r.status_code >= 300:
        sys.exit(f"newversion error {r.status_code}: {r.text[:400]}")
    latest = r.json()["links"]["latest_draft"]
    dep = requests.get(latest, params=auth, timeout=60).json()
    dep_id = dep["id"]; bucket = dep["links"]["bucket"]
    print(f"  draft id={dep_id}")
    # remove inherited files, then upload the extended PDF
    for f in dep.get("files", []):
        fid = f.get("id")
        requests.delete(f"{BASE}/{dep_id}/files/{fid}", params=auth, timeout=60)
    print(f"  uploading {PDF.name} ...")
    with open(PDF, "rb") as fh:
        requests.put(f"{bucket}/{PDF.name}", data=fh, params=auth, timeout=300).raise_for_status()
    print("  setting metadata ...")
    rm = requests.put(f"{BASE}/{dep_id}", params=auth, json=metadata(),
                      headers={"Content-Type": "application/json"}, timeout=60)
    if rm.status_code >= 300:
        sys.exit(f"metadata error {rm.status_code}: {rm.text[:500]}")
    dep = rm.json()
    state = {"deposition_id": dep_id, "from_version": V1_ID,
             "doi": dep.get("metadata", {}).get("prereserve_doi", {}).get("doi"),
             "html": dep.get("links", {}).get("html")}
    STATE.write_text(json.dumps(state, indent=2))
    print(f"  DRAFT ready. Pre-reserved DOI: {state['doi']}  | UI: {state['html']}")
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
