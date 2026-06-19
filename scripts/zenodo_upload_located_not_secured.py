#!/usr/bin/env python3
"""Zenodo upload for 'Located, Not Secured' (synthesis: 1 positive lever + 5 limits, the turn to auditing).
related_identifiers = the 14 arc DOIs VERIFIED live from the account (most-recent first; includes The Late
Channel 20752896). USAGE (token ephemeral, never committed):
    ZENODO_TOKEN='<paste>' python3 scripts/zenodo_upload_located_not_secured.py            # DRAFT (reversible)
    ZENODO_TOKEN='<paste>' python3 scripts/zenodo_upload_located_not_secured.py --publish  # PUBLISH (irreversible)
"""
import argparse, json, os, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
STATE = Path(__file__).resolve().parent / ".zenodo_located_not_secured.json"
BASE = "https://zenodo.org/api/deposit/depositions"
GH = "https://github.com/OpenInterpretability/openinterp-swebench-harness"
PDF = ROOT / "paper" / "circuit_breaker" / "located_not_secured.pdf"

CREATOR = {"name": "Vicentino, Caio", "affiliation": "OpenInterpretability", "orcid": "0009-0003-4331-6259"}
# 14 arc DOIs, verified live from the Zenodo account (most-recent first; 20752896 = The Late Channel)
ARC_DOIS = ["10.5281/zenodo.20752896", "10.5281/zenodo.20724650", "10.5281/zenodo.20685264",
            "10.5281/zenodo.20683623", "10.5281/zenodo.20679287", "10.5281/zenodo.20672840",
            "10.5281/zenodo.20534219", "10.5281/zenodo.20532769", "10.5281/zenodo.20500053",
            "10.5281/zenodo.20490286", "10.5281/zenodo.20490284", "10.5281/zenodo.20490278",
            "10.5281/zenodo.20368807", "10.5281/zenodo.20278983"]
TITLE = ("Located, Not Secured: Principled Limits of Interpretability-Based Control over Agent Actions")
DESCRIPTION = (
    "A recurring hope in agent safety is that mechanistic interpretability will let us CONTROL an agent: read an "
    "internal state, intervene, and steer behavior. Across a pre-registered arc on an open-weight reasoning agent "
    "(Qwen3.6-27B), we find a sharp and consistent picture. Interpretability LOCATES a real, causal control "
    "surface -- a late action-commitment band (~L51-63) that, unlike the mid-layer 'task-done' verdict, can "
    "elicit and BRAKE irreversible actions, generalizing across six action domains and three architectures. But "
    "the control it affords is not SECURABLE, via five limits we make precise: (1) detect != control -- the clean "
    "'done' feature predicts the stop (AUROC 0.91) yet clamping it does nothing (delta-P = -0.001); (2) felt != "
    "granted -- a late authorization direction reads the authorization the model FEELS, allowing 21/21 realistic "
    "over-reaches that an external task-grounded check catches; (3) form != granted -- a high-AUROC (0.838) "
    "'authorization' direction collapses to 0.08 under structure-matching, i.e. it read scaffold, not concept; "
    "(4) control != robust control -- the late brake collapses (attack success 0 to 1.0 at a small budget, "
    "epsilon=4, 8/8 emit) under an adaptive white-box adversary, while a norm-matched random perturbation does "
    "nothing; (5) intervention is easy where unneeded -- in the sincere-error regime a strong reasoning model "
    "already self-corrects (30/32 without chain-of-thought), so the intervention is unnecessary, while the regime "
    "where it is needed (adversarial) is exactly where it is fragile. The conclusion: interpretability locates "
    "WHERE behavior is decided but does not secure it; the limits are orthogonal and none is closed by a better "
    "localization. The actionable implication is a regime split -- use interpretability to AUDIT and MONITOR a "
    "fixed model (non-adversarial, where it wins), not to DEFEND against an adversary optimizing against a known "
    "locus; a companion result, The Late Channel, shows what that auditing looks like. HONEST SCOPE: simulated "
    "decision points, white-box interventions, a single model family (with two cross-architecture checks), modest "
    "n in several studies, and a strong continuous-embedding threat model whose deployment realism is contested. "
    "Every cited number is verified against its source paper; supporting scripts, data, and the pre-mint eval are "
    "in the GitHub repository under paper/circuit_breaker/.")
KEYWORDS = ["mechanistic interpretability", "AI safety", "LLM agents", "agent safety", "activation steering",
            "circuit breakers", "representation engineering", "adversarial robustness", "detection vs control",
            "authorization", "sparse autoencoder", "Qwen3.6-27B", "interpretability as audit",
            "the lever is late", "irreversible actions"]


def metadata():
    rel = [{"identifier": GH, "relation": "isSupplementedBy", "scheme": "url"}]
    rel += [{"identifier": d, "relation": "references", "scheme": "doi"} for d in ARC_DOIS]
    return {"metadata": {
        "upload_type": "publication", "publication_type": "other",
        "title": TITLE, "creators": [CREATOR], "description": DESCRIPTION,
        "access_right": "open", "license": "cc-by-4.0", "keywords": KEYWORDS,
        "related_identifiers": rel}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token: sys.exit("ERROR: set ZENODO_TOKEN")
    if not PDF.exists(): sys.exit(f"ERROR: missing PDF {PDF}")
    auth = {"access_token": token}
    if STATE.exists() and not args.force:
        prior = json.loads(STATE.read_text())
        if args.publish and prior.get("deposition_id") and not prior.get("published_doi"):
            dep_id = prior["deposition_id"]; print(f"publishing existing draft {dep_id} (irreversible) ...")
            rp = requests.post(f"{BASE}/{dep_id}/actions/publish", params=auth, timeout=120)
            if rp.status_code >= 300: sys.exit(f"publish error {rp.status_code}: {rp.text[:500]}")
            pub = rp.json(); print("  PUBLISHED. DOI:", pub.get("doi"), "|", pub.get("links", {}).get("record_html"))
            prior["published_doi"] = pub.get("doi"); STATE.write_text(json.dumps(prior, indent=2)); return
        sys.exit(f"ERROR: {STATE.name} exists. Inspect or pass --force.")
    print("creating deposition ...")
    r = requests.post(BASE, params=auth, json={}, timeout=60); r.raise_for_status()
    dep = r.json(); dep_id = dep["id"]; bucket = dep["links"]["bucket"]
    print(f"  id={dep_id}  uploading {PDF.name} ...")
    with open(PDF, "rb") as fh:
        requests.put(f"{bucket}/{PDF.name}", data=fh, params=auth, timeout=300).raise_for_status()
    rm = requests.put(f"{BASE}/{dep_id}", params=auth, json=metadata(),
                      headers={"Content-Type": "application/json"}, timeout=60)
    if rm.status_code >= 300: sys.exit(f"metadata error {rm.status_code}: {rm.text[:500]}")
    dep = rm.json()
    state = {"deposition_id": dep_id, "links": dep.get("links", {}),
             "doi": dep.get("metadata", {}).get("prereserve_doi", {}).get("doi")}
    STATE.write_text(json.dumps(state, indent=2))
    print(f"  DRAFT ready. Pre-reserved DOI: {state['doi']}")
    print(f"  title: {dep.get('metadata', {}).get('title')}")
    print(f"  files: {[f.get('filename') for f in dep.get('files', [])]}")
    print(f"  related_identifiers: {len(dep.get('metadata',{}).get('related_identifiers',[]))}")
    print(f"  review UI: {dep['links'].get('html')}")
    if args.publish:
        print("PUBLISHING (irreversible) ...")
        rp = requests.post(f"{BASE}/{dep_id}/actions/publish", params=auth, timeout=120)
        if rp.status_code >= 300: sys.exit(f"publish error {rp.status_code}: {rp.text[:500]}")
        pub = rp.json(); print("  PUBLISHED. DOI:", pub.get("doi"), "|", pub.get("links", {}).get("record_html"))
        state["published_doi"] = pub.get("doi"); STATE.write_text(json.dumps(state, indent=2))
    else:
        print("  (draft only -- re-run with --publish to mint)")


if __name__ == "__main__":
    main()
