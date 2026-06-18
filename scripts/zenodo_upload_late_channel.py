#!/usr/bin/env python3
"""Zenodo upload for 'The Late Channel' (first paper of the interp-as-audit phase, bylined Caio Vicentino).
related_identifiers = the 13 arc DOIs VERIFIED live from the account (2026-06-18) -- the #1 Tool-Entropy DOI
was 20368807 (not 20368601 as a stale script/memory had it). USAGE (token ephemeral, never committed):
    ZENODO_TOKEN='<paste>' python3 scripts/zenodo_upload_late_channel.py            # DRAFT (reversible)
    ZENODO_TOKEN='<paste>' python3 scripts/zenodo_upload_late_channel.py --publish  # PUBLISH (irreversible)
"""
import argparse, json, os, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
STATE = Path(__file__).resolve().parent / ".zenodo_late_channel.json"
BASE = "https://zenodo.org/api/deposit/depositions"
GH = "https://github.com/OpenInterpretability/openinterp-swebench-harness"
PDF = ROOT / "paper" / "faithfulness" / "late_channel.pdf"

CREATOR = {"name": "Vicentino, Caio", "affiliation": "OpenInterpretability", "orcid": "0009-0003-4331-6259"}
# 13 arc DOIs, verified live from the Zenodo account on 2026-06-18 (most-recent first)
ARC_DOIS = ["10.5281/zenodo.20724650", "10.5281/zenodo.20685264", "10.5281/zenodo.20683623",
            "10.5281/zenodo.20679287", "10.5281/zenodo.20672840", "10.5281/zenodo.20534219",
            "10.5281/zenodo.20532769", "10.5281/zenodo.20500053", "10.5281/zenodo.20490286",
            "10.5281/zenodo.20490284", "10.5281/zenodo.20490278", "10.5281/zenodo.20368807",
            "10.5281/zenodo.20278983"]
TITLE = ("The Late Channel: Chain-of-Thought Becomes Causal and Decodable Only Late in a 27B Reasoning Agent")
DESCRIPTION = (
    "Chain-of-thought (CoT) monitorability is a leading safety bet for reasoning agents, but it is usually "
    "tested behaviorally. We ask the mechanistic version on an open-weight 27B reasoning model (Qwen3.6-27B) "
    "with a full-stack sparse autoencoder (11 layers, d_sae=40960): are the features at the post-reasoning "
    "decision point causal for what the model decides, and where do they live? A single causal-patch test "
    "(decode-re-encode of the CoT decision state into a no-CoT run, controlled for SAE reconstruction error and "
    "a random-feature baseline) gives a consistent answer for two outcomes. (1) For the answer to hard "
    "multi-step problems (n=60, 16 CoT-flips), patching the CoT features recovers the reasoned answer with "
    "delta-logp +2.72 [+2.18, +3.28] at layer 59, with the effect ~0 through layer 47 (late +2.13 vs early "
    "-0.02, disjoint 95% CIs; 16 of 16 flipped items consistent). (2) For an agent action (tool call) in "
    "trap scenarios (n=32, 10 flips), the same test gives late +1.65 [+1.17, +2.12] vs early +0.08. A "
    "method-independent control rules out the obvious confound -- that late-layer patches simply survive while "
    "early ones are washed out: a logit-lens of the decision token shows the reasoned answer is anti-decodable "
    "early (margin -0.9 at mid layers, the model leaning to the fast System-1 answer) and only becomes "
    "decodable at layer 51, reaching +6.6 at layer 63 -- the deciding information genuinely CONSOLIDATES late, "
    "it is not present earlier. Faithfulness is CONDITIONAL: causal when the CoT changes the outcome, ~0 "
    "(performative) when the model already knew the answer without thinking (44 of 60 items). The late band is "
    "thus a readable, causal locus for mechanistic monitoring of reasoning agents -- instantiating the call to "
    "'inspect the model's inner workings' -- while, per a companion result of this arc, the same band is NOT "
    "adversarially robust if used as a control point (a late action brake collapses 0 to 1.0 attack success "
    "under an adaptive white-box attack). HONEST SCOPE: curated stimulus sets (serial-computation MCQs; "
    "hand-built agent traps), modest flip counts (16 and 10), a single model, a last-token decision-residual "
    "patch; the action-class prediction AUROC is underpowered (5 positives) and is future work. Reproducibility: "
    "37 of 37 numeric claims recomputed from the released data, and the 'late = consolidation' claim is closed "
    "by the logit-lens control. This is the first mechanistic-faithfulness result of the arc's shift from "
    "interpretability-as-control (adversarial, where it loses) to interpretability-as-audit (non-adversarial, "
    "where it wins). Code, per-item data, figures, and the recompute eval are released in the GitHub repository "
    "under paper/faithfulness/.")
KEYWORDS = ["mechanistic interpretability", "chain-of-thought", "CoT faithfulness", "reasoning models",
            "sparse autoencoder", "activation patching", "logit lens", "LLM agents", "AI safety",
            "monitoring", "performative reasoning", "the lever is late", "Qwen3.6-27B", "tool-calling agents",
            "WANDERING"]


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
