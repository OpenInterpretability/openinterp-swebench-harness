#!/usr/bin/env python3
"""Generic Zenodo NEW VERSION: replace the PDF on an existing concept record, append a short version note to the
description, set a version label. Preserves ALL inherited metadata (title, creators, keywords,
related_identifiers) by reading the draft's inherited metadata and only modifying description + version.

USAGE (token ephemeral -- paste inline, never commit):
  ZENODO_TOKEN='...' python3 scripts/zenodo_newversion_generic.py --state .zenodo_late_channel.json \
      --pdf paper/faithfulness/late_channel.pdf --version v2 --note "..."            # DRAFT (reversible)
  add --publish to mint (IRREVERSIBLE).
"""
import argparse, json, os, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
BASE = "https://zenodo.org/api/deposit/depositions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="state json filename in scripts/ (has deposition_id of v1)")
    ap.add_argument("--pdf", required=True, help="path (repo-relative) to the new PDF")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--note", required=True, help="short note appended to the description for this version")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        sys.exit("ERROR: set ZENODO_TOKEN")
    auth = {"access_token": token}
    st_path = HERE / args.state
    v1 = json.loads(st_path.read_text())
    v1_id = v1["deposition_id"]
    pdf = ROOT / args.pdf
    if not pdf.exists():
        sys.exit(f"ERROR: missing PDF {pdf}")
    new_state_path = HERE / (st_path.stem + "_v2.json")
    if new_state_path.exists():
        prior = json.loads(new_state_path.read_text())
        if args.publish and prior.get("deposition_id") and not prior.get("published_doi"):
            dep_id = prior["deposition_id"]
            print(f"publishing existing v2 draft {dep_id} (irreversible) ...")
            rp = requests.post(f"{BASE}/{dep_id}/actions/publish", params=auth, timeout=120)
            if rp.status_code >= 300:
                sys.exit(f"publish error {rp.status_code}: {rp.text[:500]}")
            pub = rp.json(); print("  PUBLISHED. DOI:", pub.get("doi"), "|", pub.get("links", {}).get("record_html"))
            prior["published_doi"] = pub.get("doi"); new_state_path.write_text(json.dumps(prior, indent=2)); return
        sys.exit(f"ERROR: {new_state_path.name} exists. Inspect or remove.")

    print(f"opening new version of deposition {v1_id} ...")
    r = requests.post(f"{BASE}/{v1_id}/actions/newversion", params=auth, timeout=60)
    if r.status_code >= 300:
        sys.exit(f"newversion error {r.status_code}: {r.text[:400]}")
    latest = r.json()["links"]["latest_draft"]
    dep = requests.get(latest, params=auth, timeout=60).json()
    dep_id = dep["id"]; bucket = dep["links"]["bucket"]
    md = dep.get("metadata", {})  # INHERITED metadata -- preserve everything
    print(f"  draft id={dep_id} | inherited related_identifiers={len(md.get('related_identifiers',[]))}")

    # remove inherited files, upload the new PDF
    for f in dep.get("files", []):
        requests.delete(f"{BASE}/{dep_id}/files/{f.get('id')}", params=auth, timeout=60)
    print(f"  uploading {pdf.name} ...")
    with open(pdf, "rb") as fh:
        requests.put(f"{bucket}/{pdf.name}", data=fh, params=auth, timeout=300).raise_for_status()

    # modify ONLY description + version
    desc = md.get("description", "")
    if args.note not in desc:
        desc = desc + " " + args.note
    md["description"] = desc
    md["version"] = args.version
    rm = requests.put(f"{BASE}/{dep_id}", params=auth, json={"metadata": md},
                      headers={"Content-Type": "application/json"}, timeout=60)
    if rm.status_code >= 300:
        sys.exit(f"metadata error {rm.status_code}: {rm.text[:500]}")
    dep = rm.json()
    state = {"deposition_id": dep_id, "from_version": v1_id,
             "doi": dep.get("metadata", {}).get("prereserve_doi", {}).get("doi"),
             "html": dep.get("links", {}).get("html")}
    new_state_path.write_text(json.dumps(state, indent=2))
    print(f"  DRAFT ready. Pre-reserved DOI: {state['doi']} | files: {[f.get('filename') for f in dep.get('files',[])]}")
    print(f"  related_identifiers now: {len(dep.get('metadata',{}).get('related_identifiers',[]))} | version: {dep.get('metadata',{}).get('version')}")
    print(f"  UI: {state['html']}")
    if args.publish:
        print("PUBLISHING (irreversible) ...")
        rp = requests.post(f"{BASE}/{dep_id}/actions/publish", params=auth, timeout=120)
        if rp.status_code >= 300:
            sys.exit(f"publish error {rp.status_code}: {rp.text[:500]}")
        pub = rp.json(); print("  PUBLISHED. DOI:", pub.get("doi"), "|", pub.get("links", {}).get("record_html"))
        state["published_doi"] = pub.get("doi"); new_state_path.write_text(json.dumps(state, indent=2))
    else:
        print("  (draft only -- re-run with --publish to mint)")


if __name__ == "__main__":
    main()
