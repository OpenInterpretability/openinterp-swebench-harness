#!/bin/bash
# Zenodo upload script for Tool-Entropy Collapse paper.
# Usage: ZENODO_TOKEN='<token>' bash paper/zenodo_upload.sh
#
# Workflow:
#   1. Create empty deposition
#   2. Upload PDF + 5 figures
#   3. Set metadata (title, authors, license, etc.)
#   4. Publish (mints DOI — IRREVERSIBLE)
#
# Per zenodo-publishing-workflow memory: bucket locks after publish.
# Plan all files upfront.

set -euo pipefail

if [ -z "${ZENODO_TOKEN:-}" ]; then
    echo "ERROR: set ZENODO_TOKEN env var first"
    echo "  export ZENODO_TOKEN='<your-token>'"
    echo "  bash paper/zenodo_upload.sh"
    exit 1
fi

ROOT="/Volumes/SSD Major/fish/openinterp-swebench-harness"
cd "$ROOT"

API="https://zenodo.org/api/deposit/depositions"

echo "=== Step 1: Create deposition ==="
RESP=$(curl -s -X POST "${API}?access_token=${ZENODO_TOKEN}" \
    -H "Content-Type: application/json" -d '{}')
DEP_ID=$(echo "$RESP" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
BUCKET=$(echo "$RESP" | python3 -c "import sys,json;print(json.load(sys.stdin)['links']['bucket'])")
echo "  Created deposition: $DEP_ID"
echo "  Bucket: $BUCKET"

echo ""
echo "=== Step 2: Upload PDF + figures ==="
upload() {
    local file="$1"
    local name="$(basename "$file")"
    echo "  Uploading $name..."
    curl -s -X PUT "${BUCKET}/${name}?access_token=${ZENODO_TOKEN}" \
        -H "Content-Type: application/octet-stream" \
        --data-binary @"$file" > /dev/null
}
upload "paper/inflection_wandering.pdf"
upload "paper/inflection_wandering.tex"
upload "paper/figures/fig1_cross_arch_entropy.pdf"
upload "paper/figures/fig2_disagreement_trajectory.pdf"
upload "paper/figures/fig3_detector_comparison.pdf"
upload "paper/figures/fig4_lab_summary.pdf"
upload "paper/figures/fig5_venn_orthogonality.pdf"

echo ""
echo "=== Step 3: Set metadata ==="
curl -s -X PUT "${API}/${DEP_ID}?access_token=${ZENODO_TOKEN}" \
    -H "Content-Type: application/json" \
    -d @paper/zenodo_metadata.json > /tmp/zenodo_metadata_resp.json
echo "  Metadata set."

echo ""
echo "=== Step 4: Publish (IRREVERSIBLE, mints DOI) ==="
read -p "Type 'PUBLISH' to proceed (or Ctrl-C to abort): " CONFIRM
if [ "$CONFIRM" != "PUBLISH" ]; then
    echo "Aborted. Deposition $DEP_ID remains as draft."
    echo "View/edit: https://zenodo.org/deposit/${DEP_ID}"
    exit 0
fi

RESP=$(curl -s -X POST "${API}/${DEP_ID}/actions/publish?access_token=${ZENODO_TOKEN}")
DOI=$(echo "$RESP" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('doi','ERR: '+str(d)))")
DOI_URL=$(echo "$RESP" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('doi_url',''))")
echo ""
echo "==============================================="
echo "PUBLISHED. DOI: $DOI"
echo "URL: $DOI_URL"
echo "==============================================="
echo ""
echo "Cite as:"
echo "  Vicentino, C. (2026). Tool-Entropy Collapse: A Cross-Architecture"
echo "  Signature of Agent WANDERING Failure. Zenodo. ${DOI_URL}"
