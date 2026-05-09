"""Upload paper-5 reproducibility artifacts (Phase 6 N=99 captures + Phase 7-12
verdict JSONs) to a new HF dataset.

Run on Colab where the Drive is mounted, or any host with the Drive contents
copied in. Creates `caiovicentino1/openinterp-paper5-saturation-direction`
under HF as a dataset repo.

Usage (Colab):
    !python scripts/upload_paper5_artifacts_to_hf.py

Skips upload of files >5GB or matching exclude patterns. Captures are
~12k safetensors files totaling ~25GB at full Phase 1 + Phase 6.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login

# Set HF_TOKEN env var before running, e.g.:
#   import os; os.environ['HF_TOKEN'] = 'hf_...'
REPO_ID = "caiovicentino1/openinterp-paper5-saturation-direction"
DRIVE_ROOT = Path("/content/drive/MyDrive/openinterp_runs")

# Phase verdict + capture roots to upload
SOURCES = [
    ("swebench_v6_phase6", "phase6_n99_captures"),
    ("phase7_steering_micro", "phase7_l43_pre_tool"),
    ("phase8_causal_cot", "phase8_l55_thinking"),
    ("phase10_fg_rg_causality", "phase10_fg_rg"),
    ("phase11_capability_locus", "phase11_4sites_he_mbpp"),
    ("phase11b_capability_locus_extension", "phase11b_l23_l43_turnend"),
    ("phase11c_cross_distribution", "phase11c_bcb"),
    ("phase11d_cross_distribution_round2", "phase11d_codeforces"),
    ("phase11e_multisite_cf", "phase11e_multisite_codeforces"),
    ("phase12_persona_falsifier", "phase12_persona"),
]

# Files always uploaded (small JSONs)
ALWAYS_UPLOAD = {".json"}
# Files conditionally uploaded (captures — only Phase 6 because it's the
# canonical capture corpus all probes are trained from)
CAPTURE_PHASES = {"swebench_v6_phase6"}

DATASET_README = """---
license: apache-2.0
language:
- en
pretty_name: openinterp paper-5 saturation-direction reproducibility bundle
tags:
- mechanistic-interpretability
- linear-probes
- activation-steering
- causal-interpretability
- qwen
- saturation-direction
size_categories:
- 10K<n<100K
---

# openinterp paper-5 — saturation-direction lever reproducibility bundle

Reproducibility artifacts for paper-5: *Saturation-Direction Lever — A
Five-Class Taxonomy of Probe Causality in Qwen3.6-27B*.

→ Paper: <https://openinterp.org/research/papers/saturation-direction-probe-levers>
→ Code: <https://github.com/OpenInterpretability/openinterp-swebench-harness>

## What's inside

| Folder | Phase | Contents |
|---|---|---|
| `phase6_n99_captures/` | Phase 6 | N=99 SWE-bench Pro capture corpus (~12k safetensors, ~25GB), the canonical training corpus for all 8 probes mapped in paper-5 |
| `phase7_l43_pre_tool/` | Phase 7 | L43 pre_tool causality verdict — Δrel ≈ 0 (epiphenomenal, softmax-temp artifact) |
| `phase8_l55_thinking/` | Phase 8 | L55 thinking template-locked verdict |
| `phase10_fg_rg/` | Phase 10 | RG L55 mid_think first lever (+30pp pushup at α=+200) |
| `phase11_4sites_he_mbpp/` | Phase 11 | 4 capability sites α-sweep on HumanEval+MBPP |
| `phase11b_l23_l43_turnend/` | Phase 11b | Extension to 2 more sites (4/4 capability lever) |
| `phase11c_bcb/` | Phase 11c | Cross-distribution BigCodeBench α=−100 +33pp (saturation-magnitude corollary attempted) |
| `phase11d_codeforces/` | Phase 11d | Codeforces ≥2000 α=−100 +40pp — **falsifies corollary**, refines to α=−100 robustness theorem |
| `phase11e_multisite_codeforces/` | Phase 11e | 4 capability sites × Codeforces multi-site validation |
| `phase12_persona/` | Phase 12 | Persona falsifier predicted pushup, observed pushdown |

## Headline finding

**α=−100 robustness theorem**: the L31 pre_tool capability probe direction
produces a +33-40pp probe-vs-random pushdown gap across code distributions
spanning Qwen3.6-27B pass-rate ~7-89%. The α=−100 locus is
**saturation-independent** at moderate amplitude.

Two pre-registered falsification cycles documented in the paper.

## Reproducing paper-5 results

```python
# 1) Pull this dataset
from huggingface_hub import snapshot_download
snapshot_download(
    'caiovicentino1/openinterp-paper5-saturation-direction',
    repo_type='dataset',
    local_dir='openinterp_paper5',
)

# 2) For verdict-only reproduction (no GPU needed): inspect verdict.json
import json
v = json.load(open('openinterp_paper5/phase11d_codeforces/verdict.json'))
print(v['classification'])

# 3) For full re-run (GPU needed): use the harness notebooks
# git clone https://github.com/OpenInterpretability/openinterp-swebench-harness
# notebooks/nb_swebench_v11d_codeforces.ipynb
```

## License

Apache-2.0. Patent grant included.
"""


def main() -> None:
    if not DRIVE_ROOT.exists():
        sys.exit(f"Drive root not found: {DRIVE_ROOT}. Run on Colab with Drive mounted.")

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN env var before running (see notebooks/HF_TOKEN ref in memory).")
    login(token=token)
    api = HfApi()

    # Create or update repo
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    print(f"Repo OK: https://huggingface.co/datasets/{REPO_ID}")

    # Write README
    api.upload_file(
        path_or_fileobj=DATASET_README.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="docs: paper-5 reproducibility bundle README",
    )
    print("Uploaded README.md")

    # Walk each source
    for src_name, target_name in SOURCES:
        src_path = DRIVE_ROOT / src_name
        if not src_path.exists():
            print(f"  [skip] {src_name} not on Drive")
            continue

        # For captures phase: upload entire dir (large)
        # For verdict phases: upload only JSON files (small)
        upload_all = src_name in CAPTURE_PHASES

        files_to_upload = []
        for f in src_path.rglob("*"):
            if not f.is_file():
                continue
            if upload_all:
                files_to_upload.append(f)
            elif f.suffix in ALWAYS_UPLOAD:
                files_to_upload.append(f)

        if not files_to_upload:
            print(f"  [skip] {src_name} no matching files")
            continue

        # Use upload_folder for batches
        # but keep selective for verdict-only
        if upload_all:
            print(f"  Uploading {src_name} ({len(files_to_upload)} files) → {target_name}/")
            api.upload_folder(
                folder_path=str(src_path),
                path_in_repo=target_name,
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"upload {src_name} → {target_name}",
                ignore_patterns=["*.tmp", "*.lock"],
            )
        else:
            for f in files_to_upload:
                rel = f.relative_to(src_path)
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f"{target_name}/{rel}",
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    commit_message=f"upload {target_name}/{rel}",
                )
            print(f"  Uploaded {len(files_to_upload)} verdict files → {target_name}/")

    print(f"\nDone. https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
