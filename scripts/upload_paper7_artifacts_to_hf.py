"""Upload paper-7 reproducibility artifacts to HF dataset.

Creates `caiovicentino1/openinterp-paper7-nla-two-tier-verbalization` dataset
with the three Phase 16 result JSONs (V1 Qwen-7B + V2 Gemma-12B + V3 Gemma-27B)
plus AV explanations for each, the captured activations, and a README.

Source paths assume Mac Drive mount or Colab /content/drive.

Usage (local Mac):
    python3 scripts/upload_paper7_artifacts_to_hf.py

Usage (Colab):
    !python3 scripts/upload_paper7_artifacts_to_hf.py
"""
from __future__ import annotations

import os
import sys
import platform
from pathlib import Path

from huggingface_hub import HfApi, login

REPO_ID = "caiovicentino1/openinterp-paper7-nla-two-tier-verbalization"

# Auto-detect Drive root: Colab vs Mac
COLAB_ROOT = Path("/content/drive/MyDrive/openinterp_runs")
MAC_ROOT = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs")

if COLAB_ROOT.exists():
    DRIVE_ROOT = COLAB_ROOT
elif MAC_ROOT.exists():
    DRIVE_ROOT = MAC_ROOT
else:
    raise SystemExit(f"Could not find Drive root. Tried: {COLAB_ROOT}, {MAC_ROOT}")

# Files to upload per phase
SOURCES = [
    # (drive_subfolder, hf_subdir, [(file_glob, required)])
    ("track_a_nla_qwen25_poc", "v1_qwen2.5-7b_L20", [
        ("phase16_results_v2.json", True),
        ("phase16_controls.json", True),
        ("phase16_direction_interp.json", True),
        ("phase16_explanations.json", True),
    ]),
    ("track_a_phase16_gemma", "v2_gemma-3-12b_L32", [
        ("phase16_full_results.json", True),
        ("phase16_explanations.json", True),
    ]),
    ("track_a_phase16_gemma27b", "v3_gemma-3-27b_L41", [
        ("phase16_full_results.json", True),
        ("phase16_explanations.json", True),
    ]),
]

DATASET_README = """---
license: apache-2.0
language:
- en
tags:
- mechanistic-interpretability
- natural-language-autoencoders
- nla
- activation-decoding
- format-priors
- decoupling-magnification
- Qwen2.5-7B
- Gemma-3-12B
- Gemma-3-27B
size_categories:
- n<1K
---

# Paper-7: Reconstruction Without Recall — NLA Two-Tier Verbalization

Reproducibility artifacts for **"Reconstruction Without Recall: Two-Tier
Verbalization in Natural Language Autoencoders"** (Vicentino, May 2026).

> NLA's headline metric `fve_nrm` (reconstruction loss) decouples from
> semantic content fidelity (keyword recall) across three NLA pairs from
> the kitft release spanning two model families and three scales. As NLA
> training quality improves, `fve_nrm` saturates toward its 1.0 ceiling
> while per-category recall spread grows then plateaus at a
> training-distribution-imbalance limit. Better NLA training makes
> `fve_nrm` *less*, not more, informative about explanation quality.

## Paper

- **Live**: https://openinterp.org/research/papers/nla-two-tier-verbalization
- **Source markdown**: [openinterpretability-web/content/papers/nla-two-tier-verbalization.md](https://github.com/OpenInterpretability/web/blob/main/content/papers/nla-two-tier-verbalization.md)

## Reproducibility notebooks

| Notebook | Model | Compute |
|---|---|---|
| [V1 — Qwen2.5-7B-L20](https://github.com/OpenInterpretability/openinterp-swebench-harness/blob/main/notebooks/nb_track_a_phase16_decoupling.ipynb) | `kitft/nla-qwen2.5-7b-L20-{av,ar}` + `Qwen/Qwen2.5-7B-Instruct` | ~30 min H100 |
| [V2 — Gemma-3-12B-L32](https://github.com/OpenInterpretability/openinterp-swebench-harness/blob/main/notebooks/nb_track_a_phase16_gemma_crossmodel.ipynb) | `kitft/nla-gemma3-12b-L32-{av,ar}` + `google/gemma-3-12b-it` | ~45 min H100 |
| [V3 — Gemma-3-27B-L41](https://github.com/OpenInterpretability/openinterp-swebench-harness/blob/main/notebooks/nb_track_a_phase16_gemma27b_v3.ipynb) | `kitft/nla-gemma3-27b-L41-{av,ar}` + `google/gemma-3-27b-it` | ~60 min RTX 6000 96GB |

## Three-model headline trajectory

| Metric | V1 Qwen-7B | V2 Gemma-12B | V3 Gemma-27B |
|---|---|---|---|
| Overall fve_nrm | 0.880 | **0.992** | 0.982 |
| fve_nrm category spread | 0.017 | 0.005 | 0.010 |
| Overall recall | 0.336 | 0.420 | **0.474** |
| Recall category spread | 0.490 | 0.649 | **0.654** (saturates) |
| Permutation gap above floor | +0.27 | +0.38 | **+0.43** (monotonic) |
| Random Gaussian fve_nrm | -0.949 | -0.992 | **-1.000** |
| Random Gaussian cos | +0.026 | +0.004 | **+0.000** |
| Direction-injection self-cat | 4/4 | 3/4 (agent→code) | 3/4 (agent→chat) |

## Per-category recall trajectory (4 categories × 3 models)

| Category | V1 Qwen-7B | V2 Gemma-12B | V3 Gemma-27B |
|---|---|---|---|
| chat | 0.578 | 0.782 | 0.813 |
| code | 0.351 | 0.404 | 0.492 |
| agent | **0.088** (floor) | **0.133** (floor) | **0.160** (floor) |
| reasoning | 0.325 | 0.361 | 0.432 |

## Three differential scaling axes

The decoupling magnification thesis evolves from single-axis (V1+V2) to
three-axis differential (V1+V2+V3):

1. **Overall content-fidelity (permutation gap)** — monotonic up, no ceiling visible
2. **Per-category recall spread** — saturates at training-distribution-imbalance ceiling (~0.65 between 12B-27B)
3. **Tier 1 fve_nrm** — peaks at moderate model size, slight regression at 27B (layer-extraction-dependent, not pure scale-dependent)

## Format-prior contraction (V3 finding)

As Tier 1 saturates toward fve_nrm ceiling, the verbalization template space CONTRACTS rather than expanding:
- V1 Qwen-7B: heterogeneous formats ("Wikipedia article", "game description", "ISO standard", "math content")
- V2 Gemma-12B: 6/6 random Gaussian explanations begin with "Structured X format"
- V3 Gemma-27B: 6/6 random Gaussian explanations begin with "Educational/X article format" — single hyper-template attractor

Better-trained NLA's Tier 1 prior becomes MORE narrow, not more diverse.

## Files

```
v1_qwen2.5-7b_L20/
├── phase16_results_v2.json    # 150 (act, explanation, fve_nrm, cos, recall) tuples
├── phase16_controls.json      # permutation + random Gaussian results
├── phase16_direction_interp.json  # 10 direction-injection results
└── phase16_explanations.json  # raw 150 AV explanations

v2_gemma-3-12b_L32/
├── phase16_full_results.json  # all-in-one (150 tuples + controls + direction)
└── phase16_explanations.json

v3_gemma-3-27b_L41/
├── phase16_full_results.json
└── phase16_explanations.json
```

## Citation

```bibtex
@article{vicentino2026nla,
  author = {Vicentino, Caio},
  title = {Reconstruction Without Recall: Two-Tier Verbalization in Natural Language Autoencoders},
  journal = {OpenInterpretability — workshop draft for NeurIPS 2026 MI Workshop},
  year = {2026},
  url = {https://openinterp.org/research/papers/nla-two-tier-verbalization}
}
```

## License

Apache-2.0 throughout.

The released kitft NLA pairs are Apache-2.0 (Fraser-Taliente et al. 2026).
The Qwen2.5-7B-Instruct target is Apache-2.0 (Alibaba). The Gemma-3-12B-IT
and Gemma-3-27B-IT targets are licensed under the Gemma Terms of Use (Google).

## Related papers (4-paper Anthropic-aligned methodology corpus)

1. [saturation-direction-probe-levers](https://openinterp.org/research/papers/saturation-direction-probe-levers) — five-class probe causality taxonomy
2. [activation-bounded-cot-monitorability](https://openinterp.org/research/papers/activation-bounded-cot-monitorability) — text-only CoT structural ceiling
3. [two-forms-epiphenomenal-probes](https://openinterp.org/research/papers/two-forms-epiphenomenal-probes) — softmax-temp + template-lock probe failures
4. **[nla-two-tier-verbalization](https://openinterp.org/research/papers/nla-two-tier-verbalization)** ← this paper
"""


def upload():
    if not os.environ.get("HF_TOKEN"):
        try:
            login()
        except Exception as e:
            raise SystemExit(f"HF login failed: {e}. Set HF_TOKEN env var first.")
    else:
        login(token=os.environ["HF_TOKEN"])

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    print(f"✓ repo: https://huggingface.co/datasets/{REPO_ID}")

    # Write README to a temp local path then upload
    readme_path = Path("/tmp/paper7_readme.md")
    readme_path.write_text(DATASET_README)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Add paper-7 README",
    )
    print(f"✓ uploaded README.md")

    total_uploaded = 0
    for sub_drive, sub_hf, files in SOURCES:
        src_dir = DRIVE_ROOT / sub_drive
        if not src_dir.exists():
            print(f"⚠ skipping {sub_drive} (not found in {DRIVE_ROOT})")
            continue
        for fname, required in files:
            src_file = src_dir / fname
            if not src_file.exists():
                msg = f"⚠ missing {sub_drive}/{fname}"
                if required:
                    print(f"FAIL: {msg}")
                    raise SystemExit(1)
                else:
                    print(msg)
                    continue
            api.upload_file(
                path_or_fileobj=str(src_file),
                path_in_repo=f"{sub_hf}/{fname}",
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"upload {sub_hf}/{fname}",
            )
            sz_kb = src_file.stat().st_size / 1024
            print(f"  ✓ {sub_hf}/{fname}  ({sz_kb:.0f} KB)")
            total_uploaded += 1

    print(f"\n✓ uploaded {total_uploaded} files + README")
    print(f"  https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    upload()
