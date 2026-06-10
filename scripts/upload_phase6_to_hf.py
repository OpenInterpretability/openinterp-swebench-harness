"""Upload Phase 6 SWE-bench Pro trajectory data from Drive → HF dataset.

Creates `caiovicentino1/swebench-pro-qwen36-27b-phase6` with:
  - 99 traces JSON (full per-turn tool_calls, thinking, content, tool_results)
  - 99 capture safetensors (residuals at L11/L23/L31/L43/L55 per turn)
  - 99 capture metadata JSON (token positions, structure)
  - 99 agent patches (.patch files)
  - phase6_results.json (per-instance index: finish_reason, n_turns, n_captures, etc.)
  - inflection_results.json (sub-class labels: WANDERING/SUCCESS/LOCKED)
  - kappa_t features (per-trace + cluster outputs)
  - phase6b/eval data (Docker eval verdicts)
  - phase6c methodology sweep
  - selected_iids.json (which 99 instances)
  - README.md (dataset card with usage)

Use cases:
  1. Inspect Evals tool_entropy_collapse eval (loads traces + labels)
  2. Mech-interp researchers wanting raw residuals (safetensors)
  3. SWE-bench Pro failure-mode analysis reproducibility
  4. Companion data for Tool-Entropy Collapse paper + Paper #2 (Two Honest Nulls)

Total upload: ~910 MB (traces 26MB + captures 884MB + features <1MB)
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file

REPO_ID = "caiovicentino1/swebench-pro-qwen36-27b-phase6"
REPO_TYPE = "dataset"

DRIVE = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
LOCAL = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")

def main():
    api = HfApi()

    print(f"=== Creating repo {REPO_ID} ===")
    try:
        create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True, private=False)
        print(f"  ✓ Repo ready")
    except Exception as e:
        print(f"  ! {e}")

    # Step 0: Upload README first so dataset page renders immediately
    print(f"\n=== Step 0: Upload README ===")
    readme_src = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/paper/paper2/hf_dataset_phase6_README.md")
    upload_file(path_or_fileobj=str(readme_src), path_in_repo="README.md",
                repo_id=REPO_ID, repo_type=REPO_TYPE,
                commit_message="Add dataset card with usage examples + schema docs")
    print(f"  ✓ README.md uploaded ({readme_src.stat().st_size//1024} KB)")

    # Step 1: Upload top-level metadata files (small)
    print(f"\n=== Step 1: Upload top-level Phase 6 metadata ===")
    top_files = [
        "phase6_results.json",
        "phase6_aggregate.json",
        "phase6_n99_verdict.json",
        "selected_iids.json",
        "kappa_t_per_trace.json",
        "kappa_t_failure_clusters.json",
        "kappa_t_success_clusters.json",
        "phase6c_methodology_sweep.json",
        "phase6c_preview.json",
        "phase7_steering_pilot.json",
    ]
    for fname in top_files:
        src = DRIVE / fname
        if not src.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        print(f"  → {fname} ({src.stat().st_size//1024} KB)")
        upload_file(path_or_fileobj=str(src), path_in_repo=fname,
                    repo_id=REPO_ID, repo_type=REPO_TYPE,
                    commit_message=f"Add {fname}")

    # Step 2: Upload locally-computed inflection features
    print(f"\n=== Step 2: Upload inflection_turn_out features (locally computed) ===")
    feature_files = [
        "inflection_results.json",
        "complementary_monitor.json",
        "cross_task_validation.json",
        "early_warning_results.json",
        "early_warning_v2_results.json",
        "early_warning_v3_persistence.json",
        "early_warning_v4_cross_layer.json",
        "early_warning_v4_midlayer.json",
        "early_warning_v4_op_sweep.json",
        "exp_b_determinism_check.json",
        "exp_d_forced_finish_counterfactual.json",
    ]
    for fname in feature_files:
        src = LOCAL / fname
        if not src.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        print(f"  → features/{fname} ({src.stat().st_size//1024} KB)")
        upload_file(path_or_fileobj=str(src), path_in_repo=f"features/{fname}",
                    repo_id=REPO_ID, repo_type=REPO_TYPE,
                    commit_message=f"Add features/{fname}")

    # Step 3: Upload phase6b Docker eval results
    print(f"\n=== Step 3: Upload phase6b Docker eval ===")
    p6b_dir = DRIVE / "phase6b"
    if p6b_dir.exists():
        for fname in os.listdir(p6b_dir):
            src = p6b_dir / fname
            if src.is_file():
                print(f"  → phase6b/{fname}")
                upload_file(path_or_fileobj=str(src), path_in_repo=f"phase6b/{fname}",
                            repo_id=REPO_ID, repo_type=REPO_TYPE,
                            commit_message=f"Add phase6b/{fname}")

    # Step 4: Upload traces directory (99 JSON + 99 patches = 26MB)
    print(f"\n=== Step 4: Upload traces/ (99 traces + 99 patches, ~26MB) ===")
    upload_folder(folder_path=str(DRIVE / "traces"), path_in_repo="traces",
                  repo_id=REPO_ID, repo_type=REPO_TYPE,
                  commit_message="Add 99 trajectories (traces JSON + patches)")
    print(f"  ✓ traces/ uploaded")

    # Step 5: Upload captures directory (99 safetensors + 99 meta = ~900MB)
    print(f"\n=== Step 5: Upload captures/ (~900 MB, residuals at L11/23/31/43/55) ===")
    upload_folder(folder_path=str(DRIVE / "captures"), path_in_repo="captures",
                  repo_id=REPO_ID, repo_type=REPO_TYPE,
                  commit_message="Add 99 residual captures (L11/23/31/43/55 per turn)")
    print(f"  ✓ captures/ uploaded")

    # Step 6: Print final summary + revision SHA for pinning
    print(f"\n=== Upload complete ===")
    info = api.dataset_info(REPO_ID)
    print(f"  Repo: https://huggingface.co/datasets/{REPO_ID}")
    print(f"  SHA pin (use revision= in hf_dataset): {info.sha}")
    print(f"  Files: {len(info.siblings)}")
    print(f"\nSave this SHA in code: REVISION = \"{info.sha}\"")


if __name__ == "__main__":
    main()
