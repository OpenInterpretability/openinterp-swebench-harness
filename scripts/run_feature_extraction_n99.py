#!/usr/bin/env python3
"""Run multi-channel feature extraction on full N=99 and save CSV."""

import json
import sys
import time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from extract_multi_channel_features import (
    extract_all_features,
    _build_precomputed,
    PHASE6,
)

INFLECTION = Path(
    "/Volumes/SSD Major/fish/openinterp-swebench-harness/"
    "scripts/inflection_turn_out/inflection_results.json"
)
V4 = Path(
    "/Volumes/SSD Major/fish/openinterp-swebench-harness/"
    "scripts/inflection_turn_out/early_warning_v4_cross_layer.json"
)
PHASE6_DOC = PHASE6 / "phase6_results.json"
OUT_CSV = Path(
    "/Volumes/SSD Major/fish/openinterp-swebench-harness/"
    "scripts/inflection_turn_out/features_n99.csv"
)


def sub_class_from_inflection(t):
    if t["label"] == 1:
        return "success"
    if t.get("lock_fail_0.40") is not None:
        return "locked"
    return "wandering"


def main():
    inflection_doc = json.loads(INFLECTION.read_text())
    v4_doc = json.loads(V4.read_text())
    phase6_doc = json.loads(PHASE6_DOC.read_text())

    rows = []
    t_start = time.time()
    errs = []

    for i, traj in enumerate(inflection_doc["per_trajectory"]):
        iid = traj["iid"]
        sub_cls = sub_class_from_inflection(traj)
        trace_path = PHASE6 / "traces" / f"{iid}.json"
        st_path = PHASE6 / "captures" / f"{iid}.safetensors"

        if not trace_path.exists() or not st_path.exists():
            errs.append((iid, "missing_files"))
            continue

        try:
            precomputed = _build_precomputed(iid, inflection_doc, v4_doc, phase6_doc)
            t0 = time.time()
            row = extract_all_features(iid, trace_path, st_path, precomputed)
            dt = time.time() - t0
            row["sub_class"] = sub_cls
            rows.append(row)
            elapsed = time.time() - t_start
            print(f"[{i+1}/99] {sub_cls[:5]:<5} {dt:5.2f}s {iid[20:60]}...")
        except Exception as e:
            errs.append((iid, str(e)[:120]))
            print(f"[{i+1}/99] FAIL {iid[20:60]}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n=== DONE ===")
    print(f"Rows: {len(rows)} / 99")
    print(f"Errors: {len(errs)}")
    if errs:
        for iid, e in errs[:10]:
            print(f"  {iid}: {e}")
    print(f"Wall: {time.time()-t_start:.1f}s")
    print(f"Output: {OUT_CSV}")
    print(f"Columns: {list(df.columns)[:5]}...{list(df.columns)[-3:]}")
    print(f"Sub-class counts: {df['sub_class'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
