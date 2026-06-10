#!/usr/bin/env python3
"""
Full N=99 Liu et al. taxonomy judge run via OpenRouter (Opus 4.7).

Iterates over all 99 Phase 6 trajectories, classifies each via Opus 4.7,
saves outputs to liu_taxonomy_assignments.json with per-trajectory parsed result.

Usage:
    OPENROUTER_API_KEY=sk-or-... python3 paper3_liu_judge_fullrun.py
"""

import json
import os
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

# Reuse test script helpers
sys.path.insert(0, str(Path(__file__).parent))
from paper3_liu_judge_test import (
    load_prompt,
    load_tool_entropy_index,
    compress_trajectory,
    call_openrouter,
    parse_judge_output,
    TRACES_DIR,
)

OUT_PATH = Path(
    "/Volumes/SSD Major/fish/openinterp-swebench-harness/"
    "scripts/inflection_turn_out/liu_taxonomy_assignments.json"
)
MODEL = os.environ.get("LIU_JUDGE_MODEL", "anthropic/claude-opus-4.7")


def main():
    system_prompt, user_template = load_prompt()
    te_idx = load_tool_entropy_index()
    print(f"Loaded {len(te_idx)} trajectories from tool_entropy index")
    print(f"Model: {MODEL}")
    print(f"Output: {OUT_PATH}\n")

    # Resume support: if output exists, skip already-classified iids
    existing = {}
    if OUT_PATH.exists():
        try:
            existing = json.loads(OUT_PATH.read_text())
            print(f"Resume: found {len(existing)} existing classifications, skipping those\n")
        except Exception:
            existing = {}

    results = dict(existing)
    total_in = 0
    total_out = 0
    n_done = 0
    n_err = 0
    n_skip = len(existing)

    iids = sorted(te_idx.keys())
    t_start = time.time()

    for i, iid in enumerate(iids):
        if iid in existing:
            continue
        te_row = te_idx[iid]
        trace_path = TRACES_DIR / f"{iid}.json"
        patch_path = TRACES_DIR / f"{iid}.patch"

        if not trace_path.exists():
            print(f"[{i+1}/99] MISSING trace: {iid}")
            results[iid] = {"_error": "trace_missing", "sub_class": te_row.get("sub_class")}
            n_err += 1
            continue

        try:
            fields = compress_trajectory(trace_path, te_row, patch_path)
        except Exception as e:
            print(f"[{i+1}/99] COMPRESS FAIL: {iid} -- {e}")
            results[iid] = {"_error": f"compress_fail: {e}", "sub_class": te_row.get("sub_class")}
            n_err += 1
            continue

        user_prompt = user_template.format(**fields)

        try:
            resp = call_openrouter(system_prompt, user_prompt, MODEL)
        except Exception as e:
            print(f"[{i+1}/99] API FAIL: {iid} -- {str(e)[:120]}")
            results[iid] = {"_error": f"api_fail: {e}", "sub_class": te_row.get("sub_class")}
            n_err += 1
            # save progress + brief sleep on API error
            OUT_PATH.write_text(json.dumps(results, indent=2))
            time.sleep(2)
            continue

        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        total_in += in_tok
        total_out += out_tok

        parsed = parse_judge_output(content)
        parsed["_sub_class"] = te_row.get("sub_class")
        parsed["_n_turns"] = fields["n_turns"]
        parsed["_repo"] = fields["repo"]
        parsed["_usage"] = {"in": in_tok, "out": out_tok}
        results[iid] = parsed

        n_done += 1
        elapsed = time.time() - t_start
        pace = n_done / max(elapsed, 1)
        eta = (99 - i - 1) / max(pace, 0.01)

        prim = parsed.get("primary_category", "?")
        conf = parsed.get("confidence", "?")
        sc = te_row.get("sub_class")
        print(f"[{i+1}/99] {sc[:5]:<5} {iid[20:60]:<40} -> {prim} (conf={conf}) | {in_tok}+{out_tok} tok | eta {eta:.0f}s")

        # save progress every 5 trajectories
        if n_done % 5 == 0:
            OUT_PATH.write_text(json.dumps(results, indent=2))

    # final save
    OUT_PATH.write_text(json.dumps(results, indent=2))
    elapsed = time.time() - t_start

    print(f"\n=== DONE ===")
    print(f"Trajectories: {n_done} classified, {n_err} errors, {n_skip} skipped (resume)")
    print(f"Wall: {elapsed:.1f}s")
    print(f"Tokens: in={total_in}, out={total_out}")
    cost = (total_in * 15.0 + total_out * 75.0) / 1_000_000
    print(f"Cost (opus-4.7 @ $15in/$75out per MTok): ${cost:.2f}")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
