"""
v5 cross-model validation — Claude 3.7 Sonnet + GPT-5/5-mini.

Tests if tool-entropy WANDERING signal generalizes to ALL 4 major model
families (Qwen ✓, Llama ✓, Claude ?, GPT ?).

Datasets:
  - SWE-bench/SWE-smith-trajectories — Claude 3.7 Sonnet (3,260/shard × 8 = ~26k)
  - JetBrains-Research/agent-trajectories-swesmith-random-subset — GPT-5 + GPT-5-mini
    router (1,465 total)

Both have `messages` arrays + `resolved` (success bool). Need to handle
multiple tool-call formats:
  - OpenAI-style: assistant message with `tool_calls` field (name + arguments)
  - mini-swe-agent: assistant message with bash code block (first command word)
  - Mixed: try both per message

Sub-classification (behavioral):
  SUCCESS: resolved=True
  WANDERING (proxy): resolved=False AND patch non-empty AND n_messages high (ran out with patch)
  LOCKED: resolved=False AND (no patch OR short trajectory)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import mannwhitneyu

OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")
CODE_BLOCK_RE = re.compile(r"```(?:bash|shell|sh)?\s*\n(.*?)```", re.DOTALL)
# Claude SWE-smith XML-style: <function=bash>, <function=str_replace_editor>, etc.
XML_FUNCTION_RE = re.compile(r"<function=([a-zA-Z_][a-zA-Z0-9_]*)\b")


def shannon_entropy(items):
    if not items:
        return 0.0
    cnt = Counter(items)
    total = sum(cnt.values())
    return -sum((c/total) * np.log2(c/total) for c in cnt.values() if c > 0)


def extract_tool_calls_unified(messages):
    """Try multiple formats; returns list of per-turn tool names (list-of-lists)."""
    turn_tools = []
    if messages is None:
        return []
    for entry in messages:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        if role not in ("assistant", "ai"):
            continue
        names = []
        # Format 1: OpenAI-style tool_calls
        tc = entry.get("tool_calls")
        if tc is not None:
            try:
                for c in tc:
                    if isinstance(c, dict):
                        fn = c.get("function", c)
                        if isinstance(fn, dict):
                            n = fn.get("name", "")
                            if n:
                                names.append(n.lower())
            except Exception:
                pass
        # Get content as string
        content = entry.get("content") or entry.get("text") or ""
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not isinstance(content, str):
            content = ""

        # Format 2: XML-style <function=X> (Claude SWE-smith)
        if not names:
            xml_funcs = XML_FUNCTION_RE.findall(content)
            for fn in xml_funcs:
                names.append(fn.lower())
            # For str_replace_editor, the actual subtool is in parameter=command — extract too
            for fn in xml_funcs:
                if fn == "str_replace_editor":
                    cmd_match = re.search(r"<parameter=command>\s*(\w+)", content)
                    if cmd_match:
                        # Replace last 'str_replace_editor' with more specific
                        if names and names[-1] == "str_replace_editor":
                            names[-1] = f"editor_{cmd_match.group(1).lower()}"

        # Format 3: bash code block in content (GPT-5, Llama mini-swe-agent)
        if not names and content:
            blocks = CODE_BLOCK_RE.findall(content)
            for blk in blocks:
                first_line = blk.strip().split("\n")[0].strip()
                first_word = first_line.split()[0] if first_line.split() else ""
                if first_word and len(first_word) < 40:
                    names.append(first_word.strip(":").lower())
        if names:
            turn_tools.append(names)
    return turn_tools


def sub_classify(resolved, patch, n_turns=None, exit_status=None):
    if resolved:
        return "success"
    has_patch = bool(patch and isinstance(patch, str) and len(patch.strip()) > 0)
    # WANDERING proxy: failed but produced patch and ran for a while
    # Use n_turns >= 5 OR exit status indicates context/budget exhaustion
    long_traj = (n_turns is not None and n_turns >= 5)
    ctx_exhaust = exit_status and any(
        s in str(exit_status).lower() for s in ("exit_context", "max_turns", "budget", "submitted (exit", "limitsexceeded", "limits")
    )
    if has_patch and (long_traj or ctx_exhaust):
        return "wandering"
    return "locked"


def process_claude():
    print("\n=== CLAUDE 3.7 SONNET (SWE-bench/SWE-smith-trajectories) ===")
    # Download just one shard to start
    p = hf_hub_download('SWE-bench/SWE-smith-trajectories', 'data/train-00000-of-00008.parquet', repo_type='dataset')
    df = pd.read_parquet(p)
    print(f"Shape: {df.shape}")
    print(f"resolved: True={df['resolved'].sum()}, False={(~df['resolved']).sum()}")
    print(f"model values: {df['model'].value_counts().head(5).to_dict()}")

    metrics = []
    for i, row in df.iterrows():
        n_msgs = len(row['messages']) if row['messages'] is not None else 0
        sc = sub_classify(row['resolved'], row['patch'], n_turns=n_msgs)
        tt = extract_tool_calls_unified(row['messages'])
        last10 = tt[-10:] if len(tt) >= 10 else tt
        all_tools = [t for sub_l in last10 for t in sub_l]
        metrics.append({
            "iid": row["instance_id"],
            "sub_class": sc,
            "model": row["model"],
            "n_msgs": n_msgs,
            "n_turns_extracted": len(tt),
            "tool_entropy_last10": shannon_entropy(all_tools),
            "n_tools_last10": len(all_tools),
        })
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(df)}")

    mdf = pd.DataFrame(metrics)
    print(f"\nSub-class distribution (all): {mdf['sub_class'].value_counts().to_dict()}")
    mdf = mdf[mdf['n_turns_extracted'] >= 5].copy()
    print(f"After n_turns_extracted>=5: {len(mdf)} ({mdf['sub_class'].value_counts().to_dict()})")
    print(f"Sample extraction stats: n_tools_last10 mean={mdf['n_tools_last10'].mean():.1f}, median={mdf['n_tools_last10'].median():.0f}")
    return mdf


def process_gpt():
    print("\n=== GPT-5 + GPT-5-mini (JetBrains-Research/agent-trajectories-swesmith-random-subset) ===")
    p = hf_hub_download('JetBrains-Research/agent-trajectories-swesmith-random-subset', 'data/train-00000-of-00001.parquet', repo_type='dataset')
    df = pd.read_parquet(p)
    print(f"Shape: {df.shape}")
    # Handle nulls in resolved
    df = df.dropna(subset=['resolved']).copy()
    df['resolved'] = df['resolved'].astype(bool)
    print(f"After drop nulls: {df.shape}")
    print(f"resolved: True={df['resolved'].sum()}, False={(~df['resolved']).sum()}")
    print(f"exit_status: {df['exit_status'].value_counts().to_dict()}")

    # GPT-specific sub-classification (no patch field, use exit_status):
    #   SUCCESS: resolved=True
    #   WANDERING: resolved=False AND exit_status='LimitsExceeded' (budget exhaustion = WANDERING analog)
    #   LOCKED: resolved=False AND exit_status='Submitted' (failed clean submit — different failure mode)
    def gpt_sub_classify(resolved, exit_status):
        if resolved:
            return "success"
        if "limit" in str(exit_status).lower():
            return "wandering"
        return "locked"

    metrics = []
    for i, row in df.iterrows():
        sc = gpt_sub_classify(row['resolved'], row['exit_status'])
        tt = extract_tool_calls_unified(row['messages'])
        last10 = tt[-10:] if len(tt) >= 10 else tt
        all_tools = [t for sub_l in last10 for t in sub_l]
        metrics.append({
            "iid": row["instance_id"],
            "sub_class": sc,
            "model": "gpt-5-router",
            "n_turns": int(row['n_turns']),
            "n_turns_extracted": len(tt),
            "tool_entropy_last10": shannon_entropy(all_tools),
            "n_tools_last10": len(all_tools),
        })
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(df)}")

    mdf = pd.DataFrame(metrics)
    print(f"\nSub-class distribution (all): {mdf['sub_class'].value_counts().to_dict()}")
    mdf = mdf[mdf['n_turns_extracted'] >= 5].copy()
    print(f"After n_turns_extracted>=5: {len(mdf)} ({mdf['sub_class'].value_counts().to_dict()})")
    print(f"Sample extraction stats: n_tools_last10 mean={mdf['n_tools_last10'].mean():.1f}, median={mdf['n_tools_last10'].median():.0f}")
    return mdf


def analyze(mdf, model_label):
    print(f"\n=== {model_label} v5 results ===")
    if mdf is None or mdf.empty:
        print("  (no data)")
        return None
    s = mdf[mdf['sub_class'] == 'success']['tool_entropy_last10'].dropna()
    w = mdf[mdf['sub_class'] == 'wandering']['tool_entropy_last10'].dropna()
    l = mdf[mdf['sub_class'] == 'locked']['tool_entropy_last10'].dropna()
    if len(s) > 0 and len(w) > 0:
        u, p = mannwhitneyu(w, s, alternative="two-sided")
    else:
        u, p = -1, 1.0
    print(f"  N: success={len(s)}, locked={len(l)}, wandering={len(w)}")
    print(f"  tool_entropy_last10 median:")
    print(f"    SUCCESS:   {s.median() if len(s) else float('nan'):.3f}")
    print(f"    LOCKED:    {l.median() if len(l) else float('nan'):.3f}")
    print(f"    WANDERING: {w.median() if len(w) else float('nan'):.3f}")
    print(f"  W vs S Mann-Whitney p={p:.4e}")
    if len(s) > 0 and len(w) > 0:
        ratio = w.median() / s.median() if s.median() > 0 else float('nan')
        print(f"  WANDERING/SUCCESS ratio: {ratio:.3f}")
    return {
        "model": model_label,
        "n_success": int(len(s)), "n_locked": int(len(l)), "n_wandering": int(len(w)),
        "median_entropy": {"success": float(s.median()) if len(s) else None,
                           "locked": float(l.median()) if len(l) else None,
                           "wandering": float(w.median()) if len(w) else None},
        "mean_entropy": {"success": float(s.mean()) if len(s) else None,
                         "locked": float(l.mean()) if len(l) else None,
                         "wandering": float(w.mean()) if len(w) else None},
        "p_w_vs_s": float(p),
        "ratio_w_over_s": float(w.median() / s.median()) if (len(s) and s.median() > 0) else None,
    }


def main():
    claude_mdf = process_claude()
    gpt_mdf = process_gpt()

    res_claude = analyze(claude_mdf, "Claude 3.7 Sonnet")
    res_gpt = analyze(gpt_mdf, "GPT-5 router")

    print("\n=== 4-MODEL UNIFIED SUMMARY ===")
    table = [
        ("Qwen3.6-27B", 0.469, 1.157, "4.0e-5", 0.41),
        ("Llama-70b", 1.000, 2.522, "<1e-15", 0.41),
    ]
    if res_claude:
        table.append((f"Claude 3.7 Sonnet",
                      res_claude["median_entropy"]["wandering"],
                      res_claude["median_entropy"]["success"],
                      f"{res_claude['p_w_vs_s']:.2e}",
                      res_claude["ratio_w_over_s"]))
    if res_gpt:
        table.append((f"GPT-5 router",
                      res_gpt["median_entropy"]["wandering"],
                      res_gpt["median_entropy"]["success"],
                      f"{res_gpt['p_w_vs_s']:.2e}",
                      res_gpt["ratio_w_over_s"]))
    print(f"  {'Model':>20}  {'W med':>8}  {'S med':>8}  {'p(W vs S)':>10}  {'W/S ratio':>10}")
    for m, wm, sm, pv, ratio in table:
        wm_str = f"{wm:.3f}" if wm is not None else " -- "
        sm_str = f"{sm:.3f}" if sm is not None else " -- "
        rt_str = f"{ratio:.3f}" if ratio is not None else " -- "
        print(f"  {m:>20}  {wm_str:>8}  {sm_str:>8}  {pv:>10}  {rt_str:>10}")

    # Save
    out = {"qwen_27b": {"w_med": 0.469, "s_med": 1.157, "p": 4e-5, "ratio": 0.41},
           "llama_70b": {"w_med": 1.000, "s_med": 2.522, "p": "<1e-15", "ratio": 0.41},
           "claude_3_7_sonnet": res_claude,
           "gpt_5_router": res_gpt}
    (OUT_DIR / "v5_cross_model_4labs.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {OUT_DIR}/v5_cross_model_4labs.json")
    claude_mdf.to_csv(OUT_DIR / "v5_claude_metrics.csv", index=False)
    gpt_mdf.to_csv(OUT_DIR / "v5_gpt_metrics.csv", index=False)


if __name__ == "__main__":
    main()
