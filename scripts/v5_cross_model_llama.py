"""
v5 cross-model validation on Llama 70b SWE-bench trajectories.

Tests if tool-entropy signal that discriminates WANDERING from SUCCESS in
Qwen3.6-27B (p=4×10⁻⁵) generalizes to a different model architecture.

Dataset: nebius/SWE-agent-trajectories shard 0
Model: swe-agent-llama-70b (n=6018)

Behavioral sub-classification (no probes available):
  SUCCESS: target=True
  WANDERING: target=False AND exit_status='submitted (exit_context)' AND patch non-empty
             (ran out of context having produced a patch — closest analog to our WANDERING)
  LOCKED: target=False AND (no patch OR early_exit / exit_context without submission)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import mannwhitneyu

OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")


def shannon_entropy(items):
    if not items:
        return 0.0
    cnt = Counter(items)
    total = sum(cnt.values())
    H = 0.0
    for c in cnt.values():
        p = c / total
        if p > 0:
            H -= p * np.log2(p)
    return H


import re
CODE_BLOCK_RE = re.compile(r"```(?:bash|shell|sh)?\s*\n(.*?)```", re.DOTALL)


def extract_tool_calls(trajectory):
    """SWE-agent format: role='ai' with text containing ```...``` bash blocks.
       Extracts the first command word of each code block as the tool name."""
    turn_tools = []
    if trajectory is None:
        return []
    for entry in trajectory:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        if role != "ai":
            continue
        text = entry.get("text") or ""
        if not isinstance(text, str):
            continue
        blocks = CODE_BLOCK_RE.findall(text)
        names = []
        for blk in blocks:
            first_line = blk.strip().split("\n")[0].strip()
            first_word = first_line.split()[0] if first_line.split() else ""
            if first_word and len(first_word) < 40:
                # Normalize SWE-agent special commands (edit, view, open, search_file, etc.)
                cmd = first_word.strip(":").lower()
                names.append(cmd)
        if names:
            turn_tools.append(names)
    return turn_tools


def sub_classify(target, exit_status, patch):
    """Behavioral analog of our WANDERING/LOCKED/SUCCESS taxonomy."""
    if target:
        return "success"
    has_patch = bool(patch and len(patch.strip()) > 0)
    if exit_status == "submitted (exit_context)" and has_patch:
        # Ran out of context with patch produced — WANDERING analog
        return "wandering"
    if exit_status in ("early_exit", "exit_context", "submitted_no_patch") or not has_patch:
        return "locked"
    return "other"  # 'submitted' with target=False = failed-but-clean-submission, ambiguous


def main():
    print("=== Downloading nebius/SWE-agent-trajectories shard 0 ===")
    path = hf_hub_download('nebius/SWE-agent-trajectories', 'data/train-00000-of-00012.parquet', repo_type='dataset')
    df = pd.read_parquet(path)
    print(f"Shape: {df.shape}")

    # Filter to Llama-70b for cleanest single-model comparison
    df = df[df['model_name'] == 'swe-agent-llama-70b'].copy()
    print(f"After filter to llama-70b: {df.shape}")

    # Sub-classify
    print("\n=== Sub-classifying trajectories ===")
    df['sub_class'] = df.apply(lambda r: sub_classify(r['target'], r['exit_status'], r['generated_patch']), axis=1)
    print(df['sub_class'].value_counts())
    print()
    print("Cross: sub_class × exit_status")
    print(df.groupby(['sub_class', 'exit_status']).size().unstack(fill_value=0))

    # Sample to make analysis tractable (still meaningful N per class)
    # Take all wandering + matched samples of success + locked
    n_wander = (df['sub_class'] == 'wandering').sum()
    n_success_sample = min(500, (df['sub_class'] == 'success').sum())
    n_locked_sample = min(500, (df['sub_class'] == 'locked').sum())
    print(f"\nWANDERING n={n_wander}, sampling {n_success_sample} SUCCESS + {n_locked_sample} LOCKED")

    sub = pd.concat([
        df[df['sub_class'] == 'wandering'],
        df[df['sub_class'] == 'success'].sample(n=n_success_sample, random_state=42),
        df[df['sub_class'] == 'locked'].sample(n=n_locked_sample, random_state=42),
    ]).reset_index(drop=True)

    # Compute v5 per trajectory
    print(f"\n=== Computing v5 tool-entropy on {len(sub)} trajectories ===")
    metrics = []
    for i, row in sub.iterrows():
        turn_tools = extract_tool_calls(row['trajectory'])
        n_turns = len(turn_tools)
        last10 = turn_tools[-10:] if n_turns >= 10 else turn_tools
        all_tools_last10 = [t for sub_l in last10 for t in sub_l]
        entropy_last10 = shannon_entropy(all_tools_last10)
        diversity_last10 = len(set(all_tools_last10)) / max(1, len(last10))
        repetition_last10 = (max(Counter(all_tools_last10).values()) / len(all_tools_last10)) if all_tools_last10 else 0.0
        metrics.append({
            "iid": row["instance_id"],
            "sub_class": row["sub_class"],
            "exit_status": row["exit_status"],
            "n_turns": n_turns,
            "n_tools_last10": len(all_tools_last10),
            "tool_entropy_last10": entropy_last10,
            "tool_diversity_last10": diversity_last10,
            "tool_repetition_last10": repetition_last10,
        })
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(sub)}")

    mdf = pd.DataFrame(metrics)
    print(f"\nMetrics computed: {len(mdf)}")

    # Filter to trajectories with enough turns (need at least 10 turns for last10 to be meaningful)
    mdf_full = mdf[mdf['n_turns'] >= 5].copy()
    print(f"After n_turns >= 5 filter: {len(mdf_full)}")
    print(mdf_full['sub_class'].value_counts())

    # Statistical comparison
    print("\n=== v5 cross-model results: Llama-70b ===")
    for metric in ["tool_entropy_last10", "tool_diversity_last10", "tool_repetition_last10"]:
        s = mdf_full[mdf_full['sub_class'] == 'success'][metric].dropna()
        w = mdf_full[mdf_full['sub_class'] == 'wandering'][metric].dropna()
        l = mdf_full[mdf_full['sub_class'] == 'locked'][metric].dropna()
        if len(s) > 0 and len(w) > 0:
            u_ws, p_ws = mannwhitneyu(w, s, alternative="two-sided")
        else:
            u_ws, p_ws = -1, 1.0
        if len(w) > 0 and len(l) > 0:
            u_wl, p_wl = mannwhitneyu(w, l, alternative="two-sided")
        else:
            u_wl, p_wl = -1, 1.0
        print(f"\n  {metric}")
        print(f"    SUCCESS   (n={len(s)}): mean={s.mean():.3f}, median={s.median():.3f}")
        print(f"    LOCKED    (n={len(l)}): mean={l.mean():.3f}, median={l.median():.3f}")
        print(f"    WANDERING (n={len(w)}): mean={w.mean():.3f}, median={w.median():.3f}")
        print(f"    W vs S: p={p_ws:.5f}")
        print(f"    W vs L: p={p_wl:.5f}")

    # Direction check
    print("\n=== DIRECTION CHECK (does Llama match Qwen?) ===")
    s = mdf_full[mdf_full['sub_class'] == 'success']['tool_entropy_last10'].dropna()
    w = mdf_full[mdf_full['sub_class'] == 'wandering']['tool_entropy_last10'].dropna()
    qwen_direction = "WANDERING < SUCCESS (entropy collapse)"
    llama_direction = "WANDERING < SUCCESS" if w.median() < s.median() else "WANDERING > SUCCESS"
    print(f"  Qwen3.6-27B: {qwen_direction} (p=4e-5)")
    print(f"  Llama-70b:   {llama_direction} (W median={w.median():.3f}, S median={s.median():.3f})")
    if w.median() < s.median():
        print("  ✓ SIGNAL DIRECTION GENERALIZES")
    else:
        print("  ✗ Signal direction does NOT match Qwen — model-specific")

    # Save
    mdf_full.to_csv(OUT_DIR / "v5_cross_model_llama.csv", index=False)
    summary = {
        "model": "swe-agent-llama-70b",
        "dataset": "nebius/SWE-agent-trajectories shard 0",
        "n_total": len(mdf_full),
        "n_success": int((mdf_full['sub_class'] == 'success').sum()),
        "n_locked": int((mdf_full['sub_class'] == 'locked').sum()),
        "n_wandering": int((mdf_full['sub_class'] == 'wandering').sum()),
        "tool_entropy_last10": {
            "success_median": float(s.median()) if len(s) else None,
            "success_mean": float(s.mean()) if len(s) else None,
            "wandering_median": float(w.median()) if len(w) else None,
            "wandering_mean": float(w.mean()) if len(w) else None,
        },
    }
    (OUT_DIR / "v5_cross_model_llama_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {OUT_DIR}/v5_cross_model_llama.csv + summary.json")


if __name__ == "__main__":
    main()
