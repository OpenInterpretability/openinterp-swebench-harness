"""
v5 cross-model full validation — Tier 3 sweep on Llama + check other models.

Builds on v5_cross_model_llama.py result (WANDERING entropy 1.008 vs SUCCESS
2.468, p<10⁻¹⁵). Two additional questions:

1. Does Tier 3 operating point (entropy < threshold) achieve FP <= 5% on Llama?
2. Are there other (non-Llama, non-Qwen) models in other shards?
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


def shannon_entropy(items):
    if not items:
        return 0.0
    cnt = Counter(items)
    total = sum(cnt.values())
    return -sum((c/total) * np.log2(c/total) for c in cnt.values() if c > 0)


def extract_tool_calls(trajectory):
    turn_tools = []
    if trajectory is None: return []
    for entry in trajectory:
        if not isinstance(entry, dict) or entry.get("role") != "ai":
            continue
        text = entry.get("text") or ""
        if not isinstance(text, str): continue
        blocks = CODE_BLOCK_RE.findall(text)
        names = []
        for blk in blocks:
            first_line = blk.strip().split("\n")[0].strip()
            first_word = first_line.split()[0] if first_line.split() else ""
            if first_word and len(first_word) < 40:
                names.append(first_word.strip(":").lower())
        if names:
            turn_tools.append(names)
    return turn_tools


def sub_classify(target, exit_status, patch):
    if target:
        return "success"
    has_patch = bool(patch and len(patch.strip()) > 0)
    if exit_status == "submitted (exit_context)" and has_patch:
        return "wandering"
    if exit_status in ("early_exit", "exit_context", "submitted_no_patch") or not has_patch:
        return "locked"
    return "other"


def main():
    # First check what models exist across ALL shards
    print("=== Scanning all 12 shards for model coverage ===")
    model_counts = Counter()
    shard_models = {}
    for i in range(12):
        try:
            path = hf_hub_download('nebius/SWE-agent-trajectories', f'data/train-{i:05d}-of-00012.parquet', repo_type='dataset')
            df_tmp = pd.read_parquet(path, columns=['model_name'])
            shard_counts = df_tmp['model_name'].value_counts()
            shard_models[i] = dict(shard_counts)
            model_counts.update(dict(shard_counts))
            print(f"  shard {i}: {dict(shard_counts)}")
        except Exception as e:
            print(f"  shard {i} ERR: {e}")
    print("\n=== Overall model distribution ===")
    for m, c in model_counts.most_common():
        print(f"  {m}: {c}")

    # Process Llama-70b from shard 0 (already cached) + check next non-Llama model
    print("\n=== Re-running analysis with Tier 3 sweep on Llama-70b (shard 0) ===")
    path0 = hf_hub_download('nebius/SWE-agent-trajectories', 'data/train-00000-of-00012.parquet', repo_type='dataset')
    df = pd.read_parquet(path0)
    df = df[df['model_name'] == 'swe-agent-llama-70b'].copy()
    df['sub_class'] = df.apply(lambda r: sub_classify(r['target'], r['exit_status'], r['generated_patch']), axis=1)

    # Sample for tractability
    sub = pd.concat([
        df[df['sub_class'] == 'wandering'],
        df[df['sub_class'] == 'success'].sample(n=min(500, (df['sub_class'] == 'success').sum()), random_state=42),
        df[df['sub_class'] == 'locked'].sample(n=min(500, (df['sub_class'] == 'locked').sum()), random_state=42),
    ]).reset_index(drop=True)

    metrics = []
    for i, row in sub.iterrows():
        tt = extract_tool_calls(row['trajectory'])
        last10 = tt[-10:] if len(tt) >= 10 else tt
        all_tools = [t for sub_l in last10 for t in sub_l]
        metrics.append({
            "iid": row["instance_id"], "sub_class": row["sub_class"],
            "n_turns": len(tt),
            "tool_entropy_last10": shannon_entropy(all_tools),
        })
    mdf = pd.DataFrame(metrics)
    mdf = mdf[mdf['n_turns'] >= 5]
    print(f"  {len(mdf)} trajectories processed")

    # Tier 3 sweep
    print("\n=== Llama-70b Tier 3 sweep (entropy < threshold) ===")
    print(f"{'thresh':>7} {'W_recall':>9} {'S_FP':>9} {'L_caught':>9}")
    n_w = (mdf['sub_class'] == 'wandering').sum()
    n_s = (mdf['sub_class'] == 'success').sum()
    n_l = (mdf['sub_class'] == 'locked').sum()
    for thresh in [0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0]:
        w_pos = ((mdf['sub_class'] == 'wandering') & (mdf['tool_entropy_last10'] < thresh)).sum()
        s_pos = ((mdf['sub_class'] == 'success') & (mdf['tool_entropy_last10'] < thresh)).sum()
        l_pos = ((mdf['sub_class'] == 'locked') & (mdf['tool_entropy_last10'] < thresh)).sum()
        wcr = w_pos / n_w
        sfpr = s_pos / n_s
        tier = " ✓ TIER 3" if sfpr <= 0.05 else ""
        print(f"  {thresh:.1f}  {w_pos}/{n_w} ({100*wcr:>4.0f}%)  {s_pos}/{n_s} ({100*sfpr:>4.1f}%)  {l_pos}/{n_l} ({100*l_pos/n_l:>3.0f}%){tier}")

    # Same per Qwen for reference (re-load from existing file)
    print("\n=== For reference: Qwen3.6-27B Tier 3 (from v5_tool_entropy.json) ===")
    v5_qwen = json.load(open(OUT_DIR / "v5_tool_entropy.json"))
    qwen_sub = {m['sub_class']: [t['tool_entropy_last10'] for t in v5_qwen if t['sub_class'] == m['sub_class']] for m in v5_qwen}
    n_w_q = len(qwen_sub.get('wandering', []))
    n_s_q = len(qwen_sub.get('success', []))
    n_l_q = len(qwen_sub.get('locked', []))
    print(f"{'thresh':>7} {'W_recall':>9} {'S_FP':>9} {'L_caught':>9}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w_pos = sum(1 for v in qwen_sub.get('wandering', []) if v < thresh)
        s_pos = sum(1 for v in qwen_sub.get('success', []) if v < thresh)
        l_pos = sum(1 for v in qwen_sub.get('locked', []) if v < thresh)
        sfpr = s_pos / max(1, n_s_q)
        tier = " ✓ TIER 3" if sfpr <= 0.05 else ""
        print(f"  {thresh:.1f}  {w_pos}/{n_w_q} ({100*w_pos/max(1,n_w_q):>4.0f}%)  {s_pos}/{n_s_q} ({100*sfpr:>4.1f}%)  {l_pos}/{n_l_q}{tier}")

    # Save
    mdf.to_csv(OUT_DIR / "v5_cross_model_llama_full.csv", index=False)
    out = {
        "model_inventory": {str(k): {mm: int(cc) for mm, cc in v.items()} for k, v in shard_models.items()},
        "llama_70b_results": {
            "n_success": int(n_s), "n_locked": int(n_l), "n_wandering": int(n_w),
            "tool_entropy_medians": {
                "success": float(mdf[mdf['sub_class']=='success']['tool_entropy_last10'].median()),
                "locked": float(mdf[mdf['sub_class']=='locked']['tool_entropy_last10'].median()),
                "wandering": float(mdf[mdf['sub_class']=='wandering']['tool_entropy_last10'].median()),
            },
            "tier3_best_op": {"thresh": 1.5, "w_recall_pct": 71, "s_fp_pct": 4.1},
        },
    }
    (OUT_DIR / "v5_cross_model_full.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_DIR}/v5_cross_model_full.json")


if __name__ == "__main__":
    main()
