"""
v5 cross-task validation on METR MALT-public.

Hypothesis test:
  H_universal: tool-entropy collapse signature is TASK-INVARIANT (not just
               code-repair / SWE-bench specific)
  H_code-specific: signal only works in tool-rich code tasks
  H_multi-tool-dependent: signal requires ≥3 distinct tools in normal usage

MALT dataset structure:
  - 171 rows per shard × 42 shards (~7,182 trajectories total)
  - Models: claude-3-5-sonnet, o1-preview, gpt-4o, claude-3-5-sonnet-v2
  - Tools (XML tags): <bash>, <python>, <submit>
  - Labels: 'normal' (SUCCESS-equiv), 'gives_up' (WANDERING analog)
  - 15+ task families: password_check, wordle, hackthebox, ai_rd_*, gaia,
    advent_of_code, local_research, reverse_hash, make_web_server, etc.

Per row:
  - samples = list of N turn-snapshots
  - Each sample.output = agent response (contains XML tool tags)
  - Aggregate tool calls across all samples = full trajectory tools

Output:
  - Per task family: WANDERING vs SUCCESS entropy comparison
  - Overall: pooled comparison
  - Per model breakdown
  - Tool-diversity check: is the signal stronger in multi-tool tasks?
"""
from __future__ import annotations

import os
import re
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import mannwhitneyu

OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")
OUT_DIR.mkdir(exist_ok=True)

TOKEN = (Path.home() / ".cache" / "huggingface" / "token").read_text().strip()
os.environ["HF_TOKEN"] = TOKEN

# MALT tool format: <bash>cmd</bash>, <python>code</python>, <submit>answer</submit>
# Close tags often missing (agent output truncated by scaffold). Use permissive:
# capture up to close tag OR next open tag OR end-of-string.
TOOL_TAG_RE = re.compile(r"<(bash|python|submit)>", re.IGNORECASE)
# Lazy match content up to either close tag, newline+open-tag, or 200 chars max
BASH_INNER_RE = re.compile(r"<bash>([^<]{0,500})", re.IGNORECASE)
PYTHON_OPEN_RE = re.compile(r"<python>", re.IGNORECASE)
SUBMIT_OPEN_RE = re.compile(r"<submit>", re.IGNORECASE)


def shannon_entropy(items):
    if not items:
        return 0.0
    cnt = Counter(items)
    total = sum(cnt.values())
    return -sum((c/total) * np.log2(c/total) for c in cnt.values() if c > 0)


def extract_tools_from_text(text):
    """Returns list of fine-grained tool names from a single message text.
    For <bash>X Y...</bash>: emit 'bash:X' (first word as subcommand).
    For <python>...</python>: emit 'python_exec' (single type).
    For <submit>: emit 'submit'.
    """
    if not isinstance(text, str):
        return []
    tools = []
    for m in BASH_INNER_RE.findall(text):
        first_line = m.strip().split("\n")[0].strip()
        first_word = first_line.split()[0] if first_line.split() else "empty"
        if "/" in first_word:
            first_word = first_word.rsplit("/", 1)[-1]
        first_word = first_word.rstrip(";|").lstrip("(")
        if first_word and len(first_word) < 40:
            tools.append(f"bash:{first_word.lower()}")
    # python and submit are single-token tools
    n_python = len(PYTHON_OPEN_RE.findall(text))
    n_submit = len(SUBMIT_OPEN_RE.findall(text))
    tools.extend(["python_exec"] * n_python)
    tools.extend(["submit"] * n_submit)
    return tools


def _flatten_to_dicts(obj):
    """Recursively flatten arbitrary nested list/ndarray to a flat list of dicts."""
    out = []
    if isinstance(obj, dict):
        out.append(obj)
    elif isinstance(obj, (list, np.ndarray)) and hasattr(obj, '__iter__'):
        for item in obj:
            out.extend(_flatten_to_dicts(item))
    return out


def extract_trajectory_tools(samples):
    """For a MALT row's samples, return list of per-turn tool calls.
       Each sample.output is one turn's agent response (may be nested ndarray).
       Returns list-of-lists (one list per turn)."""
    turn_tools = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        out = s.get("output")
        if out is None:
            continue
        msgs = _flatten_to_dicts(out)
        for m in msgs:
            if m.get("role") in ("assistant", "ai"):
                content = m.get("content", "")
                if isinstance(content, str):
                    tools = extract_tools_from_text(content)
                    if tools:
                        turn_tools.append(tools)
    return turn_tools


def process_shard(shard_idx):
    p = hf_hub_download('metr-evals/malt-public', f'data/public-{shard_idx:05d}-of-00042.parquet',
                        repo_type='dataset', token=TOKEN)
    df = pd.read_parquet(p)
    return df


def main(n_shards=5):
    print(f"=== Loading {n_shards} MALT shards ===")
    all_dfs = []
    for i in range(n_shards):
        try:
            df = process_shard(i)
            all_dfs.append(df)
            print(f"  shard {i}: {len(df)} rows")
        except Exception as e:
            print(f"  shard {i} ERR: {e}")
    if not all_dfs:
        print("No data loaded")
        return
    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df_all)}")

    # Pre-extract metadata + tools
    print("\n=== Extracting metadata + tool calls per trajectory ===")
    metrics = []
    for i, row in df_all.iterrows():
        md = row['metadata']
        labels = list(md.get('labels', []))
        if not labels:
            continue
        is_giveup = 'gives_up' in labels
        is_normal = 'normal' in labels
        if not (is_giveup or is_normal):
            continue
        task_id = md.get('task_id', '')
        task_family = task_id.split('/')[0] if '/' in task_id else task_id
        model = md.get('model', 'unknown')
        # Extract trajectory tools
        tt = extract_trajectory_tools(row['samples'])
        n_turns = len(tt)
        if n_turns < 3:
            continue
        last10 = tt[-10:] if n_turns >= 10 else tt
        all_tools_last10 = [t for sub in last10 for t in sub]
        all_tools_full = [t for sub in tt for t in sub]
        metrics.append({
            "task_family": task_family,
            "task_id": task_id,
            "model": model,
            "label": "gives_up" if is_giveup else "normal",
            "n_turns": n_turns,
            "n_tools_total": len(all_tools_full),
            "n_tools_last10": len(all_tools_last10),
            "tool_entropy_last10": shannon_entropy(all_tools_last10),
            "tool_diversity_last10": len(set(all_tools_last10)) / max(1, len(last10)),
            "tool_entropy_full": shannon_entropy(all_tools_full),
            "n_unique_tools_full": len(set(all_tools_full)),
        })
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(df_all)}")

    mdf = pd.DataFrame(metrics)
    print(f"\nExtracted {len(mdf)} trajectories")
    print(f"Label distribution: {mdf['label'].value_counts().to_dict()}")
    print(f"Model distribution: {mdf['model'].value_counts().to_dict()}")
    print(f"Task family distribution (top 10): {mdf['task_family'].value_counts().head(10).to_dict()}")
    print(f"n_unique_tools_full distribution: min={mdf['n_unique_tools_full'].min()}, max={mdf['n_unique_tools_full'].max()}, median={mdf['n_unique_tools_full'].median()}")

    # OVERALL: WANDERING (gives_up) vs SUCCESS (normal)
    print("\n=== OVERALL: WANDERING (gives_up) vs SUCCESS (normal) ===")
    w = mdf[mdf['label'] == 'gives_up']['tool_entropy_last10'].dropna()
    s = mdf[mdf['label'] == 'normal']['tool_entropy_last10'].dropna()
    print(f"  WANDERING n={len(w)}: median={w.median():.3f}, mean={w.mean():.3f}")
    print(f"  SUCCESS   n={len(s)}: median={s.median():.3f}, mean={s.mean():.3f}")
    if len(w) > 0 and len(s) > 0:
        u, p = mannwhitneyu(w, s, alternative="two-sided")
        ratio = w.median() / s.median() if s.median() > 0 else float('nan')
        print(f"  Mann-Whitney p={p:.5e}")
        print(f"  W/S ratio: {ratio:.3f}")

    # PER TASK FAMILY (only families with both labels)
    print("\n=== PER TASK FAMILY (only families with both labels + n_giveup >= 2) ===")
    print(f"  {'family':>28}  {'n_W':>4}  {'n_S':>4}  {'W med':>7}  {'S med':>7}  {'p':>9}  {'W/S':>6}  {'direction':>9}")
    per_family = {}
    for fam in mdf['task_family'].unique():
        sub = mdf[mdf['task_family'] == fam]
        w = sub[sub['label'] == 'gives_up']['tool_entropy_last10'].dropna()
        s = sub[sub['label'] == 'normal']['tool_entropy_last10'].dropna()
        if len(w) < 2 or len(s) < 2:
            continue
        try:
            u, p = mannwhitneyu(w, s, alternative="two-sided")
        except Exception:
            u, p = -1, 1.0
        ratio = w.median() / s.median() if s.median() > 0 else float('nan')
        direction = "✓ W<S" if w.median() < s.median() else "✗ W>S"
        print(f"  {fam:>28}  {len(w):>4}  {len(s):>4}  {w.median():>7.3f}  {s.median():>7.3f}  {p:>9.4f}  {ratio:>6.3f}  {direction:>9}")
        per_family[fam] = {
            "n_wandering": int(len(w)),
            "n_success": int(len(s)),
            "w_median": float(w.median()),
            "s_median": float(s.median()),
            "p": float(p),
            "ratio": float(ratio) if not np.isnan(ratio) else None,
            "direction_matches": bool(w.median() < s.median()),
        }

    # PER MODEL
    print("\n=== PER MODEL (only with both labels + n_giveup >= 3) ===")
    print(f"  {'model':>40}  {'n_W':>4}  {'n_S':>4}  {'W med':>7}  {'S med':>7}  {'p':>9}  {'W/S':>6}")
    per_model = {}
    for mdl in mdf['model'].unique():
        sub = mdf[mdf['model'] == mdl]
        w = sub[sub['label'] == 'gives_up']['tool_entropy_last10'].dropna()
        s = sub[sub['label'] == 'normal']['tool_entropy_last10'].dropna()
        if len(w) < 3 or len(s) < 3:
            continue
        try:
            u, p = mannwhitneyu(w, s, alternative="two-sided")
        except Exception:
            u, p = -1, 1.0
        ratio = w.median() / s.median() if s.median() > 0 else float('nan')
        print(f"  {mdl:>40}  {len(w):>4}  {len(s):>4}  {w.median():>7.3f}  {s.median():>7.3f}  {p:>9.4f}  {ratio:>6.3f}")
        per_model[mdl] = {
            "n_wandering": int(len(w)), "n_success": int(len(s)),
            "w_median": float(w.median()), "s_median": float(s.median()),
            "p": float(p), "ratio": float(ratio) if not np.isnan(ratio) else None,
        }

    # MULTI-TOOL CHECK — stratify by n_unique_tools_full
    print("\n=== MULTI-TOOL DEPENDENCY CHECK ===")
    print("  Hypothesis: signal works only when trajectory uses ≥2 unique tools")
    for min_uniq in [1, 2, 3]:
        sub = mdf[mdf['n_unique_tools_full'] >= min_uniq]
        w = sub[sub['label'] == 'gives_up']['tool_entropy_last10']
        s = sub[sub['label'] == 'normal']['tool_entropy_last10']
        if len(w) >= 3 and len(s) >= 3:
            u, p = mannwhitneyu(w, s, alternative="two-sided")
            ratio = w.median() / s.median() if s.median() > 0 else float('nan')
            print(f"  n_unique_tools_full >= {min_uniq}: n_W={len(w)}, n_S={len(s)}, W={w.median():.3f}, S={s.median():.3f}, p={p:.4e}, ratio={ratio:.3f}")

    # SUMMARY: cross-task universality verdict
    n_families = len(per_family)
    n_matches = sum(1 for v in per_family.values() if v['direction_matches'])
    n_sig = sum(1 for v in per_family.values() if v['direction_matches'] and v['p'] < 0.05)
    print(f"\n=== UNIVERSALITY VERDICT ===")
    print(f"  N task families with both labels (n_W>=2 each): {n_families}")
    print(f"  N families with direction WANDERING<SUCCESS:    {n_matches}/{n_families}")
    print(f"  N families significant (p<0.05) AND right direction: {n_sig}/{n_families}")

    out = {
        "n_trajectories": len(mdf),
        "overall_W_vs_S": {
            "n_w": int(len(mdf[mdf['label']=='gives_up'])),
            "n_s": int(len(mdf[mdf['label']=='normal'])),
        },
        "per_family": per_family,
        "per_model": per_model,
    }
    (OUT_DIR / "v5_cross_task_malt.json").write_text(json.dumps(out, indent=2, default=str))
    mdf.to_csv(OUT_DIR / "v5_cross_task_malt.csv", index=False)
    print(f"\nSaved {OUT_DIR}/v5_cross_task_malt.json + .csv")


if __name__ == "__main__":
    import sys
    n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n_shards=n_shards)
