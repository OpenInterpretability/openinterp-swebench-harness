"""
Exp B prereq — Determinism check for sandbox replay feasibility.

For each Phase 6 WANDERING trajectory:
  1. Categorize tool_calls by determinism risk:
       - READ_ONLY: cd, ls, find, cat, grep, view, etc. (state-preserving)
       - EDIT: str_replace_editor with create/str_replace, sed inline, etc.
       - WRITE: bash with > / >> redirects, touch, mkdir
       - DANGEROUS: pip install, network requests, date/random, pytest with timing
  2. Compute % of tool_calls that are DANGEROUS (non-replayable)
  3. If DANGEROUS < 10% across most trajectories → replay feasible
  4. If DANGEROUS > 30% → need Plan B (always-on hook from turn 0)

This is analytical only — no sandbox execution. Decides Exp B engineering path.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")

infl = json.load(open(OUT_DIR / "inflection_results.json"))
sub_class = {}
for t in infl['per_trajectory']:
    if t['label']==1: sub_class[t['iid']] = 'success'
    elif t.get('lock_fail_0.40') is not None: sub_class[t['iid']] = 'locked'
    else: sub_class[t['iid']] = 'wandering'

READ_ONLY_BASH = {'cd', 'ls', 'find', 'cat', 'grep', 'head', 'tail', 'less', 'more',
                  'file', 'wc', 'diff', 'pwd', 'echo', 'tree', 'stat', 'which', 'whereis'}
WRITE_BASH = {'touch', 'mkdir', 'rmdir', 'rm', 'mv', 'cp', 'chmod', 'chown'}
DANGEROUS_BASH = {'pip', 'pytest', 'python', 'python3', 'sh', 'bash', 'make', 'gcc', 'curl', 'wget',
                  'date', 'time', 'sleep', 'git', 'apt', 'apt-get', 'npm', 'yarn'}
# Note: 'git' is potentially OK if read-only (status/log/diff) but flagged as DANGEROUS conservatively


def categorize_bash_cmd(cmd: str) -> str:
    """Return READ_ONLY / EDIT_INLINE / WRITE / DANGEROUS for a bash command."""
    if not cmd or not cmd.strip():
        return 'EMPTY'
    parts = cmd.strip().split('&&')
    categories = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # First word = the actual command
        words = part.split()
        if not words:
            continue
        first = words[0]
        # Inline writes: > or >> redirect
        if '>' in part or '>>' in part:
            categories.append('WRITE')
            continue
        # sed -i or awk inline edits
        if first == 'sed' and ('-i' in part or '--in-place' in part):
            categories.append('EDIT_INLINE')
            continue
        if first in READ_ONLY_BASH:
            categories.append('READ_ONLY')
        elif first in WRITE_BASH:
            categories.append('WRITE')
        elif first in DANGEROUS_BASH:
            categories.append('DANGEROUS')
        else:
            categories.append(f'OTHER:{first}')
    # Most severe category wins
    if any(c == 'DANGEROUS' or c.startswith('OTHER:') for c in categories):
        return next((c for c in categories if c == 'DANGEROUS' or c.startswith('OTHER:')))
    if 'WRITE' in categories or 'EDIT_INLINE' in categories:
        return 'WRITE'
    return 'READ_ONLY'


def categorize_str_replace(call: dict) -> str:
    args = call.get('arguments') or call.get('args') or {}
    cmd = args.get('command', '') if isinstance(args, dict) else ''
    if cmd in ('view',):
        return 'READ_ONLY'
    if cmd in ('create', 'str_replace', 'insert', 'undo_edit'):
        return 'EDIT'
    return f'STR_REPLACE_OTHER:{cmd}'


def categorize_trajectory(trace: dict) -> dict:
    counts = Counter()
    by_turn = []
    for ti, turn in enumerate(trace['turns']):
        for call in turn.get('tool_calls', []) or []:
            if not isinstance(call, dict):
                counts['NON_DICT'] += 1
                continue
            name = call.get('name', 'unknown')
            if name == 'bash':
                args = call.get('arguments') or call.get('args') or {}
                cmd = args.get('command', '') if isinstance(args, dict) else ''
                cat = categorize_bash_cmd(cmd)
            elif name == 'str_replace_editor':
                cat = categorize_str_replace(call)
            elif name == 'finish':
                cat = 'FINISH'
            else:
                cat = f'UNKNOWN_TOOL:{name}'
            counts[cat] += 1
            by_turn.append((ti, name, cat))
    return {'counts': dict(counts), 'by_turn': by_turn}


def main():
    wandering_iids = [iid for iid, c in sub_class.items() if c == 'wandering']
    print(f"Analyzing {len(wandering_iids)} WANDERING trajectories for replay determinism...\n")

    all_counts = Counter()
    per_traj_results = []
    for iid in wandering_iids:
        trace_path = PHASE6 / "traces" / (iid + ".json")
        if not trace_path.exists():
            continue
        try:
            trace = json.load(open(trace_path))
        except Exception as e:
            print(f"  ERR loading {iid[:40]}: {e}")
            continue
        r = categorize_trajectory(trace)
        r['iid'] = iid
        per_traj_results.append(r)
        for cat, n in r['counts'].items():
            all_counts[cat] += n

    print("=== Aggregate tool-call categories across all WANDERING ===")
    total = sum(all_counts.values())
    for cat, n in all_counts.most_common():
        pct = 100 * n / max(1, total)
        print(f"  {cat:>30}: {n:>4} ({pct:>4.1f}%)")

    # Determinism score per trajectory
    print(f"\n=== Per-trajectory determinism breakdown ===")
    print(f"{'iid':>40} {'tot':>4} {'READ':>4} {'EDIT':>4} {'WRITE':>5} {'DNG':>4} {'pct_replayable':>14}")
    feasible_count = 0
    for r in per_traj_results:
        c = r['counts']
        tot = sum(c.values())
        read = c.get('READ_ONLY', 0)
        edit = c.get('EDIT', 0) + c.get('EDIT_INLINE', 0)
        write = c.get('WRITE', 0)
        # DANGEROUS = anything starting with DANGEROUS or OTHER: or UNKNOWN:
        dng = sum(v for k, v in c.items() if k == 'DANGEROUS' or k.startswith('OTHER:') or k.startswith('UNKNOWN'))
        replayable = read + edit + write
        pct_replayable = 100 * replayable / max(1, tot)
        if pct_replayable >= 80:
            feasible_count += 1
        print(f"  {r['iid'][:40]:>40} {tot:>4} {read:>4} {edit:>4} {write:>5} {dng:>4} {pct_replayable:>13.0f}%")

    print(f"\n=== Replay feasibility summary ===")
    print(f"  Trajectories with ≥80% replayable tool_calls: {feasible_count}/{len(per_traj_results)}")
    fr = feasible_count / max(1, len(per_traj_results))
    if fr >= 0.80:
        print(f"  >>> {100*fr:.0f}% feasible — REPLAY VIABLE for Exp B surgical patching")
        print(f"      Recommend: Plan A (replay sandbox to turn T, patch L55, resume)")
    elif fr >= 0.50:
        print(f"  >>> {100*fr:.0f}% feasible — PARTIAL replay")
        print(f"      Recommend: try Plan A on feasible trajectories, Plan B on others")
    else:
        print(f"  >>> Only {100*fr:.0f}% feasible — REPLAY RISKY")
        print(f"      Recommend: Plan B (always-on L55 hook from turn 0, fresh agent run)")

    # Top DANGEROUS commands (so we can decide what to do about them)
    print(f"\n=== Top DANGEROUS / OTHER commands (need handling) ===")
    danger_subcat = Counter()
    for r in per_traj_results:
        for cat, n in r['counts'].items():
            if cat == 'DANGEROUS' or cat.startswith('OTHER:') or cat.startswith('UNKNOWN'):
                danger_subcat[cat] += n
    for cat, n in danger_subcat.most_common(15):
        print(f"  {cat:>30}: {n}")

    out = {
        'aggregate_counts': dict(all_counts),
        'feasible_count': feasible_count,
        'total_trajectories': len(per_traj_results),
        'feasibility_rate': fr,
        'per_traj': [{'iid': r['iid'], 'counts': r['counts']} for r in per_traj_results],
    }
    (OUT_DIR / 'exp_b_determinism_check.json').write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_DIR}/exp_b_determinism_check.json")


if __name__ == "__main__":
    main()
