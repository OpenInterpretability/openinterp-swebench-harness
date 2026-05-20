"""Test-pass eval for Paper 11 Phase 1 — wrapper around the official SWE-bench Pro
evaluation harness (github.com/scaleapi/SWE-bench_Pro-os).

Ground truth for the correction/obfuscation/containment 2x2: does an agent's patch
actually resolve the SWE-bench Pro issue (gold fail_to_pass tests pass, pass_to_pass
stay passing)?

This script does NOT re-implement test execution. It wraps the official harness:
  1. Collects our <instance_id>.patch files from a traces dir.
  2. Builds patches.json  ([{instance_id, patch, prefix}])  + raw_sample.jsonl
     (the SWE-bench Pro dataset rows for those instances).
  3. Invokes swe_bench_pro_eval.py --use_local_docker  (per-instance Docker images
     jefzda/sweap-images:<tag> carry the repo + environment + gold tests).
  4. Parses output_dir/<uid>/<prefix>_output.json -> resolved per instance.
  5. Writes our test_pass_results.json keyed by (condition, instance_id).

Requirements: Docker running; `pip install docker datasets pandas`; a local clone of
scaleapi/SWE-bench_Pro-os. Image pulls are multi-GB per instance — use --dry-run first
to see scope, and --limit to pilot on a few instances.

Usage:
  python run_test_pass_eval.py \\
      --traces_dir "<drive>/adversarial_kappa_t_phase0/traces" \\
      --condition NOINT \\
      --swebp_repo /tmp/swebp-os \\
      --work_dir /tmp/testpass_eval \\
      [--dry-run] [--limit 3] [--num_workers 4]
"""
import argparse, json, os, subprocess, sys
from pathlib import Path


def load_dataset_rows():
    from datasets import load_dataset
    ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
    return {r['instance_id']: dict(r) for r in ds}


def collect_patches(traces_dir, condition, dataset_rows, limit=None):
    """Read <instance_id>.patch files; keep only those present in the dataset.

    An empty (0B) patch is kept — it is a valid datapoint (the agent produced no
    diff, so the instance will resolve=False). Only patches whose instance_id is
    absent from the dataset are dropped.
    """
    traces_dir = Path(traces_dir)
    patches, dropped, n_empty = [], [], 0
    for pf in sorted(traces_dir.glob('*.patch')):
        iid = pf.stem  # filename without .patch
        if iid not in dataset_rows:
            dropped.append((iid, 'not in SWE-bench Pro dataset'))
            continue
        text = pf.read_text()
        if not text.strip():
            n_empty += 1
        patches.append({'instance_id': iid, 'patch': text, 'prefix': condition})
        if limit and len(patches) >= limit:
            break
    return patches, dropped, n_empty


def _as_test_list(v):
    """fail_to_pass / pass_to_pass dataset fields are stringified python lists."""
    if isinstance(v, (list, tuple)):
        return list(v)
    try:
        out = eval(v)  # matches swe_bench_pro_eval.py line 556-557
        return list(out) if isinstance(out, (list, tuple, set)) else []
    except Exception:
        return []


def parse_eval_output(output_dir, condition, dataset_rows):
    """Extract per-instance resolution from the official harness output.

    Authoritative source: output_dir/eval_results.json — {instance_id: bool} written
    by swe_bench_pro_eval.py after running each instance's gold test suite.
    Resolution rule (harness line 558): an instance resolves iff every gold
    FAIL_TO_PASS and PASS_TO_PASS test is in the PASSED set.

    Per-instance <prefix>_output.json carries the raw {"tests": [{name, status}]}
    list — used here to also report how many f2p/p2p tests passed (diagnostic).
    """
    output_dir = Path(output_dir)
    results = {}

    agg = output_dir / 'eval_results.json'
    agg_map = {}
    if agg.exists():
        try:
            agg_map = json.loads(agg.read_text())
        except Exception:
            agg_map = {}

    for uid_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        oj = uid_dir / f'{condition}_output.json'
        if not oj.exists():
            cand = list(uid_dir.glob('*_output.json'))
            oj = cand[0] if cand else None

        passed = set()
        parse_note = None
        if oj is not None and oj.exists():
            try:
                payload = json.loads(oj.read_text())
                passed = {t['name'] for t in payload.get('tests', []) if t.get('status') == 'PASSED'}
            except Exception as e:
                parse_note = f'output.json parse error: {e}'
        else:
            parse_note = 'no <prefix>_output.json — instance likely errored (see *_stderr.log)'

        iid = uid_dir.name
        row = dataset_rows.get(iid, {})
        f2p = _as_test_list(row.get('fail_to_pass'))
        p2p = _as_test_list(row.get('pass_to_pass'))

        resolved = agg_map.get(iid)
        if resolved is None and oj is not None and oj.exists() and (f2p or p2p):
            resolved = (set(f2p) | set(p2p)) <= passed  # recompute from tests list

        results[iid] = {
            'resolved': bool(resolved) if resolved is not None else None,
            'f2p_passed': sum(1 for t in f2p if t in passed),
            'f2p_total': len(f2p),
            'p2p_passed': sum(1 for t in p2p if t in passed),
            'p2p_total': len(p2p),
            'note': parse_note,
        }
    return results


def main():
    ap = argparse.ArgumentParser(description='SWE-bench Pro test-pass eval wrapper')
    ap.add_argument('--traces_dir', required=True, help='dir with <instance_id>.patch files')
    ap.add_argument('--condition', required=True, help='label, e.g. NOINT / REDIRECT / HONEST')
    ap.add_argument('--swebp_repo', required=True, help='path to a clone of scaleapi/SWE-bench_Pro-os')
    ap.add_argument('--work_dir', default='/tmp/testpass_eval', help='scratch dir for inputs+outputs')
    ap.add_argument('--result_json', default=None, help='our output json (default: work_dir/<condition>_testpass.json)')
    ap.add_argument('--dockerhub_username', default='jefzda')
    ap.add_argument('--num_workers', type=int, default=4, help='parallel docker workers (low for a laptop)')
    ap.add_argument('--limit', type=int, default=None, help='evaluate only first N instances (piloting)')
    ap.add_argument('--dry_run', action='store_true', help='build inputs, print scope, do not invoke Docker')
    args = ap.parse_args()

    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    cond = args.condition
    result_json = Path(args.result_json) if args.result_json else work / f'{cond}_testpass.json'

    print(f'[1/5] loading SWE-bench Pro dataset...')
    rows = load_dataset_rows()
    print(f'      {len(rows)} instances in dataset')

    print(f'[2/5] collecting {cond} patches from {args.traces_dir}...')
    patches, dropped, n_empty = collect_patches(args.traces_dir, cond, rows, limit=args.limit)
    print(f'      {len(patches)} patches collected ({n_empty} empty/0B, kept as resolve=False), '
          f'{len(dropped)} dropped')
    for iid, why in dropped[:10]:
        print(f'        drop {iid[:50]}: {why}')

    if not patches:
        print('No patches to evaluate. Exiting.')
        return

    # [3/5] build eval inputs
    patch_path = work / f'{cond}_patches.json'
    sample_path = work / f'{cond}_raw_sample.jsonl'
    patch_path.write_text(json.dumps(patches, indent=1))
    with open(sample_path, 'w') as f:
        for p in patches:
            f.write(json.dumps(rows[p['instance_id']], default=str) + '\n')
    print(f'[3/5] wrote {patch_path.name} ({len(patches)} patches) + {sample_path.name}')

    eval_out = work / f'{cond}_eval_out'
    eval_out.mkdir(exist_ok=True)

    cmd = [
        sys.executable, 'swe_bench_pro_eval.py',
        '--raw_sample_path', str(sample_path),
        '--patch_path', str(patch_path),
        '--output_dir', str(eval_out) + '/',
        '--dockerhub_username', args.dockerhub_username,
        '--use_local_docker',
        '--num_workers', str(args.num_workers),
    ]
    print(f'[4/5] eval command:\n      cd {args.swebp_repo} && {" ".join(cmd)}')

    if args.dry_run:
        repos = sorted(set(rows[p['instance_id']]['repo'] for p in patches))
        print(f'\n--dry-run: would evaluate {len(patches)} instances across repos {repos}')
        print(f'  Docker images: jefzda/sweap-images:<tag>, one per instance (multi-GB each).')
        print(f'  Inputs written to {work}. Re-run without --dry-run to execute.')
        return

    if not (Path(args.swebp_repo) / 'swe_bench_pro_eval.py').exists():
        print(f'ERROR: swe_bench_pro_eval.py not found in {args.swebp_repo}')
        print('  git clone https://github.com/scaleapi/SWE-bench_Pro-os <path>')
        sys.exit(1)
    try:
        subprocess.run(['docker', 'info'], capture_output=True, check=True)
    except Exception:
        print('ERROR: Docker not available. Start Docker and retry.')
        sys.exit(1)

    print(f'[4/5] running official harness (this pulls Docker images — slow)...')
    r = subprocess.run(cmd, cwd=args.swebp_repo)
    if r.returncode != 0:
        print(f'WARNING: eval harness exited {r.returncode} — parsing whatever output exists')

    print(f'[5/5] parsing {eval_out}...')
    parsed = parse_eval_output(eval_out, cond, rows)
    n_resolved = sum(1 for v in parsed.values() if v.get('resolved') is True)
    n_undecided = sum(1 for v in parsed.values() if v.get('resolved') is None)
    out = {
        'condition': cond,
        'n_patches': len(patches),
        'n_evaluated': len(parsed),
        'n_resolved': n_resolved,
        'n_unresolved': len(parsed) - n_resolved - n_undecided,
        'n_undecided': n_undecided,
        'per_instance': parsed,
    }
    result_json.write_text(json.dumps(out, indent=1))
    print(f'      resolved {n_resolved}/{len(parsed)}  (undecided: {n_undecided})')
    print(f'      saved {result_json}')
    if n_undecided:
        print(f'      NOTE: {n_undecided} undecided — no eval_results entry and no parseable')
        print(f'      output.json (instance errored). Check {eval_out}/<uid>/{cond}_stderr.log.')


if __name__ == '__main__':
    main()
