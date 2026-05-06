"""Phase 6b — Docker evaluation of all Phase 6 patches.

Runs locally on Mac (Docker Desktop with storage on /Volumes/SSD Major). Pulls
the official SWE-bench Pro image per instance, runs 3 conditions (none/golden/agent)
inside the container, parses pytest output → JSON verdict.

Usage:
    python3 phase6b_docker_eval.py [--max-instances N] [--repo FILTER]

Output: phase6b_docker_verdicts.json on Drive (one record per instance).
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import sys
DRIVE = Path('/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
P6 = DRIVE / 'swebench_v6_phase6'
P6B_OUT = P6 / 'phase6b'
P6B_OUT.mkdir(parents=True, exist_ok=True)
WORKSPACE_ROOT = Path('/tmp/p6b_workspace')
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

UNIVERSAL_RUN_SCRIPT = '''#!/bin/bash
set +e
cd /app
export PYTHONPATH=/app
test_files="$1"
if [[ "$test_files" == *","* ]]; then
  IFS=',' read -r -a TEST_FILES <<< "$test_files"
else
  TEST_FILES=("$test_files")
fi
python3 -m pytest "${TEST_FILES[@]}" -v --tb=short --no-header
'''


def parse_pytest(stdout: str) -> dict:
    import re
    n_pass = len(re.findall(r' PASSED', stdout))
    n_fail = len(re.findall(r' FAILED', stdout))
    n_err = len(re.findall(r' ERROR', stdout))
    n_skip = len(re.findall(r' SKIPPED', stdout))
    return {
        'n_pass': n_pass, 'n_fail': n_fail, 'n_err': n_err, 'n_skip': n_skip,
        'total': n_pass + n_fail + n_err + n_skip,
    }


def run_condition(iid, dockerhub_tag, condition, patch_text, base_commit, before_cmd_last, test_files_csv):
    workdir = WORKSPACE_ROOT / iid / condition
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    (workdir / 'patch.diff').write_text(patch_text)
    (workdir / 'run_script.sh').write_text(UNIVERSAL_RUN_SCRIPT)
    entry = f'''#!/bin/bash
set -e
cd /app
git reset --hard {base_commit} 2>&1 | tail -3
git checkout {base_commit} 2>&1 | tail -3
git apply -v /workspace/patch.diff 2>&1 | tail -10 || echo "PATCH_APPLY_FAILED"
{before_cmd_last}
bash /workspace/run_script.sh {test_files_csv} > /workspace/stdout.log 2>&1
'''
    (workdir / 'entryscript.sh').write_text(entry)
    cmd = [
        'docker', 'run', '--rm', '--platform', 'linux/amd64',
        '-v', f'{workdir.resolve()}:/workspace',
        '--entrypoint', '/bin/bash',
        f'jefzda/sweap-images:{dockerhub_tag}',
        '-c', 'bash /workspace/entryscript.sh',
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        wall = time.time() - t0
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        wall, rc = 900, -1
    stdout_log = (workdir / 'stdout.log').read_text() if (workdir / 'stdout.log').exists() else ''
    parsed = parse_pytest(stdout_log)
    parsed['wall_s'] = round(wall, 1)
    parsed['rc'] = rc
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-instances', type=int, default=None)
    ap.add_argument('--repo-filter', type=str, default=None, help='only run for instances of this repo')
    args = ap.parse_args()

    print('Loading Phase 6 results...')
    p6_results_path = P6 / 'phase6_results.json'
    if not p6_results_path.exists():
        print(f'ERROR: {p6_results_path} not found. Run Phase 6 first.')
        sys.exit(1)
    p6 = json.loads(p6_results_path.read_text())
    success_iids = [iid for iid, r in p6.items() if r.get('patch_n_bytes', 0) > 0]
    print(f'Phase 6 done: {len(p6)} instances, {len(success_iids)} with patches.')

    # Load dataset for per-instance metadata
    print('Loading dataset...')
    from datasets import load_dataset
    ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
    ds_by_iid = {x['instance_id']: x for x in ds}

    if args.repo_filter:
        success_iids = [i for i in success_iids if ds_by_iid.get(i, {}).get('repo') == args.repo_filter]
        print(f'After repo filter "{args.repo_filter}": {len(success_iids)}')

    if args.max_instances:
        success_iids = success_iids[:args.max_instances]

    checkpoint = P6B_OUT / 'phase6b_results.json'
    all_results = {}
    if checkpoint.exists():
        all_results = json.loads(checkpoint.read_text())
        print(f'Resuming — {len(all_results)} done.')

    for i, iid in enumerate(success_iids):
        if iid in all_results and 'agent' in all_results[iid].get('conditions', {}):
            print(f'[{i+1}/{len(success_iids)}] SKIP {iid[:60]}')
            continue
        inst = ds_by_iid.get(iid)
        if inst is None:
            print(f'[{i+1}/{len(success_iids)}] NOT IN DATASET: {iid[:60]}')
            continue
        if inst.get('repo_language') != 'python':
            continue
        tag = inst['dockerhub_tag']
        image = f'jefzda/sweap-images:{tag}'
        base = inst['base_commit']
        before_last = inst['before_repo_set_cmd'].strip().split('\n')[-1]
        sel = inst['selected_test_files_to_run']
        test_files_csv = ','.join(eval(sel)) if isinstance(sel, str) else ','.join(sel)
        golden = inst['patch']
        agent_patch_path = P6 / 'traces' / f'{iid}.patch'
        agent_patch = agent_patch_path.read_text() if agent_patch_path.exists() else ''

        print(f'\n[{i+1}/{len(success_iids)}] {iid[:60]} repo={inst["repo"]}')
        print(f'  pulling...')
        pt0 = time.time()
        pull = subprocess.run(['docker', 'pull', '--platform', 'linux/amd64', image],
                              capture_output=True, text=True, timeout=600)
        if pull.returncode != 0:
            print(f'  PULL FAILED: {pull.stderr[-200:]}')
            all_results[iid] = {'pull_failed': True, 'error': pull.stderr[-500:]}
            checkpoint.write_text(json.dumps(all_results, indent=2))
            continue
        print(f'  pull: {time.time() - pt0:.0f}s')

        cond_results = {}
        for cond, ptext in [('none', ''), ('golden', golden), ('agent', agent_patch)]:
            r = run_condition(iid, tag, cond, ptext, base, before_last, test_files_csv)
            cond_results[cond] = r
            print(f'  {cond:>7s}: {r["n_pass"]}P/{r["n_fail"]}F/{r["n_err"]}E  wall={r["wall_s"]}s')

        all_results[iid] = {
            'repo': inst['repo'],
            'patch_bytes_phase6': len(agent_patch),
            'conditions': cond_results,
        }
        checkpoint.write_text(json.dumps(all_results, indent=2))
        print(f'  rmi...')
        subprocess.run(['docker', 'rmi', '-f', image], capture_output=True)

    print(f'\n=== Phase 6b: {len(all_results)} instances evaluated ===')

    # Quick verdict summary
    solves = partial = fails = env_mm = pull_failed = 0
    for iid, r in all_results.items():
        if r.get('pull_failed'):
            pull_failed += 1
            continue
        c = r.get('conditions', {})
        n = c.get('none', {}).get('n_pass', 0)
        g = c.get('golden', {}).get('n_pass', 0)
        a = c.get('agent', {}).get('n_pass', 0)
        if g <= n:
            env_mm += 1
        elif a >= g:
            solves += 1
        elif a > n:
            partial += 1
        else:
            fails += 1
    total = solves + partial + fails + env_mm + pull_failed
    print(f'\nVerdicts (out of {total}):')
    print(f'  SOLVES: {solves}')
    print(f'  PARTIAL: {partial}')
    print(f'  FAILS: {fails}')
    print(f'  env_mismatch: {env_mm}')
    print(f'  pull_failed: {pull_failed}')


if __name__ == '__main__':
    main()
