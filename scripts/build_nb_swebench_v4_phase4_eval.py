"""Generate notebooks/nb_swebench_v4_phase4_eval.ipynb — Phase 4 native test execution.

Validates whether Phase 3 agent-generated patches actually pass the SWE-bench Pro hidden
tests, as the ground-truth label our 'patch_bytes>0' proxy lacks.

Approach: native pip install + pytest in Colab runtime (Docker-in-Docker not available
on Colab without privileged mode). Per-instance:
  1. Clone repo @ base_commit
  2. Run before_repo_set_cmd (project setup)
  3. pip install -e .
  4. Apply test_patch (exposes hidden test cases from fail_to_pass)
  5. For each candidate patch (none / golden / agent_alpha0.5 / agent_alpha0.0):
     - Apply candidate
     - pytest selected_test_files_to_run -k '{fail_to_pass}'
     - Record pass/fail
     - Reset

Expected outcome:
  - golden → all fail_to_pass tests PASS  (validates env)
  - empty/no patch → fail_to_pass tests FAIL  (baseline check)
  - agent patch → unknown — this is what Phase 4 answers

Focus: f3b26 (openlibrary, coherent recovery candidate). Optionally extends to 233cb
(qutebrowser, destructive case — expected FAIL).
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_swebench_v4_phase4_eval.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n", "outputs": [], "execution_count": None}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# SWE-bench Pro Failure Anatomy — Phase 4 Native Test Eval

Validates Phase 3 agent-generated patches against ground-truth SWE-bench Pro tests.

**Why this matters**: Phase 1/2/3 used `patch_bytes>0` as success proxy. Phase 3 inspection
revealed this proxy is noisy — 233cb's 181KB "recovery" was destructive bulk deletion, not a fix.
Phase 4 runs the actual hidden tests (`fail_to_pass`) to give true PASS/FAIL labels.

**Approach**: Native pytest in Colab runtime (no Docker — Colab restrictions). Per-instance:
1. Clone repo @ base_commit, install deps
2. Apply test_patch (exposes hidden tests)
3. For each candidate (none / golden / agent_α0.5 / agent_α0.0): apply → pytest → reset
4. Report pass/fail breakdown

**Target**: f3b26 (openlibrary coherent recovery candidate). Extends to 233cb (qutebrowser
destructive — expected FAIL) if env supports.

**Hardware**: standard CPU runtime sufficient. **No GPU needed.** ~30-60 min total.
"""),
    code("""
# 1) Install + Drive mount
!pip install -q datasets pytest huggingface-hub
import os
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/openinterp_runs'
P1 = f'{DRIVE}/swebench_v1_phase1'
P3 = f'{DRIVE}/swebench_v3_phase3'
P4 = f'{DRIVE}/swebench_v4_phase4'
os.makedirs(P4, exist_ok=True)
assert os.path.exists(f'{P1}/phase1_report.json'), 'Phase 1 must complete first'
print(f'Phase 4 root: {P4}')
"""),
    code("""
# 2) Load Phase 1 + Phase 3 + dataset
from datasets import load_dataset
import json

ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
ds_by_iid = {x['instance_id']: dict(x) for x in ds}

with open(f'{P1}/phase1_report.json') as f:
    p1 = json.load(f)
with open(f'{P3}/phase3_report.json') as f:
    p3 = json.load(f)

# Identify the recovery candidates from Phase 3
TARGET_IIDS = {
    'f3b26': 'instance_internetarchive__openlibrary-f3b26c2c0721f8713353fe4b341230332e30008d-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4',
    '233cb': 'instance_qutebrowser__qutebrowser-233cb1cc48635130e5602549856a6fa4ab4c087f-v35616345bb8052ea303186706cec663146f0f184',
    'e4088': 'instance_ansible__ansible-e40889e7112ae00a21a2c74312b330e67a766cc0-v1055803c3a812189a1133297f7f5468579283f86',
}
for short, iid in TARGET_IIDS.items():
    assert iid in ds_by_iid, f'{short} ({iid}) not in dataset'
    inst = ds_by_iid[iid]
    print(f'{short}: {inst[\"repo\"]:32s}  fail_to_pass={len(inst.get(\"fail_to_pass\") or [])}  test_files={len(inst.get(\"selected_test_files_to_run\") or [])}')
"""),
    code("""
# 3) Helper: clone + setup + apply + test
import subprocess
import shlex
import shutil
from pathlib import Path

WORK_ROOT = Path('/content/p4_work')
WORK_ROOT.mkdir(parents=True, exist_ok=True)

def run_shell(cmd: str, cwd: str | Path | None = None, timeout: int = 600) -> tuple[int, str, str]:
    \"\"\"Run shell command, return (exit_code, stdout, stderr).\"\"\"
    try:
        r = subprocess.run(cmd, shell=True, executable='/bin/bash', cwd=str(cwd) if cwd else None,
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        return -1, e.stdout or '', f'TIMEOUT after {timeout}s'

def setup_workdir(inst: dict, force_clean: bool = False) -> Path:
    iid = inst['instance_id']
    workdir = WORK_ROOT / iid
    if workdir.exists() and not force_clean:
        return workdir
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.parent.mkdir(parents=True, exist_ok=True)
    repo = inst['repo']
    base = inst['base_commit']
    print(f'  cloning {repo}...')
    rc, _, err = run_shell(f'git clone --quiet https://github.com/{shlex.quote(repo)} {shlex.quote(str(workdir))}', timeout=300)
    if rc != 0:
        raise RuntimeError(f'clone failed: {err[-300:]}')
    rc, _, err = run_shell(f'git -c advice.detachedHead=false checkout --quiet {base}', cwd=workdir, timeout=60)
    if rc != 0:
        raise RuntimeError(f'checkout failed: {err[-300:]}')
    return workdir

def reset_workdir(workdir: Path):
    \"\"\"Reset to base_commit, drop all changes.\"\"\"
    run_shell('git reset --hard --quiet HEAD', cwd=workdir, timeout=30)
    run_shell('git clean -fdq', cwd=workdir, timeout=30)

def apply_patch(workdir: Path, patch_text: str) -> bool:
    if not patch_text.strip():
        return True
    p = workdir / '_apply.patch'
    p.write_text(patch_text)
    rc, out, err = run_shell(f'git apply --whitespace=nowarn {shlex.quote(str(p))}', cwd=workdir, timeout=60)
    if rc != 0:
        # Try with --reject for partial application
        rc2, _, err2 = run_shell(f'git apply --reject --whitespace=nowarn {shlex.quote(str(p))}', cwd=workdir, timeout=60)
        return rc2 == 0
    return True

print('Helpers ready.')
"""),
    code("""
# 4) Validate ONE instance: f3b26 (openlibrary)
import json
from pathlib import Path

short = 'f3b26'
inst = ds_by_iid[TARGET_IIDS[short]]
print(f'=== {short}: {inst[\"repo\"]} @ {inst[\"base_commit\"][:8]} ===\\n')

workdir = setup_workdir(inst, force_clean=True)
print(f'Workdir: {workdir}\\n')

# Run before_repo_set_cmd (project setup)
before = inst.get('before_repo_set_cmd') or ''
if before:
    print(f'Setup cmd (first 200 chars): {before[:200]}\\n')
    rc, out, err = run_shell(before, cwd=workdir, timeout=900)
    print(f'  setup exit={rc}, stdout {len(out)}b, stderr {len(err)}b')
    if rc != 0:
        print(f'  setup stderr (last 500): {err[-500:]}')

# Try to install package as editable
rc, _, err = run_shell('pip install -q -e . 2>&1 | tail -5', cwd=workdir, timeout=600)
print(f'\\npip install exit={rc}')

# Show fail_to_pass tests
ftp = inst.get('fail_to_pass') or []
ptp = inst.get('pass_to_pass') or []
test_files = inst.get('selected_test_files_to_run') or []
print(f'\\nfail_to_pass: {len(ftp)} tests')
for t in ftp[:5]:
    print(f'  {t}')
print(f'\\npass_to_pass: {len(ptp)} tests')
print(f'\\nselected_test_files: {len(test_files)}')
for f in test_files[:5]:
    print(f'  {f}')
"""),
    code("""
# 5) Run tests for 4 conditions: none / golden / agent_alpha0.5 / agent_alpha0.0
import json
from pathlib import Path
import re

def load_phase3_patch(short: str, alpha: str) -> str:
    iid = TARGET_IIDS[short]
    p = Path(f'{P3}/alpha_{alpha}/traces/{iid}.patch')
    return p.read_text() if p.exists() else ''

def run_pytest(workdir: Path, test_specs: list[str], timeout: int = 900) -> dict:
    \"\"\"Run pytest with given test specs (e.g. 'tests/test_x.py::test_func').
    Returns dict with pass/fail per test.\"\"\"
    if not test_specs:
        return {'error': 'no test specs'}
    arg = ' '.join(shlex.quote(t) for t in test_specs)
    cmd = f'python3 -m pytest {arg} --tb=short --no-header -q 2>&1 | tail -100'
    rc, out, err = run_shell(cmd, cwd=workdir, timeout=timeout)

    passed = []
    failed = []
    errored = []
    for line in out.split('\\n'):
        # pytest output: 'tests/x.py::test_y PASSED' or '... FAILED'
        m = re.match(r'(.+?)\\s+(PASSED|FAILED|ERROR|SKIPPED)\\s*', line.strip())
        if m:
            test_name, status = m.group(1).strip(), m.group(2)
            if status == 'PASSED':
                passed.append(test_name)
            elif status == 'FAILED':
                failed.append(test_name)
            elif status == 'ERROR':
                errored.append(test_name)
    # Pytest summary: '== 5 passed, 2 failed in 3.4s =='
    summary = ''
    for line in reversed(out.split('\\n')):
        if 'passed' in line or 'failed' in line or 'error' in line:
            summary = line.strip()
            break
    return {
        'rc': rc,
        'n_passed': len(passed),
        'n_failed': len(failed),
        'n_errored': len(errored),
        'summary': summary,
        'output_tail': out[-2000:],
    }

# 4 conditions
conditions = {
    'none': '',  # baseline broken state
    'golden': inst.get('patch') or '',  # canonical fix
    'agent_a0_5': load_phase3_patch(short, '0.5'),
    'agent_a0_0': load_phase3_patch(short, '0.0'),
}

# fail_to_pass test specs
ftp = inst.get('fail_to_pass') or []
print(f'Will run {len(ftp)} fail_to_pass tests under each of {len(conditions)} conditions.\\n')

# Apply test_patch ONCE first (it stays — adds the test cases)
test_patch = inst.get('test_patch') or ''
if test_patch:
    reset_workdir(workdir)
    ok = apply_patch(workdir, test_patch)
    print(f'test_patch apply: {\"OK\" if ok else \"FAILED\"}\\n')

# Save state with test_patch applied
run_shell('git add -A && git commit -q -m \"test_patch baseline\" --allow-empty', cwd=workdir, timeout=30)

results_p4: dict[str, dict] = {}
for cond, patch_text in conditions.items():
    print(f'\\n--- Condition: {cond} (patch_bytes={len(patch_text)}) ---')
    # Reset to test_patch baseline
    rc, _, _ = run_shell('git reset --hard --quiet HEAD', cwd=workdir, timeout=30)
    rc, _, _ = run_shell('git clean -fdq', cwd=workdir, timeout=30)

    if patch_text.strip():
        ok = apply_patch(workdir, patch_text)
        print(f'  apply: {\"OK\" if ok else \"FAILED\"}')
        if not ok:
            results_p4[cond] = {'apply_failed': True}
            continue

    r = run_pytest(workdir, ftp, timeout=600)
    results_p4[cond] = r
    print(f'  pytest: passed={r.get(\"n_passed\")}  failed={r.get(\"n_failed\")}  error={r.get(\"n_errored\")}')
    print(f'  summary: {r.get(\"summary\")[:200]}')

# Save phase4 results
out_path = Path(P4) / f'phase4_{short}.json'
out_path.write_text(json.dumps(results_p4, indent=2))
print(f'\\nWrote {out_path}')
"""),
    code("""
# 6) Verdict — compare conditions
import json

print('=========== Phase 4 Verdict (f3b26) ===========')
for cond in ['none', 'golden', 'agent_a0_0', 'agent_a0_5']:
    r = results_p4.get(cond, {})
    if r.get('apply_failed'):
        print(f'  {cond:>12s}: PATCH APPLY FAILED')
    elif r.get('error') == 'no test specs':
        print(f'  {cond:>12s}: no fail_to_pass tests defined')
    else:
        np_, nf, ne = r.get('n_passed', 0), r.get('n_failed', 0), r.get('n_errored', 0)
        total = np_ + nf + ne
        pass_pct = (np_ / total * 100) if total else 0
        print(f'  {cond:>12s}: {np_}/{total} pass ({pass_pct:.0f}%)  [P={np_} F={nf} E={ne}]')

# Decision logic
golden_pass = results_p4.get('golden', {}).get('n_passed', 0)
none_pass = results_p4.get('none', {}).get('n_passed', 0)
agent_pass = results_p4.get('agent_a0_5', {}).get('n_passed', 0)

if golden_pass > none_pass:
    print(f'\\nEnvironment OK (golden patches pass more than baseline).')
    if agent_pass >= golden_pass:
        print(f'🟢 Agent patch matches golden — TRUE recovery via steering!')
    elif agent_pass > none_pass:
        print(f'🟡 Agent patch partially fixes (more passes than baseline).')
    elif agent_pass <= none_pass:
        print(f'🔴 Agent patch does NOT fix issue — coherent-looking but ineffective.')
else:
    print(f'\\n⚠️  Env mismatch: golden patch did not improve over baseline. Cannot validate.')
    print(f'   Likely missing dependencies; rerun in proper SWE-bench Pro Docker env.')
"""),
    code("""
# 7) (optional) Same eval on 233cb destructive case — should clearly FAIL
short = '233cb'
inst = ds_by_iid[TARGET_IIDS[short]]
print(f'\\n=== {short}: {inst[\"repo\"]} ===')
print(f'Note: 233cb at α=0.5 deleted 3151 lines from configdata.yml. Expected: FAIL.\\n')

# Setup
try:
    workdir2 = setup_workdir(inst, force_clean=True)

    before = inst.get('before_repo_set_cmd') or ''
    if before:
        run_shell(before, cwd=workdir2, timeout=900)

    rc, _, err = run_shell('pip install -q -e . 2>&1 | tail -5', cwd=workdir2, timeout=600)
    print(f'pip install exit: {rc}')

    test_patch = inst.get('test_patch') or ''
    reset_workdir(workdir2)
    if test_patch:
        ok = apply_patch(workdir2, test_patch)
        print(f'test_patch: {\"OK\" if ok else \"FAILED\"}')
    run_shell('git add -A && git commit -q -m baseline --allow-empty', cwd=workdir2)

    ftp_233 = inst.get('fail_to_pass') or []
    if not ftp_233:
        print('No fail_to_pass tests for 233cb. Skipping.')
    else:
        results_233: dict[str, dict] = {}
        for cond_name, patch_text in [
            ('agent_a0_5', load_phase3_patch('233cb', '0.5')),
            ('golden', inst.get('patch') or ''),
        ]:
            run_shell('git reset --hard --quiet HEAD', cwd=workdir2)
            if patch_text.strip():
                ok = apply_patch(workdir2, patch_text)
                if not ok:
                    print(f'  {cond_name}: apply failed')
                    results_233[cond_name] = {'apply_failed': True}
                    continue
            r = run_pytest(workdir2, ftp_233, timeout=600)
            results_233[cond_name] = r
            print(f'  {cond_name}: passed={r.get(\"n_passed\")}/{r.get(\"n_passed\",0)+r.get(\"n_failed\",0)+r.get(\"n_errored\",0)}')

        Path(f'{P4}/phase4_233cb.json').write_text(json.dumps(results_233, indent=2))
        print(f'\\nSaved phase4_233cb.json')
except Exception as e:
    print(f'⚠️  233cb eval failed: {type(e).__name__}: {e}')
    print('Skipping — qutebrowser may have heavy deps (PyQt5) that are hard on Colab CPU.')
"""),
    code("""
# 8) Aggregate Phase 4 verdict
import json
from pathlib import Path

print('=========== Phase 4 Aggregate Report ===========\\n')
for short, fname in [('f3b26', f'{P4}/phase4_f3b26.json'), ('233cb', f'{P4}/phase4_233cb.json')]:
    p = Path(fname)
    if not p.exists():
        print(f'{short}: no eval result')
        continue
    r = json.loads(p.read_text())
    print(f'{short}:')
    for cond, cr in r.items():
        if cr.get('apply_failed'):
            print(f'  {cond:>12s}: APPLY FAILED')
            continue
        np_ = cr.get('n_passed', 0); nf = cr.get('n_failed', 0); ne = cr.get('n_errored', 0)
        total = np_ + nf + ne
        print(f'  {cond:>12s}: {np_}/{total} pass')
    print()

# Final aggregate
agg = {}
for short in ['f3b26', '233cb']:
    p = Path(f'{P4}/phase4_{short}.json')
    if p.exists():
        agg[short] = json.loads(p.read_text())
Path(f'{P4}/phase4_aggregate.json').write_text(json.dumps(agg, indent=2))
print(f'Wrote {P4}/phase4_aggregate.json')
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "None",
        "colab": {"name": "nb_swebench_v4_phase4_eval.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
