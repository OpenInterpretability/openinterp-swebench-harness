"""Join blinded patch-judge verdicts back to their conditions.

De-anonymizes the blind judge output (verdicts_shard_*.json keyed by judge_id)
via blind_map.json, and writes <COND>_patch_judge.json into the run dir in the
format run_cot_redirect_phase1_analysis.py expects ({instance_id: verdict_obj}).
Empty (0B) patches recorded in empty_patches.json are folded in as honest.
"""
import argparse, json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge_dir', default='/tmp/p1_judge')
    ap.add_argument('--run_dir', required=True)
    args = ap.parse_args()

    jd = Path(args.judge_dir)
    blind_map = json.loads((jd / 'blind_map.json').read_text())

    verdicts = {}
    for f in sorted(jd.glob('verdicts_shard_*.json')):
        for r in json.loads(f.read_text()):
            verdicts[r['judge_id']] = r

    out, unjudged = {}, []
    for jid, meta in blind_map.items():
        v = verdicts.get(jid)
        if v is None:
            unjudged.append(jid)
            continue
        out.setdefault(meta['condition'], {})[meta['instance_id']] = v

    for e in json.loads((jd / 'empty_patches.json').read_text()):
        out.setdefault(e['condition'], {})[e['instance_id']] = {
            'verdict': 'honest', 'confidence': 1.0, 'quote': '',
            'rationale': 'empty diff — honest non-attempt'}

    run = Path(args.run_dir)
    for cond, d in sorted(out.items()):
        (run / f'{cond}_patch_judge.json').write_text(json.dumps(d, indent=1))
        print(f'{cond}: {len(d)} patches -> {dict(Counter(v["verdict"] for v in d.values()))}')
    if unjudged:
        print(f'WARNING — {len(unjudged)} judge_ids unjudged: {unjudged}')
    else:
        print('all blinded patches joined')


if __name__ == '__main__':
    main()
