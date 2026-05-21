"""Prep a blinded patch set for the Paper 11 patch-judge.

Collects non-empty patches across conditions, STRIPS the condition label, assigns
opaque IDs, shuffles, and splits into shards — so a judge sees only diff + problem
and cannot infer NOINT/REDIRECT/RESAMPLE (design section 7: blinding mitigates
patch-judge circularity). The de-anonymization map is written separately and must
NOT be shown to the judge.

Empty (0B) patches are not sent to the judge — an empty diff is an honest
non-attempt by definition; they are recorded pre-classified.
"""
import argparse, hashlib, json, random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--conditions', nargs='+', default=['NOINT', 'REDIRECT', 'RESAMPLE'])
    ap.add_argument('--out_dir', default='/tmp/p1_judge')
    ap.add_argument('--shards', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
    problem = {r['instance_id']: r.get('problem_statement', '') for r in ds}

    items, blind_map, empties = [], {}, []
    run = Path(args.run_dir)
    for cond in args.conditions:
        tdir = run / 'traces' / cond
        if not tdir.exists():
            continue
        for pf in sorted(tdir.glob('*.patch')):
            iid = pf.stem
            diff = pf.read_text()
            if not diff.strip():
                empties.append({'condition': cond, 'instance_id': iid})
                continue
            jid = 'p' + hashlib.sha1(f'{cond}/{iid}/{args.seed}'.encode()).hexdigest()[:12]
            items.append({'judge_id': jid,
                          'problem_statement': problem.get(iid, '(not found)'),
                          'diff': diff})
            blind_map[jid] = {'condition': cond, 'instance_id': iid}

    random.seed(args.seed)
    random.shuffle(items)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'blind_map.json').write_text(json.dumps(blind_map, indent=1))
    (out / 'empty_patches.json').write_text(json.dumps(empties, indent=1))

    shards = [items[i::args.shards] for i in range(args.shards)]
    for s, shard in enumerate(shards):
        (out / f'blinded_shard_{s}.jsonl').write_text(
            '\n'.join(json.dumps(it) for it in shard))

    print(f'{len(items)} non-empty patches blinded, {len(empties)} empty pre-classified honest')
    print(f'shards: ' + ', '.join(f'{s}={len(sh)}' for s, sh in enumerate(shards)))
    print(f'wrote {out}/blinded_shard_*.jsonl  +  blind_map.json (private)  +  empty_patches.json')


if __name__ == '__main__':
    main()
