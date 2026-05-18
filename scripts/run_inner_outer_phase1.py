"""V2 — StandardScaler + saga solver + direct output."""
import os, json, sys, warnings, time
import numpy as np
warnings.filterwarnings('ignore')

from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

OUT_PATH = '/tmp/inner_outer_v2_out.txt'
out_f = open(OUT_PATH, 'w', buffering=1)  # line-buffered

def p(*args):
    msg = ' '.join(str(a) for a in args)
    out_f.write(msg + '\n')
    out_f.flush()

DRIVE_ROOT = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
PHASE6_DIR = os.path.join(DRIVE_ROOT, 'swebench_v6_phase6')
CAPTURES_DIR = os.path.join(PHASE6_DIR, 'captures')
RESULTS_JSON = os.path.join(PHASE6_DIR, 'phase6_results.json')
OUT_DIR = os.path.join(DRIVE_ROOT, 'inner_outer_phase1')
os.makedirs(OUT_DIR, exist_ok=True)

PROBE_LAYERS = [11, 23, 31, 43, 55]
POS_INNER = 'think_end'
POS_OUTER = 'turn_end'

t_start = time.time()

# === Load Phase 6 results ===
with open(RESULTS_JSON) as f:
    phase6_results = json.load(f)
p(f'[{time.time()-t_start:.0f}s] Phase 6 instances: {len(phase6_results)}')

labels = {}
for iid, info in phase6_results.items():
    finished = info.get('finished', False)
    finish_reason = info.get('finish_reason', '')
    labels[iid] = 0 if (finished and finish_reason == 'finish_tool') else 1
n_fail = sum(labels.values())
p(f'  Success (label=0): {len(labels) - n_fail}, Failed (label=1): {n_fail}')

# === Load paired residuals ===
def load_paired_residuals(iid, probe_layers=PROBE_LAYERS):
    safetensors_path = os.path.join(CAPTURES_DIR, f'{iid}.safetensors')
    meta_path = os.path.join(CAPTURES_DIR, f'{iid}.meta.json')
    if not os.path.exists(safetensors_path) or not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    by_key = {(r['turn_idx'], r['position_label'], r['layer']): r['activation_key'] for r in meta['records']}
    turn_indices = set()
    for t in range(60):
        if all((t, POS_INNER, L) in by_key for L in probe_layers) and \
           all((t, POS_OUTER, L) in by_key for L in probe_layers):
            turn_indices.add(t)
    if not turn_indices:
        return None
    paired_per_layer = {L: [] for L in probe_layers}
    with safe_open(safetensors_path, framework='pt') as f:
        for t in sorted(turn_indices):
            for L in probe_layers:
                v_i = f.get_tensor(by_key[(t, POS_INNER, L)]).float().numpy()
                v_o = f.get_tensor(by_key[(t, POS_OUTER, L)]).float().numpy()
                paired_per_layer[L].append((t, v_i, v_o))
    return paired_per_layer


X_think = {L: [] for L in PROBE_LAYERS}
X_turn = {L: [] for L in PROBE_LAYERS}
y_list, groups = [], []
skipped = 0
t_load = time.time()
for i, (iid, info) in enumerate(phase6_results.items()):
    paired = load_paired_residuals(iid)
    if paired is None:
        skipped += 1
        continue
    label = labels[iid]
    for L in PROBE_LAYERS:
        for (turn_idx, v_i, v_o) in paired[L]:
            X_think[L].append(v_i)
            X_turn[L].append(v_o)
            if L == PROBE_LAYERS[0]:
                y_list.append(label)
                groups.append(iid)
    if (i + 1) % 20 == 0:
        p(f'  [{i+1}/{len(phase6_results)}] {(time.time()-t_load)/60:.1f}min  skipped={skipped}')

X_think = {L: np.stack(X_think[L]) for L in PROBE_LAYERS}
X_turn = {L: np.stack(X_turn[L]) for L in PROBE_LAYERS}
y = np.array(y_list, dtype=int)
groups = np.array(groups)
n_inst = len(set(groups))
p(f'[{time.time()-t_start:.0f}s] Loaded {len(y)} samples from {n_inst} instances ({skipped} skipped)')
p(f'  Shape per layer: {X_think[PROBE_LAYERS[0]].shape}')
p(f'  Mean norm think: {np.linalg.norm(X_think[PROBE_LAYERS[0]], axis=1).mean():.1f}')
p(f'  Mean norm turn:  {np.linalg.norm(X_turn[PROBE_LAYERS[0]], axis=1).mean():.1f}')
p(f'  Label balance: success={int((y==0).sum())} fail={int((y==1).sum())}')

# === Train probes ===
MODES = {
    'think_only': lambda L: X_think[L],
    'turn_only': lambda L: X_turn[L],
    'paired_concat': lambda L: np.concatenate([X_think[L], X_turn[L]], axis=1),
    'paired_diff': lambda L: X_turn[L] - X_think[L],
}
SEEDS = [42, 7, 1337]

# Use liblinear (much faster for high-dim) + StandardScaler to fix overflow
def train_fold(Xtr, Xte, ytr, yte):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)
    clf = LogisticRegression(class_weight='balanced', max_iter=500, C=0.1, solver='liblinear')
    clf.fit(Xtr_s, ytr)
    try:
        return roc_auc_score(yte, clf.decision_function(Xte_s))
    except ValueError:
        return None

results = {}
p(f'\n[{time.time()-t_start:.0f}s] Training probes...')
for L in PROBE_LAYERS:
    for mode_name, mode_fn in MODES.items():
        t_mode = time.time()
        X = mode_fn(L).astype(np.float32)
        real_seeds, rand_seeds = [], []
        for seed in SEEDS:
            np.random.seed(seed)
            gkf = GroupKFold(n_splits=5)
            real_a, rand_a = [], []
            for tr, te in gkf.split(X, y, groups):
                a = train_fold(X[tr], X[te], y[tr], y[te])
                if a is not None: real_a.append(a)
                rng = np.random.default_rng(seed)
                Xr = rng.standard_normal(X.shape).astype(np.float32)
                a_r = train_fold(Xr[tr], Xr[te], y[tr], y[te])
                if a_r is not None: rand_a.append(a_r)
            real_seeds.append(float(np.mean(real_a)) if real_a else float('nan'))
            rand_seeds.append(float(np.mean(rand_a)) if rand_a else float('nan'))
        results[(L, mode_name)] = {
            'real': float(np.mean(real_seeds)),
            'real_std': float(np.std(real_seeds)),
            'rand': float(np.mean(rand_seeds)),
            'gap': float(np.mean(real_seeds) - np.mean(rand_seeds)),
        }
        p(f'  L{L:2d} {mode_name:15s}: real={results[(L,mode_name)]["real"]:.3f}±{results[(L,mode_name)]["real_std"]:.3f}  rand={results[(L,mode_name)]["rand"]:.3f}  gap={results[(L,mode_name)]["gap"]:+.3f}  ({time.time()-t_mode:.0f}s)')

# === Heatmap ===
modes_list = list(MODES.keys())
p('\n=== REAL AUROC ===')
p(f'{"Layer":<6} ' + ' '.join(f'{m:>15s}' for m in modes_list))
for L in PROBE_LAYERS:
    row = f'L{L:<5d} '
    for mode in modes_list:
        r = results.get((L, mode))
        row += f' {r["real"]:>14.3f}' if r else '               -'
    p(row)

p('\n=== GAP vs D1 RANDOM ===')
p(f'{"Layer":<6} ' + ' '.join(f'{m:>15s}' for m in modes_list))
for L in PROBE_LAYERS:
    row = f'L{L:<5d} '
    for mode in modes_list:
        r = results.get((L, mode))
        if r:
            gap = r['gap']
            flag = '*' if gap > 0.20 else ('.' if gap > 0.10 else ' ')
            row += f' {gap:>+13.3f}{flag}'
        else:
            row += '               -'
    p(row)

p('\n=== PAIRED LIFT per layer ===')
for L in PROBE_LAYERS:
    think = results[(L, 'think_only')]['real']
    turn = results[(L, 'turn_only')]['real']
    p_concat = results[(L, 'paired_concat')]['real']
    p_diff = results[(L, 'paired_diff')]['real']
    best_p = max(p_concat, p_diff)
    best_s = max(think, turn)
    lift = best_p - best_s
    flag = '*' if lift > 0.05 else ('.' if lift > 0.02 else ' ')
    p(f'  L{L}: paired={best_p:.3f} single={best_s:.3f} lift={lift:+.3f} {flag}')

# === Gate check ===
p('\n' + '=' * 78)
p('PRE-REGISTERED PREDICTION CHECK')
p('=' * 78)
paired_only = {k: v for k, v in results.items() if 'paired' in k[1]}
best_key = max(paired_only, key=lambda k: paired_only[k]['real'])
best_paired = paired_only[best_key]
best_L = best_key[0]
best_single = max(results[(best_L, 'think_only')]['real'], results[(best_L, 'turn_only')]['real'])

P1 = best_paired['real'] > 0.70
P2 = best_paired['rand'] < 0.55
P3 = (best_paired['real'] - best_single) > 0.05
P4 = best_paired['real_std'] < 0.05
preds = [
    ('P1 best paired AUROC > 0.70', best_paired['real'], '>0.70', P1),
    ('P2 random < 0.55', best_paired['rand'], '<0.55', P2),
    (f'P3 paired-best_single>0.05 (single={best_single:.3f})', best_paired['real']-best_single, '>0.05', P3),
    ('P4 3-seed std < 0.05', best_paired['real_std'], '<0.05', P4),
]
for name, val, thr, pp in preds:
    p(f'  {"PASS" if pp else "FAIL"} {name}: actual={val:+.3f} (need {thr})')
n_pass = sum(1 for _,_,_,pp in preds if pp)
p(f'\n=== {n_pass}/4 PASS — best (L{best_key[0]}, {best_key[1]}) ===')

if n_pass == 4: p('GREEN ship. NEXT: multi-model gate.')
elif n_pass == 3: p('YELLOW 1 fail.')
elif n_pass == 2: p('YELLOW 2 fail.')
else: p('RED walk-back.')

# === INSPECT-RAW ===
L_b, mode_b = best_key
X_b = MODES[mode_b](L_b).astype(np.float32)
scaler = StandardScaler()
X_b_s = scaler.fit_transform(X_b).astype(np.float32)
clf = LogisticRegression(class_weight='balanced', max_iter=500, C=0.1, solver='liblinear')
clf.fit(X_b_s, y)
proba = 1 / (1 + np.exp(-clf.decision_function(X_b_s)))
order = np.argsort(proba)[::-1]

p(f'\n=== TOP-10 predicted FAIL (L{L_b} {mode_b}) ===')
seen = set()
for i in order:
    if groups[i] in seen: continue
    seen.add(groups[i])
    iid = groups[i]
    info = phase6_results[iid]
    p(f'  prob={proba[i]:.3f}  actual={"FAIL" if y[i]==1 else "OK  "}  reason={info.get("finish_reason","?")[:20]:20s}  iid={iid[:55]}')
    if len(seen) >= 10: break

p(f'\n=== BOTTOM-5 predicted (SUCCESS) ===')
seen = set()
for i in order[::-1]:
    if groups[i] in seen: continue
    seen.add(groups[i])
    iid = groups[i]
    info = phase6_results[iid]
    p(f'  prob={proba[i]:.3f}  actual={"FAIL" if y[i]==1 else "OK  "}  reason={info.get("finish_reason","?")[:20]:20s}  iid={iid[:55]}')
    if len(seen) >= 5: break

# Save
with open(os.path.join(OUT_DIR, 'phase1_aurocs.json'), 'w') as f:
    out = {f'L{k[0]}_{k[1]}': v for k, v in results.items()}
    out['_best_paired'] = {'layer': int(best_key[0]), 'mode': best_key[1], **best_paired}
    out['_n_pass'] = n_pass
    out['_n_total'] = int(len(y))
    out['_n_positive'] = int(y.sum())
    out['_n_instances'] = int(n_inst)
    json.dump(out, f, indent=2)
p(f'\nTotal time: {(time.time()-t_start)/60:.1f}min')
p(f'Saved: {os.path.join(OUT_DIR, "phase1_aurocs.json")}')
out_f.close()
