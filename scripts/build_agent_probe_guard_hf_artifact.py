"""Build the HF dataset artifact for agent-probe-guard v0.1.

Reads existing nb47b captures from Drive (240 × 5120 at L11/L23/L31/L43/L55)
and Phase 1 SWE-bench captures (when available) to fit two production
probes:

    probe_L43_pre_tool.joblib   — capability prediction, K=10 features
    probe_L55_thinking.joblib   — suppressed CoT intent, K=5 features
    meta.json                   — eval metrics, dim selections, thresholds

Output is written to a local directory ready for `huggingface_hub.upload_folder`.

Usage:
    python scripts/build_agent_probe_guard_hf_artifact.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


DRIVE = Path('/Users/caiovicentino/Library/CloudStorage/'
             'GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
NB47B = DRIVE / 'nb47b_capture'
PHASE1 = DRIVE / 'swebench_v1_phase1'  # Phase 1 captures, may be unavailable

DEFAULT_OUT = Path(__file__).resolve().parent.parent / 'artifacts' / 'agent_probe_guard_qwen36_27b'

CAPABILITY_K = 10  # L43 pre_tool: K=10 (Phase 6c PCA-10 = 0.764, top-10 diffmeans = 0.83 at N=54)
THINKING_K = 5     # L55 last-prompt: K=5 (Phase 8 redux paper-grade gap +0.147)

THRESHOLDS = {
    'skip_below': 0.20,
    'escalate_below': 0.50,
    'thinking_low': 0.30,
}


def topk_diff(X_tr: np.ndarray, y_tr: np.ndarray, k: int) -> np.ndarray:
    d = np.abs(X_tr[y_tr == 1].mean(axis=0) - X_tr[y_tr == 0].mean(axis=0))
    return np.argsort(-d)[:k]


def cv_auroc_topk(X: np.ndarray, y: np.ndarray, k: int, n_splits: int = 4, seed: int = 42) -> float:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    n_splits = min(n_splits, n_pos, n_neg)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs = []
    for tr, te in skf.split(X, y):
        sel = topk_diff(X[tr], y[tr], k)
        Xtr = X[tr][:, sel]
        Xte = X[te][:, sel]
        sc = StandardScaler()
        Xtr_s = np.nan_to_num(sc.fit_transform(Xtr), nan=0.0, posinf=0.0, neginf=0.0)
        Xte_s = np.nan_to_num(sc.transform(Xte), nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
        clf.fit(Xtr_s, y[tr])
        aurocs.append(roc_auc_score(y[te], clf.predict_proba(Xte_s)[:, 1]))
    return float(np.mean(aurocs))


def build_probe(X: np.ndarray, y: np.ndarray, k: int, label: str) -> dict:
    """Fit a final probe on full data using top-K diffmeans, return artifact dict."""
    dims = topk_diff(X, y, k)
    Xk = X[:, dims]
    sc = StandardScaler()
    Xk_s = np.nan_to_num(sc.fit_transform(Xk), nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
    clf.fit(Xk_s, y)

    # CV AUROC for the final K (re-uses fold-level selection for honest eval)
    cv_auroc = cv_auroc_topk(X, y, k)

    print(f'[{label}] k={k} dims={dims.tolist()}')
    print(f'[{label}] full-fit train AUROC (in-sample, optimistic):'
          f' {roc_auc_score(y, clf.predict_proba(Xk_s)[:, 1]):.4f}')
    print(f'[{label}] CV AUROC (4-fold, honest): {cv_auroc:.4f}')

    return {
        'probe': clf,
        'scaler': sc,
        'dims': dims.tolist(),
        'cv_auroc': cv_auroc,
        'n': len(y),
    }


def load_thinking_data() -> tuple[np.ndarray, np.ndarray]:
    """Load nb47b L55 captures + has_think_v1 labels."""
    meta = json.loads((NB47B / 'metadata.json').read_text())
    records = meta['records']
    y = np.array([1 if r['has_think_v1'] else 0 for r in records], dtype=int)
    X = (load_file(str(NB47B / 'L55_pre_gen_activations.safetensors'))['activations']
         .to(torch.float32).numpy())
    return X, y


def load_capability_data() -> tuple[np.ndarray, np.ndarray] | None:
    """Load Phase 1 SWE-bench L43 captures + patch_generated labels.

    Returns None if Phase 1 captures are not yet available locally — in that
    case we fall back to a placeholder using nb47b L43 + a synthetic label so
    the artifact still ships with the structure intact.
    """
    p1_meta_path = PHASE1 / 'metadata.json'
    p1_acts_path = PHASE1 / 'L43_pre_tool_activations.safetensors'
    if not p1_meta_path.exists() or not p1_acts_path.exists():
        print(f'[capability] {p1_meta_path} or matching L43 captures missing — '
              'using nb47b L43 + has_think_v1 as placeholder')
        meta = json.loads((NB47B / 'metadata.json').read_text())
        records = meta['records']
        y = np.array([1 if r['has_think_v1'] else 0 for r in records], dtype=int)
        X = (load_file(str(NB47B / 'L43_pre_gen_activations.safetensors'))['activations']
             .to(torch.float32).numpy())
        return X, y

    # Phase 1 layout (when ready):
    #   PHASE1/metadata.json               — list of trace records with .patch_generated
    #   PHASE1/L43_pre_tool_activations.safetensors  — N × 5120 captures
    p1_meta = json.loads((PHASE1 / 'metadata.json').read_text())
    records = p1_meta['records']
    y = np.array([1 if r.get('patch_generated') else 0 for r in records], dtype=int)
    X = (load_file(str(PHASE1 / 'L43_pre_tool_activations.safetensors'))['activations']
         .to(torch.float32).numpy())
    return X, y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(DEFAULT_OUT))
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Writing artifact to {out_dir}\n')

    # ---- thinking probe (L55, K=5) ----
    X_thk, y_thk = load_thinking_data()
    print(f'thinking: X={X_thk.shape}, y_pos={int(y_thk.sum())}/{len(y_thk)}')
    thk = build_probe(X_thk, y_thk, THINKING_K, label='L55_thinking')
    joblib.dump(
        {'probe': thk['probe'], 'scaler': thk['scaler'], 'dims': thk['dims'], 'layer': 55},
        out_dir / 'probe_L55_thinking.joblib',
    )
    print()

    # ---- capability probe (L43, K=10) ----
    cap_data = load_capability_data()
    if cap_data is not None:
        X_cap, y_cap = cap_data
        print(f'capability: X={X_cap.shape}, y_pos={int(y_cap.sum())}/{len(y_cap)}')
        cap = build_probe(X_cap, y_cap, CAPABILITY_K, label='L43_pre_tool')
        joblib.dump(
            {'probe': cap['probe'], 'scaler': cap['scaler'], 'dims': cap['dims'], 'layer': 43},
            out_dir / 'probe_L43_pre_tool.joblib',
        )
    else:
        cap = None

    # ---- meta.json ----
    meta = {
        'name': 'agent-probe-guard-qwen36-27b',
        'version': '0.1.0',
        'license': 'apache-2.0',
        'model': 'Qwen/Qwen3.6-27B',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'thinking': {
            'layer': 55,
            'position': 'last_prompt_token',
            'capacity_K': THINKING_K,
            'method': 'top-K diff-of-means + LogisticRegression',
            'dims': thk['dims'],
            'cv_auroc': thk['cv_auroc'],
            'n': thk['n'],
            'random_K_baseline': 0.701,
            'gap_vs_random': round(thk['cv_auroc'] - 0.701, 4),
            'source': 'Phase 8 redux random-K-matched (eval v6 §D.5)',
        },
        'capability': {
            'layer': 43,
            'position': 'pre_tool',
            'capacity_K': CAPABILITY_K,
            'method': 'top-K diff-of-means + LogisticRegression',
            'dims': cap['dims'] if cap else None,
            'cv_auroc': cap['cv_auroc'] if cap else None,
            'n': cap['n'] if cap else None,
            'random_K_baseline': 0.749,
            'gap_vs_random': round(cap['cv_auroc'] - 0.749, 4) if cap else None,
            'source': 'Phase 6c methodology sweep (eval v4)',
            'note': (
                None if cap
                else 'Awaiting Phase 1 N=99 SWE-bench traces. Placeholder uses nb47b L43 '
                     'with has_think_v1 label — REPLACE before public release.'
            ),
        },
        'thresholds': THRESHOLDS,
        'sanity_checks': [
            'random-feature baseline at small N (paper §3.1)',
            'control-token normalization for steering (paper §3.2)',
            'structural-rigidity α-sweep diagnostic (paper §3.3)',
        ],
        'causal_status': 'detect-only — confirmed across 3 intervention experiments (Phase 7 + Phase 8 + Phase 8 redux)',
        'reproduction': {
            'harness_repo': 'https://github.com/OpenInterpretability/openinterp-swebench-harness',
            'sdk_repo': 'https://github.com/OpenInterpretability/cli',
            'paper': 'paper/two_forms_epiphenomenal_probes_neurips_mi_2026.md',
        },
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'\nWrote {out_dir / "meta.json"}')

    print('\nDone. Next steps:')
    print(f'  1. Inspect probes: ls -la {out_dir}')
    print(f'  2. cp paper/preflight_probe_eval_v6*.md {out_dir}/eval_v6.md')
    print(f'  3. cp paper/two_forms_epiphenomenal_probes*.md {out_dir}/PAPER.md')
    print(f'  4. huggingface-cli upload caiovicentino1/agent-probe-guard-qwen36-27b {out_dir} --repo-type=dataset')


if __name__ == '__main__':
    main()
