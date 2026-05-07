"""Phase 7 micro-pilot — log-prob proxy for causality at L43 pre_tool.

Cheap version of full steering experiment. Instead of re-running the full trace
(~10 min/run × N alphas × N traces), we measure the EFFECT of activation steering
at the pre_tool point on the next-N-token log-probs.

Logic:
  - At pre_tool position, model is about to commit to a tool call action
  - If probe direction is causal, +α should shift model toward "finish" tokens
    (signal that the agent thinks it's done) for traces that solved, and shift
    failing traces toward more exploration tokens (search, write, etc.)
  - If probe is epiphenomenal, log-prob distribution should be unchanged

Setup:
  - 8 traces selected by extreme probe scores (5 lowest = fails, 3 highest = solves)
  - α sweep: {-2, 0, +1, +2}
  - Hook L43 at pre_tool position, modify activation
  - Forward through 44-63, unembed → logits
  - Measure log-prob of (finish, search, execute, write, ...) tokens

Cost: ~10-15 min on a single GPU (RTX 6000 / A100 / L40S). Single Colab session,
NOT competing with Phase 6 trace collection.

Usage:
  python3 phase7_steering_micro_pilot.py --n-traces 8 --alphas -2 0 1 2

Output: phase7_steering_results.json on Drive.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DRIVE = Path('/content/drive/MyDrive/openinterp_runs')  # Colab path
if not DRIVE.exists():
    DRIVE = Path('/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
P1 = DRIVE / 'swebench_v1_phase1'
P5 = DRIVE / 'swebench_v5_phase5'
P6 = DRIVE / 'swebench_v6_phase6'
P6B = P6 / 'phase6b'
OUT = P6 / 'phase7_steering_pilot.json'

LAYER = 43
POSITION = 'pre_tool'
TOP_K = 10


def classify(r):
    if r.get('pull_failed'):
        return 'env_mismatch'
    c = r.get('conditions', {})
    n = c.get('none', {}).get('n_pass', 0)
    g = c.get('golden', {}).get('n_pass', 0)
    a = c.get('agent', {}).get('n_pass', 0)
    total = c.get('golden', {}).get('total', 0)
    if total == 0 or g <= n:
        return 'env_mismatch'
    if a >= g:
        return 'solves'
    if a > n:
        return 'partial'
    return 'fails'


def load_pos_mean(iid, layer, position):
    for root in [P6 / 'captures', P1 / 'captures']:
        if not root.exists():
            continue
        meta_glob = list(root.glob(f'{iid}*.meta.json'))
        if not meta_glob:
            continue
        meta_path = meta_glob[0]
        weights = meta_path.with_suffix('').with_suffix('.safetensors')
        if not weights.exists():
            continue
        m = json.loads(meta_path.read_text())
        t = load_file(str(weights))
        vecs = [t[r['activation_key']].to(torch.float32).numpy()
                for r in m['records']
                if r['layer'] == layer and r['position_label'] == position
                and r['activation_key'] in t]
        if vecs:
            return np.mean(np.stack(vecs, axis=0), axis=0)
    return None


def fit_probe_and_get_direction(verdicts):
    """Fit LR probe on top-10 features, return weighted direction unit vector."""
    valid_iids = [iid for iid, v in verdicts.items() if v != 'env_mismatch']
    Xs = []
    ys = []
    keep_iids = []
    for iid in valid_iids:
        v = load_pos_mean(iid, LAYER, POSITION)
        if v is None:
            continue
        Xs.append(v)
        ys.append(1 if verdicts[iid] == 'solves' else 0)
        keep_iids.append(iid)
    X = np.stack(Xs, axis=0)
    y = np.array(ys, dtype=int)
    print(f'Probe fit: N={len(keep_iids)}, pos={int(y.sum())}, neg={int((1-y).sum())}')

    # Top-K diff-of-means
    diff = np.abs(X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0))
    sel = np.argsort(-diff)[:TOP_K]

    sc = StandardScaler()
    X_sel = sc.fit_transform(X[:, sel])
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
    clf.fit(X_sel, y)

    # Build full-dim direction: weighted by LR coefs in selected dims, normalized
    direction = np.zeros(X.shape[1], dtype=np.float32)
    direction[sel] = clf.coef_.flatten() / sc.scale_  # undo standardization
    norm = np.linalg.norm(direction)
    direction /= norm
    print(f'Direction L2 norm before normalize: {norm:.3f}')
    print(f'Top-K feature indices: {sel.tolist()}')

    # Also return per-instance probe scores (in original feature scale)
    scores = clf.predict_proba(X_sel)[:, 1]
    iid_to_score = {iid: float(s) for iid, s in zip(keep_iids, scores)}
    return direction, sel, iid_to_score


def select_test_traces(verdicts, scores, n_fails=5, n_solves=3):
    """Pick traces with extreme scores: lowest among fails, highest among solves."""
    fails = [(iid, scores[iid]) for iid, v in verdicts.items()
             if v == 'fails' and iid in scores]
    solves = [(iid, scores[iid]) for iid, v in verdicts.items()
              if v == 'solves' and iid in scores]
    fails.sort(key=lambda x: x[1])  # lowest first
    solves.sort(key=lambda x: -x[1])  # highest first
    return [iid for iid, _ in fails[:n_fails]] + [iid for iid, _ in solves[:n_solves]]


def reconstruct_prompt_up_to_pre_tool(trace_path, tokenizer):
    """Replay the trace JSON to get the prompt token IDs ending at pre_tool position
    of the FIRST tool call in the FIRST turn. Returns input_ids (1, T) tensor."""
    with open(trace_path) as f:
        d = json.load(f)
    if not d.get('turns'):
        return None, None
    turn0 = d['turns'][0]
    # The captured prompt_tokens for turn 0 represents what the model saw
    # before generating turn 0. Add the model's thinking + the start of first
    # tool call to get up to pre_tool position.
    prompt_tokens = turn0.get('prompt_tokens', 0)
    raw = turn0.get('raw_response', '')
    # Use raw response up to first <tool_call> to define pre_tool position
    if '<tool_call>' in raw:
        prefix = raw.split('<tool_call>')[0] + '<tool_call>'
    else:
        return None, None
    # We need the tokenizer to actually rebuild the prompt; for the pilot we use
    # the captured input_ids if available; otherwise rebuild via decode/encode.
    # Simplification: re-encode the prefix.
    # NOTE: this assumes the trace's raw_response is decodable cleanly.
    full_text_at_pre_tool = raw[:raw.index('<tool_call>') + len('<tool_call>')]
    # We rebuild a chat-template string up to that point. For Qwen3.6 chat template
    # we need the system+user portion (lost in the trace JSON, must be rebuilt
    # from instance metadata). For pilot we approximate by treating the whole
    # raw_response prefix as the assistant content.
    return full_text_at_pre_tool, len(full_text_at_pre_tool)


def measure_logprob_shift(model, tokenizer, prompt_text, direction_t, alphas, device):
    """For a single prompt, measure log-prob of finish vs other tool tokens
    under different α modifications at L43 pre_tool position."""
    # Simplistic version: feed prompt + measure next-token log-prob distributions
    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
    target_token_strs = ['finish', 'search', 'execute', 'write', 'read', 'wait']
    target_token_ids = []
    for s in target_token_strs:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            target_token_ids.append((s, ids[0]))

    # Hook the L43 output, capture for α=0, then add α * direction at last position
    captured = {}
    def make_hook(alpha):
        def hook(module, inp, out):
            # out is tuple from transformer block; out[0] is hidden states (B, T, D)
            h = out[0]
            modified = h.clone()
            modified[:, -1, :] = h[:, -1, :] + alpha * direction_t
            return (modified,) + out[1:]
        return hook

    results_per_alpha = {}
    for alpha in alphas:
        hook = model.model.layers[LAYER].register_forward_hook(make_hook(alpha))
        try:
            with torch.no_grad():
                out = model(**inputs, use_cache=False)
            logits = out.logits[0, -1, :]  # (vocab_size,)
            log_probs = F.log_softmax(logits, dim=-1)
            results_per_alpha[alpha] = {
                s: float(log_probs[tid].item())
                for s, tid in target_token_ids
            }
        finally:
            hook.remove()

    return results_per_alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-traces', type=int, default=8)
    ap.add_argument('--alphas', type=float, nargs='+', default=[-2.0, 0.0, 1.0, 2.0])
    ap.add_argument('--model', default='Qwen/Qwen3.6-27B')
    args = ap.parse_args()

    print(f'=== Phase 7 micro-pilot: log-prob proxy at L{LAYER} {POSITION} ===')
    print(f'  alphas: {args.alphas}')
    print(f'  n_traces: {args.n_traces}')
    print(f'  drive: {DRIVE}')

    # Build verdict map
    p5a = json.loads((P5 / 'phase5a_results.json').read_text())
    p6b_path = P6B / 'phase6b_results.json'
    p6b = json.loads(p6b_path.read_text()) if p6b_path.exists() else {}
    verdicts = {iid: classify(r) for iid, r in p5a.items()}
    for iid, r in p6b.items():
        verdicts[iid] = classify(r)

    # Fit probe + get direction
    direction_np, top_idx, iid_to_score = fit_probe_and_get_direction(verdicts)
    print(f'\nProbe fit done. Direction shape: {direction_np.shape}')

    # Select test traces
    n_solves = max(2, args.n_traces // 3)
    n_fails = args.n_traces - n_solves
    test_iids = select_test_traces(verdicts, iid_to_score, n_fails=n_fails, n_solves=n_solves)
    print(f'\nSelected test traces: {len(test_iids)}')
    for iid in test_iids:
        print(f'  {verdicts[iid]:8s} score={iid_to_score[iid]:.3f}  {iid[:65]}')

    # Load model
    print(f'\nLoading {args.model}...')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map={'': 0}, trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    direction_t = torch.from_numpy(direction_np).to(device).to(torch.bfloat16)
    print(f'Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params')

    # Measure log-prob shifts
    all_results = {}
    for iid in test_iids:
        # Find the trace file
        trace_root = P6 / 'traces'
        if not trace_root.exists():
            trace_root = P1 / 'traces'
        trace_glob = list(trace_root.glob(f'{iid}*.json'))
        if not trace_glob:
            print(f'  TRACE MISSING: {iid[:60]}')
            continue
        trace_path = trace_glob[0]

        prompt_text, _ = reconstruct_prompt_up_to_pre_tool(trace_path, tok)
        if prompt_text is None:
            print(f'  NO TOOL CALL FOUND: {iid[:60]}')
            continue

        print(f'\n--- {iid[:60]} ({verdicts[iid]}) ---')
        try:
            results = measure_logprob_shift(model, tok, prompt_text, direction_t, args.alphas, device)
            all_results[iid] = {
                'verdict': verdicts[iid],
                'probe_score': iid_to_score.get(iid),
                'logprobs_per_alpha': {str(a): r for a, r in results.items()},
            }
            print(f'  α=0  {results[0.0]}')
            print(f'  α=+2 {results[2.0]}')
            shift_finish = results[2.0]['finish'] - results[0.0]['finish']
            print(f'  Δlog-prob(finish) at α=+2: {shift_finish:+.3f}')
        except Exception as e:
            print(f'  ERROR: {type(e).__name__}: {e}')

    # Aggregate
    print(f'\n=== AGGREGATE ===')
    fails_shifts = []
    solves_shifts = []
    for iid, r in all_results.items():
        lp = r['logprobs_per_alpha']
        if '0.0' not in lp or '2.0' not in lp:
            continue
        delta = lp['2.0']['finish'] - lp['0.0']['finish']
        if r['verdict'] == 'fails':
            fails_shifts.append(delta)
        elif r['verdict'] == 'solves':
            solves_shifts.append(delta)
    if fails_shifts:
        print(f'  FAILS  : Δlog-prob(finish) at α=+2: mean={np.mean(fails_shifts):+.3f}, n={len(fails_shifts)}')
    if solves_shifts:
        print(f'  SOLVES : Δlog-prob(finish) at α=+2: mean={np.mean(solves_shifts):+.3f}, n={len(solves_shifts)}')

    print(f'\nInterpretation:')
    print(f'  - If FAILS shift > 0 (especially > +0.5), probe direction is CAUSALLY pushing toward finish')
    print(f'  - If FAILS shift ~ 0, probe direction is epiphenomenal at this layer')
    print(f'  - If FAILS shift < 0, probe direction is REVERSED (probe reads "I will fail")')

    out_dict = {
        'layer': LAYER, 'position': POSITION, 'top_k': TOP_K,
        'alphas': args.alphas,
        'top_indices': top_idx.tolist(),
        'results': all_results,
        'aggregate': {
            'fails_mean_shift_finish_alpha2': float(np.mean(fails_shifts)) if fails_shifts else None,
            'solves_mean_shift_finish_alpha2': float(np.mean(solves_shifts)) if solves_shifts else None,
            'n_fails': len(fails_shifts), 'n_solves': len(solves_shifts),
        },
    }
    OUT.write_text(json.dumps(out_dict, indent=2))
    print(f'\nWrote {OUT}')


if __name__ == '__main__':
    main()
