#!/usr/bin/env python3
"""
Verdict-circuit Stage-0 (CPU, existing captures + TopK SAE). Pre-registered in
paper/verdict_circuit/PREREG_verdict_circuit.md.

Question (§13 first half): is there an interpretable SAE "task-done" feature, selected
from SUCCESS (finish-turn vs early), that WANDERING also activates (holds the verdict)
while LOCKED does not (genuinely not done)? Selection is SUCCESS-only; W/L are held-out.

Controls: turn-index confound, random-feature null, per-trajectory (no pseudo-replication),
and an SAE-usability reconstruction gate (the captures are per-position; the SAE is per-token).

--smoke plants a done-feature in synthetic FEATURE space and must recover it + reject a
length-only null before the real run is trusted.

Output: scripts/verdict_circuit_out/stage0_L{layer}.json
"""
import sys, json, re, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_rot_stage1 as s1   # reuse PHASE6 path + label csv

OUTDIR = Path(__file__).resolve().parent / "verdict_circuit_out"
SAE_REPO = "caiovicentino1/qwen36-27b-sae-fullstack"
KEY = re.compile(r"t(\d+)_(\w+?)_p(\d+)_L(\d+)$")
K_TOPK = 128
RNG = np.random.default_rng(20260602)
N_RAND_NULL = 500
DONE_MIN_FIRE_FRAC = 0.30     # done-feature must fire in >=30% of SUCCESS finals
TOPK_DONE = 20                # secondary aggregate
POS = "pre_tool"

# ---------------------------------------------------------------- SAE
def load_sae(layer):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(SAE_REPO, f"sae_L{layer}_latest.safetensors")
    sae = {}
    with safe_open(p, "pt") as f:
        for k in f.keys():
            sae[k] = np.ascontiguousarray(f.get_tensor(k).float().numpy())
    return sae  # W_enc[5120,40960] W_dec[40960,5120] b_enc[40960] b_dec[5120]

def encode(X, sae, want_recon=False):
    """TopK SAE: z = TopK_k(relu((X-b_dec)@W_enc + b_enc)). X:[n,5120] -> z:[n,40960].
    np.errstate: macOS Accelerate BLAS raises spurious FP flags on matmul even when the
    result is finite (verified: inputs have 0 nan/inf; outputs are finite). TopK feature
    SELECTION is invariant to a global input scale, so the firing-pattern analysis is robust
    to the capture-vs-training scale mismatch the FVU reveals; only magnitudes are affected."""
    with np.errstate(all="ignore"):
        z = (np.ascontiguousarray(X) - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(z, 0, out=z)
    k = K_TOPK
    idx = np.argpartition(z, -k, axis=1)[:, -k:]
    mask = np.zeros_like(z, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    z[~mask] = 0.0
    if want_recon:
        with np.errstate(all="ignore"):
            return z, z @ sae["W_dec"] + sae["b_dec"]
    return z

# ---------------------------------------------------------------- captures
def load_pre_tool(path, layer):
    """{turn: mean pre_tool residual [5120]} for the given layer."""
    by_turn = defaultdict(list)
    with safe_open(path, "pt") as f:
        for k in f.keys():
            m = KEY.match(k)
            if m and m.group(2) == POS and int(m.group(4)) == layer:
                by_turn[int(m.group(1))].append(f.get_tensor(k).float().numpy())
    return {t: np.mean(v, axis=0) for t, v in by_turn.items()}

def build_windows(layer, sae):
    """Per trajectory: final/early SAE-feature vectors at pre_tool. Returns dict by class."""
    labels = s1.load_labels()
    caps = sorted(s1.PHASE6.glob("captures/*.safetensors"))
    data = {"success": [], "wandering": [], "locked": []}
    recon_sample = []   # (x, recon) for G0
    for c in caps:
        iid = c.stem
        lab = labels.get(iid) or next((labels[k] for k in labels if iid.startswith(k) or k.startswith(iid)), None)
        if lab not in data:
            continue
        tv = load_pre_tool(c, layer)
        if len(tv) < 4:
            continue
        turns = sorted(tv)
        final_t = turns[-min(2, len(turns)):]
        early_t = turns[:min(3, len(turns))]
        Xf = np.stack([tv[t] for t in final_t]); Xe = np.stack([tv[t] for t in early_t])
        if len(recon_sample) < 200:
            zf, rf = encode(Xf, sae, want_recon=True)
            for xi, ri in zip(Xf, rf): recon_sample.append((xi, ri))
            zf_acts = zf
        else:
            zf_acts = encode(Xf, sae)
        ze_acts = encode(Xe, sae)
        data[lab].append({
            "iid": iid, "final_turn": float(turns[-1]),
            "final": zf_acts.mean(0), "early": ze_acts.mean(0),
        })
    return data, recon_sample

# ---------------------------------------------------------------- analysis (shared by smoke + real)
def auroc(a, b):
    """P(random a > random b), a/b 1-D arrays."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    from scipy.stats import rankdata
    s = np.concatenate([a, b]); r = rankdata(s)
    n1 = len(a)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(b))

def analyze(S_final, S_early, W_final, W_early, L_final, final_turns_W, final_turns_L):
    """All inputs [n, D] feature matrices (per-trajectory rows). Returns verdict dict."""
    D = S_final.shape[1]
    # ---- selection: done-feature from SUCCESS only (final vs early firing freq) ----
    fire_final = (S_final > 0).mean(0)            # firing freq at success final
    fire_early = (S_early > 0).mean(0)
    done_score = fire_final - fire_early
    eligible = fire_final >= DONE_MIN_FIRE_FRAC
    done_score_masked = np.where(eligible, done_score, -np.inf)
    if not np.any(np.isfinite(done_score_masked)):
        return {"verdict": "INCONCLUSIVE", "reason": "no SUCCESS-final feature fires in >=30% of finals"}
    primary = int(np.argmax(done_score_masked))
    topk = [int(i) for i in np.argsort(done_score_masked)[::-1][:TOPK_DONE] if np.isfinite(done_score_masked[i])]

    def feat(M, f): return np.asarray(M[:, f], float)
    # ---- P1: WANDERING final > early (paired) ----
    wf, we = feat(W_final, primary), feat(W_early, primary)
    try: p1_p = float(wilcoxon(wf, we, alternative="greater").pvalue)
    except Exception: p1_p = None
    # ---- P2: WANDERING final > LOCKED final ----
    lf = feat(L_final, primary)
    p2_p = float(mannwhitneyu(wf, lf, alternative="greater").pvalue)
    p2_auroc = float(auroc(wf, lf))
    # ---- C2: random-feature null on the W>L effect (AUROC) ----
    rand = RNG.choice(D, size=min(N_RAND_NULL, D), replace=False)
    rand_aurocs = np.array([auroc(feat(W_final, r), feat(L_final, r)) for r in rand])
    null_95 = float(np.percentile(rand_aurocs, 95))
    beats_null = bool(p2_auroc > null_95)
    # ---- C1: turn-index control (W vs L final activation, partial out final-turn) ----
    y = np.r_[np.ones(len(wf)), np.zeros(len(lf))]
    act = np.r_[wf, lf]
    turn = np.r_[np.asarray(final_turns_W, float), np.asarray(final_turns_L, float)]
    A = np.c_[np.ones(len(turn)), turn]
    def resid(v):
        b, *_ = np.linalg.lstsq(A, v, rcond=None); return v - A @ b
    ra, ry = resid(act), resid(y)
    partial_r = float(np.corrcoef(ra, ry)[0, 1]) if np.std(ra) > 1e-9 and np.std(ry) > 1e-9 else 0.0
    survives_turn = bool(partial_r > 0.10)
    # ---- secondary: top-K aggregate W vs L ----
    agg_w = W_final[:, topk].mean(1); agg_l = L_final[:, topk].mean(1)
    agg_auroc = float(auroc(agg_w, agg_l))

    p1_ok = bool(p1_p is not None and p1_p < 0.05)
    p2_ok = bool(p2_p < 0.05)
    go = bool(p1_ok and p2_ok and beats_null and survives_turn)
    return {
        "verdict": "GO" if go else "NO-GO (honest negative)",
        "primary_feature": primary, "done_score": float(done_score[primary]),
        "P1_wander_final_gt_early_p": p1_p, "P1_ok": p1_ok,
        "P2_wander_final_gt_locked_p": p2_p, "P2_auroc": p2_auroc, "P2_ok": p2_ok,
        "C2_randfeat_null_auroc95": null_95, "C2_beats_null": beats_null,
        "C1_partial_r_given_turn": partial_r, "C1_survives_turn": survives_turn,
        "topK_aggregate_auroc": agg_auroc, "topK": topk[:8],
        "means": {"S_final": float(feat(S_final, primary).mean()),
                  "W_final": float(wf.mean()), "W_early": float(we.mean()),
                  "L_final": float(lf.mean())},
    }

# ---------------------------------------------------------------- smoke (feature-space synthetic)
def build_synth(planted=True):
    D = 500; f0 = 137
    def vec(active_done, n_rows=2):
        v = np.zeros(D)
        bg = RNG.choice(D, 8, replace=False); v[bg] = RNG.uniform(0.2, 1.0, len(bg))
        if active_done: v[f0] = RNG.uniform(1.5, 2.5)
        return v
    S_final = np.stack([vec(True) for _ in range(40)])
    S_early = np.stack([vec(False) for _ in range(40)])
    # planted: done-feature ALSO on in wandering-final, OFF in locked-final
    W_final = np.stack([vec(planted) for _ in range(20)])
    W_early = np.stack([vec(False) for _ in range(20)])
    L_final = np.stack([vec(False) for _ in range(39)])
    # null arm differs only in final-turn index (W longer), feature process identical
    ftW = RNG.integers(30, 50, 20).astype(float)
    ftL = RNG.integers(5, 15, 39).astype(float)
    return S_final, S_early, W_final, W_early, L_final, ftW, ftL

def smoke():
    print("=== SMOKE planted (gate MUST fire) ===")
    r = analyze(*build_synth(planted=True)); print(json.dumps({k: r[k] for k in ("verdict","primary_feature","P1_ok","P2_ok","C2_beats_null","C1_survives_turn")}, indent=1))
    ok_go = r["verdict"] == "GO" and r["primary_feature"] == 137
    print("=== SMOKE null (gate MUST NOT fire) ===")
    r2 = analyze(*build_synth(planted=False)); print(json.dumps({k: r2[k] for k in ("verdict","P2_ok","C2_beats_null")}, indent=1))
    ok_null = r2["verdict"].startswith("NO-GO")
    print("SMOKE PASS" if (ok_go and ok_null) else "SMOKE FAIL", file=sys.stderr)
    sys.exit(0 if (ok_go and ok_null) else 1)

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--layer", type=int, default=43)
    args = ap.parse_args()
    if args.smoke: smoke()
    if not s1.PHASE6.exists(): sys.exit(f"Phase-6 not found at {s1.PHASE6}")

    print(f"loading SAE L{args.layer} ...", file=sys.stderr)
    sae = load_sae(args.layer)
    print("encoding captures (pre_tool) ...", file=sys.stderr)
    data, recon_sample = build_windows(args.layer, sae)
    n = {k: len(v) for k, v in data.items()}
    print("n by class:", n, file=sys.stderr)

    # G0 SAE-usability gate
    xs = np.stack([x for x, _ in recon_sample]); rs = np.stack([r for _, r in recon_sample])
    cos = (xs * rs).sum(1) / (np.linalg.norm(xs, axis=1) * np.linalg.norm(rs, axis=1) + 1e-9)
    fvu = ((xs - rs) ** 2).sum(1) / (((xs - xs.mean(0)) ** 2).sum(1) + 1e-9)
    g0 = {"median_recon_cosine": float(np.median(cos)), "median_FVU": float(np.median(fvu)),
          "usable": bool(np.median(cos) >= 0.5)}
    print("G0 SAE-usability:", g0, file=sys.stderr)

    def mat(cls, key): return np.stack([d[key] for d in data[cls]])
    res = analyze(
        mat("success", "final"), mat("success", "early"),
        mat("wandering", "final"), mat("wandering", "early"), mat("locked", "final"),
        [d["final_turn"] for d in data["wandering"]], [d["final_turn"] for d in data["locked"]],
    )
    if not g0["usable"]:
        res = {"verdict": "INCONCLUSIVE (G0: SAE cannot read these captures -> need per-token re-capture)",
               "G0": g0, "stage0_if_usable": res}
    else:
        res["G0"] = g0
    res["layer"] = args.layer; res["n_by_class"] = n
    OUTDIR.mkdir(exist_ok=True)
    out = OUTDIR / f"stage0_L{args.layer}.json"; out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in res if k not in ("stage0_if_usable",)}, indent=2))
    print("full ->", out)

if __name__ == "__main__":
    main()
