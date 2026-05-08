"""Colab cell: end-to-end agent-probe-guard latency + accuracy eval on GPU.

Paste this into a Colab cell after running cells 0-5 of nb_swebench_v9_phase8
(model + tok + device + test_data + train_pool + build_messages all defined).
Or after the Phase 8 redux bootstrap in the current session.

Measures:
  1. GPU forward pass latency (model + 2 hooks + scaler + LR) per call
  2. AUROC on held-out N=240 prompts via the SDK API
  3. Decision distribution at default thresholds

Total compute: ~3 minutes for N=240 prompts at <2k tokens each.
"""
COLAB_CELL = '''# === END-TO-END AGENT-PROBE-GUARD EVAL (paste in Colab after model loaded) ===
import json, time
import numpy as np
import torch
from pathlib import Path
from collections import Counter
import joblib
from sklearn.metrics import roc_auc_score

# 1) Load artifact (assumes it has been uploaded to HF or copied to Drive)
ART = Path(DRIVE_ROOT) / "agent_probe_guard_qwen36_27b"  # local copy on Drive
if not ART.exists():
    print(f"Artifact not on Drive at {ART}, falling back to git clone")
    !git clone --depth=1 https://github.com/OpenInterpretability/openinterp-swebench-harness /tmp/sw 2>&1 | tail -3
    ART = Path("/tmp/sw/artifacts/agent_probe_guard_qwen36_27b")

meta = json.loads((ART / "meta.json").read_text())
thk_art = joblib.load(ART / "probe_L55_thinking.joblib")
cap_art = joblib.load(ART / "probe_L43_pre_tool.joblib")
print(f"Loaded artifact v{meta[\\"version\\"]} from {ART}")
print(f"  L55 thinking: K={len(thk_art[\\"dims\\"])}")
print(f"  L43 capability: K={len(cap_art[\\"dims\\"])}")


# 2) Build hooks for L43 + L55
class TwoProbeGate:
    def __init__(self, model, tok, thk_art, cap_art):
        self.model = model
        self.tok = tok
        self.thk = thk_art
        self.cap = cap_art
        self.buf = {}
        self.hooks = []
        for layer in (cap_art["layer"], thk_art["layer"]):
            h = model.model.layers[layer].register_forward_hook(self._mk(layer))
            self.hooks.append(h)

    def _mk(self, L):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            self.buf[L] = h.detach()
        return hook

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def assess(self, prompt_text):
        enc = self.tok(prompt_text, return_tensors="pt", add_special_tokens=False,
                       truncation=True, max_length=4096).to(self.model.device)
        with torch.no_grad():
            self.buf = {}
            self.model(**enc, use_cache=False)
        last = enc["attention_mask"].sum(dim=1).item() - 1

        h_thk = self.buf[self.thk["layer"]][0, last].float().cpu().numpy().reshape(1, -1)
        h_thk_k = h_thk[:, self.thk["dims"]]
        s_thk = float(self.thk["probe"].predict_proba(self.thk["scaler"].transform(h_thk_k))[0, 1])

        h_cap = self.buf[self.cap["layer"]][0, last].float().cpu().numpy().reshape(1, -1)
        h_cap_k = h_cap[:, self.cap["dims"]]
        s_cap = float(self.cap["probe"].predict_proba(self.cap["scaler"].transform(h_cap_k))[0, 1])

        thr = meta["thresholds"]
        if s_cap < thr["skip_below"]:
            action = "skip"
        elif s_cap < thr["escalate_below"]:
            action = "escalate"
        else:
            action = "proceed"
        return {"action": action, "thinking": s_thk, "capability": s_cap}


gate = TwoProbeGate(model, tok, thk_art, cap_art)


# 3) Eval on N=240 nb47b prompts
test_by_id = {d["id"]: d for d in test_data}
NB47B = Path(DRIVE_ROOT) / "nb47b_capture"
nb47b_meta = json.loads((NB47B / "metadata.json").read_text())
nb47b_records = nb47b_meta["records"]
y = np.array([1 if r["has_think_v1"] else 0 for r in nb47b_records], dtype=int)

scores_thk, scores_cap, actions, latencies = [], [], [], []
t_start = time.time()

for i, rec in enumerate(nb47b_records):
    src = test_by_id.get(rec["id"])
    if src is None:
        continue
    msgs = build_messages(src["question"], src.get("memories_used", []), train_pool)
    p_text = prompt_text_no_autothink(msgs)

    t0 = time.perf_counter()
    out = gate.assess(p_text)
    latencies.append((time.perf_counter() - t0) * 1000)

    scores_thk.append(out["thinking"])
    scores_cap.append(out["capability"])
    actions.append(out["action"])

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(nb47b_records) - i - 1)
        print(f"  [{i+1:3d}/{len(nb47b_records)}] elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

print()
print("=== END-TO-END EVAL RESULTS ===")
print(f"N = {len(scores_thk)}")
print(f"thinking AUROC: {roc_auc_score(y[:len(scores_thk)], scores_thk):.4f}")
print(f"capability AUROC: {roc_auc_score(y[:len(scores_cap)], scores_cap):.4f}")
counter = Counter(actions)
n = len(actions)
print(f"\\nDecision distribution at default thresholds:")
for a in ("skip", "escalate", "proceed"):
    print(f"  {a:10s}: {counter.get(a, 0):>3d}/{n} ({counter.get(a, 0)/n*100:.1f}%)")

lat = np.array(latencies)
print(f"\\nFull assess() latency (GPU forward + 2 probes):")
print(f"  p50:  {np.percentile(lat, 50):.1f} ms")
print(f"  p95:  {np.percentile(lat, 95):.1f} ms")
print(f"  p99:  {np.percentile(lat, 99):.1f} ms")
print(f"  mean: {lat.mean():.1f} ms")

gate.detach()

# Compare against local sklearn-only eval
print("\\nSanity (must match local eval):")
print(f"  thinking AUROC expected ≈ 0.855")
print(f"  capability AUROC expected ≈ 0.863")
print(f"  skip% expected ≈ 21.2%, proceed% expected ≈ 48.8%")
'''

if __name__ == '__main__':
    print(COLAB_CELL)
