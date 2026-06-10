#!/usr/bin/env python3
"""Local driver: runs the commitment-lever Tier-1 cell-by-cell on Colab CLI sessions,
verifying every step, resuming from the HF ledger across VM deaths (keep-alive 403 bug
kills VMs every ~25-40 min — this loop just reprovisions and continues).
"""
import json, subprocess, sys, time

DATA_REPO = "caiovicentino1/swebench-phase6-verdict-circuit"
BLOCKS = ["h1_baseline", "h1_editdonor_L55", "h1_bashnull_L55", "h1_crosstask_L55",
          "h1_editdonor_L59", "h1_bashnull_L59", "h1_crosstask_L59",
          "h2_baseline", "h2_suppress_L55", "h2_suppress_bashrate_L55", "h2_ctl_L55",
          "h2_suppress_L59", "h2_suppress_bashrate_L59", "h2_ctl_L59"]
MAX_WALL = 4 * 3600

def log(*a): print(f"[driver {time.strftime('%H:%M:%S')}]", *a, flush=True)

def sh(cmd, input_text=None, timeout=None):
    try:
        r = subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "local timeout"

def hf_token():
    from huggingface_hub import get_token
    return get_token()

def ledger():
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(DATA_REPO, "results/commit_lever_partial.json", repo_type="dataset",
                            token=hf_token(), force_download=True)
        return json.load(open(p))
    except Exception:
        return {}

def pending_blocks(L):
    done = set((L.get("blocks") or {}).keys())
    return [b for b in BLOCKS if b not in done]

def ex(sess, code, timeout):
    return sh(["colab", "exec", "-s", sess, "--timeout", str(timeout)], input_text=code, timeout=timeout + 120)

def provision(sess):
    for gpu in ("G4", "G4", "A100"):
        rc, out = sh(["colab", "new", "--gpu", gpu, "-s", sess], timeout=600)
        if "READY" in out:
            log(f"provisioned {gpu} ({sess})"); return gpu
        log(f"  {gpu} failed ({'TooMany' if 'TooMany' in out else 'unavail'}) — retrying"); time.sleep(20)
    return None

def setup_session(sess):
    rc, out = ex(sess, (
        "import subprocess\n"
        "subprocess.run(['pip','-q','install','git+https://github.com/huggingface/transformers.git',"
        "'safetensors','huggingface_hub','datasets','accelerate'],check=True)\n"
        "subprocess.run(['git','clone','-q','https://github.com/OpenInterpretability/openinterp-swebench-harness',"
        "'/content/openinterp-swebench-harness'])\n"
        "print('CELL1_OK')"), 700)
    if "CELL1_OK" not in out:
        log("cell1 FAILED:", out.strip().splitlines()[-1:] if out.strip() else rc); return False
    t = hf_token()
    rc, out = ex(sess, (
        f"import os, sys\n"
        f"os.environ['HF_TOKEN']={t!r}; os.environ['HUGGING_FACE_HUB_TOKEN']={t!r}\n"
        f"sys.path.insert(0,'/content/openinterp-swebench-harness/scripts')\n"
        f"import commit_lever_stages as S\nS.setup()"), 1800)
    if "SETUP_OK" not in out:
        log("setup FAILED:", out.strip().splitlines()[-1:] if out.strip() else rc); return False
    log("SETUP_OK")
    rc, out = ex(sess, "import commit_lever_stages as S\nS.capture()", 1800)
    if "CAPTURE_OK" not in out and "CAPTURE_RESUMED" not in out:
        log("capture FAILED:", out.strip().splitlines()[-2:] if out.strip() else rc); return False
    log("capture ready (" + ("resumed" if "RESUMED" in out else "fresh, uploaded") + ")")
    return True

def main():
    t0 = time.time(); n = 6
    while time.time() - t0 < MAX_WALL:
        L = ledger()
        todo = pending_blocks(L)
        need_dense = not L.get("dense_dP")
        if not todo and not need_dense:
            log("ALL BLOCKS + DENSE DONE — finalizing")
            sess = f"lever{n}"
            if provision(sess) and setup_session(sess):
                ex(sess, "import commit_lever_stages as S\nS.finalize()", 600)
                sh(["colab", "stop", "-s", sess], timeout=120)
            log("DRIVER_DONE"); return
        log(f"pending: {len(todo)} blocks{' + dense' if need_dense else ''} -> {todo[:4]}...")
        sess = f"lever{n}"; n += 1
        if not provision(sess):
            log("no GPU available; sleeping 5 min"); time.sleep(300); continue
        try:
            if not setup_session(sess):
                continue
            for b in todo:
                rc, out = ex(sess, f"import commit_lever_stages as S\nS.block({b!r})", 1500)
                if f"BLOCK_OK {b}" in out or "BLOCK_SKIP" in out:
                    line = [l for l in out.splitlines() if "BLOCK_OK" in l or "BLOCK_SKIP" in l]
                    log(line[-1] if line else f"{b} ok")
                else:
                    log(f"block {b} FAILED/session died -> reprovision"); break
            else:
                if need_dense:
                    rc, out = ex(sess, "import commit_lever_stages as S\nS.dense()", 1800)
                    log("dense:", "OK" if "DENSE_OK" in out else "FAILED")
        finally:
            sh(["colab", "stop", "-s", sess], timeout=120)
    log("MAX_WALL reached")

if __name__ == "__main__":
    main()
