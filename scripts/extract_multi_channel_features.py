"""
Multi-Channel Feature Extraction (Paper #3 — Multi-Channel Mechanistic Signatures of WANDERING).

Single entry point: `extract_all_features(iid, trace_path, safetensors_path, precomputed)`
returns one flat dict (~50 features) across 4 channels:

  Channel 1 — TEXT       : doubt/exploration/completion word density, thinking/content lengths
  Channel 2 — TOOL-USE   : Shannon entropy (full/first10/last10), diversity, switching, max-run
  Channel 3 — RESIDUAL   : per-layer norm/cosine/drift @ L11/23/31/43/55
  Channel 4 — TEMPORAL   : n_turns, patch_n_bytes, probe-score dynamics, detector fire turns

Defensive: any missing/corrupt input -> NaN, never raises. Single safetensors load
serves all 5 layers (v6_residual_stability.py loaded 5x; we don't).

Tested on 2 trajectories at __main__: one SUCCESS, one WANDERING (from inflection_results.json).
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import safetensors.torch as st

# --- Constants -------------------------------------------------------------
LAYERS = [11, 23, 31, 43, 55]
POSITION = "pre_tool"
LATE_WINDOW = 10  # "last-N turns" window for late-half features
TENSOR_NAME_RE = re.compile(r"t(\d+)_(\w+?)_p(\d+)_L(\d+)(?:__dup\d+)?$")

# Lexical patterns (case-insensitive substring); calibrated against v1 forensic priors
DOUBT_TERMS = (
    "maybe", "i'm not sure", "im not sure", "let me try",
    "the issue is", "wait,", "actually", "perhaps", "i think the",
    "not sure", "hmm",
)
EXPLORATION_TERMS = (
    "let me look", "let me check", "let me search", "let me see",
    "let me find", "let me read", "let me run", "let me verify",
    "try", "check", "search", "investigate", "explore",
)
COMPLETION_TERMS = (
    "done", "complete", "completed", "fixed", "ready", "submitting",
    "all set", "finished", "the fix", "should work", "should be fixed",
)

# --- Small numeric utilities (defensive) ----------------------------------
def _safe_mean(xs):
    if not xs:
        return float("nan")
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _safe_std(xs):
    if not xs or len(xs) < 2:
        return float("nan")
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if arr.size >= 2 else float("nan")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _shannon(names: list[str]) -> float:
    if not names:
        return 0.0
    c = Counter(names)
    total = sum(c.values())
    return float(-sum((cnt / total) * math.log2(cnt / total) for cnt in c.values() if cnt > 0))


# --- CHANNEL 1: TEXT ------------------------------------------------------
def _join_turn_text(turn) -> str:
    return (turn.get("content") or "") + "\n" + (turn.get("thinking") or "")


def _word_density(turns, terms) -> float:
    """Fraction of turns whose joined text contains ANY of the given terms (case-insensitive)."""
    if not turns:
        return float("nan")
    hits = 0
    for t in turns:
        txt = _join_turn_text(t).lower()
        if any(term in txt for term in terms):
            hits += 1
    return hits / len(turns)


def _ngram_overlap(texts: list[str], n: int = 3) -> float:
    """Mean Jaccard overlap of n-gram sets between consecutive texts. NaN if <2 texts."""
    if len(texts) < 2:
        return float("nan")
    def grams(s):
        toks = re.findall(r"\w+", s.lower())
        return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}
    sims = []
    for i in range(1, len(texts)):
        a, b = grams(texts[i - 1]), grams(texts[i])
        if not a or not b:
            continue
        sims.append(len(a & b) / len(a | b))
    return _safe_mean(sims)


def extract_text_features(trace) -> dict:
    turns = trace.get("turns") or []
    if not turns:
        # All NaN
        return {
            "text_doubt_density": float("nan"),
            "text_exploration_density": float("nan"),
            "text_completion_density": float("nan"),
            "text_thinking_length_mean": float("nan"),
            "text_thinking_length_std": float("nan"),
            "text_thinking_length_final": float("nan"),
            "text_content_length_mean": float("nan"),
            "text_content_length_final": float("nan"),
            "text_thinking_to_content_ratio_mean": float("nan"),
            "text_doubt_density_late": float("nan"),
            "text_ngram_overlap_late": float("nan"),
        }

    late = turns[-LATE_WINDOW:] if len(turns) >= 1 else []
    think_lens = [len(t.get("thinking") or "") for t in turns]
    content_lens = [len(t.get("content") or "") for t in turns]
    ratios = [
        (len(t.get("thinking") or "") / max(1, len(t.get("content") or ""))) for t in turns
    ]
    late_texts = [_join_turn_text(t) for t in late]

    return {
        "text_doubt_density": _word_density(turns, DOUBT_TERMS),
        "text_exploration_density": _word_density(turns, EXPLORATION_TERMS),
        "text_completion_density": _word_density(turns, COMPLETION_TERMS),
        "text_thinking_length_mean": _safe_mean(think_lens),
        "text_thinking_length_std": _safe_std(think_lens),
        "text_thinking_length_final": float(think_lens[-1]) if think_lens else float("nan"),
        "text_content_length_mean": _safe_mean(content_lens),
        "text_content_length_final": float(content_lens[-1]) if content_lens else float("nan"),
        "text_thinking_to_content_ratio_mean": _safe_mean(ratios),
        "text_doubt_density_late": _word_density(late, DOUBT_TERMS),
        "text_ngram_overlap_late": _ngram_overlap(late_texts, n=3),
    }


# --- CHANNEL 2: TOOL-USE --------------------------------------------------
def _tool_names_per_turn(turns) -> list[list[str]]:
    out = []
    for t in turns:
        names = []
        for tc in (t.get("tool_calls") or []):
            if isinstance(tc, dict):
                names.append(tc.get("name", "unknown"))
        out.append(names)
    return out


def extract_tool_features(trace) -> dict:
    turns = trace.get("turns") or []
    per_turn = _tool_names_per_turn(turns)
    flat_all = [n for turn_names in per_turn for n in turn_names]
    flat_first10 = [n for turn_names in per_turn[:10] for n in turn_names]
    flat_last10 = [n for turn_names in per_turn[-10:] for n in turn_names]

    # First-call-per-turn sequence (for switching/run-length)
    first_per_turn = [tn[0] for tn in per_turn if tn]

    # tool_switching_rate: transitions per turn (between consecutive turns that both have a tool)
    switches = 0
    transitions = 0
    for i in range(1, len(first_per_turn)):
        transitions += 1
        if first_per_turn[i] != first_per_turn[i - 1]:
            switches += 1
    switching_rate = (switches / transitions) if transitions > 0 else float("nan")

    # Max consecutive same-tool run (on first-call-per-turn sequence)
    max_run = 0
    cur_run = 0
    prev = None
    for name in first_per_turn:
        if name == prev:
            cur_run += 1
        else:
            cur_run = 1
            prev = name
        max_run = max(max_run, cur_run)

    diversity = len(set(flat_all))
    top_frac = (max(Counter(flat_all).values()) / len(flat_all)) if flat_all else float("nan")
    calls_per_turn = [len(tn) for tn in per_turn]

    return {
        "tool_entropy_full": _shannon(flat_all) if flat_all else float("nan"),
        "tool_entropy_first10": _shannon(flat_first10) if flat_first10 else float("nan"),
        "tool_entropy_last10": _shannon(flat_last10) if flat_last10 else float("nan"),
        "tool_diversity_count": float(diversity),
        "tool_top_fraction": top_frac,
        "tool_switching_rate": switching_rate,
        "tool_repetition_max_run": float(max_run),
        "tool_calls_per_turn_mean": _safe_mean(calls_per_turn),
        "tool_total_calls": float(len(flat_all)),
    }


# --- CHANNEL 3: RESIDUAL --------------------------------------------------
def load_per_turn_residuals_all_layers(path: Path) -> dict[int, dict[int, np.ndarray]]:
    """Single safetensors load -> {layer: {turn: mean-pooled residual (np.float32)}}.

    Mean-pools over all token positions captured for a given (turn, layer, position=pre_tool).
    Returns empty dict on any failure.
    """
    out: dict[int, dict[int, list[np.ndarray]]] = {L: defaultdict(list) for L in LAYERS}
    try:
        tensors = st.load_file(str(path))
    except Exception:
        return {}
    for name, t in tensors.items():
        m = TENSOR_NAME_RE.match(name)
        if not m:
            continue
        turn, pos, _, layer = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        if pos != POSITION or layer not in LAYERS:
            continue
        out[layer][turn].append(t.float().numpy())
    # Reduce list -> single mean
    reduced: dict[int, dict[int, np.ndarray]] = {}
    for L, by_turn in out.items():
        reduced[L] = {turn: np.mean(np.stack(vs), axis=0) for turn, vs in by_turn.items() if vs}
    return reduced


def _residual_layer_features(by_turn: dict[int, np.ndarray], layer: int) -> dict:
    """Per-layer features. If <1 turn, all NaN; if <2 turns, cosine fields NaN."""
    prefix = f"L{layer}"
    if not by_turn:
        return {
            f"{prefix}_norm_mean": float("nan"),
            f"{prefix}_norm_std": float("nan"),
            f"{prefix}_norm_late_mean": float("nan"),
            f"{prefix}_cosine_consec_mean": float("nan"),
            f"{prefix}_cosine_consec_late": float("nan"),
            f"{prefix}_drift_first_last": float("nan"),
        }
    turns_sorted = sorted(by_turn.keys())
    residuals = [by_turn[t] for t in turns_sorted]
    norms = [float(np.linalg.norm(r)) for r in residuals]
    late_norms = norms[-LATE_WINDOW:]

    cos_consec = []
    for i in range(1, len(residuals)):
        cos_consec.append(_cosine(residuals[i - 1], residuals[i]))
    late_cos = cos_consec[-LATE_WINDOW:] if cos_consec else []

    drift = _cosine(residuals[0], residuals[-1]) if len(residuals) >= 2 else float("nan")

    return {
        f"{prefix}_norm_mean": _safe_mean(norms),
        f"{prefix}_norm_std": _safe_std(norms),
        f"{prefix}_norm_late_mean": _safe_mean(late_norms),
        f"{prefix}_cosine_consec_mean": _safe_mean(cos_consec),
        f"{prefix}_cosine_consec_late": _safe_mean(late_cos),
        f"{prefix}_drift_first_last": drift,
    }


def extract_residual_features(safetensors_path: Path) -> dict:
    """5 features × 5 layers = 25 residual features.

    Plus 2 cross-layer late-window features computed from per-turn residuals
    (operate on layer-stack at each turn, averaged over last-10).
    """
    by_layer = load_per_turn_residuals_all_layers(safetensors_path)

    out: dict = {}
    for L in LAYERS:
        out.update(_residual_layer_features(by_layer.get(L, {}), L))

    # Cross-layer features (computed inline; v4 cache provides PROBE-based versions,
    # these are RAW-residual versions for backup / orthogonal signal).
    # For each turn present at ALL 5 layers, compute range/sign-disagreement of NORMS.
    all_turns = sorted(set.intersection(*[set(by_layer.get(L, {}).keys()) for L in LAYERS])) \
        if all(by_layer.get(L) for L in LAYERS) else []
    if all_turns:
        per_turn_norms = []  # shape (n_turns, 5)
        for tn in all_turns:
            per_turn_norms.append([float(np.linalg.norm(by_layer[L][tn])) for L in LAYERS])
        arr = np.asarray(per_turn_norms)  # (T, 5)
        late = arr[-LATE_WINDOW:]
        out["cross_layer_norm_range_late_mean"] = float(np.mean(late.max(axis=1) - late.min(axis=1)))
        # "Sign disagreement" on raw residuals doesn't make sense without a probe,
        # but we can compute CV (std/mean) of layer norms per turn as a layer-coupling proxy.
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(late.mean(axis=1) > 0, late.std(axis=1) / late.mean(axis=1), np.nan)
        out["cross_layer_norm_cv_late_mean"] = float(np.nanmean(cv)) if np.isfinite(cv).any() else float("nan")
    else:
        out["cross_layer_norm_range_late_mean"] = float("nan")
        out["cross_layer_norm_cv_late_mean"] = float("nan")

    return out


# --- CHANNEL 4: TEMPORAL --------------------------------------------------
def extract_temporal_features(trace, precomputed: dict) -> dict:
    """Pulls from trace (n_turns, patch_n_bytes) + precomputed caches.

    precomputed expects:
      - inflection_entry: dict | None  (from inflection_results.json['per_trajectory'])
      - v4_entry: dict | None          (from early_warning_v4_cross_layer.json['per_trajectory'])
      - phase6_entry: dict | None      (from phase6_results.json[iid], for patch_n_bytes)

    Detector fire turns (v1, v5) are computed locally on the trace using the
    detectors module if available; falls back gracefully if import fails.
    """
    n_turns = trace.get("n_turns") or len(trace.get("turns") or []) or 0
    phase6 = precomputed.get("phase6_entry") or {}
    inflection = precomputed.get("inflection_entry") or {}
    v4 = precomputed.get("v4_entry") or {}

    patch_n_bytes = phase6.get("patch_n_bytes")
    patch_n_bytes = float(patch_n_bytes) if patch_n_bytes is not None else float("nan")

    # Probe score features (from inflection_results)
    score_traj = inflection.get("score_trajectory") or {}
    # keys may be str (json) — normalize
    score_items = sorted([(int(k), float(v)) for k, v in score_traj.items()]) if score_traj else []
    scores = [v for _, v in score_items]
    late_scores = scores[-LATE_WINDOW:] if scores else []

    first_cross = float("nan")
    for turn_idx, s in score_items:
        if s > 0.5:
            first_cross = float(turn_idx)
            break

    # Detector fire turns — use local detectors module if importable
    v1_fire = float("nan")
    v5_fire = float("nan")
    v4_fire = float("nan")
    try:
        sys.path.insert(0, str(Path("/Volumes/SSD Major/fish/inspect-tool-entropy-collapse/src").resolve()))
        from tool_entropy_collapse.detectors import v1_forensic, v5_tool_entropy  # type: ignore

        r1 = v1_forensic(trace)
        if r1.get("fire_turn") is not None:
            v1_fire = float(r1["fire_turn"])
        r5 = v5_tool_entropy(trace, threshold=0.8, window=10)
        if r5.get("fire_turn") is not None:
            v5_fire = float(r5["fire_turn"])
    except Exception:
        pass

    # v4 fire turn from cached ranges_per_turn (paper threshold 0.30)
    rpt = v4.get("ranges_per_turn") or []
    if rpt:
        # Cast (json may store strings)
        rpt_f = [float(x) for x in rpt]
        late_start = len(rpt_f) // 2
        for i in range(late_start, len(rpt_f)):
            if rpt_f[i] > 0.30:
                v4_fire = float(i)
                break

    return {
        "n_turns": float(n_turns),
        "patch_n_bytes": patch_n_bytes,
        "probe_score_final": float(scores[-1]) if scores else float("nan"),
        "probe_score_first": float(scores[0]) if scores else float("nan"),
        "probe_score_trajectory_mean": _safe_mean(scores),
        "probe_score_trajectory_late_mean": _safe_mean(late_scores),
        "probe_score_volatility": _safe_std(scores),
        "probe_first_threshold_cross_turn": first_cross,
        "detector_v1_fire_turn": v1_fire,
        "detector_v4_fire_turn": v4_fire,
        "detector_v5_fire_turn": v5_fire,
    }


# --- TOP-LEVEL ------------------------------------------------------------
def extract_all_features(
    iid: str,
    trace_path: Path,
    safetensors_path: Path,
    precomputed: dict,
) -> dict:
    """Returns a flat dict ~50 features. Defensive: loads everything in try/except."""
    row = {"iid": iid}

    try:
        trace = json.loads(Path(trace_path).read_text())
    except Exception as e:
        # Return iid + NaN-only row if trace unreadable
        row["__error__"] = f"trace_load_failed: {e}"
        return row

    # Inject instance_id into trace for detectors that look it up
    trace.setdefault("instance_id", iid)

    try:
        row.update(extract_text_features(trace))
    except Exception as e:
        row["__text_error__"] = str(e)

    try:
        row.update(extract_tool_features(trace))
    except Exception as e:
        row["__tool_error__"] = str(e)

    try:
        row.update(extract_residual_features(Path(safetensors_path)))
    except Exception as e:
        row["__residual_error__"] = str(e)

    try:
        row.update(extract_temporal_features(trace, precomputed))
    except Exception as e:
        row["__temporal_error__"] = str(e)

    return row


# --- Self-test ------------------------------------------------------------
PHASE6 = Path(
    "/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6"
)
INFLECT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")


def _build_precomputed(iid: str, inflection_doc, v4_doc, phase6_doc) -> dict:
    inflection_entry = next((t for t in inflection_doc["per_trajectory"] if t["iid"] == iid), None)
    v4_entry = next((t for t in v4_doc["per_trajectory"] if t["iid"] == iid), None)
    phase6_entry = phase6_doc.get(iid)
    return {
        "inflection_entry": inflection_entry,
        "v4_entry": v4_entry,
        "phase6_entry": phase6_entry,
    }


def _pick_test_iids(inflection_doc):
    succ = next(t["iid"] for t in inflection_doc["per_trajectory"] if t["label"] == 1)
    wand = next(
        t["iid"] for t in inflection_doc["per_trajectory"]
        if t["label"] == 0 and t.get("lock_fail_0.40") is None
    )
    return succ, wand


def _validate(row: dict) -> dict:
    """Returns counts: total, nan, inf, non_numeric."""
    total = 0
    nan = []
    inf = []
    non_numeric = []
    for k, v in row.items():
        if k == "iid" or k.startswith("__"):
            continue
        total += 1
        if isinstance(v, (int, float)):
            if isinstance(v, float):
                if math.isnan(v):
                    nan.append(k)
                elif math.isinf(v):
                    inf.append(k)
        else:
            non_numeric.append((k, type(v).__name__))
    return {"total": total, "nan": nan, "inf": inf, "non_numeric": non_numeric}


def main():
    print("=== Loading precomputed caches ===")
    inflection_doc = json.loads((INFLECT_DIR / "inflection_results.json").read_text())
    v4_doc = json.loads((INFLECT_DIR / "early_warning_v4_cross_layer.json").read_text())
    phase6_doc = json.loads((PHASE6 / "phase6_results.json").read_text())

    succ_iid, wand_iid = _pick_test_iids(inflection_doc)
    print(f"  SUCCESS iid:   {succ_iid}")
    print(f"  WANDERING iid: {wand_iid}")

    for label, iid in [("SUCCESS", succ_iid), ("WANDERING", wand_iid)]:
        trace_path = PHASE6 / "traces" / f"{iid}.json"
        st_path = PHASE6 / "captures" / f"{iid}.safetensors"
        precomputed = _build_precomputed(iid, inflection_doc, v4_doc, phase6_doc)

        t0 = time.time()
        row = extract_all_features(iid, trace_path, st_path, precomputed)
        elapsed = time.time() - t0

        print(f"\n=== {label} ({iid[:80]}...) ===")
        print(f"  elapsed: {elapsed:.2f}s")
        validation = _validate(row)
        print(f"  features: {validation['total']} total")
        print(f"  NaN:      {len(validation['nan'])}  {validation['nan'] if validation['nan'] else ''}")
        print(f"  Inf:      {len(validation['inf'])}  {validation['inf'] if validation['inf'] else ''}")
        print(f"  non-num:  {len(validation['non_numeric'])}  {validation['non_numeric']}")
        print(f"\n  --- feature values ---")
        for k in sorted(row.keys()):
            if k == "iid" or k.startswith("__"):
                continue
            v = row[k]
            if isinstance(v, float):
                print(f"    {k:42s} = {v:>14.6g}")
            else:
                print(f"    {k:42s} = {v}")

    # Time estimate for full N=99
    print("\n=== Time estimate for N=99 ===")
    print(f"  ~{elapsed:.2f}s per trajectory observed; expect ~{elapsed * 99:.0f}s total = {elapsed*99/60:.1f}min")
    print(f"  memory: peak ~50 MB per traj (safetensors 7-8 MB on disk, ×4 bf16->float32 expansion)")


if __name__ == "__main__":
    main()
