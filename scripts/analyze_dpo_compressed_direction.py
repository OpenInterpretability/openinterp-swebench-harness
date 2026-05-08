"""
Paper-2 follow-up — identify the compressed direction in multi-probe DPO.

THESIS: paper-2 found fresh-probe AUROC progression 0.472 → 0.528 across nb37 v2
DPO checkpoints, with original FG/RG probes invariant (variance 7e-8). The
"compressed direction" is the residual subspace where DPO learning lives,
orthogonal to FG/RG.

This script identifies that direction by SVD on the difference between
step-0 (base) and step-200 (final) activations, projects it onto our
paper-grade SAE features, and reports the top SAE features that align with
the compressed axis.

Inputs (from Drive or HF):
  - nb37 v2 captures at step-0 and step-200, layers L31 and L55, end_of_think
    position. Each tensor [N, d_model=5120] for N=20 hold-out prompts.
  - Qwen3.6-27B paper-grade SAE encoder weights at L31 and L55 (HF
    caiovicentino1/qwen36-27b-sae-papergrade).

Outputs:
  - top-K singular vectors of (step200 - step0) at L31 and L55
  - alignment cosines vs FG probe direction (L31), RG probe direction (L55),
    and top-100 SAE features at each layer
  - candidate "compressed feature" report — the SAE feature with highest
    cosine to the top-1 singular vector of the diff at each layer

CPU-runnable. ~3 min on M1 Mac with captures local.

Run when GPU returns and Drive mounted, or on Colab cell with captures loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ─── Config ─────────────────────────────────────────────────────────────────


DEFAULT_LAYERS = ("L31", "L55")
DEFAULT_POSITION = "end_of_think"
DEFAULT_TOP_K_SVD = 5
DEFAULT_TOP_K_SAE = 100


# ─── Data loading ───────────────────────────────────────────────────────────


def load_captures(
    drive_root: Path,
    step: int,
    layer: str,
    position: str,
) -> np.ndarray:
    """Load captures for nb37 v2 at given step + layer + position.

    Expected layout (per nb41 v2 forward sweep):
      drive_root / step_{step:04d} / {layer}_{position}.npy
    Each .npy file is shape [N, d_model] = [20, 5120].
    """
    path = drive_root / f"step_{step:04d}" / f"{layer}_{position}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing capture: {path}. "
            "Re-run nb41 v2 forward sweep with all-layer capture, or check "
            "Drive mount."
        )
    return np.load(path)


def load_probe_direction(weights_path: Path) -> np.ndarray:
    """Load FG or RG probe direction from saved sklearn classifier joblib.

    Expects a joblib with `clf.coef_` (shape [1, d_model]) and `sc.scale_`
    (shape [d_model]) for input scaling. Returns L2-normalized direction in
    raw residual space.
    """
    import joblib

    bundle = joblib.load(weights_path)
    coef = bundle["clf"].coef_.ravel()
    if "scaler" in bundle and bundle["scaler"] is not None:
        coef = coef / bundle["scaler"].scale_
    direction = coef / np.linalg.norm(coef)
    return direction


def load_sae_decoder(sae_path: Path, layer: str) -> np.ndarray:
    """Load decoder weights for paper-grade SAE at given layer.

    Returns matrix [d_sae, d_model] where each row is a feature decoder
    direction. Caller normalizes if needed.
    """
    import safetensors.numpy as st_np

    decoder_file = sae_path / f"{layer.lower()}_decoder.safetensors"
    if not decoder_file.exists():
        raise FileNotFoundError(f"Missing SAE decoder: {decoder_file}")
    tensors = st_np.load_file(str(decoder_file))
    return tensors["decoder_weight"]  # shape [d_sae, d_model]


# ─── Core analysis ──────────────────────────────────────────────────────────


def compute_diff_svd(
    step0: np.ndarray, step_final: np.ndarray, top_k: int
) -> dict:
    """SVD of (step_final - step0). Returns top-k singular triples.

    step0, step_final: shape [N, d_model] aligned per-prompt.
    """
    if step0.shape != step_final.shape:
        raise ValueError(
            f"shape mismatch: {step0.shape} vs {step_final.shape}"
        )
    diff = step_final - step0  # [N, d_model]
    U, S, Vt = np.linalg.svd(diff, full_matrices=False)
    return {
        "U": U[:, :top_k],
        "singular_values": S[:top_k],
        "Vt": Vt[:top_k, :],
        "explained_variance": (S[:top_k] ** 2)
        / float(np.sum(S**2) + 1e-12),
        "n_components_returned": top_k,
        "frobenius_norm_diff": float(np.linalg.norm(diff)),
        "frobenius_norm_step0": float(np.linalg.norm(step0)),
        "frobenius_norm_final": float(np.linalg.norm(step_final)),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a, b)
        / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    )


def analyze_layer(
    drive_root: Path,
    layer: str,
    position: str,
    step_pair: tuple[int, int],
    probe_direction: np.ndarray | None,
    sae_decoder: np.ndarray | None,
    top_k_svd: int,
    top_k_sae: int,
) -> dict:
    step0_act = load_captures(drive_root, step_pair[0], layer, position)
    final_act = load_captures(drive_root, step_pair[1], layer, position)
    svd_result = compute_diff_svd(step0_act, final_act, top_k=top_k_svd)

    report = {
        "layer": layer,
        "position": position,
        "step_pair": step_pair,
        "n_prompts": int(step0_act.shape[0]),
        "d_model": int(step0_act.shape[1]),
        "frobenius_diff": svd_result["frobenius_norm_diff"],
        "frobenius_step0": svd_result["frobenius_norm_step0"],
        "frobenius_final": svd_result["frobenius_norm_final"],
        "relative_diff_norm": (
            svd_result["frobenius_norm_diff"]
            / svd_result["frobenius_norm_step0"]
        ),
        "singular_values": svd_result["singular_values"].tolist(),
        "explained_variance": svd_result["explained_variance"].tolist(),
    }

    top_v = svd_result["Vt"][0, :]  # principal compressed direction

    if probe_direction is not None:
        report["cosine_top1_vs_probe"] = cosine(top_v, probe_direction)
        report["cosine_top1_vs_random_baseline"] = cosine(
            top_v, np.random.default_rng(0).standard_normal(top_v.shape[0])
        )

    if sae_decoder is not None:
        # Cosine of each SAE feature vs top-1 compressed direction
        sae_norms = np.linalg.norm(sae_decoder, axis=1, keepdims=True) + 1e-12
        sae_unit = sae_decoder / sae_norms
        cosines = sae_unit @ top_v / (np.linalg.norm(top_v) + 1e-12)
        top_idx = np.argsort(np.abs(cosines))[::-1][:top_k_sae]
        report["top_sae_features"] = [
            {"feature_idx": int(i), "cosine_signed": float(cosines[i])}
            for i in top_idx
        ]

    return report


# ─── Orchestration ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Identify DPO compressed direction at L31 and L55."
    )
    parser.add_argument(
        "--drive-root",
        type=Path,
        required=True,
        help="path to nb41 v2 capture root, e.g. "
        "/content/drive/MyDrive/openinterp_runs/nb41v2_grokking_extended",
    )
    parser.add_argument(
        "--fg-probe",
        type=Path,
        default=None,
        help="path to FabricationGuard probe joblib for L31",
    )
    parser.add_argument(
        "--rg-probe",
        type=Path,
        default=None,
        help="path to ReasoningGuard probe joblib for L55",
    )
    parser.add_argument(
        "--sae-root",
        type=Path,
        default=None,
        help="path to paper-grade SAE root with {l31,l55}_decoder.safetensors",
    )
    parser.add_argument(
        "--step-pair",
        type=int,
        nargs=2,
        default=(0, 200),
        metavar=("BASE", "FINAL"),
    )
    parser.add_argument(
        "--top-k-svd", type=int, default=DEFAULT_TOP_K_SVD
    )
    parser.add_argument(
        "--top-k-sae", type=int, default=DEFAULT_TOP_K_SAE
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("compressed_direction_report.json"),
    )
    args = parser.parse_args()

    probes = {}
    if args.fg_probe is not None:
        probes["L31"] = load_probe_direction(args.fg_probe)
    if args.rg_probe is not None:
        probes["L55"] = load_probe_direction(args.rg_probe)

    sae_decoders = {}
    if args.sae_root is not None:
        for layer in DEFAULT_LAYERS:
            sae_decoders[layer] = load_sae_decoder(args.sae_root, layer)

    full_report = {
        "step_pair": list(args.step_pair),
        "position": DEFAULT_POSITION,
        "layers": [],
    }
    for layer in DEFAULT_LAYERS:
        layer_report = analyze_layer(
            drive_root=args.drive_root,
            layer=layer,
            position=DEFAULT_POSITION,
            step_pair=tuple(args.step_pair),
            probe_direction=probes.get(layer),
            sae_decoder=sae_decoders.get(layer),
            top_k_svd=args.top_k_svd,
            top_k_sae=args.top_k_sae,
        )
        full_report["layers"].append(layer_report)

        # Inline summary
        print(f"=== {layer} {DEFAULT_POSITION} ===")
        print(f"  Frobenius |diff| / |step0|: "
              f"{layer_report['relative_diff_norm']:.4f}")
        print(
            f"  Top-{args.top_k_svd} singular values: "
            + ", ".join(f"{v:.3f}" for v in layer_report['singular_values'])
        )
        if "cosine_top1_vs_probe" in layer_report:
            print(
                f"  Cosine(top1 SVD, probe direction): "
                f"{layer_report['cosine_top1_vs_probe']:+.4f}"
            )
        if "top_sae_features" in layer_report:
            top5_sae = layer_report["top_sae_features"][:5]
            print("  Top-5 SAE features by |cosine| with top1 SVD:")
            for entry in top5_sae:
                print(
                    f"    feature {entry['feature_idx']:6d}  "
                    f"cosine = {entry['cosine_signed']:+.4f}"
                )

    args.output.write_text(json.dumps(full_report, indent=2))
    print(f"\nWrote full report to {args.output}")


# ─── Predicted outputs (recorded here for paper-2 follow-up) ─────────────────
#
# Hypothesis A — DPO compresses on a clean orthogonal direction:
#   - cosine(top1 SVD, FG probe) ≈ 0 (orthogonal, paper-2 invariance)
#   - cosine(top1 SVD, RG probe) ≈ 0
#   - top-1 SAE feature has |cosine| > 0.5 → causally implicated single feature
#   - relative_diff_norm at L31 small (FG signal flat) but top-1 SVD still
#     has signal → orthogonal compression confirmed
#
# Hypothesis B — DPO smears across many features:
#   - top-K singular values decay slowly (no dominant direction)
#   - explained_variance[0] < 0.20 → no single feature explains compression
#   - Implication: compressed direction is high-rank, not a clean lever.
#     Explains why fresh-probe AUROC progresses but never becomes a single
#     feature switch.
#
# Hypothesis C — compression aligns with one of the original probes after all:
#   - cosine(top1 SVD, FG probe) > 0.3 at L31
#   - Would falsify paper-2's "orthogonal" claim at finer resolution
#   - Mitigation: the probe scores were flat (var 7e-8), so even if top1 SVD
#     aligns, the projection magnitude must be tiny — investigate.
#
# Causal-locus connection:
#   - If hypothesis A or B, the locus for the DPO-learned behavior is
#     wherever the top1 SVD direction exerts maximum behavioral lever.
#   - Run Phase 8 protocol (paper5_causal_locus_protocol.md) with
#     d_probe = top1 SVD direction at L31 and L55. If φ ≥ 0.30 vs random:
#     the compressed direction IS a lever (paper-2 missing piece).

if __name__ == "__main__":
    main()
