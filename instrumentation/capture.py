from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import save_file, load_file


@dataclass
class CaptureRecord:
    instance_id: str
    turn_idx: int
    position_label: str
    token_pos: int
    layer: int
    activation_key: str  # key in the safetensors blob


@dataclass
class CaptureBuffer:
    instance_id: str
    records: list[CaptureRecord] = field(default_factory=list)
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    def add(
        self,
        *,
        turn_idx: int,
        position_label: str,
        token_pos: int,
        snapshot: dict[int, torch.Tensor],
    ) -> None:
        for layer, tensor in snapshot.items():
            key = f"t{turn_idx:03d}_{position_label}_p{token_pos:06d}_L{layer:02d}"
            n = 0
            base_key = key
            while key in self.tensors:
                n += 1
                key = f"{base_key}__dup{n}"
            self.tensors[key] = tensor.detach().to(dtype=torch.bfloat16, device="cpu").clone().contiguous()
            self.records.append(
                CaptureRecord(
                    instance_id=self.instance_id,
                    turn_idx=turn_idx,
                    position_label=position_label,
                    token_pos=token_pos,
                    layer=layer,
                    activation_key=key,
                )
            )

    def num_records(self) -> int:
        return len(self.records)


def save_captures(buf: CaptureBuffer, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights_path = out / f"{buf.instance_id}.safetensors"
    meta_path = out / f"{buf.instance_id}.meta.json"

    if buf.tensors:
        save_file(buf.tensors, str(weights_path))
    else:
        weights_path.write_bytes(b"")

    meta = {
        "instance_id": buf.instance_id,
        "n_records": len(buf.records),
        "records": [asdict(r) for r in buf.records],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return weights_path, meta_path


def audit_captures(meta_path: str | Path) -> dict:
    """Verify that every record in the meta file has a corresponding tensor in the .safetensors
    file with the right shape (1-D, length matches d_model).
    """
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    weights_path = meta_path.with_suffix("").with_suffix(".safetensors")
    if not weights_path.exists():
        return {"ok": False, "error": f"safetensors file missing: {weights_path}"}
    tensors = load_file(str(weights_path))
    issues: list[str] = []
    d_model_seen: set[int] = set()
    for rec in meta["records"]:
        key = rec["activation_key"]
        if key not in tensors:
            issues.append(f"missing tensor key: {key}")
            continue
        t = tensors[key]
        if t.dim() != 1:
            issues.append(f"{key}: expected 1-D, got {tuple(t.shape)}")
        d_model_seen.add(int(t.shape[-1]))
    return {
        "ok": len(issues) == 0,
        "n_records": len(meta["records"]),
        "n_tensors": len(tensors),
        "d_model_seen": sorted(d_model_seen),
        "issues": issues,
    }
