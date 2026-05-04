from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HarnessConfig:
    model: str = "Qwen/Qwen3.6-27B"
    dtype: str = "bfloat16"
    attn_impl: str = "flash_attention_2"

    temperature: float = 1.0
    top_p: float = 0.95

    max_context: int = 200_000
    max_turns: int = 50
    max_invalid_tools_in_row: int = 3
    bash_timeout_default: int = 120
    bash_max_output_bytes: int = 64_000

    thinking_mode: bool = True

    capture_layers: tuple[int, ...] = (11, 23, 31, 43, 55)
    capture_think_mid_every_n_tokens: int = 200

    work_root: Path = field(default_factory=lambda: Path("/tmp/swebench_work"))
    capture_root: Path = field(default_factory=lambda: Path("/tmp/swebench_captures"))
    trace_root: Path = field(default_factory=lambda: Path("/tmp/swebench_traces"))

    def seed_for(self, instance_id: str) -> int:
        import hashlib
        h = hashlib.sha256(instance_id.encode()).hexdigest()
        return int(h[:8], 16)


DEFAULT = HarnessConfig()
