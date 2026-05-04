from __future__ import annotations
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from agent import AgentLoop
from agent.prompts import render_problem
from instrumentation import LayerTap, CaptureBuffer, save_captures
from sandbox import BashSession
from config import HarnessConfig, DEFAULT
from verdict import generate_patch


class Runner:
    """Run one (or many) SWE-bench Pro instances against Qwen3.6-27B with full instrumentation.

    The model + tokenizer are loaded once externally and passed in to amortize cost across
    many problems.
    """

    def __init__(self, *, model, tokenizer, config: HarnessConfig = DEFAULT):
        self.model = model
        self.tok = tokenizer
        self.cfg = config
        self.cfg.work_root.mkdir(parents=True, exist_ok=True)
        self.cfg.capture_root.mkdir(parents=True, exist_ok=True)
        self.cfg.trace_root.mkdir(parents=True, exist_ok=True)

    def run_one(
        self,
        instance: dict,
        *,
        prepare_workdir_fn: Callable[[dict, Path], None] | None = None,
    ) -> dict:
        instance_id = instance["instance_id"]
        seed = self.cfg.seed_for(instance_id)

        workdir = self.cfg.work_root / instance_id
        workdir.mkdir(parents=True, exist_ok=True)
        if prepare_workdir_fn is not None:
            prepare_workdir_fn(instance, workdir)

        instance_with_workdir = {**instance, "__workdir__": str(workdir)}

        cap_buf = CaptureBuffer(instance_id=instance_id)
        tap = LayerTap(self.model, self.cfg.capture_layers).attach()
        bash: BashSession | None = None
        result: Any = None
        wall = 0.0
        try:
            bash = BashSession(
                workdir,
                default_timeout=self.cfg.bash_timeout_default,
                max_output_bytes=self.cfg.bash_max_output_bytes,
            )
            loop = AgentLoop(
                model=self.model,
                tokenizer=self.tok,
                config=self.cfg,
                bash_session=bash,
                tap=tap,
                capture_buffer=cap_buf,
                instance_id=instance_id,
                seed=seed,
            )
            problem_msg = render_problem(instance_with_workdir)
            t0 = time.time()
            result = loop.run(problem_msg)
            wall = time.time() - t0
        finally:
            tap.detach()
            if bash is not None:
                bash.close()

        cap_paths = save_captures(cap_buf, self.cfg.capture_root)

        trace_path = self.cfg.trace_root / f"{instance_id}.json"
        trace_payload: dict[str, Any] = {
            "instance_id": instance_id,
            "seed": seed,
            "config": {
                "model": self.cfg.model,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "thinking_mode": self.cfg.thinking_mode,
                "capture_layers": list(self.cfg.capture_layers),
            },
            "finished": getattr(result, "finished", False),
            "finish_reason": getattr(result, "finish_reason", "error"),
            "wall_seconds": wall,
            "n_turns": len(getattr(result, "turns", [])),
            "n_captures": cap_buf.num_records(),
            "error": getattr(result, "error", None),
            "turns": [asdict(t) for t in getattr(result, "turns", [])],
        }
        trace_path.write_text(json.dumps(trace_payload, indent=2, default=str))

        patch = generate_patch(workdir)
        patch_path: Path | None = None
        if patch.get("ok"):
            patch_path = self.cfg.trace_root / f"{instance_id}.patch"
            patch_path.write_text(patch["patch"])

        return {
            "instance_id": instance_id,
            "finished": trace_payload["finished"],
            "finish_reason": trace_payload["finish_reason"],
            "wall_seconds": wall,
            "n_turns": trace_payload["n_turns"],
            "n_captures": trace_payload["n_captures"],
            "trace_path": str(trace_path),
            "captures_safetensors": str(cap_paths[0]),
            "captures_meta": str(cap_paths[1]),
            "patch_path": str(patch_path) if patch_path else None,
            "patch_n_bytes": patch.get("n_bytes", 0),
        }
