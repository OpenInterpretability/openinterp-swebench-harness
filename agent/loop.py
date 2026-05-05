from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import torch

from .tools import TOOLS, dispatch_tool, TOOL_NAMES
from .parser import parse_assistant_message, _strip_think
from .prompts import SYSTEM_PROMPT


@dataclass
class TurnLog:
    turn_idx: int
    prompt_tokens: int
    new_tokens: int
    wall_seconds: float
    raw_response: str
    thinking: str | None
    content: str
    tool_calls: list[dict]
    tool_results: list[dict]
    capture_token_pos: dict
    n_capture_steps: int


@dataclass
class AgentResult:
    instance_id: str
    finished: bool
    finish_reason: str
    turns: list[TurnLog] = field(default_factory=list)
    error: str | None = None


class AgentLoop:
    """Multi-turn chat loop with thinking, tool calls, and per-token capture."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        config,
        bash_session,
        tap,
        capture_buffer,
        instance_id: str,
        seed: int,
    ):
        self.model = model
        self.tok = tokenizer
        self.cfg = config
        self.bash = bash_session
        self.tap = tap
        self.cap = capture_buffer
        self.instance_id = instance_id
        self.seed = seed

        self._think_open_ids = self.tok.encode("<think>", add_special_tokens=False)
        self._think_close_ids = self.tok.encode("</think>", add_special_tokens=False)
        self._tool_open_ids = self.tok.encode("<tool_call>", add_special_tokens=False)

    @staticmethod
    def _find_first_seq(tokens: list[int], needle: list[int]) -> int | None:
        n, m = len(tokens), len(needle)
        if m == 0 or n < m:
            return None
        for i in range(n - m + 1):
            if tokens[i:i + m] == needle:
                return i
        return None

    @staticmethod
    def _find_all_seq(tokens: list[int], needle: list[int]) -> list[int]:
        n, m = len(tokens), len(needle)
        if m == 0 or n < m:
            return []
        out = []
        i = 0
        while i <= n - m:
            if tokens[i:i + m] == needle:
                out.append(i)
                i += m
            else:
                i += 1
        return out

    def _capture_at(self, *, turn_idx: int, label: str, token_pos: int) -> bool:
        n = self.tap.n_steps()
        if token_pos < 0 or token_pos >= n:
            return False
        snap = self.tap.get_activation_at(token_pos)
        self.cap.add(turn_idx=turn_idx, position_label=label, token_pos=token_pos, snapshot=snap)
        return True

    def _do_captures_for_turn(
        self,
        *,
        turn_idx: int,
        new_token_ids: list[int],
    ) -> dict:
        log: dict = {}
        if not new_token_ids:
            return log

        if self._capture_at(turn_idx=turn_idx, label="think_start", token_pos=0):
            log["think_start"] = 0

        think_close_idx = self._find_first_seq(new_token_ids, self._think_close_ids) if self._think_close_ids else None
        if think_close_idx is not None and think_close_idx > 0:
            stride = self.cfg.capture_think_mid_every_n_tokens
            mids: list[int] = []
            for tp in range(stride, think_close_idx, stride):
                if self._capture_at(turn_idx=turn_idx, label="think_mid", token_pos=tp):
                    mids.append(tp)
            if mids:
                log["think_mid"] = mids
            if self._capture_at(turn_idx=turn_idx, label="think_end", token_pos=think_close_idx):
                log["think_end"] = think_close_idx

        if self._tool_open_ids:
            tcs: list[int] = []
            for tc_idx in self._find_all_seq(new_token_ids, self._tool_open_ids):
                if self._capture_at(turn_idx=turn_idx, label="pre_tool", token_pos=tc_idx):
                    tcs.append(tc_idx)
            if tcs:
                log["pre_tool"] = tcs

        last_idx = len(new_token_ids) - 1
        if self._capture_at(turn_idx=turn_idx, label="turn_end", token_pos=last_idx):
            log["turn_end"] = last_idx

        return log

    def _build_input_ids(self, messages: list[dict]) -> torch.Tensor:
        # transformers 5.x returns a BatchEncoding (dict-like) when tools= is set,
        # even with return_tensors="pt" — extract input_ids defensively.
        out = self.tok.apply_chat_template(
            messages,
            tools=TOOLS,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=self.cfg.thinking_mode,
        )
        if torch.is_tensor(out):
            return out
        if hasattr(out, "data") and isinstance(out.data, dict) and "input_ids" in out.data:
            return out["input_ids"]
        if isinstance(out, dict) and "input_ids" in out:
            return out["input_ids"]
        if isinstance(out, list):
            return torch.tensor([out], dtype=torch.long)
        raise RuntimeError(f"unexpected apply_chat_template return type: {type(out)}")

    def run(self, problem_message: str) -> AgentResult:
        result = AgentResult(instance_id=self.instance_id, finished=False, finish_reason="error")

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem_message},
        ]

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        invalid_in_row = 0

        for turn_idx in range(self.cfg.max_turns):
            try:
                input_ids = self._build_input_ids(messages)
            except Exception as e:
                result.error = f"chat template error: {type(e).__name__}: {e}"
                result.finish_reason = "error"
                return result

            input_ids = input_ids.to(self.model.device)
            prompt_len = int(input_ids.shape[-1])
            if prompt_len >= self.cfg.max_context - 64:
                result.finish_reason = "context_overflow"
                return result

            self.tap.reset()
            t0 = time.time()
            try:
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids,
                        max_new_tokens=min(8192, self.cfg.max_context - prompt_len - 64),
                        do_sample=True,
                        temperature=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                        pad_token_id=self.tok.eos_token_id,
                    )
            except torch.cuda.OutOfMemoryError:
                result.error = "CUDA OOM during generation"
                result.finish_reason = "error"
                return result
            except Exception as e:
                result.error = f"generate failed: {type(e).__name__}: {e}"
                result.finish_reason = "error"
                return result

            wall = time.time() - t0
            new_token_ids = output[0, prompt_len:].tolist()
            decode_ids = (
                new_token_ids[:-1]
                if new_token_ids and new_token_ids[-1] == self.tok.eos_token_id
                else new_token_ids
            )
            raw_response = self.tok.decode(decode_ids, skip_special_tokens=False)
            capture_log = self._do_captures_for_turn(turn_idx=turn_idx, new_token_ids=new_token_ids)
            n_capture_steps = self.tap.n_steps()

            parsed = parse_assistant_message(raw_response, fallback_id=f"call_t{turn_idx}")

            tool_calls_log: list[dict] = []
            tool_results_log: list[dict] = []

            # Embed the assistant body as raw text (with <tool_call> blocks intact). Qwen3.6's
            # chat template chokes on structured tool_calls field with stringified arguments
            # ("Can only get item pairs from a mapping"); replaying the model's own tokens
            # avoids re-rendering and preserves Hermes format the model emitted.
            _, body_with_tools = _strip_think(raw_response)
            messages.append({"role": "assistant", "content": body_with_tools})

            if not parsed.tool_calls:
                turn_log = TurnLog(
                    turn_idx=turn_idx,
                    prompt_tokens=prompt_len,
                    new_tokens=len(new_token_ids),
                    wall_seconds=wall,
                    raw_response=raw_response,
                    thinking=parsed.thinking,
                    content=parsed.content,
                    tool_calls=tool_calls_log,
                    tool_results=tool_results_log,
                    capture_token_pos=capture_log,
                    n_capture_steps=n_capture_steps,
                )
                result.turns.append(turn_log)

                invalid_in_row += 1
                if invalid_in_row >= self.cfg.max_invalid_tools_in_row:
                    result.finish_reason = "invalid_tools"
                    return result
                messages.append({
                    "role": "user",
                    "content": "You did not call a tool. Use bash, str_replace_editor, or finish.",
                })
                continue

            invalid_in_row = 0
            finish_called = False
            for tc in parsed.tool_calls:
                if tc.name not in TOOL_NAMES:
                    res: dict[str, Any] = {"ok": False, "error": f"unknown tool: {tc.name}"}
                else:
                    try:
                        res = dispatch_tool(tc.name, tc.arguments, bash_session=self.bash)
                    except Exception as e:
                        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}

                tool_calls_log.append({"id": tc.id, "name": tc.name, "arguments": tc.arguments})
                tool_results_log.append({"id": tc.id, "result": res})

                payload = json.dumps(res, ensure_ascii=False)
                if len(payload) > 32_000:
                    payload = payload[:32_000] + "...[truncated]"
                messages.append({
                    "role": "tool",
                    "content": payload,
                })

                if tc.name == "finish":
                    finish_called = True

            result.turns.append(TurnLog(
                turn_idx=turn_idx,
                prompt_tokens=prompt_len,
                new_tokens=len(new_token_ids),
                wall_seconds=wall,
                raw_response=raw_response,
                thinking=parsed.thinking,
                content=parsed.content,
                tool_calls=tool_calls_log,
                tool_results=tool_results_log,
                capture_token_pos=capture_log,
                n_capture_steps=n_capture_steps,
            ))

            if finish_called:
                result.finished = True
                result.finish_reason = "finish_tool"
                return result

        result.finish_reason = "max_turns"
        return result
