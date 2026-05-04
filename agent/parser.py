from __future__ import annotations
import json
import re
from dataclasses import dataclass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class AssistantMessage:
    raw: str
    thinking: str | None
    content: str
    tool_calls: list[ToolCall]


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def _strip_think(text: str) -> tuple[str | None, str]:
    """Extract the last </think>-closed block as the thinking content; return (thinking, rest)."""
    matches = list(_THINK_RE.finditer(text))
    if not matches:
        return None, text
    last = matches[-1]
    thinking = last.group(1).strip()
    rest = (text[:last.start()] + text[last.end():]).strip()
    return thinking, rest


def parse_assistant_message(raw: str, *, fallback_id: str = "call_0") -> AssistantMessage:
    """Parse a Qwen3.6 assistant turn into thinking, content, and tool calls.

    Qwen3.6 emits tool calls in `<tool_call>{...json...}</tool_call>` blocks (Qwen-Agent format).
    Multiple tool calls per turn are supported. Thinking is in <think>...</think>.
    """
    thinking, body = _strip_think(raw)

    tool_calls: list[ToolCall] = []
    content_parts: list[str] = []
    cursor = 0
    for i, m in enumerate(_TOOL_RE.finditer(body)):
        if m.start() > cursor:
            content_parts.append(body[cursor:m.start()])
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            content_parts.append(m.group(0))
        else:
            name = payload.get("name") or ""
            args = payload.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(
                ToolCall(id=f"{fallback_id}_{i}", name=name, arguments=args)
            )
        cursor = m.end()
    if cursor < len(body):
        content_parts.append(body[cursor:])

    content = "".join(content_parts).strip()
    return AssistantMessage(
        raw=raw,
        thinking=thinking,
        content=content,
        tool_calls=tool_calls,
    )
