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


# Match either <think>...</think> (open + close) or just leading text up to a stray </think>.
_THINK_OPEN_CLOSE_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_CLOSE_ONLY_RE = re.compile(r"^(.*?)</think>", re.DOTALL)

# JSON-format tool call: <tool_call>{...}</tool_call>
_TOOL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Hermes/XML-format tool call: <tool_call><function=NAME>...</function></tool_call>
# Qwen3.6 emits this format. Parameters are inside as <parameter=KEY>VALUE</parameter>.
_TOOL_HERMES_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)


def _strip_think(text: str) -> tuple[str | None, str]:
    """Extract thinking content; handle both <think>...</think> and orphan </think>."""
    matches = list(_THINK_OPEN_CLOSE_RE.finditer(text))
    if matches:
        last = matches[-1]
        thinking = last.group(1).strip()
        rest = (text[:last.start()] + text[last.end():]).strip()
        return thinking, rest

    m = _THINK_CLOSE_ONLY_RE.match(text)
    if m and m.group(1).strip():
        thinking = m.group(1).strip()
        rest = text[m.end():].strip()
        return thinking, rest

    return None, text


def _coerce_param_value(raw: str) -> object:
    """Try to coerce a string parameter value to a more useful type, falling back to str."""
    s = raw.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() == "null" or s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return raw


def _parse_hermes_args(body: str) -> dict:
    """Parse <parameter=KEY>VALUE</parameter> blocks inside a Hermes <function> body."""
    args: dict[str, object] = {}
    for m in _PARAM_RE.finditer(body):
        key = m.group(1).strip()
        val = m.group(2)
        # Strip a single leading/trailing newline (common in formatted output) but keep
        # internal whitespace which is meaningful for code commands or file contents.
        if val.startswith("\n"):
            val = val[1:]
        if val.endswith("\n"):
            val = val[:-1]
        args[key] = _coerce_param_value(val)
    return args


def parse_assistant_message(raw: str, *, fallback_id: str = "call_0") -> AssistantMessage:
    """Parse a Qwen3.6 assistant turn into thinking, content, and tool calls.

    Supports both tool-call formats:
    - Hermes/XML (Qwen3.6 default): <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>
    - OpenAI JSON: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    thinking, body = _strip_think(raw)

    tool_calls: list[ToolCall] = []
    found_spans: list[tuple[int, int]] = []

    for m in _TOOL_HERMES_RE.finditer(body):
        name = m.group(1).strip()
        inner = m.group(2)
        args = _parse_hermes_args(inner)
        tool_calls.append(
            ToolCall(id=f"{fallback_id}_{len(tool_calls)}", name=name, arguments=args)
        )
        found_spans.append((m.start(), m.end()))

    for m in _TOOL_JSON_RE.finditer(body):
        if any(s <= m.start() < e for s, e in found_spans):
            continue
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name") or ""
        args = payload.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        tool_calls.append(
            ToolCall(id=f"{fallback_id}_{len(tool_calls)}", name=name, arguments=args)
        )
        found_spans.append((m.start(), m.end()))

    found_spans.sort()
    content_parts: list[str] = []
    cursor = 0
    for s, e in found_spans:
        if s > cursor:
            content_parts.append(body[cursor:s])
        cursor = e
    if cursor < len(body):
        content_parts.append(body[cursor:])
    content = "".join(content_parts).strip()

    return AssistantMessage(
        raw=raw,
        thinking=thinking,
        content=content,
        tool_calls=tool_calls,
    )
