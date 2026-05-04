from __future__ import annotations
from pathlib import Path
from typing import Any


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in the persistent shell session for this problem. "
            "Working directory and environment persist between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120)",
                    "default": 120,
                },
            },
            "required": ["command"],
        },
    },
}

EDITOR_TOOL = {
    "type": "function",
    "function": {
        "name": "str_replace_editor",
        "description": (
            "View or edit files. Subcommands: "
            "view (read file or list dir), "
            "create (write a new file), "
            "str_replace (replace one unique snippet), "
            "insert (insert lines after a given line)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert"],
                },
                "path": {"type": "string", "description": "Absolute path"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
                "file_text": {"type": "string"},
                "insert_line": {"type": "integer"},
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [start, end] line range for view (1-indexed)",
                },
            },
            "required": ["command", "path"],
        },
    },
}

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Signal that the task is complete. Provide a brief summary.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
}

TOOLS = [BASH_TOOL, EDITOR_TOOL, FINISH_TOOL]
TOOL_NAMES = {"bash", "str_replace_editor", "finish"}


def _editor_view(path: str, view_range: list[int] | None) -> dict:
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": f"path does not exist: {path}"}
    if p.is_dir():
        try:
            items = sorted(it.name + ("/" if it.is_dir() else "") for it in p.iterdir())
        except OSError as e:
            return {"ok": False, "error": f"cannot read dir: {e}"}
        return {"ok": True, "content": "\n".join(items), "kind": "dir"}
    try:
        text = p.read_text(errors="replace")
    except OSError as e:
        return {"ok": False, "error": f"cannot read file: {e}"}
    lines = text.splitlines()
    start, end = 1, len(lines)
    if view_range:
        if len(view_range) != 2:
            return {"ok": False, "error": "view_range must be [start, end]"}
        start, end = view_range
        start = max(1, start)
        end = min(len(lines), end)
    sliced = lines[start - 1:end]
    numbered = "\n".join(f"{i+start:6d}\t{ln}" for i, ln in enumerate(sliced))
    return {"ok": True, "content": numbered, "kind": "file", "lines_total": len(lines)}


def _editor_create(path: str, file_text: str) -> dict:
    p = Path(path)
    if p.exists():
        return {"ok": False, "error": f"path exists: {path}"}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_text)
    except OSError as e:
        return {"ok": False, "error": f"cannot write: {e}"}
    return {"ok": True, "content": f"created {path} ({len(file_text)} bytes)"}


def _editor_str_replace(path: str, old_str: str, new_str: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": f"path does not exist: {path}"}
    try:
        text = p.read_text(errors="replace")
    except OSError as e:
        return {"ok": False, "error": f"cannot read: {e}"}
    n = text.count(old_str)
    if n == 0:
        return {"ok": False, "error": "old_str not found in file"}
    if n > 1:
        return {"ok": False, "error": f"old_str matches {n} times — must be unique"}
    new_text = text.replace(old_str, new_str)
    try:
        p.write_text(new_text)
    except OSError as e:
        return {"ok": False, "error": f"cannot write: {e}"}
    return {"ok": True, "content": f"replaced 1 occurrence in {path}"}


def _editor_insert(path: str, insert_line: int, new_str: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": f"path does not exist: {path}"}
    try:
        text = p.read_text(errors="replace")
    except OSError as e:
        return {"ok": False, "error": f"cannot read: {e}"}
    lines = text.splitlines(keepends=True)
    if insert_line < 0 or insert_line > len(lines):
        return {"ok": False, "error": f"insert_line out of range [0, {len(lines)}]"}
    chunk = new_str if new_str.endswith("\n") else new_str + "\n"
    lines.insert(insert_line, chunk)
    try:
        p.write_text("".join(lines))
    except OSError as e:
        return {"ok": False, "error": f"cannot write: {e}"}
    return {"ok": True, "content": f"inserted at line {insert_line} in {path}"}


def _dispatch_editor(args: dict) -> dict:
    cmd = args.get("command")
    path = args.get("path")
    if not path:
        return {"ok": False, "error": "path is required"}
    if cmd == "view":
        return _editor_view(path, args.get("view_range"))
    if cmd == "create":
        if "file_text" not in args:
            return {"ok": False, "error": "file_text required for create"}
        return _editor_create(path, args["file_text"])
    if cmd == "str_replace":
        if "old_str" not in args or "new_str" not in args:
            return {"ok": False, "error": "old_str and new_str required for str_replace"}
        return _editor_str_replace(path, args["old_str"], args["new_str"])
    if cmd == "insert":
        if "insert_line" not in args or "new_str" not in args:
            return {"ok": False, "error": "insert_line and new_str required for insert"}
        return _editor_insert(path, args["insert_line"], args["new_str"])
    return {"ok": False, "error": f"unknown editor command: {cmd}"}


def dispatch_tool(name: str, args: dict[str, Any], *, bash_session=None) -> dict:
    """Execute a tool call. Returns a dict that will be JSON-serialized into the tool result."""
    if name == "bash":
        if bash_session is None:
            return {"ok": False, "error": "bash session not initialized"}
        cmd = args.get("command")
        if not cmd:
            return {"ok": False, "error": "command is required"}
        timeout = int(args.get("timeout") or 120)
        return bash_session.run(cmd, timeout=timeout)
    if name == "str_replace_editor":
        return _dispatch_editor(args)
    if name == "finish":
        return {"ok": True, "summary": args.get("summary", "")}
    return {"ok": False, "error": f"unknown tool: {name}"}
