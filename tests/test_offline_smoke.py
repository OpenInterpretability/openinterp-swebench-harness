"""Offline smoke tests — validate the non-GPU parts of the harness.

Run with:  pytest tests/test_offline_smoke.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

# Allow importing from repo root when running pytest from the repo
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_import_all_modules():
    import config  # noqa: F401
    import verdict  # noqa: F401
    from agent import AgentLoop, TOOLS, dispatch_tool  # noqa: F401
    from agent.parser import parse_assistant_message  # noqa: F401
    from sandbox import BashSession  # noqa: F401
    from instrumentation import LayerTap, CaptureBuffer  # noqa: F401


def test_parser_thinking_only():
    from agent.parser import parse_assistant_message
    raw = "<think>let me reason about this</think>\nHello, world."
    msg = parse_assistant_message(raw)
    assert msg.thinking == "let me reason about this"
    assert msg.content == "Hello, world."
    assert msg.tool_calls == []


def test_parser_with_tool_call():
    from agent.parser import parse_assistant_message
    raw = (
        "<think>I should list files</think>\n"
        "Here we go.\n"
        '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n</tool_call>'
    )
    msg = parse_assistant_message(raw)
    assert msg.thinking == "I should list files"
    assert "Here we go" in msg.content
    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc.name == "bash"
    assert tc.arguments == {"command": "ls -la"}


def test_parser_two_tool_calls():
    from agent.parser import parse_assistant_message
    raw = (
        '<tool_call>{"name": "bash", "arguments": {"command": "pwd"}}</tool_call>\n'
        '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
    )
    msg = parse_assistant_message(raw)
    assert len(msg.tool_calls) == 2
    assert msg.tool_calls[0].name == "bash"
    assert msg.tool_calls[1].name == "finish"


def test_editor_create_view_replace(tmp_path):
    from agent.tools import dispatch_tool
    p = tmp_path / "hello.txt"
    r = dispatch_tool("str_replace_editor", {"command": "create", "path": str(p), "file_text": "hello\nworld\n"})
    assert r["ok"], r

    r = dispatch_tool("str_replace_editor", {"command": "view", "path": str(p)})
    assert r["ok"]
    assert "hello" in r["content"] and "world" in r["content"]

    r = dispatch_tool("str_replace_editor", {"command": "str_replace", "path": str(p), "old_str": "hello", "new_str": "HELLO"})
    assert r["ok"]
    assert p.read_text() == "HELLO\nworld\n"


def test_editor_str_replace_unique_required(tmp_path):
    from agent.tools import dispatch_tool
    p = tmp_path / "dup.txt"
    p.write_text("foo\nfoo\n")
    r = dispatch_tool("str_replace_editor", {"command": "str_replace", "path": str(p), "old_str": "foo", "new_str": "bar"})
    assert not r["ok"]
    assert "must be unique" in r["error"]


def test_editor_view_dir(tmp_path):
    from agent.tools import dispatch_tool
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    r = dispatch_tool("str_replace_editor", {"command": "view", "path": str(tmp_path)})
    assert r["ok"]
    assert "a.txt" in r["content"]
    assert "b.txt" in r["content"]


def test_bash_session_persistence(tmp_path):
    from sandbox import BashSession
    with BashSession(tmp_path) as bash:
        r = bash.run("echo hello", timeout=10)
        assert r["exit_code"] == 0
        assert "hello" in r["stdout"]

        r = bash.run("export FOO=42 && echo set", timeout=10)
        assert r["exit_code"] == 0
        r = bash.run("echo $FOO", timeout=10)
        assert "42" in r["stdout"]

        r = bash.run("cd /tmp && pwd", timeout=10)
        assert "/tmp" in r["stdout"]
        r = bash.run("pwd", timeout=10)
        assert "/tmp" in r["stdout"]


def test_bash_timeout(tmp_path):
    from sandbox import BashSession
    with BashSession(tmp_path) as bash:
        r = bash.run("sleep 5", timeout=1)
        assert r.get("timeout") is True
        r = bash.run("echo recovered", timeout=5)
        assert "recovered" in r["stdout"]


def test_bash_nonzero_exit(tmp_path):
    from sandbox import BashSession
    with BashSession(tmp_path) as bash:
        r = bash.run("false", timeout=5)
        assert r["exit_code"] == 1


def test_capture_buffer_save_audit(tmp_path):
    import torch
    from instrumentation.capture import CaptureBuffer, save_captures, audit_captures
    buf = CaptureBuffer(instance_id="test_inst")
    snap = {11: torch.zeros(5120, dtype=torch.bfloat16), 31: torch.ones(5120, dtype=torch.bfloat16)}
    buf.add(turn_idx=0, position_label="think_start", token_pos=0, snapshot=snap)
    buf.add(turn_idx=0, position_label="turn_end", token_pos=42, snapshot=snap)
    weights_path, meta_path = save_captures(buf, tmp_path)
    assert weights_path.exists() and meta_path.exists()

    audit = audit_captures(meta_path)
    assert audit["ok"], audit
    assert audit["n_records"] == 4
    assert audit["d_model_seen"] == [5120]


def test_seed_for_deterministic():
    from config import DEFAULT
    s1 = DEFAULT.seed_for("django__django-12345")
    s2 = DEFAULT.seed_for("django__django-12345")
    s3 = DEFAULT.seed_for("django__django-67890")
    assert s1 == s2
    assert s1 != s3
    assert 0 <= s1 < 2**32
