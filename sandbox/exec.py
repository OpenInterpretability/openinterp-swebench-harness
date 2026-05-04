from __future__ import annotations
import os
import re
import uuid
from pathlib import Path

import pexpect


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


class BashSession:
    """Persistent interactive bash session.

    The session stays alive across tool calls so that `cd`, `export`, and background
    processes persist as a real user shell would. Output capture relies on a unique
    PS1 sentinel + an exit-code echo line per command.
    """

    def __init__(
        self,
        cwd: str | os.PathLike,
        *,
        max_output_bytes: int = 64_000,
        default_timeout: int = 120,
        env: dict[str, str] | None = None,
    ):
        cwd_path = Path(cwd)
        cwd_path.mkdir(parents=True, exist_ok=True)
        self.cwd = cwd_path
        self.max_output = max_output_bytes
        self.default_timeout = default_timeout

        self._sentinel_id = uuid.uuid4().hex[:12]
        self._ps1 = f"__OISH_PROMPT_{self._sentinel_id}__$ "
        self._ec_marker = f"__OISH_EC_{self._sentinel_id}__"

        full_env = {
            **os.environ,
            "PS1": self._ps1,
            "PS2": "",
            "TERM": "dumb",
            "PROMPT_COMMAND": "",
            "BASH_SILENCE_DEPRECATION_WARNING": "1",
        }
        if env:
            full_env.update(env)

        self.shell = pexpect.spawn(
            "/bin/bash",
            ["--noprofile", "--norc"],
            cwd=str(cwd_path),
            env=full_env,
            encoding="utf-8",
            codec_errors="replace",
            echo=False,
            timeout=default_timeout,
            dimensions=(40, 200),
        )
        # Drain the initial prompt
        self.shell.expect_exact(self._ps1, timeout=5)

    def run(self, command: str, *, timeout: int | None = None) -> dict:
        if self.shell is None or not self.shell.isalive():
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": "shell session is dead",
                "truncated": False,
            }
        t = timeout if timeout is not None else self.default_timeout
        wrapped = f"{command}\necho {self._ec_marker}$?"
        self.shell.sendline(wrapped)
        try:
            self.shell.expect(rf"{self._ec_marker}(-?\d+)\r?\n", timeout=t)
        except pexpect.TIMEOUT:
            self.shell.sendintr()
            try:
                self.shell.expect_exact(self._ps1, timeout=5)
            except pexpect.TIMEOUT:
                pass
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"command timed out after {t}s",
                "truncated": False,
                "timeout": True,
            }
        except pexpect.EOF:
            return {
                "exit_code": -1,
                "stdout": self.shell.before or "",
                "stderr": "shell exited unexpectedly",
                "truncated": False,
            }

        body = _strip_ansi(self.shell.before or "")
        try:
            ec = int(self.shell.match.group(1))
        except (ValueError, IndexError, AttributeError):
            ec = -1

        try:
            self.shell.expect_exact(self._ps1, timeout=5)
        except pexpect.TIMEOUT:
            pass

        truncated = False
        if len(body) > self.max_output:
            head = body[: self.max_output // 2]
            tail = body[-self.max_output // 2:]
            body = head + "\n... [truncated] ...\n" + tail
            truncated = True

        return {
            "exit_code": ec,
            "stdout": body,
            "stderr": "",
            "truncated": truncated,
        }

    def cwd_now(self) -> str:
        out = self.run("pwd", timeout=5)
        return out.get("stdout", "").strip()

    def close(self) -> None:
        if self.shell is None:
            return
        try:
            if self.shell.isalive():
                self.shell.sendline("exit")
                try:
                    self.shell.expect(pexpect.EOF, timeout=2)
                except pexpect.TIMEOUT:
                    pass
        finally:
            self.shell.close(force=True)
            self.shell = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
