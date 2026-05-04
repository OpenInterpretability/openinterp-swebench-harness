from __future__ import annotations
import subprocess
from pathlib import Path


def generate_patch(workdir: str | Path, *, base_commit: str | None = None) -> dict:
    """Run git diff in workdir and return the patch string. If base_commit is None, diffs HEAD."""
    workdir = Path(workdir)
    if not (workdir / ".git").exists():
        return {"ok": False, "error": f"not a git repo: {workdir}"}
    args = ["git", "diff", base_commit] if base_commit else ["git", "diff", "HEAD"]
    try:
        result = subprocess.run(
            args,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "git diff timed out"}
    except FileNotFoundError:
        return {"ok": False, "error": "git not installed"}
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip() or "git diff non-zero exit"}
    return {"ok": True, "patch": result.stdout, "n_bytes": len(result.stdout)}
