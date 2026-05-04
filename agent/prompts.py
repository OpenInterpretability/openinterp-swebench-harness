from __future__ import annotations


SYSTEM_PROMPT = """You are a software engineering agent solving a real GitHub issue. You have access to the project repository at the working directory and three tools.

Tools:
- `bash`: run a shell command. The shell session persists across calls — cwd, environment variables, and processes are preserved.
- `str_replace_editor`: view or edit files. Subcommands: view, create, str_replace, insert.
- `finish`: when you believe the task is complete, call this with a brief summary.

Workflow:
1. Read the issue carefully.
2. Explore the repo to locate relevant files.
3. Reproduce the bug if applicable.
4. Make the minimal fix.
5. Verify by running the project's tests.
6. Call `finish` when done.

Constraints:
- Modify only what is necessary to resolve the issue.
- Do not reformat unrelated code.
- Run tests with the project's own test command (often `pytest`).
- If a tool returns an error, read the error and adjust — do not retry the same broken call.
"""


def render_problem(instance: dict) -> str:
    """Render a SWE-bench Pro instance into the initial user message."""
    repo = instance.get("repo", "<unknown>")
    issue_title = instance.get("issue_title") or instance.get("problem_statement_title") or ""
    problem = instance.get("problem_statement", "")
    base_commit = instance.get("base_commit", "<unknown>")
    workdir = instance.get("__workdir__", "/workspace")

    return f"""# Repository
{repo} @ {base_commit}

# Working directory
{workdir}

# Issue
{issue_title}

{problem}

Resolve the issue. The repository is already cloned and checked out at the working directory."""
