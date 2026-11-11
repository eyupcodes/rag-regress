"""The composite action's PR comment step must never decide the job result."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# The step runs under bash on Linux and macOS runners and needs a POSIX fake ``gh``.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the step script is tested with POSIX shell tools"
)

ACTION = Path(__file__).resolve().parent.parent / "action.yml"


def _comment_script() -> str:
    """The ``run`` block of the comment step, dedented, read without a YAML parser."""
    lines = ACTION.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if "Comment on the pull request" in ln)
    run = next(i for i in range(start, len(lines)) if lines[i].strip() == "run: |")
    block: list[str] = []
    for line in lines[run + 1 :]:
        if line.strip() and not line.startswith("        "):
            break
        block.append(line)
    return textwrap.dedent("\n".join(block))


def _run(tmp_path: Path, gh_body: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(f"#!/bin/sh\n{gh_body}\n", encoding="utf-8")
    gh.chmod(0o755)
    report = tmp_path / "report.md"
    report.write_text("report", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "REPO": "o/r",
        "PR": "1",
        "REPORT": str(report),
    }
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _comment_script()],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_comment_step_posts_when_gh_works(tmp_path: Path) -> None:
    result = _run(tmp_path, 'echo "$@" >> "$0.log"')
    assert result.returncode == 0
    assert "::warning" not in result.stdout
    log = (tmp_path / "bin" / "gh.log").read_text(encoding="utf-8")
    assert "--method POST" in log


def test_comment_step_does_not_fail_when_gh_is_forbidden(tmp_path: Path) -> None:
    result = _run(tmp_path, 'echo "HTTP 403: Resource not accessible" >&2; exit 1')
    assert result.returncode == 0
    assert "::warning" in result.stdout
    assert "job summary" in result.stdout
