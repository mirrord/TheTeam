"""End-to-end integration test for the general_coding flowchart.

The test wires real ``ListFilesNode``/``EditFileNode``/``GitNode``/
``ToolCallNode``/``JsonParseNode``/``RouterNode`` instances together via
the shipped ``configs/flowcharts/general_coding.yaml`` and drives them
with stub agents that return canned JSON responses.

A small fixture repository is created in ``tmp_path`` with a deliberately
buggy ``add`` function and a corresponding failing pytest case.  The
canned coder response replaces the buggy implementation; the canned
reviewer response approves the change.  We assert that:

* The flowchart runs to completion.
* The buggy file is patched and a ``.bak`` snapshot is written.
* The fixture's pytest goes green after the edits.
* A git commit is created on the fixture branch.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml

from pithos.config_manager import ConfigManager
from pithos.flowchart import Flowchart

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubAgent:
    """Minimal agent stub satisfying ``AgentPromptNode``'s ``.send`` API."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def send(
        self,
        prompt: str,
        context_name: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError(
                f"StubAgent exhausted; received prompt of length {len(prompt)}"
            )
        # Re-use the final response if the flowchart loops more than expected
        # so that the test fails loudly via assertions instead of mysteriously.
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


class StubToolRegistry:
    """Registry stub allowing only ``python`` for the RunTests node."""

    def __init__(self, python_path: str) -> None:
        self._python = python_path

    def get_tool(self, name: str):
        if name != "python":
            return None
        from types import SimpleNamespace

        return SimpleNamespace(name="python", path=self._python)

    def list_tools(self) -> list[str]:
        return ["python"]


# ---------------------------------------------------------------------------
# Fixture repository
# ---------------------------------------------------------------------------


_BUGGY_SOURCE = textwrap.dedent("""\
    def add(a, b):
        # Deliberate bug: subtraction instead of addition.
        return a - b
    """)

_FIXED_SOURCE = textwrap.dedent("""\
    def add(a, b):
        return a + b
    """)

_TEST_SOURCE = textwrap.dedent("""\
    from mymod import add


    def test_add_basic():
        assert add(2, 3) == 5


    def test_add_zero():
        assert add(0, 0) == 0
    """)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    )


def _make_fixture_repo(root: Path) -> Path:
    repo = root / "fixture_repo"
    repo.mkdir()
    (repo / "mymod.py").write_text(_BUGGY_SOURCE, encoding="utf-8")
    (repo / "test_mymod.py").write_text(_TEST_SOURCE, encoding="utf-8")

    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "ci@example.test")
    _git(repo, "config", "user.name", "Coding Flowchart Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial buggy state")
    return repo


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


_FLOWCHART_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "flowcharts"
    / "general_coding.yaml"
)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_general_coding_flowchart_drives_repo_to_green(tmp_path: Path) -> None:
    repo = _make_fixture_repo(tmp_path)

    # ---- canned agent responses -----------------------------------------
    plan_json = """```json
{
  "summary": "Fix add() to return the sum instead of the difference",
  "target_files": ["mymod.py"],
  "steps": [
    {"id": 1, "goal": "Replace mymod.add body with a + b", "files": ["mymod.py"]}
  ]
}
```"""

    coder_json = (
        "```json\n"
        + yaml.safe_dump(
            [
                {
                    "path": "mymod.py",
                    "mode": "replace",
                    "content": _FIXED_SOURCE,
                }
            ],
            default_flow_style=False,
        )
        # safe_dump produces YAML, not JSON; replace with explicit JSON dump.
    )

    import json as _json

    coder_json = (
        "```json\n"
        + _json.dumps(
            [
                {
                    "path": "mymod.py",
                    "mode": "replace",
                    "content": _FIXED_SOURCE,
                }
            ]
        )
        + "\n```"
    )

    reviewer_json = (
        "```json\n"
        + _json.dumps({"decision": "APPROVE", "message": "Tests pass; bug fixed."})
        + "\n```"
    )

    repairer_json = coder_json  # not expected to be hit on the happy path

    agents = {
        "planner": StubAgent([plan_json]),
        "coder": StubAgent([coder_json]),
        "reviewer": StubAgent([reviewer_json]),
        "repairer": StubAgent([repairer_json]),
    }

    # ---- load the real flowchart YAML -----------------------------------
    cm = ConfigManager()
    with _FLOWCHART_PATH.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    fc = Flowchart.from_dict(data, cm, validate=True)

    # ---- inject runtime dependencies ------------------------------------
    from pithos.tools.executor import ToolExecutor

    # Pre-set shared_context BEFORE run() so the templated fields resolve.
    # run() calls reset() but the message router preserves shared_context
    # across resets, so values set here survive.
    sc = fc.message_router.shared_context
    sc["repo_root"] = repo.as_posix()
    sc["test_target"] = (repo / "test_mymod.py").as_posix()
    sc["task"] = "Fix the add() function so that the tests pass."
    sc["tool_executor"] = ToolExecutor(timeout=30)
    sc["tool_registry"] = StubToolRegistry(python_path=sys.executable)

    # ---- run ------------------------------------------------------------
    fc.run(agents=agents, initial_input=sc["task"], max_steps=80)

    # ---- assertions -----------------------------------------------------
    # 1. Source file was rewritten with the fixed implementation.
    assert (repo / "mymod.py").read_text(encoding="utf-8") == _FIXED_SOURCE

    # 2. A .bak snapshot of the original was written by EditFileNode.
    assert (repo / "mymod.py.bak").exists()
    assert (repo / "mymod.py.bak").read_text(encoding="utf-8") == _BUGGY_SOURCE

    # 3. The test passes when run directly.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "test_mymod.py", "-q"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"Expected fixture tests to pass after repair.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    # 4. A second commit landed on the branch (GitAdd + GitCommit).
    log = _git(repo, "log", "--oneline").stdout.strip().splitlines()
    assert len(log) >= 2, f"Expected at least 2 commits, got: {log!r}"
    assert "auto:" in log[0], f"Top commit should be the auto-commit, got: {log[0]!r}"

    # 5. Each canned agent was invoked at least once on the happy path.
    assert agents["planner"].calls, "planner should have been invoked"
    assert agents["coder"].calls, "coder should have been invoked"
    assert agents["reviewer"].calls, "reviewer should have been invoked"
