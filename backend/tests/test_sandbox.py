"""Sandbox tests.

These assert the boundary itself rather than the guardrails around it: with
isolation on, a command that is genuinely trying to escape must fail even
though the allowlist and the argument check both let it through. That is the
whole point of having a boundary — the layers above it are advisory, and only
this one holds when they are wrong.

Skipped where bubblewrap is absent. A skip here is honest: it means the host
cannot run the sandbox, which is exactly the condition that makes the office
raise its degraded-posture banner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools.sandbox import (
    Sandbox,
    SandboxUnavailable,
    _toolchain_prefixes,
    available,
    build,
)
from app.tools.shell import (
    DEFAULT_ALLOWLIST,
    EXECUTING_ALLOWLIST,
    INERT_ALLOWLIST,
    RunCommandInput,
    ShellTool,
)
from app.tools.workspace import Workspace

needs_bwrap = pytest.mark.skipif(not available(), reason="bubblewrap not installed")


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "workspace"
    root.mkdir(parents=True)
    (root / "hello.txt").write_text("inside\n")
    (tmp_path / "secret.txt").write_text("do not read me\n")
    return Workspace(root)


@pytest.fixture
def sandboxed(ws: Workspace) -> ShellTool:
    return ShellTool(ws, sandbox=Sandbox(workspace_root=ws.root))


# -- mode resolution -------------------------------------------------------


def test_off_means_off(tmp_path: Path) -> None:
    assert build("off", tmp_path) is None


def test_explicit_bwrap_fails_loudly_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking for isolation and silently not getting it is the one outcome
    that must not be possible."""
    monkeypatch.setattr("app.tools.sandbox.available", lambda: False)
    with pytest.raises(SandboxUnavailable, match="bubblewrap is not installed"):
        build("bwrap", tmp_path)


def test_auto_degrades_rather_than_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.tools.sandbox.available", lambda: False)
    assert build("auto", tmp_path) is None


# -- the allowlist follows the isolation -----------------------------------


def test_no_sandbox_means_inert_tools_only(ws: Workspace) -> None:
    assert ShellTool(ws, sandbox=None).allowlist == INERT_ALLOWLIST


def test_a_sandbox_unlocks_the_executing_tools(ws: Workspace) -> None:
    tool = ShellTool(ws, sandbox=Sandbox(workspace_root=ws.root))
    assert tool.allowlist == DEFAULT_ALLOWLIST
    assert tool.allowlist >= EXECUTING_ALLOWLIST


def test_the_two_lists_do_not_overlap() -> None:
    """An executable in both would be reachable unsandboxed, which is the
    exact mistake the split exists to prevent."""
    assert not (INERT_ALLOWLIST & EXECUTING_ALLOWLIST)


# -- containment -----------------------------------------------------------


@needs_bwrap
def test_the_host_filesystem_is_not_there(sandboxed: ShellTool) -> None:
    result = sandboxed.run_command(
        RunCommandInput(
            command="python3 -c \"print(open('/etc/hostname').read())\""
        )
    )
    # Allowed to run, and fails inside the sandbox: the file does not exist
    # there at all, which is stronger than a permission denial.
    assert not result.is_error
    assert "FileNotFoundError" in result.content


@needs_bwrap
def test_a_write_outside_the_workspace_does_not_reach_the_host(
    sandboxed: ShellTool, ws: Workspace
) -> None:
    target = ws.root.parent / "ESCAPED.txt"
    sandboxed.run_command(
        RunCommandInput(
            command=f"python3 -c \"open('{target}','w').write('x')\""
        )
    )
    assert not target.exists()


@needs_bwrap
def test_there_is_no_network(sandboxed: ShellTool) -> None:
    """`npx` and `git push` are only safe to allow because of this."""
    result = sandboxed.run_command(
        RunCommandInput(
            command=(
                "python3 -c \"import socket;"
                "socket.create_connection(('1.1.1.1',443),timeout=3)\""
            ),
            timeout_s=20,
        )
    )
    assert "Error" in result.content or "error" in result.content
    assert "exit code: 0" not in result.content


@needs_bwrap
def test_the_backends_environment_is_not_inherited(sandboxed: ShellTool) -> None:
    """The backend runs with the provider API key in its environment. An
    agent that can read it can exfiltrate it the moment it gets a network."""
    result = sandboxed.run_command(
        RunCommandInput(
            command=(
                "python3 -c \"import os;"
                "print('KEY=' + os.environ.get('OPENAI_API_KEY','absent'))\""
            )
        )
    )
    assert "KEY=absent" in result.content


# -- the project's own toolchain -------------------------------------------
#
# Both of these were found by a live run, and neither is hypothetical: with
# the sandbox on and nothing below in place, an agent asked to "verify it by
# running the tests" simply could not, and got told so in bwrap's own words.


def test_a_missing_tool_gets_an_actionable_refusal(ws: Workspace) -> None:
    """`bwrap: execvp pytest: No such file or directory` names an internal
    detail and suggests nothing. An agent reading it retries the same call."""
    tool = ShellTool(ws, allowlist=DEFAULT_ALLOWLIST, sandbox=Sandbox(ws.root))
    result = tool.run_command(RunCommandInput(command="pytest -q", timeout_s=30))
    assert "bwrap" not in result.content
    assert "not available in this workspace" in result.content
    assert "python3 -m" in result.content


def test_the_base_interpreter_of_a_workspace_venv_is_mounted(
    tmp_path: Path,
) -> None:
    """A venv inside the workspace is bound with it, but `bin/python3` is a
    symlink to the interpreter it was built from — usually outside /usr. With
    that target absent every script in the venv fails ENOENT despite being
    right there, because the shebang cannot resolve.
    """
    import venv as venv_module

    root = tmp_path / "workspace"
    root.mkdir()
    venv_module.create(root / ".venv", with_pip=False, symlinks=True)

    prefixes = _toolchain_prefixes(root)
    interpreter = (root / ".venv" / "bin" / "python3").resolve()
    assert any(interpreter.is_relative_to(p) for p in prefixes), (
        f"{interpreter} is not covered by {prefixes}"
    )


def test_a_home_directory_is_never_bound_as_a_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The widening is bounded on purpose. A venv built from an interpreter
    sitting directly in $HOME must not drag the whole home directory — that
    is precisely what the sandbox exists to keep out of reach."""
    root = tmp_path / "workspace"
    (root / ".venv").mkdir(parents=True)
    (root / ".venv" / "pyvenv.cfg").write_text(f"home = {Path.home()}/bin\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path.parent)

    assert Path.home() not in _toolchain_prefixes(root)


@needs_bwrap
def test_a_workspace_venv_can_actually_run(tmp_path: Path) -> None:
    """The end-to-end version: a project that ships its own tools can use
    them. This is what an agent verifying its own work depends on."""
    import venv as venv_module

    root = tmp_path / "workspace"
    root.mkdir()
    venv_module.create(root / ".venv", with_pip=False, symlinks=True)
    (root / "check.py").write_text("print('ran from the workspace venv')\n")

    tool = ShellTool(
        Workspace(root), allowlist=DEFAULT_ALLOWLIST, sandbox=Sandbox(root)
    )
    result = tool.run_command(
        RunCommandInput(command="python3 check.py", timeout_s=60)
    )
    assert "ran from the workspace venv" in result.content


# -- the workspace still works ---------------------------------------------


@needs_bwrap
def test_the_workspace_is_readable_and_writable(
    sandboxed: ShellTool, ws: Workspace
) -> None:
    """Isolation that also breaks the agent's actual job is not a fix."""
    read = sandboxed.run_command(RunCommandInput(command="cat hello.txt"))
    assert "inside" in read.content

    sandboxed.run_command(
        RunCommandInput(command="python3 -c \"open('made.txt','w').write('ok')\"")
    )
    assert (ws.root / "made.txt").read_text() == "ok", "workspace writes must persist"


@needs_bwrap
def test_paths_keep_their_real_names_inside(
    sandboxed: ShellTool, ws: Workspace
) -> None:
    """The workspace is bound at its own absolute path, so a traceback names
    a file the operator can open. Remapping it to /workspace would make every
    error message refer to a path that does not exist outside."""
    result = sandboxed.run_command(
        RunCommandInput(command="python3 -c \"import os;print(os.getcwd())\"")
    )
    assert str(ws.root) in result.content
