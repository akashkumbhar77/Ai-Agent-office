"""Confinement tests.

These are the security boundary for everything an agent does to the
filesystem. Each case here is an escape a model has plausibly emitted.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.tools.workspace import Workspace, WorkspaceViolation


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "auth.py").write_text("x = 1\n")
    (tmp_path / "outside.txt").write_text("secret\n")
    return Workspace(root)


# -- the happy path --------------------------------------------------------


def test_resolves_a_plain_relative_path(ws: Workspace) -> None:
    assert ws.resolve("src/auth.py").read_text() == "x = 1\n"


def test_resolves_a_path_that_does_not_exist_yet(ws: Workspace) -> None:
    # Writes need to resolve before the file exists.
    target = ws.resolve("src/new/deep.py")
    assert target.parent.parent == ws.root / "src"


def test_dot_segments_inside_the_root_are_fine(ws: Workspace) -> None:
    assert ws.resolve("src/../src/auth.py") == ws.root / "src" / "auth.py"


def test_relative_round_trips(ws: Workspace) -> None:
    assert ws.relative(ws.resolve("src/auth.py")) == "src/auth.py"


# -- escapes ---------------------------------------------------------------


def test_parent_traversal_is_rejected(ws: Workspace) -> None:
    with pytest.raises(WorkspaceViolation, match="escapes"):
        ws.resolve("../outside.txt")


def test_deep_traversal_is_rejected(ws: Workspace) -> None:
    with pytest.raises(WorkspaceViolation, match="escapes"):
        ws.resolve("src/../../../../../../etc/passwd")


def test_absolute_path_is_rejected_not_silently_rerooted(ws: Workspace) -> None:
    """Path('/root') / '/etc/passwd' == '/etc/passwd' — joining an absolute
    path discards the root entirely, so this must be caught before the join."""
    with pytest.raises(WorkspaceViolation, match="must be relative"):
        ws.resolve("/etc/passwd")


def test_symlink_pointing_outside_is_rejected(ws: Workspace) -> None:
    link = ws.root / "escape"
    link.symlink_to(ws.root.parent / "outside.txt")
    with pytest.raises(WorkspaceViolation, match="escapes"):
        ws.resolve("escape")


def test_write_through_a_symlinked_directory_is_rejected(ws: Workspace) -> None:
    """The subtler case: the file does not exist, but its parent is a symlink
    out of the workspace, so creating it would write outside."""
    outside_dir = ws.root.parent / "elsewhere"
    outside_dir.mkdir()
    (ws.root / "linkdir").symlink_to(outside_dir, target_is_directory=True)
    with pytest.raises(WorkspaceViolation, match="escapes"):
        ws.resolve("linkdir/payload.txt")


def test_symlink_staying_inside_is_allowed(ws: Workspace) -> None:
    (ws.root / "alias").symlink_to(ws.root / "src")
    assert ws.resolve("alias/auth.py") == ws.root / "src" / "auth.py"


def test_empty_and_whitespace_paths_are_rejected(ws: Workspace) -> None:
    for bad in ("", "   ", "\n"):
        with pytest.raises(WorkspaceViolation, match="empty"):
            ws.resolve(bad)


def test_windows_style_absolute_is_rejected(ws: Workspace) -> None:
    for bad in (r"C:\Windows\System32", r"\\server\share"):
        with pytest.raises(WorkspaceViolation, match="must be relative"):
            ws.resolve(bad)


def test_the_root_itself_resolves(ws: Workspace) -> None:
    assert ws.resolve(".") == ws.root


# -- root handling ---------------------------------------------------------


def test_missing_root_fails_loudly_at_construction(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, WorkspaceViolation)):
        Workspace(tmp_path / "does-not-exist")


def test_root_that_is_a_file_is_rejected(tmp_path: Path) -> None:
    f = tmp_path / "afile"
    f.write_text("")
    with pytest.raises(WorkspaceViolation, match="not a directory"):
        Workspace(f)


def test_symlinked_root_is_canonicalized(tmp_path: Path) -> None:
    """If root itself is a symlink, containment must compare canonical paths
    on both sides or every resolve() would look like an escape."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "f.txt").write_text("ok")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    ws = Workspace(link)
    assert ws.root == real.resolve()
    assert ws.resolve("f.txt").read_text() == "ok"


@pytest.mark.skipif(os.name == "nt", reason="POSIX path semantics")
def test_null_byte_in_path_is_rejected(ws: Workspace) -> None:
    with pytest.raises((WorkspaceViolation, ValueError)):
        ws.resolve("src/auth\x00.py")
