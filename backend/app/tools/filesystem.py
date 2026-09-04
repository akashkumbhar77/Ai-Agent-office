"""File tools.

Every path goes through Workspace.resolve() — no tool here calls open() on a
model-supplied string. Tools return structured results rather than raising on
expected failures: a missing file or a failed match is information the agent
should get back and act on, not a crash in the graph runtime (CLAUDE.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.protocol.events import FileOp
from app.tools.workspace import Workspace, WorkspaceViolation

# A generous cap that still prevents one runaway read from filling the context
# window. Truncation is reported to the agent, never silent.
MAX_READ_BYTES = 256_000
MAX_LIST_ENTRIES = 500


@dataclass
class FileEffect:
    """What a tool did to the filesystem, for the world to emit as file.change."""

    path: str
    op: FileOp
    added: int = 0
    removed: int = 0


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    effect: FileEffect | None = None


# -- input schemas (CLAUDE.md §3 rule 6: every tool has a validated schema) --


class ReadFileInput(BaseModel):
    path: str = Field(description="Workspace-relative path to read.")


class WriteFileInput(BaseModel):
    path: str = Field(description="Workspace-relative path to write.")
    content: str = Field(description="Full new contents of the file.")


class EditFileInput(BaseModel):
    path: str = Field(description="Workspace-relative path to edit.")
    old_text: str = Field(description="Exact text to replace. Must appear exactly once.")
    new_text: str = Field(description="Replacement text.")


class ListDirInput(BaseModel):
    path: str = Field(default=".", description="Workspace-relative directory.")


class FileTools:
    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace

    def read_file(self, args: ReadFileInput) -> ToolResult:
        try:
            target = self.ws.resolve(args.path)
        except WorkspaceViolation as exc:
            return ToolResult(str(exc), is_error=True)

        if not target.exists():
            return ToolResult(f"No such file: {args.path}", is_error=True)
        if target.is_dir():
            return ToolResult(
                f"{args.path} is a directory; use list_dir", is_error=True
            )

        raw = target.read_bytes()
        truncated = len(raw) > MAX_READ_BYTES
        text = raw[:MAX_READ_BYTES].decode("utf-8", errors="replace")
        if truncated:
            text += (
                f"\n\n[truncated: file is {len(raw)} bytes, "
                f"showing the first {MAX_READ_BYTES}]"
            )
        return ToolResult(text)

    def write_file(self, args: WriteFileInput) -> ToolResult:
        try:
            target = self.ws.resolve(args.path)
        except WorkspaceViolation as exc:
            return ToolResult(str(exc), is_error=True)

        if target.is_dir():
            return ToolResult(f"{args.path} is a directory", is_error=True)

        existed = target.exists()
        previous = target.read_text(errors="replace") if existed else ""
        self.ws.ensure_parent(target)
        target.write_text(args.content)

        added = len(args.content.splitlines())
        removed = len(previous.splitlines())
        return ToolResult(
            f"Wrote {added} lines to {args.path}",
            effect=FileEffect(
                path=self.ws.relative(target),
                op=FileOp.EDIT if existed else FileOp.CREATE,
                added=added,
                removed=removed,
            ),
        )

    def edit_file(self, args: EditFileInput) -> ToolResult:
        try:
            target = self.ws.resolve(args.path)
        except WorkspaceViolation as exc:
            return ToolResult(str(exc), is_error=True)

        if not target.exists():
            return ToolResult(f"No such file: {args.path}", is_error=True)

        original = target.read_text(errors="replace")
        occurrences = original.count(args.old_text)

        # Ambiguity is an error, not a coin flip. Replacing the first of three
        # matches silently edits the wrong line and the agent cannot tell.
        if occurrences == 0:
            return ToolResult(
                f"old_text not found in {args.path}. The file may have changed "
                f"since you read it; read it again before editing.",
                is_error=True,
            )
        if occurrences > 1:
            return ToolResult(
                f"old_text appears {occurrences} times in {args.path}; it must "
                f"match exactly once. Include more surrounding context.",
                is_error=True,
            )

        updated = original.replace(args.old_text, args.new_text)
        target.write_text(updated)

        return ToolResult(
            f"Edited {args.path}",
            effect=FileEffect(
                path=self.ws.relative(target),
                op=FileOp.EDIT,
                added=len(args.new_text.splitlines()),
                removed=len(args.old_text.splitlines()),
            ),
        )

    def list_dir(self, args: ListDirInput) -> ToolResult:
        try:
            target = self.ws.resolve(args.path)
        except WorkspaceViolation as exc:
            return ToolResult(str(exc), is_error=True)

        if not target.exists():
            return ToolResult(f"No such directory: {args.path}", is_error=True)
        if not target.is_dir():
            return ToolResult(f"{args.path} is a file; use read_file", is_error=True)

        entries = sorted(
            target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
        lines = [
            f"{p.name}/" if p.is_dir() else f"{p.name} ({p.stat().st_size}b)"
            for p in entries[:MAX_LIST_ENTRIES]
        ]
        if len(entries) > MAX_LIST_ENTRIES:
            lines.append(f"[{len(entries) - MAX_LIST_ENTRIES} more entries omitted]")
        if not lines:
            return ToolResult(f"{args.path} is empty")
        return ToolResult("\n".join(lines))
