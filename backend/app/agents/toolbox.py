"""Tool registry and dispatch.

One place that knows every tool an agent can call, what its arguments look
like, and how to run it. Two kinds live here:

- **Effect tools** touch the filesystem or run commands. They come from
  `app/tools/` and may produce a `FileEffect` the world turns into a
  `file.change` event.
- **Control tools** produce structured decisions the graph consumes — the PM's
  task list, the reviewer's verdict. They exist as tools rather than as parsed
  prose because a tool call is already a validated, typed channel; asking a
  model to emit JSON in free text and parsing it back is strictly worse.

Every dispatch validates arguments against a Pydantic schema first. A
validation failure is returned to the agent as an error result it can correct
from — never raised into the graph (PLAN.md §7, Scenario 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.llm.base import ToolCall, ToolSpec
from app.tools.filesystem import (
    EditFileInput,
    FileEffect,
    FileTools,
    ListDirInput,
    ReadFileInput,
    ToolResult,
    WriteFileInput,
)
from app.tools.shell import RunCommandInput, ShellTool

# -- control tool schemas --------------------------------------------------


class TaskDraft(BaseModel):
    title: str = Field(description="Short imperative summary of the task.")
    description: str = Field(
        default="", description="What done looks like, and any constraints."
    )


class CreateTasksInput(BaseModel):
    tasks: list[TaskDraft] = Field(
        min_length=1, description="Ordered tasks. Earlier tasks run first."
    )


class SubmitReviewInput(BaseModel):
    approved: bool = Field(description="True if the task's outcome is met.")
    reasons: list[str] = Field(
        default_factory=list,
        description="One entry per finding. Required when approved is false.",
    )


# -- control signals the graph reacts to -----------------------------------


@dataclass
class TasksCreated:
    tasks: list[TaskDraft]


@dataclass
class ReviewSubmitted:
    approved: bool
    reasons: list[str]


ControlSignal = TasksCreated | ReviewSubmitted


@dataclass
class Dispatch:
    result: ToolResult
    control: ControlSignal | None = None

    @property
    def effect(self) -> FileEffect | None:
        return self.result.effect


class Toolbox:
    """Tools available to one agent, already narrowed to its persona."""

    def __init__(
        self,
        files: FileTools,
        shell: ShellTool,
        allowed: tuple[str, ...],
    ) -> None:
        self.files = files
        self.shell = shell
        self.allowed = allowed

        self._schemas: dict[str, type[BaseModel]] = {
            "read_file": ReadFileInput,
            "write_file": WriteFileInput,
            "edit_file": EditFileInput,
            "list_dir": ListDirInput,
            "run_command": RunCommandInput,
            "create_tasks": CreateTasksInput,
            "submit_review": SubmitReviewInput,
        }
        self._descriptions: dict[str, str] = {
            "read_file": "Read a file from the workspace.",
            "write_file": (
                "Write a file, replacing its entire contents. Creates parent "
                "directories."
            ),
            "edit_file": (
                "Replace an exact snippet in a file. The snippet must appear "
                "exactly once."
            ),
            "list_dir": "List the entries of a workspace directory.",
            "run_command": (
                "Run a single command in the workspace root. No shell syntax: "
                "no pipes, redirects, chaining or substitution."
            ),
            "create_tasks": "Submit the ordered task breakdown for an objective.",
            "submit_review": "Submit the review verdict for the current task.",
        }

        unknown = set(allowed) - set(self._schemas)
        if unknown:
            raise ValueError(f"unknown tool names: {sorted(unknown)}")

    def specs(self) -> list[ToolSpec]:
        """Tool definitions for the model, in a deterministic order.

        Order is fixed because the tool list is part of the cached prompt
        prefix on providers that cache; reordering it silently costs every
        cache hit.
        """
        return [
            ToolSpec(
                name=name,
                description=self._descriptions[name],
                input_schema=_json_schema(self._schemas[name]),
            )
            for name in self.allowed
        ]

    def dispatch(self, call: ToolCall) -> Dispatch:
        if call.name not in self.allowed:
            return Dispatch(
                ToolResult(
                    f"{call.name!r} is not a tool you can use. Available: "
                    f"{', '.join(self.allowed)}",
                    is_error=True,
                    misuse=True,
                )
            )

        if "__malformed__" in call.arguments:
            # The provider could not decode the model's JSON. Say so plainly;
            # a schema error here would be misleading.
            return Dispatch(
                ToolResult(
                    f"The arguments for {call.name} were not valid JSON. Re-send "
                    f"the call with a well-formed JSON object matching the "
                    f"tool's schema.",
                    is_error=True,
                    misuse=True,
                )
            )

        schema = self._schemas[call.name]
        try:
            args = schema.model_validate(call.arguments)
        except ValidationError as exc:
            return Dispatch(
                ToolResult(_explain(call.name, exc), is_error=True, misuse=True)
            )

        return self._run(call.name, args)

    def _run(self, name: str, args: BaseModel) -> Dispatch:
        match name:
            case "read_file":
                assert isinstance(args, ReadFileInput)
                return Dispatch(self.files.read_file(args))
            case "write_file":
                assert isinstance(args, WriteFileInput)
                return Dispatch(self.files.write_file(args))
            case "edit_file":
                assert isinstance(args, EditFileInput)
                return Dispatch(self.files.edit_file(args))
            case "list_dir":
                assert isinstance(args, ListDirInput)
                return Dispatch(self.files.list_dir(args))
            case "run_command":
                assert isinstance(args, RunCommandInput)
                return Dispatch(self.shell.run_command(args))
            case "create_tasks":
                assert isinstance(args, CreateTasksInput)
                return Dispatch(
                    ToolResult(f"Recorded {len(args.tasks)} tasks."),
                    control=TasksCreated(tasks=list(args.tasks)),
                )
            case "submit_review":
                assert isinstance(args, SubmitReviewInput)
                if not args.approved and not args.reasons:
                    return Dispatch(
                        ToolResult(
                            "A rejection needs at least one reason. Re-submit "
                            "with the specific findings.",
                            is_error=True,
                            misuse=True,
                        )
                    )
                verdict = "approved" if args.approved else "changes requested"
                return Dispatch(
                    ToolResult(f"Review recorded: {verdict}."),
                    control=ReviewSubmitted(
                        approved=args.approved, reasons=list(args.reasons)
                    ),
                )
        raise AssertionError(f"unhandled tool: {name}")


def _json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for a tool's arguments, with $defs inlined.

    Nested models produce `$ref`/`$defs`, which several providers reject or
    silently ignore. Inlining keeps one schema shape that works everywhere.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    if defs:
        schema = _inline_refs(schema, defs)
    schema.pop("title", None)
    return schema


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = defs.get(ref.rsplit("/", 1)[1], {})
            merged = {**_inline_refs(target, defs)}
            # Preserve siblings of $ref (e.g. a description on the property).
            merged.update(
                {k: _inline_refs(v, defs) for k, v in node.items() if k != "$ref"}
            )
            return merged
        return {k: _inline_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(item, defs) for item in node]
    return node


def _explain(tool_name: str, exc: ValidationError) -> str:
    """Turn a Pydantic error into something a model can act on."""
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        problems.append(f"{location}: {error['msg']}")
    return (
        f"Invalid arguments for {tool_name}:\n"
        + "\n".join(f"  - {p}" for p in problems)
        + "\nCheck the tool schema and call it again."
    )
