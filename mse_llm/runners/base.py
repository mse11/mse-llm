"""Runner building blocks shared by every engine and client (§3.3)."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any

from mse_llm.util import IS_WINDOWS


def display_argv(argv: list[str]) -> str:
    """One printable line, quoted for the platform's usual shell (cmd/PowerShell on Windows)."""
    return subprocess.list2cmdline(argv) if IS_WINDOWS else shlex.join(argv)


@dataclass
class Command:
    """A fully built command line (never run through a shell)."""

    tool: str                                  # llama-server | llama-bench | llama-benchy | inspect | router
    argv: list[str]
    label: str = ""
    model: str | None = None
    variant: str | None = None
    ctx: int | None = None
    env: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)    # flag validation against the help snapshot
    server: Command | None = None              # background server a client command runs against
    snapshot: str | None = None                # help snapshot the flags were checked against

    @property
    def ok(self) -> bool:
        return not self.errors and (self.server is None or self.server.ok)

    def display(self) -> str:
        return display_argv(self.argv)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "tool": self.tool, "label": self.label, "model": self.model, "variant": self.variant,
            "ctx": self.ctx, "argv": self.argv, "env": self.env, "warnings": self.warnings,
            "errors": self.errors, "snapshot": self.snapshot,
        }
        if self.server is not None:
            d["server"] = self.server.as_dict()
        return d
