"""Check results shared by `hw` and `doctor`."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Status = Literal["pass", "warn", "fail", "info", "skip"]
ORDER = {"fail": 0, "warn": 1, "pass": 2, "info": 3, "skip": 4}
ICON = {"pass": "[green]OK[/]", "warn": "[yellow]WARN[/]", "fail": "[red]FAIL[/]",
        "info": "[cyan]info[/]", "skip": "[dim]skip[/]"}
PLAIN_ICON = {"pass": "✓", "warn": "⚠", "fail": "✗", "info": "·", "skip": "-"}


@dataclass
class Check:
    section: str
    name: str
    status: Status
    detail: str = ""
    hint: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def verdict(checks: list[Check]) -> str:
    if any(c.status == "fail" for c in checks):
        return "below minimum"
    if any(c.status == "warn" for c in checks):
        return "runs, with limits"
    return "can run comfortably"
