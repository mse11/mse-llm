"""Small shared helpers: project root, never-fatal subprocess runs, stable hashing."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

IS_WINDOWS = sys.platform == "win32"


@cache
def project_root() -> Path:
    """The mse-llm repo root: MSE_LLM_PROJECT, else the folder with our pyproject.toml, else cwd."""
    env = os.environ.get("MSE_LLM_PROJECT")
    if env:
        return Path(env).resolve()
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for d in (start, *start.parents):
            pp = d / "pyproject.toml"
            if pp.is_file() and 'name = "mse-llm"' in pp.read_text(encoding="utf-8", errors="replace"):
                return d
    return Path.cwd().resolve()


@dataclass
class RunOut:
    argv: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.returncode == 0


def run(argv: list[str], timeout: float = 15.0, cwd: Path | None = None) -> RunOut:
    """Run an argv list (never a shell). Never raises: failures are reported in `error`."""
    flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
    try:
        p = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd,
            creationflags=flags,
        )
        return RunOut(argv, p.returncode, p.stdout, p.stderr)
    except FileNotFoundError:
        return RunOut(argv, None, "", "", error=f"not found: {argv[0]}")
    except subprocess.TimeoutExpired:
        return RunOut(argv, None, "", "", error=f"timeout after {timeout:.0f} s")
    except OSError as e:
        return RunOut(argv, None, "", "", error=str(e))


def stable_hash(obj: Any, length: int = 16) -> str:
    """SHA-256 of canonical JSON (sorted keys), truncated."""
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:length]


def exe(path: Path) -> Path:
    """Add .exe on Windows if the path has no suffix."""
    if IS_WINDOWS and not path.suffix:
        return path.with_suffix(".exe")
    return path
