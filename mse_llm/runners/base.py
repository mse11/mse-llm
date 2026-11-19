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


# ------------------------------------------------- ServerHandle (§3.3, bash run_serve_params_on_background)

class ServerError(RuntimeError):
    def __init__(self, message: str, log_tail: list[str] | None = None) -> None:
        super().__init__(message)
        self.log_tail = log_tail or []


@dataclass
class Endpoint:
    base_url: str              # http://localhost:8033/v1 (what clients use)
    port: int
    pid: int
    ready_s: float = 0.0

    @property
    def root(self) -> str:
        from mse_llm.health import root_of

        return root_of(self.base_url)


class ServerHandle:
    """A server process as an async context manager:

        async with ServerHandle(argv, port=8033) as ep:   # port check, spawn, wait for /health
            ...                                            # use ep.base_url
        # on exit: graceful stop, kill of the whole process tree, port verified free

    `detached=True` starts a server that outlives mse-llm (`presets serve --background`); leave the
    context with `keep=True` set (or call `start()` without the context manager) to keep it running."""

    def __init__(self, argv: list[str], *, port: int, client_host: str = "localhost", health_path: str = "/health",
                 ready_timeout: float = 120, log_path=None, sink=None, detached: bool = False,
                 env: dict[str, str] | None = None, grace: float = 5.0, progress=None) -> None:
        from mse_llm.proc import Spawned

        self.port = port
        host = "localhost" if client_host in ("0.0.0.0", "::", "") else client_host
        self.base_url = f"http://{host}:{port}/v1"
        self.health_path = health_path
        self.ready_timeout = ready_timeout
        self.grace = grace
        self.progress = progress
        self.keep = False
        self.proc = Spawned(argv, log_path=log_path, sink=sink, detached=detached, env=env)
        self.endpoint: Endpoint | None = None
        self.stop_report = None

    @property
    def pid(self) -> int:
        return self.proc.pid

    async def start(self) -> Endpoint:
        import asyncio

        from mse_llm import proc as pr
        from mse_llm.health import NotReady, root_of, wait_ready

        if await asyncio.to_thread(pr.port_in_use, self.port):
            raise ServerError(pr.describe_port(self.port) + "; stop it first (`mse-llm hw processes --stop`, "
                              "`mse-llm presets stop`) or pick another --port")
        try:
            await self.proc.start()
        except pr.ProcError as e:
            raise ServerError(str(e)) from e
        try:
            ready = await wait_ready(root_of(self.base_url) + self.health_path, self.ready_timeout,
                                     exited=lambda: self.proc.returncode, progress=self.progress)
        except NotReady as e:
            tail = self.proc.tail_log()
            await self.stop()
            raise ServerError(str(e), tail) from None
        except BaseException:
            await self.stop()
            raise
        self.endpoint = Endpoint(self.base_url, self.port, self.pid, ready)
        return self.endpoint

    async def stop(self):
        if self.proc.proc is None:
            return None
        self.stop_report = await self.proc.stop(grace=self.grace, port=self.port)
        return self.stop_report

    async def wait(self) -> int:
        return await self.proc.wait()

    async def __aenter__(self) -> Endpoint:
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        if not self.keep:
            await self.stop()
