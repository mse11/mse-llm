"""Async subprocesses without a shell: spawn, log tee + output filter, process-tree stop, port checks (§3.3).

Replaces the bash `EXEC_cmd_with_output_mode`, `server_get_pid_by_ppid`, `server_stop_windows`
(`ps` / `tasklist` / `taskkill` loops) and the `/tmp` log files.

Windows details:
* A server is started in its own process group (CREATE_NEW_PROCESS_GROUP), so Ctrl+C in the console
  reaches only mse-llm, which then stops the whole tree itself.
* An attached server is also put in a Job Object with KILL_ON_JOB_CLOSE: if mse-llm dies without
  cleaning up, Windows ends the server and every child (router models) with it. No orphans.
* A detached server (`presets serve --background`) gets neither, so it outlives mse-llm; it gets a
  hidden console (CREATE_NO_WINDOW), which its children (router model instances) share.
"""

from __future__ import annotations

import asyncio
import collections
import os
import re
import signal
import socket
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

import psutil

from mse_llm.util import IS_WINDOWS

OutputMode = Literal["all", "filter", "none"]

# bash run_serve_params_FILTER (grep -E), in the same order
SERVER_FILTER = [
    # Memory & fit - the most useful diagnostics
    "common_memory_breakdown_print", "common_params_fit_impl", "common_fit_params",
    # Model & context sizing
    "llama_context: n_ctx", "llama_kv_cache: size", "load_tensors: offload", "print_info: model type",
    "print_info: model params", "print_info: file size",
    # Server ready signals
    "server is listening", "model loaded", "all slots are idle",
    # Errors / warnings (always want these)
    "[Ee]rror", "[Ww]arning", "WARN",
]


class ProcError(RuntimeError):
    pass


# ---------------------------------------------------------------- output filter (bash EXEC_cmd_with_output_mode)

class OutputFilter:
    """all = every line, filter = lines matching one of the patterns (all when there are none), none = nothing."""

    def __init__(self, mode: OutputMode = "filter", patterns: Iterable[str] | None = None) -> None:
        if mode not in ("all", "filter", "none"):
            raise ValueError(f"output mode {mode!r}: use all, filter or none")
        pats = list(SERVER_FILTER if patterns is None else patterns)
        self.mode: OutputMode = "all" if mode == "filter" and not pats else mode
        self._re = re.compile("|".join(f"(?:{p})" for p in pats)) if pats else None

    def __call__(self, line: str) -> bool:
        if self.mode == "all":
            return True
        if self.mode == "none":
            return False
        return bool(self._re and self._re.search(line))


# ---------------------------------------------------------------- ports

def port_in_use(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    """Something accepts connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def port_owner(port: int) -> tuple[int, str] | None:
    """(pid, name) of the process listening on a TCP port, if the OS tells us."""
    try:
        conns = psutil.net_connections(kind="tcp")
    except (psutil.Error, OSError):
        return None
    for c in conns:
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port and c.pid:
            try:
                return c.pid, psutil.Process(c.pid).name()
            except psutil.Error:
                return c.pid, "?"
    return None


def describe_port(port: int) -> str:
    owner = port_owner(port)
    return f"port {port} is in use" + (f" by {owner[1]} (pid {owner[0]})" if owner else "")


def wait_port_free(port: int, timeout: float = 10.0, host: str = "127.0.0.1") -> bool:
    end = time.monotonic() + timeout
    while True:
        if not port_in_use(port, host):
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(0.1)


# ---------------------------------------------------------------- process tree

def tree(pid: int) -> list[psutil.Process]:
    """The process and all its descendants (parent first). Empty if it is gone."""
    try:
        p = psutil.Process(pid)
        return [p, *p.children(recursive=True)]
    except psutil.Error:
        return []


def alive(procs: Iterable[psutil.Process]) -> list[psutil.Process]:
    out = []
    for p in procs:
        try:
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                out.append(p)
        except psutil.Error:
            pass
    return out


@dataclass
class StopReport:
    pid: int
    pids: list[int] = field(default_factory=list)      # every process of the tree that was seen
    graceful: list[int] = field(default_factory=list)  # ended after the signal
    killed: list[int] = field(default_factory=list)    # needed a kill
    left: list[int] = field(default_factory=list)      # still running afterwards (should never happen)
    port_free: bool | None = None
    seconds: float = 0.0

    @property
    def clean(self) -> bool:
        return not self.left and self.port_free is not False

    def summary(self) -> str:
        s = f"stopped pid {self.pid} ({len(self.pids)} process{'es' if len(self.pids) != 1 else ''}"
        if self.killed:
            s += f", {len(self.killed)} killed"
        s += f") in {self.seconds:.1f} s"
        if self.left:
            s += f"; STILL RUNNING: {', '.join(map(str, self.left))}"
        if self.port_free is False:
            s += "; port still in use"
        return s


def stop_tree(pid: int, *, grace: float = 5.0, signal_first: bool = True, port: int | None = None,
              extra: Iterable[psutil.Process] = (), job: object | None = None, kill_wait: float = 20.0) -> StopReport:
    """Stop a process and every descendant: graceful signal (CTRL_BREAK on Windows, SIGTERM to the group
    elsewhere), wait `grace`, kill what is left (children first), wait, then check the port is free.

    `extra`: processes seen earlier (e.g. right after start) that may have been re-parented since.
    `job`: the Windows Job Object of an attached server; terminated last, which catches anything
    the scan missed."""
    t0 = time.monotonic()
    seen: dict[int, psutil.Process] = {p.pid: p for p in extra}
    for p in tree(pid):
        seen[p.pid] = p
    rep = StopReport(pid, sorted(seen))
    procs = alive(seen.values())
    if procs and signal_first:
        _signal(pid)
        _, still = psutil.wait_procs(procs, timeout=grace)
        rep.graceful = sorted(p.pid for p in procs if p not in still)
        procs = still
    for p in tree(pid):                     # children started meanwhile
        if p.pid not in seen:
            seen[p.pid] = p
            procs.append(p)
            rep.pids.append(p.pid)
    procs = alive(procs)
    for p in reversed(procs):               # leaves first, so nothing respawns a child
        try:
            p.kill()
            rep.killed.append(p.pid)
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            pass
    # a killed process holding tens of GiB (a model loaded with -lm none) takes Windows several seconds to end
    _, still = psutil.wait_procs(procs, timeout=kill_wait)
    if job is not None:
        _terminate_job(job)
        _, still = psutil.wait_procs(alive(still), timeout=2.0)
    rep.left = sorted(p.pid for p in alive(still))
    rep.killed = sorted(set(rep.killed) - set(rep.graceful))
    rep.pids = sorted(set(rep.pids))
    if port is not None:
        rep.port_free = wait_port_free(port, timeout=10.0)
    rep.seconds = time.monotonic() - t0
    return rep


def _signal(pid: int) -> None:
    try:
        if IS_WINDOWS:
            os.kill(pid, signal.CTRL_BREAK_EVENT)   # only reaches a process group that shares our console
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (OSError, ProcessLookupError, ValueError):
        try:
            psutil.Process(pid).terminate()
        except psutil.Error:
            pass


# ---------------------------------------------------------------- Windows Job Object

def _job_for(pid: int, *, kill_on_close: bool = True, memory_cap_mib: int | None = None) -> object | None:
    """Job Object holding pid (Windows only, best effort): KILL_ON_JOB_CLOSE for attached servers, and an
    optional cap on the committed memory of the whole tree (children are in the job too), so a server can't
    push the system into the pagefile: an allocation past the cap fails inside the server instead."""
    if not IS_WINDOWS or not (kill_on_close or memory_cap_mib):
        return None
    try:
        import win32api
        import win32con
        import win32job

        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        if kill_on_close:
            info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_cap_mib:
            info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_JOB_MEMORY
            info["JobMemoryLimit"] = int(memory_cap_mib) * 2**20
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
        h = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, pid)
        try:
            win32job.AssignProcessToJobObject(job, h)
        finally:
            win32api.CloseHandle(h)
        return job
    except Exception:
        return None


def _terminate_job(job: object) -> None:
    try:
        import win32job

        win32job.TerminateJobObject(job, 1)
    except Exception:
        pass


# ---------------------------------------------------------------- spawn

LineSink = Callable[[str], None]


def resolve_program(prog: str, cwd: Path | None = None) -> str:
    """argv[0] -> an absolute path when it can be found, else unchanged (the start then reports "not found").

    * A path with a directory part (`builds/llama.cpp/.../llama-bench`, `.\\x.exe`) is taken relative to
      `cwd` (default: the current directory). Windows CreateProcess doesn't find a relative path written
      with forward slashes, which is how Git Bash users type it.
    * A bare name (`llama-bench`, `uvx`) is looked up on PATH with the PATHEXT extensions (.exe, .cmd ...).
    * On Windows a missing `.exe` is added (`.../llama-bench` -> `.../llama-bench.exe`)."""
    import shutil

    p = Path(prog)
    if p.parent != Path(".") or prog.startswith((".", "/", "\\")) or p.is_absolute():
        base = p if p.is_absolute() else (cwd or Path.cwd()) / p
        cands = [base]
        if IS_WINDOWS and not base.suffix:
            cands.insert(0, base.with_suffix(".exe"))
        for c in cands:
            if c.is_file():
                return str(c.resolve())
        return prog
    return shutil.which(prog) or prog


class Spawned:
    """A running command: stdout+stderr merged, each line written to the log file and passed to `sink`.

    attached (default): output read through a pipe, process in a kill-on-close Job Object (Windows).
    detached: output goes straight to the log file, the process outlives mse-llm."""

    def __init__(self, argv: list[str], *, log_path: Path | None = None, sink: LineSink | None = None,
                 detached: bool = False, env: dict[str, str] | None = None, cwd: Path | None = None,
                 keep_lines: int = 400, memory_cap_mib: int | None = None) -> None:
        self.memory_cap_mib = memory_cap_mib
        self.argv = argv
        self.log_path = log_path
        self.sink = sink
        self.detached = detached
        self.env = env
        self.cwd = cwd
        self.lines: collections.deque[str] = collections.deque(maxlen=keep_lines)
        self.watchers: list[LineSink] = []
        self.proc: asyncio.subprocess.Process | subprocess.Popen | None = None
        self.job: object | None = None
        self.started_procs: list[psutil.Process] = []
        self._reader: asyncio.Task | None = None
        self._log: IO[str] | None = None

    @property
    def pid(self) -> int:
        if self.proc is None:
            raise ProcError("not started")
        return self.proc.pid

    @property
    def returncode(self) -> int | None:
        if self.proc is None:
            return None
        if isinstance(self.proc, subprocess.Popen):
            return self.proc.poll()
        return self.proc.returncode

    def _env(self) -> dict[str, str] | None:
        return {**os.environ, **self.env} if self.env else None

    async def start(self) -> None:
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.argv:
            self.argv = [resolve_program(self.argv[0], self.cwd), *self.argv[1:]]
        try:
            if self.detached:
                self._start_detached()
            else:
                await self._start_attached()
        except FileNotFoundError:
            raise ProcError(f"not found: {self.argv[0]}") from None
        except OSError as e:
            raise ProcError(f"can't start {self.argv[0]}: {e}") from e
        self.started_procs = tree(self.pid)

    async def _start_attached(self) -> None:
        kw: dict = {}
        if IS_WINDOWS:
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        self.proc = await asyncio.create_subprocess_exec(
            *self.argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, env=self._env(), cwd=self.cwd, limit=4 * 2**20, **kw)
        self.job = _job_for(self.proc.pid, memory_cap_mib=self.memory_cap_mib)
        if self.log_path:
            self._log = self.log_path.open("w", encoding="utf-8", errors="replace")
        self._reader = asyncio.create_task(self._read())

    def _start_detached(self) -> None:
        if not self.log_path:
            raise ProcError("a detached process needs a log file")
        log = self.log_path.open("wb")
        common = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=self._env(), cwd=self.cwd)
        try:
            if IS_WINDOWS:
                base = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                try:   # leave the job of our own parent (terminal), so closing it doesn't end the server
                    self.proc = subprocess.Popen(self.argv, creationflags=base | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                                                 **common)
                except PermissionError:
                    self.proc = subprocess.Popen(self.argv, creationflags=base, **common)
            else:
                self.proc = subprocess.Popen(self.argv, start_new_session=True, **common)
        finally:
            log.close()
        if self.memory_cap_mib:     # cap only: the job must not end the server when mse-llm exits
            self.job = _job_for(self.proc.pid, kill_on_close=False, memory_cap_mib=self.memory_cap_mib)

    async def _read(self) -> None:
        assert isinstance(self.proc, asyncio.subprocess.Process) and self.proc.stdout is not None
        while True:
            try:
                raw = await self.proc.stdout.readline()
            except (ValueError, asyncio.LimitOverrunError):
                continue
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            self.lines.append(line)
            if self._log:
                self._log.write(line + "\n")
                self._log.flush()
            for w in list(self.watchers):
                w(line)
            if self.sink:
                self.sink(line)
        if self._log:
            self._log.close()
            self._log = None

    def tail_log(self, n: int = 40) -> list[str]:
        """Last lines of the output (from the log file for a detached process)."""
        if self.detached and self.log_path and self.log_path.is_file():
            return self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        return list(self.lines)[-n:]

    async def wait(self) -> int:
        if self.proc is None:
            raise ProcError("not started")
        if isinstance(self.proc, subprocess.Popen):
            while self.proc.poll() is None:
                await asyncio.sleep(0.2)
            return self.proc.returncode
        rc = await self.proc.wait()
        if self._reader:
            await self._reader
        return rc

    async def stop(self, *, grace: float = 5.0, port: int | None = None) -> StopReport:
        if self.proc is None:
            raise ProcError("not started")
        rep = await asyncio.to_thread(stop_tree, self.pid, grace=grace, port=port, extra=self.started_procs,
                                      job=self.job, signal_first=not self.detached)
        if isinstance(self.proc, asyncio.subprocess.Process):
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except TimeoutError:
                pass
            if self._reader:
                try:
                    await asyncio.wait_for(self._reader, 5)
                except TimeoutError:
                    self._reader.cancel()
        if self.job is not None:
            try:
                import win32api

                win32api.CloseHandle(self.job)
            except Exception:
                pass
            self.job = None
        return rep
