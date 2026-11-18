"""Other inference engines and GPU compute processes that would skew a benchmark."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import psutil

from mse_llm.hw.gpu_nvidia import compute_processes

ENGINE_RE = re.compile(
    r"^(llama-(server|bench|cli|perplexity|batched-bench|fit-params)|strata|ollama|ollama app|ollama_llama_server|"
    r"lms|lm studio|koboldcpp|vllm|text-generation-launcher|tabbyapi|exllamav2|mlx_lm)(\.exe)?$",
    re.I,
)


@dataclass
class ProcInfo:
    pid: int
    name: str
    kind: str               # engine | gpu-compute
    gpu_mib: int | None = None
    cmdline: str | None = None


def _cmd(p: psutil.Process) -> str | None:
    try:
        return " ".join(p.cmdline())[:200] or None
    except (psutil.Error, OSError):
        return None


def detect_processes() -> list[ProcInfo]:
    me = os.getpid()
    out: dict[int, ProcInfo] = {}
    for p in psutil.process_iter(["pid", "name"]):
        name = p.info.get("name") or ""
        if p.info["pid"] != me and ENGINE_RE.match(name):
            out[p.pid] = ProcInfo(p.pid, name, "engine", cmdline=_cmd(p))
    for pid, mib in compute_processes():
        if pid in out:
            out[pid].gpu_mib = mib
            continue
        try:
            name = psutil.Process(pid).name()
        except psutil.Error:
            name = "?"
        out[pid] = ProcInfo(pid, name, "gpu-compute", gpu_mib=mib)
    return sorted(out.values(), key=lambda x: (x.kind, x.name.lower()))


@dataclass
class StopResult:
    pid: int
    name: str
    result: str             # stopped | killed | gone | denied | failed: ...


def stop_processes(procs: list[ProcInfo], timeout: float = 5.0) -> list[StopResult]:
    """Stop engine processes and their children: terminate, wait, then kill what is left.

    GPU compute processes that are not engines (dwm.exe, explorer.exe ...) are never touched.
    The tray app (e.g. "ollama app.exe") is stopped too, otherwise it restarts the server.
    """
    targets: dict[int, psutil.Process] = {}
    for info in procs:
        if info.kind != "engine":
            continue
        try:
            p = psutil.Process(info.pid)
            targets[p.pid] = p
            for c in p.children(recursive=True):
                targets.setdefault(c.pid, c)
        except psutil.NoSuchProcess:
            continue
        except psutil.Error:
            pass
    names: dict[int, str] = {}
    out: dict[int, StopResult] = {}
    for pid, p in targets.items():
        try:
            names[pid] = p.name()
            p.terminate()
        except psutil.NoSuchProcess:
            out[pid] = StopResult(pid, names.get(pid, "?"), "gone")
        except psutil.AccessDenied:
            out[pid] = StopResult(pid, names.get(pid, "?"), "denied")
        except psutil.Error as e:
            out[pid] = StopResult(pid, names.get(pid, "?"), f"failed: {e}")
    alive = [p for pid, p in targets.items() if pid not in out]
    _, still = psutil.wait_procs(alive, timeout=timeout)
    for p in alive:
        if p not in still:
            out[p.pid] = StopResult(p.pid, names.get(p.pid, "?"), "stopped")
    for p in still:
        try:
            p.kill()
            p.wait(timeout=timeout)
            out[p.pid] = StopResult(p.pid, names.get(p.pid, "?"), "killed")
        except psutil.NoSuchProcess:
            out[p.pid] = StopResult(p.pid, names.get(p.pid, "?"), "stopped")
        except psutil.Error as e:
            out[p.pid] = StopResult(p.pid, names.get(p.pid, "?"), f"failed: {e}")
    return sorted(out.values(), key=lambda r: (r.name.lower(), r.pid))
