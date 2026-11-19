"""PC capability detection (§5.1), modelled on Strata (MIT): never-fatal probes, hard/soft hardware
identity, checks -> verdict, and a test override (MSE_LLM_TEST_HARDWARE / --hw-fixture)."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import socket
import sys
import time
from pathlib import Path

from mse_llm.checks import Check
from mse_llm.hw.models import DiskInfo, GpuInfo, HardwareInfo, HostInfo
from mse_llm.util import IS_WINDOWS, stable_hash

ENV_FIXTURE = "MSE_LLM_TEST_HARDWARE"
CACHE_SECONDS = 30.0
_cache: tuple[float, tuple, HardwareInfo] | None = None


LOW_POWER_RE = __import__("re").compile(r"quiet|silent|saver|saving|eco\b|battery|whisper|low.?power", 2)


def low_power_plan(plan: str | None) -> bool:
    """A power plan that caps CPU / GPU power (Lenovo Legion "Quiet Mode", "Power saver", ...)."""
    return bool(plan and LOW_POWER_RE.search(plan))


ALLOW_LOW_POWER_ENV = "MSE_LLM_ALLOW_LOW_POWER"


def bench_power_problem(p=None) -> str | None:
    """Why benchmark numbers from this power state would be useless (battery, quiet / power-saving plan),
    or None. Read fresh (not the hw cache): the plan can be switched between two commands."""
    if p is None:
        from mse_llm.hw.power import detect_power

        p = detect_power()
    if p.ac_online is False:
        return f"on battery ({p.battery_percent} %)" if p.battery_percent is not None else "on battery"
    if low_power_plan(p.power_plan):
        return f"power plan {p.power_plan!r} caps CPU / GPU power"
    return None


def _host() -> HostInfo:
    ver = platform.version()
    release = platform.release()
    if IS_WINDOWS:
        try:
            build = int(ver.split(".")[2])
            release = "11" if build >= 22000 else release
        except (IndexError, ValueError):
            pass
    return HostInfo(
        hostname=socket.gethostname(),
        os=platform.system(),
        os_release=release,
        os_version=ver,
        arch=platform.machine().lower(),
        python=sys.version.split()[0],
    )


def _merge_gpus(nv: list[GpuInfo], dx: list[GpuInfo]) -> list[GpuInfo]:
    out = list(nv)
    for d in dx:
        if d.vendor == "nvidia":
            for g in nv:
                a, b = (g.name or "").lower(), (d.name or "").lower()
                if a and b and (a in b or b in a):
                    g.luid = g.luid or d.luid
            if nv:
                continue
        out.append(d)
    return out


def _disks(paths: dict[str, Path] | None) -> list[DiskInfo]:
    out = []
    for label, p in (paths or {}).items():
        probe = Path(p)
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            u = shutil.disk_usage(probe)
            out.append(DiskInfo(label=label, path=str(p), total_bytes=u.total, free_bytes=u.free))
        except OSError as e:
            out.append(DiskInfo(label=label, path=str(p), error=str(e)))
    return out


def load_fixture(source: str) -> HardwareInfo:
    """`source` is a JSON string or a path to a JSON file."""
    text = source.strip()
    if not text.startswith("{"):
        text = Path(source).read_text(encoding="utf-8")
    info = HardwareInfo.model_validate(json.loads(text))
    info.fixture = info.fixture or (source if not source.strip().startswith("{") else "inline")
    return info


def detect(force: bool = False, disk_paths: dict[str, Path] | None = None) -> HardwareInfo:
    global _cache
    fx = os.environ.get(ENV_FIXTURE)
    if fx:
        return load_fixture(fx)
    key = tuple(sorted((k, str(v)) for k, v in (disk_paths or {}).items()))
    if not force and _cache and _cache[1] == key and time.monotonic() - _cache[0] < CACHE_SECONDS:
        return _cache[2]

    from mse_llm.hw.cpu import detect_cpu
    from mse_llm.hw.driver import read_driver_settings
    from mse_llm.hw.gpu_nvidia import detect_nvidia
    from mse_llm.hw.memory import detect_memory
    from mse_llm.hw.power import detect_power

    info = HardwareInfo(detected_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"))
    probes = {
        "host": _host,
        "cpu": detect_cpu,
        "memory": detect_memory,
        "power": detect_power,
    }
    for name, fn in probes.items():
        try:
            setattr(info, name, fn())
        except Exception as e:  # never fatal
            info.errors.append(f"{name}: {e}")
    nv, err = detect_nvidia()
    if err:
        info.errors.append(f"gpu: {err}")
    dx: list[GpuInfo] = []
    if IS_WINDOWS:
        from mse_llm.hw.gpu_windows import detect_dx_adapters

        dx, err = detect_dx_adapters()
        if err:
            info.errors.append(f"gpu-dx: {err}")
    info.gpus = _merge_gpus(nv, dx)
    if nv:
        try:
            info.driver = read_driver_settings()
        except Exception as e:
            info.errors.append(f"driver: {e}")
    info.disks = _disks(disk_paths)
    _cache = (time.monotonic(), key, info)
    return info


# ---------------------------------------------------------------- identity

def identity(info: HardwareInfo) -> dict:
    """`hard`: a change means stored settings may not start (new GPU, RAM, CPU, OS).
    `soft`: a change means a re-check is due (driver, OS build). Hash = stable_hash(hard)."""
    total = info.memory.total_bytes or 0
    gpus = sorted(
        (
            {
                "name": g.name,
                "vendor": g.vendor,
                "vram_mib": g.vram_total_mib,
                "pcie_max": f"gen{g.pcie_gen_max}x{g.pcie_width_max}" if g.pcie_gen_max else None,
                "cc": g.compute_capability,
            }
            for g in info.gpus
        ),
        key=lambda d: (d["vendor"] or "", d["name"] or ""),
    )
    hard = {
        "os": info.host.os,
        "arch": info.host.arch,
        "cpu": {
            "model": info.cpu.model,
            "physical_cores": info.cpu.physical_cores,
            "logical_cpus": info.cpu.logical_cpus,
            "hybrid": info.cpu.hybrid,
        },
        "ram_gib": (total // (2 * 2**30)) * 2,   # rounded down to 2 GiB
        "gpus": gpus,
    }
    nv = info.nvidia_gpus()
    soft = {
        "os_release": info.host.os_release,
        "os_version": info.host.os_version,
        "driver": nv[0].driver if nv else None,
        "cuda_driver": nv[0].cuda_driver if nv else None,
    }
    return {"hash": stable_hash(hard), "soft_hash": stable_hash(soft), "hard": hard, "soft": soft}


# ---------------------------------------------------------------- checks

GIB = 2**30


def checks(info: HardwareInfo, thresholds=None) -> list[Check]:
    from mse_llm.config import HwChecksCfg

    t = thresholds or HwChecksCfg()
    out: list[Check] = []
    s = "hardware"

    gpu = info.primary_gpu()
    if gpu and gpu.vendor == "nvidia":
        out.append(Check(s, "GPU", "pass", f"{gpu.name}, {gpu.vram_total_mib} MiB, cc {gpu.compute_capability}"))
    elif gpu:
        out.append(Check(s, "GPU", "warn", f"{gpu.name} (no NVIDIA GPU: CUDA builds won't run)"))
    else:
        out.append(Check(s, "GPU", "warn", "no discrete GPU detected (CPU only)"))

    ram = (info.memory.total_bytes or 0) / GIB
    st = "fail" if ram < t.ram_warn_gb else "warn" if ram < t.ram_pass_gb else "pass"
    out.append(Check(s, "RAM", st, f"{ram:.1f} GiB" + (f" ({info.memory.layout})" if info.memory.layout else "")))

    if info.memory.pressure:
        avail = (info.memory.available_bytes or 0) / GIB
        out.append(Check(s, "memory pressure", "warn" if info.memory.pressure == "high" else "pass",
                         f"{info.memory.pressure} ({avail:.1f} GiB available)"))

    cores = info.cpu.physical_cores or 0
    st = "fail" if cores < t.cores_warn else "warn" if cores < t.cores_pass else "pass"
    out.append(Check(s, "CPU cores", st, f"{cores}C/{info.cpu.logical_cpus}T {info.cpu.model or ''}".strip()))

    for d in info.disks:
        if d.error or d.free_bytes is None:
            out.append(Check(s, f"disk ({d.label})", "warn", f"{d.path}: {d.error or 'unknown'}"))
            continue
        free = d.free_bytes / GIB
        st = "fail" if free < t.disk_warn_gb else "warn" if free < t.disk_pass_gb else "pass"
        out.append(Check(s, f"disk ({d.label})", st, f"{free:.0f} GiB free at {d.path}"))

    p = info.power
    if p.ac_online is False:
        out.append(Check(s, "power", "warn", f"on battery ({p.battery_percent} %)",
                         "benches refuse to start on battery unless --allow-battery"))
    elif p.ac_online and low_power_plan(p.power_plan):
        out.append(Check(s, "power", "warn", f"AC, plan {p.power_plan}",
                         "a quiet / power-saving plan caps CPU and GPU power: TG with experts on the CPU drops and "
                         "varies run to run (measured: gemma QAT 6-36 t/s); switch to a performance plan before "
                         "benchmarking"))
    elif p.ac_online:
        out.append(Check(s, "power", "pass", "AC" + (f", plan {p.power_plan}" if p.power_plan else "")))
    else:
        out.append(Check(s, "power", "warn", f"unknown ({p.error or 'no data'})"))

    for e in info.errors:
        out.append(Check(s, "probe error", "warn", e))
    return out
