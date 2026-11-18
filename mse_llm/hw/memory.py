"""RAM total/available, pressure label, commit charge and the DIMM layout (asymmetric channels)."""

from __future__ import annotations

import ctypes

import psutil

from mse_llm.hw.models import MemModule, MemoryInfo
from mse_llm.util import IS_WINDOWS

_SMBIOS_TYPES = {24: "DDR3", 26: "DDR4", 34: "DDR5", 35: "LPDDR5"}


def pressure_label(available: int, total: int) -> str:
    """Strata's thresholds: available/total > 35 % low, > 18 % medium, else high."""
    frac = available / total if total else 0
    return "low" if frac > 0.35 else "medium" if frac > 0.18 else "high"


class _PerfInfo(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", ctypes.c_uint32),
        ("ProcessCount", ctypes.c_uint32),
        ("ThreadCount", ctypes.c_uint32),
    ]


def windows_commit() -> tuple[int, int]:
    pi = _PerfInfo()
    pi.cb = ctypes.sizeof(pi)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        raise OSError("GetPerformanceInfo failed")
    return pi.CommitTotal * pi.PageSize, pi.CommitLimit * pi.PageSize


def windows_modules() -> list[MemModule]:
    import pythoncom
    import win32com.client

    # Initialise COM for this thread; not uninitialised here, because releasing the WMI objects
    # after CoUninitialize prints "Win32 exception occurred releasing IUnknown".
    pythoncom.CoInitialize()
    wmi = win32com.client.GetObject(r"winmgmts:\\.\root\cimv2")
    rows = wmi.ExecQuery(
        "SELECT DeviceLocator, BankLabel, Capacity, Speed, ConfiguredClockSpeed, SMBIOSMemoryType,"
        " Manufacturer, PartNumber FROM Win32_PhysicalMemory"
    )
    out = []
    for r in rows:
        out.append(MemModule(
            locator=(r.DeviceLocator or "").strip() or None,
            bank=(r.BankLabel or "").strip() or None,
            capacity_gb=round(int(r.Capacity) / 2**30, 1) if r.Capacity else None,
            rated_mts=int(r.Speed) if r.Speed else None,
            configured_mts=int(r.ConfiguredClockSpeed) if r.ConfiguredClockSpeed else None,
            type=_SMBIOS_TYPES.get(int(r.SMBIOSMemoryType or 0)),
            manufacturer=(r.Manufacturer or "").strip() or None,
            part=(r.PartNumber or "").strip() or None,
        ))
    del rows, wmi
    return out


def describe_layout(modules: list[MemModule]) -> str | None:
    """Two DIMMs of different size (one per channel, as in most laptops): the first 2*min GB are
    interleaved over both channels, the rest is single-channel."""
    caps = [m.capacity_gb for m in modules if m.capacity_gb]
    if not caps:
        return None
    sizes = " + ".join(f"{c:g}" for c in caps) + " GB"
    if len(caps) == 2 and abs(caps[0] - caps[1]) > 0.5:
        lo, hi = sorted(caps)
        return f"{sizes}, asymmetric: {2 * lo:g} GB dual-channel, {hi - lo:g} GB single-channel"
    if len(caps) == 1:
        return f"{sizes}, single-channel"
    return f"{sizes}, symmetric"


def detect_memory() -> MemoryInfo:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    info = MemoryInfo(
        total_bytes=vm.total,
        available_bytes=vm.available,
        pressure=pressure_label(vm.available, vm.total),
        swap_total_bytes=sw.total,
        swap_used_bytes=sw.used,
    )
    sources, errors = ["psutil"], []
    if IS_WINDOWS:
        try:
            info.commit_total_bytes, info.commit_limit_bytes = windows_commit()
            sources.append("GetPerformanceInfo")
        except Exception as e:
            errors.append(f"commit: {e}")
        try:
            info.modules = windows_modules()
            info.layout = describe_layout(info.modules)
            sources.append("WMI Win32_PhysicalMemory")
        except Exception as e:
            errors.append(f"modules: {e}")
    info.source = "+".join(sources)
    info.error = "; ".join(errors) or None
    return info
