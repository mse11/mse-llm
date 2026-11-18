"""CPU model, cores, hybrid (P/E) layout, L3 groups (CCDs) and ISA flags.

Hybrid detection follows Strata's tune_machine.py (MIT): on Windows
GetLogicalProcessorInformationEx(RelationProcessorCore) and more than one EfficiencyClass means
P+E cores; on Linux /sys/devices/cpu_core + cpu_atom. Reimplemented here, not copied.
"""

from __future__ import annotations

import ctypes
import platform
import re
from pathlib import Path

import psutil

from mse_llm.hw import cpuid
from mse_llm.hw.models import CpuInfo
from mse_llm.util import IS_WINDOWS

_RELATION_PROCESSOR_CORE = 0
_RELATION_CACHE = 2


def _mask_to_cpus(mask: int, group: int) -> list[int]:
    return [group * 64 + i for i in range(64) if (mask >> i) & 1]


def _win_logical_info(relation: int) -> list[bytes]:
    k32 = ctypes.windll.kernel32
    k32.GetLogicalProcessorInformationEx.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)
    ]
    size = ctypes.c_uint32(0)
    k32.GetLogicalProcessorInformationEx(relation, None, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if not k32.GetLogicalProcessorInformationEx(relation, buf, ctypes.byref(size)):
        raise OSError(ctypes.get_last_error() or "GetLogicalProcessorInformationEx failed")
    raw = buf.raw[: size.value]
    out, off = [], 0
    while off < len(raw):
        rec_size = int.from_bytes(raw[off + 4 : off + 8], "little")
        out.append(raw[off : off + rec_size])
        off += rec_size
    return out


def _group_masks(rec: bytes, count_off: int, first_off: int) -> list[int]:
    count = int.from_bytes(rec[count_off : count_off + 2], "little") or 1
    cpus: list[int] = []
    for i in range(count):
        o = first_off + 16 * i
        mask = int.from_bytes(rec[o : o + 8], "little")
        group = int.from_bytes(rec[o + 8 : o + 10], "little")
        cpus += _mask_to_cpus(mask, group)
    return cpus


def _windows_topology(info: CpuInfo) -> None:
    cores = _win_logical_info(_RELATION_PROCESSOR_CORE)
    by_class: dict[int, list[int]] = {}
    for rec in cores:
        eff = rec[9]
        by_class.setdefault(eff, []).extend(_group_masks(rec, 30, 32))
    info.hybrid = len(by_class) > 1
    if info.hybrid:
        top = max(by_class)
        info.p_core_cpus = sorted(by_class[top])
        info.e_core_cpus = sorted(c for k, v in by_class.items() if k != top for c in v)
    l3: list[list[int]] = []
    for rec in _win_logical_info(_RELATION_CACHE):
        if rec[8] == 3:   # Level
            l3.append(sorted(_group_masks(rec, 38, 40)))
    info.l3_groups = sorted(l3) or None


def _windows_model() -> str | None:
    import winreg

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
        return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()


def _parse_cpu_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def _linux_topology(info: CpuInfo) -> None:
    core = Path("/sys/devices/cpu_core/cpus")
    atom = Path("/sys/devices/cpu_atom/cpus")
    info.hybrid = core.exists() and atom.exists()
    if info.hybrid:
        info.p_core_cpus = _parse_cpu_list(core.read_text())
        info.e_core_cpus = _parse_cpu_list(atom.read_text())
    groups = set()
    for f in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cache/index3/shared_cpu_list"):
        groups.add(tuple(_parse_cpu_list(f.read_text())))
    info.l3_groups = sorted(list(g) for g in groups) or None


def _linux_model() -> str | None:
    txt = Path("/proc/cpuinfo").read_text(errors="replace")
    m = re.search(r"^model name\s*:\s*(.+)$", txt, re.M)
    return m.group(1).strip() if m else None


def detect_cpu() -> CpuInfo:
    info = CpuInfo(
        physical_cores=psutil.cpu_count(logical=False),
        logical_cpus=psutil.cpu_count(logical=True),
    )
    sources, errors = ["psutil"], []
    try:
        if IS_WINDOWS:
            info.model = _windows_model()
            _windows_topology(info)
            sources.append("winreg+GetLogicalProcessorInformationEx")
        elif platform.system() == "Linux":
            info.model = _linux_model()
            _linux_topology(info)
            sources.append("procfs/sysfs")
        else:
            info.model = platform.processor() or None
    except Exception as e:  # never fatal
        errors.append(f"topology: {e}")
    try:
        info.vendor = cpuid.vendor_string()
        info.isa = cpuid.isa_flags()
        if info.isa is not None:
            sources.append("cpuid")
    except Exception as e:
        errors.append(f"cpuid: {e}")
    info.source = "+".join(sources)
    info.error = "; ".join(errors) or None
    return info


def isa_summary(isa: dict[str, bool] | None) -> str:
    if not isa:
        return "unknown"
    if isa.get("avx512f"):
        ext = [n for n, k in (("VNNI", "avx512_vnni"), ("VBMI", "avx512_vbmi"), ("BF16", "avx512_bf16"))
               if isa.get(k)]
        return "AVX-512" + (f" + {'/'.join(ext)}" if ext else "")
    if isa.get("avx2"):
        return "AVX2" + (" + AVX-VNNI" if isa.get("avx_vnni") else "")
    return "AVX" if isa.get("avx") else "SSE"
