"""All display adapters on Windows from the DirectX registry (HKLM\\SOFTWARE\\Microsoft\\DirectX).

Approach taken from Strata's windows-gpu.mjs (MIT): DedicatedVideoMemory is a 64-bit value, unlike
Win32_VideoController.AdapterRAM (32-bit, wrong for big cards), and AdapterLuid matches the PDH
"GPU Adapter Memory" counter instances. Reimplemented in Python with winreg.
"""

from __future__ import annotations

import re

from mse_llm.hw.models import GpuInfo

VIRTUAL_RE = re.compile(r"virtual|remote|indirect|mirror|basic render|parsec|displaylink|idd", re.I)
_VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x1022: "amd", 0x8086: "intel"}


def luid_str(v: int) -> str:
    """PDH instance format: luid_0xHHHHHHHH_0xLLLLLLLL."""
    return f"luid_0x{(v >> 32) & 0xFFFFFFFF:08X}_0x{v & 0xFFFFFFFF:08X}"


def detect_dx_adapters() -> tuple[list[GpuInfo], str | None]:
    try:
        import winreg
    except ImportError:
        return [], "not Windows"
    out: list[GpuInfo] = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\DirectX")
    except OSError as e:
        return [], f"DirectX registry: {e}"
    with root:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            try:
                with winreg.OpenKey(root, sub) as k:
                    vals = {}
                    j = 0
                    while True:
                        try:
                            name, val, _ = winreg.EnumValue(k, j)
                        except OSError:
                            break
                        vals[name] = val
                        j += 1
            except OSError:
                continue
            desc = vals.get("Description")
            if not desc:
                continue
            ded = int(vals.get("DedicatedVideoMemory") or 0)
            if VIRTUAL_RE.search(desc) or ded == 0:
                continue
            vendor = _VENDORS.get(int(vals.get("VendorId") or 0), "other")
            luid = vals.get("AdapterLuid")
            out.append(GpuInfo(
                vendor=vendor,
                name=str(desc),
                vram_total_mib=ded // 2**20,
                luid=luid_str(int(luid)) if luid is not None else None,
                integrated=vendor != "nvidia" and ded <= 2 * 2**30,
                source="dx-registry",
            ))
    # The same adapter can appear under several GUIDs; keep one per LUID/name.
    seen, uniq = set(), []
    for g in out:
        key = g.luid or g.name
        if key not in seen:
            seen.add(key)
            uniq.append(g)
    return uniq, None
