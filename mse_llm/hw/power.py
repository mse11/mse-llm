"""Power source (AC/battery) and the active power plan. Read-only."""

from __future__ import annotations

import ctypes
import re

from mse_llm.hw.models import PowerInfo
from mse_llm.util import IS_WINDOWS, run


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_uint32),
        ("BatteryFullLifeTime", ctypes.c_uint32),
    ]


def detect_power() -> PowerInfo:
    info = PowerInfo()
    errors = []
    if IS_WINDOWS:
        st = _SystemPowerStatus()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
            info.ac_online = {0: False, 1: True}.get(st.ACLineStatus)
            info.has_battery = st.BatteryFlag != 128
            info.battery_percent = st.BatteryLifePercent if st.BatteryLifePercent != 255 else None
        else:
            errors.append("GetSystemPowerStatus failed")
        r = run(["powercfg", "/getactivescheme"], timeout=5)
        m = re.search(r"([0-9a-fA-F-]{36})\s*\((.+)\)", r.stdout)
        if m:
            info.power_plan_guid, info.power_plan = m.group(1).lower(), m.group(2).strip()
        elif r.error or not r.ok:
            errors.append(f"powercfg: {r.error or r.stderr.strip()}")
        info.source = "GetSystemPowerStatus+powercfg"
    else:
        try:
            import psutil

            b = psutil.sensors_battery()
            info.has_battery = b is not None
            info.ac_online = True if b is None else bool(b.power_plugged)
            info.battery_percent = round(b.percent) if b else None
            info.source = "psutil"
        except Exception as e:
            errors.append(str(e))
    info.error = "; ".join(errors) or None
    return info
