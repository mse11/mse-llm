"""NVIDIA driver profile settings (read-only) through NVAPI DRS: CUDA - Sysmem Fallback Policy.

With "Prefer No Sysmem Fallback" a config that doesn't fit fails with CUDA out-of-memory instead
of silently spilling into system RAM, which is what froze the laptop (§9 item 7). mse-llm only
reads the setting; you change it in NVIDIA Control Panel -> Manage 3D settings.

The setting is hidden (not in the driver's named setting list), so its ID comes from community
metadata (nvidiaProfileInspector issue #166): 0x10ECECC9, values 0 = driver default (behaves like
"Prefer Sysmem Fallback"), 1 = Prefer No Sysmem Fallback, 2 = Prefer Sysmem Fallback. NVAPI
returns "setting not found" both for an unset value and for an unknown ID, so `doctor` shows
"not set (driver default)" until the option is set once in the Control Panel.
"""

from __future__ import annotations

import ctypes as C

from mse_llm.hw.models import DriverSettings
from mse_llm.util import IS_WINDOWS

SYSMEM_FALLBACK_ID = 0x10ECECC9
PROGRAMS = ("llama-server.exe", "llama-bench.exe", "llama-perplexity.exe")
_VALUES = {0: "driver_default", 1: "prefer_no_fallback", 2: "prefer_fallback"}

_NVAPI_OK = 0
_NVAPI_SETTING_NOT_FOUND = -160
_SETTING_SIZE = 12320              # sizeof(NVDRS_SETTING_V1)
_APP_SIZE = 12296                  # sizeof(NVDRS_APPLICATION_V1)

_IDS = {
    "Initialize": 0x0150E828,
    "Unload": 0xD22BDD7E,
    "CreateSession": 0x0694D52E,
    "DestroySession": 0xDAD9CFF8,
    "LoadSettings": 0x375DBD6B,
    "GetCurrentGlobalProfile": 0x617BFF9F,
    "GetSetting": 0x73BF8338,
    "FindApplicationByName": 0xEEE566B2,
}


class _NvApi:
    def __init__(self) -> None:
        self.dll = C.CDLL("nvapi64.dll")
        self.qi = self.dll.nvapi_QueryInterface
        self.qi.restype = C.c_void_p
        self.qi.argtypes = [C.c_uint32]

    def fn(self, name: str, *argtypes):
        p = self.qi(_IDS[name])
        if not p:
            raise OSError(f"NVAPI function {name} not available")
        return C.CFUNCTYPE(C.c_int, *argtypes)(p)


def _read_dword(api: _NvApi, session, profile, sid: int) -> int | None:
    get = api.fn("GetSetting", C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p)
    buf = C.create_string_buffer(_SETTING_SIZE)
    C.c_uint32.from_buffer(buf, 0).value = _SETTING_SIZE | (1 << 16)
    r = get(session, profile, sid, buf)
    if r == _NVAPI_SETTING_NOT_FOUND:
        return None
    if r != _NVAPI_OK:
        raise OSError(f"NvAPI_DRS_GetSetting -> {r}")
    # layout: version(4) name(4096) id type location isCurrentPredefined isPredefinedValid
    #         predefined(4100) current(4100)
    return C.c_uint32.from_buffer(buf, 4 + 4096 + 20 + 4100).value


def read_driver_settings() -> DriverSettings:
    out = DriverSettings(setting_id=f"{SYSMEM_FALLBACK_ID:#010x}")
    if not IS_WINDOWS:
        out.error = "NVAPI driver profiles are Windows only"
        return out
    try:
        api = _NvApi()
    except OSError as e:
        out.error = f"nvapi64.dll not loadable: {e}"
        return out
    session = C.c_void_p()
    try:
        if api.fn("Initialize")() != _NVAPI_OK:
            raise OSError("NvAPI_Initialize failed")
        if api.fn("CreateSession", C.POINTER(C.c_void_p))(C.byref(session)) != _NVAPI_OK:
            raise OSError("NvAPI_DRS_CreateSession failed")
        if api.fn("LoadSettings", C.c_void_p)(session) != _NVAPI_OK:
            raise OSError("NvAPI_DRS_LoadSettings failed")
        prof = C.c_void_p()
        if api.fn("GetCurrentGlobalProfile", C.c_void_p, C.POINTER(C.c_void_p))(session, C.byref(prof)):
            raise OSError("NvAPI_DRS_GetCurrentGlobalProfile failed")
        g = _read_dword(api, session, prof, SYSMEM_FALLBACK_ID)
        out.sysmem_fallback_global = _VALUES.get(g, f"unknown({g})") if g is not None else "driver_default"

        find = api.fn("FindApplicationByName", C.c_void_p, C.c_wchar_p, C.POINTER(C.c_void_p), C.c_void_p)
        for exe in PROGRAMS:
            app = C.create_string_buffer(_APP_SIZE)
            C.c_uint32.from_buffer(app, 0).value = _APP_SIZE | (1 << 16)
            p = C.c_void_p()
            if find(session, exe, C.byref(p), app) != _NVAPI_OK:
                continue   # no program profile for this exe
            v = _read_dword(api, session, p, SYSMEM_FALLBACK_ID)
            if v is not None:
                out.sysmem_fallback_programs[exe] = _VALUES.get(v, f"unknown({v})")

        progs = set(out.sysmem_fallback_programs.values())
        if out.sysmem_fallback_programs and len(out.sysmem_fallback_programs) == len(PROGRAMS) \
                and progs == {"prefer_no_fallback"}:
            out.sysmem_fallback = "prefer_no_fallback"
        else:
            out.sysmem_fallback = out.sysmem_fallback_global
        out.source = "nvapi-drs"
    except OSError as e:
        out.error = str(e)
    finally:
        try:
            if session:
                api.fn("DestroySession", C.c_void_p)(session)
            api.fn("Unload")()
        except OSError:
            pass
    return out
