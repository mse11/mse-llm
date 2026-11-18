"""Raw x86-64 CPUID through a few bytes of machine code (ctypes), so no extra package is needed.

Windows does not expose AVX-512 VNNI/VBMI/BF16 through IsProcessorFeaturePresent, so CPUID is read
directly. Returns None on non-x86-64 machines or if executable memory can't be allocated.
"""

from __future__ import annotations

import ctypes
import mmap
import platform
import sys
from functools import cache

# void cpuid(uint32 leaf, uint32 subleaf, uint32 out[4])  -> out = eax, ebx, ecx, edx
_WIN64 = bytes.fromhex(
    "53"            # push rbx
    "89c8"          # mov eax, ecx        (leaf)
    "89d1"          # mov ecx, edx        (subleaf)
    "0fa2"          # cpuid
    "418900"        # mov [r8], eax
    "41895804"      # mov [r8+4], ebx
    "41894808"      # mov [r8+8], ecx
    "4189500c"      # mov [r8+12], edx
    "5b"            # pop rbx
    "c3"            # ret
)
_SYSV64 = bytes.fromhex(
    "53"            # push rbx
    "4989d0"        # mov r8, rdx         (out)
    "89f8"          # mov eax, edi        (leaf)
    "89f1"          # mov ecx, esi        (subleaf)
    "0fa2"          # cpuid
    "418900" "41895804" "41894808" "4189500c"
    "5b" "c3"
)

_keep: list[object] = []   # keep the executable buffer alive


@cache
def _func():
    if platform.machine().lower() not in ("amd64", "x86_64"):
        return None
    code = _WIN64 if sys.platform == "win32" else _SYSV64
    proto = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))
    if sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.VirtualAlloc.restype = ctypes.c_void_p
        k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
        addr = k32.VirtualAlloc(None, len(code), 0x3000, 0x40)  # MEM_COMMIT|MEM_RESERVE, RWX
        if not addr:
            return None
        ctypes.memmove(addr, code, len(code))
        return proto(addr)
    buf = mmap.mmap(-1, mmap.PAGESIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
    buf.write(code)
    _keep.append(buf)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    return proto(addr)


def cpuid(leaf: int, subleaf: int = 0) -> tuple[int, int, int, int] | None:
    f = _func()
    if f is None:
        return None
    out = (ctypes.c_uint32 * 4)()
    f(leaf, subleaf, out)
    return out[0], out[1], out[2], out[3]


def _bit(v: int, b: int) -> bool:
    return bool((v >> b) & 1)


def isa_flags() -> dict[str, bool] | None:
    """Instruction-set flags relevant to llama.cpp's CPU backend."""
    l0 = cpuid(0)
    if l0 is None:
        return None
    max_leaf = l0[0]
    _, _, c1, d1 = cpuid(1) or (0, 0, 0, 0)
    flags = {
        "sse4_2": _bit(c1, 20),
        "fma": _bit(c1, 12),
        "f16c": _bit(c1, 29),
        "avx": _bit(c1, 28),
    }
    if max_leaf >= 7:
        _, b7, c7, d7 = cpuid(7, 0) or (0, 0, 0, 0)
        a71, *_ = cpuid(7, 1) or (0, 0, 0, 0)
        flags |= {
            "avx2": _bit(b7, 5),
            "bmi2": _bit(b7, 8),
            "avx512f": _bit(b7, 16),
            "avx512dq": _bit(b7, 17),
            "avx512cd": _bit(b7, 28),
            "avx512bw": _bit(b7, 30),
            "avx512vl": _bit(b7, 31),
            "avx512_vbmi": _bit(c7, 1),
            "avx512_vbmi2": _bit(c7, 6),
            "avx512_vnni": _bit(c7, 11),
            "avx512_bitalg": _bit(c7, 12),
            "avx512_vpopcntdq": _bit(c7, 14),
            "avx512_fp16": _bit(d7, 23),
            "amx_bf16": _bit(d7, 22),
            "amx_int8": _bit(d7, 25),
            "avx_vnni": _bit(a71, 4),
            "avx512_bf16": _bit(a71, 5),
        }
    return flags


def vendor_string() -> str | None:
    r = cpuid(0)
    if r is None:
        return None
    _, b, c, d = r
    return b"".join(x.to_bytes(4, "little") for x in (b, d, c)).decode("ascii", "replace")
