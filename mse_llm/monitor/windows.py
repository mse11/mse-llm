"""PDH sampler (Windows): GPU shared / non-local memory, adapter-wide and per server process (§5).

No powershell: the counters are read through win32pdh with open queries (one Collect per tick).

    nonlocal_mib     sum of \\GPU Non Local Adapter Memory(*)\\Non Local Usage: the bash script's "shared"
                     number (Win32_PerfRawData_GPUPerformanceCounters_GPUNonLocalAdapterMemory), for parity
    shared_mib       \\GPU Adapter Memory(<NVIDIA luid>)\\Shared Usage: the NVIDIA GPU only (the iGPU's own
                     shared memory is not mixed in)
    proc_shared_mib  \\GPU Process Memory(pid_<server tree>_<luid>)\\Shared Usage
    proc_vram_mib    \\GPU Process Memory(pid_<server tree>_<luid>)\\Dedicated Usage

Shared memory of a CUDA process is not only a spill: llama.cpp's pinned host buffers (CUDA_Host, the
experts that -ncmoe keeps on the CPU) count too. A fitting Qwen3.6-28B `moe41` shows ~15.9 GB here,
equal to the host part from `fit` (15.8 GB). See monitor.Guard for how a real spill is told apart.
"""

from __future__ import annotations

import time
from typing import Any

MIB = 2 ** 20


class PdhSampler:
    name = "pdh"

    def __init__(self, luid: str | None = None, refresh_s: float = 2.0) -> None:
        self.luid = luid             # e.g. luid_0x00000000_0x00011C84 (NVIDIA adapter); None = sum of all
        self.refresh_s = refresh_s
        self.ok = False
        self.error: str | None = None
        self._pdh: Any = None
        self._q: Any = None
        self._adapter: list[tuple[str, Any]] = []
        self._pq: Any = None
        self._proc: list[tuple[str, Any]] = []
        self._proc_pids: frozenset[int] = frozenset()
        self._proc_t = 0.0

    def open(self) -> None:
        try:
            import win32pdh as pdh
        except ImportError as e:
            self.error = f"PDH: {e}"
            return
        self._pdh = pdh
        try:
            self._q = pdh.OpenQuery()
            nonlocal_paths = pdh.ExpandCounterPath(r"\GPU Non Local Adapter Memory(*)\Non Local Usage")
            paths = [("nonlocal", p) for p in nonlocal_paths]
            inst = f"{self.luid}*" if self.luid else "*"
            paths += [("shared", p) for p in pdh.ExpandCounterPath(rf"\GPU Adapter Memory({inst})\Shared Usage")]
            paths += [("dedicated", p) for p in pdh.ExpandCounterPath(rf"\GPU Adapter Memory({inst})\Dedicated Usage")]
            self._adapter = [(k, pdh.AddCounter(self._q, p)) for k, p in paths]
            pdh.CollectQueryData(self._q)
            self.ok = True
        except Exception as e:
            self.error = f"PDH: {e}"

    def close(self) -> None:
        for q in (self._q, self._pq):
            if q is not None:
                try:
                    self._pdh.CloseQuery(q)
                except Exception:
                    pass
        self._q = self._pq = None
        self.ok = False

    def _value(self, c: Any) -> int | None:
        try:
            return self._pdh.GetFormattedCounterValue(c, self._pdh.PDH_FMT_LARGE)[1]
        except Exception:
            return None

    def _proc_counters(self, pids: set[int]) -> None:
        """(Re)build the per-process query: new pids, or every refresh_s (instances appear once a process
        allocates GPU memory)."""
        key = frozenset(pids)
        if key == self._proc_pids and time.monotonic() - self._proc_t < self.refresh_s:
            return
        pdh = self._pdh
        if self._pq is not None:
            try:
                pdh.CloseQuery(self._pq)
            except Exception:
                pass
        self._pq, self._proc = None, []
        self._proc_pids, self._proc_t = key, time.monotonic()
        if not pids:
            return
        luid = self.luid or ""
        self._pq = pdh.OpenQuery()
        for pid in pids:
            for kind, counter in (("proc_shared", "Shared Usage"), ("proc_vram", "Dedicated Usage")):
                try:
                    paths = pdh.ExpandCounterPath(rf"\GPU Process Memory(pid_{pid}_*)\{counter}")
                except Exception:
                    continue
                for p in paths:
                    if luid and luid.lower() not in p.lower():
                        continue
                    try:
                        self._proc.append((kind, pdh.AddCounter(self._pq, p)))
                    except Exception:
                        pass

    def sample(self, pids: set[int]) -> dict[str, Any]:
        if not self.ok:
            return {}
        pdh = self._pdh
        out: dict[str, Any] = {}
        try:
            pdh.CollectQueryData(self._q)
        except Exception:
            return {}
        sums: dict[str, int] = {}
        for kind, c in self._adapter:
            v = self._value(c)
            if v is not None:
                sums[kind] = sums.get(kind, 0) + v
        if "nonlocal" in sums:
            out["nonlocal_mib"] = sums["nonlocal"] // MIB
        if "shared" in sums:
            out["shared_mib"] = sums["shared"] // MIB
        if "dedicated" in sums:
            out["pdh_dedicated_mib"] = sums["dedicated"] // MIB
        self._proc_counters(pids)
        if self._pq is not None and self._proc:
            try:
                pdh.CollectQueryData(self._pq)
                ps: dict[str, int] = {}
                for kind, c in self._proc:
                    v = self._value(c)
                    if v is not None:
                        ps[kind] = ps.get(kind, 0) + v
                for kind, v in ps.items():
                    out[f"{kind}_mib"] = v // MIB
            except Exception:
                pass
        if pids and "proc_shared_mib" not in out:
            out["proc_shared_mib"] = 0           # no GPU allocation yet
        return out


def nvidia_luid() -> str | None:
    """PDH instance prefix of the first NVIDIA adapter (DirectX registry)."""
    try:
        from mse_llm.hw.gpu_windows import detect_dx_adapters

        gpus, _ = detect_dx_adapters()
        return next((g.luid for g in gpus if g.vendor == "nvidia" and g.luid), None)
    except Exception:
        return None
