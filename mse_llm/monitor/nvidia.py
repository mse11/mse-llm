"""NVML sampler: dedicated VRAM, temperature, power, clocks, utilization, throttle reasons, PCIe (§5)."""

from __future__ import annotations

from typing import Any


def _try(f, *a):
    try:
        return f(*a)
    except Exception:
        return None


# nvmlClocksEventReasons / nvmlClocksThrottleReasons bits worth naming
THROTTLE_BITS = {
    0x1: "idle", 0x2: "app-clocks", 0x4: "sw-power-cap", 0x8: "hw-slowdown", 0x10: "sync-boost",
    0x20: "sw-thermal", 0x40: "hw-thermal", 0x80: "hw-power-brake", 0x100: "display-clocks",
}


def throttle_names(mask: int | None) -> list[str]:
    if not mask:
        return []
    return [n for b, n in THROTTLE_BITS.items() if mask & b and b != 0x1]


class NvmlSampler:
    """One NVIDIA GPU (index), NVML initialised once for the monitor's lifetime."""

    name = "nvml"

    def __init__(self, index: int = 0, pcie: bool = True) -> None:
        self.index = index
        self.pcie = pcie
        self.ok = False
        self.error: str | None = None
        self.total_mib: int | None = None
        self._n: Any = None
        self._h: Any = None

    def open(self) -> None:
        try:
            import pynvml as n

            n.nvmlInit()
            self._n, self._h = n, n.nvmlDeviceGetHandleByIndex(self.index)
            mem = n.nvmlDeviceGetMemoryInfo(self._h)
            self.total_mib = mem.total // 2**20
            self.ok = True
        except Exception as e:
            self.error = f"NVML: {e}"

    def close(self) -> None:
        if self._n is not None:
            _try(self._n.nvmlShutdown)
            self._n = None
            self.ok = False

    def sample(self, pids: set[int]) -> dict[str, Any]:
        if not self.ok:
            return {}
        n, h = self._n, self._h
        out: dict[str, Any] = {}
        mem = _try(n.nvmlDeviceGetMemoryInfo, h)
        if mem:
            out["vram_used_mib"] = mem.used // 2**20
            out["vram_free_mib"] = mem.free // 2**20
            out["vram_total_mib"] = mem.total // 2**20
        out["gpu_temp_c"] = _try(n.nvmlDeviceGetTemperature, h, n.NVML_TEMPERATURE_GPU)
        p = _try(n.nvmlDeviceGetPowerUsage, h)
        out["gpu_power_w"] = round(p / 1000, 1) if p is not None else None
        out["gpu_sm_mhz"] = _try(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_SM)
        out["gpu_mem_mhz"] = _try(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_MEM)
        u = _try(n.nvmlDeviceGetUtilizationRates, h)
        out["gpu_util_pct"] = u.gpu if u else None
        reasons = getattr(n, "nvmlDeviceGetCurrentClocksEventReasons", None) or getattr(
            n, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
        out["gpu_throttle"] = _try(reasons, h) if reasons else None
        if self.pcie:
            tx = _try(n.nvmlDeviceGetPcieThroughput, h, n.NVML_PCIE_UTIL_TX_BYTES)
            rx = _try(n.nvmlDeviceGetPcieThroughput, h, n.NVML_PCIE_UTIL_RX_BYTES)
            out["pcie_tx_mbs"] = round(tx / 1024, 1) if tx is not None else None   # KB/s -> MB/s
            out["pcie_rx_mbs"] = round(rx / 1024, 1) if rx is not None else None
        if pids:   # per-process VRAM (None on Windows WDDM: PDH gives it instead)
            used = 0
            seen = False
            for pr in _try(n.nvmlDeviceGetComputeRunningProcesses, h) or []:
                v = getattr(pr, "usedGpuMemory", None)
                if pr.pid in pids and isinstance(v, int) and v > 0:
                    used += v
                    seen = True
            if seen:
                out["proc_vram_mib"] = used // 2**20
        return {k: v for k, v in out.items() if v is not None}
