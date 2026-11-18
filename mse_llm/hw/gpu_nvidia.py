"""NVIDIA GPU inventory through NVML (nvidia-ml-py), no nvidia-smi."""

from __future__ import annotations

from mse_llm.hw.models import GpuInfo


def _try(f, *a):
    try:
        return f(*a)
    except Exception:
        return None


def _s(v) -> str | None:
    if v is None:
        return None
    return v.decode() if isinstance(v, bytes) else str(v)


def detect_nvidia() -> tuple[list[GpuInfo], str | None]:
    try:
        import pynvml as n
    except ImportError as e:
        return [], f"nvidia-ml-py not installed: {e}"
    try:
        n.nvmlInit()
    except Exception as e:
        return [], f"NVML init failed: {e}"
    try:
        driver = _s(_try(n.nvmlSystemGetDriverVersion))
        cuda = _try(n.nvmlSystemGetCudaDriverVersion_v2) or _try(n.nvmlSystemGetCudaDriverVersion)
        cuda_s = f"{cuda // 1000}.{(cuda % 1000) // 10}" if cuda else None
        out = []
        for i in range(n.nvmlDeviceGetCount()):
            h = n.nvmlDeviceGetHandleByIndex(i)
            mem = _try(n.nvmlDeviceGetMemoryInfo, h)
            cc = _try(n.nvmlDeviceGetCudaComputeCapability, h)
            pci = _try(n.nvmlDeviceGetPciInfo, h)
            width = _try(n.nvmlDeviceGetMemoryBusWidth, h)
            mclk = _try(n.nvmlDeviceGetMaxClockInfo, h, n.NVML_CLOCK_MEM)
            plimit = _try(n.nvmlDeviceGetEnforcedPowerLimit, h)
            # GDDR6/6X: NVML reports half the per-pin data rate in MHz -> Gbps = 2 * MHz / 1000.
            bw = round(width / 8 * mclk * 2 / 1000, 1) if width and mclk else None
            out.append(GpuInfo(
                index=i,
                vendor="nvidia",
                name=_s(_try(n.nvmlDeviceGetName, h)),
                uuid=_s(_try(n.nvmlDeviceGetUUID, h)),
                pci_bus_id=_s(pci.busId) if pci else None,
                vram_total_mib=mem.total // 2**20 if mem else None,
                vram_free_mib=mem.free // 2**20 if mem else None,
                compute_capability=f"{cc[0]}.{cc[1]}" if cc else None,
                pcie_gen_current=_try(n.nvmlDeviceGetCurrPcieLinkGeneration, h),
                pcie_gen_max=_try(n.nvmlDeviceGetGpuMaxPcieLinkGeneration, h)
                or _try(n.nvmlDeviceGetMaxPcieLinkGeneration, h),
                pcie_width_current=_try(n.nvmlDeviceGetCurrPcieLinkWidth, h),
                pcie_width_max=_try(n.nvmlDeviceGetMaxPcieLinkWidth, h),
                mem_bus_width_bits=width,
                mem_clock_max_mhz=mclk,
                bandwidth_gbs_est=bw,
                power_limit_w=round(plimit / 1000, 1) if plimit else None,
                temperature_c=_try(n.nvmlDeviceGetTemperature, h, n.NVML_TEMPERATURE_GPU),
                driver=driver,
                cuda_driver=cuda_s,
                source="nvml",
            ))
        return out, None
    except Exception as e:
        return [], f"NVML query failed: {e}"
    finally:
        _try(n.nvmlShutdown)


def compute_processes() -> list[tuple[int, int | None]]:
    """(pid, used MiB or None) of processes with a CUDA context, on every NVIDIA GPU."""
    try:
        import pynvml as n

        n.nvmlInit()
    except Exception:
        return []
    try:
        out = []
        for i in range(n.nvmlDeviceGetCount()):
            h = n.nvmlDeviceGetHandleByIndex(i)
            for p in _try(n.nvmlDeviceGetComputeRunningProcesses, h) or []:
                used = getattr(p, "usedGpuMemory", None)
                out.append((p.pid, used // 2**20 if isinstance(used, int) and used > 0 else None))
        return out
    finally:
        _try(n.nvmlShutdown)
