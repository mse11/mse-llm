"""HardwareInfo schema. Every probe is never-fatal: fields are Optional, each section has `source`
and `error` so a missing tool shows as unknown instead of being assumed fine (rule taken from Strata)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class HostInfo(_M):
    hostname: str | None = None
    os: str | None = None            # Windows / Linux / Darwin
    os_release: str | None = None    # "11"
    os_version: str | None = None    # "10.0.26300"
    arch: str | None = None
    python: str | None = None


class CpuInfo(_M):
    model: str | None = None
    vendor: str | None = None
    physical_cores: int | None = None
    logical_cpus: int | None = None
    hybrid: bool | None = None
    p_core_cpus: list[int] | None = None       # logical CPU ids on performance cores (hybrid only)
    e_core_cpus: list[int] | None = None
    l3_groups: list[list[int]] | None = None   # logical CPUs per L3 cache (= one CCD on AMD)
    isa: dict[str, bool] | None = None
    source: str | None = None
    error: str | None = None


class MemModule(_M):
    locator: str | None = None
    bank: str | None = None
    capacity_gb: float | None = None
    rated_mts: int | None = None
    configured_mts: int | None = None
    type: str | None = None
    manufacturer: str | None = None
    part: str | None = None


class MemoryInfo(_M):
    total_bytes: int | None = None
    available_bytes: int | None = None
    pressure: str | None = None                # low / medium / high
    commit_total_bytes: int | None = None
    commit_limit_bytes: int | None = None
    swap_total_bytes: int | None = None
    swap_used_bytes: int | None = None
    modules: list[MemModule] = Field(default_factory=list)
    layout: str | None = None                  # e.g. "16 + 64 GB, asymmetric: 32 GB dual-channel, 48 GB single"
    source: str | None = None
    error: str | None = None


class GpuInfo(_M):
    index: int | None = None                   # NVML index (CUDA order with default CUDA_DEVICE_ORDER)
    vendor: str | None = None                  # nvidia / amd / intel / other
    name: str | None = None
    integrated: bool = False
    uuid: str | None = None
    pci_bus_id: str | None = None
    luid: str | None = None
    vram_total_mib: int | None = None
    vram_free_mib: int | None = None
    compute_capability: str | None = None
    pcie_gen_current: int | None = None
    pcie_gen_max: int | None = None
    pcie_width_current: int | None = None
    pcie_width_max: int | None = None
    mem_bus_width_bits: int | None = None
    mem_clock_max_mhz: int | None = None
    bandwidth_gbs_est: float | None = None
    power_limit_w: float | None = None
    temperature_c: int | None = None
    driver: str | None = None
    cuda_driver: str | None = None
    source: str | None = None


class PowerInfo(_M):
    ac_online: bool | None = None
    has_battery: bool | None = None
    battery_percent: int | None = None
    power_plan: str | None = None
    power_plan_guid: str | None = None
    source: str | None = None
    error: str | None = None


class DiskInfo(_M):
    label: str
    path: str
    total_bytes: int | None = None
    free_bytes: int | None = None
    error: str | None = None


class DriverSettings(_M):
    sysmem_fallback: str | None = None         # prefer_no_fallback / prefer_fallback / driver_default
    sysmem_fallback_global: str | None = None
    sysmem_fallback_programs: dict[str, str] = Field(default_factory=dict)
    setting_id: str | None = None
    source: str | None = None
    error: str | None = None


class HardwareInfo(_M):
    schema_version: int = 1
    detected_at: str | None = None
    fixture: str | None = None                 # set when loaded from MSE_LLM_TEST_HARDWARE / --hw-fixture
    host: HostInfo = Field(default_factory=HostInfo)
    cpu: CpuInfo = Field(default_factory=CpuInfo)
    memory: MemoryInfo = Field(default_factory=MemoryInfo)
    gpus: list[GpuInfo] = Field(default_factory=list)
    power: PowerInfo = Field(default_factory=PowerInfo)
    disks: list[DiskInfo] = Field(default_factory=list)
    driver: DriverSettings = Field(default_factory=DriverSettings)
    errors: list[str] = Field(default_factory=list)

    def nvidia_gpus(self) -> list[GpuInfo]:
        return [g for g in self.gpus if g.vendor == "nvidia"]

    def primary_gpu(self) -> GpuInfo | None:
        nv = self.nvidia_gpus()
        if nv:
            return min(nv, key=lambda g: g.index if g.index is not None else 99)
        discrete = [g for g in self.gpus if not g.integrated]
        return discrete[0] if discrete else None
