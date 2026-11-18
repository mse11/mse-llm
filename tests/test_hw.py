import platform

import pytest

from mse_llm import hw
from mse_llm.config import HwChecksCfg
from mse_llm.hw.gpu_windows import VIRTUAL_RE, luid_str
from mse_llm.hw.memory import describe_layout, pressure_label
from mse_llm.hw.models import MemModule

# Hash of the captured reference laptop (tests/fixtures/hw/rtx4080-laptop.json). If this changes,
# stored sweet spots / build profiles keyed by the hash would no longer match: change on purpose only.
REFERENCE_HASH = "20725a44cc6f18bc"


def test_fixture_reference_laptop(hw_fixture):
    info = hw.detect()
    assert info.fixture
    gpu = info.primary_gpu()
    assert gpu.name == "NVIDIA GeForce RTX 4080 Laptop GPU"
    assert gpu.vram_total_mib == 12282
    assert gpu.compute_capability == "8.9"
    assert (gpu.pcie_gen_max, gpu.pcie_width_max) == (4, 16)
    assert info.cpu.physical_cores == 16 and info.cpu.logical_cpus == 32
    assert info.cpu.hybrid is False
    assert info.cpu.l3_groups == [list(range(16)), list(range(16, 32))]
    assert all(info.cpu.isa[k] for k in ("avx512f", "avx512_vnni", "avx512_vbmi", "avx512_bf16"))
    assert info.memory.layout == "16 + 64 GB, asymmetric: 32 GB dual-channel, 48 GB single-channel"
    assert info.power.ac_online is True
    assert any(g.integrated and g.vendor == "amd" for g in info.gpus)


def test_identity_hash_is_stable(hw_fixture):
    a = hw.identity(hw.detect())
    b = hw.identity(hw.detect())
    assert a["hash"] == b["hash"] == REFERENCE_HASH


def test_identity_ignores_soft_and_live_values(hw_fixture):
    info = hw.detect()
    base = hw.identity(info)["hash"]
    info.gpus[0].driver = "999.99"
    info.gpus[0].vram_free_mib = 1
    info.gpus[0].temperature_c = 90
    info.gpus[0].power_limit_w = 55.0
    info.memory.available_bytes = 1
    info.host.hostname = "other"
    assert hw.identity(info)["hash"] == base
    info.gpus[0].vram_total_mib = 8192
    assert hw.identity(info)["hash"] != base


def test_checks_reference_laptop_pass(hw_fixture):
    checks = hw.checks(hw.detect(), HwChecksCfg())
    assert {c.status for c in checks} == {"pass"}


def test_checks_battery_and_small_ram(hw_fixture):
    info = hw.detect()
    info.power.ac_online = False
    info.memory.total_bytes = 8 * 2**30
    st = {c.name: c.status for c in hw.checks(info)}
    assert st["power"] == "warn"
    assert st["RAM"] == "fail"


def test_inline_json_fixture(monkeypatch):
    monkeypatch.setenv("MSE_LLM_TEST_HARDWARE", '{"cpu": {"model": "Test CPU", "physical_cores": 2}}')
    info = hw.detect()
    assert info.cpu.model == "Test CPU" and info.fixture == "inline"
    assert {c.name: c.status for c in hw.checks(info)}["CPU cores"] == "fail"


def test_pressure_label():
    assert pressure_label(40, 100) == "low"
    assert pressure_label(20, 100) == "medium"
    assert pressure_label(10, 100) == "high"


def test_describe_layout():
    mods = [MemModule(capacity_gb=16), MemModule(capacity_gb=64)]
    assert describe_layout(mods) == "16 + 64 GB, asymmetric: 32 GB dual-channel, 48 GB single-channel"
    assert describe_layout([MemModule(capacity_gb=32), MemModule(capacity_gb=32)]) == "32 + 32 GB, symmetric"


def test_luid_and_virtual_filter():
    assert luid_str(0x00000000_00011C84) == "luid_0x00000000_0x00011C84"
    assert VIRTUAL_RE.search("Microsoft Basic Render Driver")
    assert not VIRTUAL_RE.search("NVIDIA GeForce RTX 4080 Laptop GPU")


@pytest.mark.skipif(platform.machine().lower() not in ("amd64", "x86_64"), reason="x86-64 only")
def test_cpuid_live():
    from mse_llm.hw import cpuid

    assert len(cpuid.vendor_string()) == 12
    flags = cpuid.isa_flags()
    assert flags["sse4_2"] is True


def test_live_detect_identity_stable():
    """Real probes on this machine: never raise, and the hash is stable across runs."""
    a = hw.identity(hw.detect(force=True))["hash"]
    b = hw.identity(hw.detect(force=True))["hash"]
    assert a == b
