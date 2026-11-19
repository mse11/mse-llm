"""`mse-llm doctor`: environment, toolchain, engine, hardware and driver checks."""

from __future__ import annotations

import re
import shutil
import socket
import sys
from pathlib import Path

from mse_llm import hw
from mse_llm.buildprofiles import BuildProfile, list_profiles
from mse_llm.checks import Check
from mse_llm.config import Config
from mse_llm.hw.models import HardwareInfo
from mse_llm.util import IS_WINDOWS, exe, run

# Deepest path seen relative to the build dir (CUDA .obj/.tlog files), §0.4.
DEEPEST_REL_DEFAULT = 165
PATH_WARN = 240
WIN_MAX_PATH = 260


def _ver(text: str) -> tuple[int, ...] | None:
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    return tuple(int(x) for x in m.groups() if x is not None) if m else None


def _tool_check(name: str, argv: list[str], need: tuple[int, ...] | None, why: str) -> Check:
    r = run(argv, timeout=20)
    if r.error:
        return Check("toolchain", name, "warn", r.error, f"needed for {why}")
    v = _ver(r.stdout + r.stderr)
    vs = ".".join(map(str, v)) if v else "?"
    if need and v and v < need:
        return Check("toolchain", name, "warn", f"{vs} (< {'.'.join(map(str, need))})", f"needed for {why}")
    return Check("toolchain", name, "pass", vs)


def environment_checks() -> list[Check]:
    out = []
    py = sys.version_info
    out.append(Check("environment", "python", "pass" if py >= (3, 12) else "fail",
                     f"{py.major}.{py.minor}.{py.micro} ({sys.executable})"))
    for tool, why in (("uv", "the project"), ("uvx", "llama-benchy / inspect-ai")):
        p = shutil.which(tool)
        out.append(Check("environment", tool, "pass" if p else "warn", p or "not on PATH",
                         None if p else f"install uv; needed for {why}"))
    return out


def _nvcc() -> list[str]:
    import os

    cp = os.environ.get("CUDA_PATH")
    if cp and Path(cp, "bin", "nvcc.exe" if IS_WINDOWS else "nvcc").is_file():
        return [str(Path(cp, "bin", "nvcc")), "--version"]
    return ["nvcc", "--version"]


def toolchain_checks() -> list[Check]:
    why = "mse-llm build"
    out = [
        _tool_check("git", ["git", "--version"], (2, 25), why),
        _tool_check("cmake", ["cmake", "--version"], (3, 21), why),
        _tool_check("nvcc", _nvcc(), (12, 4), why),
    ]
    if IS_WINDOWS:
        vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
        if not vswhere.is_file():
            out.append(Check("toolchain", "MSVC", "warn", "vswhere.exe not found", f"VS2022 C++ workload, {why}"))
        else:
            r = run([str(vswhere), "-latest", "-products", "*", "-requires",
                     "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                     "-property", "catalog_productDisplayVersion"])
            ver = r.stdout.strip()
            st = "pass" if ver.startswith("17.") else "warn"
            out.append(Check("toolchain", "MSVC", st, f"VS {ver}" if ver else "no VS with C++ tools",
                             None if st == "pass" else f"VS2022 C++ workload, {why}"))
    return out


def long_path_checks(profiles: list[BuildProfile]) -> list[Check]:
    if not IS_WINDOWS:
        return []
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            enabled = int(winreg.QueryValueEx(k, "LongPathsEnabled")[0]) == 1
    except OSError:
        enabled = False
    if enabled:
        return [Check("environment", "long paths", "pass", "LongPathsEnabled = 1")]
    out = []
    for p in profiles:
        variants = list((p.data.get("variants") or {}).keys())
        longest = max([p.build_dir.name] + [f"build-without-{v}" for v in variants], key=len)
        depth = len(str(p.src / longest)) + 1 + DEEPEST_REL_DEFAULT
        st = "warn" if depth > PATH_WARN else "pass"
        out.append(Check(
            "environment", f"path length ({p.name})", st,
            f"long paths disabled: deepest build path ≈ {depth} of {WIN_MAX_PATH} chars",
            "set a shorter MSE_LLM_BUILDS_ROOT (e.g. D:\\b) or enable long paths in Windows"
            if st == "warn" else None,
        ))
    return out


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "") else host, port)) != 0


def engine_checks(cfg: Config, info: HardwareInfo) -> list[Check]:
    out = []
    if not cfg.engines:
        return [Check("engine", "engines", "warn", "no engine configured")]
    gpu_match = {p.name: p.gpu_match for p in list_profiles()}
    for name, eng in cfg.engines.items():
        sec = f"engine {name}"
        bdir = eng.bin_path()
        missing = [t for t in ("llama-server", "llama-bench") if not exe(bdir / t).is_file()]
        if missing:
            hint = (f"mse-llm build run {eng.build_profile}" if eng.build_profile else "build llama.cpp")
            out.append(Check(sec, "binaries", "warn", f"not built yet: {', '.join(missing)} missing in {bdir}",
                             hint))
        else:
            r = run([str(exe(bdir / "llama-server")), "--version"], timeout=30)
            txt = (r.stdout + r.stderr).strip().splitlines()
            vline = next((x for x in txt if "version" in x), txt[-1] if txt else "?")
            out.append(Check(sec, "binaries", "pass" if r.ok else "fail", f"{bdir} ({vline.strip()})"))
            r = run([str(exe(bdir / "llama-bench")), "--list-devices"], timeout=60)
            m = re.search(r"CUDA0:\s*([^(\n]+)", r.stdout + r.stderr)
            want = gpu_match.get(eng.build_profile or "")
            if m:
                dev0 = m.group(1).strip()
                ok = not want or want.lower() in dev0.lower()
                out.append(Check(sec, "CUDA device 0", "pass" if ok else "warn", dev0,
                                 None if ok else f"expected {want}; check CUDA_VISIBLE_DEVICES"))
            else:
                out.append(Check(sec, "CUDA device 0", "warn", "no CUDA0 in llama-bench --list-devices"))
        free = _port_free(eng.host, eng.port)
        out.append(Check(sec, f"port {eng.port}", "pass" if free else "warn",
                         "free" if free else "in use (a server is already running?)"))
    return out


def monitor_checks() -> list[Check]:
    out = []
    try:
        import pynvml

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        out.append(Check("monitoring", "NVML", "pass", f"{n} device(s)"))
    except Exception as e:
        out.append(Check("monitoring", "NVML", "warn", str(e), "no VRAM/temperature monitoring"))
    if IS_WINDOWS:
        try:
            import win32pdh

            paths = win32pdh.ExpandCounterPath(r"\GPU Adapter Memory(*)\Shared Usage")
            out.append(Check("monitoring", "PDH shared GPU memory", "pass", f"{len(paths)} adapter counter(s)"))
        except Exception as e:
            out.append(Check("monitoring", "PDH shared GPU memory", "warn", str(e),
                             "spills into shared memory can't be detected"))
    return out


def driver_checks(info: HardwareInfo) -> list[Check]:
    if not info.nvidia_gpus():
        return []
    d = info.driver
    sec = "driver"
    gpu = info.nvidia_gpus()[0]
    out = [Check(sec, "NVIDIA driver", "pass", f"{gpu.driver} (CUDA {gpu.cuda_driver})")]
    if d.error:
        out.append(Check(sec, "CUDA sysmem fallback", "warn", f"can't read: {d.error}"))
        return out
    hint = ("NVIDIA Control Panel -> Manage 3D settings -> CUDA - Sysmem Fallback Policy = "
            "Prefer No Sysmem Fallback (Global, or per program for llama-server.exe, llama-bench.exe, "
            "llama-perplexity.exe)")
    progs = ", ".join(f"{k}={v}" for k, v in d.sysmem_fallback_programs.items())
    detail = f"global={d.sysmem_fallback_global}" + (f"; {progs}" if progs else "")
    if d.sysmem_fallback == "prefer_no_fallback":
        out.append(Check(sec, "CUDA sysmem fallback", "pass", detail))
    else:
        note = ("" if d.sysmem_fallback != "driver_default" else
                " - not set: the driver may spill VRAM into system RAM (laptop freezes). "
                f"Setting id {d.setting_id} is from community metadata; once set in the Control Panel "
                "this check confirms it")
        out.append(Check(sec, "CUDA sysmem fallback", "warn", detail + note, hint))
    return out


def target_checks(info: HardwareInfo) -> list[Check]:
    out = []
    gpu = info.primary_gpu()
    for p in list_profiles():
        msgs = []
        if p.gpu_match and not (gpu and gpu.name and p.gpu_match.lower() in gpu.name.lower()):
            msgs.append(f"GPU {gpu.name if gpu else None!r} doesn't match {p.gpu_match!r}")
        if p.cpu_match and p.cpu_match.lower() not in (info.cpu.model or "").lower():
            msgs.append(f"CPU {info.cpu.model!r} doesn't match {p.cpu_match!r}")
        out.append(Check("hardware", f"build target {p.name}", "warn" if msgs else "pass",
                         "; ".join(msgs) or f"matches {p.gpu_match} / {p.cpu_match}"))
    return out


def process_checks() -> list[Check]:
    from mse_llm.hw.processes import detect_processes

    engines = [p for p in detect_processes() if p.kind == "engine"]
    if engines:
        names = ", ".join(f"{p.name} (pid {p.pid})" for p in engines)
        return [Check("hardware", "other engines", "warn", names, "a second engine changes every number")]
    return [Check("hardware", "other engines", "pass", "none running")]


def disk_paths(cfg: Config) -> dict[str, Path]:
    from mse_llm.buildprofiles import builds_root

    paths: dict[str, Path] = {"builds": builds_root()}
    for m in cfg.models.values():
        anchor = Path(m.path).anchor
        if anchor and all(Path(p).anchor.lower() != anchor.lower() for p in paths.values()):
            paths[f"models {anchor.rstrip(chr(92))}"] = Path(m.path).parent
    return paths


def model_checks(cfg: Config) -> list[Check]:
    if not cfg.models:
        return [Check("config", "models", "info", "no models configured")]
    missing = [k for k, m in cfg.models.items() if not Path(m.path).is_file()]
    if missing:
        return [Check("config", "models", "warn", f"{len(missing)}/{len(cfg.models)} GGUF missing: "
                      + ", ".join(missing))]
    return [Check("config", "models", "pass", f"{len(cfg.models)} GGUF files found")]


def build_checks(cfg: Config, info: HardwareInfo) -> list[Check]:
    """Per engine built by `mse-llm build`: vcpkg toolchain file, FA_ALL_QUANTS and CUDA arch in CMakeCache."""
    from mse_llm import build as b

    out: list[Check] = []
    gpu = info.primary_gpu()
    for name, eng in cfg.engines.items():
        m = b.read_manifest(eng.bin_path())
        if not m:
            continue
        sec = f"engine {name}"
        bdir = Path(m.get("build_dir", ""))
        cache = b.read_cmake_cache(bdir)
        if not cache:
            out.append(Check(sec, "CMakeCache", "warn", f"missing in {bdir}"))
            continue
        tc = cache.get("CMAKE_TOOLCHAIN_FILE")
        if tc:
            ok = Path(tc).is_file()
            out.append(Check(sec, "vcpkg toolchain", "pass" if ok else "warn", tc,
                             None if ok else "vcpkg is gone; `mse-llm build clone` restores it"))
        fa = b.same_value("ON", cache.get("GGML_CUDA_FA_ALL_QUANTS"))
        out.append(Check(sec, "FA all quants", "pass" if fa else "warn",
                         f"GGML_CUDA_FA_ALL_QUANTS={cache.get('GGML_CUDA_FA_ALL_QUANTS', 'unset')}",
                         None if fa else "asymmetric K/V sweeps (Kq8_Vq4 ...) would measure a slow fallback"))
        archs = cache.get("CMAKE_CUDA_ARCHITECTURES", "")
        cov = b.cuda_arch_covers(archs, gpu.compute_capability if gpu else None)
        if cov is not None:
            out.append(Check(sec, "CUDA arch", "pass" if cov else "fail",
                             f"{archs} vs cc {gpu.compute_capability}",
                             None if cov else "this build can't run on the detected GPU; use another profile"))
        out.append(Check(sec, "build", "info", f"{m.get('describe')} built {m.get('built_at')} ({m.get('target')})"))
    return out


def run_doctor(cfg: Config, info: HardwareInfo | None = None) -> list[Check]:
    info = info or hw.detect(disk_paths=disk_paths(cfg))
    profiles = list_profiles()
    out: list[Check] = []
    out += environment_checks()
    out += long_path_checks(profiles)
    out.append(Check("config", "config file", "pass" if cfg.source else "info",
                     str(cfg.source) if cfg.source else "none (built-in defaults)"))
    out.append(Check("config", "build profiles", "pass" if profiles else "warn",
                     ", ".join(p.name for p in profiles) or "none in config/builds"))
    out += model_checks(cfg)
    out += toolchain_checks()
    out += hw.checks(info, cfg.hw.checks)
    out += target_checks(info)
    out += process_checks()
    out += driver_checks(info)
    out += monitor_checks()
    out += engine_checks(cfg, info)
    out += build_checks(cfg, info)
    return group_by_section(out)


def group_by_section(checks: list[Check]) -> list[Check]:
    """Keep the checks of one section together (in the order sections first appear), so e.g. all
    `engine cuda_flash` rows come before `engine cuda_flash-without-…`."""
    first: dict[str, int] = {}
    for i, c in enumerate(checks):
        first.setdefault(c.section, i)
    return sorted(checks, key=lambda c: first[c.section])   # stable: order inside a section is kept
