"""`mse-llm build`: clone, configure, compile and verify llama.cpp from a build profile (§0.4).

A *target* is a build profile (config/builds/<name>.toml) plus variants:
  rtx4080-laptop                          default variants (apply_variants) -> build/
  rtx4080-laptop --without zen4-avx512x   comparison build                  -> build-without-zen4-avx512x/
  rtx4080-laptop+<variant>                an optional variant on top        -> build-<variant>/
  rtx4080-laptop --ref b12468 --keep-previous                               -> build-b12468/

`plan_steps` only reads files and runs read-only git queries; `execute` runs the steps.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import tomllib
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mse_llm import __version__
from mse_llm.buildprofiles import BuildProfile, builds_config_dir, list_profiles, load_profile
from mse_llm.checks import Check
from mse_llm.util import IS_WINDOWS, exe, project_root, run

STEPS: tuple[tuple[str, str], ...] = (
    ("clone", "Clone llama.cpp"),
    ("checkout", "Checkout version"),
    ("vcpkg-clone", "Clone vcpkg"),
    ("vcpkg-bootstrap", "Bootstrap vcpkg"),
    ("vcpkg-install", "Install libraries"),
    ("configure", "Configure (CMake)"),
    ("compile", "Compile"),
    ("smoke", "Smoke test"),
    ("verify", "Verify flags"),
    ("manifest", "Write manifest"),
    ("help-snapshot", "Help snapshot"),
    ("engine", "Update engine"),
)
STEP_IDS = tuple(s for s, _ in STEPS)
CLONE_STEPS = STEP_IDS[:5]
PATH_KEYS = ("project", "builds_root", "src", "vcpkg_root", "build_dir", "bin")
MAIN_BINARIES = ("llama-server", "llama-bench", "llama-fit-params", "llama-perplexity")
MAIN_LIBS = ("llama", "ggml", "ggml-base", "ggml-cpu", "ggml-cuda")
# Deepest path seen relative to a build dir (CUDA .obj/.tlog files), §0.4.
DEEPEST_REL = 165
PATH_WARN = 240


class BuildError(RuntimeError):
    pass


# ---------------------------------------------------------------- target resolution

@dataclass
class Flag:
    key: str
    value: str
    source: str          # base | profile | variant:<name> | auto


@dataclass
class Target:
    profile: BuildProfile
    label: str
    applied: list[str]   # variants in this build (defaults minus --without, plus +extras)
    without: list[str]
    defaults: list[str]  # apply_variants of the profile
    ref: str
    ref_given: bool
    src: Path
    build_dir: Path
    bin_dir: Path
    builds_root: Path
    vcpkg_root: Path
    data: dict[str, Any]  # merged + placeholder-expanded profile
    flags: list[Flag] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def engine_name(self) -> str | None:
        base = self.profile.engine.get("name")
        if not base:
            return None
        default_dir = self.data.get("build_dir", "build")
        suffix = self.build_dir.name[len(default_dir):] if self.build_dir.name.startswith(default_dir) else ""
        return base + suffix

    def defs(self) -> dict[str, str]:
        return {f.key: f.value for f in self.flags if f.source != "auto"}

    @property
    def manifest_path(self) -> Path:
        return self.bin_dir / "build-manifest.json"


def _str(v: Any) -> str:
    if isinstance(v, bool):
        return "ON" if v else "OFF"
    return str(v)


def _expand(obj: Any, ctx: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: _expand(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v, ctx) for v in obj]
    if isinstance(obj, str) and "{" in obj:
        is_path = any("{" + k + "}" in obj for k in PATH_KEYS)
        for k, v in ctx.items():
            obj = obj.replace("{" + k + "}", v)
        return os.path.normpath(obj) if is_path else obj
    return obj


def load_chain(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Profile files from the root base to the profile itself (cycle already checked by load_profile)."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    parent = data.get("extends")
    return (load_chain(path.parent / parent) if parent else []) + [(path, data)]


def _sanitize(ref: str) -> str:
    return re.sub(r"[^\w.-]", "_", ref)


def resolve_target(spec: str, without: tuple[str, ...] | list[str] = (), ref: str | None = None,
                   keep_previous: bool = False) -> Target:
    name, *extras = spec.split("+")
    prof = load_profile(name)
    data = prof.data
    vtable = data.get("variants") or {}
    for v in (*extras, *without):
        if v not in vtable:
            raise BuildError(f"unknown variant {v!r} for {name}; known: {', '.join(vtable) or 'none'}")
    defaults = list(data.get("apply_variants") or [])
    for w in without:
        if w not in defaults:
            raise BuildError(f"--without {w}: not a default variant of {name} "
                             f"(defaults: {', '.join(defaults) or 'none'}); "
                             f"optional variants are added as {name}+<variant>")
    extras = [v for v in extras if v not in defaults]
    applied = [v for v in defaults if v not in without] + extras
    src_cfg = data.get("source") or {}
    ref_given = ref is not None
    ref = ref or src_cfg.get("ref") or "master"

    suffix = "".join(f"-{v}" for v in extras) + "".join(f"-without-{w}" for w in without)
    if keep_previous and ref_given:
        suffix += f"-{_sanitize(ref)}"
    root = prof.builds_root
    ctx = {"project": str(project_root()), "builds_root": str(root)}
    src = Path(_expand(src_cfg.get("src", "{builds_root}/llama.cpp"), ctx))
    ctx["src"] = str(src)
    vroot = Path(_expand((data.get("vcpkg") or {}).get("root", "{builds_root}/vcpkg"), ctx))
    ctx["vcpkg_root"] = str(vroot)
    build_dir = src / (data.get("build_dir", "build") + suffix)
    ctx["build_dir"] = str(build_dir)
    bin_dir = Path(_expand((data.get("outputs") or {}).get("bin_dir", "{build_dir}/bin/Release"), ctx))
    ctx["bin"] = str(bin_dir)
    expanded = _expand(data, ctx)

    label = spec.split("+")[0] + "".join(f"+{v}" for v in extras) + "".join(f" --without {w}" for w in without)
    if ref_given:
        label += f" --ref {ref}"
    t = Target(prof, label, applied, list(without), defaults, ref, ref_given, src, build_dir, bin_dir,
               root, vroot, expanded)
    t.flags = _flags(prof.path, expanded, applied)
    return t


def _flags(profile_path: Path, data: dict[str, Any], applied: list[str]) -> list[Flag]:
    chain = load_chain(profile_path)
    source: dict[str, str] = {}
    tc_source = "base"
    for path, d in chain:
        label = "base" if path.name.startswith("_") else "profile"
        for k in d.get("cmake") or {}:
            source[k] = label
        if (d.get("toolchain") or {}).get("vcpkg_toolchain"):
            tc_source = label
    out: dict[str, Flag] = {}
    tc = (data.get("toolchain") or {}).get("vcpkg_toolchain")
    if tc:
        out["CMAKE_TOOLCHAIN_FILE"] = Flag("CMAKE_TOOLCHAIN_FILE", tc, tc_source)
    for k, v in (data.get("cmake") or {}).items():
        out[k] = Flag(k, _str(v), source.get(k, "profile"))
    for v in applied:
        for k, val in ((data.get("variants") or {}).get(v, {}).get("cmake") or {}).items():
            out[k] = Flag(k, _str(val), f"variant:{v}")
    return list(out.values())


def pick_target(info) -> str:
    """The build profile whose [target] matches the detected GPU and CPU."""
    gpus = [g.name or "" for g in info.gpus]
    cpu = info.cpu.model or ""
    hits = [p.name for p in list_profiles()
            if (not p.gpu_match or any(p.gpu_match.lower() in g.lower() for g in gpus))
            and (not p.cpu_match or p.cpu_match.lower() in cpu.lower())]
    if len(hits) == 1:
        return hits[0]
    names = ", ".join(p.name for p in list_profiles())
    if not hits:
        raise BuildError(f"no build profile matches this PC ({', '.join(gpus) or 'no GPU'}, {cpu}); "
                         f"give a TARGET ({names}) or add config/builds/<name>.toml")
    raise BuildError(f"several profiles match this PC: {', '.join(hits)}; give a TARGET")


def all_targets() -> list[str]:
    """Buildable target specs: each profile, its --without comparisons and +optional variants."""
    out = []
    for p in list_profiles():
        out.append(p.name)
        defaults = p.data.get("apply_variants") or []
        for v in p.data.get("variants") or {}:
            out.append(f"{p.name} --without {v}" if v in defaults else f"{p.name}+{v}")
    return out


# ---------------------------------------------------------------- read-only probes

def git(src: Path, *args: str, timeout: float = 30) -> str | None:
    r = run(["git", "-C", str(src), *args], timeout=timeout)
    return r.stdout.strip() if r.ok else None


def describe(src: Path, rev: str = "HEAD") -> str | None:
    return git(src, "describe", "--tags", "--always", rev)


def is_dirty(src: Path) -> bool:
    out = git(src, "status", "--porcelain", "--untracked-files=no")
    return bool(out)


def read_cmake_cache(build_dir: Path) -> dict[str, str]:
    p = build_dir / "CMakeCache.txt"
    out: dict[str, str] = {}
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"([A-Za-z_][\w.-]*):(\w+)=(.*)$", line)
        if m:
            out[m.group(1)] = m.group(3)
    return out


_TRUE = {"on", "1", "true", "yes", "y"}
_FALSE = {"off", "0", "false", "no", "n", ""}


def same_value(want: str, have: str | None) -> bool:
    if have is None:
        return False
    w, h = want.strip().lower(), have.strip().lower()
    if w in _TRUE or w in _FALSE:
        return (w in _TRUE) == (h in _TRUE) and (h in _TRUE or h in _FALSE)
    if ("\\" in want or "/" in want) and ("\\" in have or "/" in have):
        return os.path.normcase(os.path.normpath(want)) == os.path.normcase(os.path.normpath(have))
    return w == h


DEFS_FILE = ".mse-llm-cmake-defs.json"   # flags the last `build run` configure passed (in the build dir)


def previous_defs(build_dir: Path) -> dict[str, str]:
    try:
        return json.loads((build_dir / DEFS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def stale_defs(t: Target) -> list[str]:
    """Flags an earlier configure passed that the profile no longer sets (removed with -U<KEY>)."""
    cur = t.defs()
    return [k for k in previous_defs(t.build_dir) if k not in cur]


def cache_mismatches(t: Target, cache: dict[str, str]) -> list[str]:
    out = [f"{k}: {cache.get(k, 'unset')} -> {v}" for k, v in t.defs().items() if not same_value(v, cache.get(k))]
    out += [f"{k}: {cache.get(k, 'unset')} -> default (no longer set)" for k in stale_defs(t) if k in cache]
    gen = (t.data.get("toolchain") or {}).get("generator")
    if gen and cache.get("CMAKE_GENERATOR") not in (None, gen):
        out.append(f"generator: {cache.get('CMAKE_GENERATOR')} -> {gen}")
    return out


def vcpkg_installed(root: Path, triplet: str) -> set[tuple[str, str]]:
    """(package, feature) pairs installed for the triplet; feature "" = the core package."""
    p = root / "installed" / "vcpkg" / "status"
    out: set[tuple[str, str]] = set()
    if not p.is_file():
        return out
    for stanza in p.read_text(encoding="utf-8", errors="replace").split("\n\n"):
        f = dict(re.findall(r"^([\w-]+):\s*(.*)$", stanza, re.M))
        if f.get("Architecture") == triplet and f.get("Status", "").endswith(" installed") and f.get("Package"):
            out.add((f["Package"], f.get("Feature", "")))
    return out


def vcpkg_missing(root: Path, triplet: str, packages: list[str]) -> list[str]:
    have = vcpkg_installed(root, triplet)
    out = []
    for spec in packages:
        m = re.fullmatch(r"([\w-]+)(?:\[([\w,-]+)\])?", spec)
        if not m:
            out.append(spec)
            continue
        need = [(m.group(1), "")] + [(m.group(1), f) for f in (m.group(2) or "").split(",") if f]
        if any(n not in have for n in need):
            out.append(spec)
    return out


def cuda_arch_covers(archs: str, cc: str | None) -> bool | None:
    if not cc or not archs:
        return None
    if archs.strip().lower() in ("native", "all", "all-major"):
        return True
    want = int(cc.replace(".", ""))
    for a in re.split(r"[;,\s]+", archs):
        m = re.fullmatch(r"(\d+)(?:-(real|virtual))?", a.strip())
        if not m:
            continue
        n, kind = int(m.group(1)), m.group(2)
        if n == want or (kind != "real" and n < want):   # PTX of an older arch is JIT-compiled
            return True
    return False


# ---------------------------------------------------------------- plan

@dataclass
class Step:
    n: int
    id: str
    title: str
    status: str                               # RUN | SKIP | BLOCK
    commands: list[list[str]] = field(default_factory=list)
    cwd: str | None = None
    action: str | None = None
    why: str | None = None
    what: str | None = None
    cost: str | None = None


@dataclass
class RunOpts:
    pull: bool = False
    clean: bool = False
    allow_dirty: bool = False
    quiet: bool = False
    step: bool = False
    only: str | None = None
    from_: str | None = None


def rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root.parent))
    except ValueError:
        return str(p)


def plan_steps(t: Target, opts: RunOpts | None = None) -> list[Step]:
    opts = opts or RunOpts()
    d = t.data
    s_cfg, v_cfg, tc = d.get("source") or {}, d.get("vcpkg") or {}, d.get("toolchain") or {}
    steps: list[Step] = []
    root = t.builds_root

    def add(sid: str, status: str, **kw) -> Step:
        st = Step(len(steps) + 1, sid, dict(STEPS)[sid], status, **kw)
        steps.append(st)
        return st

    have_src = t.src.is_dir()
    # 1 clone
    if have_src:
        add("clone", "SKIP", why=f"{rel(t.src, root)} exists")
    elif s_cfg.get("on_missing", "clone") == "error":
        add("clone", "BLOCK", why=f"{t.src} does not exist and [source].on_missing = \"error\"")
    else:
        argv = ["git", "clone"]
        if s_cfg.get("clone_filter"):
            argv.append(f"--filter={s_cfg['clone_filter']}")
        argv += [s_cfg.get("repo", "https://github.com/ggml-org/llama.cpp"), str(t.src)]
        add("clone", "RUN", commands=[argv], why=f"{rel(t.src, root)} does not exist (on_missing = \"clone\")",
            cost="network download, ~0.5 GB disk")

    # 2 checkout
    cmds: list[list[str]] = []
    g = ["git", "-C", str(t.src)]
    if opts.pull and have_src:
        cmds.append([*g, "fetch", "--tags", "origin"])
    cmds.append([*g, "checkout", t.ref])
    is_branch = (git(t.src, "show-ref", "--verify", "--quiet", f"refs/heads/{t.ref}") is not None) if have_src \
        else t.ref in ("master", "main")
    if opts.pull and have_src and is_branch:
        cmds.append([*g, "merge", "--ff-only", f"origin/{t.ref}"])
    if have_src and is_dirty(t.src) and not opts.allow_dirty:
        add("checkout", "BLOCK", commands=cmds,
            why=f"uncommitted changes in {rel(t.src, root)}: commit/stash them or pass --allow-dirty")
    elif have_src and not opts.pull:
        head = git(t.src, "rev-parse", "HEAD")
        want = git(t.src, "rev-parse", f"{t.ref}^{{commit}}")
        if want is None:
            add("checkout", "RUN", commands=cmds, why=f"{t.ref} not found locally (try --pull)")
        elif head == want:
            add("checkout", "SKIP", why=f"already at {t.ref} = {head[:9]} ({describe(t.src)})")
        else:
            add("checkout", "RUN", commands=cmds,
                why=f"HEAD {head[:9] if head else '?'} ({describe(t.src)}) -> "
                    f"{t.ref} {want[:9]} ({describe(t.src, want)})")
    else:
        add("checkout", "RUN", commands=cmds,
            why="builds whatever the ref points to at clone time (shown after step 1)" if not have_src
            else f"fetch, then check out {t.ref}" + (" and fast-forward" if is_branch else ""))

    # 3-5 vcpkg
    if not v_cfg:
        for sid in ("vcpkg-clone", "vcpkg-bootstrap", "vcpkg-install"):
            add(sid, "SKIP", why="no [vcpkg] in profile")
    else:
        vexe = exe(t.vcpkg_root / "vcpkg")
        triplet = v_cfg.get("triplet", "x64-windows")
        packages = list(v_cfg.get("packages") or [])
        if t.vcpkg_root.is_dir():
            add("vcpkg-clone", "SKIP", why=f"{rel(t.vcpkg_root, root)} exists")
        elif v_cfg.get("on_missing", "clone") == "error":
            add("vcpkg-clone", "BLOCK", why=f"{t.vcpkg_root} does not exist and [vcpkg].on_missing = \"error\"")
        else:
            c = [["git", "clone", v_cfg.get("repo", "https://github.com/microsoft/vcpkg"), str(t.vcpkg_root)]]
            if v_cfg.get("ref") and v_cfg["ref"] not in ("master", "main"):
                c.append(["git", "-C", str(t.vcpkg_root), "checkout", v_cfg["ref"]])
            add("vcpkg-clone", "RUN", commands=c, why=f"{rel(t.vcpkg_root, root)} does not exist",
                cost="network download, ~0.3 GB disk")
        if vexe.is_file():
            add("vcpkg-bootstrap", "SKIP", why=f"{vexe.name} exists")
        else:
            add("vcpkg-bootstrap", "RUN", commands=[list(v_cfg.get("bootstrap") or [])], cwd=str(t.vcpkg_root),
                what="downloads the vcpkg.exe tool from GitHub (Microsoft)")
        missing = vcpkg_missing(t.vcpkg_root, triplet, packages)
        if not missing:
            add("vcpkg-install", "SKIP", why=f"{', '.join(packages)} already installed ({triplet})")
        else:
            add("vcpkg-install", "RUN", commands=[[str(vexe), "install", *missing, "--triplet", triplet]],
                cwd=str(t.vcpkg_root), what=f"compiles {', '.join(missing)} from source (for -hf downloads / https)",
                cost="~10-20 min, ~3 GB disk (once)")

    # 6 configure
    argv = ["cmake", "-S", str(t.src), "-B", str(t.build_dir)]
    if tc.get("generator"):
        argv += ["-G", tc["generator"]]
    if opts.clean:
        argv.append("--fresh")
    argv += [f"-D{k}={v}" for k, v in t.defs().items()]
    argv += [f"-U{k}" for k in stale_defs(t)]
    cache = read_cmake_cache(t.build_dir)
    if cache and not opts.clean:
        mism = cache_mismatches(t, cache)
        if not mism:
            add("configure", "SKIP", commands=[argv], why="CMakeCache.txt already has the requested flags")
        else:
            add("configure", "RUN", commands=[argv], why="CMakeCache.txt differs: " + "; ".join(mism))
    else:
        add("configure", "RUN", commands=[argv],
            why="--clean: --fresh drops CMakeCache.txt" if opts.clean and cache else
            f"{rel(t.build_dir, root)} is new",
            what="checks the compilers, picks CPU/GPU features, writes the build files")

    # 7 compile
    argv = ["cmake", "--build", str(t.build_dir), "--config", tc.get("config", "Release")]
    if tc.get("jobs"):
        argv += ["-j", str(tc["jobs"])]
    if opts.clean:
        argv.append("--clean-first")
    fa = same_value("ON", t.defs().get("GGML_CUDA_FA_ALL_QUANTS"))
    add("compile", "RUN", commands=[argv],
        cost=("long (FA_ALL_QUANTS compiles many CUDA kernels), several GB disk" if fa else "several GB disk")
        if not t.bin_dir.is_dir() or opts.clean else "incremental: only changed files are recompiled")

    # 8-12
    smoke = [list(c) for c in ((d.get("smoke_test") or {}).get("commands") or [])]
    add("smoke", "RUN", commands=smoke, what="expects the binaries to start and the target GPU to be CUDA0")
    add("verify", "RUN", action="read CMakeCache.txt, ggml-cpu.vcxproj and the CPU features of the build")
    add("manifest", "RUN", action=f"{t.manifest_path} + build-plan.txt")
    if (d.get("outputs") or {}).get("help_snapshot", True):
        add("help-snapshot", "RUN", action="llama-server/llama-bench --help -> tests/fixtures/help/<commit>/, "
                                           "diff vs the previous snapshot, audit script + INI")
    else:
        add("help-snapshot", "SKIP", why="[outputs].help_snapshot = false")
    if t.engine_name:
        add("engine", "RUN", action=f"engine \"{t.engine_name}\".bin_dir -> {t.bin_dir}")
    else:
        add("engine", "SKIP", why="no [engine] in profile")

    _select(steps, opts)
    return steps


def _step_index(s: str) -> int:
    if s.isdigit() and 1 <= int(s) <= len(STEP_IDS):
        return int(s) - 1
    if s in STEP_IDS:
        return STEP_IDS.index(s)
    raise BuildError(f"unknown step {s!r}; use a number 1-{len(STEP_IDS)} or one of: {', '.join(STEP_IDS)}")


def _select(steps: list[Step], opts: RunOpts) -> None:
    if not (opts.only or opts.from_):
        return
    if opts.only:
        keep = {_step_index(opts.only)}
    else:
        keep = set(range(_step_index(opts.from_), len(steps)))
    for i, st in enumerate(steps):
        if i not in keep and st.status == "RUN":
            st.status, st.why = "SKIP", "not selected (--only/--from)"
        elif i in keep and opts.only and st.status == "SKIP" and st.commands:
            st.status, st.why = "RUN", "selected with --only"


# ---------------------------------------------------------------- checks shown with the plan

def path_estimate(build_dir: Path) -> int:
    return len(str(build_dir)) + 1 + DEEPEST_REL


def long_paths_enabled() -> bool:
    if not IS_WINDOWS:
        return True
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return int(winreg.QueryValueEx(k, "LongPathsEnabled")[0]) == 1
    except OSError:
        return False


def plan_checks(t: Target, info, toolchain: bool = True) -> list[Check]:
    from mse_llm.doctor import toolchain_checks

    out: list[Check] = []
    gpu = info.primary_gpu()
    if t.profile.gpu_match:
        names = [g.name or "" for g in info.gpus]
        ok = any(t.profile.gpu_match.lower() in n.lower() for n in names)
        out.append(Check("hardware", "GPU", "pass" if ok else "warn",
                         f"{gpu.name if gpu else 'no GPU'}"
                         + ("" if ok else f" (profile is for {t.profile.gpu_match})")))
    if t.profile.cpu_match:
        ok = t.profile.cpu_match.lower() in (info.cpu.model or "").lower()
        out.append(Check("hardware", "CPU", "pass" if ok else "warn",
                         f"{info.cpu.model}" + ("" if ok else f" (profile is for {t.profile.cpu_match})")))
    archs = t.defs().get("CMAKE_CUDA_ARCHITECTURES", "")
    if archs and gpu and gpu.vendor == "nvidia":
        cov = cuda_arch_covers(archs, gpu.compute_capability)
        if cov is not None:
            out.append(Check("hardware", "CUDA arch", "pass" if cov else "fail",
                             f"CMAKE_CUDA_ARCHITECTURES={archs} vs cc {gpu.compute_capability}",
                             None if cov else "the binaries would not run on this GPU"))
    if toolchain:
        out += toolchain_checks()
    anchor = next((p for p in (t.builds_root, *t.builds_root.parents) if p.exists()), None)
    if anchor:
        free = shutil.disk_usage(anchor).free
        st = "pass" if free > 30 * 2**30 else "warn"
        out.append(Check("disk", f"free on {anchor.anchor}", st, f"{free / 2**30:.0f} GiB",
                         None if st == "pass" else "a first build needs ~10-15 GB (llama.cpp + vcpkg)"))
    if not long_paths_enabled():
        est = path_estimate(t.build_dir)
        st = "warn" if est > PATH_WARN else "pass"
        out.append(Check("environment", "long paths", st,
                         f"Windows long paths disabled: deepest path ≈ {est} of 260 chars"
                         + (" (OK, little margin)" if st == "pass" and est > 200 else ""),
                         "set a shorter MSE_LLM_BUILDS_ROOT (e.g. D:\\b)" if st == "warn" else None))
    return out


# ---------------------------------------------------------------- printouts (plain text)

RULE = 92


def _hdr(title: str, ch: str = "═") -> str:
    s = f"{ch * 2} {title} "
    return s + ch * max(3, RULE - len(s))


def _cmd(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv) if IS_WINDOWS else " ".join(argv)


def _wrap_cmd(argv: list[str], indent: str, width: int = RULE + 8) -> list[str]:
    """Join argv into lines no longer than width (continuation lines are indented further)."""
    lines, cur = [], indent
    for i, tok in enumerate(argv):
        piece = subprocess.list2cmdline([tok]) if IS_WINDOWS else tok
        if i and len(cur) + 1 + len(piece) > width and cur.strip():
            lines.append(cur)
            cur = indent + "  " + piece
        else:
            cur = cur + ("" if cur == indent else " ") + piece
    lines.append(cur)
    return lines


def load_flag_docs() -> dict[str, dict[str, Any]]:
    p = builds_config_dir() / "_cmake-flags.toml"
    if not p.is_file():
        return {}
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def auto_flags(t: Target, info) -> list[Flag]:
    """Flags CMake decides itself that matter for the printout (GGML_AVX512 under GGML_NATIVE)."""
    defs = t.defs()
    isa = info.cpu.isa or {}
    if same_value("ON", defs.get("GGML_NATIVE", "ON")) and "GGML_AVX512" not in defs and isa.get("avx512f"):
        return [Flag("GGML_AVX512", "(ON)", "auto")]
    return []


def toolchain_versions(vcpkg_root: Path | None = None) -> dict[str, str | None]:
    from mse_llm.doctor import _nvcc, _ver

    def v(argv: list[str]) -> str | None:
        r = run(argv, timeout=20)
        x = _ver(r.stdout + r.stderr) if r.ok else None
        return ".".join(map(str, x)) if x else None

    out: dict[str, str | None] = {"cmake": v(["cmake", "--version"]), "nvcc": None, "git": v(["git", "--version"])}
    r = run(_nvcc(), timeout=20)
    m = re.search(r"release [\d.]+, V([\d.]+)", r.stdout)
    out["nvcc"] = m.group(1) if m else None
    out["msvc"] = None
    if IS_WINDOWS:
        vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
        if vswhere.is_file():
            r = run([str(vswhere), "-latest", "-products", "*", "-requires",
                     "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-format", "json"])
            try:
                vs = json.loads(r.stdout)[0]
                out["msvc"] = f"{vs['displayName']} {vs['catalog']['productDisplayVersion']}"
            except (ValueError, IndexError, KeyError):
                pass
    if vcpkg_root and vcpkg_root.is_dir():
        out["vcpkg"] = git(vcpkg_root, "log", "-1", "--format=%h (%cs)")
    return out


def render_show(t: Target, info, explain: bool = True, tools: dict[str, str | None] | None = None) -> str:
    docs = load_flag_docs()
    gpu = info.primary_gpu()
    L: list[str] = []
    dv = [v for v in t.applied if v in t.defaults]
    title = f"BUILD PLAN: {t.label}"
    if dv:
        title += f" (+ default variant{'s' if len(dv) > 1 else ''} {', '.join(dv)})"
    L.append(_hdr(title))
    if gpu:
        ok = not t.profile.gpu_match or t.profile.gpu_match.lower() in (gpu.name or "").lower()
        L.append(f"Hardware   {gpu.name} (cc {gpu.compute_capability}, {gpu.vram_total_mib} MiB)   "
                 + ("✓ matches detected GPU" if ok else f"⚠ profile is for {t.profile.gpu_match}"))
    from mse_llm.hw.cpu import isa_summary

    cok = not t.profile.cpu_match or t.profile.cpu_match.lower() in (info.cpu.model or "").lower()
    L.append(f"           {info.cpu.model} ({info.cpu.physical_cores}C/{info.cpu.logical_cpus}T, "
             f"{isa_summary(info.cpu.isa)})  "
             + ("✓ matches detected CPU" if cok else f"⚠ profile is for {t.profile.cpu_match}"))
    if t.src.is_dir():
        head = git(t.src, "rev-parse", "--short=9", "HEAD")
        L.append(f"Source     {t.src}   ref {t.ref}, checked out {head} ({describe(t.src)})")
    else:
        L.append(f"Source     {t.src}   ref {t.ref} (not cloned yet)")
    L.append(f"Build dir  {t.build_dir}   " + ("(exists)" if t.build_dir.is_dir() else "(new)"))
    if tools is None:
        tools = toolchain_versions(t.vcpkg_root)
    drv = gpu.driver if gpu else None
    L.append(f"Toolchain  CMake {tools.get('cmake') or '?'} · CUDA {tools.get('nvcc') or '?'} · "
             f"{tools.get('msvc') or 'MSVC ?'} · driver {drv or '?'}")
    if tools.get("vcpkg"):
        L.append(f"           vcpkg {tools['vcpkg']}")
    L.append("")
    L.append(_hdr("CMake flags", "─"))
    flags = t.flags + auto_flags(t, info)
    groups: dict[str, list[Flag]] = {}
    for f in flags:
        groups.setdefault((docs.get(f.key) or {}).get("group", "other"), []).append(f)
    L.append(f" {'FLAG':<27} {'VALUE':<12} {'FROM':<8} {'WHAT IT MEANS' if explain else ''}".rstrip())
    for gname in ("build", "gpu", "cpu", "debug", "other"):
        if gname not in groups:
            continue
        L.append(f" {gname}")
        for f in groups[gname]:
            val = f.value
            if f.key == "CMAKE_TOOLCHAIN_FILE":
                val = "…" + Path(val).name
            src = "variant" if f.source.startswith("variant") else f.source
            doc = docs.get(f.key)
            meaning = ""
            if explain:
                if f.source == "auto" and f.key == "GGML_AVX512":
                    meaning = "detected by FindSIMD -> /arch:AVX512 (cache still shows OFF)"
                elif doc:
                    meaning = doc.get("summary", "")
                else:
                    meaning = "? not in config/builds/_cmake-flags.toml: add an entry"
            L.append(f"  {f.key:<26} {val:<12} {src:<8} {meaning}".rstrip())
    if dv:
        L.append("")
        L.append(_hdr(f"Differences vs the comparison build (--without {' --without '.join(dv)})", "─"))
        plus = [f"+ {f.key}={f.value}" for f in t.flags if f.source.startswith("variant:")
                and f.source.split(":", 1)[1] in dv]
        L.append("  " + "  ".join(plus))
        if any(k.startswith("GGML_AVX512") for k in (p.split("=")[0][2:] for p in plus)):
            L.append("  Affects: CPU-side work only (-ncmoe / -cmoe / --moe-cache-cpu-share). "
                     "GPU-only runs: no change expected.")
    L.append("")
    L.append(_hdr("Commands", "─"))
    for st in plan_steps(t):
        if st.id in ("configure", "compile"):
            for c in st.commands:
                L.extend(_wrap_cmd(c, "  "))
    L.append("═" * RULE)
    return "\n".join(L)


def render_plan(t: Target, steps: list[Step], checks: list[Check], what: str = "build run") -> str:
    from mse_llm.checks import PLAIN_ICON as ICON

    L = [_hdr(f"WHAT `{what} {t.label}` WILL DO (nothing has been executed)")]
    chain = " (extends " + ", ".join(p.name for p, _ in load_chain(t.profile.path)[:-1]) + ")" \
        if len(load_chain(t.profile.path)) > 1 else ""
    L.append(f"Profile   {rel(t.profile.path, project_root() / 'x')}{chain}")
    env = os.environ.get("MSE_LLM_BUILDS_ROOT")
    L.append(f"Root      {t.builds_root}   " + (f"(MSE_LLM_BUILDS_ROOT={env})" if env else
                                              "(from [paths].builds_root; MSE_LLM_BUILDS_ROOT not set)"))
    if t.applied or t.without:
        L.append(f"Variants  {', '.join(t.applied) or 'none'}"
                 + (f"   without: {', '.join(t.without)}" if t.without else ""))
    L.append("")
    L.append(f" {'#':>2}  {'STEP':<20} {'STATUS':<7}  COMMAND / ACTION")
    pad = " " * 34
    for st in steps:
        first = True
        body: list[str] = []
        for c in st.commands:
            body.extend(_wrap_cmd(c, ""))
        if st.action:
            body.append(st.action)
        if not body:
            body = [""]
        for line in body:
            if first:
                L.append(f" {st.n:>2}  {st.title:<20} {st.status:<7}  {line}".rstrip())
                first = False
            else:
                L.append(pad + line)
        if st.cwd and st.status == "RUN":
            L.append(pad + f"in: {st.cwd}")
        for k in ("why", "what", "cost"):
            v = getattr(st, k)
            if v:
                L.append(pad + f"{k}: {v}")
    L.append("")
    for i, c in enumerate(checks):
        lead = "Checks    " if i == 0 else "          "
        L.append(f"{lead}{ICON[c.status]} {c.section}/{c.name}: {c.detail}" + (f"  -> {c.hint}" if c.hint else ""))
    L.append(f"Will NOT  touch anything outside {t.builds_root}")
    L.append("          change any Windows setting, or delete anything (--clean only drops CMakeCache via --fresh)")
    L.append("")
    L.append(f"Run it:   mse-llm build run {t.label}          (asks once before step 1)")
    L.append(f"          mse-llm build run {t.label} --step   (asks before every step)")
    L.append("═" * RULE)
    return "\n".join(L)


def plan_json(t: Target, steps: list[Step], checks: list[Check]) -> dict[str, Any]:
    return {
        "target": t.label,
        "profile": t.name,
        "profile_path": str(t.profile.path),
        "variants": t.applied,
        "without": t.without,
        "ref": t.ref,
        "src": str(t.src),
        "build_dir": str(t.build_dir),
        "bin_dir": str(t.bin_dir),
        "vcpkg_root": str(t.vcpkg_root),
        "engine": t.engine_name,
        "cmake": [asdict(f) for f in t.flags],
        "steps": [asdict(s) for s in steps],
        "checks": [c.as_dict() for c in checks],
    }


# ---------------------------------------------------------------- execution

def run_streamed(argv: list[str], log, echo: Callable[[str], None] | None, cwd: str | None = None
                 ) -> tuple[int | None, list[str]]:
    """Run argv (no shell), tee stdout+stderr to the log and echo. Returns (exit code, last 30 lines)."""
    tail: deque[str] = deque(maxlen=30)
    log.write(f"\n$ {_cmd(argv)}\n")
    log.flush()
    try:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             cwd=cwd, text=True, encoding="utf-8", errors="replace", bufsize=1)
    except OSError as e:
        msg = f"cannot start {argv[0]}: {e}"
        log.write(msg + "\n")
        return None, [msg]
    try:
        assert p.stdout is not None
        for line in p.stdout:
            line = line.rstrip("\r\n")
            log.write(line + "\n")
            tail.append(line)
            if echo:
                echo(line)
        return p.wait(), list(tail)
    except KeyboardInterrupt:
        _kill_tree(p.pid)
        raise


def _kill_tree(pid: int) -> None:
    import psutil

    try:
        parent = psutil.Process(pid)
        procs = [*parent.children(recursive=True), parent]
    except psutil.Error:
        return
    for q in procs:
        try:
            q.kill()
        except psutil.Error:
            pass


@dataclass
class RunState:
    started: float = field(default_factory=time.time)
    smoke: dict[str, str] = field(default_factory=dict)        # command -> output
    verify: list[dict[str, Any]] = field(default_factory=list)
    help: dict[str, Any] = field(default_factory=dict)
    engine: dict[str, Any] = field(default_factory=dict)
    durations: dict[str, float] = field(default_factory=dict)


class UI:
    """Callbacks the CLI supplies (printing and questions)."""

    def line(self, s: str) -> None: ...
    def out(self, s: str) -> None: ...          # streamed tool output
    def ask_step(self, st: Step) -> str: return "y"   # y | s | q


class _TeeUI(UI):
    """Step messages go to the screen and the log (tool output is logged by run_streamed itself)."""

    def __init__(self, inner: UI, log) -> None:
        self.inner, self.log = inner, log

    def line(self, s: str) -> None:
        self.inner.line(s)
        self.log.write(s + "\n")
        self.log.flush()

    def out(self, s: str) -> None:
        self.inner.out(s)

    def ask_step(self, st: Step) -> str:
        return self.inner.ask_step(st)


def log_path(t: Target) -> Path:
    d = t.builds_root / "logs"
    d.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w.+-]+", "_", t.label).strip("_")
    return d / f"{slug}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"


def execute(t: Target, steps: list[Step], opts: RunOpts, ui: UI, plan_text: str, info) -> RunState:
    state = RunState()
    lp = log_path(t)
    ui.line(f"log: {lp}")
    total = len(steps)
    with open(lp, "w", encoding="utf-8") as log:
        log.write(plan_text + "\n")
        for st in steps:
            tag = f"[{st.n}/{total}] {st.title}"
            if st.status != "RUN":
                msg = f"· {tag}: skipped ({st.why or st.status})"
                ui.line(msg)
                log.write(msg + "\n")
                continue
            if opts.step:
                a = ui.ask_step(st)
                if a == "q":
                    raise BuildError(f"stopped before step {st.n}; resume with --from {st.id}")
                if a == "s":
                    ui.line(f"· {tag}: skipped by you")
                    continue
            ui.line(f"▶ {tag}")
            log.write(f"\n▶ {tag}  ({_dt.datetime.now():%Y-%m-%d %H:%M:%S})\n")
            log.flush()
            t0 = time.time()
            ui = _TeeUI(ui, log)
            try:
                _run_step(t, st, opts, ui, log, state, plan_text, info)
            except BuildError as e:
                ui.line(f"✗ {tag} failed: {e}")
                ui.line(f"  log: {lp}")
                ui.line(f"  resume: mse-llm build run {t.label} --from {st.id}")
                raise
            dt = time.time() - t0
            state.durations[st.id] = round(dt, 1)
            ui.line(f"✓ done in {_fmt_dur(dt)}")
            ui = ui.inner
    return state


def _fmt_dur(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    m, s = divmod(int(s), 60)
    if m < 90:
        return f"{m} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def _run_cmds(st: Step, opts: RunOpts, ui: UI, log) -> dict[str, str]:
    outputs = {}
    for argv in st.commands:
        ui.line(f"  $ {_cmd(argv)}")
        lines: list[str] = []

        def echo(s: str, lines: list[str] = lines) -> None:
            lines.append(s)
            if not opts.quiet:
                ui.out(s)

        code, tail = run_streamed(argv, log, echo, cwd=st.cwd)
        if code != 0:
            for x in tail:
                ui.out(x)
            raise BuildError(f"exit {code}" if code is not None else tail[-1])
        outputs[_cmd(argv)] = "\n".join(lines)
    return outputs


def _run_step(t: Target, st: Step, opts: RunOpts, ui: UI, log, state: RunState, plan_text: str, info) -> None:
    if st.id == "checkout":
        if t.src.is_dir() and is_dirty(t.src) and not opts.allow_dirty:
            raise BuildError(f"uncommitted changes in {t.src}; commit/stash or --allow-dirty")
        _run_cmds(st, opts, ui, log)
        ui.line(f"  now at {git(t.src, 'rev-parse', '--short=9', 'HEAD')} ({describe(t.src)})")
    elif st.id == "configure":
        _run_cmds(st, opts, ui, log)
        (t.build_dir / DEFS_FILE).write_text(json.dumps(t.defs(), indent=2), encoding="utf-8")
    elif st.id == "smoke":
        state.smoke = _run_cmds(st, RunOpts(quiet=True), ui, log)
        for cmd, out in state.smoke.items():
            first = next((x for x in out.splitlines() if x.strip()), "")
            ui.line(f"  {Path(cmd.split()[0].strip('\"')).name}: {first.strip()}")
        devs = "\n".join(state.smoke.values())
        m = re.search(r"CUDA0:\s*([^(\n]+)", devs)
        want = t.profile.gpu_match
        if m:
            ui.line(f"  CUDA0 = {m.group(1).strip()}")
            if want and want.lower() not in m.group(1).lower():
                raise BuildError(f"CUDA0 is {m.group(1).strip()}, expected {want} (CUDA_VISIBLE_DEVICES?)")
        elif same_value("ON", t.defs().get("GGML_CUDA")):
            raise BuildError("no CUDA0 device in llama-bench --list-devices: the CUDA backend did not load")
    elif st.id == "verify":
        state.verify = verify_build(t, info, state.smoke)
        for v in state.verify:
            ui.line(f"  {'✓' if v['ok'] else '✗'} {v['name']}: {v['detail']}")
        bad = [v for v in state.verify if not v["ok"]]
        if bad:
            raise BuildError(f"{len(bad)} flag(s) did not get into the build: {', '.join(v['name'] for v in bad)}")
    elif st.id == "manifest":
        m = write_manifest(t, info, state, plan_text, opts)
        ui.line(f"  {t.manifest_path}  ({m['describe']}, {m['commit'][:9]})")
    elif st.id == "help-snapshot":
        state.help = help_snapshot(t)
        h = state.help
        ui.line(f"  snapshot {h['snapshot']} -> {h['path']}")
        if h.get("previous"):
            for tool, d in h["diff"].items():
                n = sum(len(x) for x in d.values())
                ui.line(f"  {tool} vs {h['previous']}: " + ("no changes" if not n else
                        f"{len(d['added'])} added, {len(d['removed'])} removed, {len(d['changed'])} changed"))
                for k in ("added", "removed", "changed"):
                    for x in d[k]:
                        ui.line(f"    {k:8} {x}")
        elif h.get("replaced"):
            ui.line(f"  same llama.cpp commit as the stored snapshot {h['snapshot']} (updated it)")
        else:
            ui.line("  no previous snapshot to diff against (first build)")
        f = h.get("findings") or []
        ui.line("  audit of llama_cpp_benchmark.sh + llama_cpp_presets.ini: "
                + (f"{len(f)} finding(s), see `mse-llm doctor flags`" if f else "all flags accepted"))
    elif st.id == "engine":
        state.engine = engine_report(t)
        for msg in state.engine["messages"]:
            ui.line(f"  {msg}")
    else:
        _run_cmds(st, opts, ui, log)


# ---------------------------------------------------------------- verify / manifest / snapshot / engine

def _vcxproj_release(path: Path) -> dict[str, str]:
    """EnableEnhancedInstructionSet + PreprocessorDefinitions of the Release|x64 ClCompile settings."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"<ItemDefinitionGroup Condition=\"'\$\(Configuration\)\|\$\(Platform\)'=='Release\|x64'\">"
                  r"(.*?)</ItemDefinitionGroup>", text, re.S)
    block = m.group(1) if m else text
    out = {}
    for tag in ("EnableEnhancedInstructionSet", "PreprocessorDefinitions", "AdditionalOptions"):
        mm = re.search(rf"<ClCompile>.*?<{tag}>(.*?)</{tag}>", block, re.S)
        if mm:
            out[tag] = mm.group(1)
    return out


def verify_build(t: Target, info, smoke: dict[str, str] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cache = read_cmake_cache(t.build_dir)

    def add(name: str, ok: bool, detail: str) -> None:
        out.append({"name": name, "ok": ok, "detail": detail})

    if not cache:
        add("CMakeCache.txt", False, f"missing in {t.build_dir}")
        return out
    mism = cache_mismatches(t, cache)
    add("CMakeCache.txt", not mism, "all requested flags present" if not mism else "; ".join(mism))
    defs = t.defs()
    if "CMAKE_CUDA_ARCHITECTURES" in defs:
        add("CUDA arch", same_value(defs["CMAKE_CUDA_ARCHITECTURES"], cache.get("CMAKE_CUDA_ARCHITECTURES")),
            f"CMAKE_CUDA_ARCHITECTURES={cache.get('CMAKE_CUDA_ARCHITECTURES')}")
    if same_value("ON", defs.get("GGML_CUDA_FA_ALL_QUANTS")):
        add("FA all quants", same_value("ON", cache.get("GGML_CUDA_FA_ALL_QUANTS")),
            f"GGML_CUDA_FA_ALL_QUANTS={cache.get('GGML_CUDA_FA_ALL_QUANTS')}")
    vcx = _vcxproj_release(t.build_dir / "ggml" / "src" / "ggml-cpu.vcxproj")
    if vcx:
        arch = vcx.get("EnableEnhancedInstructionSet", "")
        defs_cpp = vcx.get("PreprocessorDefinitions", "")
        isa = info.cpu.isa or {}
        if same_value("ON", defs.get("GGML_NATIVE", "ON")) and isa.get("avx512f"):
            add("/arch:AVX512", "512" in arch or "/arch:AVX512" in vcx.get("AdditionalOptions", ""),
                f"EnableEnhancedInstructionSet={arch or '?'}")
        left_out = {k for v in t.without for k in ((t.data.get("variants") or {}).get(v, {}).get("cmake") or {})}
        for flag in ("GGML_AVX512_VNNI", "GGML_AVX512_VBMI", "GGML_AVX512_BF16"):
            want = defs.get(flag) or ("OFF" if flag in left_out else None)
            if want is None:
                continue
            # MSVC path of ggml-cpu/CMakeLists.txt: VNNI/BF16 add GGML_AVX512_<X> and __AVX512<X>__,
            # VBMI only __AVX512VBMI__. Either macro means the extension is compiled in.
            macros = (flag, "__AVX512" + flag.removeprefix("GGML_AVX512_") + "__")
            found = [m for m in macros if re.search(rf"(^|;){m}(;|$)", defs_cpp)]
            add(flag, bool(found) == same_value("ON", want),
                f"{' + '.join(found) or 'not defined'} in ggml-cpu.vcxproj (requested {want})")
    elif IS_WINDOWS:
        add("ggml-cpu.vcxproj", False, "not found; cannot check the CPU flags")
    if smoke:
        m = re.search(r"CUDA0:\s*([^(\n]+)", "\n".join(smoke.values()))
        if m:
            add("CUDA0", True, m.group(1).strip())
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(t: Target, info, state: RunState, plan_text: str, opts: RunOpts | None = None) -> dict[str, Any]:
    from mse_llm import hw

    cache = read_cmake_cache(t.build_dir)
    keep = re.compile(r"^(GGML_|LLAMA_|CMAKE_CUDA_|CMAKE_TOOLCHAIN_FILE$|CMAKE_GENERATOR$|CMAKE_BUILD_TYPE$|"
                      r"VCPKG_TARGET_TRIPLET$|AVX512_FOUND$|BUILD_SHARED_LIBS$)")
    bins = {}
    for name in (*MAIN_BINARIES, *MAIN_LIBS):
        for p in (exe(t.bin_dir / name), t.bin_dir / f"{name}.dll"):
            if p.is_file():
                bins[p.name] = sha256(p)
                break
    gpu = info.primary_gpu()
    m = {
        "schema": 1,
        "target": t.label,
        "profile": t.name,
        "variants": t.applied,
        "without": t.without,
        "ref": t.ref,
        "commit": git(t.src, "rev-parse", "HEAD") or "",
        "describe": describe(t.src) or "",
        "src": str(t.src),
        "build_dir": str(t.build_dir),
        "bin_dir": str(t.bin_dir),
        "engine": t.engine_name,
        "cmake_requested": t.defs(),
        "cmake_cache": {k: v for k, v in sorted(cache.items()) if keep.match(k)},
        "toolchain": toolchain_versions(t.vcpkg_root),
        "driver": gpu.driver if gpu else None,
        "gpu": gpu.name if gpu else None,
        "compute_capability": gpu.compute_capability if gpu else None,
        "cpu": info.cpu.model,
        "cpu_isa": info.cpu.isa,
        "hw_identity": hw.identity(info).get("hash"),
        "verified": state.verify,
        "binaries_sha256": bins,
        "built_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - state.started, 1),
        "step_durations_s": state.durations,
        # Set only when the run was limited with --from/--only: duration_s then covers just those steps.
        "resumed_from": opts.from_ if opts else None,
        "only_step": opts.only if opts else None,
        "mse_llm": __version__,
    }
    t.bin_dir.mkdir(parents=True, exist_ok=True)
    t.manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
    (t.bin_dir / "build-plan.txt").write_text(plan_text + "\n", encoding="utf-8")
    return m


def read_manifest(bin_dir: Path) -> dict[str, Any] | None:
    p = bin_dir / "build-manifest.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def help_snapshot(t: Target) -> dict[str, Any]:
    from mse_llm import helpsnap as hs

    before = hs.list_snapshots()
    sid, dest = hs.capture(t.bin_dir)
    prev = next((p for p in reversed(before) if p.name != sid), None)
    rep: dict[str, Any] = {"snapshot": sid, "path": str(dest), "previous": prev.name if prev else None, "diff": {},
                           "replaced": any(p.name == sid for p in before)}
    cur = hs.load_snapshot(dest)
    if prev:
        old = hs.load_snapshot(prev)
        for tool in hs.TOOLS:
            if tool in cur and tool in old:
                d = hs.diff_help(old[tool], cur[tool])
                rep["diff"][tool] = {"added": d.added, "removed": d.removed, "changed": d.changed}
    root = project_root()
    findings = []
    if (root / "llama_cpp_benchmark.sh").is_file():
        findings += hs.audit_script(root / "llama_cpp_benchmark.sh", cur)
    if (root / "llama_cpp_presets.ini").is_file() and "llama-server" in cur:
        findings += hs.audit_ini(root / "llama_cpp_presets.ini", cur["llama-server"])
    rep["findings"] = [f.__dict__ for f in findings]
    return rep


def engine_report(t: Target) -> dict[str, Any]:
    """Engines follow the build profile automatically (config.py reads bin_dir from the profile and from
    build manifests). This reports what now points where, and anything that overrides it."""
    from mse_llm.config import load_config

    msgs = []
    name = t.engine_name
    try:
        cfg = load_config()
        eng = cfg.engines.get(name or "")
    except Exception as e:  # noqa: BLE001 - report, never fail the build here
        return {"engine": name, "messages": [f"config could not be loaded: {e}"]}
    if eng is None:
        msgs.append(f"engine {name!r} not found in the config")
    elif os.path.normcase(str(eng.bin_path())) != os.path.normcase(str(t.bin_dir)):
        msgs.append(f"⚠ engine {name!r} points at {eng.bin_path()} (overridden in {cfg.source}); "
                    f"this build is in {t.bin_dir}")
    else:
        msgs.append(f"engine {name!r} -> {t.bin_dir}")
    script = project_root() / "llama_cpp_benchmark.sh"
    if script.is_file() and t.build_dir.name == t.data.get("build_dir", "build"):
        text = script.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^MSE_LLAMA_BIN_DIR=['\"]?([^'\"\n]+)", text, re.M)
        if m:
            same = os.path.normcase(os.path.normpath(m.group(1))) == os.path.normcase(str(t.bin_dir))
            msgs.append("bash script MSE_LLAMA_BIN_DIR points here too" if same else
                         f"⚠ bash script MSE_LLAMA_BIN_DIR = {m.group(1)} (not this build)")
    return {"engine": name, "messages": msgs}


def built_dirs(src: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Build dirs under a checkout that have a manifest (build/, build-*/)."""
    out = []
    if not src.is_dir():
        return out
    for d in sorted(src.glob("build*")):
        for m in d.glob("bin/*/build-manifest.json"):
            try:
                out.append((m.parent, json.loads(m.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                pass
    return out


# ---------------------------------------------------------------- versions

@dataclass
class Versions:
    src: Path
    head: str | None
    head_describe: str | None
    branch: str | None
    releases: list[tuple[str, str, int | None]]   # (tag, date, commits HEAD is ahead of it)
    builds: list[tuple[str, str, int | None]]
    built: list[tuple[str, str, str]]            # (target, describe, bin_dir)


def versions(src: Path, limit: int = 8, remote: bool = False) -> Versions:
    if not src.is_dir():
        raise BuildError(f"{src} does not exist yet; run `mse-llm build clone` first")
    if remote:
        r = run(["git", "-C", str(src), "fetch", "--tags", "origin"], timeout=300)
        if not r.ok:
            raise BuildError(f"git fetch --tags failed: {r.error or r.stderr.strip()}")
    refs = git(src, "for-each-ref", "--sort=-creatordate", "--format=%(refname:short)|%(creatordate:short)",
               "refs/tags") or ""
    rel_tags, b_tags = [], []
    for line in refs.splitlines():
        tag, _, date = line.partition("|")
        if re.fullmatch(r"v\d+\.\d+\.\d+", tag):
            rel_tags.append((tag, date))
        elif re.fullmatch(r"b\d+", tag):
            b_tags.append((tag, date))

    def ahead(tag: str) -> int | None:
        n = git(src, "rev-list", "--count", f"{tag}..HEAD")
        return int(n) if n and n.isdigit() else None

    rel_out = [(tg, dt, ahead(tg)) for tg, dt in rel_tags[:limit]]
    b_out = [(tg, dt, ahead(tg)) for tg, dt in b_tags[:limit]]
    built = [(m.get("target", "?"), m.get("describe", "?"), str(d)) for d, m in built_dirs(src)]
    return Versions(src, git(src, "rev-parse", "--short=9", "HEAD"), describe(src),
                    git(src, "rev-parse", "--abbrev-ref", "HEAD"), rel_out, b_out, built)


# ---------------------------------------------------------------- clean

@dataclass
class CleanItem:
    path: Path
    what: str
    size: int


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def all_builds_roots() -> list[Path]:
    roots = {p.builds_root.resolve() for p in list_profiles()}
    return sorted(roots)


def clean_items(kind: str, roots: list[Path] | None = None, dirs: tuple[str, ...] = ()) -> list[CleanItem]:
    """What `build clean --logs|--builds|--all|--dir NAME` would delete. Only paths inside a builds root.

    --dir takes a build dir name (build-a49db739f) or its path; only real build dirs (CMakeCache.txt or
    bin/ inside a checkout) qualify, never a checkout or vcpkg itself."""
    if kind == "dir":
        builds = clean_items("builds", roots)
        out, unknown = [], []
        for d in dirs:
            want = os.path.normcase(os.path.abspath(d))
            hit = [it for it in builds if it.path.name.lower() == d.lower().rstrip("\\/")
                   or os.path.normcase(str(it.path)) == want]
            if not hit:
                unknown.append(d)
            out += [it for it in hit if it not in out]
        if unknown:
            have = ", ".join(it.path.name for it in builds) or "none"
            raise BuildError(f"not a build dir: {', '.join(unknown)} (build dirs: {have})")
        return out
    out: list[CleanItem] = []
    for root in roots or all_builds_roots():
        if not root.is_dir():
            continue
        if kind == "all":
            out.append(CleanItem(root, "everything: llama.cpp + vcpkg checkouts, all builds, logs", dir_size(root)))
            continue
        if kind == "logs":
            logs = root / "logs"
            if logs.is_dir():
                out.append(CleanItem(logs, "build logs", dir_size(logs)))
            continue
        # builds: build dirs (build/, build-*/) of every checkout under the root
        for src in sorted(p for p in root.iterdir() if p.is_dir() and (p / ".git").exists()):
            for d in sorted(src.glob("build*")):
                if d.is_dir() and ((d / "CMakeCache.txt").is_file() or (d / "bin").is_dir()):
                    m = next((read_manifest(x) for x in d.glob("bin/*") if read_manifest(x)), None)
                    what = f"build of {m.get('target')} ({m.get('describe')})" if m else "build dir (no manifest)"
                    out.append(CleanItem(d, what, dir_size(d)))
    return out


def busy_processes(roots: list[Path]) -> list[str]:
    """Processes running from, or working on, a builds root (a build, a server, a bench), summarized:
    the top-level ones by pid + command, their children counted by name."""
    from collections import Counter

    import psutil

    keys = [os.path.normcase(str(r)) for r in roots]
    me = os.getpid()
    hits: dict[int, tuple[str, str, int | None]] = {}
    for p in psutil.process_iter(["pid", "ppid", "name", "exe", "cmdline"]):
        if p.info["pid"] == me:
            continue
        cmd = p.info.get("cmdline") or []
        text = os.path.normcase(" ".join([p.info.get("exe") or "", *cmd]))
        if any(k in text for k in keys):
            hits[p.info["pid"]] = (p.info["name"] or "?", " ".join(cmd[:4])[:100], p.info.get("ppid"))
    tops = [pid for pid, (_n, _c, ppid) in hits.items() if ppid not in hits]
    out = []
    for pid in sorted(tops):
        name, cmd, _ = hits[pid]
        kids = Counter(hits[q][0] for q in hits if q != pid and _descends(q, pid, hits))
        top4 = ", ".join(f"{n} x{c}" for n, c in kids.most_common(4))
        extra = f" + {sum(kids.values())} child processes ({top4})" if kids else ""
        out.append(f"{name} {pid}: {cmd}{extra}")
    return out


def _descends(pid: int, ancestor: int, hits: dict[int, tuple[str, str, int | None]]) -> bool:
    seen = set()
    while pid in hits and pid not in seen:
        seen.add(pid)
        pid = hits[pid][2]
        if pid == ancestor:
            return True
    return False


def remove_tree(path: Path, roots: list[Path]) -> None:
    """Delete a folder inside a builds root (read-only git objects included)."""
    rp = path.resolve()
    if not any(rp == r or r in rp.parents for r in (x.resolve() for x in roots)):
        raise BuildError(f"refusing to delete {rp}: not inside a builds root")

    def onexc(func, p, _exc):
        os.chmod(p, 0o700)
        func(p)

    shutil.rmtree(rp, onexc=onexc)


def fmt_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"
