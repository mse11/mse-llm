"""`serve` and `presets serve` at run time (M5): fit pre-check, what --fit would change, startup-log
check, foreground / background servers, router model loads, state files under results/servers/.

`serve` runs llama-server with `--fit on` (§0.3). Before it starts, the `fit` check (§4.6) runs for the
requested settings and warns about NO-FIT / TIGHT / UNSURE. `llama-fit-params` without `--fit off`
prints the command line llama.cpp would actually use, which is diffed against the request.

Checked on a49db739f / ffc87c3a7: --fit only adjusts arguments the user did NOT set. `serve` sets
`-ngl all` and `-c`, so for a config that doesn't fit llama.cpp aborts the fit ("n_gpu_layers already
set by user") and the server starts with exactly the requested (too large) settings. The preview says
so, and shows what --fit would pick if -ngl were left unset (`serve --let-fit`).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import re
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from mse_llm import fit as ff
from mse_llm.config import Config
from mse_llm.util import IS_WINDOWS, project_root
from mse_llm.variants import Variant

BAD_VERDICTS = ("NO-FIT", "TIGHT", "UNSURE", "ERROR", "?")


def servers_dir() -> Path:
    return project_root() / "results" / "servers"


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------- what --fit would change

@dataclass
class FitPreview:
    argv: list[str]
    ok: bool                                  # llama-fit-params found a fitting set of arguments
    fitted: list[str] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)
    locked: bool = False                      # aborted: -ngl / -ncmoe (tensor overrides) already set by the user
    message: str = ""
    unlocked: FitPreview | None = None        # the same with -ngl and -ncmoe left unset (what `--let-fit` gets)

    def as_dict(self) -> dict[str, Any]:
        return {"argv": self.argv, "ok": self.ok, "fitted": self.fitted, "changes": self.changes,
                "locked": self.locked, "message": self.message,
                "unlocked": self.unlocked.as_dict() if self.unlocked else None}


_OT_LAYER = re.compile(r"blk\\?\.(\d+)\\?\.ffn_([^=]*)=")


def _ot_layers(ot: str) -> tuple[set[int], set[int]]:
    """-ot pattern list -> (layers with all experts on CPU, layers with part of them on CPU)."""
    full, part = set(), set()
    for item in ot.split(","):
        m = _OT_LAYER.search(item)
        if not m:
            continue
        n, what = int(m.group(1)), m.group(2)
        (full if "exps" in what and "up" in what and "down" in what else part).add(n)
    return full, part - full


def _ranges(ns: set[int]) -> str:
    if not ns:
        return "none"
    s = sorted(ns)
    out, a, b = [], s[0], s[0]
    for x in s[1:]:
        if x == b + 1:
            b = x
            continue
        out.append(f"{a}–{b}" if a != b else str(a))
        a = b = x
    out.append(f"{a}–{b}" if a != b else str(a))
    return ", ".join(out)


def _arg(args: list[str], *names: str) -> str | None:
    for i, a in enumerate(args):
        if a in names and i + 1 < len(args):
            return args[i + 1]
    return None


def _ngl_text(v: str | None) -> str:
    return {"-2": "all", "-1": "auto", None: "auto"}.get(v, str(v))


def diff_fitted(requested: list[str], fitted: list[str], ncmoe: int | None) -> list[str]:
    out: list[str] = []
    rc, fc = _arg(requested, "-c", "--ctx-size"), _arg(fitted, "-c", "--ctx-size")
    if fc and rc != fc:
        out.append(f"-c {rc or 'model default'} → {fc}")
    rn, fn = _arg(requested, "-ngl", "--n-gpu-layers"), _arg(fitted, "-ngl", "--n-gpu-layers")
    if fn is not None and _ngl_text(rn) != _ngl_text(fn) and not (rn in ("all", "-2") and fn == "-2"):
        out.append(f"-ngl {_ngl_text(rn) if rn else 'auto'} → {_ngl_text(fn)}"
                   + (" (the last layers stay on the CPU)" if fn not in ("-1", "-2") else ""))
    ot = _arg(fitted, "-ot", "--override-tensor")
    want = set(range(ncmoe)) if ncmoe else set()
    if ot is not None or want:
        full, part = _ot_layers(ot or "")
        if full != want or part:
            s = f"experts of layers {_ranges(full)} on the CPU"
            if part:
                s += f", part of layer {_ranges(part)}"
            s += f" (requested: {_ranges(want) if want else 'none'}"
            s += f", -ncmoe {ncmoe})" if ncmoe else ")"
            out.append(s)
    return out


PLACEMENT_FLAGS = {"-ngl": 1, "--n-gpu-layers": 1, "--gpu-layers": 1, "-ncmoe": 1, "--n-cpu-moe": 1,
                   "-cmoe": 0, "--cpu-moe": 0}


def without_placement(argv: list[str]) -> list[str]:
    """argv minus -ngl / -ncmoe / -cmoe: what llama.cpp's --fit needs to be free to decide (`serve --let-fit`)."""
    out, i = [], 0
    while i < len(argv):
        n = PLACEMENT_FLAGS.get(argv[i])
        if n is None:
            out.append(argv[i])
            i += 1
        else:
            i += 1 + n
    return out


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
    return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=120, creationflags=flags)


def fit_preview(cfg: Config, engine: str | None, model_path: str, v: Variant, ctx: int, *,
                runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
                unlock: bool = False, _second: bool = True) -> FitPreview:
    """Run llama-fit-params with --fit on (the default) and diff its fitted command line with the request."""
    runner = runner or _run
    argv = ff.fit_params_argv(cfg, engine, model_path, v, ctx)
    i = argv.index("--fit")
    argv = argv[:i] + argv[i + 4:]                      # drop "--fit off --fit-print on"
    requested = argv
    if unlock:
        argv = without_placement(argv)
    try:
        cp = runner(argv)
    except (OSError, subprocess.SubprocessError) as e:
        return FitPreview(argv, False, message=f"can't run llama-fit-params: {e}")
    lines = [ln for ln in (cp.stdout or "").splitlines() if ln.strip().startswith("-")]
    if cp.returncode == 0 and lines:
        fitted = [a.strip('"') for a in shlex.split(lines[-1], posix=False)]
        p = FitPreview(argv, True, fitted, diff_fitted(requested, fitted, v.n_cpu_moe))
        p.message = "no change: the requested settings fit" if not p.changes else "--fit would change:"
        return p
    err = cp.stderr or ""
    m = re.search(r"failed to fit params[^\n]*", err)
    p = FitPreview(argv, False, message=m.group(0).strip() if m else ff._error_text(err, cp.returncode))
    if "already set by user" in err:
        p.locked = True
        what = "-ncmoe (tensor overrides)" if "tensor_buft_overrides" in err else "-ngl (serve passes -ngl all)"
        p.message = (f"--fit on can't adjust this config: {what} is set, so llama.cpp aborts the fit and the "
                     "server starts with exactly these settings")
        if _second and not unlock:
            p.unlocked = fit_preview(cfg, engine, model_path, v, ctx, runner=runner, unlock=True, _second=False)
    return p


# ---------------------------------------------------------------- fit pre-check

@dataclass
class FitCheck:
    model: str
    variant: str
    ctx: int
    verdict: str
    row: ff.FitRow | None = None
    report: ff.FitReport | None = None
    preview: FitPreview | None = None
    error: str | None = None

    @property
    def bad(self) -> bool:
        return self.verdict.rstrip("~") in BAD_VERDICTS

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "variant": self.variant, "ctx": self.ctx, "verdict": self.verdict,
                "bad": self.bad, "error": self.error, "row": self.row.as_dict() if self.row else None,
                "preview": self.preview.as_dict() if self.preview else None}


def fit_check(cfg: Config, model: str, v: Variant, ctx: int, engine: str | None = None, *,
              draft: str | None = None, mmproj: str | None = None, margin_mib: int = ff.DEFAULT_MARGIN,
              free: tuple[int | None, str] | None = None) -> FitCheck:
    """The `fit` verdict for one (model, variant, ctx); for a bad one also --suggest and the --fit preview."""
    free_mib, free_src = free or ff.free_vram_now()
    try:
        rep = ff.run_fit(cfg, model, [ff.FitInput(v)], [ctx], engine=engine, free_mib=free_mib, free_source=free_src,
                         margin_mib=margin_mib, draft=draft, mmproj=mmproj)
    except (ff.FitError, KeyError) as e:
        return FitCheck(model, v.name, ctx, "ERROR", error=str(e.args[0] if isinstance(e, KeyError) else e))
    row = rep.rows[0]
    chk = FitCheck(model, v.name, ctx, row.verdict, row, rep, error=row.error)
    if chk.bad and not row.error:
        try:
            rep = ff.run_fit(cfg, model, [ff.FitInput(v)], [ctx], engine=engine, free_mib=free_mib,
                             free_source=free_src, margin_mib=margin_mib, draft=draft, mmproj=mmproj, suggest=True)
            chk.row, chk.report = rep.rows[0], rep
        except ff.FitError:
            pass
        chk.preview = fit_preview(cfg, engine, rep.path, v, ctx)
    return chk


# ---------------------------------------------------------------- startup log

_N_CTX = re.compile(r"llama_context:\s+n_ctx\s*=\s*(\d+)")
_OFFLOAD = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")


@dataclass
class StartupInfo:
    n_ctx: int | None = None
    offloaded: tuple[int, int] | None = None
    fit_lines: list[str] = field(default_factory=list)

    def feed(self, line: str) -> None:
        m = _N_CTX.search(line)
        if m and self.n_ctx is None:
            self.n_ctx = int(m.group(1))
        m = _OFFLOAD.search(line)
        if m and self.offloaded is None:
            self.offloaded = (int(m.group(1)), int(m.group(2)))
        if ("fit_params" in line or "params_fit" in line) and len(self.fit_lines) < 20:
            self.fit_lines.append(line.strip())

    def differences(self, ctx: int | None) -> list[str]:
        out = []
        if ctx and self.n_ctx and self.n_ctx != ctx:
            out.append(f"n_ctx is {self.n_ctx}, requested -c {ctx} (--fit or the model's limit changed it)")
        if self.offloaded and self.offloaded[0] < self.offloaded[1]:
            out.append(f"only {self.offloaded[0]}/{self.offloaded[1]} layers on the GPU")
        return out


# ---------------------------------------------------------------- engines already running

def other_engines(port: int | None = None) -> list[str]:
    try:
        from mse_llm.hw.processes import detect_processes

        procs = detect_processes()
    except Exception:
        return []
    return [f"{p.name} (pid {p.pid}{f', {p.gpu_mib} MiB VRAM' if p.gpu_mib else ''})"
            for p in procs if p.kind == "engine"]


# ---------------------------------------------------------------- state files (background servers)

def state_path(kind: str, port: int) -> Path:
    return servers_dir() / f"{kind}-{port}.json"


def write_state(kind: str, port: int, data: dict[str, Any]) -> Path:
    p = state_path(kind, port)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def read_state(kind: str, port: int | None = None) -> list[dict[str, Any]]:
    d = servers_dir()
    if not d.is_dir():
        return []
    files = [state_path(kind, port)] if port else sorted(d.glob(f"{kind}-*.json"))
    out = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_file"] = str(f)
            out.append(data)
        except (OSError, ValueError):
            continue
    return out


def state_alive(st: dict[str, Any]) -> bool:
    import psutil

    try:
        p = psutil.Process(int(st["pid"]))
        return abs(p.create_time() - float(st.get("create_time", p.create_time()))) < 2
    except (psutil.Error, KeyError, ValueError, TypeError):
        return False


# ---------------------------------------------------------------- router model loads

def _status(m: dict[str, Any]) -> str:
    s = m.get("status")
    if isinstance(s, dict):
        return str(s.get("value") or "?") + (" (failed)" if s.get("failed") else "")
    return str(s or "?")


async def router_models(root: str) -> list[dict[str, Any]]:
    from mse_llm.health import get_json

    data = await get_json(root + "/models")
    return list(data.get("data") or [])


async def load_model(root: str, name: str, *, timeout: float = 600, wait: bool = True,
                     progress: Callable[[str], None] | None = None) -> str:
    """POST /models/load and wait until the model is `loaded` (or fails / times out). Returns its status."""
    from mse_llm.health import post_json

    code, body = await post_json(root + "/models/load", {"model": name})
    if code >= 400:
        msg = body.get("error", {}).get("message") if isinstance(body, dict) else str(body)
        raise RuntimeError(f"load {name}: HTTP {code} {msg or ''}".strip())
    if not wait:
        return "loading"
    t0 = asyncio.get_running_loop().time()
    last = ""
    while True:
        models = await router_models(root)
        m = next((x for x in models if x.get("id") == name), None)
        st = _status(m) if m else "?"
        if st != last and progress:
            progress(st)
        last = st
        if st == "loaded":
            return st
        if "fail" in st or st in ("error",) or (m and isinstance(m.get("status"), dict) and m["status"].get("failed")):
            raise RuntimeError(f"load {name}: {st}")
        if st == "unloaded" and asyncio.get_running_loop().time() - t0 > 5:
            raise RuntimeError(f"load {name}: the router unloaded it again (see its log)")
        if asyncio.get_running_loop().time() - t0 > timeout:
            raise RuntimeError(f"load {name}: not loaded within {timeout:.0f} s (status {st})")
        await asyncio.sleep(0.5)


async def unload_model(root: str, name: str) -> None:
    from mse_llm.health import post_json

    code, body = await post_json(root + "/models/unload", {"model": name})
    if code >= 400:
        msg = body.get("error", {}).get("message") if isinstance(body, dict) else str(body)
        raise RuntimeError(f"unload {name}: HTTP {code} {msg or ''}".strip())


def gpu_mib(mem: dict[str, Any] | None) -> int | None:
    """Router memory measurement (bytes per device) -> MiB on the GPUs."""
    if not mem:
        return None
    devs = [d for d in mem.get("devices") or [] if d.get("kind") == "device"]
    if not devs:
        return None
    return round(sum((d.get("model") or 0) + (d.get("context") or 0) + (d.get("compute") or 0) for d in devs) / 2**20)


def variant_without_defaults(v: Variant, cfg: Config) -> Variant:
    """Drop the load mode when it is just [defaults].load_mode, so names compare with default_serve."""
    return replace(v, load_mode=None) if v.load_mode and v.load_mode == cfg.defaults.load_mode else v
