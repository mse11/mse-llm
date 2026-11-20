"""`mse-llm fit` (§4.6, M4): expected VRAM / RAM per variant and context, before anything runs.

The main model is measured by llama.cpp itself: `llama-fit-params --fit off --fit-print on` prints the
per-device split (model, context, compute in MiB) in a few seconds, without loading the weights.
llama-fit-params rejects `--spec-*` and doesn't know about a vision projector, so those parts are added
here, each with its source:

    measured     llama-fit-params, or exact tensor bytes from the GGUF header
    estimated    from GGUF metadata (draft KV + compute, mmproj compute); verdict gets a `~`
    none         no VRAM (n-gram speculation runs on the CPU)
    NOT COUNTED  unknown (e.g. a draft-* variant without a drafter); verdict UNSURE
    not used     the part isn't present

Verdict: fits / TIGHT (less than `margin` MiB left) / NO-FIT, with `~` when a part is estimated, and
UNSURE when a part is NOT COUNTED (unless it already doesn't fit).
"""

from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from mse_llm import gguf
from mse_llm.config import Config, ModelCfg
from mse_llm.util import exe
from mse_llm.variants import DRAFT_BUILTIN, Variant

MIB = 2 ** 20
DEFAULT_UB = 512                    # llama.cpp default -ub
DEFAULT_MARGIN = 768                # MiB below which a fit counts as TIGHT
SUGGEST_UB_STEP = 256

# bytes per element of a KV cache type (block size 32 for the quantized ones)
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "f8_e4m3": 1.0, "q8_0": 34 / 32, "q5_1": 24 / 32,
            "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}
DRAFT_KV_DEFAULT = "f16"            # llama-server -ctkd / -ctvd default

MEASURED, ESTIMATED, NONE, NOT_COUNTED, NOT_USED = "measured", "estimated", "none", "NOT COUNTED", "not used"

Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


class FitError(RuntimeError):
    pass


# ---------------------------------------------------------------- llama-fit-params

def fit_params_argv(cfg: Config, engine: str | None, model_path: str, v: Variant, ctx: int) -> list[str]:
    """llama-fit-params for the main model: the variant's flags minus --spec-* (which it rejects)."""
    _, eng = cfg.engine(engine)
    plain = v.without_spec()
    flags, _ = plain.flags("server")
    argv = [str(exe(eng.bin_path() / "llama-fit-params")), "-m", model_path, "-c", str(ctx),
            "-ngl", str(cfg.defaults.ngl), *flags]
    lm = v.load_mode or cfg.defaults.load_mode
    if lm:
        argv += ["-lm", lm]
    return argv + ["--fit", "off", "--fit-print", "on"]


_LINE = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")


@dataclass
class DeviceMem:
    device: str
    model: int
    context: int
    compute: int

    @property
    def total(self) -> int:
        return self.model + self.context + self.compute

    @property
    def is_host(self) -> bool:
        return self.device.lower() in ("host", "cpu")


def parse_fit_print(stdout: str) -> list[DeviceMem]:
    """`CUDA0 1286 745 4040` / `Host 15167 0 640` (device, model, context, compute in MiB)."""
    out = []
    for line in stdout.splitlines():
        m = _LINE.match(line.strip())
        if m:
            out.append(DeviceMem(m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))))
    return out


def _error_text(stderr: str, rc: int) -> str:
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("error:"):
            return ln
    errs = [re.sub(r"^[\d.]+\s+E\s+", "", ln) for ln in lines if re.match(r"^[\d.]+\s+E\s", ln)]
    if errs:
        return errs[0]
    return f"llama-fit-params exited with {rc}" + (f": {lines[-1]}" if lines else "")


def run_fit_params(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)


# ---------------------------------------------------------------- parts

@dataclass
class Part:
    name: str
    gpu_mib: float | None             # None = unknown (NOT COUNTED)
    source: str
    detail: str = ""
    host_mib: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def kv_mib(info: gguf.ModelInfo, layers: int, ctx: int, ctk: str, ctv: str) -> float:
    if not layers or not info.n_head_kv or not info.key_length:
        return 0.0
    k = info.n_head_kv * info.key_length * KV_BYTES.get(ctk, 2.0)
    vv = info.n_head_kv * (info.value_length or info.key_length) * KV_BYTES.get(ctv, 2.0)
    return layers * (k + vv) * ctx / MIB


def logits_mib_per_ub_token(info: gguf.ModelInfo) -> float:
    """Compute buffer per ubatch token ≈ the logits row: vocab × 4 bytes (§4.6, ~1 MiB for 248k-262k)."""
    return (info.vocab or 0) * 4 / MIB


def draft_parts(model_key: str | None, drafter: str | None, v: Variant, ctx: int,
                main_info: gguf.ModelInfo | None, info_of: Callable[[str], gguf.ModelInfo]) -> list[Part]:
    """Parts for speculative decoding: n-gram = no VRAM; draft-* = drafter weights + its KV / compute."""
    if not v.spec_type:
        return []
    if not v.uses_draft_model:
        return [Part(f"spec {v.spec_type}", 0, NONE, "runs on the CPU, no extra VRAM")]
    where = f"models.{model_key}.draft" if model_key else "--draft"
    if not drafter:
        return [Part("drafter", None, NOT_COUNTED, f"{v.spec_type} requested, but {where} is not set")]
    ub = v.ub or DEFAULT_UB
    if drafter.lower() in (DRAFT_BUILTIN, "build-in"):
        if main_info is None:
            return [Part("drafter", None, NOT_COUNTED, "builtin MTP, but the model GGUF couldn't be read")]
        if not main_info.nextn_layers:
            return [Part("drafter", None, NOT_COUNTED,
                         f"{where} = builtin, but the GGUF has no MTP / NextN layers")]
        kv = kv_mib(main_info, main_info.nextn_layers, ctx, v.draft_ctk or DRAFT_KV_DEFAULT,
                    v.draft_ctv or DRAFT_KV_DEFAULT)
        comp = ub * logits_mib_per_ub_token(main_info)
        return [Part("draft weights", 0, MEASURED,
                     f"built-in MTP ({main_info.nextn_layers} layer): inside the main model weights"),
                Part("draft KV+compute", kv + comp, ESTIMATED,
                     f"KV {kv:.0f} ({main_info.nextn_layers} layer, {v.draft_ctk or DRAFT_KV_DEFAULT}, "
                     f"ctx {ctx}) + compute {comp:.0f} (ub {ub})")]
    p = Path(drafter)
    try:
        di = info_of(str(p))
    except FileNotFoundError:
        return [Part("drafter", None, NOT_COUNTED, f"draft GGUF not found: {drafter}")]
    except (OSError, gguf.GgufError) as e:
        return [Part("drafter", None, NOT_COUNTED, f"can't read draft GGUF: {e}")]
    own_kv = max(0, (di.n_layers or 0) - di.shared_kv_layers)
    kv = kv_mib(di, own_kv, ctx, v.draft_ctk or DRAFT_KV_DEFAULT, v.draft_ctv or DRAFT_KV_DEFAULT)
    comp = ub * logits_mib_per_ub_token(di)
    kv_note = (f"KV {kv:.0f} ({own_kv} layers, f16)" if own_kv
               else f"KV 0 (all {di.n_layers} layers share the main model's KV)")
    return [Part("draft weights", di.weights_mib, MEASURED, f"{p.name} (GGUF tensor sizes, from {where})"),
            Part("draft KV+compute", kv + comp, ESTIMATED, f"{kv_note} + compute {comp:.0f} (ub {ub})")]


def mmproj_part(mmproj: str | None, info_of: Callable[[str], gguf.ModelInfo]) -> Part:
    if not mmproj:
        return Part("mmproj", 0, NOT_USED, "no --mmproj")
    p = Path(mmproj)
    try:
        mi = info_of(str(p))
    except FileNotFoundError:
        return Part("mmproj", None, NOT_COUNTED, f"mmproj not found: {mmproj}")
    except (OSError, gguf.GgufError) as e:
        return Part("mmproj", None, NOT_COUNTED, f"can't read mmproj: {e}")
    return Part("mmproj", mi.weights_mib, ESTIMATED,
                f"{p.name}: weights {mi.weights_mib:.0f}; image compute buffer not included")


# ---------------------------------------------------------------- one row

@dataclass
class FitRow:
    variant: str
    ctx: int
    note: str = ""
    argv: list[str] = field(default_factory=list)
    devices: list[DeviceMem] = field(default_factory=list)
    parts: list[Part] = field(default_factory=list)
    free_mib: int | None = None
    margin_mib: int = DEFAULT_MARGIN
    error: str | None = None
    suggest: list[str] = field(default_factory=list)

    @property
    def main_gpu(self) -> DeviceMem | None:
        gpus = [d for d in self.devices if not d.is_host]
        if not gpus:
            return None
        return DeviceMem("+".join(d.device for d in gpus), sum(d.model for d in gpus),
                         sum(d.context for d in gpus), sum(d.compute for d in gpus))

    @property
    def gpu_mib(self) -> float:
        return sum(p.gpu_mib or 0 for p in self.parts)

    @property
    def host_mib(self) -> float:
        return sum(p.host_mib for p in self.parts)

    @property
    def host_unsure_mib(self) -> float:
        """Drafter / mmproj buffers: fit knows their size but not whether llama.cpp puts them in VRAM or in pinned
        host memory (which shows up as shared GPU memory). Measured: gemma QAT + its 240 MiB MTP drafter added
        +360 MiB of shared memory beyond the main model's host part."""
        return sum(p.gpu_mib or 0 for p in self.parts if p.name.startswith(("draft", "mmproj")))

    @property
    def left_mib(self) -> float | None:
        return None if self.free_mib is None or self.error else self.free_mib - self.gpu_mib

    @property
    def uncounted(self) -> list[Part]:
        return [p for p in self.parts if p.source == NOT_COUNTED]

    @property
    def estimated(self) -> list[Part]:
        return [p for p in self.parts if p.source == ESTIMATED]

    @property
    def verdict(self) -> str:
        if self.error:
            return "ERROR"
        left = self.left_mib
        if left is None:
            return "?"
        base = "NO-FIT" if left < 0 else "TIGHT" if left < self.margin_mib else "fits"
        if self.uncounted and base != "NO-FIT":
            return "UNSURE"
        return base + ("~" if self.estimated else "")

    def footer(self) -> list[str]:
        out = []
        for p in self.parts:
            if p.source == ESTIMATED:
                out.append(f"~ {p.name} estimated (+{p.gpu_mib:.0f} MiB): {p.detail}")
            elif p.source == NOT_COUNTED:
                fix = ('set draft = "builtin" (MTP inside the GGUF) or a draft GGUF path'
                       if p.name == "drafter" else "")
                out.append(f"! {p.name} NOT COUNTED: {p.detail}" + (f"; {fix}" if fix else ""))
        return out

    def as_dict(self) -> dict[str, Any]:
        mg = self.main_gpu
        return {
            "variant": self.variant, "ctx": self.ctx, "note": self.note, "argv": self.argv,
            "gpu_mib": round(self.gpu_mib), "host_mib": round(self.host_mib),
            "main": asdict(mg) if mg else None,
            "devices": [asdict(d) for d in self.devices],
            "parts": [p.as_dict() for p in self.parts],
            "free_mib": self.free_mib, "left_mib": None if self.left_mib is None else round(self.left_mib),
            "verdict": self.verdict, "error": self.error, "footer": self.footer(), "suggest": self.suggest,
        }


# ---------------------------------------------------------------- suggest

MAX_CHECKS = 6                      # extra llama-fit-params calls per row to confirm an -ncmoe suggestion


NOTHING_FITS = (-1, 0.0)


def _check_ncmoe(start: int, L: int, fits: Callable[[int], tuple[bool, float] | None]) -> tuple[int, float] | None:
    """From the linear-rule estimate, walk -ncmoe with real llama-fit-params calls to the lowest value that
    fits with the margin. (n, MiB left); NOTHING_FITS when even L doesn't; None when it can't be checked
    (a call failed, or MAX_CHECKS ran out before an answer)."""
    n = max(0, min(start, L))
    r = fits(n)
    if r is None:
        return None
    calls, best = 1, (n, r[1]) if r[0] else None
    if r[0]:
        while n > 0 and calls < MAX_CHECKS:
            r = fits(n - 1)
            calls += 1
            if r is None or not r[0]:
                break
            n -= 1
            best = (n, r[1])
    else:
        while n < L and calls < MAX_CHECKS:
            n += 1
            r = fits(n)
            calls += 1
            if r is None:
                return None
            if r[0]:
                return n, r[1]
        return NOTHING_FITS if n >= L else None
    return best


def suggestions(row: FitRow, v: Variant, info: gguf.ModelInfo | None,
                measure: Callable[[int], float | None] | None = None) -> list[str]:
    """The two linear rules (§4.6): one expert layer per -ncmoe step, ~vocab × 4 B per ubatch token.
    `measure(n)` (main-model GPU MiB at -ncmoe n, from llama-fit-params) confirms the -ncmoe value:
    expert layers differ in size (quant mix), so the linear step alone can be off by a layer."""
    left = row.left_mib
    if info is None or left is None or row.error:
        return []
    if row.uncounted:
        return ["no suggestion: a part is NOT COUNTED, so the VRAM left is unknown"]
    out: list[str] = []
    budget = left - row.margin_mib
    L, layer = info.moe_layers, info.expert_layer_mib
    on_cpu = L if v.cpu_moe else (min(v.n_cpu_moe, L) if v.n_cpu_moe is not None else None)
    mg = row.main_gpu
    others = row.gpu_mib - (mg.total if mg else 0)            # drafter / mmproj: don't depend on -ncmoe

    def fits(n: int) -> tuple[bool, float] | None:
        g = measure(n) if measure else None
        if g is None:
            return None
        lft = (row.free_mib or 0) - g - others
        return lft >= row.margin_mib, lft

    if L and layer and on_cpu is not None:
        if budget >= layer:
            est = on_cpu - min(on_cpu, int(budget // layer))
            chk = _check_ncmoe(est, on_cpu, fits) if measure else None
            chk = None if chk == NOTHING_FITS else chk
            lowest, how = (chk[0], f"checked: {chk[1]:.0f} MiB left") if chk else (est, "linear estimate")
            steps = on_cpu - lowest
            if steps:
                tries = sorted({on_cpu - round(k * steps / 3) for k in (1, 2, 3)} - {on_cpu}, reverse=True)
                out.append(f"{budget:.0f} MiB unused above the margin: lowest safe -ncmoe "
                           f"{'=' if chk else '≈'} {lowest} ({how}; {_layers(steps)} back on the GPU, "
                           f"~{layer:.0f} MiB each) → try {_moe_pattern(v, tries)}")
        elif budget < 0:
            est = on_cpu + math.ceil(-budget / layer)
            chk = _check_ncmoe(min(est, L), L, fits) if measure else None
            if chk == NOTHING_FITS or (chk is None and est > L):
                out.append("doesn't fit even with all experts on the CPU; lower -ub or the context")
            elif chk:
                out.append(f"needs -ncmoe ≥ {chk[0]} to fit with the margin (checked: {chk[1]:.0f} MiB left; "
                           f"{_layers(chk[0] - on_cpu)} more on the CPU)")
            else:
                out.append(f"needs -ncmoe ≥ {est} to fit with the margin (linear estimate; "
                           f"{_layers(est - on_cpu)} more on the CPU, ~{layer:.0f} MiB each)")
    ub_now = v.ub or DEFAULT_UB
    measured_rate = mg.compute / ub_now if mg and ub_now >= 1024 else 0.0   # small ub: fixed buffers dominate
    with_draft = any(p.name == "draft KV+compute" for p in row.parts)   # the draft context uses the same -ub
    per_tok = max(logits_mib_per_ub_token(info), measured_rate) * (2 if with_draft else 1)
    if per_tok and budget > 0:
        ub = v.ub or DEFAULT_UB
        ub_max = (ub + int(budget // per_tok)) // SUGGEST_UB_STEP * SUGGEST_UB_STEP
        if ub_max > ub:
            out.append(f"or largest -ub ≈ {ub_max} (each ubatch token ≈ {per_tok:.2f} MiB of compute buffer"
                       f"{', main + draft' if with_draft else ''}); -ncmoe and -ub share the same VRAM")
    elif per_tok and budget < 0 and (v.ub or DEFAULT_UB) > DEFAULT_UB:
        ub = v.ub or DEFAULT_UB
        ub_fit = (ub - math.ceil(-budget / per_tok)) // SUGGEST_UB_STEP * SUGGEST_UB_STEP
        if ub_fit >= SUGGEST_UB_STEP:
            out.append(f"or -ub ≤ {ub_fit} ({per_tok:.2f} MiB per ubatch token)")
    return out


def _layers(n: int) -> str:
    return f"{n} expert layer{'' if n == 1 else 's'}"


def _moe_pattern(v: Variant, values: list[int]) -> str:
    if not values:
        return v.name
    marker = 987654321
    name = replace(v, n_cpu_moe=marker, cpu_moe=False).name
    inner = str(values[0]) if len(values) == 1 else "{" + ",".join(map(str, values)) + "}"
    return name.replace(f"moe{marker}", f"moe{inner}")


# ---------------------------------------------------------------- the whole command

@dataclass
class FitInput:
    variant: Variant
    note: str = ""


@dataclass
class FitReport:
    model: str
    model_key: str | None
    path: str
    free_mib: int | None
    free_source: str
    margin_mib: int
    info: gguf.ModelInfo | None
    rows: list[FitRow]
    warnings: list[str] = field(default_factory=list)

    @property
    def any_uncounted(self) -> bool:
        return any(r.uncounted for r in self.rows)

    @property
    def any_error(self) -> bool:
        return any(r.error for r in self.rows)

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "model_key": self.model_key, "path": self.path, "free_mib": self.free_mib,
                "free_source": self.free_source, "margin_mib": self.margin_mib,
                "info": self.info.as_dict() if self.info else None, "warnings": self.warnings,
                "rows": [r.as_dict() for r in self.rows]}


def resolve_model(cfg: Config, name: str) -> tuple[str | None, ModelCfg | None, str]:
    """GGUF_OR_MODEL: a configured model (key / alias), else a path to a .gguf."""
    try:
        key, m = cfg.model(name)
        return key, m, m.path
    except KeyError:
        pass
    if Path(name).suffix.lower() == ".gguf" or Path(name).is_file():
        return None, None, name
    raise FitError(f"{name!r} is neither a configured model nor a .gguf path")


def apply_kv(v: Variant, kv: str | None) -> Variant:
    """--kv K/V: the cache types for variants that don't set their own."""
    if not kv:
        return v
    k, _, val = kv.partition("/")
    return replace(v, ctk=v.ctk or k, ctv=v.ctv or (val or k))


def run_fit(cfg: Config, model: str, inputs: list[FitInput], ctxs: list[int], *, engine: str | None = None,
            free_mib: int | None = None, free_source: str = "", margin_mib: int = DEFAULT_MARGIN,
            draft: str | None = None, mmproj: str | None = None, suggest: bool = False,
            runner: Runner | None = None, info_of: Callable[[str], gguf.ModelInfo] | None = None,
            progress: Callable[[int, int, str], None] | None = None) -> FitReport:
    runner = runner or run_fit_params
    info_of = info_of or gguf.info
    key, m, path = resolve_model(cfg, model)
    if not ctxs:
        if m is None or not m.ctx:
            raise FitError(f"{model}: no context size (--ctx, or [models.*].ctx)")
        ctxs = [m.ctx]
    warnings: list[str] = []
    info: gguf.ModelInfo | None = None
    try:
        info = info_of(path)
    except FileNotFoundError:
        raise FitError(f"model file not found: {path}") from None
    except (OSError, gguf.GgufError) as e:
        warnings.append(f"can't read the GGUF header ({e}); --suggest and draft estimates are off")
    drafter = draft or (m.draft if m else None)
    mm = mmproj or (m.mmproj or m.serve_extra.get("mmproj") if m else None)
    cache: dict[tuple[str, ...], tuple[list[DeviceMem], str | None]] = {}

    def measure(argv: list[str]) -> tuple[list[DeviceMem], str | None]:
        k = tuple(argv)
        if k not in cache:
            try:
                cp = runner(argv)
                devs = parse_fit_print(cp.stdout or "")
                err = None if cp.returncode == 0 and devs else _error_text(cp.stderr or "", cp.returncode)
            except FileNotFoundError:
                raise FitError(f"llama-fit-params not found: {argv[0]} (build it with `mse-llm build run`, "
                               "or pick another --engine)") from None
            except (OSError, subprocess.SubprocessError) as e:
                devs, err = [], f"can't run llama-fit-params: {e}"
            cache[k] = (devs, err)
        return cache[k]

    def main_gpu_at(v: Variant, c: int) -> Callable[[int], float | None]:
        def at(n: int) -> float | None:
            devs, err = measure(fit_params_argv(cfg, engine, path, replace(v, n_cpu_moe=n, cpu_moe=False), c))
            return None if err else sum(d.total for d in devs if not d.is_host)
        return at

    rows: list[FitRow] = []
    total, done = len(ctxs) * len(inputs), 0
    for c in ctxs:
        for inp in inputs:
            v = inp.variant
            row = FitRow(v.name, c, inp.note, free_mib=free_mib, margin_mib=margin_mib)
            row.argv = fit_params_argv(cfg, engine, path, v, c)
            if progress:
                progress(done, total, f"{v.name} ctx {c}")
            row.devices, row.error = measure(row.argv)
            done += 1
            if not row.error:
                mg = row.main_gpu
                host = sum(d.total for d in row.devices if d.is_host)
                row.parts.append(Part("main", mg.total if mg else 0, MEASURED,
                                      f"{mg.model}+{mg.context}+{mg.compute} (model+KV+compute)" if mg else "",
                                      host_mib=host))
                row.parts += draft_parts(key, drafter, v, c, info, info_of)
                row.parts.append(mmproj_part(mm, info_of))
                if suggest:
                    row.suggest = suggestions(row, v, info, main_gpu_at(v, c) if free_mib is not None else None)
            rows.append(row)
    if progress:
        progress(total, total, "")
    if info and info.moe_layers:
        over = sorted({i.variant.n_cpu_moe for i in inputs
                       if i.variant.n_cpu_moe is not None and i.variant.n_cpu_moe > info.moe_layers})
        if over:
            warnings.append(f"{', '.join(f'moe{n}' for n in over)} ≥ the model's {info.moe_layers} MoE layers: "
                            "all experts on the CPU")
    return FitReport(model, key, path, free_mib, free_source, margin_mib, info, rows, warnings)


def free_vram_now() -> tuple[int | None, str]:
    """Free VRAM summed over the NVIDIA GPUs (NVML), or the hardware fixture's value in tests."""
    from mse_llm import hw

    info = hw.detect()
    gpus = [g for g in info.gpus if g.vendor == "nvidia" and g.vram_free_mib is not None]
    if not gpus:
        return None, "no NVIDIA GPU / NVML: pass --free MiB"
    src = "NVML now" if not info.fixture else f"hw fixture {Path(str(info.fixture)).name}"
    return sum(g.vram_free_mib for g in gpus), src
