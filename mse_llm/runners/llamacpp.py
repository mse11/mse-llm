"""llama.cpp command builders: llama-server, llama-bench, router presets, help (§3.3).

The argv order follows the bash script (`run_serve_params_cmd_build`, `run_bench_params_cmd_build`,
`run_presets_serve`), so `--dry-run` output can be compared line by line. Deliberate differences
(bash bugs and §0.2 updates) are listed in docs/PARITY.md, "M3: argv differences".
"""

from __future__ import annotations

from typing import Any, Literal

from mse_llm import flagcheck
from mse_llm.config import Config, ModelCfg
from mse_llm.runners.base import Command
from mse_llm.util import exe
from mse_llm.variants import Variant, VariantError, decode

Fit = Literal["on", "off"]


def as_variant(v: Variant | str | None) -> Variant:
    if v is None:
        return Variant()
    return v if isinstance(v, Variant) else decode(v)


def _bench_ngl(ngl: str) -> str:
    """llama-bench -ngl takes a number only: all -> 999 (as the bash script), auto -> -1 (its default)."""
    return {"all": "999", "auto": "-1"}.get(str(ngl), str(ngl))


def option_flags(opts: dict[str, Any], helps: dict, tool: str = "llama-server") -> list[str]:
    """TOML table -> flags, e.g. {jinja=true, reasoning-format="none", temp=0.3}
    -> --jinja --reasoning-format none --temp 0.3. A key gets `--key` or `-key`, whichever the
    engine's --help knows (default `--key`); true = flag only, false = --no-key, a list = comma list."""
    h = helps.get(tool)
    out: list[str] = []
    for key, val in opts.items():
        if key.startswith("-"):
            flag = key
        elif h is not None and f"--{key}" not in h and f"-{key}" in h:
            flag = f"-{key}"
        else:
            flag = f"--{key}"
        if val is True:
            out.append(flag)
        elif val is False:
            out.append(f"--no-{flag.lstrip('-')}")
        elif isinstance(val, list):
            out += [flag, ",".join(str(x) for x in val)]
        else:
            out += [flag, str(val)]
    return out


def _finish(cmd: Command, tool: str, helps: dict, snap: str | None) -> Command:
    cmd.snapshot = snap
    cmd.errors += flagcheck.check_argv(cmd.argv, tool, helps)
    return cmd


def _variant_flags(cmd: Command, v: Variant, tool: Literal["server", "bench"], model: ModelCfg) -> list[str]:
    try:
        flags, warn = v.flags(tool, draft=model.draft)
    except VariantError as e:
        cmd.errors.append(str(e))
        return []
    cmd.warnings += warn
    return flags


def server_command(cfg: Config, model: str, variant: Variant | str | None, ctx: int | None = None,
                   engine: str | None = None, *, fit: Fit = "on", port: int | None = None,
                   purpose: str = "serve") -> Command:
    """llama-server for one model (bash run_serve_params_cmd_build).

    fit: "on" for `serve` (a too-large config is shrunk, §0.3), "off" for the background server of
    benchy / eval, so the measured config is exactly the requested one."""
    mkey, m = cfg.model(model)
    ename, eng = cfg.engine(engine)
    v = as_variant(variant)
    d = cfg.defaults
    ctx = ctx or m.ctx
    snap, helps = flagcheck.load_helps(eng.help_snapshot, eng.bin_path())
    cmd = Command("llama-server", [], label=f"{purpose} {mkey} {v.name}", model=mkey, variant=v.name, ctx=ctx)
    if ctx is None:
        cmd.errors.append(f"{mkey}: no context size (--ctx, or [models.{mkey}].ctx)")
    argv = [str(exe(eng.bin_path() / "llama-server")), "-t", str(d.threads)]
    if ctx is not None:
        argv += ["-c", str(ctx)]
    argv += ["--parallel", str(d.parallel), "-ngl", str(d.ngl), "-m", m.path]
    argv += _variant_flags(cmd, v, "server", m)
    lm = v.load_mode or d.load_mode
    if lm:
        argv += ["-lm", lm]
    argv += option_flags(m.serve_extra, helps)
    argv += ["--fit", fit, "--host", eng.host, "--port", str(port or eng.port)]
    cmd.argv = argv
    return _finish(cmd, "llama-server", helps, snap)


def bench_command(cfg: Config, model: str, variant: Variant | str | None, ctx: int | None = None,
                  engine: str | None = None, *, tg: list[int] | None = None, repetitions: int | None = None,
                  output: str | None = "jsonl") -> Command:
    """llama-bench (bash run_bench_params_cmd_build). -p is the prompt size (bash runner_ctx)."""
    mkey, m = cfg.model(model)
    ename, eng = cfg.engine(engine)
    v = as_variant(variant)
    d = cfg.defaults
    ctx = ctx or m.ctx
    snap, helps = flagcheck.load_helps(eng.help_snapshot, eng.bin_path())
    cmd = Command("llama-bench", [], label=f"bench {mkey} {v.name}", model=mkey, variant=v.name, ctx=ctx)
    if ctx is None:
        cmd.errors.append(f"{mkey}: no prompt size (--ctx, or [models.{mkey}].ctx)")
    tgs = tg or d.bench_tg
    argv = [str(exe(eng.bin_path() / "llama-bench")), "-t", str(d.threads)]
    if ctx is not None:
        argv += ["-p", str(ctx)]
    argv += ["-n", ",".join(str(x) for x in tgs), "-ngl", _bench_ngl(d.ngl), "-m", m.path]
    argv += _variant_flags(cmd, v, "bench", m)
    lm = v.load_mode or d.load_mode
    if lm:
        argv += ["-lm", lm]
    argv += list(m.bench_extra)
    reps = repetitions if repetitions is not None else d.bench_repetitions
    argv += ["-r", str(reps)]
    cmd.params = {"tg": list(tgs), "repetitions": reps}
    if output:
        argv += ["-o", output]
    cmd.argv = argv
    return _finish(cmd, "llama-bench", helps, snap)


def presets_serve_command(cfg: Config, file: str | None = None, engine: str | None = None, *,
                          port: int | None = None, models_max: int | None = None,
                          memory_margin_mib: int | None = None) -> Command:
    """Router: llama-server --models-preset INI (bash run_presets_serve)."""
    from pathlib import Path

    ename, eng = cfg.engine(engine or cfg.presets.engine)
    path = Path(file) if file else cfg.presets.path()
    snap, helps = flagcheck.load_helps(eng.help_snapshot, eng.bin_path())
    cmd = Command("router", [], label=f"presets serve {path.name}")
    if not path.is_file():
        cmd.warnings.append(f"presets file not found: {path}")
    argv = [str(exe(eng.bin_path() / "llama-server")), "--models-preset", str(path),
            "--host", eng.host, "--port", str(port or eng.port)]
    if models_max is not None:
        argv += ["--models-max", str(models_max)]
    if memory_margin_mib is not None:
        argv += ["--models-memory-margin", str(memory_margin_mib)]
    cmd.argv = argv
    return _finish(cmd, "llama-server", helps, snap)


def help_command(cfg: Config, tool: Literal["server", "bench"], engine: str | None = None) -> Command:
    ename, eng = cfg.engine(engine)
    return Command(f"llama-{tool}", [str(exe(eng.bin_path() / f"llama-{tool}")), "--help"], label=f"help {tool}")
