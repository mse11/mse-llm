"""`serve` and `presets *` commands (M5, §4.3). Registered on the main group by mse_llm.cli."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import click
from rich.markup import escape
from rich.table import Table

from mse_llm.cli import Ctx, _plain, _print_commands, _variants_for, console, json_option, main, pass_ctx

OUTPUT = click.Choice(["all", "filter", "none"])


# ---------------------------------------------------------------- shared helpers

def _engine(ctx: Ctx, engine: str | None):
    from mse_llm import flagcheck

    try:
        name, eng = ctx.cfg.engine(engine)
    except KeyError as e:
        raise click.ClickException(str(e.args[0])) from e
    snap, helps = flagcheck.load_helps(eng.help_snapshot, eng.bin_path())
    return name, eng, snap, helps


def _filter(ctx: Ctx, mode: str | None):
    from mse_llm.proc import OutputFilter

    d = ctx.cfg.defaults
    return OutputFilter(mode or d.output, d.output_filter)


def _printer(filt):
    def sink(line: str) -> None:
        if filt(line):
            _plain(line)
    return sink


def _interactive() -> bool:
    """stdin is a real terminal (on Windows NUL passes isatty(), so ask the console API)."""
    if not sys.stdin.isatty():
        return False
    from mse_llm.util import IS_WINDOWS

    if not IS_WINDOWS:
        return True
    import ctypes
    import msvcrt

    try:
        mode = ctypes.c_uint32()
        return bool(ctypes.windll.kernel32.GetConsoleMode(msvcrt.get_osfhandle(sys.stdin.fileno()), ctypes.byref(mode)))
    except (OSError, ValueError):
        return False


def _need_yes(yes: bool, question: str) -> None:
    if yes:
        return
    if not _interactive():
        raise click.ClickException(f"{question} (no terminal to ask: pass --yes to start anyway)")
    if not click.confirm(question, default=False):
        raise click.Abort()


def _prestart_checks(port: int) -> None:
    from mse_llm import proc as pr
    from mse_llm import serve as sv

    if pr.port_in_use(port):
        raise click.ClickException(pr.describe_port(port) + " (`mse-llm presets stop`, "
                                   "`mse-llm hw processes --stop-ollama`, or another --port)")
    others = sv.other_engines()
    if others:
        hint = ("`mse-llm hw processes --stop-ollama` ends Ollama"
                if any(o.lower().startswith("ollama") for o in others) else "stop them before benching")
        console.print(f"[yellow]![/] other inference engines are running: {escape(', '.join(others))} "
                      f"(they hold VRAM; {hint})")


def _print_stop(rep) -> None:
    if rep is None:
        return
    style = "green" if rep.clean else "red"
    console.print(f"[{style}]■[/] {escape(rep.summary())}")


# ---------------------------------------------------------------- serve

def _monitor_options(f):
    for opt in reversed([
        click.option("--no-monitor", is_flag=True, help="No VRAM / shared memory / RAM monitoring."),
        click.option("--no-guard", is_flag=True,
                     help="Monitor, but don't stop the server on a spill / RAM / commit trip."),
        click.option("--mem-cap", type=int, default=None, metavar="MiB",
                     help="Windows Job Object cap on the server tree's committed memory "
                          "(default: [monitor].job_memory_cap_mib)."),
    ]):
        f = opt(f)
    return f


def _print_fit_check(chk) -> None:
    from mse_llm.serve import BAD_VERDICTS

    v = chk.verdict
    style = "green" if v.rstrip("~") not in BAD_VERDICTS else ("magenta" if v == "UNSURE" else "red")
    r = chk.row
    if chk.error and (r is None or r.error):
        console.print(f"[red]fit check failed:[/] {escape(chk.error)}")
        return
    head = f"fit check: [{style}]{v}[/]  GPU {r.gpu_mib:.0f} MiB"
    if r.free_mib is not None:
        head += f" of {r.free_mib} MiB free ({escape(chk.report.free_source)}), left {r.left_mib:.0f}"
    console.print(head)
    for line in r.footer():
        console.print(f"  {escape(line)}")
    for s in r.suggest:
        console.print(f"  → {escape(s)}")
    p = chk.preview
    if p is not None:
        console.print(f"  [bold]--fit on:[/] {escape(p.message)}")
        for c in p.changes:
            console.print(f"    {escape(c)}")
        u = p.unlocked
        if u is not None:
            if u.ok and u.changes:
                console.print("  with -ngl and -ncmoe left unset ([bold]serve --let-fit[/]), --fit would change:")
                for c in u.changes:
                    console.print(f"    {escape(c)}")
            elif u.ok:
                console.print("  with -ngl and -ncmoe left unset (serve --let-fit): no change")
            else:
                console.print(f"  with -ngl and -ncmoe left unset (serve --let-fit): {escape(u.message)}")


def _drop_flag(argv: list[str], *names: str) -> list[str]:
    out, i = [], 0
    while i < len(argv):
        if argv[i] in names:
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def _preset_server_command(ctx: Ctx, name: str, file_, engine, ctx_, port):
    """`serve --preset NAME`: one model outside the router with that section's merged args."""
    from mse_llm import flagcheck
    from mse_llm import presets as ps
    from mse_llm.runners import Command
    from mse_llm.util import exe

    path = Path(file_) if file_ else ctx.cfg.presets.path()
    if not path.is_file():
        raise click.ClickException(f"presets file not found: {path}")
    _, eng, snap, helps = _engine(ctx, engine)
    pf = ps.parse(path)
    try:
        d = ps.dump(pf, name, helps.get("llama-server"), ctx.cfg)
    except KeyError as e:
        raise click.ClickException(str(e.args[0])) from e
    flags = list(d.flags)
    if ctx_:
        flags = _drop_flag(flags, "-c", "--ctx-size") + ["--ctx-size", str(ctx_)]
    merged = pf.merged(d.name)
    if "fit" not in merged:
        flags += ["--fit", "on"]
    argv = [str(exe(eng.bin_path() / "llama-server")), *flags, "--host", eng.host, "--port", str(port or eng.port)]
    cmd = Command("llama-server", argv, label=f"serve --preset {d.name}", model=d.model_key, variant=d.variant,
                  ctx=ctx_ or d.ctx, snapshot=snap)
    cmd.errors += flagcheck.check_argv(argv, "llama-server", helps)
    if d.skipped:
        cmd.warnings.append(f"router-only keys left out: {', '.join(d.skipped)}")
    return cmd, d, merged


def _serve_list(ctx: Ctx, file_, engine) -> None:
    """Models from [models.*] and sections of the presets INI, each with the command that serves it."""
    from mse_llm import presets as ps

    models = []
    for key, m in ctx.cfg.models.items():
        models.append({"name": key, "alias": m.alias, "file": Path(m.path).name, "found": Path(m.path).is_file(),
                       "ctx": m.ctx, "default_serve": m.default_serve, "default_serve_spec": m.default_serve_spec,
                       "tags": m.tags, "command": f"mse-llm serve {key}" if m.default_serve else None,
                       "command_spec": f"mse-llm serve {key} --spec" if m.default_serve_spec else None})
    path = Path(file_) if file_ else ctx.cfg.presets.path()
    presets = []
    if path.is_file():
        _, _, _, helps = _engine(ctx, engine or ctx.cfg.presets.engine)
        pf = ps.parse(path)
        for s in pf.models:
            d = ps.dump(pf, s.name, helps.get("llama-server"), ctx.cfg)
            presets.append({"name": s.name, "ctx": d.ctx, "variant": d.variant, "model_key": d.model_key,
                            "found": bool(d.model_path) and Path(d.model_path).is_file(),
                            "command": f"mse-llm serve --preset {s.name}"})
    if ctx.json:
        click.echo(json.dumps({"models": models, "presets_file": str(path), "presets": presets}, indent=2))
        return
    ok = "[green]✓[/]"
    bad = "[red]✗ file missing[/]"
    t = Table(box=None, pad_edge=False, header_style="bold", title="models   →  mse-llm serve NAME [--spec]",
              title_justify="left", title_style="bold")
    for c in ("NAME", "CTX", "default_serve", "--spec", "TAGS", ""):
        t.add_column(c, overflow="fold")
    for r in models:
        t.add_row(escape(r["name"]), str(r["ctx"] or "-"), escape(r["default_serve"] or "-"),
                  escape(r["default_serve_spec"] or "-"), escape(",".join(r["tags"])), ok if r["found"] else bad)
    console.print(t)
    if presets:
        console.print()
        title = f"presets in {path.name}   →  mse-llm serve --preset NAME   (all of them: mse-llm presets serve)"
        t = Table(box=None, pad_edge=False, header_style="bold", title_justify="left", title_style="bold",
                  title=title)
        for c in ("NAME", "CTX", "VARIANT ≈", "CONFIG MODEL", ""):
            t.add_column(c, overflow="fold")
        for r in presets:
            t.add_row(escape(r["name"]), str(r["ctx"] or "-"), escape(r["variant"]), escape(r["model_key"] or "-"),
                      ok if r["found"] else bad)
        console.print(t)
    console.print("\n[dim]Other variants: -v VARIANT (`mse-llm variants expand PATTERN`), other context: --ctx N. "
                  "Check memory first: `mse-llm fit NAME`. Preview: --dry-run.[/]")


@main.command("serve")
@click.argument("model", required=False)
@click.option("-v", "--variant", "patterns", multiple=True, help="Variant (default: [models].default_serve).")
@click.option("--spec", is_flag=True, help="Use default_serve_spec (bash SD_ENABLED=1).")
@click.option("--preset", default=None, metavar="NAME",
              help="Serve one section of the presets INI (outside the router).")
@click.option("--file", "file_", type=click.Path(dir_okay=False), default=None, help="Presets INI for --preset.")
@click.option("--ctx", "ctx_", type=int, default=None, help="Context -c (default: [models].ctx).")
@click.option("--port", type=int, default=None)
@click.option("--output", type=OUTPUT, default=None,
              help="Console output: all | filter | none (default: [defaults].output).")
@click.option("--yes", "-y", is_flag=True, help="Start without asking, even after a bad fit check.")
@click.option("--no-fit-check", is_flag=True, help="Skip the llama-fit-params pre-check.")
@click.option("--let-fit", is_flag=True,
              help="Leave -ngl and -ncmoe out, so llama.cpp's --fit decides the GPU/CPU placement.")
@click.option("--all", "all_", is_flag=True,
              help="Every model with a default_serve (bash run_PRESETS_dump; with --dry-run).")
@click.option("--list", "list_", is_flag=True,
              help="What can be served: models and presets, with the command for each.")
@_monitor_options
@click.option("--engine", default=None)
@click.option("--dry-run", is_flag=True,
              help="Print the command (flag-checked) and stop (bash MSE_SERVER_PRESETS_DUMP).")
@json_option
@pass_ctx
def serve_cmd(ctx: Ctx, model, patterns, spec, preset, file_, ctx_, port, output, yes, no_fit_check, let_fit, all_,
              list_, no_monitor, no_guard, mem_cap, engine, dry_run) -> None:
    """llama-server for one model with --fit on, after a fit check (bash run_serve_params / run_SERVER_*).

    Without MODEL (or with --list) it lists what can be served."""
    from mse_llm import serve as sv
    from mse_llm import variants as vv
    from mse_llm.runners import server_command

    if list_ or not (model or preset or all_):
        _serve_list(ctx, file_, engine)
        return
    if all_:
        if not dry_run:
            raise click.ClickException("--all lists the commands of every model: use it with --dry-run")
        keys = [k for k, m in ctx.cfg.models.items() if (m.default_serve_spec if spec else m.default_serve)]
        cmds = [server_command(ctx.cfg, k, _variants_for(ctx, k, (), None, spec)[0], ctx_, engine, port=port)
                for k in keys]
        _print_commands(ctx, cmds)
        sys.exit(0 if all(c.ok for c in cmds) else 1)
    merged: dict = {}
    if preset:
        cmd, dump, merged = _preset_server_command(ctx, preset, file_, engine, ctx_, port)
        fit_model, variant = dump.model_path or "", vv.decode(dump.variant)
    elif model:
        vs = _variants_for(ctx, model, patterns, None, spec)
        if len(vs) > 1:
            raise click.ClickException("serve runs one variant; got " + ", ".join(v.name for v in vs))
        try:
            cmd = server_command(ctx.cfg, model, vs[0], ctx_, engine, port=port)
        except KeyError as e:
            raise click.ClickException(str(e.args[0])) from e
        fit_model, variant = model, vs[0]
    else:
        raise click.ClickException("give a MODEL, --preset NAME or --all")
    if let_fit:
        cmd.argv = sv.without_placement(cmd.argv)
        variant = sv.replace(variant, n_cpu_moe=None, cpu_moe=False)
        cmd.warnings.append("--let-fit: -ngl / -ncmoe left out, so --fit decides the GPU/CPU placement")
    _print_commands(ctx, [cmd])
    if not cmd.ok:
        console.print("[red]the command failed the flag check[/]")
        sys.exit(1)
    if dry_run:
        return
    if ctx.json:
        raise click.ClickException("--json is for --dry-run; a running server streams its log")
    _, eng, _, _ = _engine(ctx, engine)
    the_port = port or eng.port
    expected_host = None
    if let_fit and not no_fit_check and cmd.ctx:
        with console.status("llama-fit-params --fit on …"):
            path = fit_model if preset else ctx.cfg.model(fit_model)[1].path
            pv = sv.fit_preview(ctx.cfg, engine, path, variant, cmd.ctx, unlock=True, _second=False)
        msg = pv.message if pv.ok else "nothing that fits: " + pv.message
        console.print(f"  [bold]--fit on picks:[/] {escape(msg)}")
        for c in pv.changes:
            console.print(f"    {escape(c)}")
        if not pv.ok:
            _need_yes(yes, "llama.cpp found no fitting placement. Start anyway?")
    elif not no_fit_check and cmd.ctx:
        with console.status("fit check (llama-fit-params) …"):
            chk = sv.fit_check(ctx.cfg, fit_model, variant, cmd.ctx, engine,
                               draft=merged.get("model-draft") or merged.get("md"), mmproj=merged.get("mmproj"))
        _print_fit_check(chk)
        if chk.row is not None and not chk.row.error:
            expected_host = chk.row.host_mib      # pinned host buffers: shared GPU memory that is not a spill
        if chk.bad:
            _need_yes(yes, f"fit check says {chk.verdict}. Start anyway?")
    _prestart_checks(the_port)
    log = sv.servers_dir() / f"serve-{the_port}.log"
    mon = _MonOpts(not no_monitor and ctx.cfg.defaults.monitor, not no_guard, expected_host,
                   mem_cap if mem_cap is not None else ctx.cfg.monitor.job_memory_cap_mib)
    asyncio_run(_serve(ctx, cmd, eng, the_port, log, _filter(ctx, output), mon))


def asyncio_run(coro) -> None:
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        sys.exit(130)


@dataclass
class _MonOpts:
    monitor: bool = True
    guard: bool = True
    expected_host_mib: float | None = None
    mem_cap_mib: int | None = None


async def _serve(ctx: Ctx, cmd, eng, port: int, log: Path, filt, mon: _MonOpts) -> None:
    from mse_llm import serve as sv
    from mse_llm.runners import ServerHandle

    info = sv.StartupInfo()
    h = ServerHandle(cmd.argv, port=port, client_host=eng.client_host,
                     ready_timeout=ctx.cfg.defaults.server_ready_timeout,
                     log_path=log, sink=_printer(filt), memory_cap_mib=mon.mem_cap_mib)
    h.proc.watchers.append(info.feed)

    async def ready(ep) -> None:
        console.print(f"[green]✓[/] ready after {ep.ready_s:.1f} s: [bold]{ep.base_url}[/]  pid {ep.pid}")
        for d in info.differences(cmd.ctx):
            console.print(f"[yellow]! --fit changed the config:[/] {escape(d)}")

    await run_server(ctx, h, log, mon, ready, what=cmd.model or cmd.label, hint="Ctrl+C stops the server")


async def run_server(ctx: Ctx, h, log: Path, mon: _MonOpts, on_ready, *, what: str, hint: str) -> None:
    """Foreground server: monitor (baseline before the start), guard stop, stop report, summary, voice message.
    Exit codes: 1 = the server failed to start, 2 = stopped by the memory guard."""
    from contextlib import AsyncExitStack

    from mse_llm import notify
    from mse_llm.cli_monitor import make_monitor, print_summary
    from mse_llm.runners import ServerError

    async def trip(reason: str) -> None:
        console.print(f"[red]■ memory guard:[/] {escape(reason)} → stopping the server")
        _print_stop(await h.stop())

    m = None
    if mon.monitor:
        m = make_monitor(ctx, guard=mon.guard, expected_host_mib=mon.expected_host_mib,
                         csv_path=log.with_name(log.stem + ".monitor.csv"), on_trip=trip)
    note = "" if m is None else (f"; monitor every {ctx.cfg.monitor.interval:g} s, guard "
                                 + ("on" if mon.guard and ctx.cfg.monitor.guard else "off"))
    console.print(f"[dim]log: {escape(str(log))}  ({hint}){note}[/]")
    failure = None
    exited = None
    interrupted = True
    tripped = None
    try:
        async with AsyncExitStack() as stack:
            if m is not None:
                await stack.enter_async_context(m)
                m.track(lambda: h.proc.proc.pid if h.proc.proc is not None else None)
            try:
                ep = await h.start()
            except ServerError as e:
                failure = e
            else:
                if m is not None and m.tripped:      # tripped before the process existed: stop it now
                    _print_stop(await h.stop())
                else:
                    await on_ready(ep)
                    exited = await h.wait()
            finally:
                if h.stop_report is None or h.proc.returncode is None:
                    _print_stop(await h.stop())
        interrupted = False
    finally:
        if m is not None:
            tripped = m.tripped
            if m.samples:
                print_summary(m.summary(), title=f"monitor {what}")
    if tripped:
        notify.notify(f"{what}: stopped by the memory guard", ctx.cfg.defaults.notify)
        raise click.exceptions.Exit(2)
    if failure is not None:
        notify.notify(f"{what}: server failed to start", ctx.cfg.defaults.notify)
        _fail(failure, log)
    if not interrupted and exited is not None:
        console.print(f"[yellow]server exited with code {exited}[/]")
        notify.notify(f"{what}: server exited", ctx.cfg.defaults.notify)


def _fail(e, log: Path):
    console.print(f"[red]✗ {escape(str(e))}[/]")
    for line in getattr(e, "log_tail", [])[-25:]:
        _plain("  | " + line)
    console.print(f"[dim]full log: {escape(str(log))}[/]")
    raise click.exceptions.Exit(1)


# ---------------------------------------------------------------- presets

def _presets_file(ctx: Ctx, file_) -> Path:
    path = Path(file_) if file_ else ctx.cfg.presets.path()
    if not path.is_file():
        raise click.ClickException(f"presets file not found: {path}")
    return path


def _validate(ctx: Ctx, file_, engine):
    from mse_llm import presets as ps

    path = _presets_file(ctx, file_)
    _, _, snap, helps = _engine(ctx, engine or ctx.cfg.presets.engine)
    pf = ps.parse(path)
    return pf, ps.validate(pf, helps.get("llama-server"), snap), helps


_ICONS = {"error": "[red]✗[/]", "warn": "[yellow]⚠[/]", "info": "[cyan]ℹ[/]", "ok": "[green]✓[/]"}


def _print_validation(val, only_problems: bool = False) -> None:
    console.print(f"[bold]{escape(val.path.name)}[/] → help snapshot {val.snapshot or '(none)'}")
    for f in val.findings:
        if only_problems and f.level in ("info", "ok"):
            continue
        where = f"{escape(f.where):<20} " if f.where else ""
        line = f"[dim]:{f.line}[/] " if f.line else ""
        console.print(f" {_ICONS[f.level]} {where}{line}{escape(f.message)}")
    e, w = len(val.errors), len(val.warnings)
    tail = "  `mse-llm presets fix` shows the corrections." if val.fixable else ""
    console.print(f"Result: {e} error{'s' if e != 1 else ''}, {w} warning{'s' if w != 1 else ''}.{tail}")


def _router_port(ctx: Ctx, port, engine=None) -> tuple[int, str]:
    _, eng = ctx.cfg.engine(engine or ctx.cfg.presets.engine)
    p = port or ctx.cfg.presets.port or eng.port
    host = "localhost" if eng.client_host in ("0.0.0.0", "") else eng.client_host
    return p, f"http://{host}:{p}"


@main.group()
def presets() -> None:
    """Multi-model router from an INI file (§4.3): one endpoint, the model picked per request."""


@presets.command("list")
@click.option("--file", "file_", type=click.Path(dir_okay=False), default=None)
@click.option("--engine", default=None)
@json_option
@pass_ctx
def presets_list(ctx: Ctx, file_, engine) -> None:
    """Sections, model file found?, key args, variant ≈, router-owned keys."""
    from mse_llm import presets as ps

    pf, _, helps = _validate(ctx, file_, engine)
    rows = []
    for s in pf.models:
        d = ps.dump(pf, s.name, helps.get("llama-server"), ctx.cfg)
        found = bool(d.model_path) and ps._path_exists(d.model_path, Path.cwd())
        owned = [k for k in s.keys if k in ps.ROUTER_OWNED]
        rows.append({"name": s.name, "alias": s.keys.get("alias"), "model": d.model_path, "found": found,
                     "ctx": d.ctx, "variant": d.variant, "model_key": d.model_key, "default_serve": d.default_serve,
                     "matches_default": d.matches_default, "router_owned": owned})
    if ctx.json:
        click.echo(json.dumps({"file": str(pf.path), "globals": pf.globals, "presets": rows}, indent=2))
        return
    glob = " ".join(f"{k}={v}" for k, v in pf.globals.items())
    console.print(f"[bold]{escape(str(pf.path))}[/]   [*] {escape(glob)}")
    t = Table(box=None, pad_edge=False, header_style="bold")
    for c in ("PRESET", "FILE", "CTX", "VARIANT ≈", "CONFIG MODEL"):
        t.add_column(c, overflow="fold")
    for r in rows:
        match = {True: "[green]= default_serve[/]", False: "[yellow]≠ default_serve[/]", None: ""}[r["matches_default"]]
        t.add_row(escape(r["name"]), "[green]✓[/]" if r["found"] else "[red]✗ missing[/]", str(r["ctx"] or "-"),
                  escape(r["variant"]), f"{escape(r['model_key'] or '-')} {match}")
    console.print(t)
    owned = sorted({k for r in rows for k in r["router_owned"]})
    here = "; set here: " + ", ".join(owned) if owned else ""
    console.print(f"[dim]router-owned keys (ignored in the INI): host, port, api-key{escape(here)}[/]")


@presets.command("validate")
@click.option("--file", "file_", type=click.Path(dir_okay=False), default=None)
@click.option("--engine", default=None)
@json_option
@pass_ctx
def presets_validate(ctx: Ctx, file_, engine) -> None:
    """Every key vs the engine's --help, model files, section names; exit 1 on errors."""
    _, val, _ = _validate(ctx, file_, engine)
    if ctx.json:
        click.echo(json.dumps(val.as_dict(), indent=2))
    else:
        _print_validation(val)
    sys.exit(1 if val.errors else 0)


@presets.command("fix")
@click.option("--file", "file_", type=click.Path(dir_okay=False), default=None)
@click.option("--write", is_flag=True, help="Apply the fixes (the original is kept as FILE.bak).")
@click.option("--engine", default=None)
@pass_ctx
def presets_fix(ctx: Ctx, file_, write, engine) -> None:
    """Show the fixes as a diff; --write applies them. Comments are kept."""
    import shutil

    from mse_llm import presets as ps

    pf, val, _ = _validate(ctx, file_, engine)
    if not val.fixable:
        console.print("nothing to fix" + (f" ({len(val.errors)} error(s) need a manual change)" if val.errors else ""))
        return
    new = ps.fixed_text(pf, val)
    _plain(ps.diff(pf, new))
    if write:
        bak = pf.path.with_name(pf.path.name + ".bak")
        shutil.copy2(pf.path, bak)
        pf.path.write_text(new, encoding="utf-8")
        console.print(f"[green]✓[/] wrote {escape(str(pf.path))} (backup {escape(bak.name)})")
    else:
        console.print("[dim]--write applies this[/]")


@presets.command("dump")
@click.argument("name", required=False)
@click.option("--file", "file_", type=click.Path(dir_okay=False), default=None)
@click.option("--engine", default=None)
@json_option
@pass_ctx
def presets_dump(ctx: Ctx, name, file_, engine) -> None:
    """Merged [*] + section → the equivalent single-model llama-server command (bash run_PRESETS_dump)."""
    from mse_llm import presets as ps
    from mse_llm.runners import display_argv

    pf, _, helps = _validate(ctx, file_, engine)
    try:
        names = [pf.find(name).name] if name else [s.name for s in pf.models]
    except KeyError as e:
        raise click.ClickException(str(e.args[0])) from e
    dumps = [ps.dump(pf, n, helps.get("llama-server"), ctx.cfg) for n in names]
    if ctx.json:
        click.echo(json.dumps([{"name": d.name, "argv": ["llama-server", *d.flags], "variant": d.variant,
                                "model_key": d.model_key, "default_serve": d.default_serve,
                                "matches_default": d.matches_default, "skipped": d.skipped} for d in dumps], indent=2))
        return
    for d in dumps:
        console.print(f"[bold][{escape(d.name)}][/]")
        _plain("  llama-server " + display_argv(d.flags))
        m = ""
        if d.model_key:
            m = (f"   (matches models.{d.model_key}.default_serve)" if d.matches_default else
                 f"   (models.{d.model_key}.default_serve = {d.default_serve})" if d.default_serve else
                 f"   (models.{d.model_key})")
        _plain(f"  variant ≈ {d.variant}{m}")
        if d.skipped:
            console.print(f"  [dim]router-only keys not on the command line: {escape(', '.join(d.skipped))}[/]")


@presets.command("generate")
@click.option("--out", "out", type=click.Path(dir_okay=False), default=None, help="Write here (default: print).")
@click.option("--force", is_flag=True, help="Overwrite an existing --out file.")
@click.option("--engine", default=None)
@pass_ctx
def presets_generate(ctx: Ctx, out, force, engine) -> None:
    """Write an INI from [models.*] + each model's default_serve (one source of truth)."""
    from mse_llm import presets as ps

    text, warnings = ps.generate(ctx.cfg, engine)
    for w in warnings:
        click.echo(f"! {w}", err=True)
    if not out:
        click.echo(text, nl=False)
        return
    p = Path(out)
    if p.exists() and not force:
        raise click.ClickException(f"{p} exists (--force overwrites it)")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    _, _, snap, helps = _engine(ctx, engine)
    val = ps.validate(ps.parse(p), helps.get("llama-server"), snap)
    console.print(f"[green]✓[/] wrote {escape(str(p))}")
    _print_validation(val, only_problems=True)


def _resolve_loads(pf, names) -> list[str]:
    out = []
    for n in names:
        try:
            out.append(pf.find(n).name)
        except KeyError as e:
            raise click.ClickException(str(e.args[0])) from e
    return out


@presets.command("serve")
@click.option("--file", "file_", type=click.Path(dir_okay=False), default=None,
              help="Presets INI (default: [presets].file = llama_cpp_presets.ini).")
@click.option("--port", type=int, default=None)
@click.option("--models-max", type=int, default=None, help="Default: [presets].models_max.")
@click.option("--memory-margin", "memory_margin", type=int, default=None,
              help="MiB (--models-memory-margin). Default: [presets].memory_margin_mib.")
@click.option("--no-autoload", is_flag=True, help="Load models only via --load / `presets load`.")
@click.option("--load", "loads", multiple=True, metavar="NAME", help="Load this preset after start (repeatable).")
@click.option("--background", is_flag=True, help="Start, wait for the loads, return. `presets stop` ends it.")
@click.option("--foreground", is_flag=True, help="Stream the log; Ctrl+C stops the router and every model (default).")
@click.option("--output", type=OUTPUT, default=None, help="Foreground console output (default: [defaults].output).")
@click.option("--force", is_flag=True, help="Start even when `presets validate` finds errors.")
@_monitor_options
@click.option("--engine", default=None)
@click.option("--dry-run", is_flag=True)
@json_option
@pass_ctx
def presets_serve(ctx: Ctx, file_, port, models_max, memory_margin, no_autoload, loads, background, foreground, output,
                  force, no_monitor, no_guard, mem_cap, engine, dry_run) -> None:
    """llama-server --models-preset INI (bash run_PRESETS_serve)."""
    from mse_llm import fit as ff
    from mse_llm.runners import presets_serve_command

    if background and foreground:
        raise click.UsageError("--background or --foreground, not both")
    pc = ctx.cfg.presets
    engine = engine or pc.engine
    the_port, root = _router_port(ctx, port, engine)
    cmd = presets_serve_command(ctx.cfg, file_, engine, port=the_port,
                                models_max=models_max if models_max is not None else pc.models_max,
                                memory_margin_mib=memory_margin if memory_margin is not None else pc.memory_margin_mib)
    if no_autoload or not pc.autoload:
        cmd.argv.append("--no-models-autoload")
    path = Path(file_) if file_ else pc.path()
    pf = val = None
    if path.is_file():
        pf, val, _ = _validate(ctx, file_, engine)
    if ctx.json and dry_run:
        click.echo(json.dumps({**cmd.as_dict(), "validation": val.as_dict() if val else None}, indent=2))
        sys.exit(0 if cmd.ok and (val is None or not val.errors) else 1)
    if val is not None and (val.errors or val.warnings):
        _print_validation(val, only_problems=True)
    _print_commands(ctx, [cmd])
    _, eng, snap, _ = _engine(ctx, engine)
    names = [s.name for s in pf.models] if pf else []
    console.print(f"  engine     {escape(eng.bin_path().as_posix())} ({snap or '?'})")
    console.print(f"  models     {escape(', '.join(names) or '(none)')}")
    console.print(f"  endpoint   {root}/v1   (model = section name)")
    if not cmd.ok or pf is None:
        sys.exit(1)
    if val.errors and not force:
        raise click.ClickException("the presets file has errors (`presets fix`, or --force to try anyway)")
    load = _resolve_loads(pf, list(loads) or list(pc.load_on_start))
    if load:
        console.print(f"  load       {escape(', '.join(load))}")
    if dry_run:
        return
    _prestart_checks(the_port)
    margin = memory_margin if memory_margin is not None else pc.memory_margin_mib
    free, src = ff.free_vram_now()
    if margin and free is not None and free < margin:
        console.print(f"[yellow]![/] only {free} MiB VRAM free ({src}), less than the {margin} MiB memory margin")
    cap = mem_cap if mem_cap is not None else ctx.cfg.monitor.job_memory_cap_mib
    if background:
        asyncio_run(_router_background(ctx, cmd, eng, the_port, path, load, cap))
    else:
        mon = _MonOpts(not no_monitor and ctx.cfg.defaults.monitor, not no_guard, None, cap)
        asyncio_run(_router_foreground(ctx, cmd, eng, the_port, load, _filter(ctx, output), mon))


async def _do_loads(root: str, load: list[str]) -> None:
    from mse_llm import serve as sv

    for n in load:
        t0 = time.monotonic()
        console.print(f"  loading {escape(n)} …")
        try:
            await sv.load_model(root, n)
            console.print(f"  [green]✓[/] {escape(n)} loaded in {time.monotonic() - t0:.1f} s")
        except RuntimeError as e:
            console.print(f"  [red]✗ {escape(str(e))}[/]")


async def _router_foreground(ctx: Ctx, cmd, eng, port: int, load: list[str], filt, mon: _MonOpts) -> None:
    from mse_llm import serve as sv
    from mse_llm.runners import ServerHandle

    log = sv.servers_dir() / f"router-{port}.log"
    h = ServerHandle(cmd.argv, port=port, client_host=eng.client_host,
                     ready_timeout=ctx.cfg.defaults.server_ready_timeout,
                     log_path=log, sink=_printer(filt), memory_cap_mib=mon.mem_cap_mib)

    async def ready(ep) -> None:
        console.print(f"[green]✓[/] router ready after {ep.ready_s:.1f} s: [bold]{ep.base_url}[/]  pid {ep.pid}")
        await _do_loads(ep.root, load)

    await run_server(ctx, h, log, mon, ready, what="router", hint="Ctrl+C stops the router and all its models")


async def _router_background(ctx: Ctx, cmd, eng, port: int, ini: Path, load: list[str],
                             mem_cap_mib: int | None = None) -> None:
    import psutil

    from mse_llm import serve as sv
    from mse_llm.runners import ServerError, ServerHandle

    log = sv.servers_dir() / f"router-{port}.log"
    h = ServerHandle(cmd.argv, port=port, client_host=eng.client_host,
                     ready_timeout=ctx.cfg.defaults.server_ready_timeout,
                     log_path=log, detached=True, memory_cap_mib=mem_cap_mib)
    try:
        ep = await h.start()
    except ServerError as e:
        _fail(e, log)
    st = sv.write_state("router", port, {
        "pid": ep.pid, "create_time": psutil.Process(ep.pid).create_time(), "port": port, "endpoint": ep.base_url,
        "argv": cmd.argv, "file": str(ini), "log": str(log), "started": sv.timestamp()})
    console.print(f"[green]✓[/] router running in the background: [bold]{ep.base_url}[/]  pid {ep.pid}")
    await _do_loads(ep.root, load)
    console.print(f"[dim]state {escape(str(st))}, log {escape(str(log))}; `mse-llm presets stop` ends it[/]")


@presets.command("stop")
@click.option("--port", type=int, default=None, help="Default: every recorded background router.")
@pass_ctx
def presets_stop(ctx: Ctx, port) -> None:
    """Stop a --background router (its whole process tree) and check the port is free."""
    from mse_llm import proc as pr
    from mse_llm import serve as sv

    states = sv.read_state("router", port)
    if not states:
        p = port or _router_port(ctx, None)[0]
        msg = "no background router recorded" + (f" for port {port}" if port else "")
        if pr.port_in_use(p):
            msg += f"; {pr.describe_port(p)} (`mse-llm hw processes --stop-ollama` ends Ollama)"
        console.print(msg)
        return
    for st in states:
        f = Path(st["_file"])
        if not sv.state_alive(st):
            console.print(f"router pid {st['pid']} (port {st['port']}) is not running any more")
            f.unlink(missing_ok=True)
            continue
        rep = pr.stop_tree(int(st["pid"]), signal_first=False, port=int(st["port"]))
        _print_stop(rep)
        if rep.clean:
            f.unlink(missing_ok=True)
        else:
            sys.exit(1)


def _http(coro):
    import httpx

    try:
        return asyncio.run(coro)
    except httpx.HTTPError as e:
        raise click.ClickException(f"router not reachable: {e}") from None
    except RuntimeError as e:
        raise click.ClickException(str(e)) from None


@presets.command("status")
@click.option("--port", type=int, default=None)
@json_option
@pass_ctx
def presets_status(ctx: Ctx, port) -> None:
    """GET /models: loaded / unloaded, memory min / max / effective per model."""
    from mse_llm import serve as sv

    _, root = _router_port(ctx, port)
    models = _http(sv.router_models(root))
    if ctx.json:
        click.echo(json.dumps(models, indent=2))
        return
    t = Table(box=None, pad_edge=False, header_style="bold", title=f"{root}/models", title_justify="left")
    for c, j in (("MODEL", "left"), ("STATUS", "left"), ("GPU MiB eff", "right"), ("min", "right"),
                 ("max", "right"), ("n_ctx", "right")):
        t.add_column(c, justify=j, overflow="fold")
    for m in models:
        mem = m.get("memory") or {}
        eff = mem.get("effective") or {}
        st = sv._status(m)
        style = {"loaded": "green", "loading": "yellow"}.get(st, "")
        t.add_row(escape(str(m.get("id"))), f"[{style}]{escape(st)}[/]" if style else escape(st),
                  str(sv.gpu_mib(eff) or "-"), str(sv.gpu_mib(mem.get("min")) or "-"),
                  str(sv.gpu_mib(mem.get("max")) or "-"), str(eff.get("n_ctx") or "-"))
    console.print(t)


@presets.command("load")
@click.argument("name")
@click.option("--port", type=int, default=None)
@click.option("--no-wait", is_flag=True)
@click.option("--timeout", type=float, default=600, show_default=True)
@pass_ctx
def presets_load(ctx: Ctx, name, port, no_wait, timeout) -> None:
    """POST /models/load and wait until it is loaded."""
    from mse_llm import serve as sv

    _, root = _router_port(ctx, port)
    t0 = time.monotonic()
    st = _http(sv.load_model(root, name, timeout=timeout, wait=not no_wait,
                             progress=lambda s: console.print(f"  [dim]{escape(name)}: {escape(s)}[/]")))
    console.print(f"[green]✓[/] {escape(name)}: {st} ({time.monotonic() - t0:.1f} s)")


@presets.command("unload")
@click.argument("name")
@click.option("--port", type=int, default=None)
@pass_ctx
def presets_unload(ctx: Ctx, name, port) -> None:
    """POST /models/unload."""
    from mse_llm import serve as sv

    _, root = _router_port(ctx, port)
    _http(sv.unload_model(root, name))
    console.print(f"[green]✓[/] {escape(name)}: unload requested")


@presets.command("watch")
@click.option("--port", type=int, default=None)
@pass_ctx
def presets_watch(ctx: Ctx, port) -> None:
    """Follow /models/sse: loads, unloads, memory measurements (Ctrl+C ends)."""
    from mse_llm import serve as sv
    from mse_llm.health import sse_events

    _, root = _router_port(ctx, port)

    async def go() -> None:
        async for ev in sse_events(root + "/models/sse"):
            data = ev.get("data")
            if ev.get("event") == "model_memory" and isinstance(data, dict):
                extra = (f"min {sv.gpu_mib(data.get('min'))} max {sv.gpu_mib(data.get('max'))} "
                         f"effective {sv.gpu_mib(data.get('effective'))} MiB")
            elif isinstance(data, dict):
                extra = " ".join(f"{k}={v}" for k, v in data.items() if k != "info")
            else:
                extra = ""
            what = ev.get("event", ev.get("raw", "?"))
            _plain(f"{time.strftime('%H:%M:%S')}  {ev.get('model', '?')}  {what}  {extra}")

    try:
        _http(go())
    except KeyboardInterrupt:
        pass
