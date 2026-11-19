"""mse-llm command line (click). Commands stay thin; logic lives in the modules."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from mse_llm import __version__
from mse_llm.checks import ICON, Check, verdict


def _fix_streams() -> None:
    """Redirected output (pipes, files, CI) gets UTF-8; a legacy console keeps its code page but replaces
    characters it can't show instead of crashing."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_fix_streams()
console = Console(highlight=False, width=None if sys.stdout.isatty() else 160)


class Ctx:
    def __init__(self, config: str | None, as_json: bool, verbose: bool) -> None:
        self.config_path = config
        self.json = as_json
        self.verbose = verbose
        self._cfg = None

    @property
    def cfg(self):
        if self._cfg is None:
            from mse_llm.config import load_config

            try:
                self._cfg = load_config(self.config_path)
            except Exception as e:
                raise click.ClickException(f"config: {e}") from e
        return self._cfg


pass_ctx = click.make_pass_decorator(Ctx)


def _set_json(cctx: click.Context, _param, value: bool) -> bool:
    if value:
        cctx.find_object(Ctx).json = True
    return value


# `--json` works both globally (mse-llm --json hw) and per command (mse-llm hw --json).
json_option = click.option("--json", "_json", is_flag=True, expose_value=False, callback=_set_json,
                           help="Machine-readable output.")


class SectionedGroup(click.Group):
    """Root group whose --help lists the commands by area. Same names, grouped only in the help text;
    a command missing from SECTIONS shows under "Other" so nothing new gets hidden."""

    SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Inference", ("serve", "presets", "fit", "predict")),
        ("Benchmark", ("bench", "benchy", "sweep", "results", "report", "eval")),
        ("System", ("doctor", "hw", "monitor", "notify", "help")),
        ("Config", ("models", "variants")),
        ("Build", ("build",)),
    )

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        cmds = {n: c for n in self.list_commands(ctx)
                if (c := self.get_command(ctx, n)) is not None and not c.hidden}
        if not cmds:
            return
        width = max(len(n) for n in cmds)       # one description column across all sections
        limit = formatter.width - 6 - width
        sections = [(title, [n for n in names if n in cmds]) for title, names in self.SECTIONS]
        listed = {n for _, names in sections for n in names}
        sections.append(("Other", [n for n in cmds if n not in listed]))
        for title, names in sections:
            if names:
                with formatter.section(title):
                    formatter.write_dl([(n.ljust(width), cmds[n].get_short_help_str(limit)) for n in names])


@click.group(cls=SectionedGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", "config", type=click.Path(dir_okay=False), envvar="MSE_LLM_CONFIG",
              help="Config TOML (default: config/config.toml, else config/config.example.toml).")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@click.option("-v", "--verbose", is_flag=True)
@click.option("--no-notify", is_flag=True, help="No voice message at the end of a run (MSE_LLM_NO_NOTIFY=1).")
@click.option("--hw-fixture", type=click.Path(exists=True, dir_okay=False),
              help="Use a HardwareInfo JSON instead of probing (tests/CI). Same as MSE_LLM_TEST_HARDWARE.")
@click.version_option(__version__, prog_name="mse-llm")
@click.pass_context
def main(ctx: click.Context, config, as_json, verbose, no_notify, hw_fixture) -> None:
    """Benchmark, tune and serve local LLMs (llama.cpp)."""
    if no_notify:
        os.environ["MSE_LLM_NO_NOTIFY"] = "1"
    if hw_fixture:
        os.environ["MSE_LLM_TEST_HARDWARE"] = str(Path(hw_fixture).resolve())
    ctx.obj = Ctx(config, as_json, verbose)


# ---------------------------------------------------------------- output helpers

def print_checks(checks: list[Check], title: str) -> None:
    t = Table(title=title, show_lines=False, expand=False)
    t.add_column("section", style="dim")
    t.add_column("check")
    t.add_column("", no_wrap=True)
    t.add_column("detail", overflow="fold")
    prev = None
    for c in checks:
        detail = escape(c.detail)
        if c.hint and c.status in ("warn", "fail"):
            detail += f"\n[dim]-> {escape(c.hint)}[/]"
        if prev is not None and c.section != prev:
            t.add_section()                       # a line between sections
        t.add_row(c.section if c.section != prev else "", c.name, ICON[c.status], detail)
        prev = c.section
    console.print(t)
    counts = {s: sum(c.status == s for c in checks) for s in ("pass", "warn", "fail")}
    console.print(f"{counts['pass']} ok, {counts['warn']} warnings, {counts['fail']} failures. "
                  f"Verdict: [bold]{verdict(checks)}[/]")


def gib(b: int | None) -> str:
    return f"{b / 2**30:.1f} GiB" if b is not None else "?"


# ---------------------------------------------------------------- doctor

@main.group(invoke_without_command=True)
@json_option
@pass_ctx
@click.pass_context
def doctor(cctx: click.Context, ctx: Ctx) -> None:
    """Check environment, toolchain, engine binaries, hardware and the NVIDIA driver setting."""
    if cctx.invoked_subcommand:
        return
    from mse_llm.doctor import run_doctor

    checks = run_doctor(ctx.cfg)
    if ctx.json:
        click.echo(json.dumps({"verdict": verdict(checks), "checks": [c.as_dict() for c in checks]}, indent=2))
    else:
        print_checks(checks, "mse-llm doctor")
    sys.exit(1 if any(c.status == "fail" for c in checks) else 0)


@doctor.command("flags")
@json_option
@click.option("--engine", default=None, help="Engine name (default: cuda_flash).")
@click.option("--capture", is_flag=True, help="Store the current build's --help as a snapshot.")
@click.option("--against", default=None, help="Snapshot id to diff against (default: engine's, else newest).")
@click.option("--script", type=click.Path(dir_okay=False), default=None,
              help="Bash script to audit (default: llama_cpp_benchmark.sh if present).")
@click.option("--ini", type=click.Path(dir_okay=False), default=None,
              help="Presets INI to audit (default: llama_cpp_presets.ini if present).")
@pass_ctx
def doctor_flags(ctx: Ctx, engine, capture, against, script, ini) -> None:
    """Diff llama.cpp --help against a stored snapshot and audit the flags the bash script/INI use."""
    from mse_llm import helpsnap as hs
    from mse_llm.util import exe, project_root

    name, eng = ctx.cfg.engine(engine)
    bdir = eng.bin_path()
    have_bins = all(exe(bdir / t).is_file() for t in hs.TOOLS)
    report: dict = {"engine": name, "bin_dir": str(bdir), "binaries": have_bins}

    current: dict[str, hs.HelpInfo] = {}
    if have_bins:
        for t in hs.TOOLS:
            try:
                current[t] = hs.parse_help(hs.read_help(exe(bdir / t)))
            except RuntimeError as e:
                raise click.ClickException(str(e)) from e
        if capture:
            sid, dest = hs.capture(bdir)
            report["captured"] = str(dest)
    elif capture:
        raise click.ClickException(f"no binaries in {bdir}; build first: mse-llm build run")

    snaps = hs.list_snapshots()
    want = against or eng.help_snapshot
    snap_path = next((s for s in snaps if s.name == want), None) if want else (snaps[-1] if snaps else None)
    if want and snap_path is None:
        raise click.ClickException(f"snapshot {want!r} not found in {hs.snapshots_dir()}")
    stored = hs.load_snapshot(snap_path) if snap_path else {}
    report["snapshot"] = snap_path.name if snap_path else None

    diffs = {}
    if current and stored:
        for t in hs.TOOLS:
            if t in current and t in stored:
                d = hs.diff_help(stored[t], current[t])
                diffs[t] = {"added": d.added, "removed": d.removed, "changed": d.changed}
    report["diff"] = diffs

    helps = current or stored
    findings: list[hs.Finding] = []
    root = project_root()
    script_p = Path(script) if script else root / "llama_cpp_benchmark.sh"
    ini_p = Path(ini) if ini else root / "llama_cpp_presets.ini"
    if helps:
        if script_p.is_file():
            findings += hs.audit_script(script_p, helps)
        if ini_p.is_file() and "llama-server" in helps:
            findings += hs.audit_ini(ini_p, helps["llama-server"])
    report["findings"] = [f.__dict__ for f in findings]

    if ctx.json:
        click.echo(json.dumps(report, indent=2))
    else:
        console.print(f"engine [bold]{name}[/]: {bdir}")
        if not have_bins:
            console.print("[yellow]WARN[/] binaries not built yet (mse-llm build run): "
                          "using the stored snapshot only" if stored else
                          "[yellow]WARN[/] no binaries and no stored snapshot: nothing to check yet. "
                          "Build llama.cpp with `mse-llm build run` (it captures the help snapshot).")
        if snap_path:
            console.print(f"snapshot: {snap_path}")
        if "captured" in report:
            console.print(f"[green]captured[/] current help into {report['captured']}")
        for t, d in diffs.items():
            if not any(d.values()):
                console.print(f"{t}: no changes vs snapshot")
                continue
            console.print(f"[bold]{t}[/] vs snapshot {snap_path.name}:")
            for k, color in (("added", "green"), ("removed", "red"), ("changed", "yellow")):
                for x in d[k]:
                    console.print(f"  [{color}]{k:8}[/] {escape(x)}")
        if helps:
            if findings:
                console.print(f"[bold]audit[/]: {len(findings)} finding(s)")
                for f in findings:
                    console.print(f"  [yellow]{f.kind:13}[/] {f.source:32} {escape(f.message)}")
            else:
                console.print("[green]audit[/]: script and INI flags all accepted")
    if findings:
        sys.exit(1)


# ---------------------------------------------------------------- help

@main.command("help")
@click.argument("tool", type=click.Choice(["server", "bench", "benchy", "eval"]))
@click.option("--engine", default=None)
@pass_ctx
def help_cmd(ctx: Ctx, tool, engine) -> None:
    """Show --help of llama-server / llama-bench / llama-benchy / inspect eval (bash run_HELP_*)."""
    from mse_llm import helpsnap as hs
    from mse_llm.util import exe, run

    if tool in ("server", "bench"):
        name, eng = ctx.cfg.engine(engine)
        binary = exe(eng.bin_path() / f"llama-{tool}")
        if binary.is_file():
            r = run([str(binary), "--help"], timeout=30)
            click.echo(r.stdout + r.stderr, nl=False)
            sys.exit(r.returncode or 0)
        snaps = hs.list_snapshots()
        snap = next((s for s in snaps if s.name == eng.help_snapshot), snaps[-1] if snaps else None)
        if snap and (snap / f"llama-{tool}.txt").is_file():
            click.echo(f"# {binary} not built; showing snapshot {snap.name}", err=True)
            click.echo((snap / f"llama-{tool}.txt").read_text(encoding="utf-8"), nl=False)
            return
        raise click.ClickException(f"{binary} not found and no help snapshot; build llama.cpp first: mse-llm build run")
    if tool == "benchy":
        argv = ["uvx", ctx.cfg.defaults.benchy.version, "--help"]
    else:
        argv = ["uvx", "--from", "inspect-ai", "inspect", "eval", "--help"]
    r = run(argv, timeout=300)
    if r.error:
        raise click.ClickException(f"{' '.join(argv)}: {r.error}")
    click.echo(r.stdout + r.stderr, nl=False)
    sys.exit(r.returncode or 0)


# ---------------------------------------------------------------- hw

def _detect(ctx: Ctx):
    from mse_llm import hw
    from mse_llm.doctor import disk_paths

    return hw.detect(disk_paths=disk_paths(ctx.cfg))


@main.group(invoke_without_command=True)
@json_option
@pass_ctx
@click.pass_context
def hw(cctx: click.Context, ctx: Ctx) -> None:
    """PC capability detection: CPU, RAM, GPUs, PCIe, power, disk, checks and verdict."""
    if cctx.invoked_subcommand:
        return
    from mse_llm import hw as hwmod
    from mse_llm.hw.cpu import isa_summary

    info = _detect(ctx)
    checks = hwmod.checks(info, ctx.cfg.hw.checks)
    ident = hwmod.identity(info)
    if ctx.json:
        out = info.model_dump(mode="json")
        out["identity"] = {"hash": ident["hash"], "soft_hash": ident["soft_hash"]}
        out["checks"] = [c.as_dict() for c in checks]
        out["verdict"] = verdict(checks)
        click.echo(json.dumps(out, indent=2))
        return

    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_column(style="bold")
    t.add_column(overflow="fold")
    h = info.host
    t.add_row("host", f"{h.hostname} - {h.os} {h.os_release} ({h.os_version}), {h.arch}"
              + (f"  [yellow](fixture {info.fixture})[/]" if info.fixture else ""))
    c = info.cpu
    hyb = "unknown" if c.hybrid is None else ("hybrid" if c.hybrid else "not hybrid")
    t.add_row("cpu", f"{c.model} - {c.physical_cores}C/{c.logical_cpus}T, {hyb}, {isa_summary(c.isa)}")
    if c.hybrid and c.p_core_cpus:
        t.add_row("", f"P-core CPUs {c.p_core_cpus}; E-core CPUs {c.e_core_cpus}")
    if c.l3_groups and len(c.l3_groups) > 1:
        groups = ", ".join(f"L3#{i}: {g[0]}-{g[-1]}" if g == list(range(g[0], g[-1] + 1)) else f"L3#{i}: {g}"
                           for i, g in enumerate(c.l3_groups))
        t.add_row("", f"{len(c.l3_groups)} L3 groups (CCDs): {groups}")
    m = info.memory
    t.add_row("ram", f"{gib(m.total_bytes)} total, {gib(m.available_bytes)} available, pressure {m.pressure}")
    if m.modules:
        mods = "; ".join(f"{x.bank or x.locator}: {x.capacity_gb:g} GB {x.type or ''} "
                         f"{x.configured_mts}/{x.rated_mts} MT/s" for x in m.modules)
        t.add_row("", mods)
    if m.layout:
        t.add_row("", m.layout)
    if m.commit_limit_bytes:
        t.add_row("", f"commit {gib(m.commit_total_bytes)} of {gib(m.commit_limit_bytes)}, "
                  f"pagefile/swap {gib(m.swap_used_bytes)} of {gib(m.swap_total_bytes)} used")
    for g in info.gpus:
        if g.vendor == "nvidia":
            pcie = (f"PCIe {g.pcie_gen_max}.0 x{g.pcie_width_max} (now gen{g.pcie_gen_current} "
                    f"x{g.pcie_width_current})") if g.pcie_gen_max else "PCIe ?"
            line = (f"[{g.index}] {g.name} - {g.vram_total_mib} MiB ({g.vram_free_mib} free), cc "
                    f"{g.compute_capability}, {pcie}, {g.mem_bus_width_bits}-bit ~{g.bandwidth_gbs_est} GB/s, "
                    f"power limit {g.power_limit_w} W, {g.temperature_c} C, driver {g.driver}")
        else:
            line = f"{g.name} ({g.vendor}{', integrated' if g.integrated else ''}) - {g.vram_total_mib} MiB"
        t.add_row("gpu", line)
    p = info.power
    pw = "AC" if p.ac_online else "battery" if p.ac_online is False else "unknown"
    t.add_row("power", f"{pw}" + (f", battery {p.battery_percent} %" if p.has_battery else "")
              + (f", plan {p.power_plan}" if p.power_plan else ""))
    for d in info.disks:
        t.add_row("disk", f"{d.label}: {gib(d.free_bytes)} free of {gib(d.total_bytes)} ({d.path})")
    if info.nvidia_gpus():
        dr = info.driver
        t.add_row("driver", f"CUDA sysmem fallback: {dr.sysmem_fallback or dr.error}")
    t.add_row("identity", f"{ident['hash']} (soft {ident['soft_hash']})")
    console.print(t)
    console.print()
    print_checks(checks, "hardware checks")


@hw.command("identity")
@json_option
@click.option("--profile", default=None, help="Build profile to compare with (default: all).")
@pass_ctx
def hw_identity(ctx: Ctx, profile) -> None:
    """Hard/soft hardware identity and its hash; compare with the build profiles' [target]."""
    from mse_llm import hw as hwmod
    from mse_llm.buildprofiles import list_profiles, load_profile
    from mse_llm.doctor import target_checks

    info = _detect(ctx)
    ident = hwmod.identity(info)
    profs = [load_profile(profile)] if profile else list_profiles()
    tchecks = [c for c in target_checks(info) if not profile or c.name.endswith(profile)]
    if ctx.json:
        click.echo(json.dumps({**ident, "targets": [c.as_dict() for c in tchecks]}, indent=2))
        return
    console.print(f"hard identity hash: [bold]{ident['hash']}[/]")
    console.print_json(json.dumps(ident["hard"]))
    console.print(f"soft identity hash: [bold]{ident['soft_hash']}[/]")
    console.print_json(json.dumps(ident["soft"]))
    if profs:
        print_checks(tchecks, "build profile targets")


@hw.command("processes")
@click.option("--stop-ollama", "stop", is_flag=True,
              help="Stop Ollama (server, tray app and model runners), like taskkill /IM ollama.exe /F.")
@click.option("-y", "--yes", is_flag=True, help="Don't ask before stopping.")
@json_option
@pass_ctx
def hw_processes(ctx: Ctx, stop: bool, yes: bool) -> None:
    """Other inference engines and GPU compute processes that would skew a bench."""
    from mse_llm.hw.processes import detect_processes, stop_processes

    procs = detect_processes()
    if stop:
        engines = [p for p in procs if p.kind == "engine" and p.name.lower().startswith("ollama")]
        if not engines:
            if ctx.json:
                click.echo("[]")
            else:
                console.print("Ollama is not running")
            return
        if not yes:
            if ctx.json:
                raise click.UsageError("--stop-ollama with --json needs --yes")
            for p in engines:
                console.print(f"  {p.pid:>7}  {p.name}  {p.cmdline or ''}")
            click.confirm(f"Stop {len(engines)} Ollama process(es)?", abort=True)
        results = stop_processes(engines)
        if ctx.json:
            click.echo(json.dumps([r.__dict__ for r in results], indent=2))
        else:
            for r in results:
                console.print(f"  {r.pid:>7}  {r.name:<24} {r.result}")
        if any(r.result.startswith("failed") or r.result == "denied" for r in results):
            raise SystemExit(1)
        return
    if ctx.json:
        click.echo(json.dumps([p.__dict__ for p in procs], indent=2))
        return
    if not procs:
        console.print("no other engines or GPU compute processes")
        return
    t = Table("kind", "pid", "name", "GPU MiB", "command line")
    for p in procs:
        t.add_row(p.kind, str(p.pid), p.name, str(p.gpu_mib or "-"), p.cmdline or "")
    console.print(t)


# ---------------------------------------------------------------- build

class _BuildUI:
    def line(self, s: str) -> None:
        console.print(s, markup=False, highlight=False, soft_wrap=True)

    def out(self, s: str) -> None:
        console.print("    " + s, markup=False, highlight=False, style="dim", soft_wrap=True)

    def ask_step(self, st) -> str:
        return click.prompt(f"Run step {st.n} ({st.title})? [y]es / [s]kip / [q]uit", default="y",
                            type=click.Choice(["y", "s", "q"]), show_choices=False)


def _target(spec: str | None, without=(), ref=None, keep_previous=False):
    from mse_llm import build as b
    from mse_llm import hw

    info = hw.detect()
    try:
        if not spec:
            spec = b.pick_target(info)
            click.echo(f"target: {spec} (matches this PC)", err=True)
        return b.resolve_target(spec, without, ref, keep_previous), info
    except (b.BuildError, FileNotFoundError) as e:
        raise click.ClickException(str(e)) from e


def _plain(text: str) -> None:
    console.print(text, markup=False, highlight=False, soft_wrap=True)


@main.group()
def build() -> None:
    """Build llama.cpp from config/builds/*.toml (clone, configure, compile, verify, manifest)."""


@build.command("list")
@json_option
@pass_ctx
def build_list(ctx: Ctx) -> None:
    """Targets from config/builds (+ variants) and what is built."""
    from mse_llm import build as b

    rows = []
    for spec in b.all_targets():
        name, _, rest = spec.partition(" --without ")
        t = b.resolve_target(name, (rest,) if rest else ())
        m = b.read_manifest(t.bin_dir)
        rows.append({"target": spec, "build_dir": str(t.build_dir), "engine": t.engine_name,
                     "built": m.get("describe") if m else None, "built_at": m.get("built_at") if m else None})
    # other builds on disk, e.g. --ref <x> --keep-previous (build-<ref>)
    listed = {os.path.normcase(r["build_dir"]) for r in rows}
    for src in {Path(r["build_dir"]).parent for r in rows}:
        for bin_dir, m in b.built_dirs(src):
            bdir = bin_dir.parent.parent
            if os.path.normcase(str(bdir)) not in listed:
                rows.append({"target": m.get("target", "?"), "build_dir": str(bdir), "engine": m.get("engine"),
                             "built": m.get("describe"), "built_at": m.get("built_at")})
    if ctx.json:
        click.echo(json.dumps(rows, indent=2))
        return
    tb = Table("target", "build dir", "engine", "built")
    for r in rows:
        tb.add_row(r["target"], r["build_dir"], r["engine"] or "-",
                   f"{r['built']} ({r['built_at']})" if r["built"] else "[dim]not built[/]")
    console.print(tb)


def _run_options(f):
    for opt in reversed([
        click.argument("target", required=False),
        click.option("--without", multiple=True, help="Leave out a default variant (comparison build)."),
        click.option("--ref", default=None, help="llama.cpp branch, tag (b12468, v1.0.0) or commit."),
        click.option("--keep-previous", is_flag=True, help="With --ref: build into build-<ref>, keep build/."),
        click.option("--pull", is_flag=True, help="git fetch --tags (+ fast-forward a branch) before building."),
        click.option("--clean", is_flag=True, help="cmake --fresh + --clean-first (full rebuild)."),
        click.option("--allow-dirty", is_flag=True, help="Build even with uncommitted changes in the checkout."),
    ]):
        f = opt(f)
    return f


def _plan(ctx: Ctx, target, without, ref, keep_previous, pull, clean, allow_dirty, only=None, from_=None,
          keep_steps=None, what="build run"):
    from mse_llm import build as b

    t, info = _target(target, tuple(without), ref, keep_previous)
    opts = b.RunOpts(pull=pull, clean=clean, allow_dirty=allow_dirty, only=only, from_=from_)
    try:
        steps = b.plan_steps(t, opts)
    except b.BuildError as e:
        raise click.ClickException(str(e)) from e
    if keep_steps is not None:
        steps = [s for s in steps if s.id in keep_steps]
    checks = b.plan_checks(t, info, toolchain=not ctx.json)
    return t, info, opts, steps, checks, b.render_plan(t, steps, checks, what)


@build.command("plan")
@_run_options
@json_option
@pass_ctx
def build_plan(ctx: Ctx, target, without, ref, keep_previous, pull, clean, allow_dirty) -> None:
    """Preview every step `build run` would execute. Nothing runs."""
    from mse_llm import build as b

    t, _info, _o, steps, checks, text = _plan(ctx, target, without, ref, keep_previous, pull, clean, allow_dirty)
    if ctx.json:
        click.echo(json.dumps(b.plan_json(t, steps, checks), indent=2))
    else:
        _plain(text)


@build.command("run")
@_run_options
@click.option("--only", "only", default=None, help="Run just this step (number or id, e.g. compile).")
@click.option("--from", "from_", default=None, help="Resume from this step (number or id, e.g. configure).")
@click.option("--step", "step_mode", is_flag=True, help="Ask before every step.")
@click.option("-y", "--yes", is_flag=True, help="Don't ask before step 1.")
@click.option("--dry-run", is_flag=True, help="Same as `build plan`: show the steps, run nothing.")
@click.option("-q", "--quiet", is_flag=True, help="Don't stream tool output (full log in builds/logs/).")
@json_option
@pass_ctx
def build_run(ctx: Ctx, target, without, ref, keep_previous, pull, clean, allow_dirty, only, from_, step_mode, yes,
              dry_run, quiet) -> None:
    """Show the plan, ask, then clone / configure / compile / verify llama.cpp."""
    _do_run(ctx, target, without, ref, keep_previous, pull, clean, allow_dirty, only, from_, step_mode, yes,
            dry_run, quiet)


def _do_run(ctx, target, without, ref, keep_previous, pull, clean, allow_dirty, only, from_, step_mode, yes,
            dry_run, quiet, keep_steps=None, what="build run") -> None:
    from mse_llm import build as b

    t, info, opts, steps, checks, text = _plan(ctx, target, without, ref, keep_previous, pull, clean, allow_dirty,
                                               only, from_, keep_steps, what)
    if dry_run:
        if ctx.json:
            click.echo(json.dumps(b.plan_json(t, steps, checks), indent=2))
        else:
            _plain(text)
        return
    _plain(text)
    blocked = [s for s in steps if s.status == "BLOCK"]
    if blocked:
        raise click.ClickException(f"step {blocked[0].n} ({blocked[0].title}) cannot run: {blocked[0].why}")
    if any(c.status == "fail" for c in checks):
        raise click.ClickException("a check failed (see Checks above)")
    if not any(s.status == "RUN" for s in steps):
        console.print("nothing to do")
        return
    if not yes and not step_mode:
        click.confirm("Proceed?", abort=True)
    opts.quiet, opts.step = quiet, step_mode
    try:
        state = b.execute(t, steps, opts, _BuildUI(), text, info)
    except b.BuildError as e:
        raise click.ClickException(str(e)) from e
    except KeyboardInterrupt:
        raise click.ClickException("interrupted; the running tool was stopped") from None
    if "manifest" in state.durations:
        m = b.read_manifest(t.bin_dir) or {}
        console.print(f"[green]Built[/] {m.get('describe')} -> {t.bin_dir} "
                      f"(this run {b._fmt_dur(m.get('duration_s', 0))}{_steps_note(m)})")


def _steps_note(m: dict) -> str:
    """Only for runs limited with --from / --only: their time doesn't cover the whole build."""
    if m.get("only_step"):
        return f", only step {m['only_step']}"
    if m.get("resumed_from"):
        return f", resumed from step {m['resumed_from']}"
    return ""


@build.command("clone")
@click.argument("target", required=False)
@click.option("--vcpkg-only", is_flag=True, help="Only vcpkg (steps 3-5).")
@click.option("--no-vcpkg", is_flag=True, help="Only llama.cpp (steps 1-2).")
@click.option("--ref", default=None, help="llama.cpp branch, tag or commit to check out.")
@click.option("-y", "--yes", is_flag=True)
@click.option("--dry-run", is_flag=True)
@json_option
@pass_ctx
def build_clone(ctx: Ctx, target, vcpkg_only, no_vcpkg, ref, yes, dry_run) -> None:
    """Only prepare sources: clone llama.cpp + vcpkg if missing, install the vcpkg libraries."""
    from mse_llm import build as b

    if vcpkg_only and no_vcpkg:
        raise click.UsageError("--vcpkg-only and --no-vcpkg exclude each other")
    keep = b.CLONE_STEPS[2:] if vcpkg_only else b.CLONE_STEPS[:2] if no_vcpkg else b.CLONE_STEPS
    _do_run(ctx, target, (), ref, False, False, False, False, None, None, False, yes, dry_run, False,
            keep_steps=keep, what="build clone")


@build.command("clean")
@click.option("--logs", "kind", flag_value="logs", help="Delete builds/logs.")
@click.option("--builds", "kind", flag_value="builds",
              help="Delete the build dirs (build/, build-*/); keep the llama.cpp and vcpkg checkouts.")
@click.option("--all", "kind", flag_value="all", help="Delete the whole builds folder (checkouts, vcpkg, logs).")
@click.option("--dir", "dirs", multiple=True, metavar="NAME",
              help="Delete one build dir, e.g. build-a49db739f (name or path; repeatable).")
@click.option("-y", "--yes", is_flag=True, help="Don't ask.")
@click.option("--dry-run", is_flag=True, help="Only show what would be deleted.")
def build_clean(kind, dirs, yes, dry_run) -> None:
    """Delete build logs, build dirs or everything under builds/ (shows sizes and asks first)."""
    from mse_llm import build as b

    if dirs and kind:
        raise click.UsageError(f"--dir can't be combined with --{kind}")
    if dirs:
        kind = "dir"
    if not kind:
        raise click.UsageError("choose what to delete: --logs, --builds, --all or --dir NAME")
    roots = b.all_builds_roots()
    try:
        items = b.clean_items(kind, roots, tuple(dirs))
    except b.BuildError as e:
        raise click.ClickException(str(e)) from e
    if not items:
        console.print("nothing to delete")
        return
    tb = Table("delete", "what", "size")
    for it in items:
        tb.add_row(str(it.path), it.what, b.fmt_size(it.size))
    console.print(tb)
    total = sum(it.size for it in items)
    _plain(f"total {b.fmt_size(total)}" + ("" if kind != "all" else
           "   (the next `build run` clones again and rebuilds vcpkg's curl/openssl)"))
    if dry_run:
        return
    busy = b.busy_processes([it.path for it in items])
    if busy:
        raise click.ClickException("in use by running processes (a build, server or bench); wait for them "
                                   "to finish or stop them first:\n  " + "\n  ".join(busy))
    if not yes:
        click.confirm("Delete these permanently?", abort=True)
    for it in items:
        try:
            b.remove_tree(it.path, roots)
        except (OSError, b.BuildError) as e:
            raise click.ClickException(f"could not delete {it.path}: {e}") from e
        _plain(f"deleted {it.path}")


@build.command("show")
@click.argument("target", required=False)
@click.option("--without", multiple=True)
@click.option("--explain/--no-explain", default=True, help="Show the meaning column.")
@json_option
@pass_ctx
def build_show(ctx: Ctx, target, without, explain) -> None:
    """Merged profile: every CMake flag with where it comes from and what it means."""
    from dataclasses import asdict

    from mse_llm import build as b

    t, info = _target(target, tuple(without))
    if ctx.json:
        click.echo(json.dumps({**b.plan_json(t, b.plan_steps(t), []),
                               "auto": [asdict(f) for f in b.auto_flags(t, info)]}, indent=2))
        return
    _plain(b.render_show(t, info, explain))


@build.command("info")
@click.argument("target", required=False)
@click.option("--without", multiple=True)
@click.option("--ref", default=None)
@click.option("--keep-previous", is_flag=True)
@json_option
@pass_ctx
def build_info(ctx: Ctx, target, without, ref, keep_previous) -> None:
    """Show build-manifest.json: commit, flags, toolchain versions, binary hashes."""
    from mse_llm import build as b

    t, _ = _target(target, tuple(without), ref, keep_previous)
    m = b.read_manifest(t.bin_dir)
    if m is None:
        raise click.ClickException(f"no build-manifest.json in {t.bin_dir}; run `mse-llm build run {t.label}`")
    if ctx.json:
        click.echo(json.dumps(m, indent=2))
        return
    tc = m.get("toolchain") or {}
    _plain(f"{m['target']}  {m['describe']}  ({m['commit'][:9]})  built {m['built_at']}, "
           f"last run {b._fmt_dur(m.get('duration_s', 0))}{_steps_note(m)}")
    _plain(f"bin dir    {m['bin_dir']}")
    _plain(f"engine     {m.get('engine')}")
    _plain(f"variants   {', '.join(m.get('variants') or []) or 'none'}"
           + (f"   without: {', '.join(m['without'])}" if m.get("without") else ""))
    _plain(f"toolchain  CMake {tc.get('cmake')} · CUDA {tc.get('nvcc')} · {tc.get('msvc')} · "
           f"git {tc.get('git')} · vcpkg {tc.get('vcpkg')} · driver {m.get('driver')}")
    _plain(f"hardware   {m.get('gpu')} (cc {m.get('compute_capability')}) · {m.get('cpu')} · "
           f"hw identity {m.get('hw_identity')}")
    tb = Table("CMake (from CMakeCache)", "value", "requested")
    req = m.get("cmake_requested") or {}
    for k, v in (m.get("cmake_cache") or {}).items():
        tb.add_row(k, v if len(v) < 60 else "…" + v[-57:], req.get(k, ""))
    console.print(tb)
    for v in m.get("verified") or []:
        _plain(f"  {'✓' if v['ok'] else '✗'} {v['name']}: {v['detail']}")
    tb = Table("binary", "sha256")
    for k, v in (m.get("binaries_sha256") or {}).items():
        tb.add_row(k, v)
    console.print(tb)


@build.command("versions")
@click.argument("target", required=False)
@click.option("--remote", is_flag=True, help="git fetch --tags first.")
@click.option("--limit", default=8, show_default=True)
@json_option
@pass_ctx
def build_versions(ctx: Ctx, target, remote, limit) -> None:
    """llama.cpp versions you can build (vX.Y.Z releases, b<N> tags) and what is built here."""
    from dataclasses import asdict

    from mse_llm import build as b

    t, _ = _target(target)
    try:
        v = b.versions(t.src, limit, remote)
    except b.BuildError as e:
        raise click.ClickException(str(e)) from e
    if ctx.json:
        d = asdict(v)
        d["src"] = str(v.src)
        click.echo(json.dumps(d, indent=2))
        return
    _plain(f"llama.cpp at {v.src}   checked out: {v.head} ({v.head_describe}), branch {v.branch}")

    def rows(title, items):
        console.print(f"[bold]{title}[/]")
        if not items:
            _plain("  (none)")
        for tag, date, ahead in items:
            note = "" if ahead is None else ("  ← your checkout" if ahead == 0 else
                                             f"  ← your checkout is {ahead} commits newer")
            _plain(f"  {tag:<10} {date}{note}")

    rows("Releases", v.releases)
    rows("Build tags (newest first)", v.builds)
    console.print("[bold]Built here[/]")
    for tgt, desc, d in v.built:
        _plain(f"  {tgt:<40} {desc:<22} {d}")
    if not v.built:
        _plain("  (nothing built yet)")
    _plain(f"\nBuild one:  mse-llm build run {t.name} --ref <tag> [--keep-previous]")


@build.command("explain")
@click.argument("flags", nargs=-1)
def build_explain(flags) -> None:
    """Plain-language meaning of CMake flags (all if none given), from config/builds/_cmake-flags.toml."""
    from mse_llm import build as b

    docs = b.load_flag_docs()
    names = [f.upper().removeprefix("-D").split("=")[0] for f in flags] or list(docs)
    for n in names:
        d = docs.get(n)
        if not d:
            console.print(f"[yellow]{n}[/]: not in config/builds/_cmake-flags.toml")
            continue
        console.print(f"[bold]{n}[/]  [dim]({d.get('group', '?')}, default: {d.get('default', '?')})[/]")
        for k in ("summary", "effect", "note", "source"):
            if d.get(k):
                _plain(f"  {k:8} {d[k]}")
        if d.get("affects"):
            _plain(f"  affects  {', '.join(d['affects'])}")


# ---------------------------------------------------------------- variants

def _variant_names(patterns, from_file, include_commented=False) -> list[str]:
    from mse_llm import variants as vv

    names: list[str] = []
    for p in patterns:
        names += vv.expand(p)
    if from_file:
        text = sys.stdin.read() if from_file == "-" else Path(from_file).read_text(encoding="utf-8")
        names += [e.name for e in vv.parse_list(text, include_disabled=include_commented)]
    return names


@main.group()
def variants() -> None:
    """Variant names (fa_Kq8_Vq4_b4096_ub4096_moe41_sd3cache, legacy extra_* too): decode, encode, expand."""


@variants.command("decode")
@json_option
@click.argument("names", nargs=-1, required=True)
@click.option("--tool", type=click.Choice(["server", "bench"]), default="server", show_default=True)
@click.option("--model", default=None, help="Model whose drafter fills --model-draft for sd*mtp.")
@pass_ctx
def variants_decode(ctx: Ctx, names, tool, model) -> None:
    """Show the fields and llama.cpp flags of variant names (brace patterns allowed)."""
    from mse_llm import variants as vv

    draft = ctx.cfg.model(model)[1].draft if model else None
    out, failed = [], False
    for pat in names:
        for n in vv.expand(pat):
            row: dict = {"input": n}
            try:
                v = vv.decode(n)
                flags, warn = v.flags(tool, draft=draft)
                row |= {"name": v.name, "fields": v.as_dict(), "flags": flags, "warnings": warn}
            except vv.VariantError as e:
                row["error"] = str(e)
                failed = True
            out.append(row)
    if ctx.json:
        click.echo(json.dumps(out, indent=2))
    else:
        for r in out:
            if "error" in r:
                console.print(f"[red]✗[/] {escape(r['input'])}: {escape(r['error'])}")
                continue
            same = "" if r["name"] == r["input"] else f"  [dim](canonical {r['name']})[/]"
            console.print(f"[bold]{escape(r['input'])}[/]{same}")
            _plain("  " + (" ".join(r["flags"]) or "(no flags)"))
            for w in r["warnings"]:
                console.print(f"  [yellow]![/] {escape(w)}")
    sys.exit(1 if failed else 0)


@variants.command("encode", context_settings={"ignore_unknown_options": True})
@json_option
@click.argument("flags", nargs=-1, type=click.UNPROCESSED, required=True)
@pass_ctx
def variants_encode(ctx: Ctx, flags) -> None:
    """Variant name for llama.cpp flags, e.g. `variants encode -- -fa on -ctk q8_0 -b 2048 -ub 2048 -ncmoe 16`."""
    from mse_llm import variants as vv

    v, rest = vv.from_flags(list(flags))
    if ctx.json:
        click.echo(json.dumps({"name": v.name, "fields": v.as_dict(), "ignored": rest}, indent=2))
        return
    _plain(v.name)
    if rest:
        console.print(f"[dim]not part of a variant: {escape(' '.join(rest))}[/]")


@variants.command("expand")
@json_option
@click.argument("patterns", nargs=-1)
@click.option("--from", "from_file", type=click.Path(allow_dash=True), default=None,
              help="Read names from a file or - (stdin): one per line, pasted bash arrays and # comments OK.")
@click.option("--include-commented", is_flag=True, help="Also the commented-out entries of a pasted bash array.")
@click.option("--sweep", default=None, help="The variants of [sweeps.NAME] (names + matrix).")
@pass_ctx
def variants_expand(ctx: Ctx, patterns, from_file, include_commented, sweep) -> None:
    """Expand brace patterns (moe{18,21}_sd{3..5}mtp), pasted lists or a sweep into canonical names."""
    from mse_llm import variants as vv

    try:
        names = _variant_names(patterns, from_file, include_commented)
        vs = vv.resolve(names)
        if sweep:
            sw = ctx.cfg.sweeps.get(sweep)
            if sw is None:
                raise click.ClickException(f"unknown sweep {sweep!r}; known: {', '.join(ctx.cfg.sweeps) or '-'}")
            vs += [v for v in vv.resolve([e.v for e in sw.entries()]) if v.name not in {x.name for x in vs}]
    except vv.VariantError as e:
        raise click.ClickException(str(e)) from e
    if ctx.json:
        click.echo(json.dumps([v.name for v in vs], indent=2))
    else:
        for v in vs:
            _plain(v.name)


# ---------------------------------------------------------------- models

@main.group()
def models() -> None:
    """Configured models ([models.*], bash MODELS / DRAFTER_MODELS / *_EXTRA)."""


@models.command("list")
@json_option
@pass_ctx
def models_list(ctx: Ctx) -> None:
    ms = ctx.cfg.models
    if ctx.json:
        data = {k: m.model_dump() | {"exists": Path(m.path).is_file()} for k, m in ms.items()}
        click.echo(json.dumps(data, indent=2))
        return
    t = Table(title="models", expand=False)
    for col in ("name", "file", "ctx", "draft", "default_serve"):
        t.add_column(col, overflow="fold")
    for k, m in ms.items():
        draft = "-" if not m.draft else ("builtin" if m.draft == "builtin" else Path(m.draft).name)
        t.add_row(k, ("" if Path(m.path).is_file() else "[red]missing[/] ") + escape(Path(m.path).name),
                  str(m.ctx or "-"), escape(draft), escape(m.default_serve or "-")
                  + (f"\n[dim]spec: {escape(m.default_serve_spec)}[/]" if m.default_serve_spec else ""))
    console.print(t)


@models.command("show")
@json_option
@click.argument("name")
@pass_ctx
def models_show(ctx: Ctx, name) -> None:
    try:
        key, m = ctx.cfg.model(name)
    except KeyError as e:
        raise click.ClickException(str(e.args[0])) from e
    d = m.model_dump() | {"name": key, "label": m.label, "exists": Path(m.path).is_file()}
    if ctx.json:
        click.echo(json.dumps(d, indent=2))
        return
    for k, val in d.items():
        if val not in (None, [], {}):
            _plain(f"{k:19} {val}")


# ---------------------------------------------------------------- bench / benchy / serve / eval (dry-run, M3)

def _show_commands(ctx: Ctx, cmds: list, dry_run: bool, runs_in: str) -> None:
    """Print built commands (with flag-check errors and warnings); without --dry-run say when running lands."""
    _print_commands(ctx, cmds)
    bad = [c for c in cmds if not c.ok]
    if bad:
        if not ctx.json:
            console.print(f"[red]{len(bad)} of {len(cmds)} command(s) failed the flag check[/]")
        sys.exit(1)
    if not dry_run:
        raise click.ClickException(f"running this arrives in {runs_in}; for now use --dry-run to see the commands")


def _print_commands(ctx: Ctx, cmds: list) -> None:
    from mse_llm.runners import display_argv

    if ctx.json:
        click.echo(json.dumps([c.as_dict() for c in cmds], indent=2))
    else:
        for c in cmds:
            ctxs = f"  [dim](ctx {c.ctx})[/]" if c.ctx else ""
            console.print(f"[bold]▶ {escape(c.label)}[/]{ctxs}")
            parts = [("server", c.server), ("client", c)] if c.server else [("", c)]
            for role, x in parts:
                env = " ".join(f"{k}={v}" for k, v in x.env.items())
                _plain(f"  {role + '  ' if role else ''}{env + ' ' if env else ''}{display_argv(x.argv)}")
                for w in x.warnings:
                    console.print(f"    [yellow]![/] {escape(w)}")
                for e in x.errors:
                    console.print(f"    [red]✗[/] {escape(e)}")
        snaps = {c.snapshot or (c.server.snapshot if c.server else None) for c in cmds} - {None}
        if snaps:
            console.print(f"[dim]flags checked against help snapshot {', '.join(sorted(snaps))}[/]")


def _variants_for(ctx: Ctx, model: str, patterns, from_file, spec: bool = False) -> list:
    from mse_llm import variants as vv

    try:
        _, m = ctx.cfg.model(model)
        names = _variant_names(patterns, from_file)
        if not names:
            default = m.default_serve_spec if spec else m.default_serve
            if not default:
                raise click.ClickException(f"{model}: no -v VARIANT and no "
                                           f"{'default_serve_spec' if spec else 'default_serve'} in [models.{model}]")
            names = [default]
        return vv.resolve(names)
    except vv.VariantError as e:
        raise click.ClickException(str(e)) from e
    except KeyError as e:
        raise click.ClickException(str(e.args[0])) from e


def _variant_options(f):
    for opt in reversed([
        click.option("-v", "--variant", "patterns", multiple=True,
                     help="Variant name or brace pattern (repeatable). Legacy extra_* names work."),
        click.option("--from", "from_file", type=click.Path(allow_dash=True), default=None,
                     help="Variant list from a file or - (stdin), e.g. a pasted bash extras_list."),
        click.option("--engine", default=None, help="Engine (default: cuda_flash)."),
        click.option("--dry-run", is_flag=True, help="Print the commands (flag-checked) and stop."),
        json_option,
    ]):
        f = opt(f)
    return f


@main.command("eval")
@click.argument("model")
@click.argument("task")
@click.option("-v", "--variant", "patterns", multiple=True, help="Variant (default: [models].default_serve).")
@click.option("--ctx", "ctx_", type=int, default=None)
@click.option("--engine", default=None)
@click.option("--dry-run", is_flag=True)
@json_option
@pass_ctx
def eval_cmd(ctx: Ctx, model, task, patterns, ctx_, engine, dry_run) -> None:
    """inspect-ai TASK (mmlu_pro, gpqa_diamond, math_aime_24, ...) against a background server (bash run_INSPECT_*)."""
    from mse_llm.runners import inspect_command

    vs = _variants_for(ctx, model, patterns, None)
    cmds = [inspect_command(ctx.cfg, model, task, v, ctx_, engine) for v in vs]
    _show_commands(ctx, cmds, dry_run, "M12 (eval)")


# ---------------------------------------------------------------- fit (M4, §4.6)

_VERDICT_STYLE = {"fits": "green", "TIGHT": "yellow", "NO-FIT": "red", "UNSURE": "magenta", "ERROR": "red"}


def _fit_inputs(ctx: Ctx, model: str, patterns, from_file, include_commented: bool) -> list:
    from mse_llm import fit as ff
    from mse_llm import variants as vv

    try:
        entries: list[tuple[str, str]] = []
        for p in patterns:
            entries += [(n, "") for n in vv.expand(p)]
        if from_file:
            text = sys.stdin.read() if from_file == "-" else Path(from_file).read_text(encoding="utf-8")
            entries += [(e.name, e.note) for e in vv.parse_list(text, include_disabled=include_commented)]
        if not entries:
            try:
                _, m = ctx.cfg.model(model)
            except KeyError:
                raise click.ClickException(f"{model}: give -v VARIANT or --from FILE (no [models.*] defaults "
                                           "for a plain .gguf path)") from None
            entries = [(n, "default_serve_spec" if n == m.default_serve_spec else "default_serve")
                       for n in (m.default_serve, m.default_serve_spec) if n]
            if not entries:
                raise click.ClickException(f"{model}: no -v VARIANT and no default_serve in [models.{model}]")
            if from_file and not include_commented:
                console.print("[yellow]![/] every entry in the list is commented out; "
                              "--include-commented checks them anyway. Showing the model's defaults.")
        out, seen = [], set()
        for name, note in entries:
            v = vv.decode(name)
            if v.name not in seen:
                seen.add(v.name)
                out.append(ff.FitInput(v, note))
        return out
    except vv.VariantError as e:
        raise click.ClickException(str(e)) from e


@main.command("fit")
@click.argument("model", metavar="GGUF_OR_MODEL")
@click.option("-v", "--variant", "patterns", multiple=True,
              help="Variant name or brace pattern (repeatable). Default: the model's default_serve(_spec).")
@click.option("--from", "from_file", type=click.Path(allow_dash=True), default=None,
              help="Variant list from a file or - (stdin): a pasted bash extras_list, comments kept as notes.")
@click.option("--include-commented", is_flag=True, help="Also check the #-commented entries of --from.")
@click.option("--ctx", "ctxs", type=int, multiple=True, help="Context size (repeatable). Default: [models].ctx.")
@click.option("--kv", default=None, metavar="K/V",
              help="Cache types for variants that set none, e.g. q8_0/q8_0 (llama.cpp default: f16).")
@click.option("--free", "free_", default="auto", show_default=True, metavar="auto|MiB",
              help="Free VRAM to fit into: auto = NVML right now.")
@click.option("--margin", type=int, default=768, show_default=True,
              help="MiB of headroom below which a fit counts as TIGHT.")
@click.option("--draft", default=None, metavar="PATH|builtin",
              help="Drafter for sd*mtp variants (default: [models.*].draft).")
@click.option("--mmproj", default=None, type=click.Path(dir_okay=False),
              help="Count a vision projector (default: [models.*].mmproj).")
@click.option("--suggest", is_flag=True, help="Lowest -ncmoe / largest -ub that still fits, per variant.")
@click.option("--strict", is_flag=True, help="Exit 1 when any part is NOT COUNTED (verdict UNSURE).")
@click.option("--engine", default=None, help="Engine whose llama-fit-params is used (default: cuda_flash).")
@json_option
@pass_ctx
def fit_cmd(ctx: Ctx, model, patterns, from_file, include_commented, ctxs, kv, free_, margin, draft, mmproj,
            suggest, strict, engine) -> None:
    """Expected VRAM / RAM and fit per variant and context (llama-fit-params, no model load)."""
    from mse_llm import fit as ff

    inputs = [ff.FitInput(ff.apply_kv(i.variant, kv), i.note)
              for i in _fit_inputs(ctx, model, patterns, from_file, include_commented)]
    if str(free_).lower() == "auto":
        free_mib, free_src = ff.free_vram_now()
    else:
        try:
            free_mib, free_src = int(free_), "--free"
        except ValueError:
            raise click.BadParameter("auto or a number of MiB", param_hint="--free") from None
    try:
        if ctx.json or not sys.stdout.isatty():
            rep = ff.run_fit(ctx.cfg, model, inputs, list(ctxs), engine=engine, free_mib=free_mib,
                             free_source=free_src, margin_mib=margin, draft=draft, mmproj=mmproj, suggest=suggest)
        else:
            with console.status("llama-fit-params …") as st:
                rep = ff.run_fit(ctx.cfg, model, inputs, list(ctxs), engine=engine, free_mib=free_mib,
                                 free_source=free_src, margin_mib=margin, draft=draft, mmproj=mmproj,
                                 suggest=suggest,
                                 progress=lambda i, n, what: st.update(f"llama-fit-params {i + 1}/{n} {what}"))
    except ff.FitError as e:
        raise click.ClickException(str(e)) from e
    except KeyError as e:
        raise click.ClickException(str(e.args[0])) from e
    if ctx.json:
        click.echo(json.dumps(rep.as_dict(), indent=2))
    else:
        _print_fit(rep)
    if rep.any_error or (strict and rep.any_uncounted):
        sys.exit(1)


def _print_fit(rep) -> None:
    for w in rep.warnings:
        console.print(f"[yellow]![/] {escape(w)}")
    free = (f"free VRAM {rep.free_mib} MiB ({rep.free_source})" if rep.free_mib is not None
            else f"[yellow]free VRAM unknown[/] ({rep.free_source})")
    for c in dict.fromkeys(r.ctx for r in rep.rows):
        rows = [r for r in rep.rows if r.ctx == c]
        console.print(f"\nmodel [bold]{escape(Path(rep.path).name)}[/]  ctx {c}  {free}  margin {rep.margin_mib}")
        t = Table(box=None, pad_edge=False, header_style="bold")
        for col, just in (("VARIANT", "left"), ("GPU MiB", "right"), ("(model+KV+compute)", "right"),
                          ("host MiB", "right"), ("left", "right"), ("VERDICT", "left"), ("NOTE", "left")):
            t.add_column(col, justify=just, overflow="fold")
        for r in rows:
            v = r.verdict
            style = _VERDICT_STYLE.get(v.rstrip("~"), "")
            mg = r.main_gpu
            notes = [r.note] if r.note else []
            for p in r.parts:
                if p.name == "main" or p.source == "not used":
                    continue
                if p.source == "none":
                    notes.append(f"+{p.name}: no extra VRAM")
                elif p.source == "NOT COUNTED":
                    notes.append(f"+{p.name} NOT COUNTED")
                else:
                    notes.append(f"+{p.name} {p.gpu_mib:.0f}" + ("~" if p.source == "estimated" else ""))
            if r.error:
                t.add_row(escape(r.variant), "-", "", "", "", f"[red]{v}[/]", escape(r.error))
                continue
            t.add_row(escape(r.variant), f"{r.gpu_mib:.0f}",
                      f"({mg.model}+{mg.context}+{mg.compute})" if mg else "", f"{r.host_mib:.0f}",
                      "-" if r.left_mib is None else f"{r.left_mib:.0f}",
                      f"[{style}]{v}[/]" if style else v, escape("; ".join(notes)))
        console.print(t)
        for r in rows:
            extra = r.footer() + [f"→ {s}" for s in r.suggest]
            if extra:
                console.print(f"  [bold]{escape(r.variant)}[/]")
                for line in extra:
                    console.print(f"    {escape(line)}")
    info = rep.info
    if info and any(r.suggest for r in rep.rows):
        console.print(f"\n[dim]{info.moe_layers} MoE layers, 1 layer = {info.expert_layer_mib:.0f} MiB, "
                      f"1 ubatch token ≈ {info.vocab * 4 / 2**20 if info.vocab else 0:.2f} MiB "
                      f"(n_used/n_expert = {info.n_expert_used}/{info.n_expert})[/]")
    legend = ("fits = all parts measured · fits~ = some estimated · UNSURE = a part NOT COUNTED · "
              f"TIGHT = under {rep.margin_mib} MiB left")
    console.print(f"[dim]{legend}[/]")


from mse_llm import cli_bench as _cli_bench  # noqa: E402,F401  (registers bench, benchy, sweep, results)
from mse_llm import cli_monitor as _cli_monitor  # noqa: E402,F401  (registers monitor + notify)
from mse_llm import cli_predict as _cli_predict  # noqa: E402,F401  (registers predict, results calibrate, sweep search)
from mse_llm import cli_report as _cli_report  # noqa: E402,F401  (registers report, results compare/export)
from mse_llm import cli_serve as _cli_serve  # noqa: E402,F401  (registers serve + presets)

if __name__ == "__main__":
    main()
