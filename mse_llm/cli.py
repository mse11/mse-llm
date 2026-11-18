"""mse-llm command line (click). Commands stay thin; logic lives in the modules."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from mse_llm import __version__
from mse_llm.checks import ICON, Check, verdict

console = Console(highlight=False)


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


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", "config", type=click.Path(dir_okay=False), envvar="MSE_LLM_CONFIG",
              help="Config TOML (default: config/config.toml, else config/config.example.toml).")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@click.option("-v", "--verbose", is_flag=True)
@click.option("--hw-fixture", type=click.Path(exists=True, dir_okay=False),
              help="Use a HardwareInfo JSON instead of probing (tests/CI). Same as MSE_LLM_TEST_HARDWARE.")
@click.version_option(__version__, prog_name="mse-llm")
@click.pass_context
def main(ctx: click.Context, config, as_json, verbose, hw_fixture) -> None:
    """Benchmark, tune and serve local LLMs (llama.cpp)."""
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
    for c in checks:
        detail = c.detail + (f"\n[dim]-> {c.hint}[/]" if c.hint and c.status in ("warn", "fail") else "")
        t.add_row(c.section, c.name, ICON[c.status], detail)
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
    from mse_llm.util import exe, project_root, run

    name, eng = ctx.cfg.engine(engine)
    bdir = eng.bin_path()
    have_bins = all(exe(bdir / t).is_file() for t in hs.TOOLS)
    report: dict = {"engine": name, "bin_dir": str(bdir), "binaries": have_bins}

    current: dict[str, hs.HelpInfo] = {}
    if have_bins:
        for t in hs.TOOLS:
            r = run([str(exe(bdir / t)), "--help"], timeout=30)
            current[t] = hs.parse_help(r.stdout + r.stderr)
        if capture:
            sid, dest = hs.capture(bdir)
            report["captured"] = str(dest)
    elif capture:
        raise click.ClickException(f"no binaries in {bdir}; build first (mse-llm build run, M2)")

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
            console.print("[yellow]WARN[/] binaries not built yet (mse-llm build run, M2): "
                          "using the stored snapshot only" if stored else
                          "[yellow]WARN[/] no binaries and no stored snapshot: nothing to check yet. "
                          "Build llama.cpp (M2), then run `mse-llm doctor flags --capture`.")
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
                    console.print(f"  [{color}]{k:8}[/] {x}")
        if helps:
            if findings:
                console.print(f"[bold]audit[/]: {len(findings)} finding(s)")
                for f in findings:
                    console.print(f"  [yellow]{f.kind:13}[/] {f.source:32} {f.message}")
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
        raise click.ClickException(f"{binary} not found and no help snapshot; build llama.cpp first (M2)")
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


if __name__ == "__main__":
    main()
