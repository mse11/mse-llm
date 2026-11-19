"""`monitor` and `notify` commands, and the monitor helpers `serve` / `presets serve` use (M6, §5)."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import click
from rich.markup import escape

from mse_llm.cli import Ctx, _plain, console, json_option, main, pass_ctx

_LVL = {"pass": "[green]✓[/]", "warn": "[yellow]⚠[/]", "fail": "[red]![/]"}


def guard_cfg(ctx: Ctx, *, enabled: bool = True, expected_host_mib: float | None = None):
    from mse_llm.monitor import guard_cfg_from

    return guard_cfg_from(ctx.cfg.monitor, enabled=enabled, expected_host_mib=expected_host_mib)


def thresholds(ctx: Ctx):
    from mse_llm.monitor import thresholds_from

    return thresholds_from(ctx.cfg.monitor)


def make_monitor(ctx: Ctx, *, guard: bool, expected_host_mib: float | None = None, csv_path: Path | None = None,
                 on_trip=None, on_sample=None, host_extra_mib: float = 0.0):
    from mse_llm.monitor import monitor_from

    return monitor_from(ctx.cfg.monitor, guard=guard, expected_host_mib=expected_host_mib, csv_path=csv_path,
                        on_trip=on_trip, on_sample=on_sample, host_extra_mib=host_extra_mib)


def _dur(s: float) -> str:
    return f"{int(s // 60)}m {int(s % 60)}s"


def _stat(sm, key: str, unit: str = "MiB", base: bool = True) -> str | None:
    st = sm.get(key)
    if st is None:
        return None
    s = f"peak {st.peak:.0f} {unit}"
    if base and st.baseline is not None:
        s += f" (baseline {st.baseline:.0f}, {st.delta:+.0f})"
    return s


def print_summary(sm, *, title: str = "monitor") -> None:
    """The bash RESULTS block, from the in-process samples."""
    console.print(f"[bold]──── {escape(title)}: {sm.samples} samples every {sm.interval:g} s, "
                  f"{_dur(sm.duration_s)} ────[/]")
    for e in sm.errors:
        console.print(f"  [yellow]![/] {escape(e)}")
    st = {s.area: [] for s in sm.statuses}
    for s in sm.statuses:
        st[s.area].append(s)
    ded = _stat(sm, "vram_used_mib")
    if ded:
        tot = sm.get("vram_total_mib")
        console.print(f"GPU memory   dedicated {ded}" + (f" of {tot.peak:.0f} MiB" if tot else ""))
    sh = _stat(sm, "nonlocal_mib") or _stat(sm, "shared_mib")
    if sh:
        line = f"             shared    {sh}"
        ps = sm.get("proc_shared_mib")
        if ps:
            line += f"; server {ps.peak:.0f} MiB"
        if sm.expected_host_mib is not None:
            line += f", host buffers expected {sm.expected_host_mib:.0f} MiB"
        console.print(line)
    for s in st.get("gpu", []):
        console.print(f"  {_LVL[s.level]} {escape(s.text)}")
    ram = sm.get("ram_used_mib")
    if ram:
        tot = sm.get("ram_total_mib")
        pct = sm.get("ram_pct")
        line = f"System RAM   peak {ram.peak:.0f}" + (f" / {tot.peak:.0f}" if tot else "") + " MiB"
        if pct:
            line += f" ({pct.peak:.0f} %)"
        if ram.delta is not None:
            line += f", delta {ram.delta:+.0f} MiB"
        c = sm.get("commit_pct")
        if c:
            line += f"; commit peak {c.peak:.0f} %"
        console.print(line)
    for s in st.get("ram", []):
        console.print(f"  {_LVL[s.level]} {escape(s.text)}")
    t = sm.get("gpu_temp_c")
    if t:
        line = f"GPU temp     peak {t.peak:.0f} °C, mean {t.mean:.0f} °C"
        p = sm.get("gpu_power_w")
        if p:
            line += f"; power peak {p.peak:.0f} W"
        if sm.throttle:
            line += f"; throttle: {', '.join(sm.throttle)}"
        console.print(line)
    for s in st.get("temp", []):
        console.print(f"  {_LVL[s.level]} {escape(s.text)}")
    parts = []
    for key, label in (("proc_rss_mib", "RSS"), ("proc_private_mib", "private"), ("proc_vram_mib", "VRAM"),
                       ("pcie_rx_mbs", "PCIe RX MB/s"), ("pcie_tx_mbs", "PCIe TX MB/s")):
        s = sm.get(key)
        if s:
            parts.append(f"{label} peak {s.peak:.0f}")
    if parts:
        console.print("Server       " + ", ".join(parts))
    if sm.csv_path:
        console.print(f"[dim]samples: {escape(sm.csv_path)}[/]")


def live_line(s: dict) -> str:
    def g(k, fmt="{:.0f}"):
        v = s.get(k)
        return "-" if v is None else fmt.format(v)
    return (f"{s['t']:6.1f}s  VRAM {g('vram_used_mib')} MiB  shared {g('shared_mib')} (server {g('proc_shared_mib')})"
            f"  RAM {g('ram_pct')} %  commit {g('commit_pct')} %  {g('gpu_temp_c')} °C  {g('gpu_power_w')} W")


@main.command("monitor", context_settings={"ignore_unknown_options": True, "allow_interspersed_args": False})
@click.argument("cmd", nargs=-1, type=click.UNPROCESSED)
@click.option("--duration", type=float, default=None, help="Seconds to sample (no CMD). Default: until Ctrl+C.")
@click.option("--interval", type=float, default=None, help="Seconds between samples (default: [monitor].interval).")
@click.option("--pid", type=int, default=None, help="Follow this process tree (server VRAM / shared / RSS).")
@click.option("--csv", "csv_", type=click.Path(dir_okay=False), default=None, help="Write every sample here.")
@click.option("--expect-host", type=float, default=None, metavar="MiB",
              help="Host buffers the server should pin (fit's host MiB): tells -ncmoe buffers from a spill.")
@click.option("--guard/--no-guard", default=False,
              help="Stop CMD (or --pid's tree) on a spill / RAM / commit trip. Off by default here.")
@click.option("--live/--no-live", default=True, help="Print a sample line every 2 s.")
@click.option("--message", default=None, help="Spoken at the end (default: the command's name and result).")
@json_option
@pass_ctx
def monitor_cmd(ctx: Ctx, cmd, duration, interval, pid, csv_, expect_host, guard, live, message) -> None:
    """VRAM / shared GPU memory / RAM / temperature sampling, in-process (bash EXEC_cmd_with_gpu_and_ram_monitoring).

    \b
    mse-llm monitor --duration 30                      # sample the system
    mse-llm monitor --pid 1234 --expect-host 15807     # follow a running server
    mse-llm monitor --guard -- llama-bench.exe -m X.gguf ...   # run a command under monitor + guard
    """
    import json

    from mse_llm import notify
    from mse_llm import proc as pr

    if interval:
        ctx.cfg.monitor.interval = interval
    last = [0.0]

    def on_sample(s):
        if live and not ctx.json and s["t"] - last[0] >= 2.0:
            last[0] = s["t"]
            _plain(live_line(s))

    async def run() -> tuple[object, int | None]:
        sp = None
        trip_note: list[str] = []

        async def on_trip(reason: str):
            trip_note.append(reason)
            console.print(f"[red]■ memory guard:[/] {escape(reason)}")
            if sp is not None and sp.proc is not None:
                rep = await sp.stop(grace=2)
                console.print(f"  {escape(rep.summary())}")
            elif pid:
                rep = await asyncio.to_thread(pr.stop_tree, pid, grace=2)
                console.print(f"  {escape(rep.summary())}")

        mon = make_monitor(ctx, guard=guard, expected_host_mib=expect_host, csv_path=Path(csv_) if csv_ else None,
                           on_trip=on_trip, on_sample=on_sample)
        rc = None
        async with mon:
            if cmd:
                sp = pr.Spawned(list(cmd), sink=_plain)
                try:
                    await sp.start()
                except pr.ProcError as e:
                    raise click.ClickException(str(e)) from None
                mon.track(sp.pid)
                try:
                    rc = await sp.wait()
                finally:
                    if sp.returncode is None:
                        await sp.stop()
            else:
                mon.track(pid)
                end = time.monotonic() + duration if duration else None
                while end is None or time.monotonic() < end:
                    if mon.trip_event.is_set() and pid:
                        break
                    await asyncio.sleep(0.2)
        return mon, rc

    try:
        mon, rc = asyncio.run(run())
    except KeyboardInterrupt:
        sys.exit(130)
    sm = mon.summary()
    if ctx.json:
        click.echo(json.dumps({**sm.as_dict(), "returncode": rc}, indent=2))
    else:
        print_summary(sm, title="monitor" + (f" {Path(cmd[0]).name}" if cmd else ""))
    if cmd:
        what = message or (f"{Path(cmd[0]).stem} stopped by the memory guard" if sm.tripped else
                           f"{Path(cmd[0]).stem} finished" + (f" with exit code {rc}" if rc else ""))
        notify.notify(what, ctx.cfg.defaults.notify)
    if sm.tripped:
        sys.exit(2)
    if rc:
        sys.exit(rc)


@main.command("notify")
@click.argument("text", nargs=-1, required=True)
@pass_ctx
def notify_cmd(ctx: Ctx, text) -> None:
    """Speak TEXT (the end-of-run voice message; bash EXEC_user_notification_WIN_T2S)."""
    from mse_llm import notify

    if notify.disabled():
        console.print("notifications are off (--no-notify / MSE_LLM_NO_NOTIFY)")
        return
    if not notify.speak(" ".join(text)):
        raise click.ClickException("no speech engine found (Windows SAPI, say, spd-say, espeak, notify-send)")
