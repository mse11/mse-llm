"""`tune moe-cache` (M11, §4.4): llama.cpp's moe-cache-tune.py around the variant's server argv."""

from __future__ import annotations

import asyncio
import json
import sys
import time

import click
from rich.markup import escape

from mse_llm.cli import Ctx, _plain, console, json_option, main, pass_ctx
from mse_llm.cli_bench import allow_low_power_option, power_gate


@main.group()
def tune() -> None:
    """Tuners that pick settings by measuring (llama.cpp's own scripts, run on the normal argv)."""


@tune.command("moe-cache")
@click.argument("model")
@click.option("-v", "--variant", "variant", required=True,
              help="Base variant with experts in host memory (cmoe or moe<N>); mc / cs in it are ignored.")
@click.option("--ctx", "ctx_", type=int, default=None, help="Server context -c (default: [models].ctx).")
@click.option("--engine", default=None, help="Engine (default: cuda_flash).")
@click.option("--max-mib", type=int, default=None, help="Largest cache to tune (default: what fits on the GPU).")
@click.option("--markdown", is_flag=True, help="The tuner's tables as Markdown.")
@click.option("--no-monitor", is_flag=True, help="No VRAM / RAM monitoring.")
@click.option("--no-guard", is_flag=True, help="Monitor, but don't stop the tuner on a spill / RAM / commit trip.")
@click.option("--no-save", is_flag=True, help="Don't write the recommendation to the results store.")
@click.option("--dry-run", is_flag=True, help="Print the command and stop.")
@allow_low_power_option
@json_option
@pass_ctx
def tune_moe_cache(ctx: Ctx, model, variant, ctx_, engine, max_mib, markdown, no_monitor, no_guard, no_save,
                   dry_run, allow_low_power) -> None:
    """Pick --moe-cache-mib / --moe-cache-cpu-share with llama.cpp's scripts/moe-cache-tune.py (6 server runs:
    2 cache sizes x shares 0 / 0.5 / 0.95, then a replay of the routing for other sizes and shares). The
    recommendation is stored as a variant that staged sweeps use as mc{tuned} / cs{tuned}."""
    from mse_llm import tune as tu
    from mse_llm.runners import display_argv
    from mse_llm.variants import VariantError, decode

    try:
        v = decode(variant)
        t = tu.tune_command(ctx.cfg, model, v, ctx_, engine, max_mib=max_mib, markdown=markdown)
    except (VariantError, tu.TuneError, KeyError) as e:
        raise click.ClickException(str(e.args[0] if isinstance(e, KeyError) else e)) from e
    if t["errors"]:
        raise click.ClickException("flag check: " + "; ".join(t["errors"]))
    if dry_run or ctx.json and dry_run:
        if ctx.json:
            click.echo(json.dumps({"argv": t["argv"], "server_argv": t["server_argv"], "out": str(t["out"]),
                                   "base": t["base"].name}, indent=2))
            return
        console.print(f"[bold]tune moe-cache[/] {escape(t['model'])} {escape(t['base'].name)}")
        _plain("  " + display_argv(t["argv"]))
        console.print(f"[dim]run directory (reused while the command stays the same): {escape(str(t['out']))}[/]")
        return
    power_gate(allow_low_power)
    for w in t["warnings"]:
        console.print(f"[yellow]![/] {escape(w)}")
    console.print(f"[bold]tune moe-cache[/] {escape(t['model'])} {escape(t['base'].name)}: 6 server runs "
                  f"(reused from {escape(str(t['out']))} when the command is the same)")
    try:
        res = asyncio.run(_run(ctx, t, monitor=not no_monitor, guard=not no_guard, write=not no_save))
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/]")
        sys.exit(130)
    rec = res.tune or {}
    if ctx.json:
        click.echo(json.dumps(res.model_dump(by_alias=True, exclude_none=True), indent=2))
    elif rec.get("variant"):
        console.print(f"[green]✓ recommended[/] --moe-cache-mib {rec['recommended_mib']} --moe-cache-cpu-share "
                      f"{rec['recommended_share']:g} ({rec['tg_tps']} t/s"
                      + (f", {rec['tg_tps_share0']} with share 0" if rec.get("tg_tps_share0") else "") + ")")
        console.print(f"  variant [bold]{escape(rec['variant'])}[/]"
                      + ("  → mc{tuned} / cs{tuned} in staged sweeps" if not no_save else ""))
    elif rec.get("all_fit"):
        console.print(f"[yellow]{escape(rec['advice'])}[/]")
    if res.status != "ok":
        console.print(f"[red]✗ {res.status}[/]: {escape(res.reason or '')}")
        sys.exit(2 if res.status == "aborted" else 1)


async def _run(ctx: Ctx, t: dict, *, monitor: bool, guard: bool, write: bool):
    from mse_llm import bench as bb
    from mse_llm import notify
    from mse_llm import proc as pr
    from mse_llm import results as rs
    from mse_llm import tune as tu
    from mse_llm.util import project_root

    cfg = ctx.cfg
    env = bb.RunEnv()
    now = rs.now()
    run_id = rs.new_run_id("tune", now)
    raw = rs.raw_dir(run_id)
    raw.mkdir(parents=True, exist_ok=True)
    srv = t["server"]
    mkey, m = cfg.model(t["model"])
    ename, _ = cfg.engine(t["engine"])
    res = rs.RunResult(run_id=run_id, key=rs.run_key("tune", mkey, t["base"].name, srv.ctx, {"argv": t["argv"][3:]}),
                       timestamp=now.isoformat(timespec="seconds"), tool="tune", engine=ename, model=mkey,
                       model_path=m.path, variant=t["base"].name, variant_fields=t["base"].as_dict(), ctx=srv.ctx,
                       argv=list(t["argv"]), server_argv=list(t["server_argv"]), build=env.build_for(cfg, t["engine"]),
                       hw=env.hw_info())
    log = raw / "tune.log"
    res.logs["tool"] = rs.rel(log) or str(log)
    lines: list[str] = []
    # unbuffered: the tuner's tables (stdout) and its messages (stderr) stay in order
    sp = pr.Spawned(t["argv"], log_path=log, sink=lambda ln: _plain("  │ " + ln), cwd=project_root(),
                    env={"PYTHONUNBUFFERED": "1"})
    sp.watchers.append(lines.append)
    mon = None
    if monitor and cfg.defaults.monitor:
        from mse_llm.monitor import monitor_from

        mon = monitor_from(cfg.monitor, guard=guard, csv_path=raw / "monitor.csv")
    t0 = time.monotonic()

    async def trip(reason: str) -> None:
        console.print(f"[red]■ memory guard:[/] {reason} → stopping the tuner and its server")
        await sp.stop(grace=2)

    async def body() -> None:
        try:
            await sp.start()
        except pr.ProcError as e:
            res.reason = str(e)
            return
        rc = await sp.wait()
        if rc != 0 and not (mon is not None and mon.tripped):
            res.reason = f"moe-cache-tune.py exit code {rc}: {tu.failure_reason(lines)}"

    try:
        await bb._with_monitor(mon, lambda: sp.proc.pid if sp.proc is not None else None, body, trip)
    finally:
        bb._finish_monitor(res, mon)
    res.duration_s = round(time.monotonic() - t0, 1)
    res.tune = {**tu.parse_output(lines, t["base"]), "out_dir": rs.rel(t["out"]) or str(t["out"]),
                "script": rs.rel(t["script"]) or str(t["script"])}
    if res.tune.get("variant"):
        res.metrics = {"tg_tps": res.tune.get("tg_tps")}
    if mon is not None and mon.tripped:
        res.status, res.reason = "aborted", f"memory guard: {mon.tripped}"
    elif res.reason:
        res.status = "failed"
    elif not res.tune.get("variant") and not res.tune.get("all_fit"):
        res.status, res.reason = "failed", "no recommendation in the tuner output"
    if write:
        rs.write_run(res)
        if res.status == "ok" and res.tune.get("variant"):
            tu.save_tuned(mkey, res.tune, run_id)
    notify.notify(f"tune moe-cache {mkey}: {res.status}", cfg.defaults.notify)
    return res


__all__ = ["tune"]
