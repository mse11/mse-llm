"""`bench`, `benchy`, `sweep` and `results` commands (M7, §6, §8)."""

from __future__ import annotations

import asyncio
import json
import re
import sys

import click
from rich.markup import escape
from rich.table import Table

from mse_llm.cli import Ctx, _plain, _print_commands, _variants_for, console, json_option, main, pass_ctx

# llama-bench (stderr) and llama-benchy lines worth showing with --output filter
BENCH_FILTER = [r"[Ee]rror", r"[Ww]arn", r"[Ff]ail", r"out of memory", r"^Running test", r"^\s*Run \d+/\d+",
                r"Benchmarking model", r"Saving results", r"Device \d+:"]

# benchy looks up the GGUF name on Hugging Face as a tokenizer and falls back to gpt2 (as in bash): not an error
BENCH_QUIET = re.compile(r"Error loading tokenizer|private repository|Repository Not Found|Please make sure you|"
                         r"If you are trying to access|For more details, see https://huggingface")

_ICON = {"ok": "[green]✓[/]", "skipped": "[yellow]–[/]", "aborted": "[red]■[/]", "failed": "[red]✗[/]"}


def _run_options(f):
    for opt in reversed([
        click.option("--no-fit-check", is_flag=True, help="Don't run llama-fit-params before each job."),
        click.option("--force", is_flag=True,
                     help="Start jobs the fit check says don't fit (and sweep entries with status skip)."),
        click.option("--output", type=click.Choice(["all", "filter", "none"]), default=None,
                     help="Tool output on the console (default: [defaults].output; logs keep everything)."),
        click.option("--no-save", is_flag=True, help="Don't write to the results store."),
        click.option("--no-monitor", is_flag=True, help="No VRAM / shared memory / RAM monitoring."),
        click.option("--no-guard", is_flag=True, help="Monitor, but don't stop a job on a spill / RAM / commit trip."),
        click.option("--mem-cap", type=int, default=None, metavar="MiB",
                     help="Windows Job Object cap on the tool's committed memory "
                          "(default: [monitor].job_memory_cap_mib)."),
    ]):
        f = opt(f)
    return f


def _opts(ctx: Ctx, *, no_fit_check, force, output, no_save, no_monitor, no_guard, mem_cap, port=None):
    from mse_llm.bench import RunOpts
    from mse_llm.proc import OutputFilter

    mode = output or ctx.cfg.defaults.output
    filt = OutputFilter(mode, BENCH_FILTER)

    def sink(line: str) -> None:
        if filt(line) and (mode == "all" or not BENCH_QUIET.search(line)):
            _plain("  │ " + line)

    return RunOpts(fit_check=not no_fit_check, force=force, monitor=not no_monitor, guard=not no_guard,
                   mem_cap_mib=mem_cap if mem_cap is not None else ctx.cfg.monitor.job_memory_cap_mib, port=port,
                   write=not no_save, sink=sink, say=lambda s: console.print("  " + s))


def _fmt(x, nd: int = 1) -> str:
    return "-" if x is None else f"{x:,.{nd}f}"


def _result_line(r) -> str:
    m = r.metrics
    parts = []
    if m.get("pp_tps") is not None:
        parts.append(f"pp {m.get('n_prompt')}: [bold]{_fmt(m['pp_tps'])}[/] t/s")
    if m.get("tg_tps") is not None:
        parts.append(f"tg {m.get('n_gen')}: [bold]{_fmt(m['tg_tps'], 2)}[/] t/s")
    mon = r.monitor or {}
    vram = ((mon.get("stats") or {}).get("vram_used_mib") or {}).get("peak")
    if vram is not None:
        parts.append(f"VRAM peak {vram:.0f} MiB")
    if mon.get("spill_mib") is not None and mon.get("expected_host_mib") is not None:
        parts.append(f"shared {mon['spill_mib']:+.0f} MiB vs fit")
    if r.duration_s is not None:
        parts.append(f"{r.duration_s:.0f} s")
    head = f"{_ICON[r.status]} {r.status}"
    if r.reason:
        head += f": {escape(r.reason)}"
    return head + ("  " + "  ".join(parts) if parts else "")


def _summary_table(results) -> Table:
    t = Table("#", "variant", "ctx", "PP t/s", "TG t/s", "VRAM peak", "status", "note", title="results")
    for i, r in enumerate(results, 1):
        m = r.metrics
        vram = (((r.monitor or {}).get("stats") or {}).get("vram_used_mib") or {}).get("peak")
        st = f"{_ICON[r.status]} {r.status}"
        t.add_row(str(i), r.variant, str(r.ctx or "-"), _fmt(m.get("pp_tps")), _fmt(m.get("tg_tps"), 2),
                  "-" if vram is None else f"{vram:.0f}", st, escape((r.reason or r.note or "")[:60]))
    return t


async def _run_planned(ctx: Ctx, planned, opts, title: str) -> list:
    from mse_llm import bench as bb
    from mse_llm import notify
    from mse_llm import results as rs

    env = bb.RunEnv()
    todo = [p for p in planned if p.action == "run"]
    out = []
    for i, p in enumerate(todo, 1):
        console.print(f"[bold]▶ [{i}/{len(todo)}] {escape(p.job.label)}[/]"
                      + (f"  [dim]# {escape(p.job.note)}[/]" if p.job.note else ""))
        r = await bb.run_job(ctx.cfg, p.job, opts, env)
        out.append(r)
        console.print("  " + _result_line(r))
        if opts.write:
            console.print(f"  [dim]→ {escape(rs.rel(r.path()) or '')}[/]")
    if len(out) > 1:
        console.print(_summary_table(out))
    if out:
        ok = sum(r.status == "ok" for r in out)
        notify.notify(f"{title}: {ok} of {len(out)} runs ok", ctx.cfg.defaults.notify)
    return out


def _print_plan(planned) -> None:
    skipped = [p for p in planned if p.action != "run"]
    for p in skipped:
        if p.action != "skip-filter":
            console.print(f"[dim]  – skip {escape(p.job.variant.name)}"
                          + (f" (ctx {p.job.ctx})" if p.job.ctx else "") + f": {escape(p.why)}[/]")
    n = len(planned) - len(skipped)
    console.print(f"[bold]{n} job(s) to run[/]" + (f", {len(skipped)} skipped" if skipped else ""))


def _execute(ctx: Ctx, planned, opts, title: str, dry_run: bool) -> None:
    todo = [p for p in planned if p.action == "run"]
    bad = [p for p in todo if not p.cmd.ok]
    if dry_run or ctx.json and dry_run:
        if ctx.json:
            click.echo(json.dumps([{**p.cmd.as_dict(), "action": p.action, "why": p.why, "key": p.key}
                                   for p in planned], indent=2))
        else:
            _print_commands(ctx, [p.cmd for p in todo])
            _print_plan(planned)
        sys.exit(1 if bad else 0)
    if ctx.json:
        raise click.ClickException("--json is for --dry-run, `results list` and `results show`")
    _print_plan(planned)
    if not todo:
        return
    try:
        results = asyncio.run(_run_planned(ctx, planned, opts, title))
    except KeyboardInterrupt:
        console.print("[yellow]interrupted: the running job was stopped and recorded as aborted[/]")
        sys.exit(130)
    if any(r.status == "failed" for r in results):
        sys.exit(1)
    if any(r.status == "aborted" for r in results):
        sys.exit(2)


# ---------------------------------------------------------------- bench / benchy

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


@main.command("bench")
@click.argument("model")
@_variant_options
@click.option("--ctx", "ctxs", type=int, multiple=True,
              help="Prompt size -p (repeatable: one run per size, bash _wrapper_PROMPTS). Default: [models].ctx.")
@click.option("--tg", type=int, multiple=True, help="Generated tokens -n (default: [defaults].bench_tg).")
@click.option("-r", "--repetitions", type=int, default=None,
              help="llama-bench -r (default: [defaults].bench_repetitions).")
@_run_options
@pass_ctx
def bench_cmd(ctx: Ctx, model, patterns, from_file, engine, dry_run, ctxs, tg, repetitions, **run) -> None:
    """llama-bench per variant, monitored, results into results/ (bash run_bench_params / run_BENCH_*)."""
    from mse_llm import bench as bb

    vs = _variants_for(ctx, model, patterns, from_file)
    jobs = [(bb.Job("bench", model, v, c, {"tg": list(tg), "repetitions": repetitions}, engine), "run", "")
            for c in (ctxs or [None]) for v in vs]
    _execute(ctx, bb.plan(ctx.cfg, jobs), _opts(ctx, **run), f"bench {model}", dry_run)


@main.command("benchy")
@click.argument("model")
@_variant_options
@click.option("--ctx", "ctx_", type=int, default=None, help="Server context -c (default: [models].ctx).")
@click.option("--pp", type=int, multiple=True,
              help="Prompt sizes (default: the context when pp_sync_ctx, else [defaults.benchy].pp).")
@click.option("--tg", type=int, multiple=True, help="Generated tokens (default: [defaults].bench_tg).")
@click.option("--depth", type=int, multiple=True, help="Context depths (default: [defaults.benchy].depth).")
@click.option("--runs", type=int, default=None)
@click.option("--concurrency", type=int, multiple=True, help="Concurrent requests (benchy --concurrency).")
@click.option("--port", type=int, default=None, help="Port of the background server (default: the engine's).")
@_run_options
@pass_ctx
def benchy_cmd(ctx: Ctx, model, patterns, from_file, engine, dry_run, ctx_, pp, tg, depth, runs, concurrency, port,
               **run) -> None:
    """llama-benchy against a background llama-server per variant, monitored, results into results/
    (bash run_benchy_params_with_server / run_BENCHY_WITH_SERVER_*)."""
    from mse_llm import bench as bb

    vs = _variants_for(ctx, model, patterns, from_file)
    params = {"pp": list(pp), "tg": list(tg), "depth": list(depth), "runs": runs, "concurrency": list(concurrency)}
    jobs = [(bb.Job("benchy", model, v, ctx_, dict(params), engine), "run", "") for v in vs]
    _execute(ctx, bb.plan(ctx.cfg, jobs, port=port), _opts(ctx, port=port, **run), f"benchy {model}", dry_run)


# ---------------------------------------------------------------- sweep

@main.group()
def sweep() -> None:
    """Sweeps from [sweeps.*] in the config (the bash run_BENCH_* / run_BENCHY_* functions)."""


def _sweep(ctx: Ctx, name: str):
    sw = ctx.cfg.sweeps.get(name)
    if sw is None:
        raise click.ClickException(f"unknown sweep {name!r}; known: {', '.join(ctx.cfg.sweeps) or '-'} "
                                   "(`mse-llm sweep list`)")
    return sw


@sweep.command("list")
@json_option
@pass_ctx
def sweep_list(ctx: Ctx) -> None:
    """Configured sweeps: tool, model, ctx, variants on / off / skip."""
    rows = []
    for name, sw in ctx.cfg.sweeps.items():
        es = sw.entries()
        rows.append({"name": name, "tool": sw.tool, "model": sw.model, "ctx": sw.ctx,
                     "on": sum(e.status == "on" for e in es), "off": sum(e.status == "off" for e in es),
                     "skip": sum(e.status == "skip" for e in es), "bash": sw.bash})
    if ctx.json:
        click.echo(json.dumps(rows, indent=2))
        return
    t = Table("sweep", "tool", "model", "ctx", "on", "off", "skip", "replaces")
    for r in rows:
        t.add_row(r["name"], r["tool"], r["model"], str(r["ctx"] or "-"), str(r["on"]), str(r["off"]),
                  str(r["skip"]), r["bash"] or "")
    console.print(t)
    console.print("[dim]on = runs; off = commented out in bash (--include-off); skip = freeze/skipped notes "
                  "(--force). `mse-llm sweep show NAME`, `mse-llm sweep run NAME --dry-run`[/]")


@sweep.command("show")
@click.argument("name")
@json_option
@pass_ctx
def sweep_show(ctx: Ctx, name) -> None:
    """Every variant of a sweep with its status, note and the last result for the current build."""
    from mse_llm import bench as bb

    sw = _sweep(ctx, name)
    jobs = bb.sweep_jobs(ctx.cfg, name, include_off=True, force=True)
    env = bb.RunEnv()
    from mse_llm import results as rs

    bid = env.build_for(ctx.cfg, sw.engine).get("id")
    last = rs.latest_by_key(bid)
    entries = {e.v: e for e in sw.entries()}
    rows = []
    for job, _, _ in jobs:
        cmd = job.command(ctx.cfg, save_result="x" if job.tool == "benchy" else None)
        e = entries[job.variant.name]
        prev = last.get(job.key(cmd)) or {}
        rows.append({"variant": job.variant.name, "ctx": job.ctx, "status": e.status, "tag": e.tag, "note": e.note,
                     "last": prev.get("status"), "pp_tps": prev.get("pp_tps"), "tg_tps": prev.get("tg_tps"),
                     "run_id": prev.get("run_id")})
    if ctx.json:
        click.echo(json.dumps({"name": name, **sw.model_dump(exclude={"variants", "annotations"}), "build": bid,
                               "entries": rows}, indent=2))
        return
    console.print(f"[bold]{name}[/]: {sw.tool} {sw.model}, ctx {sw.ctx}" + (f"  [dim](bash {sw.bash})[/]"
                                                                           if sw.bash else ""))
    if sw.notes:
        console.print(f"  [dim]{escape(sw.notes)}[/]")
    t = Table("variant", "ctx", "status", "note", "last run (this build)", "PP", "TG")
    style = {"on": "green", "off": "dim", "skip": "red"}
    for r in rows:
        t.add_row(r["variant"], str(r["ctx"] or "-"), f"[{style[r['status']]}]{r['status']}[/]",
                  escape(r["note"] or ""), r["last"] or "-", _fmt(r["pp_tps"]), _fmt(r["tg_tps"], 2))
    console.print(t)
    console.print(f"[dim]build {bid}[/]")


@sweep.command("run")
@click.argument("name")
@click.option("--ctx", "ctx_", type=int, default=None, help="Override the sweep's ctx (bench: -p, benchy: -c).")
@click.option("--include-off", is_flag=True, help="Also run the entries with status off (commented out in bash).")
@click.option("--match", "match", multiple=True, help="Only variants matching this glob (repeatable).")
@click.option("--resume", is_flag=True, help="Skip variants with an ok result for the same build.")
@click.option("--only-failed", is_flag=True, help="Run only variants whose last run on this build failed/aborted.")
@click.option("--port", type=int, default=None, help="benchy: port of the background server.")
@click.option("--dry-run", is_flag=True, help="Print the commands and what would be skipped, and stop.")
@json_option
@_run_options
@pass_ctx
def sweep_run(ctx: Ctx, name, ctx_, include_off, match, resume, only_failed, port, dry_run, force, **run) -> None:
    """Run a sweep: every variant with status on, one result per variant (bash run_BENCH_* / run_BENCHY_*)."""
    from mse_llm import bench as bb

    sw = _sweep(ctx, name)
    try:
        jobs = bb.sweep_jobs(ctx.cfg, name, ctx=ctx_, include_off=include_off, force=force, match=list(match))
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    planned = bb.plan(ctx.cfg, jobs, resume=resume, only_failed=only_failed, port=port)
    if not any(p.action == "run" for p in planned) and not dry_run:
        off = sum(1 for p in planned if "status off" in p.why)
        _print_plan(planned)
        hint = f"; {off} entr{'y has' if off == 1 else 'ies have'} status off (--include-off)" if off else ""
        console.print(f"[yellow]nothing to run in {name}{hint}[/]")
        return
    console.print(f"[bold]sweep {name}[/]: {sw.tool} {sw.model}, ctx {ctx_ or sw.ctx}")
    _execute(ctx, planned, _opts(ctx, port=port, force=force, **run), f"sweep {name}", dry_run)


# ---------------------------------------------------------------- results

@main.group()
def results() -> None:
    """The results store in results/ (index.jsonl + one JSON per run)."""


@results.command("list")
@click.option("--model", default=None, help="Model key (substring).")
@click.option("--sweep", "sweep_", default=None)
@click.option("--tool", type=click.Choice(["bench", "benchy", "eval"]), default=None)
@click.option("--status", type=click.Choice(["ok", "failed", "aborted", "skipped"]), default=None)
@click.option("-n", "--limit", type=int, default=30, show_default=True, help="Last N runs (0 = all).")
@json_option
@pass_ctx
def results_list(ctx: Ctx, model, sweep_, tool, status, limit) -> None:
    """Recorded runs, newest last."""
    from mse_llm import results as rs

    rows = [e for e in rs.read_index()
            if (not model or model.lower() in (e.get("model") or "").lower())
            and (not sweep_ or e.get("sweep") == sweep_) and (not tool or e.get("tool") == tool)
            and (not status or e.get("status") == status)]
    if limit:
        rows = rows[-limit:]
    if ctx.json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        console.print(f"no runs in {rs.results_root() / 'index.jsonl'}")
        return
    t = Table("run_id", "tool", "model", "variant", "ctx", "PP t/s", "TG t/s", "VRAM", "status", "build")
    for e in rows:
        t.add_row(e["run_id"], e["tool"], e["model"], e["variant"], str(e.get("ctx") or "-"), _fmt(e.get("pp_tps")),
                  _fmt(e.get("tg_tps"), 2), _fmt(e.get("vram_peak_mib"), 0),
                  f"{_ICON.get(e['status'], '')} {e['status']}", e.get("commit") or "-")
    console.print(t)


@results.command("show")
@click.argument("run_id")
@json_option
@pass_ctx
def results_show(ctx: Ctx, run_id) -> None:
    """One run: settings, tests, monitor summary, and the command to reproduce it (§9 item 1)."""
    from mse_llm import results as rs
    from mse_llm.runners import display_argv

    try:
        r = rs.find_run(run_id)
    except KeyError as e:
        raise click.ClickException(str(e.args[0])) from e
    if ctx.json:
        click.echo(r.model_dump_json(indent=2, by_alias=True, exclude_none=True))
        return
    console.print(f"[bold]{r.run_id}[/]  {_ICON[r.status]} {r.status}" + (f": {escape(r.reason)}" if r.reason else ""))
    b = r.build
    for k, val in (("tool", f"{r.tool} ({r.tool_version or '?'})"), ("model", f"{r.model}  {r.model_path}"),
                   ("variant", r.variant), ("ctx", r.ctx), ("params", json.dumps(r.params)),
                   ("sweep", r.sweep), ("note", r.note),
                   ("build", f"{b.get('id')}  {b.get('describe') or ''}  engine {b.get('engine')}"),
                   ("hw", f"{r.hw.get('hash')}  {r.hw.get('gpu') or ''}  driver {r.hw.get('driver') or '?'}"),
                   ("when", f"{r.timestamp}, {r.duration_s} s")):
        if val not in (None, ""):
            _plain(f"{k:9} {val}")
    if r.fit:
        f = r.fit
        _plain(f"{'fit':9} {f.get('verdict')}  GPU {f.get('gpu_mib')} MiB, host {f.get('host_mib')} MiB, "
               f"free {f.get('free_mib')}, left {f.get('left_mib')}")
    if r.tests:
        t = Table("test", "prompt", "gen", "depth", "conc", "PP t/s", "TG t/s", "TTFT ms")
        for x in r.tests:
            pp = _fmt(x.pp_tps) + (f" ± {x.pp_std:.1f}" if x.pp_std else "")
            tg = _fmt(x.tg_tps, 2) + (f" ± {x.tg_std:.2f}" if x.tg_std else "")
            t.add_row(x.kind, str(x.n_prompt), str(x.n_gen), str(x.depth), str(x.concurrency), pp, tg,
                      _fmt(x.ttft_ms, 0))
        console.print(t)
    if r.monitor:
        st = r.monitor.get("stats") or {}
        for key, label in (("vram_used_mib", "VRAM"), ("proc_shared_mib", "server shared"), ("ram_used_mib", "RAM"),
                           ("gpu_temp_c", "GPU °C")):
            s = st.get(key)
            if s:
                _plain(f"{label:14} peak {s['peak']:.0f}" + (f" (baseline {s['baseline']:.0f})"
                                                             if s.get("baseline") is not None else ""))
        for s in r.monitor.get("statuses") or []:
            _plain(f"  {s['level']:4} {s['text']}")
    console.print("[bold]reproduce:[/]")
    if r.server_argv:
        _plain("  server  " + display_argv(r.server_argv))
    _plain("  " + ("client  " if r.server_argv else "") + display_argv(r.argv))
    for k, p in r.logs.items():
        _plain(f"{'log ' + k:9} {p}")


__all__ = ["BENCH_FILTER", "bench_cmd", "benchy_cmd", "results", "sweep"]
