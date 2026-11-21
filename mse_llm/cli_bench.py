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


allow_low_power_option = click.option(
    "--allow-low-power", is_flag=True,
    help="Run on battery or under a quiet / power-saving plan anyway (results are slower and vary run to run; "
         "env MSE_LLM_ALLOW_LOW_POWER=1 does the same).")


def power_gate(allow: bool) -> None:
    """Stop before any benchmark when the numbers would be useless: on battery or under a quiet / power-saving
    plan (measured: gemma QAT TG 6-36 t/s for one config under Legion Quiet Mode)."""
    import os

    from mse_llm import hw

    why = hw.bench_power_problem()
    if not why:
        return
    if allow or os.environ.get(hw.ALLOW_LOW_POWER_ENV, "").strip() not in ("", "0"):
        console.print(f"[yellow]![/] {escape(why)}: running anyway (--allow-low-power); results are not "
                      "comparable with runs under a performance plan")
        return
    raise click.ClickException(
        f"{why}: benchmark results would be slower and vary run to run. Plug in AC / switch to a performance "
        "plan (Lenovo Vantage or Fn+Q on a Legion; `powercfg /list`), or pass --allow-low-power.")


def _run_options(f):
    for opt in reversed([
        click.option("--no-fit-check", is_flag=True, help="Don't run llama-fit-params before each job."),
        allow_low_power_option,
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


def _opts(ctx: Ctx, *, no_fit_check, force, output, no_save, no_monitor, no_guard, mem_cap, port=None,
          allow_low_power=False):
    from mse_llm.bench import RunOpts
    from mse_llm.proc import OutputFilter

    mode = output or ctx.cfg.defaults.output
    filt = OutputFilter(mode, BENCH_FILTER)

    def sink(line: str) -> None:
        if filt(line) and (mode == "all" or not BENCH_QUIET.search(line)):
            _plain("  │ " + line)

    return RunOpts(fit_check=not no_fit_check, force=force, monitor=not no_monitor, guard=not no_guard,
                   mem_cap_mib=mem_cap if mem_cap is not None else ctx.cfg.monitor.job_memory_cap_mib, port=port,
                   write=not no_save, sink=sink, say=lambda s: console.print("  " + s),
                   allow_low_power=allow_low_power)


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
    sp = spec_text(r.spec)
    if sp:
        parts.append(sp)
    if r.duration_s is not None:
        parts.append(f"{r.duration_s:.0f} s")
    head = f"{_ICON[r.status]} {r.status}"
    if r.reason:
        head += f": {escape(r.reason)}"
    return head + ("  " + "  ".join(parts) if parts else "")


def spec_text(s: dict | None, positions: int = 6) -> str:
    """`acc 71.2 % · len 2.85 · per pos .85/.70/.55` for a speculative-decoding run."""
    if not s:
        return ""
    if not s.get("drafted"):
        return "acc - (nothing drafted)"
    out = f"acc {s['acceptance_rate'] * 100:.1f} %"
    if s.get("mean_accept_len"):
        out += f" · len {s['mean_accept_len']:.2f}"
    rates = s.get("accept_rate_per_pos") or []
    if rates:
        out += " · per pos " + "/".join(f"{x:.2f}".lstrip("0") or "0" for x in rates[:positions])
    if s.get("prompts") and s["prompts"] != "prose":
        out += f" ({s['prompts']})"
    return out


def _summary_table(results) -> Table:
    spec = any(r.spec for r in results)
    cols = ["#", "variant", "ctx", "PP t/s", "TG t/s"] + (["acceptance"] if spec else []) + ["VRAM peak", "status",
                                                                                          "note"]
    t = Table(*cols, title="results")
    for i, r in enumerate(results, 1):
        m = r.metrics
        vram = (((r.monitor or {}).get("stats") or {}).get("vram_used_mib") or {}).get("peak")
        st = f"{_ICON[r.status]} {r.status}"
        row = [str(i), r.variant + (f" ({r.params['prompts']})" if r.params.get("prompts") else ""),
               str(r.ctx or "-"), _fmt(m.get("pp_tps")), _fmt(m.get("tg_tps"), 2)]
        if spec:
            row.append(spec_text(r.spec, 4))
        row += ["-" if vram is None else f"{vram:.0f}", st, escape((r.reason or r.note or "")[:60])]
        t.add_row(*row)
    return t


async def _run_planned(ctx: Ctx, planned, opts, title: str, notify_end: bool = True) -> list:
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
    if out and notify_end:
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
    power_gate(opts.allow_low_power)
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
@click.option("--prompts", "prompt_sets", multiple=True,
              help="Prompt set: prose (benchy's book, default) | code (llama.cpp sources) | wikitext2 | FILE. "
                   "Repeatable: one run per set (speculative-decoding acceptance depends on the content).")
@click.option("--warmup", type=int, default=None, metavar="TOKENS",
              help="MoE cache variants (mc*): generate this many tokens before benchy, so the cache is full.")
@_run_options
@pass_ctx
def benchy_cmd(ctx: Ctx, model, patterns, from_file, engine, dry_run, ctx_, pp, tg, depth, runs, concurrency, port,
               prompt_sets, warmup, **run) -> None:
    """llama-benchy against a background llama-server per variant, monitored, results into results/
    (bash run_benchy_params_with_server / run_BENCHY_WITH_SERVER_*)."""
    from mse_llm import bench as bb

    vs = _variants_for(ctx, model, patterns, from_file)
    params = {"pp": list(pp), "tg": list(tg), "depth": list(depth), "runs": runs, "concurrency": list(concurrency),
              "warmup": warmup}
    sets = [None if s == "prose" else s for s in prompt_sets] or [None]
    jobs = [(bb.Job("benchy", model, v, ctx_, dict(params) if s is None else {**params, "prompts": s}, engine),
             "run", "") for v in vs for s in sets]
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
                     "skip": sum(e.status == "skip" for e in es), "bash": sw.bash, "stages": len(sw.stages)})
    if ctx.json:
        click.echo(json.dumps(rows, indent=2))
        return
    t = Table("sweep", "tool", "model", "ctx", "on", "off", "skip", "replaces")
    for r in rows:
        on = f"{r['stages']} stages" if r["stages"] else str(r["on"])
        t.add_row(r["name"], r["tool"], r["model"], str(r["ctx"] or "-"), on, str(r["off"]),
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
    if sw.stages:
        _show_staged(ctx, name, sw)
        return
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


def _show_staged(ctx: Ctx, name: str, sw) -> None:
    """Stages with their patterns, and the winners of the last staged run on this build (if any)."""
    from mse_llm import bench as bb
    from mse_llm import staged as st

    env = bb.RunEnv()
    bid = env.build_for(ctx.cfg, sw.engine).get("id")
    p = st.summary_path(name, bid, env.hw_info().get("hash"))
    summ = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    winners = {s["index"]: s for s in (summ or {}).get("stages", [])}
    if ctx.json:
        click.echo(json.dumps({"name": name, **sw.model_dump(exclude={"variants", "annotations"}), "build": bid,
                               "last": summ}, indent=2))
        return
    console.print(f"[bold]{name}[/]: staged {sw.tool} {sw.model}, ctx {sw.ctx}, goal {sw.goal}"
                  + (f", warm-up {sw.warmup} tokens" if sw.warmup else ""))
    t = Table("stage", "patterns", "variants", "winner (this build)")
    for i, pats in enumerate(sw.stages):
        w = winners.get(i) or {}
        t.add_row(str(i), escape("\n".join(pats)), str(len(w.get("variants") or [])) if w else "-",
                  escape(w.get("winner") or w.get("unresolved") or "-"))
    console.print(t)
    if summ and summ.get("winner"):
        console.print(f"sweep winner: [bold]{escape(summ['winner'])}[/]  [dim]({escape(str(p))})[/]")
    console.print(f"[dim]build {bid}; `mse-llm sweep run {name} --dry-run` resolves the stages[/]")


@sweep.command("run")
@click.argument("name")
@click.option("--ctx", "ctx_", type=int, default=None, help="Override the sweep's ctx (bench: -p, benchy: -c).")
@click.option("--include-off", is_flag=True, help="Also run the entries with status off (commented out in bash).")
@click.option("--match", "match", multiple=True, help="Only variants matching this glob (repeatable).")
@click.option("--resume", is_flag=True, help="Skip variants with an ok result for the same build.")
@click.option("--only-failed", is_flag=True, help="Run only variants whose last run on this build failed/aborted.")
@click.option("--port", type=int, default=None, help="benchy: port of the background server.")
@click.option("--dry-run", is_flag=True, help="Print the commands and what would be skipped, and stop.")
@click.option("--prompts", "prompt_sets", multiple=True,
              help="benchy: prompt sets (prose | code | wikitext2 | FILE) instead of the sweep's prompt_sets.")
@click.option("--with-kld", is_flag=True,
              help="First a KLD run per distinct K/V combination of the sweep (quality kld; reference made if "
                   "missing).")
@click.option("--kld-ctx", type=int, default=None, help="--with-kld: KLD context (default 8192).")
@click.option("--yes", "-y", is_flag=True, help="--with-kld: don't ask before creating a reference.")
@click.option("--until-stage", type=int, default=None, metavar="N",
              help="Staged sweep: stop after stage N (0-based).")
@json_option
@_run_options
@pass_ctx
def sweep_run(ctx: Ctx, name, ctx_, include_off, match, resume, only_failed, port, dry_run, prompt_sets, with_kld,
              kld_ctx, yes, until_stage, force, **run) -> None:
    """Run a sweep: every variant with status on, one result per variant (bash run_BENCH_* / run_BENCHY_*).
    A sweep with `stages` runs stage by stage; each stage's winner fills the `{best}` / `{winner}` of the next."""
    from mse_llm import bench as bb

    sw = _sweep(ctx, name)
    if sw.stages:
        if with_kld or only_failed or include_off:
            raise click.ClickException("staged sweeps take --resume, --match, --until-stage, --ctx, --prompts, "
                                       "--dry-run (not --with-kld / --only-failed / --include-off)")
        _run_staged(ctx, name, sw, ctx_=ctx_, match=list(match), resume=resume, port=port, dry_run=dry_run,
                    prompt_sets=list(prompt_sets) or None, until=until_stage, force=force, run=run)
        return
    try:
        jobs = bb.sweep_jobs(ctx.cfg, name, ctx=ctx_, include_off=include_off, force=force, match=list(match),
                             prompt_sets=list(prompt_sets) or None)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    planned = bb.plan(ctx.cfg, jobs, resume=resume, only_failed=only_failed, port=port)
    if with_kld:
        from mse_llm import cli_quality as cq

        todo = [p.job.variant for p in planned if p.action == "run"]
        cq.sweep_kld(ctx, sw.model, todo, name, sw.engine, kld_ctx=kld_ctx, yes=yes, dry_run=dry_run,
                     output=run.get("output"), no_guard=run.get("no_guard", False))
    if not any(p.action == "run" for p in planned) and not dry_run:
        off = sum(1 for p in planned if "status off" in p.why)
        _print_plan(planned)
        hint = f"; {off} entr{'y has' if off == 1 else 'ies have'} status off (--include-off)" if off else ""
        console.print(f"[yellow]nothing to run in {name}{hint}[/]")
        return
    console.print(f"[bold]sweep {name}[/]: {sw.tool} {sw.model}, ctx {ctx_ or sw.ctx}")
    _execute(ctx, planned, _opts(ctx, port=port, force=force, **run), f"sweep {name}", dry_run)


def _staged_runs(planned, results_by_key, last) -> list:
    """StageRun per planned job: a result of this invocation, else the stored one (--resume / dry run)."""
    from mse_llm import staged as st

    out = []
    for p in planned:
        if p.action == "skip-filter":
            continue
        r = results_by_key.get(p.key)
        if r is not None:
            out.append(st.run_of(r))
        elif p.key in last and p.action != "run":
            out.append(st.run_of(last[p.key], reused=True))
        elif p.action == "skip-status":
            out.append(st.StageRun(p.job.variant.name, "skipped", reason=p.why))
    return out


def _run_staged(ctx: Ctx, name: str, sw, *, ctx_, match, resume, port, dry_run, prompt_sets, until, force,
                run) -> None:
    """Staged sweep (§4.4): stage N's patterns are resolved with the winners of stages 0..N-1, then run."""
    from mse_llm import bench as bb
    from mse_llm import notify
    from mse_llm import results as rs
    from mse_llm import staged as st
    from mse_llm.config import SweepEntry

    opts = _opts(ctx, port=port, force=force, **run)
    env = bb.RunEnv()
    build = env.build_for(ctx.cfg, sw.engine).get("id")
    tuned = st.load_tuned(sw.model)
    winners: list = []
    states: list = []
    ran: dict = {}                 # run key -> RunResult of this invocation
    last_stage = len(sw.stages) - 1 if until is None else min(until, len(sw.stages) - 1)
    say = (lambda *a, **k: None) if ctx.json else console.print
    say(f"[bold]staged sweep {name}[/]: {sw.tool} {sw.model}, ctx {ctx_ or sw.ctx}, "
        f"{len(sw.stages)} stages (running 0..{last_stage}), goal {sw.goal}"
        + ("  [dim](dry run: later stages resolve from stored results)[/]" if dry_run else ""))
    gated = False
    for i, patterns in enumerate(sw.stages[: last_stage + 1]):
        state = st.StageState(i, list(patterns))
        states.append(state)
        try:
            names = [n for pat in patterns for n in st.resolve(pat, winners, tuned)]
        except st.Unresolved as e:
            state.unresolved = str(e)
            say(f"[bold]stage {i}[/]: [yellow]{escape(str(e))}[/]")
            break
        except st.StageError as e:
            raise click.ClickException(f"stage {i}: {e}") from e
        state.variants = list(dict.fromkeys(names))
        say(f"[bold]stage {i}[/]: {len(state.variants)} variant(s) from "
            + ", ".join(f"[cyan]{escape(p)}[/]" for p in patterns))
        entries = [SweepEntry(v=n) for n in state.variants]
        jobs = bb.sweep_jobs(ctx.cfg, name, ctx=ctx_, force=force, match=match, prompt_sets=prompt_sets,
                             entries=entries, stage={"index": i, "patterns": list(patterns)})
        # a stage's winner needs its runs: a stored ok run of this build counts (always in a dry run)
        planned = bb.plan(ctx.cfg, jobs, resume=resume or dry_run, port=port, env=env)
        for p in planned:          # a variant an earlier stage of this invocation already ran (e.g. its winner)
            if p.action == "run" and p.key in ran:
                p.action, p.why = "skip-resume", f"ran in this sweep ({ran[p.key].run_id})"
        last = rs.latest_by_key(build)
        done: dict = {k: r for k, r in ran.items() if any(p.key == k for p in planned)}
        if dry_run:
            if not ctx.json:
                _print_commands(ctx, [p.cmd for p in planned if p.action == "run"])
                _print_plan(planned)
        else:
            _print_plan(planned)
            todo = [p for p in planned if p.action == "run"]
            bad = [p for p in todo if not p.cmd.ok]
            if bad:
                raise click.ClickException(f"stage {i}: {bad[0].job.variant.name}: "
                                           + "; ".join(bad[0].cmd.errors + (bad[0].cmd.server.errors
                                                                            if bad[0].cmd.server else [])))
            if todo:
                if not gated:
                    power_gate(opts.allow_low_power)
                    gated = True
                try:
                    out = asyncio.run(_run_planned(ctx, todo, opts, f"sweep {name} stage {i}", notify_end=False))
                except KeyboardInterrupt:
                    console.print("[yellow]interrupted: the running job was stopped and recorded as aborted[/]")
                    sys.exit(130)
                done.update({p.key: r for p, r in zip(todo, out, strict=True)})
                ran.update(done)
        state.runs = _staged_runs(planned, done, last)
        w = st.pick_winner(state.runs, sw.goal)
        if w is None:
            msg = "no ok run" + (" stored yet" if dry_run else "") + f" in stage {i}: later stages can't resolve"
            say(f"[yellow]{msg}[/]")
            state.unresolved = msg
            break
        state.winner = w.variant
        winners.append(bb.decode_variant(w.variant))
        say(f"  [green]stage {i} winner[/]: [bold]{escape(w.variant)}[/]  TG {_fmt(w.tg, 2)}, PP {_fmt(w.pp)}"
            + (f", hit rate {w.hit_rate * 100:.1f} %" if w.hit_rate is not None else "")
            + ("  [dim](stored run)[/]" if w.reused else ""))
    if ctx.json:
        click.echo(json.dumps([s.as_dict() for s in states], indent=2))
    allruns = [r for s in states for r in s.runs]
    best = st.pick_winner(allruns, sw.goal)
    if not dry_run and allruns and opts.write:
        path = st.write_summary(name, sw.model, sw.goal, states, build, env.hw_info().get("hash"))
        say(f"[dim]→ {escape(rs.rel(path) or str(path))}[/]")
    if best is not None:
        say(f"[bold green]sweep winner[/] ({sw.goal}): [bold]{escape(best.variant)}[/]  TG {_fmt(best.tg, 2)}, "
            f"PP {_fmt(best.pp)}")
        say(f"[dim]report: mse-llm report moe-cache --sweep {name} --chart[/]")
    if not dry_run and any(not r.reused for r in allruns):
        notify.notify(f"sweep {name}: {sum(r.status == 'ok' for r in allruns)} of {len(allruns)} runs ok",
                      ctx.cfg.defaults.notify)


# ---------------------------------------------------------------- results

@main.group()
def results() -> None:
    """The results store in results/ (index.jsonl + one JSON per run)."""


@results.command("list")
@click.option("--model", default=None, help="Model key (substring).")
@click.option("--sweep", "sweep_", default=None)
@click.option("--tool", type=click.Choice(["bench", "benchy", "eval", "quality"]), default=None)
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
    for k, val in (("tool", r.tool + (f" ({r.tool_version})" if r.tool_version else "")),
                   ("model", f"{r.model}  {r.model_path}"),
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
    if r.spec:
        _plain(f"{'spec':9} {r.spec.get('type')} n_max {r.spec.get('n_max')}: {spec_text(r.spec, 12)}  "
               f"[{r.spec.get('source')}]")
    if r.quality:
        q = r.quality
        line = f"{q.get('mode')}  corpus {(q.get('corpus') or {}).get('name')}, {q.get('chunks')} chunks"
        if q.get("mean_kld") is not None:
            line += (f"; mean KLD {q['mean_kld']:.6f}, 99 % {q.get('kld_p99')}, same top p "
                     f"{(q.get('same_top_p') or 0) * 100:.2f} %, PPL ratio {q.get('ppl_ratio')}; "
                     f"{q.get('verdict') or ''}")
        elif q.get("ppl") is not None:
            line += f"; PPL {q['ppl']}"
        _plain(f"{'quality':9} {line}")
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
