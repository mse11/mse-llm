"""`report chart|table|prompt` and `results compare|export` (M8, §6, §8, parity F10/F11)."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.markup import escape
from rich.table import Table

from mse_llm.cli import Ctx, _plain, console, json_option, main, pass_ctx
from mse_llm.cli_bench import _fmt, results


def _select_options(f):
    for opt in reversed([
        click.option("--sweep", "sweep_", default=None, help="Runs of this sweep (in its entry order)."),
        click.option("--model", default=None, help="Model key (substring)."),
        click.option("--tool", type=click.Choice(["bench", "benchy", "quality"]), default=None),
        click.option("--build", default=None, help="Build id or llama.cpp commit (prefix)."),
        click.option("--runs", "run_ids", multiple=True, help="Run ids (repeatable; overrides the filters)."),
        click.option("--last", type=int, default=None, help="Only the last N runs."),
        click.option("--all-runs", is_flag=True,
                     help="Every run, not only the newest per (tool, model, variant, ctx, params)."),
    ]):
        f = opt(f)
    return f


def _select(ctx: Ctx, sweep_, model, tool, build, run_ids, last, all_runs, statuses=("ok",)):
    from mse_llm.report.data import select

    order = None
    if sweep_:
        sw = ctx.cfg.sweeps.get(sweep_)
        if sw is not None:
            order = [e.v for e in sw.entries()]
    try:
        sel = select(run_ids=list(run_ids) or None, sweep=sweep_, model=model, tool=tool, build=build,
                     statuses=statuses, latest=not all_runs, last=last, order=order)
    except KeyError as e:
        raise click.ClickException(str(e.args[0])) from e
    if not sel.runs:
        raise click.ClickException("no matching runs (`mse-llm results list` shows what is stored)")
    return sel


@main.group()
def report() -> None:
    """Charts, tables and analyst prompts from the results store (bash run_dump_params_wrapper_*)."""


@report.command("chart")
@_select_options
@click.option("--legacy-log", "legacy", type=click.Path(exists=True, dir_okay=False), default=None,
              help="An old bash console log (.txt, or the .html page it was pasted into) instead of results.")
@click.option("--type", "bench_type", type=click.Choice(["auto", "llama-bench", "llama-benchy"]), default="auto",
              help="--legacy-log: bench type for the title (bash auto-detects from the markers).")
@click.option("--title", default=None, help="Chart title (default: the sweep / model names).")
@click.option("-o", "--out", type=click.Path(dir_okay=False), default=None,
              help="PNG path (default: results/reports/<name>/<name>.png).")
@json_option
@pass_ctx
def report_chart(ctx: Ctx, sweep_, model, tool, build, run_ids, last, all_runs, legacy, bench_type, title,
                 out) -> None:
    """The bash 5-axis chart: duration, PP, TG, peak VRAM, shared memory per configuration
    (bash run_dump_params_wrapper_PLOTER)."""
    from mse_llm.report import reports_dir
    from mse_llm.report.data import chart_row
    from mse_llm.report.plot import chart

    skipped: list[str] = []
    if legacy:
        from mse_llm.parsers.legacy_log import parse_legacy, read_log

        bt, rows = parse_legacy(read_log(legacy), bench_type)
        if not rows:
            raise click.ClickException(f"{legacy}: no benchmark blocks (needs TESTING: / BENCHY: blocks with a "
                                       "`| ppN |` and a `| tgN |` table row)")
        name = Path(legacy).stem
        title = title or "Benchmarks Performance Analysis"
        placeholders = True
    else:
        sel = _select(ctx, sweep_, model, tool, build, run_ids, last, all_runs)
        rows = [chart_row(r) for r in sel.runs]
        skipped = [f"{r.variant} ({r.status})" for r in sel.runs if not (r.metrics.get("pp_tps") and
                                                                         r.metrics.get("tg_tps"))]
        rows = [x for x in rows if x.pp and x.tg]
        if not rows:
            raise click.ClickException("no run with both PP and TG to chart")
        bt, name, placeholders = sel.bench_type, sweep_ or sel.title, False
        title = title or sel.title
    path = Path(out) if out else reports_dir(name) / f"{Path(name).name}.png"
    path = chart(rows, path, title=title, bench_type=bt, bash_placeholders=placeholders)
    if ctx.json:
        click.echo(json.dumps({"png": str(path), "bars": len(rows), "bench_type": bt, "skipped": skipped}, indent=2))
        return
    console.print(f"[green]✓[/] {len(rows)} configuration(s) → [bold]{escape(str(path))}[/]")
    for s in skipped:
        console.print(f"  [dim]not charted (no PP/TG): {escape(s)}[/]")


@report.command("table")
@_select_options
@click.option("--status", "statuses", multiple=True, type=click.Choice(["ok", "aborted", "failed", "skipped"]),
              help="Statuses to include (default: all).")
@click.option("--sort", type=click.Choice(["order", "tg", "pp", "vram", "ttft"]), default="order", show_default=True)
@click.option("--tests", "per_test", is_flag=True, help="One row per test (pp / tg line, benchy prompt × depth).")
@click.option("--format", "fmt", type=click.Choice(["table", "csv", "md"]), default="table", show_default=True)
@json_option
@pass_ctx
def report_table(ctx: Ctx, sweep_, model, tool, build, run_ids, last, all_runs, statuses, sort, per_test,
                 fmt) -> None:
    """Runs side by side, best PP / TG marked (csv / md for pasting elsewhere)."""
    from mse_llm.report.data import STATUSES, best, table_row, test_rows, to_csv

    sel = _select(ctx, sweep_, model, tool, build, run_ids, last, all_runs, tuple(statuses) or STATUSES)
    rows = [t for r in sel.runs for t in test_rows(r)] if per_test else [table_row(r) for r in sel.runs]
    keys = {"tg": "tg_tps", "pp": "pp_tps", "vram": "vram_peak_mib", "ttft": "ttft_ms"}
    if sort != "order":
        k = keys[sort]
        rows.sort(key=lambda r: (r.get(k) is None, -(r.get(k) or 0) if sort in ("tg", "pp") else (r.get(k) or 0)))
    if ctx.json:
        click.echo(json.dumps(rows, indent=2))
        return
    if fmt == "csv":
        click.echo(to_csv(rows), nl=False)
        return
    if per_test:
        cols = ["run_id", "variant", "ctx", "kind", "n_prompt", "n_gen", "depth", "concurrency", "pp_tps", "tg_tps",
                "ttft_ms", "status"]
    else:
        cols = ["variant", "ctx", "pp_tps", "tg_tps", "ttft_ms", "vram_peak_mib", "server_shared_mib", "spill_mib",
                "duration_s", "status", "note"]
        extra = [c for c in ("acceptance", "mean_kld", "same_top_p", "hit_rate")
                 if any(r.get(c) is not None for r in rows)]
        cols[4:4] = extra
    best_pp, best_tg = (None, None) if per_test else (best(rows, "pp_tps"), best(rows, "tg_tps"))

    def cell(r, c):
        v = r.get(c)
        if c in ("pp_tps", "tg_tps", "ttft_ms"):
            s = _fmt(v, 2 if c == "tg_tps" else 1)
            if (c == "pp_tps" and r.get("run_id") == best_pp) or (c == "tg_tps" and r.get("run_id") == best_tg):
                s += " ★"
            return s
        if c in ("vram_peak_mib", "server_shared_mib", "spill_mib", "duration_s"):
            return _fmt(v, 0)
        if c in ("acceptance", "same_top_p", "hit_rate"):
            return "" if v is None else f"{v * 100:.1f} %"
        if c == "mean_kld":
            return "" if v is None else f"{v:.5f}"
        return "" if v is None else str(v)

    if fmt == "md":
        click.echo("| " + " | ".join(cols) + " |")
        click.echo("|" + "---|" * len(cols))
        for r in rows:
            click.echo("| " + " | ".join(cell(r, c).replace("|", "/") for c in cols) + " |")
        return
    t = Table(*cols, title=sel.title)
    for r in rows:
        t.add_row(*(escape(cell(r, c)) for c in cols))
    console.print(t)
    if not per_test:
        console.print("[dim]★ = best PP / TG of the ok runs. MiB columns: peak; spill = shared memory beyond "
                      "fit's host buffers (> ~100 = a real spill).[/]")


@report.command("spec")
@_select_options
@click.option("--chart", "with_chart", is_flag=True,
              help="Also write the PNG (TG + acceptance vs n_max; per position).")
@click.option("-o", "--out", type=click.Path(dir_okay=False), default=None,
              help="PNG path (default: results/reports/<name>/<name>-spec.png).")
@json_option
@pass_ctx
def report_spec(ctx: Ctx, sweep_, model, tool, build, run_ids, last, all_runs, with_chart, out) -> None:
    """Speculative decoding: acceptance rate, mean accepted length and acceptance per draft position next to TG,
    with the TG of the same variant without spec (flagged when spec is slower)."""
    from mse_llm.cli_bench import spec_text
    from mse_llm.report import reports_dir
    from mse_llm.report.spec import chart, spec_rows

    sel = _select(ctx, sweep_, model, tool or "benchy", build, run_ids, last, all_runs)
    rows = spec_rows(sel.runs)
    if not rows:
        raise click.ClickException("no speculative-decoding runs among the selected ones (benchy runs of sd* "
                                   "variants record acceptance)")
    png = None
    if with_chart:
        name = sweep_ or sel.title
        png = chart(rows, Path(out) if out else reports_dir(name) / f"{Path(name).name}-spec.png", title=sel.title)
    if ctx.json:
        click.echo(json.dumps({"rows": rows, "png": str(png) if png else None}, indent=2))
        return
    t = Table("variant", "prompts", "TG t/s", "no-spec TG", "speedup", "acceptance", "temp", title=sel.title)
    for r in rows:
        sp = spec_text({"drafted": r["drafted"], "acceptance_rate": r["acceptance_rate"],
                        "mean_accept_len": r["mean_accept_len"], "accept_rate_per_pos": r["accept_rate_per_pos"]})
        speed = "-" if r["speedup"] is None else f"x{r['speedup']:.2f}"
        if r["slower_than_baseline"]:
            speed = f"[red]{speed} slower[/]"
        t.add_row(escape(r["variant"]), r["prompts"], _fmt(r["tg_tps"], 2), _fmt(r["baseline_tg"], 2), speed,
                  escape(sp), "-" if r["temp"] is None else str(r["temp"]))
    console.print(t)
    console.print("[dim]acceptance = accepted / drafted; len = tokens per target-model step (1 + accepted / "
                  "steps); per pos = accepted at draft position N / steps (under ~0.3 a longer draft stops "
                  "paying off).[/]")
    if png:
        console.print(f"[green]✓[/] chart → [bold]{escape(str(png))}[/]")


@report.command("moe-cache")
@_select_options
@click.option("--chart", "with_chart", is_flag=True,
              help="Also write the PNG (TG, PP, duration, VRAM, hit rate, uploaded MiB per variant).")
@click.option("-o", "--out", type=click.Path(dir_okay=False), default=None,
              help="PNG path (default: results/reports/<name>/<name>-moe-cache.png).")
@json_option
@pass_ctx
def report_moe_cache(ctx: Ctx, sweep_, model, tool, build, run_ids, last, all_runs, with_chart, out) -> None:
    """MoE expert cache (§4.4): TG, PP at a short and a long prompt, peak VRAM, hit rate, uploaded MiB and PCIe per
    variant; the runs without a cache are the baselines (grey in the chart)."""
    from mse_llm.report import reports_dir
    from mse_llm.report.moe_cache import cache_rows, chart

    sel = _select(ctx, sweep_, model, tool or "benchy", build, run_ids, last, all_runs,
                  statuses=("ok", "failed", "aborted"))
    rows = cache_rows(sel.runs)
    if not any(r["mc"] is not None or r["hit_rate"] is not None for r in rows):
        raise click.ClickException("no MoE cache runs among the selected ones (benchy runs of mc* variants)")
    png = None
    if with_chart:
        name = sweep_ or sel.title
        png = chart(rows, Path(out) if out else reports_dir(name) / f"{Path(name).name}-moe-cache.png",
                    title=sel.title)
    if ctx.json:
        click.echo(json.dumps({"rows": rows, "png": str(png) if png else None}, indent=2))
        return
    best_tg = max((r["tg_tps"] for r in rows if r["status"] == "ok" and r["tg_tps"]), default=None)
    base_tg = max((r["tg_tps"] for r in rows if r["status"] == "ok" and r["mc"] is None and r["tg_tps"]),
                  default=None)
    t = Table("stage", "variant", "TG t/s", "PP short", "PP long", "VRAM peak", "cache MiB", "hit rate", "uploaded",
              "PCIe RX", "status", title=sel.title)
    for r in rows:
        tg = _fmt(r["tg_tps"], 2) + (" ★" if r["tg_tps"] is not None and r["tg_tps"] == best_tg else "")
        if r["mc"] is not None and base_tg and r["tg_tps"] and r["tg_tps"] < base_tg:
            tg = f"[red]{tg}[/]"
        st = r["status"] if not r["invalid_config"] else "invalid config"
        t.add_row("-" if r["stage"] is None else str(r["stage"]), escape(r["variant"]), tg,
                  _fmt(r["pp_short_tps"]), _fmt(r["pp_long_tps"]), _fmt(r["vram_peak_mib"], 0),
                  _fmt(r["cache_mib"], 0), "-" if r["hit_rate"] is None else f"{r['hit_rate'] * 100:.1f} %",
                  _fmt(r["uploaded_mib"], 0), _fmt(r["pcie_rx_mean_mbs"], 0), escape(st))
    console.print(t)
    console.print("[dim]★ best TG; red = a cache run slower than the best run without a cache. hit rate / uploaded: "
                  "small batches (ubatch <= 8), whole server run incl. the warm-up; PCIe RX: mean MB/s while it "
                  "ran.[/]")
    for r in rows:
        if r["mc"] is not None and r["hit_rate"] is None and r["status"] == "ok":
            console.print(f"[yellow]![/] {escape(r['variant'])}: no hit / miss counters (the server did not exit "
                          "cleanly)")
    if png:
        console.print(f"[green]✓[/] chart → [bold]{escape(str(png))}[/]")


@report.command("prompt")
@_select_options
@click.option("--title", default=None)
@click.option("--chart", "with_chart", is_flag=True, help="Also draw the chart and name it in the prompt.")
@click.option("--no-tests", is_flag=True, help="Leave out the per-test CSV.")
@click.option("--status", "statuses", multiple=True, type=click.Choice(["ok", "aborted", "failed", "skipped"]),
              help="Statuses to include (default: all, so the analyst sees the aborted / skipped ones too).")
@click.option("-o", "--out", type=click.Path(dir_okay=False), default=None, help="Write to a file (default: stdout).")
@pass_ctx
def report_prompt(ctx: Ctx, sweep_, model, tool, build, run_ids, last, all_runs, title, with_chart, no_tests,
                  statuses, out) -> None:
    """AI analyst prompt with the result data embedded as CSV (bash run_dump_params_wrapper_AI_PROMPT /
    _COMMON / _EXTRAS / _PROMPTS)."""
    from mse_llm.report import render_prompt, reports_dir
    from mse_llm.report.data import STATUSES, chart_row

    sel = _select(ctx, sweep_, model, tool, build, run_ids, last, all_runs, tuple(statuses) or STATUSES)
    png = None
    if with_chart:
        from mse_llm.report.plot import chart

        rows = [chart_row(r) for r in sel.runs if r.metrics.get("pp_tps") and r.metrics.get("tg_tps")]
        if rows:
            name = sweep_ or sel.title
            png = str(chart(rows, reports_dir(name) / f"{Path(name).name}.png", title=title or sel.title,
                            bench_type=sel.bench_type))
    text = render_prompt(sel, title=title, chart=png, with_tests=not no_tests)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        console.print(f"[green]✓[/] prompt ({len(text)} chars, {len(sel.runs)} runs) → {escape(out)}")
    else:
        click.echo(text, nl=False)


# ---------------------------------------------------------------- results compare / export

@results.command("compare")
@click.argument("run_a")
@click.argument("run_b")
@json_option
@pass_ctx
def results_compare(ctx: Ctx, run_a, run_b) -> None:
    """Two runs side by side: what differs in the settings, and the PP / TG / memory deltas."""
    from mse_llm import results as rs
    from mse_llm.report.data import compare

    try:
        a, b = rs.find_run(run_a), rs.find_run(run_b)
    except KeyError as e:
        raise click.ClickException(str(e.args[0])) from e
    rows = compare(a, b)
    if ctx.json:
        click.echo(json.dumps([{"field": f, "a": x, "b": y, "delta": d} for f, x, y, d in rows], indent=2,
                              default=str))
        return
    t = Table("", a.run_id, b.run_id, "B vs A")
    for f, x, y, d in rows:
        t.add_row(f, escape(_cell(x)), escape(_cell(y)), d)
    console.print(t)


def _cell(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.2f}"
    if isinstance(v, dict):
        return json.dumps(v)
    return str(v)


@results.command("export")
@_select_options
@click.option("--status", "statuses", multiple=True, type=click.Choice(["ok", "aborted", "failed", "skipped"]),
              help="Statuses to include (default: all).")
@click.option("--tests", "per_test", is_flag=True, help="One row per test instead of one per run.")
@click.option("--csv", "out", type=click.Path(dir_okay=False), default=None, help="CSV file (default: stdout).")
@pass_ctx
def results_export(ctx: Ctx, sweep_, model, tool, build, run_ids, last, all_runs, statuses, per_test, out) -> None:
    """Runs (or every test) as CSV, e.g. for a spreadsheet."""
    from mse_llm.report.data import STATUSES, table_row, test_rows, to_csv

    sel = _select(ctx, sweep_, model, tool, build, run_ids, last, all_runs, tuple(statuses) or STATUSES)
    rows = [t for r in sel.runs for t in test_rows(r)] if per_test else [table_row(r) for r in sel.runs]
    text = to_csv(rows)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        console.print(f"[green]✓[/] {len(rows)} rows → {escape(out)}")
    else:
        _plain(text.rstrip("\n"))
