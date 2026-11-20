"""`predict`, `results calibrate` and `sweep search` (M9, §4.6, §9 item 6)."""

from __future__ import annotations

import asyncio
import json
import sys

import click
from rich.markup import escape
from rich.table import Table

from mse_llm.cli import Ctx, console, json_option, main, pass_ctx
from mse_llm.cli_bench import _ICON, _opts, allow_low_power_option, power_gate, results, sweep

_VSTYLE = {"fits": "green", "tight": "yellow", "no-fit": "red", "ERROR": "red"}


def _free(free_: str):
    from mse_llm import fit as ff

    if str(free_).lower() == "auto":
        return ff.free_vram_now()
    try:
        return int(free_), "--free"
    except ValueError:
        raise click.BadParameter("auto or a number of MiB", param_hint="--free") from None


def _base_variant(ctx: Ctx, model: str, variant: str | None, kv: str | None):
    from mse_llm import fit as ff
    from mse_llm import variants as vv

    name = variant
    if not name:
        try:
            _, m = ctx.cfg.model(model)
            name = m.default_serve
        except KeyError:
            name = None
    try:
        v = vv.decode(name or "fa_Kq8_Vq8_default")
    except vv.VariantError as e:
        raise click.ClickException(str(e)) from e
    return ff.apply_kv(v, kv)


def _ctx_for(ctx: Ctx, model: str, ctx_: int | None) -> int:
    if ctx_:
        return ctx_
    try:
        _, m = ctx.cfg.model(model)
        if m.ctx:
            return m.ctx
    except KeyError:
        pass
    raise click.ClickException(f"{model}: no context size (--ctx, or [models.*].ctx)")


def _ints(text: str | None, name: str) -> list[int] | None:
    from mse_llm import predict as pr

    try:
        return pr.parse_int_list(text)
    except ValueError:
        raise click.BadParameter("a list like 41,34,28 or a range 16:30:2", param_hint=name) from None


def _calibration(ctx: Ctx, use: bool, tflops: float | None):
    from mse_llm import hw
    from mse_llm import predict as pr

    info = hw.detect()
    hard = pr.hardware_from(info, tflops)
    try:
        hw_hash = hw.identity(info)["hash"]
    except Exception:
        hw_hash = None
    plan = info.power.power_plan
    if hw.low_power_plan(plan) and not ctx.json:
        console.print(f"[yellow]![/] power plan [bold]{escape(plan)}[/] caps CPU / GPU power: runs are slower and vary "
                      "run to run; switch to a performance plan before benchmarking (calibration uses only runs "
                      "recorded under this plan)")
    cal = pr.load_calibration(pr.Params.spec(hard), hw_hash=hw_hash, use=use, power_plan=plan)
    return hard, cal, hw_hash, plan


def _predict_options(f):
    for opt in reversed([
        click.option("-v", "--variant", "variant", default=None,
                     help="Base variant: fa / K/V / load mode (default: [models].default_serve); -ncmoe and -ub "
                          "come from the grid."),
        click.option("--ctx", "ctx_", type=int, default=None, help="Context / prompt size (default: [models].ctx)."),
        click.option("--kv", default=None, metavar="K/V", help="Cache types when the base variant sets none."),
        click.option("--ncmoe", "ncmoe_", default=None, metavar="LIST",
                     help="-ncmoe values, e.g. 41,34,28 or 16:30:2 (default: all layers down to 0 in ~7 steps)."),
        click.option("--ub", "ub_", default=None, metavar="LIST",
                     help="-ub values (b = ub); default 512,1024,2048,4096."),
        click.option("--exact", is_flag=True, help="Measure every grid point (default: endpoints + linear rule)."),
        click.option("--free", "free_", default="auto", show_default=True, metavar="auto|MiB"),
        click.option("--margin", type=int, default=768, show_default=True, help="MiB below which a fit is tight."),
        click.option("--no-calibration", is_flag=True, help="Spec bandwidths only, ignore the stored runs."),
        click.option("--tflops", type=float, default=None, help="GPU TFLOPS before calibration (default 60)."),
        click.option("--engine", default=None, help="Engine whose llama-fit-params is used (default: cuda_flash)."),
    ]):
        f = opt(f)
    return f


def _run_predict(ctx: Ctx, model, variant, ctx_, kv, ncmoe_, ub_, exact, free_, margin, no_calibration, tflops,
                 engine, extra_ubs: list[int] | None = None):
    from mse_llm import fit as ff
    from mse_llm import predict as pr

    base = _base_variant(ctx, model, variant, kv)
    c = _ctx_for(ctx, model, ctx_)
    free_mib, free_src = _free(free_)
    hard, cal, hw_hash, plan = _calibration(ctx, not no_calibration, tflops)
    ubs = _ints(ub_, "--ub") or sorted(set(pr.DEFAULT_UBS + (extra_ubs or [])))
    try:
        if ctx.json or not sys.stdout.isatty():
            pred = pr.run_predict(ctx.cfg, model, base, c, ncmoes=_ints(ncmoe_, "--ncmoe"), ubs=ubs,
                                  free_mib=free_mib, free_source=free_src, margin_mib=margin, engine=engine,
                                  exact=exact, hardware=hard, calibration=cal)
        else:
            with console.status("llama-fit-params …") as st:
                pred = pr.run_predict(ctx.cfg, model, base, c, ncmoes=_ints(ncmoe_, "--ncmoe"), ubs=ubs,
                                      free_mib=free_mib, free_source=free_src, margin_mib=margin, engine=engine,
                                      exact=exact, hardware=hard, calibration=cal,
                                      progress=lambda i, n, w: st.update(f"llama-fit-params {i + 1}/{n} {w}"))
    except (pr.PredictError, ff.FitError) as e:
        raise click.ClickException(str(e)) from e
    except KeyError as e:
        raise click.ClickException(str(e.args[0])) from e
    pred.warnings += [f"power plan: {plan}"] if hw_low(plan) else []
    return pred, hw_hash, plan


def hw_low(plan):
    from mse_llm import hw

    return hw.low_power_plan(plan)


@main.command("predict")
@click.argument("model", metavar="GGUF_OR_MODEL")
@_predict_options
@json_option
@pass_ctx
def predict_cmd(ctx: Ctx, model, **kw) -> None:
    """Expected memory, TG and PP over an -ncmoe x -ub grid, and which setting helps (no model load)."""
    pred, _, _ = _run_predict(ctx, model, **kw)
    if ctx.json:
        click.echo(json.dumps(pred.as_dict(), indent=2))
        return
    print_prediction(pred)


def _cal_line(pred) -> str:
    cal = pred.calibration
    p = cal.params
    head = (f"calibrated on {len(cal.runs)} run(s)" if cal.calibrated
            else "[yellow]not calibrated[/] (no ok runs on this PC: spec values)")
    ms = cal.model_scale.get(pred.model_key or "")
    line = (f"{head}: VRAM {p.vram_gbs:.0f} GB/s, RAM {p.ram_gbs:.0f} GB/s, PCIe {p.pcie_gbs:.1f} GB/s, "
            f"GPU {p.tflops:.0f} TFLOPS, t0 {p.t0_ms:.1f} ms")
    if ms:
        q = cal.params_for(pred.model_key)
        line += (f"; this model ({ms.get('n_tg', 0)} TG / {ms.get('n_pp', 0)} PP values): VRAM {q.vram_gbs:.0f}, "
                 f"CPU experts {q.ram_gbs:.0f} GB/s, PCIe {q.pcie_gbs:.1f} GB/s, GPU {q.tflops:.0f} TFLOPS")
    elif cal.calibrated:
        line += "; no runs of this model yet"
    return line


def print_prediction(pred) -> None:
    from pathlib import Path

    s = pred.shape
    for w in pred.warnings:
        console.print(f"[yellow]![/] {escape(w)}")
    moe = (f"{s.moe_layers} MoE layers × {s.expert_layer_mib:.0f} MiB, {s.n_expert_used}/{s.n_expert} experts used "
           f"(reads {s.active_expert_mib:.1f} MiB per layer per token)" if s.moe_layers else "dense")
    console.print(f"model [bold]{escape(Path(pred.path).name)}[/]  ctx {pred.ctx}  base {escape(pred.base)}")
    console.print(f"[dim]{moe}; dense part {s.dense_mib:.0f} MiB; free VRAM "
                  f"{pred.free_mib if pred.free_mib is not None else '?'} MiB ({escape(pred.free_source)}), "
                  f"margin {pred.margin_mib}[/]")
    console.print(f"[dim]{_cal_line(pred)}[/]")
    best = {g: pred.best(g) for g in ("tg", "pp", "both")}
    mark = {id(p): "".join(k for k, g in (("T", "tg"), ("P", "pp"), ("*", "both")) if best[g] is p)
            for p in pred.points}
    t = Table(box=None, pad_edge=False, header_style="bold")
    for col, j in (("VARIANT", "left"), ("GPU MiB", "right"), ("host MiB", "right"), ("left", "right"),
                   ("VERDICT", "left"), ("TG t/s", "right"), (f"PP t/s @{pred.ctx}", "right"), ("", "left"),
                   ("NOTE", "left")):
        t.add_column(col, justify=j, overflow="fold")
    for p in sorted(pred.points, key=lambda q: (q.ub, -(q.ncmoe or 0))):
        v = p.verdict
        st = _VSTYLE.get(v.rstrip("~"), "")
        t.add_row(escape(p.variant.name), "-" if p.gpu_mib is None else f"{p.gpu_mib:.0f}",
                  "-" if p.host_mib is None else f"{p.host_mib:.0f}",
                  "-" if p.left_mib is None else f"{p.left_mib:.0f}", f"[{st}]{v}[/]" if st else v,
                  "-" if p.tg is None else f"{p.tg:.1f}", "-" if p.pp is None else f"{p.pp:.0f}",
                  mark.get(id(p), ""), escape(p.error or p.note or ""))
    console.print(t)
    for r in pred.recommendations:
        console.print(f"→ {escape(r)}")
    console.print("[dim]~ = memory interpolated (one -ncmoe step = one expert layer); T / P / * = best fitting TG / "
                  "PP / TG×PP. TG for tg128 at depth 0, PP for a prompt of ctx tokens (llama-bench). "
                  "`mse-llm results calibrate` shows how close the predictions were.[/]")


# ---------------------------------------------------------------- results calibrate

@results.command("calibrate")
@click.option("--model", default=None, help="Only rows of this model (the fit still uses every run).")
@click.option("--tflops", type=float, default=None, help="Effective GPU compute before calibration (default 60).")
@json_option
@pass_ctx
def results_calibrate(ctx: Ctx, model, tflops) -> None:
    """Fit predict's bandwidths to the stored runs of this PC; measured vs predicted per run (in-sample and
    leave-one-run-out)."""
    from mse_llm import hw
    from mse_llm import predict as pr

    info = hw.detect()
    hard = pr.hardware_from(info, tflops)
    try:
        hw_hash = hw.identity(info)["hash"]
    except Exception:
        hw_hash = None
    plan = info.power.power_plan
    runs = pr.stored_runs(hw_hash, power_plan=plan)
    if hw.low_power_plan(plan) and not ctx.json:
        console.print(f"[yellow]![/] power plan {escape(plan)}: only runs recorded under it are used")
    samples, notes = pr.samples_from_runs(runs, pr._shape_or_none)
    cal, rows = pr.check(pr.Params.spec(hard), samples, notes)
    if model:
        rows = [r for r in rows if r.model == model or model.lower() in r.model.lower()]
    if ctx.json:
        click.echo(json.dumps({"hardware": hard.as_dict(), "calibration": cal.as_dict(),
                               "rows": [r.as_dict() for r in rows]}, indent=2))
        return
    for k, v in hard.source.items():
        console.print(f"[dim]spec {k}: {escape(v)}[/]")
    if not samples:
        console.print("[yellow]no ok bench / benchy runs on this PC yet[/]: predict uses the spec values. "
                      "Run a sweep first (`mse-llm sweep run …` or `mse-llm sweep search …`).")
        return
    p, b = cal.params, cal.base
    console.print(f"fitted from {len(cal.runs)} runs ({cal.n_samples} TG/PP values):")
    for label, a, z, unit in (("VRAM", b.vram_gbs, p.vram_gbs, "GB/s"), ("RAM", b.ram_gbs, p.ram_gbs, "GB/s"),
                              ("PCIe", b.pcie_gbs, p.pcie_gbs, "GB/s"), ("GPU", b.tflops, p.tflops, "TFLOPS"),
                              ("t0", b.t0_ms, p.t0_ms, "ms/token"), ("t_ub", b.tub_ms, p.tub_ms, "ms/ubatch")):
        console.print(f"  {label:5} {a:8.1f} → {z:8.1f} {unit}")
    for m, ms in cal.model_scale.items():
        q = cal.params_for(m)
        console.print(f"  [dim]{escape(m)} ({ms.get('n_tg', 0)} TG / {ms.get('n_pp', 0)} PP): VRAM {q.vram_gbs:.0f}, "
                      f"CPU experts {q.ram_gbs:.1f} GB/s, t0 {q.t0_ms:.1f} ms; GPU {q.tflops:.0f} TFLOPS, "
                      f"PCIe {q.pcie_gbs:.1f} GB/s, t_ub {q.tub_ms:.1f} ms[/]")
    t = Table("model", "variant", "test", "measured", "spec", "calibrated", "err", "leave-one-out", "err",
              header_style="bold")
    for r in rows:
        le = r.loo_err
        t.add_row(escape(r.model), escape(r.variant), f"{r.kind} {r.n_prompt or ''}".strip(), f"{r.measured:.1f}",
                  f"{r.spec:.1f}", f"{r.predicted:.1f}", _pct(r.err), "-" if r.loo is None else f"{r.loo:.1f}",
                  "-" if le is None else _pct(le))
    console.print(t)
    loo = [abs(r.loo_err) for r in rows if r.loo_err is not None]
    if loo:
        within = sum(x <= 0.15 for x in loo)
        console.print(f"leave-one-out: median |err| {sorted(loo)[len(loo) // 2]:.1%}, max {max(loo):.1%}, "
                      f"{within}/{len(loo)} within 15 %")
    for n in notes:
        console.print(f"[dim]  {escape(n)}[/]")


def _pct(x: float) -> str:
    s = f"{x:+.1%}"
    return f"[red]{s}[/]" if abs(x) > 0.15 else s


# ---------------------------------------------------------------- sweep search

@sweep.command("search")
@click.argument("model")
@_predict_options
@click.option("--tool", type=click.Choice(["bench", "benchy"]), default="bench", show_default=True)
@click.option("--goal", type=click.Choice(["tg", "pp", "both"]), default="both", show_default=True,
              help="What the winner maximises (both = sqrt(TG x PP)).")
@click.option("--axes", default=None, help="Order of the stages, e.g. ub,ncmoe (default: from predict).")
@click.option("--steps", type=int, default=4, show_default=True,
              help="-ncmoe steps from all-on-CPU to the predicted edge.")
@click.option("--patience", type=int, default=2, show_default=True,
              help="Stop a stage after this many steps without a gain > 1 %.")
@click.option("--max-runs", type=int, default=12, show_default=True, help="New runs at most (reused runs are free).")
@click.option("--no-step-back", is_flag=True,
              help="Take the best run even when it is right next to a run that failed / spilled.")
@click.option("--no-reuse", is_flag=True, help="Run every step, even when an ok result for this build exists.")
@click.option("--tg", type=int, multiple=True, help="Generated tokens (default: [defaults].bench_tg).")
@click.option("--apply", "apply_", is_flag=True,
              help="Write the winner as [models.*].default_serve in the config file (`presets generate` uses it).")
@click.option("--dry-run", is_flag=True, help="Show the stages with predicted results; run nothing.")
@click.option("--output", type=click.Choice(["all", "filter", "none"]), default=None)
@click.option("--no-guard", is_flag=True, help="Monitor, but don't stop a run on a spill (not recommended here).")
@click.option("--mem-cap", type=int, default=None, metavar="MiB", help="Windows Job Object memory cap per run.")
@click.option("--port", type=int, default=None, help="benchy: port of the background server.")
@allow_low_power_option
@pass_ctx
def sweep_search(ctx: Ctx, model, tool, goal, axes, steps, patience, max_runs, no_step_back, no_reuse, tg, apply_,
                 dry_run, output, no_guard, mem_cap, port, allow_low_power, **kw) -> None:
    """Safe automatic search for the sweet spot: predict picks the order, each stage steps from the safe side
    toward the edge, never starts a predicted no-fit, stops at the first spill; the winner steps back one only
    from a run that really failed."""
    from mse_llm import bench as bb
    from mse_llm import predict as pr
    from mse_llm import results as rs
    from mse_llm import search as se

    try:
        mkey, _ = ctx.cfg.model(model)
    except KeyError as e:
        raise click.ClickException(str(e.args[0])) from e
    if not dry_run:
        power_gate(allow_low_power)           # before predict: its fit calls take a while
    pred, hw_hash, plan = _run_predict(ctx, mkey, **kw)
    console.print(f"[bold]sweep search {mkey}[/]: {tool}, goal {goal}, ctx {pred.ctx}")
    console.print(f"[dim]{_cal_line(pred)}[/]")
    for r in pred.recommendations[:2]:
        console.print(f"[dim]→ {escape(r)}[/]")
    params = ({"tg": list(tg)} if tool == "bench" else {"tg": list(tg), "pp": [pred.ctx]})
    ax = [a.strip() for a in axes.split(",")] if axes else None
    if ax and any(a not in se.AXES for a in ax):
        raise click.BadParameter(f"axes are {', '.join(se.AXES)}", param_hint="--axes")
    common = dict(model=mkey, tool=tool, goal=goal, axes=ax, ubs=sorted({p.ub for p in pred.points}),
                  ncmoes=_ints(kw.get("ncmoe_"), "--ncmoe"), steps=steps, patience=patience,
                  max_runs=max_runs, step_back=not no_step_back, params=params, engine=kw.get("engine"))
    if dry_run:
        res = asyncio.run(se.search(pred, run=se.simulated_runner(pred), simulated=True, **common))
        print_search(res, goal)
        console.print("[dim]dry run: TG / PP above are predictions; the real search stops at the first spill or "
                      "failed run, and later stages start from the real winner.[/]")
        return
    env = bb.RunEnv()
    bid = env.build_for(ctx.cfg, kw.get("engine")).get("id")
    opts = _opts(ctx, no_fit_check=False, force=False, output=output, no_save=False, no_monitor=False,
                 no_guard=no_guard, mem_cap=mem_cap, port=port)
    last = rs.latest_by_key(bid)

    def reuse(job):
        if no_reuse:
            return None
        cmd = job.command(ctx.cfg, port=port, save_result="<raw>/benchy.json" if job.tool == "benchy" else None)
        e = last.get(job.key(cmd))
        if e and e.get("status") == "ok":
            try:
                r = rs.find_run(e["run_id"])
            except KeyError:
                return None
            return r if pr.same_plan(r, plan) else None
        return None

    async def run(job):
        r = await bb.run_job(ctx.cfg, job, opts, env)
        console.print(f"  {_ICON[r.status]} {r.status}" + (f": {escape(r.reason)}" if r.reason else "")
                      + (f"  TG {r.metrics.get('tg_tps')}  PP {r.metrics.get('pp_tps')}" if r.metrics else ""))
        return r

    try:
        res = asyncio.run(se.search(pred, run=run, reuse=reuse, say=lambda s: console.print(s), **common))
    except KeyboardInterrupt:
        console.print("[yellow]interrupted: the running job was stopped and recorded as aborted[/]")
        sys.exit(130)
    res.build, res.hw, res.timestamp = bid, hw_hash, rs.now().isoformat(timespec="seconds")
    se.write_search(res, pred)
    print_search(res, goal)
    console.print(f"[dim]→ {escape(res.path or '')}[/]")
    from mse_llm import notify

    notify.notify(f"sweep search {mkey}: " + (f"winner {res.winner}" if res.winner else "no winner"),
                  ctx.cfg.defaults.notify)
    if apply_ and res.winner:
        src = ctx.cfg.source
        if src is None:
            raise click.ClickException("no config file to write to (--config)")
        old = se.set_default_serve(src, mkey, res.winner)
        console.print(f"[green]✓[/] {escape(str(src))}: [models.{escape(mkey)}] default_serve = "
                      f"{escape(res.winner)} (was {escape(old or '-')}); `mse-llm presets generate` picks it up")
    if res.any_spill:
        sys.exit(2)


def print_search(res, goal: str) -> None:
    console.print(f"[dim]order: {escape(res.order_why)}[/]")
    for st in res.stages:
        fixed = ", ".join(f"{k} {v}" for k, v in st.fixed.items())
        console.print(f"\n[bold]stage {st.axis}[/] ({fixed}): values {st.values}  [dim]stop: {st.stop}[/]")
        t = Table(box=None, pad_edge=False, header_style="bold")
        for col, j in (("variant", "left"), ("status", "left"), ("TG", "right"), ("PP", "right"),
                       ("pred TG", "right"), ("pred PP", "right"), ("left MiB", "right"), ("", "left"),
                       ("run / reason", "left")):
            t.add_column(col, justify=j, overflow="fold")
        for s in st.steps:
            icon = {"ok": "[green]✓[/]", "aborted": "[red]■[/]", "failed": "[red]✗[/]",
                    "skipped": "[yellow]–[/]"}.get(s.status, "[dim]·[/]")
            win = "◀ winner" if s.variant == st.winner else ("edge" if s.variant == st.edge else "")
            t.add_row(escape(s.variant), f"{icon} {s.status}", "-" if s.tg is None else f"{s.tg:.2f}",
                      "-" if s.pp is None else f"{s.pp:.0f}",
                      "-" if s.predicted_tg is None else f"{s.predicted_tg:.1f}",
                      "-" if s.predicted_pp is None else f"{s.predicted_pp:.0f}",
                      "-" if s.predicted_left_mib is None else f"{s.predicted_left_mib:.0f}", win,
                      escape(((s.run_id or "") + (" (reused)" if s.reused else "") + "  " + (s.reason or "")).strip()))
        console.print(t)
        if st.stepped_back:
            console.print(f"  [dim]best run was right next to a failed run ({escape(st.edge or '')}): one step back[/]")
    w = res.winner_step
    if res.winner and w:
        console.print(f"\n[bold green]winner[/] {escape(res.winner)}: TG {w.tg or 0:.2f} t/s, PP {w.pp or 0:.0f} t/s"
                      + (f", VRAM peak {w.vram_peak_mib:.0f} MiB" if w.vram_peak_mib else ""))
    else:
        console.print("\n[yellow]no winner: no step ran ok[/]")
    console.print(f"[dim]{res.runs} new run(s), {res.reused} reused" + (", [red]a run spilled[/]" if res.any_spill
                                                                         else ", no run spilled") + "[/]")


__all__ = ["predict_cmd", "results_calibrate", "sweep_search"]
