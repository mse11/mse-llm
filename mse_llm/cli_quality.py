"""`quality` commands (M10, §4.5): KLD of K/V-cache choices against an f16-KV reference of the same GGUF."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys

import click
from rich.markup import escape
from rich.table import Table

from mse_llm.cli import Ctx, _plain, _print_commands, _variants_for, console, json_option, main, pass_ctx
from mse_llm.cli_bench import _ICON, _fmt


@main.group()
def quality() -> None:
    """KL divergence of K/V-cache variants vs an f16-KV reference (llama-perplexity), with a noise floor."""


def _common(f):
    for opt in reversed([
        click.option("--ctx", "ctx_", type=int, default=None,
                     help="KLD context (default 8192; KV errors grow with it)."),
        click.option("--chunks", type=int, default=None, help="Chunks of --ctx tokens (default 8; reference size "
                                                                "grows with it)."),
        click.option("--corpus", default=None, help="code (default: llama.cpp sources) | wikitext2 | FILE."),
        click.option("--engine", default=None, help="Engine (default: cuda_flash)."),
        click.option("--output", type=click.Choice(["all", "filter", "none"]), default=None,
                     help="llama-perplexity output on the console (logs keep everything)."),
        click.option("--no-guard", is_flag=True, help="Monitor, but don't stop a run on a spill / RAM trip."),
        click.option("--yes", "-y", is_flag=True, help="Don't ask before creating a reference file."),
        click.option("--dry-run", is_flag=True,
                     help="Show the reference plan, its size and the commands; run nothing."),
        json_option,
    ]):
        f = opt(f)
    return f


def _opts(ctx: Ctx, output, no_guard):
    from mse_llm import quality as qq
    from mse_llm.bench import RunOpts
    from mse_llm.proc import OutputFilter

    filt = OutputFilter(output or ctx.cfg.defaults.output, qq.PPL_FILTER)

    def sink(line: str) -> None:
        if filt(line):
            _plain("  │ " + line)

    return RunOpts(guard=not no_guard, mem_cap_mib=ctx.cfg.monitor.job_memory_cap_mib, sink=sink,
                   say=lambda s: console.print("  " + s))


def _setup(ctx: Ctx, model, ctx_, chunks, corpus, engine, base=None, plan_new=True):
    from mse_llm import quality as qq

    try:
        return qq.setup(ctx.cfg, model, ctx=ctx_ or qq.DEFAULT_CTX, chunks=chunks or qq.DEFAULT_CHUNKS,
                        corpus=corpus or qq.DEFAULT_CORPUS, engine=engine, base=base, plan_new=plan_new)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    except KeyError as e:
        raise click.ClickException(str(e.args[0])) from e


def _ref_line(st) -> str:
    r = st.ref
    if r is None:
        return "reference: none yet"
    fl = r.floor or {}
    floor = f"noise floor: mean KLD {_kld(fl['mean_kld'])} ({fl.get('change')})" if fl.get("mean_kld") is not None \
        else "no noise floor yet"
    tag = "  [yellow]q8 reference: KLD values are vs q8_0 KV, not f16[/]" if r.q8_ref else ""
    place = r.variant + (" " + " ".join(r.extra) if r.extra else "")
    return f"reference {r.id}: {r.label}, placement {escape(place)}  ({floor}){tag}"


def _plan_lines(st) -> list[str]:
    from mse_llm import quality as qq

    out = []
    p = st.placement
    if p is None:
        return out
    size = f"{st.size_gib:.1f} GiB" if st.size_gib is not None else "unknown size"
    out.append(f"new reference: {'/'.join(p.kv)} KV, ctx {st.ctx}, {st.chunks} chunks, corpus {st.corpus.label} "
               f"({st.corpus.size / 1e6:.1f} MB), placement {escape(qq.vname(p.variant))} (fit: {p.verdict})")
    for s in p.steps:
        out.append(f"  fallback: {escape(s)}")
    free = shutil.disk_usage(_existing(qq.refs_dir())).free
    out.append(f"  reference file: {size} in {qq.refs_dir()} ({free / 2**30:.0f} GiB free); "
               "--chunks trades accuracy for size")
    return out


def _existing(p):
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def _check_space(st) -> None:
    from mse_llm import quality as qq

    if st.size_bytes is None:
        return
    free = shutil.disk_usage(_existing(qq.refs_dir())).free
    if st.size_bytes * 1.1 > free:
        raise click.ClickException(f"the reference needs {st.size_gib:.1f} GiB, only {free / 2**30:.1f} GiB free on "
                                   f"{_existing(qq.refs_dir())}: use fewer --chunks, or `mse-llm quality clean`")


def _confirm(st, yes: bool) -> None:
    if st.placement is not None and not st.placement.fits:
        console.print("[yellow]![/] even the last fallback is predicted not to fit; the memory guard stops the run "
                      "if it spills")
    if yes:
        return
    if not sys.stdin.isatty():
        raise click.ClickException("creating a reference needs confirmation: pass --yes")
    if not click.confirm(f"create the reference ({st.size_gib:.1f} GiB)?", default=False):
        raise click.Abort()


def _show_dry(ctx: Ctx, st, jobs) -> None:
    for ln in _plan_lines(st):
        console.print(ln)
    _print_commands(ctx, [j.command(ctx.cfg) for j in jobs])


async def _ensure_ref(ctx: Ctx, st, opts, engine, *, floor: bool = True, env=None, sweep=None):
    """Reference (created when missing), then its noise floor (when missing and floor)."""
    from mse_llm import quality as qq

    ref = st.ref
    if ref is None:
        console.print(f"[bold]▶ reference[/] {escape(qq.vname(st.placement.variant))} (ctx {st.ctx}, "
                      f"{st.chunks} chunks)")
        ref, r = await qq.make_reference(ctx.cfg, st, opts, engine=engine, env=env, sweep=sweep)
        console.print(f"  {_ICON[r.status]} {r.status}" + (f": {escape(r.reason)}" if r.reason else "")
                      + (f"  PPL {r.quality.get('ppl')}  {r.duration_s:.0f} s" if r.status == "ok" else ""))
        if r.status != "ok":
            raise click.ClickException("the reference run failed; see " + str(r.logs.get("tool")))
        st.ref = ref
    if floor and (ref.floor or {}).get("mean_kld") is None:
        console.print("[bold]▶ noise floor[/] (reference vs reference with another placement)")
        fit = qq.default_fit(ctx.cfg, st.model, st.ctx, engine)
        r = await qq.run_floor(ctx.cfg, st, ref, opts, engine=engine, env=env, fit=fit, sweep=sweep)
        console.print(f"  {_ICON[r.status]} {r.status}" + (f": {escape(r.reason)}" if r.reason else "")
                      + (f"  mean KLD {r.quality.get('mean_kld')}" if r.status == "ok" else ""))
    return ref


@quality.command("kld-ref")
@click.argument("model")
@click.option("-v", "--variant", "placement", default=None,
              help="Placement to start from (default: the model's default_serve); K/V become f16.")
@click.option("--force", is_flag=True, help="Make a new reference even when one exists.")
@click.option("--no-floor", is_flag=True, help="Don't measure the noise floor afterwards.")
@_common
@pass_ctx
def kld_ref(ctx: Ctx, model, placement, force, no_floor, ctx_, chunks, corpus, engine, output, no_guard, yes,
            dry_run) -> None:
    """Create the f16-KV reference logits for MODEL (fallbacks when it doesn't fit: more experts on the CPU,
    KV in RAM, q8 reference) and measure the noise floor."""
    from mse_llm import quality as qq
    from mse_llm.variants import VariantError, decode

    try:
        base = decode(placement) if placement else None
    except VariantError as e:
        raise click.ClickException(str(e)) from e
    st = _setup(ctx, model, ctx_, chunks, corpus, engine, base)
    if st.ref is not None and force:
        st.ref = None
        st = _replan(ctx, st, base, engine)
    if ctx.json and dry_run:
        click.echo(json.dumps({"ref": st.ref.as_dict() if st.ref else None,
                               "plan": _plan_dict(st), "size_bytes": st.size_bytes}, indent=2))
        return
    if st.ref is not None:
        console.print(_ref_line(st))
        if dry_run or (no_floor or (st.ref.floor or {}).get("mean_kld") is not None):
            return
    else:
        for ln in _plan_lines(st):
            console.print(ln)
    if dry_run:
        meta = qq.new_ref_meta(st)
        job = qq.QualityJob(st.model, st.placement.variant, "ref", st.ctx, st.chunks, st.corpus, meta.path(),
                            engine=engine)
        _print_commands(ctx, [job.command(ctx.cfg)])
        return
    if st.ref is None:
        _check_space(st)
        _confirm(st, yes)
    opts = _opts(ctx, output, no_guard)
    _run(_ensure_ref(ctx, st, opts, engine, floor=not no_floor))
    console.print(_ref_line(st))


def _replan(ctx, st, base, engine):
    from mse_llm import quality as qq
    from mse_llm.variants import Variant, decode

    _, m = ctx.cfg.model(st.model)
    b = base if base is not None else (decode(m.default_serve) if m.default_serve else Variant(fa="on"))
    st.placement = qq.plan_reference(b, qq.default_fit(ctx.cfg, st.model, st.ctx, engine), st.moe_layers)
    return st


def _plan_dict(st):
    from mse_llm import quality as qq

    p = st.placement
    return None if p is None else {"variant": qq.vname(p.variant), "kv": list(p.kv), "steps": p.steps,
                                   "verdict": p.verdict, "fits": p.fits}


def _run(coro):
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        console.print("[yellow]interrupted: the running job was stopped and recorded as aborted[/]")
        sys.exit(130)


@quality.command("floor")
@click.argument("model")
@_common
@pass_ctx
def kld_floor(ctx: Ctx, model, ctx_, chunks, corpus, engine, output, no_guard, yes, dry_run) -> None:
    """(Re-)measure the noise floor: the reference against itself with a different placement."""
    st = _setup(ctx, model, ctx_, chunks, corpus, engine, plan_new=False)
    if st.ref is None:
        raise click.ClickException(f"no reference for {st.model} at ctx {st.ctx} / corpus {st.corpus.label} / "
                                   f"build {st.build}: run `mse-llm quality kld-ref {st.model}` first")
    st.ref.floor = None if not dry_run else st.ref.floor
    if dry_run:
        console.print(_ref_line(st))
        return
    _run(_ensure_ref(ctx, st, _opts(ctx, output, no_guard), engine))
    console.print(_ref_line(st))


@quality.command("kld")
@click.argument("model")
@click.option("-v", "--variant", "patterns", multiple=True, help="Variant (brace patterns; repeatable).")
@click.option("--from", "from_file", type=click.Path(allow_dash=True), default=None,
              help="Variant list from a file or - (stdin).")
@click.option("--sweep", "sweep_", default=None, help="Record the runs under this sweep name.")
@_common
@pass_ctx
def kld_cmd(ctx: Ctx, model, patterns, from_file, sweep_, ctx_, chunks, corpus, engine, output, no_guard, yes,
            dry_run) -> None:
    """KLD of each variant's K/V types vs the reference (created first when missing), with the noise floor:
    mean KLD within 2x the floor = no measurable loss."""
    from mse_llm import quality as qq

    vs = _variants_for(ctx, model, patterns, from_file)
    base = vs[0].without_spec()
    st = _setup(ctx, model, ctx_, chunks, corpus, engine, base)
    if dry_run:
        console.print(_ref_line(st) if st.ref else "")
        ref = st.ref or qq.new_ref_meta(st)
        jobs = [] if st.ref else [qq.QualityJob(st.model, st.placement.variant, "ref", st.ctx, st.chunks, st.corpus,
                                                ref.path(), engine=engine)]
        jobs += [qq.QualityJob(st.model, v.without_spec(), "kld", st.ctx, ref.chunks, st.corpus, ref.path(), ref,
                               engine, tested=v.name) for v in vs]
        if ctx.json:
            click.echo(json.dumps({"ref": ref.as_dict(), "plan": _plan_dict(st),
                                   "commands": [j.command(ctx.cfg).as_dict() for j in jobs]}, indent=2))
            return
        _show_dry(ctx, st, jobs)
        return
    if st.ref is None:
        for ln in _plan_lines(st):
            console.print(ln)
        _check_space(st)
        _confirm(st, yes)
    opts = _opts(ctx, "none" if ctx.json else output, no_guard)
    quiet = console.quiet
    console.quiet = quiet or ctx.json            # --json: only the JSON on stdout
    try:
        runs = _run(_kld_all(ctx, st, vs, opts, engine, sweep_))
    finally:
        console.quiet = quiet
    print_kld(ctx, st, runs)
    if any(r.status != "ok" for r in runs):
        sys.exit(2 if any(r.status == "aborted" for r in runs) else 1)


async def _kld_all(ctx: Ctx, st, vs, opts, engine, sweep=None):
    from mse_llm import bench as bb
    from mse_llm import quality as qq

    env = bb.RunEnv()
    ref = await _ensure_ref(ctx, st, opts, engine, env=env, sweep=sweep)
    fit = qq.default_fit(ctx.cfg, st.model, ref.ctx, engine)
    out = []
    done: dict[str, object] = {}
    for i, v in enumerate(vs, 1):
        tv, note = qq.tested_variant(v, ref, fit)
        key = qq.vname(tv)
        if key in done:              # same K/V + placement as an earlier variant (only spec / b differ)
            continue
        console.print(f"[bold]▶ [{i}/{len(vs)}] kld {escape(v.name)}[/]"
                      + (f"  [dim]# {escape(note)}[/]" if note else ""))
        job = qq.QualityJob(st.model, tv, "kld", ref.ctx, ref.chunks, st.corpus, ref.path(), ref, engine, sweep,
                            note=note, tested=v.name)
        r = await qq.run_quality(ctx.cfg, job, opts, env)
        q = r.quality or {}
        console.print(f"  {_ICON[r.status]} {r.status}" + (f": {escape(r.reason)}" if r.reason else "")
                      + (f"  mean KLD {q.get('mean_kld')}  same top p {_pct(q.get('same_top_p'))}  "
                         f"{q.get('verdict')}" if r.status == "ok" else ""))
        done[key] = r
        out.append(r)
    return out


def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.2f} %"


def _kld(x) -> str:
    if x is None:
        return "-"
    if x == 0:
        return "< 1e-6"
    return f"{x:.2e}" if abs(x) < 1e-4 else f"{x:.5f}"


def print_kld(ctx: Ctx, st, runs) -> None:
    from mse_llm import quality as qq

    rows = qq.kld_rows(runs)
    ref = st.ref
    if ctx.json:
        click.echo(json.dumps({"ref": ref.as_dict() if ref else None, "rows": rows}, indent=2))
        return
    console.print(_ref_line(st))
    t = Table("variant", "K/V", "mean KLD", "99% KLD", "same top p", "PPL ratio", "verdict", "run",
              title=f"KLD {st.model} (ctx {st.ctx}, corpus {st.corpus.label})")
    fl = (ref.floor or {}) if ref else {}
    if fl.get("mean_kld") is not None:
        t.add_row("[dim]noise floor[/]", "/".join(ref.kv), _kld(fl["mean_kld"]), _kld(fl.get("kld_p99")),
                  _pct(fl.get("same_top_p")), "", f"[dim]{escape(fl.get('change') or '')}[/]", fl.get("run_id", ""))
    for r in rows:
        if r["status"] != "ok":
            t.add_row(escape(r["variant"]), r["kv"], "-", "-", "-", "-",
                      f"{_ICON[r['status']]} {escape(r['reason'] or '')}", r["run_id"])
            continue
        t.add_row(escape(r["variant"]), r["kv"], _kld(r["mean_kld"]) + (f" ± {_kld(r['mean_kld_err'])}" if
                                                                          r.get("mean_kld_err") else ""),
                  _kld(r["kld_p99"]), _pct(r["same_top_p"]), _fmt(r["ppl_ratio"], 4), escape(r["verdict"] or ""),
                  r["run_id"])
    console.print(t)
    if ref and ref.q8_ref:
        console.print("[yellow]values are KLD vs a q8_0/q8_0 reference (f16 KV did not fit), not vs f16[/]")
    console.print(f"[dim]verdict: mean KLD ≤ {qq.FLOOR_FACTOR:g}x the noise floor = no measurable loss. "
                  "Same top p = how often the top token matches the reference.[/]")


@quality.command("list")
@click.argument("model", required=False)
@json_option
@pass_ctx
def quality_list(ctx: Ctx, model) -> None:
    """References on disk and the stored KLD results."""
    from mse_llm import quality as qq
    from mse_llm import results as rs

    mkey = ctx.cfg.model(model)[0] if model else None
    refs = qq.list_refs(mkey)
    idx = [e for e in rs.read_index() if e.get("tool") == "quality" and (mkey is None or e.get("model") == mkey)
           and e.get("mode") in ("kld", "floor")]
    if ctx.json:
        click.echo(json.dumps({"refs": [r.as_dict() for r in refs], "runs": idx}, indent=2))
        return
    t = Table("model", "ref", "KV", "ctx", "chunks", "corpus", "build", "size", "floor", "ready", title="references")
    for r in refs:
        size = r.path().stat().st_size / 2**30 if r.path().is_file() else None
        fl = (r.floor or {}).get("mean_kld")
        t.add_row(escape(r.model), r.id, "/".join(r.kv), str(r.ctx), str(r.chunks),
                  f"{r.corpus.get('name')}@{str(r.corpus.get('sha'))[:8]}", str(r.build or "-"),
                  "-" if size is None else f"{size:.1f} GiB", _kld(fl), "yes" if r.ready else "no")
    console.print(t)
    t = Table("run", "model", "mode", "variant", "ctx", "mean KLD", "same top p", "ref", "status", title="KLD runs")
    for e in idx[-40:]:
        t.add_row(e["run_id"], escape(e["model"]), e.get("mode") or "", escape(e["variant"]), str(e.get("ctx")),
                  _kld(e.get("mean_kld")), _pct(e.get("same_top_p")), str(e.get("ref_id") or ""), e["status"])
    console.print(t)


@quality.command("clean")
@click.argument("model", required=False)
@click.option("--all", "all_", is_flag=True, help="Every reference of every model.")
@click.option("--yes", "-y", is_flag=True)
@pass_ctx
def quality_clean(ctx: Ctx, model, all_, yes) -> None:
    """Delete reference files (they are large: ~2 GiB per chunk at ctx 8192). Results stay."""
    from mse_llm import quality as qq

    if not model and not all_:
        raise click.UsageError("name a MODEL or pass --all")
    refs = qq.list_refs(ctx.cfg.model(model)[0] if model else None)
    files = [p for r in refs for p in (r.path(), r.meta_path()) if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    if not files:
        console.print("nothing to delete")
        return
    for p in files:
        console.print(f"  {p}  ({p.stat().st_size / 2**30:.2f} GiB)")
    if not yes and not click.confirm(f"delete {len(files)} file(s), {total / 2**30:.1f} GiB? (cannot be undone)",
                                     default=False):
        raise click.Abort()
    for p in files:
        p.unlink()
    console.print(f"[green]✓[/] deleted {len(files)} file(s), {total / 2**30:.1f} GiB")


@quality.command("corpus")
@click.argument("name", required=False)
@click.option("--download", is_flag=True, help="wikitext2: download it (Hugging Face, ~4.5 MB zip).")
@click.option("--yes", "-y", is_flag=True)
@json_option
@pass_ctx
def quality_corpus(ctx: Ctx, name, download, yes) -> None:
    """The KLD corpora: code (llama.cpp sources, built locally), wikitext2 (download once), or a FILE."""
    from mse_llm import quality as qq

    if download:
        if name not in ("wikitext2", "wikitext-2", "wiki"):
            raise click.UsageError("--download is for wikitext2")
        p = qq.wikitext_path()
        if p.is_file():
            console.print(f"already there: {p}")
            return
        if not yes and not click.confirm(f"download {qq.WIKITEXT_URL} (~{qq.WIKITEXT_ZIP_MB} MB) into {p.parent}?",
                                         default=False):
            raise click.Abort()
        p = qq.download_wikitext()
        console.print(f"[green]✓[/] {p} ({p.stat().st_size / 1e6:.1f} MB)")
        return
    rows = []
    for n in [name] if name else ["code", "wikitext2"]:
        try:
            c = qq.resolve_corpus(n)
            rows.append({**c.as_dict(), "ok": True})
        except ValueError as e:
            rows.append({"name": n, "ok": False, "error": str(e)})
    if ctx.json:
        click.echo(json.dumps(rows, indent=2))
        return
    for r in rows:
        if r["ok"]:
            console.print(f"[green]✓[/] {r['name']}@{r['sha'][:8]}  {r['size'] / 1e6:.1f} MB  {escape(r['path'])}  "
                          f"[dim]{escape(r['source'])}[/]")
        else:
            console.print(f"[yellow]–[/] {r['name']}: {escape(r['error'])}")


def kv_groups(variants) -> list:
    """One variant per distinct (K, V) cache type pair, in order (sweep --with-kld)."""
    seen: dict[tuple, object] = {}
    for v in variants:
        seen.setdefault((v.ctk or "f16", v.ctv or "f16"), v)
    return list(seen.values())


def sweep_kld(ctx: Ctx, model: str, variants, sweep: str, engine, *, kld_ctx=None, yes=False, dry_run=False,
              output=None, no_guard=False) -> None:
    """`sweep run --with-kld`: a KLD run per distinct K/V combination before the speed runs."""
    from mse_llm import quality as qq

    vs = kv_groups(variants)
    if not vs:
        return
    console.print(f"[bold]KLD for {len(vs)} K/V combination(s)[/]: "
                  + ", ".join(f"{v.ctk or 'f16'}/{v.ctv or 'f16'}" for v in vs))
    st = _setup(ctx, model, kld_ctx, None, None, engine, vs[0].without_spec())
    if dry_run:
        console.print(_ref_line(st) if st.ref else "")
        ref = st.ref or qq.new_ref_meta(st)
        jobs = [] if st.ref else [qq.QualityJob(st.model, st.placement.variant, "ref", st.ctx, st.chunks, st.corpus,
                                                ref.path(), engine=engine)]
        jobs += [qq.QualityJob(st.model, v.without_spec(), "kld", st.ctx, ref.chunks, st.corpus, ref.path(), ref,
                               engine, sweep, tested=v.name) for v in vs]
        if not ctx.json:
            _show_dry(ctx, st, jobs)
        return
    if st.ref is None:
        for ln in _plan_lines(st):
            console.print(ln)
        _check_space(st)
        _confirm(st, yes)
    runs = _run(_kld_all(ctx, st, vs, _opts(ctx, output, no_guard), engine, sweep))
    print_kld(ctx, st, runs)
