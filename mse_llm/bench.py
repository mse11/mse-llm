"""Running `bench` / `benchy` jobs and sweeps (M7, §8): fit pre-check, monitor + memory guard, parsing,
one RunResult per job in the results store.

    llama-bench   bash run_bench_params (+ _wrapper_EXTRAS / _wrapper_PROMPTS)
    benchy        bash run_benchy_params_with_server (background llama-server, then llama-benchy)

Every job is checked with `fit` first (llama-fit-params, ~0.5 s). A variant that doesn't fit is recorded as
`skipped` and never started (unless forced): the freezes the bash notes describe came from such configs.
While a job runs, the monitor guard (§5 rules A-C) stops it on a GPU memory spill or a RAM / commit trip;
the result is then `aborted` with the reason.
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mse_llm import results as rs
from mse_llm.config import Config
from mse_llm.runners.base import Command
from mse_llm.variants import Variant

Say = Callable[[str], None]


@dataclass
class Job:
    tool: str                              # bench | benchy
    model: str
    variant: Variant
    ctx: int | None = None
    # bench: tg, repetitions; benchy: pp, tg, depth, runs, concurrency
    params: dict[str, Any] = field(default_factory=dict)
    engine: str | None = None
    sweep: str | None = None
    note: str | None = None

    def command(self, cfg: Config, *, port: int | None = None, save_result: str | None = None,
                book_url: str | None = None) -> Command:
        from mse_llm.runners import bench_command, benchy_command

        p = self.params
        if self.tool == "bench":
            return bench_command(cfg, self.model, self.variant, self.ctx, self.engine, tg=p.get("tg") or None,
                                 repetitions=p.get("repetitions"))
        if self.tool == "benchy":
            return benchy_command(cfg, self.model, self.variant, self.ctx, self.engine, pp=p.get("pp") or None,
                                  tg=p.get("tg") or None, depth=p.get("depth") or None, runs=p.get("runs"),
                                  concurrency=p.get("concurrency") or None, port=port, save_result=save_result,
                                  prompts=p.get("prompts"), book_url=book_url)
        raise ValueError(f"tool {self.tool!r} runs in a later milestone (eval: M12)")

    def key(self, cmd: Command) -> str:
        return rs.run_key(self.tool, cmd.model or self.model, self.variant.name, cmd.ctx, cmd.params)

    @property
    def label(self) -> str:
        return f"{self.tool} {self.model} {self.variant.name}" + (f" (ctx {self.ctx})" if self.ctx else "")


@dataclass
class RunOpts:
    fit_check: bool = True
    force: bool = False                    # start even when the fit check says it doesn't fit
    monitor: bool = True
    guard: bool = True
    mem_cap_mib: int | None = None
    port: int | None = None                # benchy server port (default: the engine's)
    write: bool = True                     # write to the results store
    sink: Callable[[str], None] | None = None   # tool output lines (already filtered by the caller)
    say: Say = lambda s: None              # status lines (rich markup)
    samplers: list[Any] | None = None      # tests: scripted monitor samplers
    fit_runner: Callable[..., Any] | None = None   # tests: replaces sv.fit_check
    allow_low_power: bool = False          # CLI: run on battery / a quiet plan (power_gate)


@dataclass
class RunEnv:
    """What is the same for every job of one invocation: build, hardware, model identities."""
    build: dict[str, dict[str, Any]] = field(default_factory=dict)   # engine -> build record
    hw: dict[str, Any] | None = None
    model_ids: dict[str, dict[str, Any] | None] = field(default_factory=dict)

    def build_for(self, cfg: Config, engine: str | None) -> dict[str, Any]:
        name, eng = cfg.engine(engine)
        if name not in self.build:
            self.build[name] = rs.build_record(eng.bin_path(), name)
        return self.build[name]

    def hw_info(self) -> dict[str, Any]:
        if self.hw is None:
            self.hw = rs.hw_record()
        return self.hw

    def model_id(self, path: str) -> dict[str, Any] | None:
        if path not in self.model_ids:
            from mse_llm.hw.model_id import model_identity

            try:
                self.model_ids[path] = model_identity(path)
            except OSError:
                self.model_ids[path] = None
        return self.model_ids[path]


# ---------------------------------------------------------------- one job

def _fit_dict(chk) -> dict[str, Any]:
    r = chk.row
    d: dict[str, Any] = {"verdict": chk.verdict, "ctx": chk.ctx}
    if chk.error:
        d["error"] = chk.error
    if r is not None and not r.error:
        d.update(gpu_mib=round(r.gpu_mib), host_mib=round(r.host_mib), host_unsure_mib=round(r.host_unsure_mib),
                 free_mib=r.free_mib,
                 left_mib=None if r.left_mib is None else round(r.left_mib), suggest=list(r.suggest))
    return d


def _fit_ctx(job: Job, cmd: Command) -> int | None:
    """The context the job really allocates: llama-bench -p N -n M needs N (pp test) or M (tg test),
    a server its -c."""
    if cmd.ctx is None:
        return None
    if job.tool == "bench":
        return max(cmd.ctx, *(cmd.params.get("tg") or [0]))
    return cmd.ctx


def _base_result(cfg: Config, job: Job, cmd: Command, env: RunEnv, run_id: str, ts: str) -> rs.RunResult:
    mkey, m = cfg.model(job.model)
    ename, _ = cfg.engine(job.engine)
    srv = cmd.server
    return rs.RunResult(
        run_id=run_id, key=job.key(cmd), timestamp=ts, sweep=job.sweep, tool=job.tool, engine=ename, model=mkey,
        model_path=m.path, model_identity=env.model_id(m.path), variant=job.variant.name,
        variant_fields=job.variant.as_dict(), ctx=cmd.ctx, params=dict(cmd.params), argv=list(cmd.argv),
        server_argv=list(srv.argv) if srv else None, env=dict(cmd.env), build=env.build_for(cfg, job.engine),
        hw=env.hw_info(), note=job.note or None)


def _tail_reason(lines: list[str], n: int = 3) -> str:
    keep = [ln.strip() for ln in lines if ln.strip() and any(w in ln.lower() for w in ("error", "fail", "abort",
                                                                                        "out of memory"))]
    return " | ".join((keep or [ln.strip() for ln in lines if ln.strip()])[-n:])[:500]


async def run_job(cfg: Config, job: Job, opts: RunOpts, env: RunEnv | None = None) -> rs.RunResult:
    """Run one job to a RunResult (written to the store when opts.write). Never raises for a failed
    benchmark; Ctrl+C (CancelledError) stops the processes, records `aborted` and re-raises."""
    env = env or RunEnv()
    t = rs.now()
    run_id = rs.new_run_id(job.tool, t)
    ts = t.isoformat(timespec="seconds")
    raw = rs.raw_dir(run_id)
    save = raw / "benchy.json" if job.tool == "benchy" else None
    book = None
    prompts = job.params.get("prompts") if job.tool == "benchy" else None
    if prompts not in (None, "prose"):
        from mse_llm import quality as qq

        try:
            book = qq.TextServer(qq.prompt_text(cfg, prompts))
        except (OSError, ValueError) as e:
            cmd = job.command(cfg, port=opts.port, save_result=str(save) if save else None)
            res = _base_result(cfg, job, cmd, env, run_id, ts)
            res.status, res.reason = "failed", f"prompts {prompts}: {e}"
            if opts.write:
                rs.write_run(res)
            return res
    cmd = job.command(cfg, port=opts.port, save_result=str(save) if save else None,
                      **({"book_url": book.url} if book else {}))
    res = _base_result(cfg, job, cmd, env, run_id, ts)
    try:
        return await _run_job(cfg, job, opts, env, cmd, res, raw, save)
    finally:
        if book is not None:
            book.close()


async def _run_job(cfg: Config, job: Job, opts: RunOpts, env: RunEnv, cmd: Command, res: rs.RunResult, raw: Path,
                   save: Path | None) -> rs.RunResult:
    from mse_llm import serve as sv

    t0 = time.monotonic()

    def done(status: str, reason: str | None = None) -> rs.RunResult:
        res.status = status                 # type: ignore[assignment]
        res.reason = reason
        res.duration_s = round(time.monotonic() - t0, 1)
        if opts.write:
            rs.write_run(res)
        return res

    if not cmd.ok:
        errs = cmd.errors + (cmd.server.errors if cmd.server else [])
        return done("failed", "flag check: " + "; ".join(errs))

    expected_host = None
    host_extra = 0.0
    fctx = _fit_ctx(job, cmd)
    if opts.fit_check and fctx:
        opts.say(f"[dim]fit check (llama-fit-params, ctx {fctx}) …[/]")
        check = opts.fit_runner or sv.fit_check
        chk = await asyncio.to_thread(check, cfg, job.model, job.variant, fctx, job.engine)
        res.fit = _fit_dict(chk)
        if chk.row is not None and not chk.row.error:
            expected_host = chk.row.host_mib
            host_extra = chk.row.host_unsure_mib
        if chk.bad and not opts.force:
            why = f"fit check: {chk.verdict}"
            if chk.row is not None and chk.row.left_mib is not None:
                why += f" (GPU {chk.row.gpu_mib:.0f} MiB, left {chk.row.left_mib:.0f})"
            if chk.row is not None and chk.row.suggest:
                why += "; " + chk.row.suggest[0]
            return done("skipped", why)

    raw.mkdir(parents=True, exist_ok=True)
    mon = None
    if opts.monitor and cfg.defaults.monitor:
        from mse_llm.monitor import monitor_from

        mon = monitor_from(cfg.monitor, guard=opts.guard, expected_host_mib=expected_host,
                           csv_path=raw / "monitor.csv", samplers=opts.samplers, stall=True, host_extra_mib=host_extra)
    try:
        if job.tool == "bench":
            await _run_bench(cfg, cmd, res, raw, mon, opts)
        else:
            await _run_benchy(cfg, job, cmd, res, raw, save, mon, opts)
    except asyncio.CancelledError:
        _finish_monitor(res, mon)
        done("aborted", "interrupted (Ctrl+C)")
        raise
    _finish_monitor(res, mon)
    res.metrics = rs.headline(res.tests)
    if mon is not None and mon.tripped:
        return done("aborted", f"memory guard: {mon.tripped}")
    if res.reason:
        return done("failed", res.reason)
    if not res.tests:
        return done("failed", "no results parsed")
    return done("ok")


def _finish_monitor(res: rs.RunResult, mon) -> None:
    if mon is not None and mon.samples:
        sm = mon.summary()
        res.monitor = sm.as_dict()
        res.monitor["csv"] = rs.rel(sm.csv_path)


async def _with_monitor(mon, pid_of: Callable[[], int | None], body, on_trip) -> None:
    """Run body() with the monitor started first (baseline before the process exists)."""
    if mon is None:
        await body()
        return
    mon.on_trip = on_trip
    async with mon:
        mon.track(pid_of)
        await body()


async def _run_bench(cfg: Config, cmd: Command, res: rs.RunResult, raw: Path, mon, opts: RunOpts) -> None:
    from mse_llm import proc as pr
    from mse_llm.parsers import parse_llama_bench, parse_llama_bench_line

    log = raw / "llama-bench.log"
    res.logs["tool"] = rs.rel(log) or str(log)
    json_lines: list[str] = []
    sp = pr.Spawned(cmd.argv, log_path=log, sink=opts.sink, memory_cap_mib=opts.mem_cap_mib)
    sp.watchers.append(lambda ln: json_lines.append(ln) if parse_llama_bench_line(ln) else None)

    stopped = asyncio.Event()

    async def trip(reason: str) -> None:
        opts.say(f"[red]■ memory guard:[/] {reason} → stopping llama-bench")
        if sp.proc is not None and sp.returncode is None:
            rep = await sp.stop(grace=2)
            if not rep.clean:
                opts.say(f"[yellow]  {rep.summary()}[/]")
        stopped.set()

    async def body() -> None:
        try:
            await sp.start()
        except pr.ProcError as e:
            res.reason = str(e)
            return
        if mon is not None and mon.tripped:
            await sp.stop(grace=2)
            return
        waiter = asyncio.create_task(sp.wait())
        try:
            rc = await _wait_or_give_up(waiter, stopped, res, sp.pid, opts)
        except asyncio.CancelledError:
            waiter.cancel()
            await sp.stop(grace=2)
            raise
        if rc is not None and rc != 0 and not (mon is not None and mon.tripped):
            res.reason = f"llama-bench exit code {rc}: {_tail_reason(list(sp.lines))}"

    await _with_monitor(mon, lambda: sp.proc.pid if sp.proc is not None else None, body, trip)
    res.tests, eff = parse_llama_bench(json_lines)
    if eff:
        res.startup = {"effective": eff}
        res.tool_version = f"llama-bench {eff.get('build_commit')} (b{eff.get('build_number')})"


STUCK_EXIT_S = 30.0      # after a guard stop: how long to wait for a process stuck in the GPU driver to exit


async def _wait_or_give_up(waiter: asyncio.Task, stopped: asyncio.Event, res: rs.RunResult, pid: int,
                           opts: RunOpts) -> int | None:
    """The process's exit code; after a guard stop at most STUCK_EXIT_S more. A process that hangs inside the
    GPU driver ignores TerminateProcess until the driver lets go (measured: ~3 min for the 35B at -ncmoe 10),
    so the job gives up waiting instead of hanging the sweep. The process ends on its own (it is in our
    kill-on-close Job Object, and was told to terminate)."""
    stop_wait = asyncio.create_task(stopped.wait())
    try:
        await asyncio.wait({waiter, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
        if waiter.done():
            return waiter.result()
        done, _ = await asyncio.wait({waiter}, timeout=STUCK_EXIT_S)
        if done:
            return waiter.result()
        waiter.cancel()
        res.note = ((res.note + "; ") if res.note else "") + (
            f"pid {pid} did not exit within {STUCK_EXIT_S:.0f} s of the kill (stuck in the GPU driver); "
            "it frees its VRAM when the driver lets go")
        opts.say(f"[yellow]  llama-bench (pid {pid}) is stuck in the GPU driver and hasn't exited yet; "
                 "moving on (it ends when the driver lets go)[/]")
        return None
    finally:
        stop_wait.cancel()


async def _run_benchy(cfg: Config, job: Job, cmd: Command, res: rs.RunResult, raw: Path, save: Path | None, mon,
                      opts: RunOpts) -> None:
    from mse_llm import proc as pr
    from mse_llm import serve as sv
    from mse_llm.parsers import parse_benchy
    from mse_llm.runners import ServerError, ServerHandle

    _, eng = cfg.engine(job.engine)
    port = opts.port or eng.port
    slog, blog = raw / "server.log", raw / "benchy.log"
    res.logs.update(server=rs.rel(slog) or str(slog), tool=rs.rel(blog) or str(blog))
    info = sv.StartupInfo()
    assert cmd.server is not None
    h = ServerHandle(cmd.server.argv, port=port, client_host=eng.client_host,
                     ready_timeout=cfg.defaults.server_ready_timeout, log_path=slog, memory_cap_mib=opts.mem_cap_mib)
    h.proc.watchers.append(info.feed)
    draft_lines: list[str] = []
    h.proc.watchers.append(lambda ln: draft_lines.append(ln) if "draft acceptance" in ln else None)
    client = pr.Spawned(cmd.argv, log_path=blog, sink=opts.sink)
    scraped: dict[str, dict | None] = {"before": None, "after": None}
    spec_on = bool(job.variant.spec_type)

    async def trip(reason: str) -> None:
        opts.say(f"[red]■ memory guard:[/] {reason} → stopping the server and benchy")
        if client.proc is not None and client.returncode is None:
            await client.stop(grace=2)
        await h.stop()

    async def body() -> None:
        try:
            opts.say(f"[dim]starting llama-server on port {port} …[/]")
            ep = await h.start()
        except ServerError as e:
            res.reason = f"server: {e}" + (f" | {_tail_reason(e.log_tail)}" if e.log_tail else "")
            return
        try:
            res.startup = {"ready_s": round(ep.ready_s, 1), "n_ctx": info.n_ctx,
                           "offloaded": list(info.offloaded) if info.offloaded else None}
            diffs = info.differences(cmd.ctx)
            if diffs:
                res.startup["differences"] = diffs
            if mon is not None and mon.tripped:
                return
            opts.say(f"[dim]server ready after {ep.ready_s:.1f} s; running llama-benchy …[/]")
            if spec_on:
                scraped["before"] = await scrape_metrics(ep.root)
            try:
                await client.start()
            except pr.ProcError as e:
                res.reason = str(e)
                return
            rc = await client.wait()
            if spec_on:
                scraped["after"] = await scrape_metrics(ep.root)
            if rc != 0 and not (mon is not None and mon.tripped):
                res.reason = f"llama-benchy exit code {rc}: {_tail_reason(list(client.lines))}"
        except asyncio.CancelledError:
            if client.proc is not None and client.returncode is None:
                await client.stop(grace=2)
            raise
        finally:
            await h.stop()

    await _with_monitor(mon, lambda: h.proc.proc.pid if h.proc.proc is not None else None, body, trip)
    if spec_on:
        res.spec = spec_record(cfg, job, scraped["before"], scraped["after"], draft_lines)
    if save is not None and save.is_file():
        try:
            res.tests, binfo = parse_benchy(save)
            res.tool_version = f"llama-benchy {binfo.get('version')}"
            res.startup = {**(res.startup or {}), "benchy": binfo}
            res.logs["result"] = rs.rel(save) or str(save)
        except ValueError as e:
            res.reason = res.reason or f"benchy JSON unreadable: {e}"


async def scrape_metrics(root: str) -> dict | None:
    """GET /metrics (Prometheus text) -> parsed counters; None when the server has it off or doesn't answer."""
    import httpx

    from mse_llm.parsers.spec import parse_metrics

    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(root + "/metrics")
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    return parse_metrics(r.text)


SAMPLING_KEYS = ("temp", "top-k", "top-p", "min-p", "repeat-penalty")


def spec_record(cfg: Config, job: Job, before: dict | None, after: dict | None,
                log_lines: list[str]) -> dict:
    """RunResult.spec (§4.5): the /metrics delta over the benchy run (per draft position), else the summed slot
    log lines; plus what produced it (type, n_max, drafter, sampling: a high temperature lowers acceptance)."""
    from mse_llm.parsers import spec as sp

    v = job.variant
    _, m = cfg.model(job.model)
    s = sp.spec_from_counters(sp.delta(before, after)) if after else None
    if s is None:
        s = sp.parse_server_log(log_lines) or {"source": "none", "drafted": 0, "accepted": 0,
                                                "note": "no draft tokens counted (server without --metrics, or "
                                                        "the drafter did not run)"}
    s.update(type=v.spec_type, n_max=v.spec_n_max, n_min=v.spec_n_min,
             draft_model=(m.draft if v.uses_draft_model else None),
             prompts=job.params.get("prompts") or "prose",
             sampling={k: m.serve_extra[k] for k in SAMPLING_KEYS if k in m.serve_extra})
    rates = s.get("accept_rate_per_pos") or []
    if rates:
        s["useful_len"] = sp.useful_draft_len(rates)
    return s


# ---------------------------------------------------------------- sweeps

@dataclass
class Planned:
    job: Job
    cmd: Command
    key: str
    action: str                     # run | skip-resume | skip-status | skip-filter
    why: str = ""


def sweep_jobs(cfg: Config, name: str, *, ctx: int | None = None, include_off: bool = False,
               force: bool = False, match: list[str] | None = None,
               prompt_sets: list[str] | None = None) -> list[tuple[Job, str, str]]:
    """Jobs of [sweeps.NAME] with the entry's status: (job, action, why). action: run | skip-status | skip-filter."""
    from mse_llm.variants import decode

    sw = cfg.sweeps[name]
    if sw.tool == "eval":
        raise ValueError(f"sweep {name}: tool eval runs in M12")
    c = ctx or sw.ctx
    out: list[tuple[Job, str, str]] = []
    if sw.tool == "bench":
        params = {"tg": sw.tg, "repetitions": sw.repetitions}
        ctxs = sw.prompts or [c]
    else:
        params = {"pp": sw.pp, "tg": sw.tg, "depth": sw.depth, "runs": sw.runs, "concurrency": sw.concurrency}
        ctxs = [c]
    sets: list[str | None] = [None]
    if sw.tool == "benchy":
        sets = [None if s == "prose" else s for s in (prompt_sets or sw.prompt_sets)] or [None]
    for e in sw.entries():
        v = decode(e.v)
        for cc, ps in ((cc, ps) for cc in ctxs for ps in sets):
            jp = dict(params) if ps is None else {**params, "prompts": ps}
            job = Job(sw.tool, sw.model, v, cc, jp, sw.engine, name, e.note or None)
            if match and not any(fnmatch.fnmatchcase(v.name, p) for p in match):
                out.append((job, "skip-filter", "not matched"))
            elif e.status == "skip" and not force:
                out.append((job, "skip-status", f"status skip{': ' + e.note if e.note else ''} (--force runs it)"))
            elif e.status == "off" and not include_off:
                out.append((job, "skip-status", "status off (--include-off runs it)"))
            else:
                out.append((job, "run", ""))
    return out


def plan(cfg: Config, jobs: list[tuple[Job, str, str]], *, resume: bool = False, only_failed: bool = False,
         port: int | None = None, env: RunEnv | None = None) -> list[Planned]:
    """Commands + resume decisions. --resume skips jobs with an `ok` result for the same key and build;
    --only-failed runs only jobs whose last result for this build failed or was aborted."""
    env = env or RunEnv()
    out: list[Planned] = []
    last: dict[str, dict[str, dict[str, Any]]] = {}
    for job, action, why in jobs:
        cmd = job.command(cfg, port=port, save_result="<raw>/benchy.json" if job.tool == "benchy" else None)
        key = job.key(cmd)
        if action == "run" and job.tool == "benchy" and job.params.get("prompts") not in (None, "prose"):
            try:
                from mse_llm import quality as qq

                qq.prompt_text(cfg, job.params["prompts"], create=False)
            except (OSError, ValueError) as e:
                cmd.errors.append(f"prompts {job.params['prompts']}: {e}")
        if action == "run" and (resume or only_failed):
            bid = env.build_for(cfg, job.engine).get("id")
            idx = last.setdefault(str(bid), rs.latest_by_key(bid))
            prev = idx.get(key)
            if resume and prev and prev.get("status") == "ok":
                action, why = "skip-resume", f"ok in {prev['run_id']}"
            elif only_failed and not (prev and prev.get("status") in ("failed", "aborted")):
                action, why = "skip-resume", "no failed run for this build" if not prev else \
                    f"last run {prev['status']} ({prev['run_id']})"
        out.append(Planned(job, cmd, key, action, why))
    return out
