"""`mse-llm sweep search` (§9 item 6, M9): the manual "SWEETSPOT" hunt, automated and safe.

    1. `predict` (§4.6) measures the memory grid and picks the order of the settings: the axis with the larger
       predicted gain first (-ncmoe for models with a high n_used / n_expert, -ub when PCIe uploads limit PP).
    2. Each stage starts from the previous winner, on the safe side (all experts on the CPU, the smallest -ub),
       and steps toward the edge. A config that `predict` marks no-fit or tight is never started; every job also
       gets the normal fit check (llama-fit-params) and the memory guard (§5).
    3. In each direction the stage stops at the edge: the first predicted no-fit / tight config, or the first run
       that spills, trips the guard, fails (CUDA out of memory with "Prefer No Sysmem Fallback") or is skipped by
       the fit check. It also stops when the goal metric stopped improving for `patience` steps.
    4. The winner is taken one step back when the best run is right next to a *real* failure (a run that spilled,
       tripped the guard, failed or was skipped by the fit check; your notes show sporadic freezes right next to
       configs that broke). An edge that is only predicted (never run) does not move the winner: the best run
       already passed the fit check with the margin (measured: gemma moe10 TG 67.9 with 1075 MiB left lost to
       moe15 at 50.8 when the predicted edge counted too).

Every step is a normal RunResult in results/ (sweep "search:<model>"), so an earlier ok run with the same
settings and build is reused instead of run again. The whole search (steps, edges, winner) is written to
results/search/<model>/<build>__<hw>.json.
"""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from mse_llm import predict as pr
from mse_llm import results as rs
from mse_llm.bench import Job
from mse_llm.variants import Variant

AXES = ("ncmoe", "ub")
MIN_GAIN = 0.01                       # a step counts as an improvement above +1 %


@dataclass
class Step:
    axis: str
    variant: str
    ncmoe: int | None
    ub: int
    status: str                       # ok | failed | aborted | skipped | predicted no-fit | predicted tight | max-runs
    predicted_tg: float | None = None
    predicted_pp: float | None = None
    predicted_left_mib: float | None = None
    tg: float | None = None
    pp: float | None = None
    run_id: str | None = None
    reused: bool = False
    reason: str | None = None
    vram_peak_mib: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def metric(self, goal: str) -> float:
        tg, pp = self.tg or 0.0, self.pp or 0.0
        return tg if goal == "tg" else pp if goal == "pp" else math.sqrt(tg * pp)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("predicted_tg", "predicted_pp", "tg", "pp", "predicted_left_mib"):
            if d[k] is not None:
                d[k] = round(d[k], 2)
        return d


@dataclass
class Stage:
    axis: str
    values: list[int]
    fixed: dict[str, int | None]
    steps: list[Step] = field(default_factory=list)
    edge: str | None = None           # variant where the edge was found
    edge_reason: str | None = None
    edge_real: bool = False           # the edge is a run that failed (not only a prediction)
    winner: str | None = None
    stepped_back: bool = False
    stop: str | None = None           # why the stage ended (edge / no gain / max runs / end of range)

    def as_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "values": self.values, "fixed": self.fixed, "edge": self.edge,
                "edge_reason": self.edge_reason, "edge_real": self.edge_real, "winner": self.winner,
                "stepped_back": self.stepped_back,
                "stop": self.stop, "steps": [s.as_dict() for s in self.steps]}


@dataclass
class SearchResult:
    model: str
    ctx: int
    tool: str
    goal: str
    base: str
    order: list[str]
    order_why: str
    stages: list[Stage]
    winner: str | None = None
    winner_step: Step | None = None
    simulated: bool = False
    build: str | None = None
    hw: str | None = None
    timestamp: str | None = None
    path: str | None = None
    runs: int = 0
    reused: int = 0

    @property
    def any_spill(self) -> bool:
        return any(s.status == "aborted" for st in self.stages for s in st.steps)

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "ctx": self.ctx, "tool": self.tool, "goal": self.goal, "base": self.base,
                "order": self.order, "order_why": self.order_why, "winner": self.winner,
                "winner_metrics": self.winner_step.as_dict() if self.winner_step else None,
                "simulated": self.simulated, "build": self.build, "hw": self.hw, "timestamp": self.timestamp,
                "runs": self.runs, "reused": self.reused, "any_spill": self.any_spill,
                "stages": [s.as_dict() for s in self.stages]}


# ---------------------------------------------------------------- plan

def axis_order(pred: pr.Prediction, goal: str, ubs: list[int]) -> tuple[list[str], str]:
    """The axis with the larger predicted gain of the goal metric, measured from the safe corner."""
    s = pred.shape
    if not s.moe_layers:
        return ["ub"], "dense model: only -ub"
    L, u0 = s.moe_layers, min(ubs)
    start = pred.at(L, u0)
    n_min = pred.lowest_fitting_ncmoe(u0)
    fit_ubs = [u for u in ubs if pred.at(L, u).fits]
    if not start.fits or n_min is None:
        return ["ncmoe", "ub"], "the safe corner doesn't fit: -ncmoe order kept"
    g_n = pr.score(pred.at(n_min, u0), goal) / (pr.score(start, goal) or 1)
    g_u = max((pr.score(pred.at(L, u), goal) for u in fit_ubs), default=0) / (pr.score(start, goal) or 1)
    order = ["ncmoe", "ub"] if g_n >= g_u else ["ub", "ncmoe"]
    why = (f"predicted {goal} gain from the safe corner (moe{L}, ub {u0}): -ncmoe → {n_min}: x{g_n:.2f}, "
           f"-ub → {max(fit_ubs) if fit_ubs else u0}: x{g_u:.2f} → {' then '.join(order)}")
    return order, why


def ncmoe_values(pred: pr.Prediction, ub: int, steps: int, start: int | None = None,
                 explicit: list[int] | None = None) -> list[int]:
    """Safe (high) → edge (low): `steps` even steps from `start` to the lowest predicted fit, plus one value past
    it (which `predict` marks no-fit, so it is the predicted edge and never run)."""
    L = pred.shape.moe_layers
    if explicit:
        return sorted(set(explicit), reverse=True)
    top = L if start is None else start
    lo = pred.lowest_fitting_ncmoe(ub)
    if lo is None:
        return [top]
    if lo >= top:
        return [top]
    n = max(1, steps)
    vals = sorted({round(top - k * (top - lo) / n) for k in range(n + 1)}, reverse=True)
    if lo > 0:
        vals.append(lo - 1)
    return vals


def ub_values(ubs: list[int], start: int) -> list[int]:
    return [u for u in sorted(set(ubs)) if u >= start]


# ---------------------------------------------------------------- the search

RunFn = Callable[[Job], Awaitable[rs.RunResult]]
ReuseFn = Callable[[Job], "rs.RunResult | None"]


def _metric_from(r: rs.RunResult) -> tuple[float | None, float | None]:
    return r.metrics.get("tg_tps"), r.metrics.get("pp_tps")


def _peak(r: rs.RunResult) -> float | None:
    return (((r.monitor or {}).get("stats") or {}).get("vram_used_mib") or {}).get("peak")


async def search(pred: pr.Prediction, *, model: str, tool: str = "bench", goal: str = "both",
                 axes: list[str] | None = None, ubs: list[int] | None = None, ncmoes: list[int] | None = None,
                 steps: int = 4, patience: int = 2, max_runs: int = 20, step_back: bool = True,
                 params: dict[str, Any] | None = None, engine: str | None = None,
                 run: RunFn, reuse: ReuseFn | None = None, say: Callable[[str], None] = lambda s: None,
                 simulated: bool = False) -> SearchResult:
    s = pred.shape
    ubs = sorted(set(ubs or pr.DEFAULT_UBS))
    if axes:
        order, why = [a for a in axes if a in AXES and (a != "ncmoe" or s.moe_layers)], "--axes"
    else:
        order, why = axis_order(pred, goal, ubs)
    base = pred.base_variant or Variant()
    cur_n: int | None = s.moe_layers if s.moe_layers else None
    cur_u = min(ubs)
    res = SearchResult(model, pred.ctx, tool, goal, pred.base, order, why, [], simulated=simulated)
    budget = [max_runs]
    seen: dict[str, Step] = {}                # a config measured in an earlier stage (the stage start)

    async def one(axis: str, n: int | None, u: int) -> Step:
        pt = pred.at(n, u, base)
        if pt.variant.name in seen:
            return replace(seen[pt.variant.name], axis=axis, reused=True)
        st = Step(axis, pt.variant.name, n, u, "", pt.tg, pt.pp, pt.left_mib)
        seen[st.variant] = st
        if not pt.fits:
            st.status, st.reason = f"predicted {pt.verdict.rstrip('~')}", (
                f"GPU {pt.gpu_mib:.0f} MiB, {pt.left_mib:.0f} MiB left (margin {pred.margin_mib})"
                if pt.left_mib is not None else pt.error)
            return st
        job = Job(tool, model, pt.variant, pred.ctx, dict(params or {}), engine, f"search:{model}",
                  f"sweep search: {axis} stage")
        prev = reuse(job) if reuse else None
        if prev is None and budget[0] <= 0:
            st.status, st.reason = "max-runs", "--max-runs reached"
            return st
        if prev is None:
            budget[0] -= 1
            say(f"[bold]▶ {axis}: {pt.variant.name}[/]  [dim](predicted TG {pt.tg:.1f}, PP {pt.pp:.0f}, "
                f"{pt.left_mib:.0f} MiB left)[/]")
            r = await run(job)
            res.runs += 1
        else:
            r, st.reused = prev, True
            res.reused += 1
            say(f"[dim]  reuse {pt.variant.name}: {prev.run_id}[/]")
        st.status, st.run_id, st.reason, st.vram_peak_mib = r.status, r.run_id, r.reason, _peak(r)
        st.tg, st.pp = _metric_from(r)
        if st.ok and st.tg is None and st.pp is None:
            st.status, st.reason = "failed", "no TG / PP in the result"
        return st

    for axis in order:
        if axis == "ncmoe":
            vals = ncmoe_values(pred, cur_u, steps, start=s.moe_layers, explicit=ncmoes)
            fixed = {"ub": cur_u}
        else:
            vals = ub_values(ubs, cur_u)
            fixed = {"ncmoe": cur_n}
        stage = Stage(axis, vals, fixed)
        res.stages.append(stage)
        best: Step | None = None
        no_gain = 0
        for val in vals:
            n, u = (val, cur_u) if axis == "ncmoe" else (cur_n, val)
            st = await one(axis, n, u)
            stage.steps.append(st)
            if st.status == "max-runs":
                stage.stop = "max runs"
                break
            if not st.ok:
                stage.edge, stage.edge_reason = st.variant, f"{st.status}: {st.reason or ''}".strip(": ")
                stage.edge_real = not st.status.startswith("predicted")
                stage.stop = "edge"
                break
            if best is None or st.metric(goal) > best.metric(goal) * (1 + MIN_GAIN):
                best, no_gain = st, 0
            else:
                no_gain += 1
                if no_gain >= patience:
                    stage.stop = f"no gain for {patience} steps"
                    break
        else:
            stage.stop = "end of range"
        ok = [x for x in stage.steps if x.ok]
        if ok:
            win = max(ok, key=lambda x: x.metric(goal))
            if step_back and stage.edge_real and win is ok[-1] and len(ok) > 1:
                win, stage.stepped_back = ok[-2], True
            stage.winner = win.variant
            if axis == "ncmoe":
                cur_n = win.ncmoe
            else:
                cur_u = win.ub
            res.winner, res.winner_step = win.variant, win
    return res


# ---------------------------------------------------------------- simulation (dry run) + storage

def simulated_runner(pred: pr.Prediction) -> RunFn:
    """--dry-run: every job "returns" the predicted TG / PP, so the stages show the path predict expects."""
    async def run(job: Job) -> rs.RunResult:
        v = job.variant
        pt = pred.at(v.n_cpu_moe, v.ub or pr.DEFAULT_UB, pred.base_variant)
        return rs.RunResult(run_id="(predicted)", key="-", timestamp="-", tool=job.tool, engine="-", model=job.model,
                            variant=v.name, status="ok", metrics={"tg_tps": pt.tg, "pp_tps": pt.pp})
    return run


def search_path(model: str, build: str | None, hw: str | None, root: Path | None = None) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in model)
    return (root or rs.results_root()) / "search" / safe / f"{build or 'nobuild'}__{hw or 'nohw'}.json"


def write_search(res: SearchResult, pred: pr.Prediction, root: Path | None = None) -> Path:
    """The search as data (per model, build and hardware): steps, edges and winner, plus the prediction used."""
    p = search_path(res.model, res.build, res.hw, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    hist: list[dict[str, Any]] = []
    if p.is_file():
        try:
            hist = json.loads(p.read_text(encoding="utf-8")).get("history", [])
        except (OSError, ValueError):
            hist = []
    doc = res.as_dict()
    doc["prediction"] = {"calibration": pred.calibration.as_dict(), "free_mib": pred.free_mib,
                         "margin_mib": pred.margin_mib, "recommendations": pred.recommendations}
    doc["history"] = hist + [{"timestamp": res.timestamp, "winner": res.winner,
                              "edges": {st.axis: st.edge for st in res.stages}}]
    p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    res.path = rs.rel(p)
    return p


def set_default_serve(config_path: Path, model_key: str, variant: str) -> str:
    """Write `default_serve = "<variant>"` into [models.<key>] of the config file (text edit, keeps comments).
    Returns the old value ('' when the key was not set)."""
    import re

    text = config_path.read_text(encoding="utf-8")
    hdr = re.compile(rf"^\[models\.{re.escape(model_key)}\]\s*$", re.M)
    m = hdr.search(text)
    if not m:
        raise KeyError(f"[models.{model_key}] not found in {config_path}")
    nxt = re.compile(r"^\[", re.M).search(text, m.end())
    end = nxt.start() if nxt else len(text)
    body = text[m.end():end]
    line = re.compile(r'^default_serve\s*=\s*"([^"]*)"(.*)$', re.M)
    lm = line.search(body)
    if lm:
        old = lm.group(1)
        new_body = body[:lm.start()] + f'default_serve = "{variant}"   # sweep search (was {old})' + body[lm.end():]
    else:
        old = ""
        new_body = body.rstrip("\n") + f'\ndefault_serve = "{variant}"   # sweep search\n' + (
            "\n" if body.endswith("\n\n") else "")
    config_path.write_text(text[:m.end()] + new_body + text[end:], encoding="utf-8")
    return old
