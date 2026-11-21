"""Picking runs from the results store and turning them into chart / table / CSV rows (M8, §8)."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

from mse_llm import results as rs
from mse_llm.parsers.legacy_log import ChartRow, config_label

STATUSES = ("ok", "aborted", "failed", "skipped")


@dataclass
class Selection:
    runs: list[rs.RunResult]
    title: str
    bench_type: str                       # llama-bench | llama-benchy (chart title, like bash)
    dropped: list[str] = field(default_factory=list)   # duplicates replaced by a newer run with the same key


def select(*, run_ids: list[str] | None = None, sweep: str | None = None, model: str | None = None,
           tool: str | None = None, build: str | None = None, statuses: tuple[str, ...] = ("ok",),
           latest: bool = True, last: int | None = None, order: list[str] | None = None) -> Selection:
    """Runs by id, or filtered from index.jsonl. latest: one run per key (the newest), so a re-run replaces the
    old bar instead of adding a second one. order: variant names (a sweep's entry order) to sort by."""
    dropped: list[str] = []
    if run_ids:
        runs = [rs.find_run(i) for i in run_ids]
    else:
        idx = [e for e in rs.read_index()
               if (sweep is None or e.get("sweep") == sweep) and (tool is None or e.get("tool") == tool)
               and (model is None or model.lower() in (e.get("model") or "").lower())
               and (build is None or (e.get("build") or "").startswith(build) or e.get("commit") == build)
               and e.get("status") in statuses]
        if latest:
            by_key: dict[str, dict[str, Any]] = {}
            for e in idx:
                if e["key"] in by_key:
                    dropped.append(by_key[e["key"]]["run_id"])
                by_key[e["key"]] = e
            idx = [e for e in idx if by_key[e["key"]] is e]
        if last:
            idx = idx[-last:]
        runs = [rs.find_run(e["run_id"]) for e in idx]
    if order:
        pos = {v: i for i, v in enumerate(order)}
        runs.sort(key=lambda r: (pos.get(r.variant, len(pos)), r.ctx or 0, r.timestamp))
    tools = {r.tool for r in runs}
    bench_type = "llama-benchy" if tools == {"benchy"} else "llama-bench"
    title = sweep or (", ".join(sorted({r.model for r in runs})) if runs else "no runs")
    return Selection(runs, title, bench_type, dropped)


def _peak(r: rs.RunResult, key: str) -> float:
    st = ((r.monitor or {}).get("stats") or {}).get(key) or {}
    return float(st.get("peak") or 0.0)


def chart_row(r: rs.RunResult) -> ChartRow:
    """A run as one bar group, labelled the way bash labels the same command (but with the real ctx: bash showed
    llama-bench's -p as "Ctx:default"). Bars: duration = time under the monitor minus the server start, startup =
    the server's ready time (benchy), VRAM = peak dedicated, shared = peak of the bash shared number
    (Non Local Usage sum)."""
    text = " ".join([*(r.server_argv or []), *r.argv])
    ready = float((r.startup or {}).get("ready_s") or 0.0)
    mon_dur = float((r.monitor or {}).get("duration_s") or r.duration_s or 0.0)
    m = r.metrics
    return ChartRow(config_id=config_label(text, r.ctx), config_a=f"{r.model} [{r.engine}]",
                    duration=max(0.0, mon_dur - ready), server_startup=ready,
                    pp=float(m.get("pp_tps") or 0.0), tg=float(m.get("tg_tps") or 0.0),
                    vram=_peak(r, "vram_used_mib"), svram=_peak(r, "nonlocal_mib") or _peak(r, "shared_mib"),
                    status=r.status)


# ---------------------------------------------------------------- table / CSV

TABLE_COLS = ("run_id", "tool", "model", "variant", "ctx", "pp_tps", "pp_std", "tg_tps", "tg_std", "ttft_ms",
              "acceptance", "mean_accept_len", "mean_kld", "same_top_p", "hit_rate", "vram_peak_mib",
              "server_shared_mib", "spill_mib", "duration_s", "fit", "status", "note", "build")


def table_row(r: rs.RunResult) -> dict[str, Any]:
    m, mon = r.metrics, r.monitor or {}
    spill = mon.get("spill_mib") if mon.get("expected_host_mib") is not None else None
    return {"run_id": r.run_id, "tool": r.tool, "model": r.model, "variant": r.variant, "ctx": r.ctx,
            "pp_tps": m.get("pp_tps"), "pp_std": m.get("pp_std"), "tg_tps": m.get("tg_tps"),
            "tg_std": m.get("tg_std"), "ttft_ms": m.get("ttft_ms"),
            "acceptance": (r.spec or {}).get("acceptance_rate"),
            "mean_accept_len": (r.spec or {}).get("mean_accept_len"),
            "mean_kld": (r.quality or {}).get("mean_kld"), "same_top_p": (r.quality or {}).get("same_top_p"),
            "hit_rate": (r.moe_cache or {}).get("hit_rate"),
            "vram_peak_mib": _peak(r, "vram_used_mib") or None,
            "server_shared_mib": _peak(r, "proc_shared_mib") or None, "spill_mib": spill,
            "duration_s": r.duration_s, "fit": (r.fit or {}).get("verdict"), "status": r.status,
            "note": r.reason or r.note, "build": (r.build.get("commit") or "")[:9] or None}


def test_rows(r: rs.RunResult) -> list[dict[str, Any]]:
    """One row per measured test (llama-bench pp / tg line, benchy prompt size × depth)."""
    base = {"run_id": r.run_id, "tool": r.tool, "model": r.model, "variant": r.variant, "ctx": r.ctx,
            "status": r.status, "build": (r.build.get("commit") or "")[:9] or None}
    return [{**base, **t.model_dump(exclude={"samples"})} for t in r.tests]


def to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    cols = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def best(rows: list[dict[str, Any]], key: str) -> str | None:
    ok = [r for r in rows if r.get(key) is not None and r.get("status") == "ok"]
    return max(ok, key=lambda r: r[key])["run_id"] if ok else None


# ---------------------------------------------------------------- compare

def compare(a: rs.RunResult, b: rs.RunResult) -> list[tuple[str, Any, Any, str]]:
    """(field, A, B, delta) for what differs or matters: settings, build, headline numbers, memory peaks."""
    out: list[tuple[str, Any, Any, str]] = []

    def add(name: str, x: Any, y: Any, numeric: bool = False) -> None:
        d = ""
        if numeric and isinstance(x, (int, float)) and isinstance(y, (int, float)) and x:
            d = f"{(y - x) / x * 100:+.1f} %"
        if numeric or x != y:
            out.append((name, x, y, d))

    add("model", a.model, b.model)
    add("variant", a.variant, b.variant)
    for k in sorted(set(a.variant_fields) | set(b.variant_fields)):
        add(f"  {k}", a.variant_fields.get(k), b.variant_fields.get(k))
    add("ctx", a.ctx, b.ctx)
    add("params", a.params, b.params)
    add("build", a.build.get("id"), b.build.get("id"))
    add("hw", a.hw.get("hash"), b.hw.get("hash"))
    add("status", a.status, b.status)
    for k in ("pp_tps", "tg_tps", "ttft_ms"):
        if k in a.metrics or k in b.metrics:
            add(k, a.metrics.get(k), b.metrics.get(k), numeric=True)
    for k in ("vram_used_mib", "proc_shared_mib", "ram_used_mib", "gpu_temp_c", "gpu_power_w"):
        x, y = _peak(a, k), _peak(b, k)
        if x or y:
            add(f"peak {k}", x, y, numeric=True)
    add("duration_s", a.duration_s, b.duration_s, numeric=True)
    return out
