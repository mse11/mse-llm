"""Results store (§8): one JSON per run under results/runs/YYYY-MM/, one line per run in results/index.jsonl.

The tool only writes files. results/ is git-ignored for now (2026-11-19); the split below is what would be
tracked later. Large raw data (logs, monitor time series, benchy's own JSON) goes to results/raw/<run_id>/.

Layout (MSE_LLM_RESULTS_DIR moves the whole tree, used by the tests):
    results/index.jsonl                 small: one line per run
    results/runs/2026-11/<run_id>.json  small: RunResult
    results/builds/<build_id>.json      small: copy of the build's build-manifest.json
    results/hw/<hw_hash>.json           small: hardware identity the runs refer to
    results/raw/<run_id>/               large: llama-bench / server / benchy logs, monitor csv
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import secrets
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mse_llm.util import project_root, stable_hash

Status = Literal["ok", "failed", "aborted", "skipped"]
SCHEMA = 1


def results_root() -> Path:
    env = os.environ.get("MSE_LLM_RESULTS_DIR")
    return Path(env) if env else project_root() / "results"


def rel(p: Path | str | None) -> str | None:
    """Path relative to the repo when inside it (results JSON stays machine-independent)."""
    if p is None:
        return None
    p = Path(p)
    try:
        return p.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return str(p)


class _M(BaseModel):
    model_config = ConfigDict(extra="allow")


class TestRow(_M):
    """One measured test: a llama-bench line (pp or tg) or one benchy benchmark (pp + tg at a depth)."""
    __test__ = False                       # not a pytest test class
    kind: Literal["pp", "tg", "pg", "pp+tg"]
    n_prompt: int = 0
    n_gen: int = 0
    depth: int = 0
    concurrency: int = 1
    pp_tps: float | None = None
    pp_std: float | None = None
    tg_tps: float | None = None
    tg_std: float | None = None
    ttft_ms: float | None = None
    peak_tps: float | None = None
    samples: list[float] = Field(default_factory=list)


class RunResult(_M):
    schema_: int = Field(SCHEMA, alias="schema")
    run_id: str
    key: str                               # same key + same build = same measurement (sweep --resume)
    timestamp: str
    sweep: str | None = None
    tool: Literal["bench", "benchy", "eval"]
    engine: str
    model: str
    model_path: str | None = None
    model_identity: dict[str, Any] | None = None
    variant: str
    variant_fields: dict[str, Any] = Field(default_factory=dict)
    ctx: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)       # tg, prompts, pp, depth, runs, repetitions …
    argv: list[str] = Field(default_factory=list)
    server_argv: list[str] | None = None
    env: dict[str, str] = Field(default_factory=dict)
    build: dict[str, Any] = Field(default_factory=dict)        # id, commit, describe, engine, binary sha
    hw: dict[str, Any] = Field(default_factory=dict)           # hash, soft_hash
    tool_version: str | None = None
    status: Status = "ok"
    reason: str | None = None
    note: str | None = None                                    # the sweep entry's note (bash comment)
    duration_s: float | None = None
    tests: list[TestRow] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)      # headline pp_tps / tg_tps of the run
    fit: dict[str, Any] | None = None
    startup: dict[str, Any] | None = None
    monitor: dict[str, Any] | None = None
    logs: dict[str, str] = Field(default_factory=dict)

    def index_line(self) -> dict[str, Any]:
        m = self.metrics
        mon = self.monitor or {}
        return {"run_id": self.run_id, "timestamp": self.timestamp, "key": self.key, "sweep": self.sweep,
                "tool": self.tool, "engine": self.engine, "model": self.model, "variant": self.variant,
                "ctx": self.ctx, "build": self.build.get("id"), "commit": (self.build.get("commit") or "")[:9] or None,
                "hw": self.hw.get("hash"), "pp_tps": m.get("pp_tps"), "tg_tps": m.get("tg_tps"),
                "n_prompt": m.get("n_prompt"), "vram_peak_mib": _peak(mon, "vram_used_mib"),
                "spill_mib": mon.get("spill_mib"), "status": self.status, "reason": self.reason}

    def path(self, root: Path | None = None) -> Path:
        return (root or results_root()) / "runs" / self.timestamp[:7] / f"{self.run_id}.json"


def _peak(mon: dict[str, Any], key: str) -> float | None:
    st = (mon.get("stats") or {}).get(key)
    return st.get("peak") if isinstance(st, dict) else None


def headline(tests: list[TestRow]) -> dict[str, Any]:
    """PP of the largest prompt and TG of the first generation test (depth 0 first): the numbers the bash
    5-axis chart shows per variant."""
    if not tests:
        return {}
    pp = [t for t in tests if t.pp_tps is not None]
    tg = [t for t in tests if t.tg_tps is not None]
    out: dict[str, Any] = {}
    if pp:
        b = max(pp, key=lambda t: (-t.depth, t.n_prompt))
        out.update(pp_tps=round(b.pp_tps, 2), pp_std=_r(b.pp_std), n_prompt=b.n_prompt)
        if b.ttft_ms is not None:
            out["ttft_ms"] = round(b.ttft_ms, 1)
    if tg:
        b = min(tg, key=lambda t: (t.depth, -t.n_prompt if t.kind == "pp+tg" else 0))
        out.update(tg_tps=round(b.tg_tps, 2), tg_std=_r(b.tg_std), n_gen=b.n_gen)
    return out


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 2)


def run_key(tool: str, model: str, variant: str, ctx: int | None, params: dict[str, Any]) -> str:
    return stable_hash({"tool": tool, "model": model, "variant": variant, "ctx": ctx,
                        "params": {k: v for k, v in sorted(params.items()) if v not in (None, [], {})}}, 12)


def now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC).astimezone()


def new_run_id(tool: str, t: _dt.datetime | None = None) -> str:
    t = t or now()
    return f"{t:%Y%m%d-%H%M%S}-{tool}-{secrets.token_hex(2)}"


def raw_dir(run_id: str, root: Path | None = None) -> Path:
    return (root or results_root()) / "raw" / run_id


# ---------------------------------------------------------------- write / read

def write_run(r: RunResult, root: Path | None = None) -> Path:
    root = root or results_root()
    p = r.path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(r.model_dump_json(indent=2, by_alias=True, exclude_none=True) + "\n", encoding="utf-8")
    with (root / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(r.index_line(), ensure_ascii=False) + "\n")
    return p


def read_index(root: Path | None = None) -> list[dict[str, Any]]:
    p = (root or results_root()) / "index.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def find_run(run_id: str, root: Path | None = None) -> RunResult:
    """By full id or a unique prefix / suffix."""
    root = root or results_root()
    ids = [e["run_id"] for e in read_index(root)]
    hits = [i for i in ids if i == run_id] or sorted({i for i in ids if i.startswith(run_id) or i.endswith(run_id)})
    if not hits:
        raise KeyError(f"no run {run_id!r} in {root / 'index.jsonl'}")
    if len(hits) > 1:
        raise KeyError(f"{run_id!r} matches {len(hits)} runs: {', '.join(hits[:5])}")
    rid = hits[0]
    p = root / "runs" / f"{rid[:4]}-{rid[4:6]}" / f"{rid}.json"
    if not p.is_file():
        found = list((root / "runs").glob(f"*/{rid}.json"))
        if not found:
            raise KeyError(f"{rid}: listed in index.jsonl, but its JSON is missing")
        p = found[0]
    return RunResult.model_validate_json(p.read_text(encoding="utf-8"))


def latest_by_key(build_id: str | None, root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Last index entry per run key for one build (None = any build)."""
    out: dict[str, dict[str, Any]] = {}
    for e in read_index(root):
        if build_id is None or e.get("build") == build_id:
            out[e["key"]] = e
    return out


# ---------------------------------------------------------------- build + hardware records

_VOLATILE = {"built_at", "duration_s", "step_durations_s", "steps_run"}


def build_record(bin_dir: Path, engine: str, root: Path | None = None) -> dict[str, Any]:
    """Build info for a run, from the engine's build-manifest.json; a copy goes to results/builds/<id>.json."""
    mp = Path(bin_dir) / "build-manifest.json"
    if not mp.is_file():
        return {"id": None, "engine": engine, "bin_dir": rel(bin_dir), "manifest": None}
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"id": None, "engine": engine, "bin_dir": rel(bin_dir), "manifest": "unreadable"}
    bid = f"{(m.get('commit') or 'unknown')[:9]}-{stable_hash({k: v for k, v in m.items() if k not in _VOLATILE}, 8)}"
    dst = (root or results_root()) / "builds" / f"{bid}.json"
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    return {"id": bid, "engine": engine, "commit": m.get("commit"), "describe": m.get("describe"),
            "profile": m.get("profile"), "variants": m.get("variants"), "without": m.get("without")}


def hw_record(root: Path | None = None) -> dict[str, Any]:
    """Hardware identity (§5.1) of this PC; the full identity goes to results/hw/<hash>.json."""
    from mse_llm import hw

    try:
        info = hw.detect()
        ident = hw.identity(info)
    except Exception as e:        # never fatal for a benchmark
        return {"hash": None, "error": str(e)}
    dst = (root or results_root()) / "hw" / f"{ident['hash']}.json"
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(ident, indent=2) + "\n", encoding="utf-8")
    nv = info.nvidia_gpus()
    ac = getattr(info.power, "ac_online", None)
    power = None if ac is None else ("AC power" if ac else "on battery")
    return {"hash": ident["hash"], "soft_hash": ident["soft_hash"], "driver": ident["soft"].get("driver"),
            "gpu": nv[0].name if nv else None, "power": power,
            "power_plan": getattr(info.power, "power_plan", None)}
