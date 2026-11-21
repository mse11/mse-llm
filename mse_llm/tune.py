"""`mse-llm tune moe-cache` (§4.4, M11): llama.cpp's scripts/moe-cache-tune.py around the normal server argv.

    mse-llm tune moe-cache MODEL -v VARIANT [--max-mib N] [--ctx N]

1. The variant's llama-server argv is built as for a benchmark (same -t, -c, placement, load mode, K/V), without
   --host / --port / --metrics / -lv and without mc / cs (the tuner sets those itself, 2 sizes x shares 0 / 0.5 /
   0.95 = 6 server runs).
2. The script runs from the engine's llama.cpp checkout through mse_llm.tune_shim (Windows: SIGINT -> Ctrl+C, so the
   servers exit cleanly and print their cache counters), under the monitor + memory guard. Its run directory is
   results/raw/tune/<model>__<hash of the argv>: the 6 calibration runs are reused while the command stays the same,
   so another --max-mib is instant.
3. `recommended: --moe-cache-mib N --moe-cache-cpu-share F` becomes the variant `<VARIANT>_mc<N>_cs<F*100>`, stored
   in the results store (tool "tune") and as results/tune/<model>.json, which a staged sweep reads as `mc{tuned}` /
   `cs{tuned}`.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from mse_llm import results as rs
from mse_llm.config import Config
from mse_llm.util import stable_hash
from mse_llm.variants import Variant

# flags the tuner sets itself, or that would make its servers differ from what it expects
_DROP_WITH_VALUE = {"--host", "--port", "-lv", "--verbosity", "--log-verbosity", "--moe-cache-mib",
                    "--moe-cache-cpu-share"}
_DROP_FLAG = {"--metrics"}

_RECOMMENDED = re.compile(r"recommended: `?--moe-cache-mib (\d+) --moe-cache-cpu-share ([0-9.]+)`? \(([0-9.]+) t/s"
                          r"(?:, ([0-9.]+) with share 0)?\)")
_ALL_FIT = re.compile(r"all experts fit on (\S+) \((\d+) MiB\)")
_MACHINE = re.compile(r"machine: uploads ([0-9.]+) GiB/s, CPU ([0-9.]+) GiB/s")
_HEAD = re.compile(r"`?([^`:]+)`?: ([0-9.]+) MiB of experts, (\S+) (\d+) MiB free, cache up to (\d+) MiB")
_CALIB = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)%\s*$")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class TuneError(ValueError):
    pass


def script_path(cfg: Config, engine: str | None) -> Path:
    """scripts/moe-cache-tune.py of the engine's checkout (bin_dir = <checkout>/build*/bin/<config>)."""
    _, eng = cfg.engine(engine)
    b = eng.bin_path()
    for up in (b, *b.parents):
        cand = up / "scripts" / "moe-cache-tune.py"
        if cand.is_file():
            return cand
    raise TuneError(f"scripts/moe-cache-tune.py not found above {b} (llama.cpp 1.0+ checkout needed)")


def strip_argv(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in _DROP_WITH_VALUE:
            i += 2
            continue
        if a not in _DROP_FLAG:
            out.append(a)
        i += 1
    return out


def base_variant(v: Variant) -> Variant:
    if v.spec_type or v.has_draft_fields:
        raise TuneError(f"{v.name}: tune without speculative decoding (the tuner times plain generation)")
    if not v.cpu_moe and v.n_cpu_moe is None:
        raise TuneError(f"{v.name}: the MoE cache needs experts in host memory: add cmoe or moe<N>")
    return replace(v, moe_cache_mib=None, moe_cache_cpu_share=None)


def tune_command(cfg: Config, model: str, v: Variant, ctx: int | None, engine: str | None, *,
                 max_mib: int | None = None, markdown: bool = False) -> dict[str, Any]:
    """The server argv for the tuner, the full command, and the run directory."""
    from mse_llm.runners import server_command

    base = base_variant(v)
    srv = server_command(cfg, model, base, ctx, engine, fit="off", purpose="tune")
    server_argv = strip_argv(srv.argv)
    mkey = srv.model or model
    out = rs.results_root() / "raw" / "tune" / f"{mkey}__{stable_hash(server_argv, 10)}"
    script = script_path(cfg, engine)
    argv = [sys.executable, "-m", "mse_llm.tune_shim", str(script), "--out", str(out)]
    if max_mib:
        argv += ["--max-mib", str(max_mib)]
    if markdown:
        argv += ["--markdown"]
    argv += ["--", *server_argv]
    return {"model": mkey, "engine": engine, "base": base, "server": srv, "server_argv": server_argv, "argv": argv,
            "out": out, "script": script, "errors": list(srv.errors), "warnings": list(srv.warnings)}


def parse_output(lines: list[str], base: Variant) -> dict[str, Any]:
    """The tuner's summary: calibration runs, machine speeds, recommendation (or "all experts fit")."""
    out: dict[str, Any] = {"calibration": []}
    for raw in lines:
        ln = _ANSI.sub("", raw)
        if m := _HEAD.search(ln):
            out.update(experts_mib=float(m.group(2)), device=m.group(3), free_mib=int(m.group(4)),
                       cap_mib=int(m.group(5)))
        elif m := _CALIB.match(ln):
            out["calibration"].append({"mib": int(m.group(1)), "share": float(m.group(2)), "tg_tps": float(m.group(3)),
                                       "hit_pct": int(m.group(4))})
        elif m := _MACHINE.search(ln):
            out["machine"] = {"upload_gib_s": float(m.group(1)), "cpu_gib_s": float(m.group(2)), "line": ln.strip()}
        elif m := _RECOMMENDED.search(ln):
            mib, share = int(m.group(1)), float(m.group(2))
            out.update(recommended_mib=mib, recommended_share=share, tg_tps=float(m.group(3)),
                       tg_tps_share0=float(m.group(4)) if m.group(4) else None)
            out["variant"] = replace(base, moe_cache_mib=mib, moe_cache_cpu_share=round(share * 100)).name
        elif m := _ALL_FIT.search(ln):
            out["all_fit"] = {"device": m.group(1), "mib": int(m.group(2))}
            out["advice"] = "all experts fit on the GPU: drop cmoe / moe<N> instead of using a cache"
    return out


_FAIL = re.compile(r"do not fit the time model|exited \(|no MoE cache|no room on|not in the log|needs -m|"
                   r"Traceback|Error", re.I)
_FIT_WITHIN = re.compile(r"fit within ([0-9.]+)%")


def failure_reason(lines: list[str]) -> str:
    """The tuner's own message (it exits with sys.exit(text)), plus how far off its time model was."""
    clean = [_ANSI.sub("", x).strip() for x in lines if x.strip()]
    msg = next((x for x in reversed(clean) if _FAIL.search(x)), clean[-1] if clean else "no output")
    fit = next((m.group(1) for x in clean if (m := _FIT_WITHIN.search(x))), None)
    if fit and "time model" in msg:
        msg += (f" (its model is {fit} % off, the limit is 10 %: run times varied, e.g. a quiet power plan or other "
                "load; the measured runs are kept, delete the run directory to measure again)")
    return msg[:500]


def save_tuned(model: str, rec: dict[str, Any], run_id: str) -> Path:
    from mse_llm.staged import tuned_path

    p = tuned_path(model)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"model": model, "variant": rec["variant"], "run_id": run_id,
                             "recommended_mib": rec.get("recommended_mib"),
                             "recommended_share": rec.get("recommended_share"), "tg_tps": rec.get("tg_tps"),
                             "timestamp": rs.now().isoformat(timespec="seconds")}, indent=2) + "\n",
                 encoding="utf-8")
    return p
