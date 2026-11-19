"""llama-benchy `--format json --save-result FILE` (checked with llama-benchy 0.4.0).

One entry in "benchmarks" per (prompt_size, context_size = depth, concurrency); each metric is
{mean, std, values}. Rows with is_context_prefill_phase=true (the depth prefill of a depth test) are
skipped: they measure building the context, not the test itself."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mse_llm.results import TestRow


def _m(b: dict[str, Any], key: str) -> tuple[float | None, float | None, list[float]]:
    v = b.get(key)
    if not isinstance(v, dict) or v.get("mean") is None:
        return None, None, []
    return float(v["mean"]), (float(v["std"]) if v.get("std") is not None else None), \
        [round(float(x), 3) for x in v.get("values") or []]


def parse_benchy(data: dict[str, Any] | str | Path) -> tuple[list[TestRow], dict[str, Any]]:
    """Rows + run info (version, latency, model) from benchy's JSON (dict, text or file)."""
    if isinstance(data, Path):
        data = data.read_text(encoding="utf-8")
    if isinstance(data, str):
        data = json.loads(data)
    rows: list[TestRow] = []
    for b in data.get("benchmarks") or []:
        if b.get("is_context_prefill_phase"):
            continue
        pp, pp_sd, _ = _m(b, "pp_throughput")
        tg, tg_sd, tg_vals = _m(b, "tg_throughput")
        ttft, _, _ = _m(b, "e2e_ttft")
        peak, _, _ = _m(b, "peak_throughput")
        rows.append(TestRow(kind="pp+tg", n_prompt=int(b.get("prompt_size") or 0),
                            n_gen=int(b.get("response_size") or 0), depth=int(b.get("context_size") or 0),
                            concurrency=int(b.get("concurrency") or 1), pp_tps=pp, pp_std=pp_sd, tg_tps=tg,
                            tg_std=tg_sd, ttft_ms=ttft, peak_tps=peak, samples=tg_vals))
    info = {k: data.get(k) for k in ("version", "timestamp", "latency_mode", "latency_ms", "model",
                                     "prefix_caching_enabled", "max_concurrency") if k in data}
    return rows, info
