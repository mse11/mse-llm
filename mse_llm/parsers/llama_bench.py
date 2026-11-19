"""llama-bench `-o jsonl`: one JSON object per test on stdout (the log goes to stderr).

Each line has n_prompt / n_gen / n_depth, avg_ts / stddev_ts and samples_ts, plus the build and the
settings llama-bench actually used (type_k, n_ubatch, n_cpu_moe, ...), which are kept as `effective`."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from mse_llm.results import TestRow

# settings llama-bench reports back: compared with what the variant asked for
EFFECTIVE_KEYS = ("build_commit", "build_number", "backends", "gpu_info", "model_type", "model_size",
                  "model_n_params", "n_batch", "n_ubatch", "n_threads", "type_k", "type_v", "n_gpu_layers",
                  "n_cpu_moe", "flash_attn", "load_mode", "tensor_buft_overrides", "no_kv_offload")


def parse_llama_bench_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not (line.startswith("{") and line.endswith("}")):
        return None
    try:
        d = json.loads(line)
    except ValueError:
        return None
    return d if isinstance(d, dict) and "avg_ts" in d else None


def to_row(d: dict[str, Any]) -> TestRow:
    n_p, n_g = int(d.get("n_prompt") or 0), int(d.get("n_gen") or 0)
    kind = "pg" if n_p and n_g else ("pp" if n_p else "tg")
    ts, sd = float(d["avg_ts"]), float(d.get("stddev_ts") or 0.0)
    row = TestRow(kind=kind, n_prompt=n_p, n_gen=n_g, depth=int(d.get("n_depth") or 0),
                  samples=[round(float(x), 3) for x in d.get("samples_ts") or []])
    if kind == "tg":
        row.tg_tps, row.tg_std = ts, sd
    else:
        row.pp_tps, row.pp_std = ts, sd       # pg: llama-bench reports one combined rate
    return row


def parse_llama_bench(lines: Iterable[str]) -> tuple[list[TestRow], dict[str, Any]]:
    """Rows + the effective settings of the first line."""
    rows: list[TestRow] = []
    eff: dict[str, Any] = {}
    for ln in lines:
        d = parse_llama_bench_line(ln)
        if d is None:
            continue
        if not eff:
            eff = {k: d[k] for k in EFFECTIVE_KEYS if k in d}
        rows.append(to_row(d))
    return rows, eff
