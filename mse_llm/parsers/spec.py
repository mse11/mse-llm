"""Speculative-decoding acceptance (§4.5): llama-server's Prometheus `/metrics` and its slot log line.

Checked against build ffc87c3a7 (tools/server/server-task.cpp, server-context.cpp):

    llamacpp:spec_decode_num_draft_tokens_total            drafted tokens
    llamacpp:spec_decode_num_accepted_tokens_total         accepted drafted tokens
    llamacpp:spec_decode_num_drafts_total                  verification steps
    llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="N"}   accepted at draft position N
    slot log:  draft acceptance = 0.71234 (  412 accepted /   578 generated), mean len =  2.85

`/metrics` needs `--metrics`; mse-llm adds it to every server it starts for a benchmark. The counters are
totals since the server started, so a run's numbers are the delta between a scrape before and after it.
mean len = 1 + accepted / verification steps (tokens gained per target-model forward pass).
"""

from __future__ import annotations

import re
from typing import Any

_LINE = re.compile(r'^llamacpp:(\w+)(?:\{position="(\d+)"\})?\s+(\S+)\s*$')
_LOG = re.compile(r"draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\),\s*"
                  r"mean len\s*=\s*([\d.]+)")

SPEC_COUNTERS = {"spec_decode_num_draft_tokens_total": "drafted",
                 "spec_decode_num_accepted_tokens_total": "accepted",
                 "spec_decode_num_drafts_total": "verif_steps"}
PER_POS = "spec_decode_num_accepted_tokens_per_pos_total"


def parse_metrics(text: str) -> dict[str, Any]:
    """Prometheus text -> {name: value, ..., "per_pos": [..]} (llamacpp: prefix dropped)."""
    out: dict[str, Any] = {}
    pos: dict[int, float] = {}
    for ln in text.splitlines():
        m = _LINE.match(ln.strip())
        if not m:
            continue
        name, p, val = m.group(1), m.group(2), m.group(3)
        try:
            x = float(val)
        except ValueError:
            continue
        if p is not None and name == PER_POS:
            pos[int(p)] = x
        else:
            out[name] = x
    if pos:
        out["per_pos"] = [pos.get(i, 0.0) for i in range(max(pos) + 1)]
    return out


def delta(before: dict[str, Any] | None, after: dict[str, Any]) -> dict[str, Any]:
    """after - before for every counter (a missing `before` = a fresh server)."""
    b = before or {}
    out: dict[str, Any] = {}
    for k, v in after.items():
        if k == "per_pos":
            bp = b.get("per_pos") or []
            out[k] = [v[i] - (bp[i] if i < len(bp) else 0.0) for i in range(len(v))]
        elif isinstance(v, (int, float)) and k.endswith("_total"):
            out[k] = v - float(b.get(k) or 0.0)
        else:
            out[k] = v
    return out


def spec_from_counters(d: dict[str, Any]) -> dict[str, Any] | None:
    """The stored `spec` numbers from a /metrics delta; None when nothing was drafted."""
    drafted = int(round(d.get("spec_decode_num_draft_tokens_total") or 0))
    if drafted <= 0:
        return None
    accepted = int(round(d.get("spec_decode_num_accepted_tokens_total") or 0))
    steps = int(round(d.get("spec_decode_num_drafts_total") or 0))
    per_pos = [int(round(x)) for x in d.get("per_pos") or []]
    while per_pos and per_pos[-1] == 0:
        per_pos.pop()
    return _finish({"source": "metrics", "drafted": drafted, "accepted": accepted, "verif_steps": steps,
                    "accepted_per_pos": per_pos})


def parse_server_log(lines: list[str]) -> dict[str, Any] | None:
    """Sum of every slot's `draft acceptance` line (one per finished request): the fallback when /metrics is off.
    No per-position numbers (those are only in the trace-level log)."""
    acc = gen = 0
    steps = 0.0
    n = 0
    for ln in lines:
        m = _LOG.search(ln)
        if not m:
            continue
        a, g, mean_len = int(m.group(2)), int(m.group(3)), float(m.group(4))
        acc, gen, n = acc + a, gen + g, n + 1
        if mean_len > 1:
            steps += a / (mean_len - 1)
    if gen <= 0:
        return None
    return _finish({"source": "server log", "drafted": gen, "accepted": acc,
                    "verif_steps": int(round(steps)) or None, "requests": n, "accepted_per_pos": []})


def _finish(s: dict[str, Any]) -> dict[str, Any]:
    d, a, steps = s["drafted"], s["accepted"], s.get("verif_steps")
    s["acceptance_rate"] = round(a / d, 4) if d else None
    s["mean_accept_len"] = round(1 + a / steps, 3) if steps else None
    pp = s.get("accepted_per_pos") or []
    s["accept_rate_per_pos"] = [round(x / steps, 4) for x in pp] if steps and pp else []
    return s


def useful_draft_len(rates: list[float], threshold: float = 0.3) -> int | None:
    """First draft position whose acceptance falls under `threshold` (1-based count of useful positions):
    a longer draft than this usually stops paying off."""
    for i, r in enumerate(rates):
        if r < threshold:
            return i
    return len(rates) if rates else None
