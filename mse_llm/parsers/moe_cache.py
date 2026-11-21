"""MoE expert cache lines of a llama-server log (§4.4, M11) -> RunResult.moe_cache.

Printed by src/llama-moe-cache.cpp, at load (needs `-lv 4`: llama's own info lines are hidden at the default 3;
the load lines carry the function name `impl:`):

    impl:      CUDA0 MoE cache size =  4093.19 MiB for 18662.00 MiB of host experts, 95% of the misses on the CPU
    impl: 37 layers, q5_K:  2069 slots (21.8%, 7.0 tokens worth)
    llama_moe_cache: no layer has all of its experts in host memory, MoE cache is disabled      (warning)
    llama_moe_cache: MoE cache budget is too small for 40 layers, they are not cached            (warning)

and when the cache is freed, i.e. only at a clean exit (Ctrl+C, see proc.send_ctrl_c):

    llama_moe_cache: ubatch <= 8: hits = H, misses = M (C on the CPU), hit rate = R%, uploaded = U MiB
    llama_moe_cache: ubatch  > 8: hits = ...
    llama_moe_cache: large batches: N experts copied from the cache, X MiB

The small-batch counters (decode, speculative drafting) are the ones the cache is for; they give `hit_rate`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_SIZE = re.compile(r"(\S+) MoE cache size =\s*([0-9.]+) MiB for ([0-9.]+) MiB of host experts, ([0-9.]+)% of the "
                   r"misses on the CPU")
_GROUP = re.compile(r"(?:impl|llama_moe_cache):\s*(\d+) layers, (\S+):\s*(\d+) slots "
                    r"\(([0-9.]+)%, ([0-9.]+) tokens worth\)")
_STATS = re.compile(r"llama_moe_cache: ubatch\s*(<=|>)\s*8: hits = (\d+), misses = (\d+)(?: \((\d+) on the CPU\))?, "
                    r"hit rate = ([0-9.]+)%, uploaded = ([0-9.]+) MiB")
_COPY = re.compile(r"llama_moe_cache: large batches: (\d+) experts copied from the cache, ([0-9.]+) MiB")
# a config the cache can't run: the server starts without a cache, or doesn't start
INVALID = [
    (re.compile(r"MoE cache is disabled"), "MoE cache is disabled (no layer has all of its experts in host memory: "
                                           "use -cmoe or -ncmoe)"),
    (re.compile(r"MoE cache budget is too small for (\d+) layers"), "MoE cache budget is too small for {0} layers "
                                                                    "(they are not cached)"),
    (re.compile(r"MoE cache is too small to hold the experts of one token"), "MoE cache is too small to hold the "
                                                                             "experts of one token"),
    (re.compile(r"MoE cache requires ([^\"\n]+)"), "MoE cache requires {0}"),
    (re.compile(r"MoE cache does not support tensor parallelism"), "MoE cache does not support tensor parallelism"),
]


def is_cache_line(line: str) -> bool:
    return "llama_moe_cache" in line or "MoE cache" in line or bool(_GROUP.search(line))


def parse_log(lines: Iterable[str]) -> dict[str, Any] | None:
    """The cache's size, slots, counters and problems; None when the log has no MoE cache line at all."""
    size_mib = host_mib = 0.0
    cpu_share = None
    devices: list[str] = []
    groups: list[dict[str, Any]] = []
    small: dict[str, float] | None = None
    large: dict[str, float] | None = None
    copied = None
    invalid: list[str] = []
    seen = False
    for ln in lines:
        if not is_cache_line(ln):
            continue
        seen = True
        if m := _SIZE.search(ln):
            devices.append(m.group(1))
            size_mib += float(m.group(2))
            host_mib += float(m.group(3))
            cpu_share = float(m.group(4)) / 100
        elif m := _GROUP.search(ln):
            groups.append({"layers": int(m.group(1)), "type": m.group(2), "slots": int(m.group(3)),
                           "slots_pct": float(m.group(4)), "tokens_worth": float(m.group(5))})
        elif m := _STATS.search(ln):
            d = _acc(small if m.group(1) == "<=" else large, m)
            if m.group(1) == "<=":
                small = d
            else:
                large = d
        elif m := _COPY.search(ln):
            copied = {"experts": int(m.group(1)), "mib": float(m.group(2))}
        else:
            for rx, text in INVALID:
                if mi := rx.search(ln):
                    msg = text.format(*mi.groups())
                    if msg not in invalid:
                        invalid.append(msg)
                    break
    if not seen:
        return None
    out: dict[str, Any] = {"size_mib": round(size_mib, 2) if devices else 0.0, "host_experts_mib": round(host_mib, 2),
                           "cpu_share": cpu_share, "devices": devices, "groups": groups}
    if small:
        out["small"] = _finish(small)
        out.update(hit_rate=out["small"]["hit_rate"], uploaded_mib=out["small"]["uploaded_mib"],
                   cpu_misses=out["small"]["cpu"])
    if large:
        out["large"] = _finish(large)
    if copied:
        out["large_copied"] = copied
    if not small and not large and devices:
        out["note"] = ("no hit / miss counters: the cache prints them only at a clean exit (the server was killed, "
                       "or nothing was generated)")
    if invalid:
        out["invalid_config"] = "; ".join(invalid)
    elif not devices and (small or large) is None:
        out["invalid_config"] = "no MoE cache size line (server without -lv 4, or the cache was not created)"
    return out


def _acc(prev: dict[str, float] | None, m: re.Match[str]) -> dict[str, float]:
    d = dict(prev or {"hits": 0, "misses": 0, "cpu": 0, "uploaded_mib": 0.0})
    d["hits"] += int(m.group(2))
    d["misses"] += int(m.group(3))
    d["cpu"] += int(m.group(4) or 0)
    d["uploaded_mib"] += float(m.group(6))
    return d


def _finish(d: dict[str, float]) -> dict[str, Any]:
    n = d["hits"] + d["misses"]
    return {"hits": int(d["hits"]), "misses": int(d["misses"]), "cpu": int(d["cpu"]),
            "hit_rate": round(d["hits"] / n, 4) if n else None, "uploaded_mib": round(d["uploaded_mib"], 2)}
