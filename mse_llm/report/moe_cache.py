"""MoE expert cache report (M11, §4.4): per variant TG, PP at a short and a long prompt, duration, peak VRAM, cache hit
rate, uploaded MiB and PCIe traffic. Runs without a cache (the baselines) are shown with an empty hit rate, so a cache
that is slower than no cache is visible next to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mse_llm import results as rs


def _stat(r: rs.RunResult, key: str, what: str = "peak") -> float | None:
    st = ((r.monitor or {}).get("stats") or {}).get(key) or {}
    x = st.get(what)
    return float(x) if isinstance(x, (int, float)) else None


def _pp_at(r: rs.RunResult, longest: bool) -> tuple[int | None, float | None]:
    pp = [t for t in r.tests if t.pp_tps is not None and t.depth == 0]
    if not pp:
        return None, None
    t = max(pp, key=lambda t: t.n_prompt) if longest else min(pp, key=lambda t: t.n_prompt)
    return t.n_prompt, t.pp_tps


def cache_rows(runs: list[rs.RunResult]) -> list[dict[str, Any]]:
    out = []
    for r in runs:
        c = r.moe_cache or {}
        v = r.variant_fields or {}
        ps, pps = _pp_at(r, False)
        pl, ppl = _pp_at(r, True)
        ready = float((r.startup or {}).get("ready_s") or 0.0)
        out.append({"run_id": r.run_id, "variant": r.variant, "stage": (r.stage or {}).get("index"),
                    "status": r.status, "reason": r.reason, "tg_tps": r.metrics.get("tg_tps"),
                    "pp_short": ps, "pp_short_tps": pps, "pp_long": pl if pl != ps else None,
                    "pp_long_tps": ppl if pl != ps else None,
                    "duration_s": r.duration_s, "ready_s": ready or None,
                    "vram_peak_mib": _stat(r, "vram_used_mib"), "ram_peak_mib": _stat(r, "ram_used_mib"),
                    "pcie_rx_mean_mbs": _stat(r, "pcie_rx_mbs", "mean"), "pcie_rx_peak_mbs": _stat(r, "pcie_rx_mbs"),
                    "mc": v.get("moe_cache_mib"), "cs": v.get("moe_cache_cpu_share"),
                    "lm": v.get("load_mode"), "placement": "cmoe" if v.get("cpu_moe") else
                    (f"moe{v['n_cpu_moe']}" if v.get("n_cpu_moe") is not None else "gpu"),
                    "cache_mib": c.get("size_mib"), "hit_rate": c.get("hit_rate"),
                    "uploaded_mib": c.get("uploaded_mib"), "cpu_misses": c.get("cpu_misses"),
                    "large_hit_rate": (c.get("large") or {}).get("hit_rate"),
                    "invalid_config": c.get("invalid_config"), "clean_exit": c.get("clean_exit"),
                    "warmup_tokens": (c.get("warmup") or {}).get("tokens")})
    return out


def chart(rows: list[dict[str, Any]], out: str | Path, title: str = "MoE expert cache") -> Path:
    """Six panels, one bar per variant: TG, PP (short / long prompt), duration (server start stacked), peak VRAM,
    small-batch hit rate, uploaded MiB with the mean PCIe RX rate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in rows if r.get("tg_tps") is not None]
    if not rows:
        raise ValueError("no runs with TG to chart")
    labels = [r["variant"].replace("fa_", "").replace("_", " ") for r in rows]
    x = np.arange(len(rows))
    n = len(rows)
    fig, axes = plt.subplots(3, 2, figsize=(max(14, 0.9 * n + 6), 13))
    (a_tg, a_pp), (a_dur, a_vram), (a_hit, a_up) = axes
    base = [r for r in rows if r.get("hit_rate") is None and r.get("mc") is None]
    colors = ["#888888" if r.get("mc") is None else "#1f77b4" for r in rows]

    def bars(ax, vals, ylabel, fmt="{:.1f}", color=None):
        vals = [0.0 if v is None else float(v) for v in vals]
        b = ax.bar(x, vals, color=color or colors)
        for rect, v in zip(b, vals, strict=True):
            if v:
                ax.annotate(fmt.format(v), (rect.get_x() + rect.get_width() / 2, v), ha="center", va="bottom",
                            fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=7)

    bars(a_tg, [r["tg_tps"] for r in rows], "TG t/s", "{:.2f}")
    if base:
        best_base = max(r["tg_tps"] for r in base)
        a_tg.axhline(best_base, color="red", linestyle="--", linewidth=1)
        a_tg.annotate(f"best without cache {best_base:.1f}", (n - 0.5, best_base), color="red", fontsize=8,
                      va="bottom", ha="right")
    a_tg.set_title("TG (grey = no cache)")
    w = 0.4
    ps = [r["pp_short_tps"] or 0 for r in rows]
    pl = [r["pp_long_tps"] or 0 for r in rows]
    short_n = next((r["pp_short"] for r in rows if r["pp_short"]), None)
    long_n = next((r["pp_long"] for r in rows if r["pp_long"]), None)
    a_pp.bar(x - w / 2, ps, w, label=f"pp {short_n}")
    if any(pl):
        a_pp.bar(x + w / 2, pl, w, label=f"pp {long_n}")
    a_pp.set_ylabel("PP t/s")
    a_pp.set_xticks(x, labels, rotation=60, ha="right", fontsize=7)
    a_pp.legend(fontsize=8)
    a_pp.set_title("PP (the cache is for small batches: should stay the same)")
    ready = [r["ready_s"] or 0 for r in rows]
    rest = [max(0.0, (r["duration_s"] or 0) - (r["ready_s"] or 0)) for r in rows]
    a_dur.bar(x, ready, color="#bbbbbb", label="server start")
    a_dur.bar(x, rest, bottom=ready, color="#2ca02c", label="warm-up + benchy")
    a_dur.set_ylabel("seconds")
    a_dur.set_xticks(x, labels, rotation=60, ha="right", fontsize=7)
    a_dur.legend(fontsize=8)
    a_dur.set_title("Duration")
    bars(a_vram, [r["vram_peak_mib"] for r in rows], "MiB", "{:.0f}")
    a_vram.set_title("Peak VRAM (dedicated)")
    bars(a_hit, [None if r["hit_rate"] is None else r["hit_rate"] * 100 for r in rows], "%", "{:.0f}")
    a_hit.set_ylim(0, 100)
    a_hit.set_title("Cache hit rate (ubatch <= 8)")
    bars(a_up, [r["uploaded_mib"] for r in rows], "MiB", "{:.0f}", color="#9467bd")
    a_up.set_title("Uploaded over PCIe (cache misses) · line: mean PCIe RX MB/s")
    ax2 = a_up.twinx()
    ax2.plot(x, [r["pcie_rx_mean_mbs"] or 0 for r in rows], "o-", color="#ff7f0e")
    ax2.set_ylabel("PCIe RX MB/s (mean)")
    fig.suptitle(title)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out
