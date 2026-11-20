"""Speculative-decoding report (M10, §4.5): acceptance next to TG, per draft position, and the no-spec baseline.

A spec run whose TG is lower than the same variant without speculative decoding (same model, ctx, tool, build
and prompt set) is flagged: the draft costs more than it saves.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from mse_llm import results as rs


def baseline_name(variant: str) -> str | None:
    """The variant without its speculative-decoding tokens (None when it has none)."""
    from mse_llm.variants import VariantError, decode

    try:
        v = decode(variant)
    except VariantError:
        return None
    if not v.spec_type:
        return None
    return replace(v.without_spec()).name


def _bkey(e: dict[str, Any] | rs.RunResult, variant: str) -> tuple:
    if isinstance(e, dict):
        return (e.get("model"), e.get("tool"), e.get("ctx"), e.get("build"), e.get("prompts") or "prose", variant)
    return (e.model, e.tool, e.ctx, e.build.get("id"), e.params.get("prompts") or "prose", variant)


def spec_rows(runs: list[rs.RunResult], index: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One row per spec run: TG, PP, acceptance, mean len, per-position rates, the no-spec TG and a flag."""
    index = rs.read_index() if index is None else index
    base: dict[tuple, dict[str, Any]] = {}
    for e in index:
        if e.get("status") == "ok" and e.get("tg_tps") is not None:
            base[_bkey(e, e.get("variant"))] = e          # newest wins
    out = []
    for r in runs:
        s = r.spec
        if not s:
            continue
        bname = baseline_name(r.variant)
        prompts = r.params.get("prompts") or "prose"
        b = base.get(_bkey(r, bname)) if bname else None
        tg = r.metrics.get("tg_tps")
        btg = b.get("tg_tps") if b else None
        out.append({"run_id": r.run_id, "model": r.model, "variant": r.variant, "ctx": r.ctx,
                    "prompts": prompts, "type": s.get("type"), "n_max": s.get("n_max"),
                    "tg_tps": tg, "pp_tps": r.metrics.get("pp_tps"),
                    "acceptance_rate": s.get("acceptance_rate"), "mean_accept_len": s.get("mean_accept_len"),
                    "drafted": s.get("drafted"), "accepted": s.get("accepted"),
                    "accept_rate_per_pos": s.get("accept_rate_per_pos") or [], "useful_len": s.get("useful_len"),
                    "source": s.get("source"), "temp": (s.get("sampling") or {}).get("temp"),
                    "baseline_variant": bname, "baseline_tg": btg, "baseline_run": b.get("run_id") if b else None,
                    "speedup": round(tg / btg, 3) if tg and btg else None,
                    "slower_than_baseline": bool(tg and btg and tg < btg), "status": r.status})
    return out


def chart(rows: list[dict[str, Any]], out: str | Path, title: str = "Speculative decoding") -> Path:
    """Two panels: TG and acceptance vs draft n_max (one line per model / prompt set), and the acceptance per
    draft position (bars per n_max). The 0.3 line marks where a longer draft usually stops paying off."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in rows if r.get("n_max") is not None and r.get("tg_tps") is not None]
    if not rows:
        raise ValueError("no spec runs with TG to chart")
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 6))
    ax2 = ax1.twinx()
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(f"{r['model']} ({r['prompts']})", []).append(r)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (g, rs_) in enumerate(groups.items()):
        rs_ = sorted(rs_, key=lambda r: r["n_max"])
        x = [r["n_max"] for r in rs_]
        c = colors[i % len(colors)]
        ax1.plot(x, [r["tg_tps"] for r in rs_], "o-", color=c, label=f"TG {g}")
        ax2.plot(x, [(r["acceptance_rate"] or 0) * 100 for r in rs_], "s--", color=c, alpha=0.6,
                 label=f"acceptance {g}")
        btg = next((r["baseline_tg"] for r in rs_ if r.get("baseline_tg")), None)
        if btg:
            ax1.axhline(btg, color=c, linestyle=":", linewidth=1)
            ax1.annotate("no spec", (x[0], btg), color=c, fontsize=8, va="bottom")
    ax1.set_xticks(sorted({r["n_max"] for r in rows}))
    ax1.set_xlabel("spec-draft-n-max")
    ax1.set_ylabel("TG t/s")
    ax2.set_ylabel("acceptance %")
    ax2.set_ylim(0, 100)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax1.set_title("TG and acceptance vs draft length")
    per = [r for r in rows if r["accept_rate_per_pos"]]
    if per:
        width = 0.8 / len(per)
        npos = max(len(r["accept_rate_per_pos"]) for r in per)
        for j, r in enumerate(per):
            rates = r["accept_rate_per_pos"]
            ax3.bar(np.arange(len(rates)) + j * width, rates, width, label=f"{r['variant'].rsplit('_', 1)[-1]} "
                                                                           f"({r['prompts']})")
        ax3.set_xticks(np.arange(npos) + 0.4 - width / 2, [str(i + 1) for i in range(npos)])
        ax3.axhline(0.3, color="red", linestyle="--", linewidth=1)
        ax3.set_ylim(0, 1)
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "no per-position counts (server without --metrics)", ha="center", transform=ax3.transAxes)
    ax3.set_xlabel("draft position")
    ax3.set_ylabel("accepted / verification steps")
    ax3.set_title("Acceptance per draft position")
    fig.suptitle(title)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
