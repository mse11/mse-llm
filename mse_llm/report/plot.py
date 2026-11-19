"""The bash 5-axis grouped bar chart (`run_dump_params_wrapper_PLOTER`), ported as is (parity F10).

Bars per configuration: duration (server startup stacked below), PP, TG, peak VRAM, shared memory, each on
its own y axis; background bands per "model [engine]"; 3-line x labels (K/V, MoE + load mode + spec, ctx +
batch). Same sizes, colours, offsets and label formats as the bash script, so an old log gives the same chart.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mse_llm.parsers.legacy_log import ChartRow

# bash: a missing value was drawn as a placeholder bar (and labelled with it)
BASH_PLACEHOLDERS = {"duration": 100.0, "vram": 4000.0, "svram": 12000.0}


def chart(rows: Sequence[ChartRow], out: str | Path, *, title: str = "Benchmarks Performance Analysis",
          bench_type: str = "llama-bench", bash_placeholders: bool = False) -> Path:
    """Write the PNG. bash_placeholders: draw missing duration / VRAM / shared as 100 s / 4000 / 12000 like
    the bash plotter did (legacy logs); new results draw 0."""
    import logging

    import matplotlib

    matplotlib.use("Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # "semibold" -> 700, as in bash
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import matplotlib.transforms as transforms
    import numpy as np

    if not rows:
        raise ValueError("nothing to chart")
    ph = BASH_PLACEHOLDERS if bash_placeholders else {"duration": 0.0, "vram": 0.0, "svram": 0.0}
    n = len(rows)
    configs = [r.config_id for r in rows]
    durations = [r.duration if r.duration > 0 else ph["duration"] for r in rows]
    startups = [r.server_startup if r.server_startup > 0 else 0 for r in rows]
    pps = [r.pp for r in rows]
    tgs = [r.tg for r in rows]
    vrams = [r.vram if r.vram > 0 else ph["vram"] for r in rows]
    svrams = [r.svram if r.svram > 0 else ph["svram"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(30, 11), facecolor="white")
    ax2, ax3, ax4, ax5 = ax1.twinx(), ax1.twinx(), ax1.twinx(), ax1.twinx()
    for ax, side, pos in ((ax2, "left", -0.06), (ax3, "left", -0.13), (ax4, "right", 1.06), (ax5, "right", 1.13)):
        ax.spines[side].set_position(("axes", pos))
        ax.yaxis.set_label_position(side)
        ax.yaxis.set_ticks_position(side)
    for ax, label, color in ((ax1, "Token Generation (t/s)", "#1565c0"), (ax2, "Prompt Processing (t/s)", "#5dade2"),
                             (ax3, "Total Duration (s)", "#e74c3c"), (ax4, "Peak VRAM (MiB)", "#81c784"),
                             (ax5, "Shared VRAM (MiB)", "#2e7d32")):
        ax.set_ylabel(label, color=color, fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", colors=color)

    ind = np.arange(n)
    width, alpha = 0.16, 0.85
    ax1.grid(True, axis="y", color="#e0e0e0", linewidth=0.5, alpha=0.7, zorder=1)
    r3s = ax3.bar(ind - 2 * width, startups, width, color="#c0392b", alpha=alpha, zorder=3, label="Server Startup")
    r3 = ax3.bar(ind - 2 * width, durations, width, bottom=startups, color="#e74c3c", alpha=alpha, zorder=3,
                 label="Benchmark Duration")
    r2 = ax2.bar(ind - width, pps, width, color="#5dade2", alpha=alpha, zorder=3)
    r1 = ax1.bar(ind, tgs, width, color="#1565c0", alpha=alpha, zorder=3)
    r4 = ax4.bar(ind + width, vrams, width, color="#81c784", alpha=alpha, zorder=3)
    r5 = ax5.bar(ind + 2 * width, svrams, width, color="#2e7d32", alpha=alpha, zorder=3)
    for ax in (ax1, ax2, ax3, ax4, ax5):
        ax.margins(y=0.25)

    unique: list[str] = []
    for r in rows:
        if r.config_a not in unique:
            unique.append(r.config_a)
    region_colors = ["#fff3e0", "#ffe0b2", "#ffcc80"]
    patch_colors = ["#ff9800", "#fb8c00", "#f57c00"]
    region_labels = ["Config Set A", "Config Set B", "Config Set C"]
    trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    for idx, name in enumerate(unique):
        indices = [i for i, r in enumerate(rows) if r.config_a == name]
        start, end = indices[0] - 0.5, indices[-1] + 0.5
        lbl = region_labels[idx] if idx < len(region_labels) else f"Config Set {idx + 1}"
        ax1.axvspan(start, end, facecolor=region_colors[idx % len(region_colors)], alpha=0.6, zorder=0)
        if start > -0.5:
            ax1.axvline(start, color="#e0e0e0", linewidth=1.2, linestyle="--", zorder=0.5)
        ax1.text((start + end) / 2, 1.02, f"{lbl}: {name}", transform=trans, ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold", color="#555555", zorder=5)

    def add_labels(rects, ax, fmt: str, color: str) -> None:
        lo, hi = ax.get_ylim()
        off = (hi - lo) * 0.008
        for rect in rects:
            val = rect.get_height()
            if fmt == "time":
                text = f"{int(val // 60):02d}:{int(val % 60):02d}"
            elif fmt == "dec":
                text = f"{val:.1f}"
            else:
                text = f"{int(val)}M"
            ax.text(rect.get_x() + rect.get_width() / 2.0, rect.get_y() + val + off, text, ha="center",
                    va="bottom", rotation=90, fontsize=7.0, color=color, fontweight="semibold", zorder=4)

    add_labels(r3s, ax3, "time", "#c0392b")
    add_labels(r3, ax3, "time", "#e74c3c")
    add_labels(r2, ax2, "dec", "#5dade2")
    add_labels(r1, ax1, "dec", "#1565c0")
    add_labels(r4, ax4, "intM", "#81c784")
    add_labels(r5, ax5, "intM", "#2e7d32")

    ax1.set_xticks(ind)
    ax1.set_xticklabels(configs, rotation=35, ha="right", fontsize=8.0)
    labels = ["Duration (Bench)", "Duration (Startup)", "PP", "TG", "VRAM", "SharedRAM"]
    colors = ["#e74c3c", "#c0392b", "#5dade2", "#1565c0", "#81c784", "#2e7d32"]
    handles = [mpatches.Patch(color=c, label=lb) for c, lb in zip(colors, labels, strict=True)]
    for idx in range(min(len(unique), 3)):
        handles.append(mpatches.Patch(color=patch_colors[idx], label=region_labels[idx]))
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=8, frameon=True,
               facecolor="white", edgecolor="#cccccc")
    plt.title(f"{title} [{bench_type.upper()}]", fontsize=14, fontweight="bold", pad=88, color="#222222")
    plt.tight_layout(rect=[0.06, 0, 0.95, 0.95])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
