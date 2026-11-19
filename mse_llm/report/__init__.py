"""Reports from the results store (M8, §8, parity F10/F11): the bash 5-axis chart, tables, analyst prompts."""

from __future__ import annotations

from pathlib import Path

from mse_llm import results as rs
from mse_llm.report.data import Selection, table_row, test_rows, to_csv

PROMPTS = Path(__file__).parent / "prompts"


def reports_dir(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:80] or "report"
    return rs.results_root() / "reports" / safe


def render_prompt(sel: Selection, *, title: str | None = None, chart: str | None = None,
                  with_tests: bool = True) -> str:
    """The bash AI analyst prompt (run_dump_params_wrapper_AI_PROMPT + _COMMON) with the data embedded as CSV
    instead of "[PASTE RAW RESULTS BELOW THIS LINE]"."""
    import jinja2

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS)), keep_trailing_newline=True,
                             autoescape=False, undefined=jinja2.StrictUndefined)
    runs = sel.runs
    hw = next((r.hw for r in runs if r.hw.get("gpu")), None)
    hw_text = None
    if hw:
        power = hw.get("power") if hw.get("power") in ("AC power", "on battery") else None
        hw_text = f"{hw.get('gpu')}, driver {hw.get('driver')}" + (f", {power}" if power else "")
    rows = [table_row(r) for r in runs]
    tests = [t for r in runs for t in test_rows(r)] if with_tests else []
    return env.get_template("analyst.md.j2").render(
        title=title or sel.title, runs=[{"variant": r.variant, "ctx": r.ctx, "note": r.note or r.reason}
                                        for r in runs],
        models=sorted({r.model for r in runs}), tools=sorted({r.tool for r in runs}),
        engines=sorted({r.engine for r in runs}),
        builds=sorted({r.build.get("describe") or (r.build.get("commit") or "")[:9] for r in runs} - {""}),
        ctxs=sorted({r.ctx for r in runs if r.ctx}), hw=hw_text, chart=chart, csv=to_csv(rows),
        tests_csv=to_csv(tests) if tests else "")
