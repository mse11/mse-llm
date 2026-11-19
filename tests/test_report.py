"""report chart / table / prompt, results compare / export, legacy log import (M8, §8, parity F10/F11)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from mse_llm import results as rs
from mse_llm.cli import main
from mse_llm.parsers.legacy_log import ChartRow, config_label, html_to_text, parse_legacy, read_log
from mse_llm.report import render_prompt
from mse_llm.report.data import chart_row, compare, select, table_row
from mse_llm.report.plot import chart
from mse_llm.util import project_root

LEGACY = Path(__file__).parent / "fixtures" / "legacy"
UD_TXT = LEGACY / "run_BENCH_gemma_4_26B_A4b_UD_Q4_K_M.txt"
QAT_HTML = LEGACY / "RESULTS_run_BENCH_gemma_4_26B_A4b_QAT_UD_Q4_K_XL.html"


# ---------------------------------------------------------------- legacy logs

def test_legacy_text_log_as_bash_parses_it():
    bt, rows = parse_legacy(read_log(UD_TXT))
    assert bt == "llama-bench" and len(rows) == 17           # bash plotter: "Processed 17 benchmark block(s)"
    r = rows[0]
    assert r.config_a == "gemma_4_26B_A4b_UD_Q4_K_M [cuda_flash]"
    assert r.config_id == "K:q8_0 / V:q8_0\nMoE:31 / mmap:0\nCtx:default / Batch:default / uBatch:default"
    assert (r.pp, r.tg, r.vram, r.svram, r.duration) == (873.96, 24.35, 4858.0, 14732.0, 497.0)
    assert rows[-1].config_id.endswith("Batch:4096 / uBatch:1024")


def test_legacy_html_page_with_benchy_blocks():
    bt, rows = parse_legacy(read_log(QAT_HTML))
    assert bt == "llama-benchy" and len(rows) == 23
    spec = next(r for r in rows if "Spec:" in r.config_id)
    assert "Spec:draft / DraftN:10 / DraftModel:mtp-gemma-4-26B-A4B-it" in spec.config_id
    assert spec.server_startup == 1.0 and spec.tg == 20.28


def test_html_to_text():
    t = html_to_text("<p>TESTING:\n&gt;&gt;&gt; Started<br>|&nbsp; pp512 |&nbsp;\n 923.19 ± 6 |<br></p>")
    assert t.splitlines() == ["TESTING: >>> Started", "| pp512 | 923.19 ± 6 |"]


def test_config_label_quirks_kept():
    argv = "llama-bench -p 65536 -fa on -ctk q8_0 -ctv q4_0 -b 2048 -ub 512 --n-cpu-moe 21 -lm none"
    assert config_label(argv) == "K:q8_0 / V:q4_0\nMoE:21 / lm:none\nCtx:default / Batch:2048 / uBatch:512"
    assert config_label(argv, 65536).endswith("Ctx:65536 / Batch:2048 / uBatch:512")


def test_chart_writes_png(tmp_path):
    _, rows = parse_legacy(read_log(UD_TXT))
    p = chart(rows, tmp_path / "c.png", title="t", bash_placeholders=True)
    import matplotlib.image as im

    h, w, _ = im.imread(p).shape
    assert w > 4000 and h > 1500                               # figsize 30x11 at dpi 180, tight bbox
    with pytest.raises(ValueError):
        chart([], tmp_path / "x.png")


@pytest.mark.skipif(os.environ.get("MSE_LLM_BASH_PARITY") != "1" or not shutil.which("bash")
                    or not shutil.which("uvx"), reason="set MSE_LLM_BASH_PARITY=1 (needs bash + uvx + matplotlib)")
def test_chart_is_pixel_identical_to_the_bash_plotter(tmp_path):
    """Runs the real bash run_dump_params_wrapper_PLOTER on the same log and compares every pixel."""
    import matplotlib.image as im
    import numpy as np

    log = tmp_path / "log.txt"
    log.write_text(read_log(UD_TXT), encoding="utf-8")
    script = (project_root() / "llama_cpp_benchmark.sh").as_posix()
    cmd = (f'source "{script}" >/dev/null 2>&1; run_dump_params_wrapper_PLOTER auto "T" '
           f'"{log.as_posix()}" "{(tmp_path / "bash.png").as_posix()}"')
    subprocess.run(["bash", "-c", cmd], check=True, capture_output=True, timeout=600)
    bt, rows = parse_legacy(read_log(log))
    chart(rows, tmp_path / "ours.png", title="T", bench_type=bt, bash_placeholders=True)
    a, b = im.imread(tmp_path / "bash.png"), im.imread(tmp_path / "ours.png")
    assert a.shape == b.shape and int((np.abs(a - b).max(axis=2) > 0.02).sum()) == 0


# ---------------------------------------------------------------- results -> rows

def _run(variant="fa_Kq8_Vq8_default", tool="bench", key=None, pp=1000.0, tg=50.0, status="ok", ctx=4096,
         sweep="s1", model="omnicoder_9b", ts="2026-11-19T10:00:00+01:00", rid=None, **kw) -> rs.RunResult:
    rid = rid or f"20261119-10{abs(hash((variant, ts))) % 10000:04d}-{tool}-{abs(hash(key or variant)) % 9999:04x}"
    argv = ["llama-bench", "-p", str(ctx), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none"]
    return rs.RunResult(run_id=rid, key=key or variant, timestamp=ts, sweep=sweep, tool=tool, engine="cuda_flash",
                        model=model, variant=variant, ctx=ctx, argv=argv, status=status,
                        metrics={"pp_tps": pp, "tg_tps": tg} if status == "ok" else {},
                        monitor={"duration_s": 30.0, "stats": {"vram_used_mib": {"peak": 6000},
                                                               "nonlocal_mib": {"peak": 900},
                                                               "proc_shared_mib": {"peak": 650}}},
                        build={"id": "b1", "commit": "ffc87c3a7aaa"}, duration_s=33.0, **kw)


def test_chart_row_from_a_result():
    r = _run(startup={"ready_s": 4.0}, server_argv=["llama-server", "-c", "4096", "--spec-type", "draft-mtp",
                                                     "--spec-draft-n-max", "4"], tool="benchy")
    row = chart_row(r)
    assert row.config_a == "omnicoder_9b [cuda_flash]"
    assert row.config_id.splitlines() == ["K:q8_0 / V:q8_0", "MoE:default / lm:none / Spec:draft / DraftN:4",
                                          "Ctx:4096 / Batch:default / uBatch:default"]
    assert (row.server_startup, row.duration, row.vram, row.svram) == (4.0, 26.0, 6000.0, 900.0)


def _store(*runs):
    for r in runs:
        rs.write_run(r)


def test_select_latest_per_key_and_sweep_order():
    old = _run("fa_Kq8_Vq4_default", pp=1.0, ts="2026-11-19T09:00:00+01:00", rid="20261119-090000-bench-0001")
    new = _run("fa_Kq8_Vq4_default", pp=2.0, ts="2026-11-19T11:00:00+01:00", rid="20261119-110000-bench-0002")
    other = _run("fa_Kq8_Vq8_default", ts="2026-11-19T10:00:00+01:00", rid="20261119-100000-bench-0003")
    bad = _run("fa_Kq4_Vq8_default", status="aborted", rid="20261119-100100-bench-0004")
    _store(old, other, new, bad)
    sel = select(sweep="s1", order=["fa_Kq8_Vq8_default", "fa_Kq8_Vq4_default"])
    assert [r.run_id for r in sel.runs] == [other.run_id, new.run_id] and sel.dropped == [old.run_id]
    assert len(select(sweep="s1", latest=False).runs) == 3
    assert {r.status for r in select(sweep="s1", statuses=("ok", "aborted")).runs} == {"ok", "aborted"}
    assert select(sweep="nope").runs == []


def test_table_row_compare_and_prompt():
    a, b = _run("fa_Kq8_Vq8_default", tg=50.0), _run("fa_Kq8_Vq4_default", tg=55.0, note="SWEETSPOT")
    row = table_row(b)
    assert row["tg_tps"] == 55.0 and row["vram_peak_mib"] == 6000 and row["note"] == "SWEETSPOT"
    diff = {f: (x, y, d) for f, x, y, d in compare(a, b)}
    assert diff["tg_tps"] == (50.0, 55.0, "+10.0 %") and "variant" in diff
    _store(a, b)
    text = render_prompt(select(sweep="s1"), title="My title")
    assert "LLM DATA ANALYST PROMPT: My title" in text
    assert "TEST ID 2: fa_Kq8_Vq4_default @ ctx 4096   # SWEETSPOT" in text
    assert "run_id,tool,model,variant" in text and "[PASTE RAW RESULTS" not in text
    assert "kind,n_prompt" not in text                         # no tests stored -> no per-test CSV


# ---------------------------------------------------------------- CLI

def test_cli_report_commands(tmp_path):
    _store(_run("fa_Kq8_Vq8_default", tg=50.0, rid="20261119-100000-bench-aaaa"),
           _run("fa_Kq8_Vq4_default", tg=60.0, rid="20261119-100001-bench-bbbb"))
    cli = CliRunner()
    r = cli.invoke(main, ["report", "table", "--sweep", "s1", "--sort", "tg", "--format", "csv"])
    assert r.exit_code == 0, r.output
    lines = r.output.splitlines()
    assert lines[0].startswith("run_id,") and "fa_Kq8_Vq4_default" in lines[1]
    r = cli.invoke(main, ["report", "table", "--sweep", "s1", "--format", "md"])
    assert "| fa_Kq8_Vq4_default |" in r.output and "60.00 ★" in r.output
    r = cli.invoke(main, ["--json", "report", "chart", "--sweep", "s1", "-o", str(tmp_path / "c.png")])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["bars"] == 2 and (tmp_path / "c.png").is_file()
    r = cli.invoke(main, ["report", "chart", "--legacy-log", str(UD_TXT), "-o", str(tmp_path / "l.png")])
    assert r.exit_code == 0 and "17 configuration(s)" in r.output
    r = cli.invoke(main, ["report", "prompt", "--sweep", "s1", "-o", str(tmp_path / "p.md")])
    assert r.exit_code == 0 and "fa_Kq8_Vq4_default" in (tmp_path / "p.md").read_text(encoding="utf-8")
    r = cli.invoke(main, ["results", "compare", "aaaa", "bbbb"])
    assert r.exit_code == 0 and "+20.0 %" in r.output
    r = cli.invoke(main, ["results", "export", "--csv", str(tmp_path / "x.csv")])
    assert r.exit_code == 0 and (tmp_path / "x.csv").read_text(encoding="utf-8").count("\n") == 3
    r = cli.invoke(main, ["report", "table", "--sweep", "nothing"])
    assert r.exit_code != 0 and "no matching runs" in r.output


def test_chartrow_is_a_plain_dataclass():
    assert ChartRow("a", "b").status == "ok" and sys.version_info >= (3, 12)
