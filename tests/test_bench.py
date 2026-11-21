"""bench / benchy / sweep / results store (M7, §8, parity F4/F5). Fake tools; no GPU."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from mse_llm import bench as bb
from mse_llm import results as rs
from mse_llm.cli import main
from mse_llm.config import SweepCfg, load_config
from mse_llm.parsers import parse_benchy, parse_llama_bench
from mse_llm.runners.base import Command
from mse_llm.util import project_root
from mse_llm.variants import decode

sys.path.insert(0, str(Path(__file__).parent))
from test_monitor import Script, row  # noqa: E402
from test_proc import fake, free_port  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "results"
FAKE_BENCH = Path(__file__).parent / "fake_bench.py"


@pytest.fixture
def store(tmp_path, monkeypatch, hw_fixture):
    monkeypatch.setenv("MSE_LLM_RESULTS_DIR", str(tmp_path / "results"))
    return tmp_path / "results"


@pytest.fixture(scope="module")
def cfg():
    return load_config(project_root() / "config" / "config.example.toml")


# ---------------------------------------------------------------- parsers

def test_parse_llama_bench_jsonl():
    lines = (FIX / "llama_bench_omnicoder.jsonl").read_text(encoding="utf-8").splitlines()
    rows, eff = parse_llama_bench([*lines, "ggml_cuda_init: found 1 CUDA devices", "{not json"])
    assert [r.kind for r in rows] == ["pp", "tg"]
    assert rows[0].n_prompt == 512 and round(rows[0].pp_tps) == 3996 and rows[0].samples == [3445.45, 4547.4]
    assert rows[1].n_gen == 32 and round(rows[1].tg_tps, 1) == 63.2
    assert eff["build_commit"] == "ffc87c3a7" and eff["type_k"] == "q8_0" and eff["n_cpu_moe"] == 0
    assert rs.headline(rows) == {"pp_tps": 3996.43, "pp_std": 779.2, "n_prompt": 512, "tg_tps": 63.22,
                                 "tg_std": 0.49, "n_gen": 32}


def test_parse_benchy_json():
    rows, info = parse_benchy(FIX / "benchy_omnicoder.json")
    assert info["version"] == "0.4.0"
    assert [(r.n_prompt, r.n_gen, r.depth) for r in rows] == [(512, 32, 0), (2048, 32, 0)]
    assert round(rows[1].pp_tps) == 3948 and round(rows[1].tg_tps, 2) == 61.97 and round(rows[1].ttft_ms) == 430
    h = rs.headline(rows)
    assert h["n_prompt"] == 2048 and h["tg_tps"] == 61.97


def test_benchy_skips_context_prefill_rows():
    d = json.loads((FIX / "benchy_omnicoder.json").read_text(encoding="utf-8"))
    d["benchmarks"][0]["is_context_prefill_phase"] = True
    rows, _ = parse_benchy(d)
    assert len(rows) == 1


# ---------------------------------------------------------------- sweep config

def test_sweep_entries_status_annotations_and_matrix():
    sw = SweepCfg(model="m", variants=["extra_fa_Kq8_Vq{8,4}_default",
                                       {"v": "fa_Kq8_Vq8_default_moe30", "status": "skip", "note": "PC freeze"}],
                  matrix={"fa": ["on"], "ctk": ["q8_0"], "ctv": ["q8_0"], "n_cpu_moe": [18]},
                  annotations={"fa_Kq8_Vq4_default": {"status": "off", "note": "SWEETSPOT", "tag": "sweetspot"}})
    es = sw.entries()
    assert [e.v for e in es] == ["fa_Kq8_Vq8_default", "fa_Kq8_Vq4_default", "fa_Kq8_Vq8_default_moe30",
                                 "fa_Kq8_Vq8_default_moe18"]
    assert [e.status for e in es] == ["on", "off", "skip", "on"]
    assert es[1].tag == "sweetspot" and es[2].note == "PC freeze"


def test_config_has_one_sweep_per_bash_function(cfg):
    text = (project_root() / "llama_cpp_benchmark.sh").read_text(encoding="utf-8")
    funcs = set(re.findall(r"^(run_(?:BENCH|BENCHY_WITH_SERVER|INSPECT)_\w+)\(\)", text, re.M))
    ported = {k: sw for k, sw in cfg.sweeps.items() if sw.bash}        # M10+ adds sweeps with no bash function
    assert {sw.bash for sw in ported.values()} == funcs
    # every extras_list entry, commented out or not, is a sweep entry
    arrays = "\n".join(m.group(1) for m in re.finditer(r"local -a extras_list=\((.*?)\n\s*\)", text, re.S))
    assert sum(len(sw.variants) for sw in ported.values()) == len(re.findall(r'^\s*#?\s*"extra_\w+"', arrays,
                                                                                 re.M))
    q = cfg.sweeps["Qwen36_35B_A3B_UD_Q4_K_M_MTP_bench"].entries()
    assert [e.v for e in q if e.status == "on"] == ["fa_Kq8_Vq8_default_moe41", "fa_Kq8_Vq4_default_moe41"]
    assert {e.v: e.tag for e in q}["fa_Kq8_Vq4_default_moe34"] == "sweetspot"
    assert {e.v: e.status for e in q}["fa_Kq8_Vq8_default_moe30"] == "skip"


def test_sweep_jobs_and_filters(cfg):
    acts = {j.variant.name: a for j, a, _ in bb.sweep_jobs(cfg, "Qwen36_35B_A3B_UD_Q4_K_M_MTP_bench")}
    assert acts["fa_Kq8_Vq8_default_moe41"] == "run" and acts["fa_Kq8_Vq8_default_moe36"] == "skip-status"
    jobs = bb.sweep_jobs(cfg, "Qwen36_35B_A3B_UD_Q4_K_M_MTP_bench", include_off=True, match=["*moe3?"])
    acts = {j.variant.name: a for j, a, _ in jobs}
    assert acts["fa_Kq8_Vq8_default_moe36"] == "run" and acts["fa_Kq8_Vq8_default_moe41"] == "skip-filter"
    assert acts["fa_Kq8_Vq8_default_moe30"] == "skip-status"          # freeze note: only with --force
    with pytest.raises(ValueError, match="M12"):
        bb.sweep_jobs(cfg, "gemma_4_26B_A4b_UD_Q4_K_M_mmlu_pro")


# ---------------------------------------------------------------- store + resume

def _result(cfg, status="ok", variant="fa_Kq8_Vq8_default", build="b1", **kw) -> rs.RunResult:
    job = bb.Job("bench", "omnicoder_9b", decode(variant), 512, {"tg": [32]})
    cmd = job.command(cfg)
    t = rs.now()
    return rs.RunResult(run_id=rs.new_run_id("bench", t), key=job.key(cmd),
                        timestamp=t.isoformat(timespec="seconds"), tool="bench", engine="cuda_flash",
                        model="omnicoder_9b", variant=variant, ctx=512, params=cmd.params, build={"id": build},
                        status=status, **kw)


def test_store_write_find_and_index(cfg, store):
    r = _result(cfg, metrics={"pp_tps": 1.0, "tg_tps": 2.0})
    p = rs.write_run(r)
    assert p.parent.parent.name == "runs" and p.parent.name == r.timestamp[:7]
    (e,) = rs.read_index()
    assert e["run_id"] == r.run_id and e["pp_tps"] == 1.0 and e["status"] == "ok"
    assert rs.find_run(r.run_id[-4:]).run_id == r.run_id
    with pytest.raises(KeyError):
        rs.find_run("nope")


def test_resume_and_only_failed(cfg, store):
    rs.write_run(_result(cfg, "ok", "fa_Kq8_Vq8_default"))
    rs.write_run(_result(cfg, "aborted", "fa_Kq8_Vq4_default"))
    rs.write_run(_result(cfg, "ok", "fa_Kq4_Vq8_default", build="other-build"))
    env = bb.RunEnv(build={"cuda_flash": {"id": "b1"}})
    jobs = [(bb.Job("bench", "omnicoder_9b", decode(v), 512, {"tg": [32]}), "run", "")
            for v in ("fa_Kq8_Vq8_default", "fa_Kq8_Vq4_default", "fa_Kq4_Vq8_default")]
    assert [p.action for p in bb.plan(cfg, jobs, resume=True, env=env)] == ["skip-resume", "run", "run"]
    assert [p.action for p in bb.plan(cfg, jobs, only_failed=True, env=env)] == ["skip-resume", "run",
                                                                                 "skip-resume"]


def test_run_key_ignores_unset_params_and_order():
    a = rs.run_key("bench", "m", "v", 512, {"tg": [32], "repetitions": 1, "x": None})
    assert a == rs.run_key("bench", "m", "v", 512, {"repetitions": 1, "tg": [32]})
    assert a != rs.run_key("bench", "m", "v", 1024, {"tg": [32], "repetitions": 1})


# ---------------------------------------------------------------- jobs with fake tools

def _fake_bench(monkeypatch, *extra: str):
    def command(self, cfg, *, port=None, save_result=None):
        return Command("llama-bench", [sys.executable, str(FAKE_BENCH), *extra], model=self.model,
                       variant=self.variant.name, ctx=self.ctx, params={"tg": [32]})
    monkeypatch.setattr(bb.Job, "command", command)


def _opts(**kw):
    return bb.RunOpts(fit_check=False, samplers=[Script([row(proc_rss_mib=100)])], **kw)


def _job(variant="fa_Kq8_Vq8_default", ctx=512, tool="bench"):
    return bb.Job(tool, "omnicoder_9b", decode(variant), ctx)


def test_bench_job_ok(cfg, store, monkeypatch):
    _fake_bench(monkeypatch)
    r = asyncio.run(bb.run_job(cfg, _job(), _opts()))
    assert r.status == "ok", r.reason
    assert r.metrics["tg_tps"] == 63.22 and r.startup["effective"]["build_commit"] == "ffc87c3a7"
    assert r.monitor and r.monitor["samples"] >= 1
    saved = json.loads(r.path().read_text(encoding="utf-8"))
    assert saved["run_id"] == r.run_id and saved["schema"] == 1 and saved["hw"]["hash"]
    assert (store / "raw" / r.run_id / "llama-bench.log").read_text(encoding="utf-8").count("avg_ts") == 2
    assert (store / "hw").is_dir() and rs.read_index()[0]["run_id"] == r.run_id
    assert len(r.path().read_bytes()) < 50_000


def test_bench_job_failure_is_recorded(cfg, store, monkeypatch):
    _fake_bench(monkeypatch, "--exit", "3")
    r = asyncio.run(bb.run_job(cfg, _job(), _opts()))
    assert r.status == "failed" and "exit code 3" in r.reason and "failed on purpose" in r.reason


def test_guard_aborts_a_job(cfg, store, monkeypatch):
    _fake_bench(monkeypatch, "--sleep", "30")
    ram = Script([row(proc_rss_mib=100, ram_pct=97.0)], wait_for_pids=True)
    r = asyncio.run(bb.run_job(cfg, _job(), bb.RunOpts(fit_check=False, samplers=[ram])))
    assert r.status == "aborted" and r.reason.startswith("memory guard: system RAM at 97")
    assert r.duration_s < 25


def test_fit_check_skips_a_no_fit_variant(cfg, store, monkeypatch):
    from mse_llm.serve import FitCheck

    _fake_bench(monkeypatch)
    calls = []

    def fit(cfg_, model, v, ctx, engine):
        calls.append(ctx)
        return FitCheck(model, v.name, ctx, "NO-FIT")
    r = asyncio.run(bb.run_job(cfg, _job("fa_Kq8_Vq8_default_moe10"), bb.RunOpts(fit_runner=fit)))
    assert r.status == "skipped" and "NO-FIT" in r.reason and calls == [512]
    assert not (store / "raw" / r.run_id).exists()
    r = asyncio.run(bb.run_job(cfg, _job("fa_Kq8_Vq8_default_moe10"),
                               bb.RunOpts(fit_runner=fit, force=True, samplers=[Script([row(proc_rss_mib=1)])])))
    assert r.status == "ok" and r.fit["verdict"] == "NO-FIT"


def test_stuck_process_is_given_up_after_a_guard_stop(monkeypatch):
    """A process hung in the GPU driver ignores the kill for minutes: the job moves on."""
    monkeypatch.setattr(bb, "STUCK_EXIT_S", 0.2)
    res = rs.RunResult(run_id="x", key="k", timestamp="2026-11-19T00:00:00", tool="bench", engine="e", model="m",
                       variant="v")

    async def go():
        never = asyncio.create_task(asyncio.sleep(3600))
        stopped = asyncio.Event()
        stopped.set()
        return await bb._wait_or_give_up(never, stopped, res, 4242, bb.RunOpts()), never
    rc, never = asyncio.run(go())
    assert rc is None and never.cancelled() and "pid 4242 did not exit" in res.note


def test_benchy_job_with_fake_server(cfg, store, monkeypatch):
    port = free_port()

    def command(self, cfg_, *, port=None, save_result=None):
        server = Command("llama-server", fake(port), model=self.model, variant=self.variant.name, ctx=4096)
        argv = [sys.executable, str(FAKE_BENCH), "--save-result", save_result]
        return Command("llama-benchy", argv, model=self.model, variant=self.variant.name, ctx=4096,
                       server=server, params={"pp": [512, 2048], "tg": [32], "depth": [0], "runs": 2})
    monkeypatch.setattr(bb.Job, "command", command)
    r = asyncio.run(bb.run_job(cfg, _job(ctx=4096, tool="benchy"), _opts(port=port)))
    assert r.status == "ok", r.reason
    assert r.tool_version == "llama-benchy 0.4.0" and len(r.tests) == 2 and r.metrics["n_prompt"] == 2048
    assert r.startup["n_ctx"] == 4096 and r.startup["ready_s"] >= 0
    assert set(r.logs) == {"server", "tool", "result"}
    from mse_llm import proc as pr

    assert not pr.port_in_use(port)


# ---------------------------------------------------------------- CLI

def test_cli_sweep_list_show_and_dry_run(store):
    r = CliRunner().invoke(main, ["--json", "sweep", "list"])
    assert r.exit_code == 0, r.output
    names = {s["name"] for s in json.loads(r.output)}
    assert "omnicoder_9b_bench" in names and len(names) == 11      # 9 bash functions + the M10 spec + M11 cache sweeps
    r = CliRunner().invoke(main, ["sweep", "run", "Qwen36_35B_A3B_UD_Q4_K_M_MTP_bench", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "2 job(s) to run, 8 skipped" in r.output
    assert "--n-cpu-moe 41" in r.output and "PC freeze" in r.output
    r = CliRunner().invoke(main, ["--json", "sweep", "show", "Qwen36_35B_A3B_UD_Q4_K_M_MTP_bench"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert len(d["entries"]) == 10 and d["entries"][5]["note"] == "SWEETSPOT (CTX==65536)"
    r = CliRunner().invoke(main, ["sweep", "run", "Qwen36_28B_REAP20_A3B_Q4_K_M_bench"])
    assert r.exit_code == 0 and "nothing to run" in r.output and "--include-off" in r.output


def test_cli_bench_and_benchy_dry_run(store):
    r = CliRunner().invoke(main, ["bench", "omnicoder_9b", "--ctx", "512", "--ctx", "1024", "--dry-run"])
    assert r.exit_code == 0 and r.output.count("llama-bench") >= 2 and "-o jsonl" in r.output
    r = CliRunner().invoke(main, ["--json", "benchy", "omnicoder_9b", "--pp", "512", "--concurrency", "2",
                                  "--port", "18099", "--dry-run"])
    assert r.exit_code == 0, r.output
    (c,) = json.loads(r.output)
    assert c["argv"][c["argv"].index("--concurrency") + 1] == "2" and "--save-result" in c["argv"]
    assert c["server"]["argv"][-2:] == ["18099", "--metrics"] and c["params"]["pp"] == [512]


def test_cli_results_list_and_show(cfg, store):
    r1 = _result(cfg, metrics={"pp_tps": 4000.0, "tg_tps": 60.0}, argv=["llama-bench", "-p", "512"])
    rs.write_run(r1)
    r = CliRunner().invoke(main, ["--json", "results", "list"])
    assert [e["run_id"] for e in json.loads(r.output)] == [r1.run_id]
    r = CliRunner().invoke(main, ["results", "show", r1.run_id])
    assert r.exit_code == 0, r.output
    assert "reproduce:" in r.output and "llama-bench -p 512" in r.output
