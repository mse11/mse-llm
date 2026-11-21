"""M11 (§4.4): MoE cache log parsing, Ctrl+C stop, staged sweeps, `tune moe-cache`, `report moe-cache`. No GPU."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from mse_llm import bench as bb
from mse_llm import proc as pr
from mse_llm import results as rs
from mse_llm import staged as st
from mse_llm import tune as tu
from mse_llm.cli import main
from mse_llm.config import SweepCfg, load_config
from mse_llm.parsers import moe_cache as mcp
from mse_llm.runners import benchy_command, server_command
from mse_llm.runners.base import Command
from mse_llm.util import IS_WINDOWS, project_root
from mse_llm.variants import CapabilityError, decode

sys.path.insert(0, str(Path(__file__).parent))
from test_monitor import Script, row  # noqa: E402
from test_proc import fake, free_port  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "moe_cache"
FAKE_BENCH = Path(__file__).parent / "fake_bench.py"
MODEL = "Qwen36_35B_A3B_UD_Q4_K_M_MTP"
SWEEP = "Qwen36_35B_A3B_UD_Q4_K_M_MTP_moecache"


@pytest.fixture
def store(tmp_path, monkeypatch, hw_fixture):
    monkeypatch.setenv("MSE_LLM_RESULTS_DIR", str(tmp_path / "results"))
    return tmp_path / "results"


@pytest.fixture(scope="module")
def cfg():
    return load_config(project_root() / "config" / "config.example.toml")


# ---------------------------------------------------------------- server log

def test_parse_real_server_log():
    """Lines of a real llama-server run (ffc87c3a7, Qwen3.6-35B -cmoe --moe-cache-mib 4096 --moe-cache-cpu-share 0.95,
    stopped with Ctrl+C)."""
    c = mcp.parse_log((FIX / "server_cmoe_mc4096_cs95.log").read_text(encoding="utf-8").splitlines())
    assert c["size_mib"] == 4093.19 and c["host_experts_mib"] == 18662.0 and c["cpu_share"] == 0.95
    assert c["devices"] == ["CUDA0"] and [g["layers"] for g in c["groups"]] == [37, 3]
    assert c["groups"][0] == {"layers": 37, "type": "q5_K", "slots": 2069, "slots_pct": 21.8, "tokens_worth": 7.0}
    assert c["hit_rate"] == 0.6218 and c["uploaded_mib"] == 7868.8 and c["cpu_misses"] == 59562
    assert c["small"]["hits"] == 104997 and c["large"]["hit_rate"] == 0.2767
    assert c["large_copied"] == {"experts": 10482, "mib": 6356.25}
    assert "invalid_config" not in c and "note" not in c


def test_parse_log_without_counters_and_invalid_configs():
    size = ("I impl:      CUDA0 MoE cache size =  2048.00 MiB for 18662.00 MiB of host experts, 0% of the misses on "
            "the CPU")
    c = mcp.parse_log(["noise", size])
    assert c["size_mib"] == 2048.0 and c["cpu_share"] == 0.0 and "clean exit" in c["note"]
    c = mcp.parse_log(["W llama_moe_cache: no layer has all of its experts in host memory, MoE cache is disabled"])
    assert c["invalid_config"].startswith("MoE cache is disabled")
    c = mcp.parse_log([size, "W impl: MoE cache budget is too small for 40 layers, they are not cached"])
    assert "too small for 40 layers" in c["invalid_config"]
    assert mcp.parse_log(["llama_kv_cache: size = 1 MiB"]) is None


def test_cache_variants_server_argv_and_bench_rejects_them(cfg):
    v = decode("fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmD")
    srv = server_command(cfg, MODEL, v, 8192, fit="off", purpose="benchy-server", metrics=True)
    a = srv.argv
    assert a[a.index("--moe-cache-mib") + 1] == "4096" and a[a.index("--moe-cache-cpu-share") + 1] == "0.95"
    assert "-cmoe" in a and a[a.index("-lm") + 1] == "dio" and a[-2:] == ["-lv", "4"] and srv.ok, srv.errors
    plain = server_command(cfg, MODEL, decode("fa_Kq8_Vq8_default_moe34"), 8192, purpose="benchy-server")
    assert "-lv" not in plain.argv
    assert "-lv" not in server_command(cfg, MODEL, v, 8192, purpose="serve").argv
    with pytest.raises(CapabilityError, match="moe-cache-mib"):
        v.flags("bench")
    cmd = benchy_command(cfg, MODEL, v, 8192, warmup=512)
    assert cmd.params["warmup"] == 512
    assert "warmup" not in benchy_command(cfg, MODEL, decode("fa_Kq8_Vq8_default_moe34"), 8192, warmup=512).params


# ---------------------------------------------------------------- Ctrl+C stop

CHILD = r"""
import sys, time
print("ready", flush=True)
try:
    time.sleep(60)
except KeyboardInterrupt:
    print("clean exit", flush=True)
    sys.exit(0)
"""


@pytest.mark.skipif(not IS_WINDOWS, reason="the Windows console path; POSIX uses SIGINT to the group")
def test_ctrl_c_stop_is_graceful():
    async def go():
        sp = pr.Spawned([sys.executable, "-c", CHILD], ctrl_c=True)
        await sp.start()
        for _ in range(50):
            if "ready" in sp.lines:
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.5)
        rep = await sp.stop(grace=15)
        return sp, rep
    sp, rep = asyncio.run(go())
    assert "clean exit" in sp.lines and sp.returncode == 0 and not rep.killed and not rep.left


def test_fast_stop_kills_without_ctrl_c():
    async def go():
        sp = pr.Spawned([sys.executable, "-c", CHILD], ctrl_c=True)
        await sp.start()
        await asyncio.sleep(1.0)
        rep = await sp.stop(grace=2, fast=True)
        return sp, rep
    sp, rep = asyncio.run(go())
    assert "clean exit" not in sp.lines and rep.killed and not rep.left


def test_benchy_cache_job_with_fake_server(cfg, store, monkeypatch):
    """Warm-up request, Ctrl+C stop, counters parsed into RunResult.moe_cache."""
    port = free_port()
    v = decode("fa_Kq8_Vq8_default_cmoe_mc4096_cs95")

    def command(self, cfg_, *, port=None, save_result=None, **kw):
        server = Command("llama-server", fake(port, "--moe-cache"), model=self.model, variant=self.variant.name,
                         ctx=4096)
        argv = [sys.executable, str(FAKE_BENCH), "--save-result", save_result]
        return Command("llama-benchy", argv, model=self.model, variant=self.variant.name, ctx=4096, server=server,
                       params={"pp": [512], "tg": [32], "warmup": 64})
    monkeypatch.setattr(bb.Job, "command", command)
    job = bb.Job("benchy", MODEL, v, 4096)
    r = asyncio.run(bb.run_job(cfg, job, bb.RunOpts(fit_check=False, port=port,
                                                    samplers=[Script([row(proc_rss_mib=100)])])))
    assert r.status == "ok", r.reason
    c = r.moe_cache
    assert c["size_mib"] == 4093.19 and c["requested_mib"] == 4096 and c["requested_cpu_share"] == 0.95
    assert c["warmup"] == {"tokens": 64, "tg_tps": 33.3}
    assert c["hit_rate"] == 0.75 and c["clean_exit"] is True
    assert rs.read_index()[-1]["hit_rate"] == 0.75
    assert not pr.port_in_use(port)


def test_invalid_cache_config_fails_the_run(cfg):
    job = bb.Job("benchy", MODEL, decode("fa_Kq8_Vq8_default_mc4096"), 4096)
    rec = bb.moe_cache_record(job, ["W impl: no layer has all of its experts in host memory, MoE cache is disabled"],
                              {})
    assert rec["invalid_config"].startswith("MoE cache is disabled")
    rec = bb.moe_cache_record(job, [], {})
    assert "add cmoe or moe<N>" in rec["invalid_config"]


# ---------------------------------------------------------------- staged sweeps

def test_resolve_best_winner_and_tuned():
    w0 = decode("fa_Kq8_Vq8_default_moe34_lmD")
    w1 = decode("fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmN")
    assert st.resolve("fa_Kq8_Vq8_default_moe34_lm{N,A,D}", []) == [
        "fa_Kq8_Vq8_default_moe34_lmN", "fa_Kq8_Vq8_default_moe34_lmA", "fa_Kq8_Vq8_default_moe34_lmD"]
    # lm{best}: the newest winner that sets lm
    assert st.resolve("fa_Kq8_Vq8_default_cmoe_mc{best}_cs{0,50}_lm{best}", [w0, w1]) == [
        "fa_Kq8_Vq8_default_cmoe_mc4096_cs0_lmN", "fa_Kq8_Vq8_default_cmoe_mc4096_cs50_lmN"]
    assert st.resolve("fa_Kq8_Vq8_default_moe34_lm{best}", [w0]) == ["fa_Kq8_Vq8_default_moe34_lmD"]
    # {winner}: the previous winner with the pattern's tokens on top; moe<N> replaces cmoe
    assert st.resolve("{winner}_K{q8,f8}_Vq4", [w0, w1]) == [
        "fa_Kq8_Vq4_default_cmoe_mc4096_cs95_lmN", "fa_Kf8_Vq4_default_cmoe_mc4096_cs95_lmN"]
    assert st.resolve("{winner}_moe37", [w1]) == ["fa_Kq8_Vq8_default_moe37_mc4096_cs95_lmN"]
    assert st.resolve("{winner}_sd3mtp", [w1]) == ["fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmN_sd3mtp"]
    with pytest.raises(st.Unresolved):
        st.resolve("x_mc{best}", [])
    with pytest.raises(st.Unresolved):
        st.resolve("{winner}_Kf8", [])
    with pytest.raises(st.StageError, match="no earlier stage winner sets mc"):
        st.resolve("fa_cmoe_mc{best}", [w0])
    tuned = decode("fa_Kq8_Vq8_default_cmoe_mc5120_cs90_lmD")
    assert st.resolve("fa_Kq8_Vq8_default_cmoe_mc{tuned}_cs{tuned}_lmD", [], tuned) == [
        "fa_Kq8_Vq8_default_cmoe_mc5120_cs90_lmD"]
    with pytest.raises(st.Unresolved, match="tune moe-cache"):
        st.resolve("fa_cmoe_mc{tuned}", [])


def test_pick_winner_ignores_failed_and_uses_the_goal():
    runs = [st.StageRun("a", "ok", tg=30, pp=1000), st.StageRun("b", "ok", tg=35, pp=500),
            st.StageRun("c", "failed", tg=50), st.StageRun("d", "skipped")]
    assert st.pick_winner(runs, "tg").variant == "b"
    assert st.pick_winner(runs, "pp").variant == "a"
    assert st.pick_winner(runs, "both").variant == "a"
    assert st.pick_winner(runs[2:], "tg") is None


def test_config_cache_sweep_is_staged(cfg):
    sw = cfg.sweeps[SWEEP]
    assert sw.tool == "benchy" and len(sw.stages) == 5 and sw.goal == "tg" and sw.warmup == 512
    assert sw.entries() == []
    assert SweepCfg(model="m", stages=[["fa_moe{1,2}"]]).stages == [["fa_moe{1,2}"]]


def _fake_run_job(tg_of):
    async def run_job(cfg, job, opts, env=None):
        env = env or bb.RunEnv()
        cmd = job.command(cfg, save_result="x")
        res = bb._base_result(cfg, job, cmd, env, rs.new_run_id(job.tool), rs.now().isoformat(timespec="seconds"))
        tg = tg_of(job.variant)
        res.status = "ok" if tg else "failed"
        res.reason = None if tg else "fake failure"
        res.metrics = {"tg_tps": tg, "pp_tps": 1000.0} if tg else {}
        if job.variant.moe_cache_mib:
            res.moe_cache = {"hit_rate": 0.5, "size_mib": float(job.variant.moe_cache_mib)}
        if opts.write:
            rs.write_run(res)
        return res
    return run_job


def test_staged_sweep_runs_stage_by_stage(store, monkeypatch):
    """Stage winners feed the next stage; --until-stage 2 stops there; --resume reuses every run."""
    from mse_llm import hw

    monkeypatch.setattr(hw, "bench_power_problem", lambda: None)
    monkeypatch.setattr("mse_llm.notify.notify", lambda *a, **k: None)

    def tg(v):
        if v.moe_cache_mib == 6144:
            return None                                       # pretend 6144 failed
        base = {"none": 30.0, "auto": 31.0, "dio": 32.0}[v.load_mode]
        return base + (v.moe_cache_mib or 0) / 1000 + (v.moe_cache_cpu_share or 0) / 100
    monkeypatch.setattr(bb, "run_job", _fake_run_job(tg))
    r = CliRunner().invoke(main, ["sweep", "run", SWEEP, "--until-stage", "2", "--allow-low-power", "--no-fit-check"])
    assert r.exit_code == 0, r.output
    assert "stage 0 winner: fa_Kq8_Vq8_default_moe34_lmD" in r.output
    assert "stage 1 winner: fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmD" in r.output
    assert "stage 2 winner: fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmD" in r.output
    idx = rs.read_index()
    assert len(idx) == 3 + 12 + 3                  # stage 2: cs0 and cs95 at mc4096 lmD ran in stage 1: reused
    assert {e["stage"] for e in idx} == {0, 1, 2}
    summ = json.loads(next((store / "staged" / SWEEP).glob("*.json")).read_text(encoding="utf-8"))
    assert summ["winner"] == "fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmD"
    assert [s["winner"] for s in summ["stages"]] == [
        "fa_Kq8_Vq8_default_moe34_lmD", "fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmD",
        "fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmD"]
    # dry run of the whole sweep: stages 0-2 resolve from stored runs, stage 3 is shown
    r = CliRunner().invoke(main, ["--json", "sweep", "run", SWEEP, "--dry-run"])
    assert r.exit_code == 0, r.output
    stages = json.loads(r.output)
    assert stages[3]["variants"] == ["fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmD",          # {winner}: stored run
                                     "fa_Kq8_Vq8_default_moe37_mc4096_cs95_lmD",
                                     "fa_Kq8_Vq8_default_moe38_mc4096_cs95_lmD",
                                     "fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmD_sd3mtp"]
    assert [r["variant"] for r in stages[3]["runs"]] == ["fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmD"]
    assert stages[4]["variants"][0] == "fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmD" and len(stages[4]["variants"]) == 6
    # resume: only the 4 failed mc6144 runs run again (like any sweep's --resume)
    r = CliRunner().invoke(main, ["sweep", "run", SWEEP, "--until-stage", "2", "--resume", "--allow-low-power"])
    assert r.exit_code == 0, r.output
    again = rs.read_index()[len(idx):]
    assert len(again) == 4 and all("mc6144" in e["variant"] for e in again)
    r = CliRunner().invoke(main, ["sweep", "show", SWEEP])
    assert r.exit_code == 0, r.output
    assert "sweep winner: fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmD" in r.output
    r = CliRunner().invoke(main, ["--json", "report", "moe-cache", "--sweep", SWEEP])
    assert r.exit_code == 0, r.output
    rows = json.loads(r.output)["rows"]
    assert {x["placement"] for x in rows} == {"moe34", "cmoe"} and any(x["hit_rate"] == 0.5 for x in rows)


def test_report_moe_cache_chart(tmp_path):
    from mse_llm.report.moe_cache import chart

    rows = [{"variant": "fa_Kq8_Vq8_default_moe34_lmD", "tg_tps": 30.0, "pp_short": 512, "pp_short_tps": 700.0,
             "pp_long": 8192, "pp_long_tps": 900.0, "duration_s": 100.0, "ready_s": 20.0, "vram_peak_mib": 9000.0,
             "hit_rate": None, "mc": None, "uploaded_mib": None, "pcie_rx_mean_mbs": 500.0},
            {"variant": "fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmD", "tg_tps": 35.0, "pp_short": 512,
             "pp_short_tps": 300.0, "pp_long": 8192, "pp_long_tps": 800.0, "duration_s": 120.0, "ready_s": 25.0,
             "vram_peak_mib": 8000.0, "hit_rate": 0.6, "mc": 4096, "uploaded_mib": 7000.0, "pcie_rx_mean_mbs": 1400}]
    p = chart(rows, tmp_path / "c.png")
    assert p.is_file() and p.stat().st_size > 10_000


# ---------------------------------------------------------------- tune moe-cache

TUNER_OUT = """1/6 409 MiB, share 0: 21.3 t/s
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`: 18662 MiB of experts, CUDA0 11341 MiB free, cache up to 6800 MiB

  cache MiB  share   t/s  hits
       1856      0  21.3   41%
       1856    0.5  24.0   40%
       5568   0.95  30.1   70%

machine: uploads 9.8 GiB/s, CPU 31.5 GiB/s, CPU part 0.40 ms, overlap 1.2 ms/GiB, rest 8.10 ms per token, fit 3%

recommended: --moe-cache-mib 6784 --moe-cache-cpu-share 0.9 (31.2 t/s, 26.0 with share 0)
"""


def test_tune_output_parsed_into_a_variant():
    base = decode("fa_Kq8_Vq8_default_cmoe_lmD")
    rec = tu.parse_output(TUNER_OUT.splitlines(), base)
    assert rec["variant"] == "fa_Kq8_Vq8_default_cmoe_mc6784_cs90_lmD"
    assert rec["recommended_mib"] == 6784 and rec["recommended_share"] == 0.9 and rec["tg_tps_share0"] == 26.0
    assert rec["experts_mib"] == 18662.0 and rec["cap_mib"] == 6800 and rec["machine"]["upload_gib_s"] == 9.8
    assert [c["share"] for c in rec["calibration"]] == [0.0, 0.5, 0.95]
    fit = tu.parse_output(["all experts fit on CUDA0 (9000 MiB): drop `-cmoe`, or tune a smaller cache"], base)
    assert fit["all_fit"] == {"device": "CUDA0", "mib": 9000} and "variant" not in fit


TUNE_SCRIPT = project_root() / "builds" / "llama.cpp" / "scripts" / "moe-cache-tune.py"
needs_script = pytest.mark.skipif(not TUNE_SCRIPT.is_file(), reason="needs the llama.cpp checkout in builds/")


@needs_script
def test_tune_failure_reason():
    lines = ["the runs do not fit the time model, was the machine busy? delete X and run again",
             "machine: uploads 3.8 GiB/s, CPU 5.4 GiB/s, CPU part 4.56 ms, overlap 20.3 ms/GiB, fit within 31.8%",
             "         7936  57.9*  45.7   45.3  44.6  43.8  42.4"]
    r = tu.failure_reason(lines)
    assert r.startswith("the runs do not fit the time model") and "31.8 % off" in r


def test_tune_command_strips_what_the_tuner_sets(cfg, store):
    t = tu.tune_command(cfg, MODEL, decode("fa_Kq8_Vq8_default_cmoe_mc4096_cs95_lmD"), 32768, None, max_mib=5000)
    sa = t["server_argv"]
    for flag in ("--host", "--port", "--metrics", "--moe-cache-mib", "--moe-cache-cpu-share", "-lv"):
        assert flag not in sa
    assert "-cmoe" in sa and sa[sa.index("-lm") + 1] == "dio" and t["base"].name == "fa_Kq8_Vq8_default_cmoe_lmD"
    assert t["argv"][1:3] == ["-m", "mse_llm.tune_shim"] and t["argv"][t["argv"].index("--max-mib") + 1] == "5000"
    assert t["argv"][t["argv"].index("--") + 1:] == sa
    same = tu.tune_command(cfg, MODEL, decode("fa_Kq8_Vq8_default_cmoe_lmD"), 32768, None)
    assert same["out"] == t["out"]                  # same server argv -> same run directory (calibration reused)
    with pytest.raises(tu.TuneError, match="host memory"):
        tu.base_variant(decode("fa_Kq8_Vq8_default"))
    with pytest.raises(tu.TuneError, match="speculative"):
        tu.base_variant(decode("fa_Kq8_Vq8_default_cmoe_sd3mtp"))


def test_tuned_recommendation_feeds_staged_sweeps(store):
    tu.save_tuned(MODEL, {"variant": "fa_Kq8_Vq8_default_cmoe_mc6784_cs90_lmD", "recommended_mib": 6784,
                          "recommended_share": 0.9, "tg_tps": 31.2}, "run-x")
    t = st.load_tuned(MODEL)
    assert t.moe_cache_mib == 6784 and t.moe_cache_cpu_share == 90
    assert st.resolve("fa_Kq8_Vq8_default_cmoe_mc{tuned}_cs{tuned}_lmD", [], t) == [
        "fa_Kq8_Vq8_default_cmoe_mc6784_cs90_lmD"]


def test_tune_shim_routes_sigint_to_ctrl_c(tmp_path):
    """The tuner's `proc.send_signal(signal.SIGINT)` (ValueError on Windows) ends a llama-server cleanly."""
    script = tmp_path / "tuner.py"
    server = tmp_path / "llama-server.py"
    server.write_text(CHILD, encoding="utf-8")
    script.write_text(
        "import signal, subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, {str(server)!r}], stdout=subprocess.PIPE, text=True)\n"
        "assert p.stdout.readline().strip() == 'ready'\n"
        "time.sleep(0.5)\n"
        "p.send_signal(signal.SIGINT)\n"
        "out = p.communicate(timeout=20)[0]\n"
        "print('server said:', out.strip(), 'rc', p.returncode)\n", encoding="utf-8")
    import subprocess

    r = subprocess.run([sys.executable, "-m", "mse_llm.tune_shim", str(script)], capture_output=True, text=True,
                       timeout=60, cwd=project_root())
    assert r.returncode == 0, r.stderr
    assert "server said: clean exit rc 0" in r.stdout


@needs_script
def test_cli_tune_dry_run(store):
    r = CliRunner().invoke(main, ["--json", "tune", "moe-cache", MODEL, "-v", "fa_Kq8_Vq8_default_cmoe_lmA",
                                  "--ctx", "32768", "--dry-run"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["base"] == "fa_Kq8_Vq8_default_cmoe_lmA" and d["argv"][3].endswith("moe-cache-tune.py")
    r = CliRunner().invoke(main, ["tune", "moe-cache", MODEL, "-v", "fa_Kq8_Vq8_default", "--dry-run"])
    assert r.exit_code != 0 and "host memory" in r.output
