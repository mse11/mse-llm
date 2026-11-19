"""Monitor, memory guard, summary statuses, notify (M6, §5, §9 item 7, parity F9/F13). Fake samplers only."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from mse_llm import notify
from mse_llm import proc as pr
from mse_llm.monitor import Guard, GuardCfg, Monitor, summarize
from mse_llm.runners import ServerError, ServerHandle

sys.path.insert(0, str(Path(__file__).parent))
from test_proc import fake, free_port  # noqa: E402


class Script:
    """A sampler that replays a list of dicts, one per tick (the last one repeats)."""

    name = "script"

    def open(self):
        pass

    def close(self):
        pass

    def __init__(self, rows: list[dict], wait_for_pids: bool = False) -> None:
        self.__init_rows(rows)
        self.wait_for_pids = wait_for_pids

    def __init_rows(self, rows):
        self.rows = rows
        self.i = 0
        self.ok = True
        self.error = None
        self.seen_pids: list[set[int]] = []

    def sample(self, pids):
        self.seen_pids.append(set(pids))
        if self.wait_for_pids and not pids:        # like PDH: no server numbers before the process exists
            return dict(BASE)
        r = self.rows[min(self.i, len(self.rows) - 1)]
        self.i += 1
        return dict(r)


BASE = {"vram_used_mib": 1081, "vram_free_mib": 11201, "vram_total_mib": 12282, "shared_mib": 108,
        "nonlocal_mib": 108, "ram_pct": 52.0, "ram_used_mib": 42000, "ram_total_mib": 81088, "gpu_temp_c": 50}


def row(**kw):
    return {**BASE, **kw}


# ---------------------------------------------------------------- guard rules

def test_pinned_host_buffers_are_not_a_spill():
    """Qwen3.6-28B moe41 measured: 15.9 GB server shared memory = fit's 15.8 GB host part."""
    g = Guard(GuardCfg(expected_host_mib=15807), row())
    for sh, used in ((12160, 2534), (15246, 2534), (15892, 4872), (15892, 6900)):
        assert g.check(row(proc_shared_mib=sh, vram_used_mib=used, vram_free_mib=12282 - used)) is None


def test_bash_rule_would_have_flagged_the_same_run():
    s = summarize([row(t=0.5, nonlocal_mib=16000, shared_mib=16000)], row(), interval=0.5,
                  guard=Guard(GuardCfg(enabled=False), row()))
    (gpu,) = [x for x in s.statuses if x.area == "gpu"]
    assert gpu.level == "warn" and "bash rule" in gpu.text


def test_spill_after_vram_full_trips_rule_a():
    """Qwen3.6-35B moe10 measured: VRAM full at 12140/12282, then the server's shared memory grows."""
    g = Guard(GuardCfg(), row())
    assert g.check(row(proc_shared_mib=1738, vram_used_mib=12140, vram_free_mib=142)) is None   # full: start
    assert g.check(row(proc_shared_mib=1820, vram_used_mib=12140, vram_free_mib=142)) is None   # +82
    r = g.check(row(proc_shared_mib=1866, vram_used_mib=12140, vram_free_mib=142))              # +128
    assert r and "dedicated VRAM is full" in r and "grew by 128 MiB" in r


def test_vram_not_full_resets_rule_a():
    g = Guard(GuardCfg(), row())
    g.check(row(proc_shared_mib=100, vram_free_mib=100))
    g.check(row(proc_shared_mib=300, vram_free_mib=2000))          # free again: reset
    assert g.check(row(proc_shared_mib=350, vram_free_mib=100)) is None


def test_beyond_expected_host_trips_rule_b():
    g = Guard(GuardCfg(expected_host_mib=5235), row())
    assert g.check(row(proc_shared_mib=5300)) is None              # within max(100, 2 %) = 104.7
    r = g.check(row(proc_shared_mib=9018))
    assert r and "3783 MiB more than the 5235 MiB" in r


def test_ram_and_commit_trip_rule_c():
    assert "system RAM at 96" in Guard(GuardCfg(), row()).check(row(ram_pct=96.0))
    assert "commit charge" in Guard(GuardCfg(), row()).check(row(commit_pct=97.0, commit_mib=1, commit_limit_mib=1))
    assert Guard(GuardCfg(enabled=False), row()).check(row(ram_pct=99.0)) is None


def test_adapter_delta_used_without_process_counters():
    g = Guard(GuardCfg(expected_host_mib=0, spill_stop_mib=100), row())
    assert g.shared(row(shared_mib=158)) == 50
    assert g.check(row(shared_mib=408)) and "300 MiB more" in g.check(row(shared_mib=408))


# ---------------------------------------------------------------- summary statuses (bash thresholds)

def test_summary_statuses_and_stats():
    samples = [row(t=0.5, ram_pct=65.0, gpu_temp_c=78, gpu_throttle=0x5), row(t=1.0, ram_pct=70.0, gpu_temp_c=80)]
    g = Guard(GuardCfg(expected_host_mib=0), row())
    s = summarize(samples, row(), interval=0.5, guard=g)
    lv = {x.area: x.level for x in s.statuses}
    assert lv == {"gpu": "pass", "ram": "warn", "temp": "warn"}
    assert s.get("ram_pct").peak == 70 and s.get("ram_pct").baseline == 52.0
    assert s.get("gpu_temp_c").mean == 79
    assert s.throttle == ["sw-power-cap"]
    hot = summarize([row(t=1, gpu_temp_c=90, ram_pct=85.0)], row(), interval=1)
    assert {x.area: x.level for x in hot.statuses if x.area != "gpu"} == {"ram": "fail", "temp": "fail"}


# ---------------------------------------------------------------- Monitor loop

def test_monitor_trips_and_calls_back(tmp_path):
    rows = [row(proc_shared_mib=0)] * 2 + [row(proc_shared_mib=1700, vram_free_mib=100)] + \
           [row(proc_shared_mib=2000, vram_free_mib=100)] * 3
    sampler = Script(rows)
    calls: list[str] = []

    async def go():
        async def on_trip(r):
            calls.append(r)
        async with Monitor(0.05, guard=GuardCfg(), samplers=[sampler], csv_path=tmp_path / "m.csv",
                           on_trip=on_trip) as m:
            await asyncio.wait_for(m.trip_event.wait(), 5)
        return m
    m = asyncio.run(go())
    assert len(calls) == 1 and "dedicated VRAM is full" in calls[0]
    assert m.summary().statuses[0].level == "fail"
    text = (tmp_path / "m.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("t,") and "proc_shared_mib" in text


def test_monitor_follows_a_server_tree_and_guard_stops_it(tmp_path):
    """End to end with the fake server: the guard callback stops the whole tree."""
    port = free_port()
    rows = [row(proc_shared_mib=0)] * 3 + [row(proc_shared_mib=50, vram_free_mib=10)] + \
           [row(proc_shared_mib=400, vram_free_mib=10)] * 50
    sampler = Script(rows, wait_for_pids=True)

    async def go():
        h = ServerHandle(fake(port, "--children", "1"), port=port, log_path=tmp_path / "s.log", grace=2)

        async def trip(reason):
            await h.stop()
        async with Monitor(0.1, guard=GuardCfg(), samplers=[sampler], on_trip=trip) as m:
            m.track(lambda: h.proc.proc.pid if h.proc.proc is not None else None)
            try:
                await h.start()                       # the guard may stop it while it is still loading
                await asyncio.wait_for(h.wait(), 20)
            except ServerError as e:
                assert "exited" in str(e)
        return m, h
    m, h = asyncio.run(go())
    assert m.tripped and h.stop_report.clean
    assert h.proc.pid in set().union(*sampler.seen_pids), "the sampler saw the server's pid"
    assert not pr.port_in_use(port)


# ---------------------------------------------------------------- notify

def test_notify_respects_switches(monkeypatch):
    said = []
    monkeypatch.setattr(notify, "speak", lambda t: said.append(t) or True)
    assert notify.notify("x") is False                   # conftest sets MSE_LLM_NO_NOTIFY=1
    monkeypatch.delenv("MSE_LLM_NO_NOTIFY")
    assert notify.notify("done", enabled=False) is False
    assert notify.notify("done") is True and said == ["done"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object")
def test_job_memory_cap_is_set():
    import win32job

    port = free_port()

    async def go():
        h = ServerHandle(fake(port), port=port, memory_cap_mib=512)
        async with h:
            info = win32job.QueryInformationJobObject(h.proc.job, win32job.JobObjectExtendedLimitInformation)
            return info
    info = asyncio.run(go())
    assert info["JobMemoryLimit"] == 512 * 2**20
    assert info["BasicLimitInformation"]["LimitFlags"] & win32job.JOB_OBJECT_LIMIT_JOB_MEMORY
