"""ServerHandle / proc: spawn, readiness, tree kill, port checks, output filter (M5, parity F6/F14)."""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path

import psutil
import pytest

from mse_llm import proc as pr
from mse_llm import serve as sv
from mse_llm.runners import ServerError, ServerHandle

FAKE = Path(__file__).parent / "fake_server.py"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake(port: int, *extra: str) -> list[str]:
    return [sys.executable, str(FAKE), "--port", str(port), *extra]


def test_output_filter_modes():
    f = pr.OutputFilter("filter")
    assert f("llama_context: n_ctx = 4096")
    assert f("main: server is listening on http://0.0.0.0:8033")
    assert f("some Error happened") and f("a warning")
    assert not f("srv  update_slots: all good")
    assert pr.OutputFilter("all")("anything")
    assert not pr.OutputFilter("none")("llama_context: n_ctx")
    assert pr.OutputFilter("filter", [])("no patterns = all")     # bash: empty filter -> all
    with pytest.raises(ValueError):
        pr.OutputFilter("loud")


CYCLES = int(os.environ.get("MSE_LLM_TEST_CYCLES", "10"))    # M5 acceptance: MSE_LLM_TEST_CYCLES=50


def test_server_handle_start_stop_cycles_no_orphans(tmp_path):
    """Parity F6 / M5: repeated start/stop cycles, every process of the tree gone, port free each time."""
    port = free_port()

    async def cycle(i: int) -> tuple[list[int], bool]:
        h = ServerHandle(fake(port, "--children", "2"), port=port, ready_timeout=20, log_path=tmp_path / "s.log",
                         grace=2)
        async with h as ep:
            assert ep.port == port and ep.base_url == f"http://localhost:{port}/v1"
            pids = [p.pid for p in pr.tree(ep.pid)]
            assert len(pids) >= 3, f"cycle {i}: expected server + 2 children, got {pids}"
        assert h.stop_report is not None and h.stop_report.clean, h.stop_report.summary()
        return pids, pr.port_in_use(port)

    async def run_all():
        out = []
        for i in range(CYCLES):
            out.append(await cycle(i))
        return out

    results = asyncio.run(run_all())
    for pids, in_use in results:
        assert not in_use
        assert not [p for p in pids if psutil.pid_exists(p) and psutil.Process(p).status() != psutil.STATUS_ZOMBIE]


def test_server_handle_waits_for_health_and_logs(tmp_path):
    port = free_port()
    log = tmp_path / "x.log"
    seen: list[str] = []
    info = sv.StartupInfo()

    async def go():
        h = ServerHandle(fake(port, "--ready-after", "1.2"), port=port, ready_timeout=20, log_path=log,
                         sink=seen.append)
        h.proc.watchers.append(info.feed)
        async with h as ep:
            assert ep.ready_s >= 1.0
    asyncio.run(go())
    assert any("server is listening" in s for s in seen)
    assert "n_ctx" in log.read_text(encoding="utf-8")
    assert info.n_ctx == 4096 and info.offloaded == (10, 10)
    assert info.differences(4096) == [] and info.differences(8192)


def test_server_handle_fails_fast_when_process_exits(tmp_path):
    port = free_port()

    async def go():
        h = ServerHandle(fake(port, "--ready-after", "60", "--exit-after", "0.5"), port=port, ready_timeout=30,
                         log_path=tmp_path / "x.log")
        with pytest.raises(ServerError) as ei:
            await h.start()
        return ei.value, h
    err, h = asyncio.run(go())
    assert "exited with code" in str(err)
    assert any("exiting on purpose" in ln for ln in err.log_tail)
    assert not pr.port_in_use(port)


def test_server_handle_timeout_kills_tree(tmp_path):
    port = free_port()

    async def go():
        h = ServerHandle(fake(port, "--ready-after", "60", "--children", "1"), port=port, ready_timeout=1.5,
                         log_path=tmp_path / "x.log", grace=1)
        with pytest.raises(ServerError, match="not ready within"):
            await h.start()
        return h
    h = asyncio.run(go())
    assert h.stop_report is not None and h.stop_report.clean and len(h.stop_report.pids) >= 2


def test_port_in_use_is_refused(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]

        async def go():
            h = ServerHandle(fake(port), port=port, log_path=tmp_path / "x.log")
            with pytest.raises(ServerError, match=f"port {port} is in use"):
                await h.start()
            assert h.proc.proc is None             # nothing was started
        asyncio.run(go())


def test_missing_binary(tmp_path):
    async def go():
        h = ServerHandle([str(tmp_path / "nope.exe")], port=free_port(), log_path=tmp_path / "x.log")
        with pytest.raises(ServerError, match="not found"):
            await h.start()
    asyncio.run(go())


def test_detached_server_outlives_handle_and_stops_by_pid(tmp_path):
    port = free_port()
    log = tmp_path / "bg.log"

    async def go():
        h = ServerHandle(fake(port, "--children", "1"), port=port, log_path=log, detached=True, ready_timeout=20)
        ep = await h.start()                       # no context manager: keep it running
        return ep.pid
    pid = asyncio.run(go())
    try:
        assert pr.port_in_use(port)
        assert "server is listening" in log.read_text(encoding="utf-8")
        assert len(pr.tree(pid)) >= 2
    finally:
        rep = pr.stop_tree(pid, signal_first=False, port=port)
    assert rep.clean and not pr.port_in_use(port)


def test_router_load_via_fake(tmp_path):
    port = free_port()

    async def go():
        async with ServerHandle(fake(port, "--models", "a,b"), port=port, log_path=tmp_path / "r.log") as ep:
            assert await sv.load_model(ep.root, "a") == "loaded"
            assert await sv.load_model(ep.root, "b") == "loaded"
            st = {m["id"]: sv._status(m) for m in await sv.router_models(ep.root)}
            assert st == {"a": "unloaded", "b": "loaded"}
            with pytest.raises(RuntimeError, match="HTTP 400"):
                await sv.load_model(ep.root, "nope")
            await sv.unload_model(ep.root, "b")
    asyncio.run(go())


def test_state_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "servers_dir", lambda: tmp_path)
    me = psutil.Process()
    sv.write_state("router", 1234, {"pid": me.pid, "create_time": me.create_time(), "port": 1234})
    (st,) = sv.read_state("router")
    assert st["port"] == 1234 and sv.state_alive(st)
    assert not sv.state_alive({"pid": me.pid, "create_time": me.create_time() - 100})
    assert sv.read_state("router", 999) == []
