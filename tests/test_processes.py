import subprocess
import sys

import psutil
from click.testing import CliRunner

from mse_llm.cli import main
from mse_llm.hw import processes
from mse_llm.hw.processes import ProcInfo, stop_processes


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_stop_engine_and_skip_gpu_compute():
    eng, gpu = _sleeper(), _sleeper()
    try:
        res = stop_processes([ProcInfo(eng.pid, "ollama.exe", "engine"),
                              ProcInfo(gpu.pid, "dwm.exe", "gpu-compute")])
        got = {r.pid: r.result for r in res}     # may include the venv launcher's child process
        assert got[eng.pid] == "stopped" and gpu.pid not in got
        assert eng.wait(timeout=5) is not None
        assert psutil.pid_exists(gpu.pid) and gpu.poll() is None
    finally:
        for p in (eng, gpu):
            p.kill()


def test_cli_stop_ollama_asks_and_skips_other_engines(monkeypatch):
    eng, other = _sleeper(), _sleeper()
    try:
        monkeypatch.setattr(processes, "detect_processes",
                            lambda: [ProcInfo(eng.pid, "ollama.exe", "engine", cmdline="ollama serve"),
                                     ProcInfo(other.pid, "llama-server.exe", "engine")])
        r = CliRunner().invoke(main, ["hw", "processes", "--stop-ollama"], input="n\n")
        assert r.exit_code == 1 and "Stop 1 Ollama" in r.output
        assert eng.poll() is None
        r = CliRunner().invoke(main, ["hw", "processes", "--stop-ollama", "--yes"])
        assert r.exit_code == 0, r.output
        assert "stopped" in r.output
        assert eng.wait(timeout=5) is not None
        assert other.poll() is None
    finally:
        eng.kill()
        other.kill()
