from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
HW_FIXTURE = FIXTURES / "hw" / "rtx4080-laptop.json"


@pytest.fixture
def hw_fixture(monkeypatch):
    monkeypatch.setenv("MSE_LLM_TEST_HARDWARE", str(HW_FIXTURE))
    return HW_FIXTURE


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for k in ("MSE_LLM_CONFIG", "MSE_LLM_BUILDS_ROOT", "MSE_LLM_THREADS", "MSE_LLM_BENCHY_PP"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MSE_LLM_NO_NOTIFY", "1")      # tests never speak
    monkeypatch.setenv("MSE_LLM_RESULTS_DIR", str(tmp_path / "results"))   # never the repo's results/


@pytest.fixture(autouse=True)
def _performance_power(monkeypatch):
    """Benchmark commands stop on battery / a quiet power plan; the machine running the tests may be on one."""
    from mse_llm import hw

    monkeypatch.setattr(hw, "bench_power_problem", lambda p=None: None)
