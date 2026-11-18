from pathlib import Path

import pytest
from pydantic import ValidationError

from mse_llm.buildprofiles import list_profiles, load_profile
from mse_llm.config import load_config

ROOT = Path(__file__).parents[1]


def test_example_config_loads():
    cfg = load_config(ROOT / "config" / "config.example.toml")
    assert cfg.defaults.threads == 4
    assert cfg.defaults.benchy.pp == [512, 2048, 8192, 32768]
    assert len(cfg.models) == 5
    name, eng = cfg.engine()
    assert name == "cuda_flash"
    assert eng.port == 8033 and eng.build_profile == "rtx4080-laptop"
    assert eng.bin_path().parts[-5:] == ("builds", "llama.cpp", "build", "bin", "Release")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MSE_LLM_THREADS", "8")
    monkeypatch.setenv("MSE_LLM_BENCHY_PP", "512 2048")
    monkeypatch.setenv("MSE_LLM_NOTIFY", "false")
    cfg = load_config(ROOT / "config" / "config.example.toml")
    assert cfg.defaults.threads == 8
    assert cfg.defaults.benchy.pp == [512, 2048]
    assert cfg.defaults.notify is False


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[defaults]\nthreds = 4\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(p)


def test_config_engine_overrides_profile(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[engines.cuda_flash]\nport = 9000\nhelp_snapshot = "a49db739f"\n', encoding="utf-8")
    _, eng = load_config(p).engine("cuda_flash")
    assert eng.port == 9000 and eng.help_snapshot == "a49db739f"
    assert eng.build_profile == "rtx4080-laptop"


def test_build_profile_extends_and_paths(monkeypatch, tmp_path):
    prof = load_profile("rtx4080-laptop")
    assert prof.data["cmake"]["GGML_CUDA"] == "ON"            # from _base-cuda-windows.toml
    assert prof.data["cmake"]["CMAKE_CUDA_ARCHITECTURES"] == "89-real"
    assert prof.gpu_match == "RTX 4080 Laptop"
    assert prof.bin_dir == prof.src / "build" / "bin" / "Release"
    monkeypatch.setenv("MSE_LLM_BUILDS_ROOT", str(tmp_path))
    assert load_profile("rtx4080-laptop").src == tmp_path / "llama.cpp"
    assert [p.name for p in list_profiles()] == ["rtx4080-laptop"]   # _base files are not targets
