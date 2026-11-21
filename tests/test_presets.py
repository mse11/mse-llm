"""presets validate / fix / dump / generate (§4.3, parity F8) and the serve fit preview (§0.3)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from mse_llm import flagcheck
from mse_llm import presets as ps
from mse_llm import serve as sv
from mse_llm import variants as vv
from mse_llm.cli import main
from mse_llm.config import load_config
from mse_llm.util import project_root
from mse_llm.variants import decode

FIX = Path(__file__).parent / "fixtures"
BROKEN = FIX / "presets" / "original_broken.ini"
FIXED = project_root() / "llama_cpp_presets.ini"


@pytest.fixture(scope="module")
def server_help():
    _, helps = flagcheck.load_helps("a49db739f")
    return helps["llama-server"]


@pytest.fixture(scope="module")
def cfg():
    return load_config(project_root() / "config" / "config.example.toml")


def _paths_exist(monkeypatch):
    monkeypatch.setattr(ps, "_path_exists", lambda p, base: True)


def test_original_ini_findings_match_plan(server_help, monkeypatch):
    _paths_exist(monkeypatch)
    val = ps.validate(ps.parse(BROKEN), server_help)
    errs = [f.message for f in val.errors]
    assert any("globals must be [*]" in m for m in errs)
    assert any(m.startswith("no-mmap = 1") and "load-mode = none" in m for m in errs)
    warns = [f.message for f in val.warnings]
    for needle in ("fa = 1", "jinja = 1", "ngl = 999", 'no "version = 1"'):
        assert any(needle in m for m in warns), needle
    assert len(val.errors) == 2


def test_fix_rewrites_only_the_broken_lines(server_help, monkeypatch, tmp_path):
    _paths_exist(monkeypatch)
    pf = ps.parse(BROKEN)
    new = ps.fixed_text(pf, ps.validate(pf, server_help))
    assert new.startswith("version = 1\n")
    assert "[*]" in new and "[global defaults]" not in new
    assert "load-mode = none" in new and "no-mmap" not in new.split("[Qwen")[0]
    assert "fa = on" in new
    # comments, the commented-out block and the original command lines are kept
    old_comments = [ln for ln in BROKEN.read_text(encoding="utf-8").splitlines() if ln.startswith(("#", ";"))]
    assert all(c in new for c in old_comments)
    p = tmp_path / "f.ini"
    p.write_text(new, encoding="utf-8")
    assert ps.validate(ps.parse(p), server_help).errors == []


def test_fixed_ini_has_no_errors(server_help, monkeypatch):
    _paths_exist(monkeypatch)
    val = ps.validate(ps.parse(FIXED), server_help)
    assert val.errors == [] and not val.fixable


def test_validate_catches_bad_values_and_router_keys(server_help, monkeypatch, tmp_path):
    _paths_exist(monkeypatch)
    p = tmp_path / "x.ini"
    p.write_text("version = 1\n[*]\nport = 8033\n[m]\nmodel = a.gguf\nctv = turbo4\nbogus-key = 1\n"
                 "[nomodel]\nctk = q8_0\n", encoding="utf-8")
    val = ps.validate(ps.parse(p), server_help)
    msgs = [f.message for f in val.findings]
    assert any("ctv = turbo4" in m and "not in" in m for m in msgs)
    assert any("bogus-key" in m for m in msgs)
    assert any("owned by the router" in m for m in msgs)
    assert any("no model" in m for m in msgs)


def test_dump_maps_every_preset_back_to_default_serve(server_help, cfg):
    pf = ps.parse(FIXED)
    for s in pf.models:
        d = ps.dump(pf, s.name, server_help, cfg)
        assert d.model_key, s.name
        bash = cfg.models[d.model_key].bash_serve      # the INI is the bash baseline: it serves the old default
        assert d.matches_default or (bash and vv.canonical(bash) == d.variant), (s.name, d.variant, d.default_serve)
    d = ps.dump(pf, "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL", server_help, cfg)
    f = d.flags
    assert f[:2] == ["-m", pf.find(d.name).model]
    assert f[f.index("-fa") + 1] == "on"                  # optional-value flag keeps its value
    assert "--jinja" in f and f[f.index("--ctx-size") + 1] == "65536"
    assert f[f.index("--n-cpu-moe") + 1] == "16"
    assert "version" not in " ".join(f) and "alias" not in d.skipped
    assert flagcheck.check_argv(["llama-server", *f], "llama-server", {"llama-server": server_help}) == []


def test_find_by_alias_and_case(server_help):
    pf = ps.parse(FIXED)
    assert pf.find("tesslate_omnicoder-9b-q4_k_m").name == "Tesslate_OmniCoder-9B-Q4_K_M"
    with pytest.raises(KeyError):
        pf.find("nope")


def test_generate_from_config_is_valid(cfg, server_help, monkeypatch, tmp_path):
    _paths_exist(monkeypatch)
    text, _ = ps.generate(cfg)
    p = tmp_path / "gen.ini"
    p.write_text(text, encoding="utf-8")
    pf = ps.parse(p)
    assert "version" in pf.top and "*" in pf.sections
    val = ps.validate(pf, server_help)
    assert val.errors == [], [f.message for f in val.errors]
    # every model with a default_serve is a section, and it decodes back to that variant
    want = {m.alias or m.label: k for k, m in cfg.models.items() if m.default_serve}
    assert {s.name for s in pf.models} == set(want)
    for s in pf.models:
        d = ps.dump(pf, s.name, server_help, cfg)
        assert d.matches_default, (s.name, d.variant, d.default_serve)


def test_cli_presets_validate_and_fix(tmp_path, monkeypatch):
    _paths_exist(monkeypatch)
    p = tmp_path / "b.ini"
    p.write_text(BROKEN.read_text(encoding="utf-8"), encoding="utf-8")
    r = CliRunner().invoke(main, ["presets", "validate", "--file", str(p)])
    assert r.exit_code == 1 and "2 errors" in r.output
    r = CliRunner().invoke(main, ["presets", "fix", "--file", str(p), "--write"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "b.ini.bak").read_text(encoding="utf-8") == BROKEN.read_text(encoding="utf-8")
    r = CliRunner().invoke(main, ["presets", "validate", "--file", str(p)])
    assert r.exit_code == 0 and "0 errors" in r.output


def test_cli_presets_serve_dry_run_refuses_on_errors(tmp_path, monkeypatch):
    _paths_exist(monkeypatch)
    p = tmp_path / "b.ini"
    p.write_text(BROKEN.read_text(encoding="utf-8"), encoding="utf-8")
    r = CliRunner().invoke(main, ["presets", "serve", "--file", str(p), "--dry-run"])
    assert r.exit_code != 0 and "has errors" in r.output
    r = CliRunner().invoke(main, ["presets", "serve", "--dry-run", "--load", "Tesslate_OmniCoder-9B-Q4_K_M"])
    assert r.exit_code == 0, r.output
    assert "--models-max 1" in r.output and "--models-memory-margin 1024" in r.output
    assert "load       Tesslate_OmniCoder-9B-Q4_K_M" in r.output


def test_cli_serve_preset_dry_run():
    r = CliRunner().invoke(main, ["serve", "--preset", "Tesslate_OmniCoder-9B-Q4_K_M", "--dry-run", "--port", "18033"])
    assert r.exit_code == 0, r.output
    assert "--fit on --host 0.0.0.0 --port 18033" in r.output


# ---------------------------------------------------------------- what --fit would change

OT = ('"' + ",".join(f"blk\\.{i}\\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU" for i in range(34)) + '"')


def _cp(rc: int, out: str = "", err: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], rc, out, err)


def test_preview_no_change_when_it_fits(cfg):
    runner = lambda argv: _cp(0, f"-c 65536 -ngl -2 -ot {OT}\n")   # noqa: E731
    p = sv.fit_preview(cfg, None, "m.gguf", decode("fa_Kq8_Vq8_default_moe34"), 65536, runner=runner)
    assert p.ok and p.changes == [] and "no change" in p.message
    assert "--fit" not in p.argv and "--fit-print" not in p.argv


def test_preview_locked_by_ngl_shows_unlocked_alternative(cfg):
    calls = []
    ot = '"' + ",".join(f"blk\\.{i}\\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU" for i in range(15, 41)) + \
         ',blk\\.14\\.ffn_down.*=CPU"'

    def runner(argv):
        calls.append(argv)
        if "-ngl" in argv:
            return _cp(1, "", "W common_fit_params: failed to fit params to free device memory: n_gpu_layers "
                       "already set by user to -2, abort\n")
        return _cp(0, f"-c 65536 -ngl 42 -ot {ot}\n")

    p = sv.fit_preview(cfg, None, "m.gguf", decode("fa_Kq8_Vq8_default_moe10"), 65536, runner=runner)
    assert not p.ok and p.locked and "can't adjust" in p.message
    assert len(calls) == 2 and "-ngl" not in calls[1] and "--n-cpu-moe" not in calls[1]
    u = p.unlocked
    assert u.ok
    assert any(c.startswith("-ngl all → 42") or c.startswith("-ngl auto → 42") for c in u.changes)
    assert any("layers 15–40 on the CPU, part of layer 14 (requested: 0–9, -ncmoe 10)" in c for c in u.changes)


def test_preview_ctx_change():
    assert sv.diff_fitted(["-c", "65536", "-ngl", "all"], ["-c", "32768", "-ngl", "-2"], None) == ["-c 65536 → 32768"]


def test_cli_serve_list(cfg):
    r = CliRunner().invoke(main, ["--json", "serve", "--list"])
    assert r.exit_code == 0, r.output
    out = __import__("json").loads(r.output)
    names = {m["name"] for m in out["models"]}
    assert names == set(cfg.models)
    mtp = next(m for m in out["models"] if m["name"] == "Qwen36_35B_A3B_UD_Q4_K_M_MTP")
    assert mtp["command"] == "mse-llm serve Qwen36_35B_A3B_UD_Q4_K_M_MTP"
    assert mtp["command_spec"].endswith("--spec")
    assert {p["name"] for p in out["presets"]} >= {"Tesslate_OmniCoder-9B-Q4_K_M"}
    r = CliRunner().invoke(main, ["serve"])                  # bare `serve` lists instead of failing
    assert r.exit_code == 0 and "mse-llm serve NAME" in r.output and "--preset NAME" in r.output
