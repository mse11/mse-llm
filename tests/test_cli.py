import json

from click.testing import CliRunner

from mse_llm import helpsnap
from mse_llm.cli import main


def test_top_level_help():
    r = CliRunner().invoke(main, ["--help"])
    assert r.exit_code == 0
    for cmd in ("doctor", "hw", "help"):
        assert cmd in r.output


def test_top_level_help_is_sectioned():
    """Every command shows up exactly once, under its area; nothing ends up in "Other"."""
    r = CliRunner().invoke(main, ["--help"])
    titles = [t for t, _ in main.SECTIONS]
    assert [line[:-1] for line in r.output.splitlines() if line[:-1] in titles + ["Other"]] == titles
    listed = [n for _, names in main.SECTIONS for n in names]
    assert sorted(listed) == sorted(n for n, c in main.commands.items() if not c.hidden)


def test_hw_json_with_fixture(hw_fixture):
    r = CliRunner().invoke(main, ["hw", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["identity"]["hash"] == "20725a44cc6f18bc"
    assert data["verdict"] == "can run comfortably"


def test_hw_fixture_option(hw_fixture, monkeypatch):
    monkeypatch.delenv("MSE_LLM_TEST_HARDWARE")
    r = CliRunner().invoke(main, ["--hw-fixture", str(hw_fixture), "--json", "hw", "identity"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["hash"] == "20725a44cc6f18bc"
    assert data["targets"][0]["status"] == "pass"


def test_hw_table_with_fixture(hw_fixture):
    r = CliRunner().invoke(main, ["hw"])
    assert r.exit_code == 0, r.output
    assert "RTX 4080 Laptop" in r.output and "AVX-512 + VNNI/VBMI/BF16" in r.output


def test_doctor_json_with_fixture(hw_fixture):
    r = CliRunner().invoke(main, ["doctor", "--json"])
    data = json.loads(r.output)
    assert r.exit_code == (1 if any(c["status"] == "fail" for c in data["checks"]) else 0)
    names = {c["name"] for c in data["checks"]}
    assert {"python", "CUDA sysmem fallback", "binaries", "RAM"} <= names


def test_doctor_flags_uses_stored_snapshot(tmp_path, monkeypatch):
    snap = tmp_path / "abc1234"
    snap.mkdir()
    src = (helpsnap.snapshots_dir().parent / "help_synthetic" / "new" / "llama-server.txt")
    (snap / "llama-server.txt").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (snap / "meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(helpsnap, "snapshots_dir", lambda: tmp_path)
    monkeypatch.setenv("MSE_LLM_BUILDS_ROOT", str(tmp_path / "nobuild"))   # no binaries
    r = CliRunner().invoke(main, ["doctor", "flags", "--json"])
    data = json.loads(r.output)
    assert data["snapshot"] == "abc1234" and data["binaries"] is False
    msgs = " | ".join(f["message"] for f in data["findings"])
    assert "turbo4" in msgs          # bash script's turbo cache types (§0.1)
    assert r.exit_code == 1


def test_doctor_groups_checks_by_section():
    from mse_llm.checks import Check
    from mse_llm.doctor import group_by_section

    cs = [Check("engine a", "binaries", "pass", ""), Check("engine b", "binaries", "pass", ""),
          Check("engine a", "build", "info", ""), Check("engine b", "build", "info", "")]
    got = [(c.section, c.name) for c in group_by_section(cs)]
    assert got == [("engine a", "binaries"), ("engine a", "build"), ("engine b", "binaries"), ("engine b", "build")]
