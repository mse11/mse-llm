from pathlib import Path

from mse_llm import helpsnap as hs

SYN = Path(__file__).parent / "fixtures" / "help_synthetic"
ROOT = Path(__file__).parents[1]


def _help(which: str) -> hs.HelpInfo:
    return hs.parse_help((SYN / which / "llama-server.txt").read_text(encoding="utf-8"))


def test_parse_names_and_values():
    h = _help("new")
    assert "-ctk" in h and "--cache-type-k" in h
    assert h.flags["-ctk"] is h.flags["--cache-type-k"]
    assert "f8_e4m3" in h.flags["-ctk"].values and "q5_1" in h.flags["-ctk"].values
    assert h.flags["-fa"].values == ["on", "off", "auto"]
    assert h.flags["-lm"].values == ["mmap", "none", "dio"]
    assert "--n-gpu-layers" in h and "--gpu-layers" in h
    assert "--no-mmap" not in h
    assert h.flags["--spec-type"].values == ["draft-simple", "draft-mtp", "ngram-cache", "ngram-mod"]


def test_diff_old_new():
    d = hs.diff_help(_help("old"), _help("new"))
    assert any("--no-mmap" in x for x in d.removed)
    assert any("--draft" in x for x in d.removed)
    assert any("--load-mode" in x for x in d.added)
    assert any("--spec-type" in x for x in d.added)
    assert any(x.startswith("--cache-type-k:") and "f8_e4m3" in x for x in d.changed)


def test_version_id():
    assert hs.version_id("version: 12468 (a49db739f)\nbuilt with MSVC") == "a49db739f"
    assert hs.version_id("nothing") is None


def test_audit_script_finds_section0_issues(tmp_path):
    script = tmp_path / "bench.sh"
    script.write_text(
        'FA="-fa 1"\n'
        'EXTRAS_add_batch_variants "extra_fa_Kt4_Vt4"   "$FA -ctk turbo4 -ctv turbo4"\n'
        'EXTRAS_add_batch_variants "extra_fa_Kq8_Vq8"   "$FA -ctk q8_0  -ctv q8_0"\n'
        'CMD="${LLAMA_SERVER} -m x.gguf --no-mmap --draft 4 -ngl 999"\n'
        'if [[ -z "$x" && $a -eq 1 ]]; then echo; fi\n'
        'curl -s --max-time 2 http://localhost\n',
        encoding="utf-8",
    )
    found = hs.audit_script(script, {"llama-server": _help("new")})
    msgs = [f.message for f in found]
    assert any("-fa 1" in m for m in msgs)
    assert any("turbo4" in m and "-ctk" in m for m in msgs)
    assert any("turbo4" in m and "-ctv" in m for m in msgs)
    assert any(m.startswith("--no-mmap") for m in msgs)
    assert any(m.startswith("--draft") for m in msgs)
    assert not any("q8_0" in m.split(":")[0] for m in msgs)
    assert not any("-eq" in m or "--max-time" in m for m in msgs)


def test_audit_ini_old_layout(tmp_path):
    ini = tmp_path / "p.ini"
    ini.write_text("[global defaults]\nno-mmap = 1\nfa = 1\n\n[m]\nmodel = x.gguf\nctv = turbo4\n",
                   encoding="utf-8")
    found = hs.audit_ini(ini, _help("new"))
    kinds = [(f.kind, f.message) for f in found]
    assert any(k == "ini-section" and "[global defaults]" in m for k, m in kinds)
    assert any(k == "ini-key" and m.startswith("no-mmap") for k, m in kinds)
    assert any(k == "bad-value" and "turbo4" in m for k, m in kinds)
    assert any(k == "ini-section" and "no [*]" in m for k, m in kinds)


def test_audit_fixed_ini_is_clean():
    """The repo's fixed INI only uses options that the (synthetic) new help knows."""
    found = hs.audit_ini(ROOT / "llama_cpp_presets.ini", _help("new"))
    assert found == []


def test_snapshot_roundtrip(tmp_path):
    d = tmp_path / "abc1234"
    d.mkdir()
    (d / "llama-server.txt").write_text((SYN / "new" / "llama-server.txt").read_text(encoding="utf-8"),
                                        encoding="utf-8")
    (d / "meta.json").write_text('{"captured_at": "2026-11-18T00:00:00+00:00"}', encoding="utf-8")
    assert hs.list_snapshots(tmp_path) == [d]
    snap = hs.load_snapshot(d)
    assert "--spec-type" in snap["llama-server"]


def test_version_id_formats():
    assert hs.version_id("version: 12468 (a49db739f)\nbuilt with MSVC") == "a49db739f"
    assert hs.version_id("version: 1.0.0-dev (build 12515, commit ffc87c3a7)") == "ffc87c3a7"
