import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from mse_llm import build as b
from mse_llm.cli import main
from mse_llm.hw import load_fixture

FIXTURE = Path(__file__).parent / "fixtures" / "hw" / "rtx4080-laptop.json"


@pytest.fixture
def broot(tmp_path, monkeypatch):
    """An empty builds root, so nothing outside tmp_path is read or written."""
    root = tmp_path / "builds"
    monkeypatch.setenv("MSE_LLM_BUILDS_ROOT", str(root))
    return root


def _cache(build_dir: Path, t: b.Target, **override) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# fake CMakeCache", "CMAKE_GENERATOR:INTERNAL=Visual Studio 17 2022"]
    for k, v in {**t.defs(), **override}.items():
        lines.append(f"{k}:STRING={v}")
    (build_dir / "CMakeCache.txt").write_text("\n".join(lines), encoding="utf-8")


def test_default_target_applies_default_variant(broot):
    t = b.resolve_target("rtx4080-laptop")
    assert t.applied == ["zen4-avx512x"]
    assert t.build_dir == broot / "llama.cpp" / "build"
    assert t.bin_dir == broot / "llama.cpp" / "build" / "bin" / "Release"
    assert t.engine_name == "cuda_flash"
    src = {f.key: f.source for f in t.flags}
    assert src["GGML_CUDA"] == "base"
    assert src["CMAKE_CUDA_ARCHITECTURES"] == "profile"
    assert src["GGML_AVX512_VNNI"] == "variant:zen4-avx512x"
    assert t.defs()["CMAKE_TOOLCHAIN_FILE"] == str(broot / "vcpkg" / "scripts" / "buildsystems" / "vcpkg.cmake")
    assert "GGML_AVX512" not in t.defs()   # must never be passed (FindSIMD overrides it)


def test_without_default_variant(broot):
    t = b.resolve_target("rtx4080-laptop", without=["zen4-avx512x"])
    assert t.applied == []
    assert t.build_dir.name == "build-without-zen4-avx512x"
    assert t.engine_name == "cuda_flash-without-zen4-avx512x"
    assert not any(k.startswith("GGML_AVX512_") for k in t.defs())


def test_variant_errors(broot):
    with pytest.raises(b.BuildError, match="unknown variant"):
        b.resolve_target("rtx4080-laptop+nope")
    with pytest.raises(b.BuildError, match="unknown variant"):
        b.resolve_target("rtx4080-laptop", without=["nope"])


def test_ref_keep_previous_dir(broot):
    t = b.resolve_target("rtx4080-laptop", ref="b12468", keep_previous=True)
    assert t.build_dir.name == "build-b12468"
    assert b.resolve_target("rtx4080-laptop", ref="b12468").build_dir.name == "build"


def test_fresh_plan_runs_everything(broot):
    t = b.resolve_target("rtx4080-laptop")
    steps = b.plan_steps(t)
    assert [s.id for s in steps] == list(b.STEP_IDS)
    assert all(s.status == "RUN" for s in steps)
    clone = steps[0].commands[0]
    assert clone[:3] == ["git", "clone", "--filter=blob:none"] and clone[-1] == str(t.src)
    install = steps[4].commands[0]
    assert install[1:] == ["install", "curl[ssl,sspi]", "openssl", "--triplet", "x64-windows"]
    conf = steps[5].commands[0]
    assert conf[:7] == ["cmake", "-S", str(t.src), "-B", str(t.build_dir), "-G", "Visual Studio 17 2022"]
    assert "-DGGML_CUDA_FA_ALL_QUANTS=ON" in conf and "-DGGML_AVX512_VNNI=ON" in conf
    assert steps[6].commands[0] == ["cmake", "--build", str(t.build_dir), "--config", "Release", "-j", "16"]
    assert not broot.exists()   # planning created nothing


def test_configure_skipped_when_cache_matches(broot):
    t = b.resolve_target("rtx4080-laptop")
    _cache(t.build_dir, t)
    conf = next(s for s in b.plan_steps(t) if s.id == "configure")
    assert conf.status == "SKIP"
    _cache(t.build_dir, t, GGML_AVX512_VNNI="OFF")
    conf = next(s for s in b.plan_steps(t) if s.id == "configure")
    assert conf.status == "RUN" and "GGML_AVX512_VNNI: OFF -> ON" in conf.why
    conf = next(s for s in b.plan_steps(t, b.RunOpts(clean=True)) if s.id == "configure")
    assert "--fresh" in conf.commands[0]


def test_only_and_from(broot):
    t = b.resolve_target("rtx4080-laptop")
    st = b.plan_steps(t, b.RunOpts(only="compile"))
    assert [s.id for s in st if s.status == "RUN"] == ["compile"]
    st = b.plan_steps(t, b.RunOpts(from_="smoke"))
    assert [s.id for s in st if s.status == "RUN"] == ["smoke", "verify", "manifest", "help-snapshot", "engine"]
    with pytest.raises(b.BuildError, match="unknown step"):
        b.plan_steps(t, b.RunOpts(only="bogus"))


def test_vcpkg_status_parsing(tmp_path):
    st = tmp_path / "installed" / "vcpkg"
    st.mkdir(parents=True)
    stanza = "Package: {p}\n{f}Architecture: x64-windows\nStatus: install ok installed\n"
    (st / "status").write_text("\n".join([
        stanza.format(p="curl", f=""), stanza.format(p="curl", f="Feature: ssl\n"),
        stanza.format(p="openssl", f=""),
    ]), encoding="utf-8")
    assert b.vcpkg_missing(tmp_path, "x64-windows", ["curl[ssl,sspi]", "openssl"]) == ["curl[ssl,sspi]"]
    assert b.vcpkg_missing(tmp_path, "x64-windows", ["curl[ssl]", "openssl"]) == []
    assert b.vcpkg_missing(tmp_path, "arm64-windows", ["openssl"]) == ["openssl"]


@pytest.mark.parametrize("archs,cc,ok", [
    ("89-real", "8.9", True), ("86-real", "8.9", False), ("86", "8.9", True),
    ("75-real;86-real", "8.6", True), ("native", "12.0", True), ("90-real", "8.9", False),
])
def test_cuda_arch_covers(archs, cc, ok):
    assert b.cuda_arch_covers(archs, cc) is ok


def test_same_value():
    assert b.same_value("ON", "TRUE") and b.same_value("OFF", "") and not b.same_value("ON", "OFF")
    assert b.same_value(r"D:\a\b.cmake", "D:/a/b.cmake")
    assert not b.same_value("89-real", "86-real")


def test_verify_build(broot):
    t = b.resolve_target("rtx4080-laptop")
    _cache(t.build_dir, t)
    proj = t.build_dir / "ggml" / "src"
    proj.mkdir(parents=True)
    (proj / "ggml-cpu.vcxproj").write_text(
        "<ItemDefinitionGroup Condition=\"'$(Configuration)|$(Platform)'=='Release|x64'\"><ClCompile>"
        "<EnableEnhancedInstructionSet>AdvancedVectorExtensions512</EnableEnhancedInstructionSet>"
        "<PreprocessorDefinitions>%(PreprocessorDefinitions);GGML_AVX512;__AVX512VNNI__;GGML_AVX512_VNNI;__AVX512VBMI__"
        "</PreprocessorDefinitions></ClCompile></ItemDefinitionGroup>", encoding="utf-8")
    info = load_fixture(str(FIXTURE))
    smoke = {"x": "CUDA0: NVIDIA GeForce RTX 4080 Laptop GPU (12281 MiB)"}
    res = {v["name"]: v["ok"] for v in b.verify_build(t, info, smoke)}
    assert res["CMakeCache.txt"] and res["/arch:AVX512"] and res["GGML_AVX512_VNNI"] and res["CUDA0"]
    assert res["GGML_AVX512_VBMI"]               # MSVC: only __AVX512VBMI__ is defined
    assert "GGML_AVX512_BF16" not in res         # not requested any more


def test_engine_from_comparison_manifest(broot):
    from mse_llm.config import load_config

    t = b.resolve_target("rtx4080-laptop", without=["zen4-avx512x"])
    t.bin_dir.mkdir(parents=True)
    t.manifest_path.write_text(json.dumps({"engine": t.engine_name, "target": t.label}), encoding="utf-8")
    engines = load_config().engines
    assert engines["cuda_flash-without-zen4-avx512x"].bin_path() == t.bin_dir
    assert engines["cuda_flash"].bin_path() == b.resolve_target("rtx4080-laptop").bin_dir


def test_cli_plan_json_and_dry_run(broot, hw_fixture):
    r = CliRunner().invoke(main, ["build", "plan", "rtx4080-laptop", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["variants"] == ["zen4-avx512x"] and len(data["steps"]) == 12
    r = CliRunner().invoke(main, ["build", "run", "--without", "zen4-avx512x", "--dry-run", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["build_dir"].endswith("build-without-zen4-avx512x")
    assert not broot.exists()


def test_cli_run_blocks_without_yes(broot, hw_fixture):
    r = CliRunner().invoke(main, ["build", "run", "rtx4080-laptop"], input="n\n")
    assert r.exit_code == 1 and "Proceed?" in r.output
    assert not broot.exists()


def test_cli_explain_and_show(broot, hw_fixture):
    r = CliRunner().invoke(main, ["build", "explain", "--", "-DGGML_AVX512_VNNI=ON"])
    assert r.exit_code == 0 and "VPDPBUSD" in r.output
    r = CliRunner().invoke(main, ["build", "show", "rtx4080-laptop", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["auto"][0]["key"] == "GGML_AVX512"


def test_dropped_flag_is_unset(broot):
    t = b.resolve_target("rtx4080-laptop")
    _cache(t.build_dir, t, GGML_OLD_FLAG="ON")
    (t.build_dir / b.DEFS_FILE).write_text(json.dumps({**t.defs(), "GGML_OLD_FLAG": "ON"}), encoding="utf-8")
    conf = next(s for s in b.plan_steps(t) if s.id == "configure")
    assert conf.status == "RUN" and "GGML_OLD_FLAG" in conf.why
    assert "-UGGML_OLD_FLAG" in conf.commands[0]


def _fake_tree(broot: Path) -> None:
    src = broot / "llama.cpp"
    (src / ".git").mkdir(parents=True)
    for d in ("build", "build-without-zen4-avx512x"):
        (src / d / "bin" / "Release").mkdir(parents=True)
        (src / d / "CMakeCache.txt").write_text("x" * 100, encoding="utf-8")
    (src / "README.md").write_text("src", encoding="utf-8")
    (broot / "vcpkg").mkdir()
    (broot / "logs").mkdir()
    (broot / "logs" / "a.log").write_text("log", encoding="utf-8")


def test_clean_builds_keeps_sources(broot):
    _fake_tree(broot)
    ro = broot / "llama.cpp" / "build" / "ro.txt"
    ro.write_text("read-only", encoding="utf-8")
    ro.chmod(0o444)
    r = CliRunner().invoke(main, ["build", "clean", "--builds", "--dry-run"])
    assert r.exit_code == 0 and "total 209 B" in r.output   # 2 x 100 B caches + 9 B file
    assert (broot / "llama.cpp" / "build").exists()
    r = CliRunner().invoke(main, ["build", "clean", "--builds"], input="y\n")
    assert r.exit_code == 0, r.output
    assert not list((broot / "llama.cpp").glob("build*"))
    assert (broot / "llama.cpp" / "README.md").exists() and (broot / "logs" / "a.log").exists()


def test_clean_logs_and_all(broot):
    _fake_tree(broot)
    r = CliRunner().invoke(main, ["build", "clean", "--logs", "-y"])
    assert r.exit_code == 0 and not (broot / "logs").exists() and (broot / "vcpkg").exists()
    r = CliRunner().invoke(main, ["build", "clean", "--all"], input="n\n")
    assert r.exit_code == 1 and broot.exists()
    r = CliRunner().invoke(main, ["build", "clean", "--all", "-y"])
    assert r.exit_code == 0 and not broot.exists()


def test_clean_needs_a_choice(broot):
    r = CliRunner().invoke(main, ["build", "clean"])
    assert r.exit_code == 2 and "--logs, --builds, --all or --dir NAME" in r.output


def test_remove_tree_refuses_outside_root(broot, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(b.BuildError, match="not inside a builds root"):
        b.remove_tree(other, [broot])


def test_verify_without_expects_flags_absent(broot):
    t = b.resolve_target("rtx4080-laptop", without=["zen4-avx512x"])
    _cache(t.build_dir, t)
    proj = t.build_dir / "ggml" / "src"
    proj.mkdir(parents=True)
    (proj / "ggml-cpu.vcxproj").write_text(
        "<ClCompile><EnableEnhancedInstructionSet>AdvancedVectorExtensions512</EnableEnhancedInstructionSet>"
        "<PreprocessorDefinitions>GGML_AVX512;__AVX512VNNI__</PreprocessorDefinitions></ClCompile>", encoding="utf-8")
    res = {v["name"]: v["ok"] for v in b.verify_build(t, load_fixture(str(FIXTURE)))}
    assert res["GGML_AVX512_VNNI"] is False      # left out, but compiled in
    assert res["GGML_AVX512_VBMI"] is True       # left out and absent


def test_steps_note_only_for_limited_runs():
    from mse_llm.cli import _steps_note

    assert _steps_note({"resumed_from": None, "only_step": None}) == ""
    assert _steps_note({}) == ""                                   # manifests from before this field
    assert _steps_note({"resumed_from": "verify"}) == ", resumed from step verify"
    assert _steps_note({"only_step": "compile"}) == ", only step compile"


def test_clean_one_dir(broot):
    _fake_tree(broot)
    src = broot / "llama.cpp"
    r = CliRunner().invoke(main, ["build", "clean", "--dir", "build-without-zen4-avx512x", "-y"])
    assert r.exit_code == 0, r.output
    assert not (src / "build-without-zen4-avx512x").exists()
    assert (src / "build").exists() and (src / "README.md").exists() and (broot / "logs").exists()
    r = CliRunner().invoke(main, ["build", "clean", "--dir", str(src / "build"), "-y"])   # by path
    assert r.exit_code == 0, r.output
    assert not (src / "build").exists()


def test_clean_dir_refuses_non_build_dirs(broot):
    _fake_tree(broot)
    for bad in ("llama.cpp", "vcpkg", "logs", "build-nope"):
        r = CliRunner().invoke(main, ["build", "clean", "--dir", bad, "-y"])
        assert r.exit_code == 1 and "not a build dir" in r.output, bad
    assert (broot / "llama.cpp" / "build").exists() and (broot / "vcpkg").exists()
    r = CliRunner().invoke(main, ["build", "clean", "--builds", "--dir", "build"])
    assert r.exit_code == 2
