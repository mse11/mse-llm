"""Golden tests (M3): the argv mse-llm builds vs what llama_cpp_benchmark.sh builds.

tests/fixtures/bash_golden/golden.jsonl is produced from the real script by dump_golden.sh (see its
header to regenerate). The port must give the same argv, except for the deliberate differences in
`fix_*` below; they're listed in docs/PARITY.md ("M3: argv differences").
"""

import json
import ntpath
from pathlib import Path

import pytest

from mse_llm import variants as vv
from mse_llm.config import load_config
from mse_llm.runners import (
    bench_command,
    benchy_command,
    help_command,
    inspect_command,
    presets_serve_command,
    server_command,
)

ROOT = Path(__file__).parents[1]
GOLDEN = [json.loads(line) for line in
          (ROOT / "tests" / "fixtures" / "bash_golden" / "golden.jsonl").read_text(encoding="utf-8").splitlines()]


def kind(k: str) -> list[dict]:
    return [r for r in GOLDEN if r["kind"] == k]


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config" / "config.example.toml")


def _ids(recs):
    return [f"{r.get('fn', r['kind'])}:{r.get('model', '')}:{r.get('extra', r.get('key', ''))}" for r in recs]


# ---------------------------------------------------------------- deliberate differences (docs/PARITY.md)

def fix_variant_flags(tokens: list[str]) -> list[str]:
    """bash EXTRAS flags -> port: `-fa 1` -> `-fa on` (§0.2); a literal `--model-draft MODEL`
    (sdNmtp on a model without a DRAFTER_MODELS entry) is dropped, with a warning instead."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "-fa" and i + 1 < len(tokens) and tokens[i + 1] == "1":
            out += ["-fa", "on"]
            i += 2
        elif tokens[i] == "--model-draft" and i + 1 < len(tokens) and tokens[i + 1] == "MODEL":
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def fix_server(argv: list[str], fit: str) -> list[str]:
    """+ `-ngl 999` -> `-ngl all` (§0.2), explicit `--fit on|off` before --host (§0.3)."""
    out = fix_variant_flags(argv[1:])
    i = out.index("-ngl")
    assert out[i + 1] == "999"
    out[i + 1] = "all"
    h = out.index("--host")
    return out[:h] + ["--fit", fit] + out[h:]


def fix_bench(argv: list[str]) -> list[str]:
    """+ `-r 1` (bash used llama-bench's implicit 5) and `-o jsonl` (structured output for M7)."""
    return fix_variant_flags(argv[1:]) + ["-r", "1", "-o", "jsonl"]


def tool_name(path: str) -> str:
    return ntpath.basename(path).lower().removesuffix(".exe")


# ---------------------------------------------------------------- F3: EXTRAS table + placeholder replacement

@pytest.mark.parametrize("rec", kind("extra"), ids=_ids(kind("extra")))
def test_every_extras_key_decodes_to_the_same_flags(rec):
    key, flags = rec["key"], rec["flags"]
    concrete = key.replace("_moeN", "_moe7").replace("_sdScache", "_sd3cache").replace("_sdSmtp", "_sd3mtp")
    if "_Kt" in key or "_Vt" in key:
        with pytest.raises(vv.DroppedVariantError):
            vv.decode(concrete)
        return
    want = (flags.replace("--n-cpu-moe N", "--n-cpu-moe 7").replace("--spec-draft-n-max S", "--spec-draft-n-max 3")
            .replace("--model-draft MODEL", "--model-draft D.gguf")).split()
    got, warn = vv.decode(concrete).flags("server", draft="D.gguf")
    assert got == fix_variant_flags(want)
    assert not warn
    # the canonical name decodes to the same thing (codec round trip)
    assert vv.decode(vv.encode(vv.decode(concrete))) == vv.decode(concrete)


def test_extras_table_covers_every_bash_combination():
    keys = {r["key"] for r in kind("extra")}
    assert len(keys) == 1249        # 13 batch variants x 6 spec/moe suffixes x 16 bases + no_extra
    assert "no_extra" in keys and vv.decode("no_extra").is_empty


@pytest.mark.parametrize("rec", kind("replaced"), ids=_ids(kind("replaced")))
def test_placeholders_replaced_per_model(rec, cfg):
    draft = cfg.models[rec["model"]].draft if rec["model"] in cfg.models else None
    got, warn = vv.decode(rec["key"]).flags("server", draft=draft)
    assert got == fix_variant_flags(rec["flags"].split())
    literal_model = "--model-draft MODEL" in rec["flags"]
    assert bool(warn) == literal_model     # the bash bug becomes a warning


def test_f2_every_bash_model_is_configured(cfg):
    bash_models = {r["model"] for r in kind("replaced")} - {"DUMMY_MODEL"}
    assert bash_models == set(cfg.models)


# ---------------------------------------------------------------- command lines

@pytest.mark.parametrize("rec", kind("serve"), ids=_ids(kind("serve")))
def test_serve_argv_matches_bash(rec, cfg):
    cmd = server_command(cfg, rec["model"], rec["extra"], rec["ctx"])
    assert tool_name(cmd.argv[0]) == tool_name(rec["argv"][0]) == "llama-server"
    assert cmd.argv[1:] == fix_server(rec["argv"], "on")
    assert cmd.errors == []
    assert bool(cmd.warnings) == ("MODEL" in rec["argv"])


@pytest.mark.parametrize("rec", kind("bench"), ids=_ids(kind("bench")))
def test_bench_argv_matches_bash(rec, cfg):
    cmd = bench_command(cfg, rec["model"], rec["extra"], rec["ctx"])
    assert tool_name(cmd.argv[0]) == tool_name(rec["argv"][0]) == "llama-bench"
    if "--spec-type" in rec["argv"]:
        # bash passed --spec-* (and a literal --model-draft MODEL) to llama-bench, which has no such flags
        assert any("llama-bench can't run sd" in e for e in cmd.errors)
        return
    assert cmd.argv[1:] == fix_bench(rec["argv"])
    assert cmd.errors == [] and cmd.warnings == []


@pytest.mark.parametrize("rec", kind("benchy"), ids=_ids(kind("benchy")))
def test_benchy_argv_matches_bash(rec, cfg):
    cmd = benchy_command(cfg, rec["model"], rec["extra"], rec["ctx"])
    assert cmd.argv == rec["argv"]               # one pp/tg/depth value each: same as bash
    assert cmd.server.argv[1:] == fix_server(rec["server"], "off") + ["--metrics"]   # M10: /metrics (spec)
    assert cmd.ok


def test_benchy_lists_go_after_one_flag(cfg):
    cmd = benchy_command(cfg, "gemma_4_26B_A4b_QAT_UD_Q4_K_XL", "fa_Kq8_Vq8_b2048_ub2048_moe18", 65536,
                         pp=[512, 2048], tg=[64, 128], depth=[0, 4096])
    a = cmd.argv
    assert a[a.index("--pp"):] == ["--pp", "512", "2048", "--tg", "64", "128", "--depth", "0", "4096"]
    assert a.count("--pp") == 1 and not cmd.warnings


@pytest.mark.parametrize("rec", kind("inspect"), ids=_ids(kind("inspect")))
def test_inspect_argv_matches_bash(rec, cfg):
    cmd = inspect_command(cfg, rec["model"], rec["task"], rec["extra"], rec["ctx"])
    assert cmd.argv == rec["argv"]
    assert cmd.env == {"OPENAI_API_KEY": "dummy"}
    assert cmd.server.argv[1:] == fix_server(rec["server"], "off")


def test_presets_serve_argv_matches_bash(cfg):
    (rec,) = kind("presets_serve")
    cmd = presets_serve_command(cfg)
    assert tool_name(cmd.argv[0]) == "llama-server"
    norm = [ntpath.normcase(x) for x in cmd.argv[1:]]
    assert norm == [ntpath.normcase(x) for x in rec["argv"][1:]] + ["--metrics"]   # M10: /metrics
    assert cmd.errors == []


@pytest.mark.parametrize("tool", ["server", "bench"])
def test_help_argv_matches_bash(tool, cfg):
    (rec,) = kind(f"help_{tool}")
    cmd = help_command(cfg, tool)
    assert tool_name(cmd.argv[0]) == tool_name(rec["argv"][0]) and cmd.argv[1:] == rec["argv"][1:]


def test_every_bash_server_function_has_a_default_serve(cfg):
    """run_SERVER_* -> `serve MODEL` (default_serve) / `serve MODEL --spec` (default_serve_spec)."""
    for rec in kind("serve"):
        if rec["fn"].startswith("synthetic"):
            continue
        m = cfg.models[rec["model"]]
        assert vv.canonical(rec["extra"]) in {vv.canonical(x) for x in (m.default_serve, m.default_serve_spec) if x}
        assert m.ctx == rec["ctx"]
