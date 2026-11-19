import json

import pytest
from click.testing import CliRunner

from mse_llm import variants as vv
from mse_llm.cli import main
from mse_llm.flagcheck import check_argv, load_helps


def test_legacy_prefix_and_canonical_name():
    v = vv.decode("extra_fa_Kq8_Vq4_b4096_ub4096_moe41_sd3cache")
    assert (v.fa, v.ctk, v.ctv, v.b, v.ub, v.n_cpu_moe) == ("on", "q8_0", "q4_0", 4096, 4096, 41)
    assert (v.spec_type, v.spec_n_max) == ("ngram-cache", 3)
    assert v.name == "fa_Kq8_Vq4_b4096_ub4096_moe41_sd3cache"
    assert vv.canonical("fa_Kq8_Vq8_moe34") == "fa_Kq8_Vq8_default_moe34"   # `default` is implied
    assert vv.decode("moe34_Vq8_fa_Kq8") == vv.decode("fa_Kq8_Vq8_default_moe34")   # any token order


def test_new_tokens():
    v = vv.decode("fa_Kf8_Vq4_default_cmoe_mc6144_cs95_lmD_nohost_fitt512_sd4eagle3")
    assert v.ctk == "f8_e4m3" and v.cpu_moe and v.moe_cache_mib == 6144 and v.moe_cache_cpu_share == 95
    assert v.load_mode == "dio" and v.no_host and v.fit_target == 512 and v.spec_type == "draft-eagle3"
    flags, warn = v.flags("server", draft="d.gguf")
    assert flags == ["-fa", "on", "-ctk", "f8_e4m3", "-ctv", "q4_0", "-cmoe", "--moe-cache-mib", "6144",
                     "--moe-cache-cpu-share", "0.95", "--no-host", "-fitt", "512", "--spec-type", "draft-eagle3",
                     "--spec-draft-n-max", "4", "--model-draft", "d.gguf"]
    assert v.name == "fa_Kf8_Vq4_default_cmoe_mc6144_cs95_lmD_nohost_fitt512_sd4eagle3"
    for t in ("q41", "q5", "q51", "iq4nl", "f16", "bf16", "f32"):
        assert vv.decode(f"K{t}").ctk == vv.CACHE_TYPES[t]
    assert vv.decode("sd3ngrammod").spec_type == "ngram-mod"
    assert vv.decode("faoff").fa == "off"


@pytest.mark.parametrize("name,msg", [
    ("fa_Kt4_Vt4_default", "turbo4 cache types are dropped"),
    ("fa_Kq8_Vt3", "turbo3"),
])
def test_turbo_dropped(name, msg):
    with pytest.raises(vv.DroppedVariantError, match=msg):
        vv.decode(name)


@pytest.mark.parametrize("name,msg", [
    ("fa_Kq7", "unknown cache type"),
    ("fa_wat", "unknown token"),
    ("fa_moe3_moe4", "conflicts"),
    ("sd3foo", "unknown speculative type"),
    ("lmX", "unknown load mode"),
    ("cs150", "percentage"),
    ("fa__Kq8", "empty token"),
])
def test_decode_errors(name, msg):
    with pytest.raises(vv.VariantError, match=msg):
        vv.decode(name)


def test_draft_handling():
    v = vv.decode("fa_Kq8_Vq8_default_moe34_sd2mtp")
    assert "--model-draft" not in v.flags(draft="builtin")[0] and not v.flags(draft="builtin")[1]
    assert v.flags(draft="x.gguf")[0][-2:] == ["--model-draft", "x.gguf"]
    flags, warn = v.flags(draft=None)
    assert "--model-draft" not in flags and "no drafter configured" in warn[0]
    assert vv.decode("sd3cache").flags(draft=None) == (["--spec-type", "ngram-cache", "--spec-draft-n-max", "3"], [])


def test_bench_capabilities():
    with pytest.raises(vv.CapabilityError, match="benchy"):
        vv.decode("fa_moe21_sd4mtp").flags("bench")
    with pytest.raises(vv.CapabilityError, match="mc"):
        vv.decode("fa_cmoe_mc2048").flags("bench")
    assert vv.decode("fa_nohost").flags("bench")[0] == ["-fa", "on", "--no-host", "1"]


def test_expand_braces_and_ranges():
    assert vv.expand("fa_Kq8_Vq8_b2048_ub2048_moe21_sd{3,4,5}mtp") == [
        f"fa_Kq8_Vq8_b2048_ub2048_moe21_sd{n}mtp" for n in (3, 4, 5)]
    assert vv.expand("moe{18,21}_sd{2..3}mtp") == ["moe18_sd2mtp", "moe18_sd3mtp", "moe21_sd2mtp", "moe21_sd3mtp"]
    assert vv.expand("moe{30..40..5}") == ["moe30", "moe35", "moe40"]
    with pytest.raises(vv.VariantError):
        vv.expand("moe{1,{2,3}}")


PASTED = '''
    local -a extras_list=(
       "extra_fa_Kq8_Vq8_default_moe41"
       "extra_fa_Kq8_Vq4_default_moe41"
#        "extra_fa_Kq8_Vq4_default_moe34" # SWEETSPOT (CTX==65536)
#        "extra_fa_Kq8_Vq8_default_moe30" # PC freeze (CTX==65536)
    )
    # NOTES: Batching Works - Generic quantization
fa_Kq8_Vq8_moe{16,18}   fa_Kq8_Vq8_default
'''


def test_parse_pasted_bash_array():
    names = [e.name for e in vv.parse_list(PASTED)]
    assert names == ["extra_fa_Kq8_Vq8_default_moe41", "extra_fa_Kq8_Vq4_default_moe41",
                     "fa_Kq8_Vq8_moe16", "fa_Kq8_Vq8_moe18", "fa_Kq8_Vq8_default"]
    all_ = vv.parse_list(PASTED, include_disabled=True)
    off = [e for e in all_ if e.disabled]
    assert [(e.name, e.note) for e in off] == [("extra_fa_Kq8_Vq4_default_moe34", "SWEETSPOT (CTX==65536)"),
                                               ("extra_fa_Kq8_Vq8_default_moe30", "PC freeze (CTX==65536)")]


def test_matrix_expansion():
    vs = vv.expand_matrix({"fa": ["on"], "ctk": ["q8_0"], "ctv": ["q8_0", "q4_0"],
                           "b_ub": [[2048, 2048], [4096, 4096], "default"], "n_cpu_moe": [18, 21],
                           "exclude": ["*_b4096_ub4096_moe18"]})
    names = [v.name for v in vs]
    assert len(names) == 2 * 3 * 2 - 2
    assert "fa_Kq8_Vq4_default_moe21" in names and "fa_Kq8_Vq8_b4096_ub4096_moe18" not in names
    with pytest.raises(vv.VariantError, match="unknown axis"):
        vv.expand_matrix({"bogus": [1]})
    base = vv.expand_matrix({"base": "fa_Kq8_Vq8_b2048_ub2048", "n_cpu_moe": [16]})
    assert [v.name for v in base] == ["fa_Kq8_Vq8_b2048_ub2048_moe16"]


def test_resolve_dedupes():
    vs = vv.resolve(["fa_Kq8_Vq8_moe34", "extra_fa_Kq8_Vq8_default_moe34"], {"fa": ["on"], "ctk": ["q8_0"],
                                                                            "ctv": ["q8_0"], "n_cpu_moe": [34]})
    assert [v.name for v in vs] == ["fa_Kq8_Vq8_default_moe34"]


def test_from_flags_round_trip():
    v = vv.decode("fa_Kq8_Vq4_b4096_ub4096_moe41_mc4096_cs80_lmD_sd3cache")
    flags, _ = v.flags("server")
    back, rest = vv.from_flags(["-m", "x.gguf", "-c", "65536", *flags, "-lm", "dio", "--jinja"])
    assert back == v and rest == ["--jinja"]
    assert vv.from_flags(["-fa", "1", "-ncmoe", "16"])[0].name == "fa_default_moe16"


def test_extra_flags_make_an_undecodable_suffix():
    v = vv.Variant(fa="on", extra=("--swa-full",))
    assert v.name.startswith("fa_default_x") and v.flags()[0] == ["-fa", "on", "--swa-full"]


# ---------------------------------------------------------------- flag check

def test_flagcheck_against_snapshot():
    snap, helps = load_helps(None)
    assert snap is not None
    errs = check_argv(["x", "-ctk", "turbo4", "-fa", "1", "--no-mmap", "-lm", "bogus", "--spec-type",
                       "draft-mtp,ngram-cache", "-fitt", "512", "-c"], "llama-server", helps)
    assert errs == ["-ctk turbo4: turbo4 not in [f32, f16, bf16, f8_e4m3, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1]",
                    "-fa 1: 1 not in [on, off, auto]", "--no-mmap: not accepted by llama-server",
                    "-lm bogus: bogus not in [auto, none, mmap, mlock, mmap+mlock, dio]", "-c: needs a value (N)"]
    # llama-bench: comma lists and ranges, cache types checked with llama-server's list
    bench = ["x", "-ctk", "q8_0,f8_e4m3", "-ncmoe", "30-40+2", "-lm", "mmap+mlock"]
    assert check_argv(bench, "llama-bench", helps) == []
    assert check_argv(["x", "-ctv", "q8_0,turbo3"], "llama-bench", helps)[0].startswith("-ctv q8_0,turbo3: turbo3")
    assert check_argv(["x", "--spec-type", "draft-mtp"], "llama-bench", helps) == [
        "--spec-type: not accepted by llama-bench"]


# ---------------------------------------------------------------- CLI

def test_cli_variants_decode_encode_expand():
    r = CliRunner().invoke(main, ["--json", "variants", "decode", "extra_fa_Kq8_Vq4_default_moe{34,41}", "fa_Kt4"])
    assert r.exit_code == 1, r.output
    data = json.loads(r.output)
    assert [d.get("name") for d in data] == ["fa_Kq8_Vq4_default_moe34", "fa_Kq8_Vq4_default_moe41", None]
    assert "dropped" in data[2]["error"]
    r = CliRunner().invoke(main, ["variants", "encode", "--", "-fa", "on", "-ctk", "q8_0", "-b", "2048", "-ub",
                                  "2048", "--n-cpu-moe", "16", "-m", "x.gguf", "--jinja"])
    assert r.exit_code == 0, r.output
    assert r.output.splitlines()[0] == "fa_Kq8_b2048_ub2048_moe16"
    r = CliRunner().invoke(main, ["variants", "expand", "--from", "-"], input=PASTED)
    assert r.output.split() == ["fa_Kq8_Vq8_default_moe41", "fa_Kq8_Vq4_default_moe41", "fa_Kq8_Vq8_default_moe16",
                                "fa_Kq8_Vq8_default_moe18", "fa_Kq8_Vq8_default"]


def _cli(args, **kw):
    from mse_llm.util import project_root

    return CliRunner().invoke(main, ["--config", str(project_root() / "config" / "config.example.toml"), *args], **kw)


def test_cli_serve_dry_run_json():
    r = _cli(["--json", "serve", "gemma_4_26B_A4b_QAT_UD_Q4_K_XL", "--spec", "--dry-run"])
    assert r.exit_code == 0, r.output
    (cmd,) = json.loads(r.output)
    assert cmd["variant"] == "fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp"
    assert cmd["argv"][cmd["argv"].index("--fit") + 1] == "on" and cmd["errors"] == []


def test_cli_bench_needs_dry_run_and_rejects_spec():
    r = _cli(["bench", "omnicoder_9b", "-v", "fa_Kq8_Vq8"])
    assert r.exit_code != 0 and "M7" in r.output
    r = _cli(["bench", "gemma_4_26B_A4b_QAT_UD_Q4_K_XL", "-v", "fa_moe21_sd4mtp", "--dry-run"])
    assert r.exit_code == 1 and "llama-bench can't run sd" in r.output


def test_cli_bench_prompts_wrapper():
    r = _cli(["--json", "bench", "omnicoder_9b", "-v", "extra_fa_Kq8_Vq8_default", "--ctx", "8192", "--ctx", "32768",
              "--dry-run"])
    assert r.exit_code == 0, r.output
    assert [c["argv"][c["argv"].index("-p") + 1] for c in json.loads(r.output)] == ["8192", "32768"]


def test_cli_benchy_eval_presets_dry_run():
    r = _cli(["--json", "benchy", "Qwen36_28B_REAP20_A3B_Q4_K_M", "-v", "fa_Kq8_Vq8_b4096_ub4096_moe41", "--pp", "512",
              "--pp", "2048", "--dry-run"])
    assert r.exit_code == 0, r.output
    (c,) = json.loads(r.output)
    assert c["argv"][c["argv"].index("--pp"):][:3] == ["--pp", "512", "2048"] and "server" in c
    r = _cli(["--json", "eval", "gemma_4_26B_A4b_UD_Q4_K_M", "mmlu_pro", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)[0]["argv"][-1] == "http://localhost:8033/v1"
    r = _cli(["--json", "presets", "serve", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)[0]["argv"][1] == "--models-preset"
    r = _cli(["serve", "--all", "--dry-run"])
    assert r.exit_code == 0 and r.output.count("▶ serve") == 7


def test_cli_models():
    r = _cli(["--json", "models", "show", "qwen3.6-28b-reap20-a3b-q4_k_m"])   # by alias, any case
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["name"] == "Qwen36_28B_REAP20_A3B_Q4_K_M"
    r = _cli(["models", "list"])
    assert r.exit_code == 0 and "GLM_47_Flash_REAP_23B_A3B_Q4_K_M" in r.output
