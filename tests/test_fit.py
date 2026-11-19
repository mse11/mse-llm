"""M4: `mse-llm fit` (§4.6). llama-fit-params output and GGUF header info are replayed from
tests/fixtures/fit/recorded.json (recorded on the RTX 4080 laptop by record_fit.py), so no model or
GPU is needed."""

import json
import struct
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from mse_llm import fit as ff
from mse_llm import gguf
from mse_llm import variants as vv
from mse_llm.cli import main
from mse_llm.config import load_config

ROOT = Path(__file__).parents[1]
FIX = ROOT / "tests" / "fixtures" / "fit"
REC = json.loads((FIX / "recorded.json").read_text(encoding="utf-8"))
QWEN28_LIST = (FIX / "qwen28_extras_list.txt").read_text(encoding="utf-8")
Q28 = "Qwen36_28B_REAP20_A3B_Q4_K_M"
GEMMA = "gemma_4_26B_A4b_QAT_UD_Q4_K_XL"
MMPROJ = r"d:\.lmstudio\models\unsloth\gemma-4-26B-A4B-it-qat-GGUF\mmproj-F32.gguf"


def replay(argv):
    out = REC["runs"].get(" ".join(argv[1:]))
    if out is None:
        return subprocess.CompletedProcess(argv, 1, "", "error: not recorded: " + " ".join(argv[1:]))
    return subprocess.CompletedProcess(argv, 0, out, "")


def info_of(path):
    if path not in REC["infos"]:
        raise FileNotFoundError(path)
    return gguf.ModelInfo.from_dict(REC["infos"][path])


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config" / "config.example.toml")


def inputs(*names, notes=None):
    return [ff.FitInput(vv.decode(n), (notes or {}).get(n, "")) for n in names]


def fit(cfg, model, ins, ctxs=(65536,), free=11500, **kw):
    return ff.run_fit(cfg, model, ins, list(ctxs), free_mib=free, free_source="test", runner=replay,
                      info_of=info_of, **kw)


# ---------------------------------------------------------------- the §4.6 table (acceptance)

# plan §4.6 "Real output for your pasted list", free VRAM 11500 MiB
PLAN_TABLE = {
    "fa_Kq8_Vq8_default_moe41": (2568, "1286+745+537", 15247, 8932),
    "fa_Kq8_Vq4_default_moe41": (2408, "1286+585+537", 15247, 9092),
    "fa_Kq8_Vq8_b6144_ub6144_moe41": (8091, "1286+745+6060", 16127, 3409),
    "fa_Kq8_Vq4_b6144_ub6144_moe41": (7931, "1286+585+6060", 16127, 3569),
    "fa_Kq8_Vq8_b4096_ub4096_moe41": (6071, "1286+745+4040", 15807, 5429),
    "fa_Kq8_Vq4_b4096_ub4096_moe41": (5911, "1286+585+4040", 15807, 5589),
    "fa_Kq8_Vq8_b4096_ub4096_moe41_sd3cache": (6071, "1286+745+4040", 15807, 5429),
    "fa_Kq8_Vq4_b4096_ub4096_moe41_sd3cache": (5911, "1286+585+4040", 15807, 5589),
    "fa_Kq8_Vq8_b4096_ub4096_moe41_sd2cache": (6071, "1286+745+4040", 15807, 5429),
    "fa_Kq8_Vq4_b4096_ub4096_moe41_sd2cache": (5911, "1286+585+4040", 15807, 5589),
}


def test_pasted_qwen28_list_reproduces_the_plan_table(cfg):
    entries = vv.parse_list(QWEN28_LIST)
    rep = fit(cfg, Q28, [ff.FitInput(vv.decode(e.name), e.note) for e in entries])
    assert [r.variant for r in rep.rows] == list(PLAN_TABLE)          # order kept, extra_ dropped
    for r in rep.rows:
        gpu, split, host, left = PLAN_TABLE[r.variant]
        mg = r.main_gpu
        assert (round(r.gpu_mib), f"{mg.model}+{mg.context}+{mg.compute}", round(r.host_mib), round(r.left_mib)) \
            == (gpu, split, host, left), r.variant
        assert r.verdict == "fits"
    notes = {r.variant: r.note for r in rep.rows}
    assert notes["fa_Kq8_Vq8_b4096_ub4096_moe41"] == "SWEETSPOT 1. step"
    ngram = [p for p in rep.rows[-1].parts if p.name.startswith("spec")]
    assert ngram and ngram[0].source == ff.NONE and ngram[0].gpu_mib == 0
    assert "moe41 ≥ the model's 40 MoE layers" in rep.warnings[0]


def test_commented_entries_are_skipped_unless_asked():
    assert "fa_Kq8_Vq8_b8192_ub8192_moe41" not in [e.name for e in vv.parse_list(QWEN28_LIST)]
    assert "extra_fa_Kq8_Vq8_b8192_ub8192_moe41" in [e.name for e in vv.parse_list(QWEN28_LIST, True)]


def test_suggest_proposes_the_lower_ncmoe(cfg):
    """§4.6: the sweet spot leaves 5.4 GiB; lowest safe -ncmoe ≈ 28 (12 expert layers back on the GPU),
    confirmed with real llama-fit-params calls at 28 (fits) and 27 (doesn't)."""
    rep = fit(cfg, Q28, inputs("fa_Kq8_Vq8_b4096_ub4096_moe41"), suggest=True)
    (r,) = rep.rows
    s = " | ".join(r.suggest)
    assert "lowest safe -ncmoe = 28 (checked: 854 MiB left; 12 expert layers back on the GPU" in s
    assert "fa_Kq8_Vq8_b4096_ub4096_moe{36,32,28}" in s          # brace pattern ready for -v
    assert "largest -ub ≈" in s


def test_suggest_checks_the_linear_rule(cfg):
    """Expert layers differ in size, so the linear step can be one layer off: with 102 MiB less free VRAM,
    moe28 is TIGHT (752 left) although the rule says it fits; the check moves to 29."""
    rep = fit(cfg, Q28, inputs("fa_Kq8_Vq8_b4096_ub4096_moe41", "fa_Kq8_Vq8_b4096_ub4096_moe28"), free=11398,
              suggest=True)
    a, b = rep.rows
    assert "lowest safe -ncmoe = 29 (checked: 1151 MiB left; 11 expert layers" in a.suggest[0]
    assert b.verdict == "TIGHT"
    assert b.suggest[0] == ("needs -ncmoe ≥ 29 to fit with the margin "
                            "(checked: 1151 MiB left; 1 expert layer more on the CPU)")


def test_suggest_falls_back_to_the_linear_estimate(cfg):
    def only_main(argv):                       # the extra check calls fail
        if argv[argv.index("--n-cpu-moe") + 1] == "41":
            return replay(argv)
        return subprocess.CompletedProcess(argv, 1, "", "error: boom")
    rep = ff.run_fit(cfg, Q28, inputs("fa_Kq8_Vq8_b4096_ub4096_moe41"), [65536], free_mib=11500,
                     runner=only_main, info_of=info_of, suggest=True)
    assert "lowest safe -ncmoe ≈ 28 (linear estimate; 12 expert layers" in rep.rows[0].suggest[0]


def test_several_contexts(cfg):
    rep = fit(cfg, Q28, inputs("fa_Kq8_Vq8_b4096_ub4096_moe41"), ctxs=(65536, 131072))
    a, b = rep.rows
    assert (a.ctx, b.ctx) == (65536, 131072)
    assert b.main_gpu.context == 1425 and b.gpu_mib > a.gpu_mib


# ---------------------------------------------------------------- drafter / mmproj parts

def test_sdmtp_without_a_drafter_is_unsure(cfg):
    rep = fit(cfg, "Qwen36_35B_A3B_UD_Q4_K_M", inputs("fa_Kq8_Vq8_default_moe34_sd2mtp"), suggest=True)
    (r,) = rep.rows
    assert r.verdict == "UNSURE"
    assert r.uncounted and r.uncounted[0].name == "drafter"
    assert "models.Qwen36_35B_A3B_UD_Q4_K_M.draft is not set" in r.footer()[0]
    assert r.suggest == ["no suggestion: a part is NOT COUNTED, so the VRAM left is unknown"]
    # the main model was still measured: --spec-* stripped for llama-fit-params
    assert "--spec-type" not in r.argv and r.main_gpu.total == 6057


def test_builtin_mtp_is_estimated(cfg):
    rep = fit(cfg, "Qwen36_35B_A3B_UD_Q4_K_M_MTP", inputs("fa_Kq8_Vq4_default_moe34_sd2mtp"))
    (r,) = rep.rows
    assert r.verdict == "fits~"
    parts = {p.name: p for p in r.parts}
    assert parts["draft weights"].gpu_mib == 0 and parts["draft weights"].source == ff.MEASURED
    est = parts["draft KV+compute"]
    assert est.source == ff.ESTIMATED
    # 1 NextN layer x 2 KV heads x (256+256) x 65536 x f16 = 128 MiB; 512 x 248320 x 4 B = 485 MiB
    assert round(est.gpu_mib) == 128 + 485
    assert round(r.gpu_mib) == 6057 - 160 + 613                     # main (Vq4) + draft


def test_separate_draft_gguf(cfg):
    rep = fit(cfg, GEMMA, inputs("fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp"))
    (r,) = rep.rows
    parts = {p.name: p for p in r.parts}
    assert r.main_gpu.total == 8178                                  # plan §4.6 (4998+1010+2170)
    assert parts["draft weights"].source == ff.MEASURED and 220 < parts["draft weights"].gpu_mib < 230
    assert "mtp-gemma-4-26B-A4B-it.gguf" in parts["draft weights"].detail
    # gemma4-assistant: all 4 layers share the main model's KV -> only the compute buffer
    assert "KV 0" in parts["draft KV+compute"].detail
    assert parts["mmproj"].source == ff.NOT_USED
    assert r.verdict.endswith("~")


def test_draft_override_and_missing_file(cfg):
    rep = fit(cfg, "Qwen36_35B_A3B_UD_Q4_K_M", inputs("fa_Kq8_Vq8_default_moe34_sd2mtp"), draft=r"d:\nope.gguf")
    assert rep.rows[0].verdict == "UNSURE" and "not found" in rep.rows[0].uncounted[0].detail


def test_mmproj_counted_as_estimated(cfg):
    rep = fit(cfg, GEMMA, inputs("fa_Kq8_Vq8_b2048_ub2048_moe16"), mmproj=MMPROJ)
    (r,) = rep.rows
    mm = {p.name: p for p in r.parts}["mmproj"]
    assert mm.source == ff.ESTIMATED and round(mm.gpu_mib) == 2185
    assert round(r.gpu_mib) == 10219 + 2185
    assert r.verdict == "NO-FIT~"


# ---------------------------------------------------------------- verdicts / suggest edge cases

def test_freeze_config_is_no_fit_with_a_hint(cfg):
    rep = fit(cfg, GEMMA, inputs("fa_Kq8_Vq8_b8192_ub8192_moe16"), suggest=True)
    (r,) = rep.rows
    assert r.main_gpu.total == 17374 and r.verdict == "NO-FIT"         # plan: "freeze TERRIBLE"
    assert any("-ub ≤" in s or "-ncmoe ≥" in s or "doesn't fit" in s for s in r.suggest)


def test_tight_and_margin(cfg):
    ins = inputs("fa_Kq8_Vq8_b2048_ub2048_moe16")                    # 10219 MiB
    assert fit(cfg, GEMMA, ins, free=10219 + 500).rows[0].verdict == "TIGHT"
    assert fit(cfg, GEMMA, ins, free=10219 + 500, margin_mib=256).rows[0].verdict == "fits"
    assert fit(cfg, GEMMA, ins, free=None).rows[0].verdict == "?"


def test_needs_higher_ncmoe(cfg):
    rep = fit(cfg, GEMMA, inputs("fa_Kq8_Vq8_b2048_ub2048_moe16"), free=9000, suggest=True)
    # 10219 - 9000 + 768 = 1987 MiB over -> 5 layers of 408 MiB
    assert "needs -ncmoe ≥ 21" in rep.rows[0].suggest[0]


def test_fit_params_error_is_reported(cfg):
    rep = fit(cfg, Q28, inputs("fa_Kq8_Vq8_b1024_ub1024_moe41"))       # not recorded -> runner fails
    assert rep.rows[0].verdict == "ERROR" and rep.any_error


def test_missing_binary_is_one_clear_error(cfg):
    def boom(argv):
        raise FileNotFoundError(argv[0])
    with pytest.raises(ff.FitError, match="llama-fit-params not found"):
        ff.run_fit(cfg, Q28, inputs("fa_Kq8_Vq8_default_moe41"), [65536], free_mib=1, runner=boom, info_of=info_of)


def test_unknown_model(cfg):
    with pytest.raises(ff.FitError, match="neither a configured model"):
        fit(cfg, "no_such_model", inputs("fa_Kq8_Vq8_default"))


# ---------------------------------------------------------------- argv / parsing

def test_fit_params_argv(cfg):
    m = cfg.models[GEMMA]
    argv = ff.fit_params_argv(cfg, None, m.path, vv.decode("fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp"), 65536)
    assert argv[1:] == ["-m", m.path, "-c", "65536", "-ngl", "all", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
                        "-b", "2048", "-ub", "2048", "--n-cpu-moe", "21", "-lm", "none",
                        "--fit", "off", "--fit-print", "on"]
    assert Path(argv[0]).stem == "llama-fit-params"


def test_parse_fit_print_and_errors():
    devs = ff.parse_fit_print("CUDA0 1286 745 4040 \nCUDA1 10 20 30\nHost 15167 0 640 \nnoise\n")
    assert [d.device for d in devs] == ["CUDA0", "CUDA1", "Host"]
    assert devs[2].is_host and devs[0].total == 6071
    assert ff._error_text("x\nerror: invalid argument: --spec-type\n", 1) == "error: invalid argument: --spec-type"
    assert "failed to open GGUF" in ff._error_text("0.00.023.647 E gguf_init_from_file: failed to open GGUF file", 127)


def test_apply_kv():
    v = ff.apply_kv(vv.decode("fa_default_moe20"), "q8_0/q4_0")
    assert (v.ctk, v.ctv) == ("q8_0", "q4_0")
    v = ff.apply_kv(vv.decode("fa_Kq4_Vq4_default"), "q8_0/q8_0")
    assert (v.ctk, v.ctv) == ("q4_0", "q4_0")                         # the variant's own types win


# ---------------------------------------------------------------- GGUF reader

def _gguf_bytes(meta: dict, tensors: list[tuple[str, int]], align: int = 32) -> bytes:
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def val(v):
        if isinstance(v, bool):
            return struct.pack("<I?", 7, v)
        if isinstance(v, int):
            return struct.pack("<II", 4, v)
        if isinstance(v, str):
            return struct.pack("<I", 8) + s(v)
        if isinstance(v, list):          # string arrays only
            return struct.pack("<IIQ", 9, 8, len(v)) + b"".join(s(x) for x in v)
        raise TypeError(v)

    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, v in meta.items():
        out += s(k) + val(v)
    off = 0
    for name, n in tensors:
        out += s(name) + struct.pack("<IQIQ", 1, n, 0, off)
        off += (n * 4 + align - 1) // align * align
    out += b"\0" * ((-len(out)) % align)
    return out + b"\0" * off


def test_gguf_reader(tmp_path):
    vocab = [f"t{i}" for i in range(1000)]
    meta = {"general.architecture": "testmoe", "testmoe.block_count": 2, "testmoe.expert_count": 16,
            "testmoe.expert_used_count": 2, "testmoe.attention.head_count": 4, "testmoe.attention.head_count_kv": 2,
            "testmoe.embedding_length": 64, "testmoe.nextn_predict_layers": 1, "tokenizer.ggml.tokens": vocab}
    tensors = [("token_embd.weight", 1000), ("blk.0.attn_k.weight", 10), ("blk.0.ffn_up_exps.weight", 256),
               ("blk.0.ffn_down_exps.weight", 256), ("blk.1.ffn_gate_exps.weight", 512)]
    p = tmp_path / "t.gguf"
    p.write_bytes(_gguf_bytes(meta, tensors))
    g = gguf.read(p)
    assert g.arch == "testmoe" and isinstance(g.metadata["tokenizer.ggml.tokens"], gguf.ArrayLen)
    assert [t.nbytes for t in g.tensors] == [4000, 64, 1024, 1024, 2048]   # f32, padded to 32
    i = gguf.model_info(g)
    assert (i.moe_layers, i.vocab, i.n_head_kv, i.key_length, i.nextn_layers) == (2, 1000, 2, 16, 1)
    assert i.expert_layer_mib == pytest.approx(2048 / 2**20)
    assert i.attn_layers == 1


def test_gguf_reader_rejects_other_files(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 20)
    with pytest.raises(gguf.GgufError):
        gguf.read(p)


# ---------------------------------------------------------------- CLI

@pytest.fixture
def cli(monkeypatch):
    monkeypatch.setattr(ff, "run_fit_params", replay)
    monkeypatch.setattr(gguf, "info", info_of)
    return CliRunner()


def test_cli_pasted_list_json(cli):
    r = cli.invoke(main, ["--config", str(ROOT / "config" / "config.example.toml"), "fit", Q28, "--ctx", "65536",
                          "--from", "-", "--free", "11500", "--json"], input=QWEN28_LIST)
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert [(x["variant"], x["gpu_mib"], x["left_mib"], x["verdict"]) for x in data["rows"]][:2] == [
        ("fa_Kq8_Vq8_default_moe41", 2568, 8932, "fits"), ("fa_Kq8_Vq4_default_moe41", 2408, 9092, "fits")]


def test_cli_table_and_strict(cli):
    args = ["--config", str(ROOT / "config" / "config.example.toml"), "fit", "Qwen36_35B_A3B_UD_Q4_K_M",
            "-v", "fa_Kq8_Vq8_default_moe34_sd2mtp", "--free", "11500"]
    r = cli.invoke(main, args)
    assert r.exit_code == 0, r.output
    assert "UNSURE" in r.output and "drafter NOT COUNTED" in r.output
    assert cli.invoke(main, args + ["--strict"]).exit_code == 1


def test_cli_defaults_to_default_serve(cli):
    r = cli.invoke(main, ["--config", str(ROOT / "config" / "config.example.toml"), "fit", GEMMA, "--free", "11500",
                          "--json"])
    assert r.exit_code == 0, r.output
    rows = json.loads(r.output)["rows"]
    assert [x["variant"] for x in rows] == ["fa_Kq8_Vq8_b2048_ub2048_moe16", "fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp"]
    assert [x["note"] for x in rows] == ["default_serve", "default_serve_spec"]


def test_cli_free_auto_uses_hw(cli, hw_fixture):
    r = cli.invoke(main, ["--config", str(ROOT / "config" / "config.example.toml"), "fit", Q28,
                          "-v", "fa_Kq8_Vq8_default_moe41", "--json"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["free_mib"] is not None and d["free_source"].startswith("hw fixture")
