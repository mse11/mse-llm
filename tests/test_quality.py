"""M10 (§4.5): KLD (llama-perplexity) and speculative-decoding acceptance. Fake tools; no GPU."""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from mse_llm import quality as qq
from mse_llm import results as rs
from mse_llm.cli import main
from mse_llm.config import load_config
from mse_llm.parsers import spec as sp
from mse_llm.parsers.perplexity import parse_perplexity
from mse_llm.runners import benchy_command, perplexity_command
from mse_llm.runners.base import Command
from mse_llm.util import project_root
from mse_llm.variants import CapabilityError, Variant, VariantError, decode

FIX = Path(__file__).parent / "fixtures" / "quality"
FAKE = Path(__file__).parent / "fake_perplexity.py"
QWEN = "Qwen36_35B_A3B_UD_Q4_K_M_MTP"


@pytest.fixture(scope="module")
def cfg():
    return load_config(project_root() / "config" / "config.example.toml")


# ---------------------------------------------------------------- parsers

def test_parse_kld_summary():
    q = parse_perplexity((FIX / "kld_omnicoder.txt").read_text(encoding="utf-8"))
    assert q["chunks"] == 2 and q["n_ctx"] == 2048 and q["n_batch"] == 2048
    assert q["mean_kld"] == pytest.approx(0.000361) and q["mean_kld_err"] == pytest.approx(0.000040, abs=1e-6)
    assert q["ppl_q"] == pytest.approx(2.208339) and q["ppl_ratio"] == pytest.approx(1.000142)
    assert q["same_top_p"] == pytest.approx(0.99316, abs=1e-4) and q["kld_p99"] > q["median_kld"]
    assert q["rms_dp_pct"] > 0 and q["max_kld"] >= q["kld_p999"] and "errors" not in q


def test_parse_reference_run():
    q = parse_perplexity((FIX / "ref_omnicoder.txt").read_text(encoding="utf-8"))
    assert q["ppl"] == pytest.approx(2.2080) and q["ppl_err"] == pytest.approx(0.08565) and q["chunks"] == 2
    assert "mean_kld" not in q


def test_parse_errors_and_short_corpus():
    q = parse_perplexity(["main: error: failed to open x.kld", "perplexity: you need at least 16384 tokens to "
                          "evaluate perplexity with a context of 8192"])
    assert q["tokens_needed"] == 16384 and q["errors"]


METRICS = """# HELP llamacpp:prompt_tokens_total x
llamacpp:prompt_tokens_total 100
llamacpp:spec_decode_num_draft_tokens_total {d}
llamacpp:spec_decode_num_accepted_tokens_total {a}
llamacpp:spec_decode_num_drafts_total {s}
llamacpp:predicted_tokens_seconds 41.5
# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{{position="0"}} {p0}
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{{position="1"}} {p1}
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{{position="2"}} {p2}
"""


def test_metrics_delta_gives_acceptance_per_position():
    before = sp.parse_metrics(METRICS.format(d=30, a=20, s=10, p0=9, p1=7, p2=4))
    after = sp.parse_metrics(METRICS.format(d=330, a=241, s=110, p0=100, p1=83, p2=58))
    assert after["per_pos"] == [100, 83, 58] and after["predicted_tokens_seconds"] == 41.5
    s = sp.spec_from_counters(sp.delta(before, after))
    assert (s["drafted"], s["accepted"], s["verif_steps"]) == (300, 221, 100)
    assert s["acceptance_rate"] == pytest.approx(0.7367, abs=1e-4) and s["mean_accept_len"] == pytest.approx(3.21)
    assert s["accept_rate_per_pos"] == [0.91, 0.76, 0.54] and s["source"] == "metrics"
    assert sp.useful_draft_len([0.91, 0.76, 0.24]) == 2 and sp.useful_draft_len([0.9]) == 1
    assert sp.spec_from_counters(sp.delta(after, after)) is None      # nothing drafted in between


def test_server_log_fallback():
    lines = ["slot release: id 0 | task 3 | draft acceptance = 0.71234 (  412 accepted /   578 generated), "
             "mean len =  2.85", "noise",
             "draft acceptance = 0.50000 (  10 accepted /   20 generated), mean len =  2.00"]
    s = sp.parse_server_log(lines)
    assert (s["drafted"], s["accepted"], s["requests"]) == (598, 422, 2) and s["source"] == "server log"
    assert s["acceptance_rate"] == pytest.approx(422 / 598, abs=1e-4) and s["accept_rate_per_pos"] == []
    assert sp.parse_server_log(["nothing"]) is None


# ---------------------------------------------------------------- variants: draft side

def test_draft_tokens_round_trip_and_flags():
    n = "fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp_sdmin2_dKq8_dVq4_dmoe3"
    v = decode(n)
    assert v.name == n and (v.spec_n_min, v.draft_ctk, v.draft_ctv, v.draft_n_cpu_moe) == (2, "q8_0", "q4_0", 3)
    flags, _ = v.flags("server", draft="builtin")
    assert flags[-8:] == ["--spec-draft-n-min", "2", "-ctkd", "q8_0", "-ctvd", "q4_0", "-ncmoed", "3"]
    assert v.without_spec().name == "fa_Kq8_Vq8_b2048_ub2048_moe21"
    assert decode("fa_default_sd3mtp_dcmoe").draft_cpu_moe
    with pytest.raises(CapabilityError):
        v.flags("bench")
    with pytest.raises(VariantError, match="speculative type"):
        decode("fa_default_dKq8").flags("server")


def test_draft_kv_types_count_in_fit():
    from mse_llm import fit as ff
    from mse_llm.gguf import ModelInfo

    info = ModelInfo("m.gguf", "qwen35moe", 1.0, 40, 40, 100.0, 256, 8, 248320, 2, 256, 256, 1, attn_layers=10)
    f16 = ff.draft_parts("m", "builtin", decode("fa_default_sd3mtp"), 65536, info, lambda p: info)
    q8 = ff.draft_parts("m", "builtin", decode("fa_default_sd3mtp_dKq8_dVq8"), 65536, info, lambda p: info)
    assert q8[1].gpu_mib < f16[1].gpu_mib and "q8_0" in q8[1].detail


# ---------------------------------------------------------------- reference planning

@dataclass
class Chk:
    bad: bool
    verdict: str


def fit_where(ok):
    return lambda v: Chk(not ok(v), "fits" if ok(v) else "NO-FIT")


def test_reference_size_formula():
    # 8 chunks x 8192 x Qwen 248320 vocab: 15.2 GiB (only the second half of each chunk is stored)
    assert qq.ref_size_bytes(248320, 8192, 8) / 2**30 == pytest.approx(15.2, abs=0.05)
    assert qq.ref_size_bytes(262144, 8192, 4) / 2**30 == pytest.approx(8.0, abs=0.05)


def test_reference_fits_as_is():
    p = qq.plan_reference(decode("fa_Kq8_Vq4_b2048_ub2048_moe16"), fit_where(lambda v: True), 30)
    assert p.kv == ("f16", "f16") and p.name == "fa_Kf16_Vf16_b2048_ub2048_moe16" and not p.steps


def test_reference_falls_back_to_more_experts_then_nkvo_then_q8():
    base = decode("fa_Kq8_Vq8_default_moe10")
    p = qq.plan_reference(base, fit_where(lambda v: (v.n_cpu_moe or 0) >= 20), 30)
    assert p.variant.n_cpu_moe == 20 and p.kv == ("f16", "f16") and "more experts" in p.steps[-1]
    p = qq.plan_reference(base, fit_where(lambda v: "-nkvo" in v.extra), 30)
    assert p.variant.extra == ("-nkvo",) and "KV in system RAM" in p.steps[-1] and p.name.endswith(" -nkvo")
    p = qq.plan_reference(base, fit_where(lambda v: v.ctk == "q8_0"), 30)
    assert p.kv == ("q8_0", "q8_0") and p.fits and p.variant.cpu_moe and "q8_0/q8_0 reference" in p.steps[-1]
    p = qq.plan_reference(decode("fa_default"), fit_where(lambda v: False), 0)      # dense, nothing fits
    assert not p.fits and p.kv == ("q8_0", "q8_0")


def test_floor_changes_only_the_placement():
    ref = decode("fa_Kf16_Vf16_default_moe16")
    fv, why = qq.floor_variant(ref, 30)
    assert fv.n_cpu_moe == 23 and (fv.ctk, fv.ctv) == ("f16", "f16") and "-ncmoe 23" in why
    fv, _ = qq.floor_variant(replace(ref, n_cpu_moe=None, cpu_moe=True), 30)
    assert fv.n_cpu_moe == 23 and not fv.cpu_moe
    fv, why = qq.floor_variant(decode("fa_Kf16_Vf16_default"), 0)                     # dense
    assert fv.extra == ("-nkvo",) and "RAM" in why
    fv, _ = qq.floor_variant(replace(decode("fa_Kf16_Vf16_default"), extra=("-nkvo",)), 0)
    assert fv.extra == ()


def test_verdict_levels():
    assert qq.verdict(0.000003, 0.000002) == "no measurable loss"
    assert qq.verdict(0.0000015, 0.0) == "no measurable loss"            # floor printed as 0.000000
    assert qq.verdict(0.00036, 0.0, 0.00004) == "negligible loss"
    assert qq.verdict(0.0035, 0.0, 0.0002) == "small loss"
    assert qq.verdict(0.02, 0.0) == "noticeable loss" and qq.verdict(0.2, 0.0) == "large loss"
    assert qq.verdict(0.0005, 0.0001, 0.0002) == "no measurable loss"   # within 2 sigma of 2x floor
    assert qq.verdict(0.1, None) == "no floor yet" and qq.verdict(None, 0.1) == "-"


def test_tested_variant_uses_ref_placement_when_its_own_does_not_fit():
    ref = qq.RefMeta("r1", "m", 8192, 8, {"sha": "x"}, "b", ("f16", "f16"), "fa_Kf16_Vf16_default_moe30",
                     ["-nkvo"])
    v = decode("fa_Kq8_Vq4_b4096_ub4096_moe10_sd3mtp")
    tv, note = qq.tested_variant(v, ref, fit_where(lambda x: False))
    assert (tv.ctk, tv.ctv, tv.n_cpu_moe, tv.extra) == ("q8_0", "q4_0", 30, ("-nkvo",)) and "doesn't fit" in note
    tv, note = qq.tested_variant(v, ref, fit_where(lambda x: True))
    assert tv.name == "fa_Kq8_Vq4_b4096_ub4096_moe10" and note is None
    assert qq.from_vname(qq.vname(replace(tv, extra=("-nkvo",)))).extra == ("-nkvo",)


# ---------------------------------------------------------------- corpus, text server, commands

def test_code_corpus_is_deterministic(tmp_path):
    src = tmp_path / "llama.cpp"
    (src / "src").mkdir(parents=True)
    (src / "common").mkdir()
    (src / "src" / "b.cpp").write_text("int b() { return 2; }\r\n" * 50, encoding="utf-8")
    (src / "src" / "a.h").write_text("int a();\n" * 50, encoding="utf-8")
    (src / "common" / "c.cpp").write_text("// c\n" * 50, encoding="utf-8")
    c1 = qq.code_corpus(src, size=600)
    c2 = qq.code_corpus(src, size=600)
    assert c1.sha == c2.sha and c1.path.is_file() and c1.path.name == f"code-{c1.sha}.txt"
    text = c1.path.read_text(encoding="utf-8")
    assert text.startswith("// ---- common/c.cpp") and "\r" not in text and "src/b.cpp" not in text   # cut at size
    meta = json.loads(c1.path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["files"] == ["common/c.cpp", "src/a.h"]
    assert qq.code_corpus(src, size=10_000).sha != c1.sha
    with pytest.raises(ValueError, match="no llama.cpp sources"):
        qq.code_corpus(tmp_path / "nothing")


def test_text_server_serves_the_corpus(tmp_path):
    p = tmp_path / "code.txt"
    p.write_text("hello code", encoding="utf-8")
    srv = qq.TextServer(p)
    try:
        assert urllib.request.urlopen(srv.url, timeout=5).read() == b"hello code"    # noqa: S310
    finally:
        srv.close()


def test_perplexity_command(cfg):
    cmd = perplexity_command(cfg, "gemma_4_26B_A4b_QAT_UD_Q4_K_XL", "fa_Kq8_Vq4_b2048_ub2048_moe16_mc512_sd4mtp",
                             8192, corpus="c.txt", chunks=8, logits_file="r.kld", kld=True)
    a = cmd.argv
    assert a[a.index("-c") + 1] == "8192" and a[a.index("--chunks") + 1] == "8" and a[-3:] == [
        "--kl-divergence-base", "r.kld", "--kl-divergence"]
    assert "--spec-type" not in a and "--moe-cache-mib" not in a and "-ctv" in a and cmd.ok, cmd.errors
    ref = perplexity_command(cfg, "omnicoder_9b", "fa_Kf16_Vf16_default", 8192, corpus="c.txt", chunks=2,
                             logits_file="r.kld", kld=False)
    assert ref.argv[-2:] == ["--kl-divergence-base", "r.kld"] and ref.params["mode"] == "ref"


def test_benchy_code_prompts_and_metrics(cfg):
    cmd = benchy_command(cfg, QWEN, "fa_Kq8_Vq8_default_moe34_sd3mtp", 16384, pp=[2048], prompts="code",
                         book_url="http://127.0.0.1:5/c.txt")
    assert cmd.argv[cmd.argv.index("--book-url") + 1] == "http://127.0.0.1:5/c.txt"
    assert cmd.params["prompts"] == "code" and cmd.server.argv[-1] == "--metrics"
    prose = benchy_command(cfg, QWEN, "fa_Kq8_Vq8_default_moe34_sd3mtp", 16384, pp=[2048])
    assert "--book-url" not in prose.argv and "prompts" not in prose.params


def test_sweep_prompt_sets(cfg):
    from mse_llm import bench as bb

    jobs = bb.sweep_jobs(cfg, "gemma_4_26B_A4b_QAT_UD_Q4_K_XL_benchy", include_off=True,
                         match=["*moe21_sd4mtp"], prompt_sets=["prose", "code"])
    run = [j for j, a, _ in jobs if a == "run"]
    assert [j.params.get("prompts") for j in run] == [None, "code"]


def test_spec_record(cfg):
    from mse_llm import bench as bb

    job = bb.Job("benchy", QWEN, decode("fa_Kq8_Vq8_default_moe34_sd3mtp"), 16384, {"prompts": "code"})
    before = sp.parse_metrics(METRICS.format(d=0, a=0, s=0, p0=0, p1=0, p2=0))
    after = sp.parse_metrics(METRICS.format(d=300, a=221, s=100, p0=91, p1=76, p2=54))
    s = bb.spec_record(cfg, job, before, after, [])
    assert s["type"] == "draft-mtp" and s["n_max"] == 3 and s["draft_model"] == "builtin"
    assert s["prompts"] == "code" and s["sampling"] == {"temp": 0.3} and s["useful_len"] == 3
    s = bb.spec_record(cfg, job, None, None, ["draft acceptance = 0.5 ( 10 accepted /  20 generated), mean len = 2"])
    assert s["source"] == "server log" and s["drafted"] == 20
    assert bb.spec_record(cfg, job, None, None, [])["source"] == "none"


# ---------------------------------------------------------------- end to end (fake llama-perplexity)

MEANS = {("q8_0", "q8_0"): 0.00036, ("q8_0", "q4_0"): 0.0035, ("f16", "f16"): 0.000004}


@pytest.fixture
def fake_ppl(monkeypatch, tmp_path, hw_fixture):
    from mse_llm import gguf

    monkeypatch.setenv("MSE_LLM_MONITOR", "0")

    def command(self, cfg):
        argv = [sys.executable, str(FAKE), "--kl-divergence-base", str(self.logits_file)]
        if self.mode != "ref":
            argv += ["--kl-divergence", "--mean", str(MEANS.get((self.variant.ctk, self.variant.ctv), 0.01))]
        return Command("llama-perplexity", argv, model=self.model, variant=self.variant.name, ctx=self.ctx)

    monkeypatch.setattr(qq.QualityJob, "command", command)
    fits = {"n": 0}

    def default_fit(cfg, model, ctx, engine):
        def f(v):
            fits["n"] += 1
            return Chk(False, "fits")
        return f

    monkeypatch.setattr(qq, "default_fit", default_fit)
    info = type("I", (), {"vocab": 1000, "moe_layers": 30})()
    monkeypatch.setattr(gguf, "info", lambda p: info)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("some text\n" * 100, encoding="utf-8")
    return corpus


def test_cli_quality_kld_end_to_end(fake_ppl):
    r = CliRunner().invoke(main, ["--no-notify", "quality", "kld", "gemma_4_26B_A4b_QAT_UD_Q4_K_XL",
                                  "-v", "fa_Kq8_Vq{8,4}_b2048_ub2048_moe16",
                                  "-v", "fa_Kq8_Vq8_b2048_ub2048_moe16_sd4mtp",
                                  "--corpus", str(fake_ppl), "--yes", "--output", "none"])
    assert r.exit_code == 0, r.output
    assert "noise floor" in r.output and "negligible loss" in r.output and "small loss" in r.output
    refs = qq.list_refs("gemma_4_26B_A4b_QAT_UD_Q4_K_XL")
    assert len(refs) == 1 and refs[0].ready and refs[0].floor["mean_kld"] == pytest.approx(4e-6)
    assert refs[0].variant == "fa_Kf16_Vf16_b2048_ub2048_moe16" and refs[0].floor["variant"].endswith("moe23")
    idx = [e for e in rs.read_index() if e["tool"] == "quality"]
    assert [e["mode"] for e in idx] == ["ref", "floor", "kld", "kld"]      # the sd4mtp variant = same K/V run
    run = rs.find_run(idx[2]["run_id"])
    assert run.quality["verdict"] == "negligible loss" and run.quality["ref_id"] == refs[0].id
    # second call: reference and floor are reused, only the variants run
    r = CliRunner().invoke(main, ["--no-notify", "--json", "quality", "kld", "gemma_4_26B_A4b_QAT_UD_Q4_K_XL", "-v",
                                  "fa_Kq8_Vq4_b2048_ub2048_moe16", "--corpus", str(fake_ppl), "--output", "none"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["ref"]["id"] == refs[0].id and [x["verdict"] for x in d["rows"]] == ["small loss"]
    r = CliRunner().invoke(main, ["quality", "list", "gemma_4_26B_A4b_QAT_UD_Q4_K_XL"])
    assert r.exit_code == 0 and refs[0].id in r.output
    r = CliRunner().invoke(main, ["quality", "clean", "gemma_4_26B_A4b_QAT_UD_Q4_K_XL", "--yes"])
    assert r.exit_code == 0 and not qq.list_refs("gemma_4_26B_A4b_QAT_UD_Q4_K_XL")


def test_cli_quality_kld_dry_run_shows_size_and_never_runs(fake_ppl):
    r = CliRunner().invoke(main, ["quality", "kld", "omnicoder_9b", "-v", "fa_Kq8_Vq4_default", "--corpus",
                                  str(fake_ppl), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "new reference: f16/f16 KV, ctx 8192, 8 chunks" in r.output and "reference file:" in r.output
    assert "fake_perplexity" in r.output and not rs.read_index()


def test_cli_quality_kld_ref_needs_confirmation(fake_ppl):
    r = CliRunner().invoke(main, ["quality", "kld-ref", "omnicoder_9b", "--corpus", str(fake_ppl)], input="n\n")
    assert r.exit_code != 0 and not qq.list_refs("omnicoder_9b")


def test_failed_reference_leaves_no_file(fake_ppl, monkeypatch, cfg):
    from mse_llm.bench import RunOpts

    st = qq.setup(cfg, "omnicoder_9b", corpus=str(fake_ppl))
    orig = qq.QualityJob.command
    monkeypatch.setattr(qq.QualityJob, "command", lambda self, c: replace(orig(self, c), argv=[
        *orig(self, c).argv, "--exit", "2"]))
    meta, r = asyncio.run(qq.make_reference(cfg, st, RunOpts(monitor=False, write=True)))
    assert r.status == "failed" and "exit code 2" in r.reason and not meta.path().exists() and not meta.ready


# ---------------------------------------------------------------- report spec

def _spec_run(variant, tg, acc=None, prompts=None, rid="r"):
    r = rs.RunResult(run_id=rid, key=rid, timestamp="2026-11-19T10:00:00+01:00", tool="benchy", engine="e", model="m",
                     variant=variant, ctx=16384, build={"id": "b"}, metrics={"tg_tps": tg},
                     params={"prompts": prompts} if prompts else {})
    if acc is not None:
        r.spec = sp.spec_from_counters({"spec_decode_num_draft_tokens_total": 300,
                                        "spec_decode_num_accepted_tokens_total": 300 * acc,
                                        "spec_decode_num_drafts_total": 100, "per_pos": [90, 70, 40]})
        r.spec.update(type="draft-mtp", n_max=int(variant.split("sd")[1][0]))
    return r


def test_spec_rows_compare_with_the_no_spec_baseline():
    from mse_llm.report.spec import baseline_name, spec_rows

    base = _spec_run("fa_Kq8_Vq8_default_moe34", 40.0, rid="b0")
    fast = _spec_run("fa_Kq8_Vq8_default_moe34_sd3mtp", 50.0, 0.7, rid="s3")
    slow = _spec_run("fa_Kq8_Vq8_default_moe34_sd6mtp", 35.0, 0.4, rid="s6")
    code = _spec_run("fa_Kq8_Vq8_default_moe34_sd3mtp", 52.0, 0.7, "code", rid="c3")
    index = [x.index_line() for x in (base, fast, slow, code)]
    rows = spec_rows([fast, slow, code, base], index)
    assert baseline_name("fa_Kq8_Vq8_default_moe34_sd3mtp_dKq8") == "fa_Kq8_Vq8_default_moe34"
    assert [r["run_id"] for r in rows] == ["s3", "s6", "c3"]
    assert rows[0]["speedup"] == pytest.approx(1.25) and not rows[0]["slower_than_baseline"]
    assert rows[1]["slower_than_baseline"] and rows[1]["baseline_run"] == "b0"
    assert rows[2]["baseline_tg"] is None                  # no code-prompt baseline: not compared with prose
    assert rows[0]["accept_rate_per_pos"] == [0.9, 0.7, 0.4]


def test_spec_chart(tmp_path):
    from mse_llm.report.spec import chart, spec_rows

    runs = [_spec_run(f"fa_Kq8_Vq8_default_moe34_sd{n}mtp", 40 + n, 0.8 - n / 20, rid=f"s{n}") for n in (2, 3, 4)]
    p = chart(spec_rows(runs, []), tmp_path / "spec.png")
    assert p.is_file() and p.stat().st_size > 10_000


def test_index_line_carries_quality_and_spec():
    r = _spec_run("fa_Kq8_Vq8_default_moe34_sd3mtp", 50.0, 0.7, "code")
    e = r.index_line()
    assert e["acceptance_rate"] == pytest.approx(0.7) and e["prompts"] == "code"
    q = rs.RunResult(run_id="q", key="q", timestamp="t", tool="quality", engine="e", model="m", variant="v",
                     quality={"mode": "kld", "mean_kld": 0.001, "same_top_p": 0.99, "ref_id": "r1"})
    assert {k: q.index_line()[k] for k in ("mode", "mean_kld", "ref_id")} == {"mode": "kld", "mean_kld": 0.001,
                                                                              "ref_id": "r1"}


def test_kv_groups_for_sweep_with_kld():
    from mse_llm.cli_quality import kv_groups

    vs = [decode(n) for n in ("fa_Kq8_Vq8_default_moe34", "fa_Kq8_Vq8_default_moe30_sd3mtp", "fa_Kq8_Vq4_default",
                              "fa_default")]
    assert [v.name for v in kv_groups(vs)] == ["fa_Kq8_Vq8_default_moe34", "fa_Kq8_Vq4_default", "fa_default"]
    assert Variant().ctk is None
