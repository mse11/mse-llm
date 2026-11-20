"""M9: `predict` (§4.6) and `sweep search` (§9 item 6). Synthetic GGUF shapes, a fake llama-fit-params that
follows the two linear rules, and fake runs; no model or GPU."""

from __future__ import annotations

import asyncio
import json
import math
import re
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from mse_llm import fit as ff
from mse_llm import gguf
from mse_llm import predict as pr
from mse_llm import results as rs
from mse_llm import search as se
from mse_llm.cli import main
from mse_llm.config import Config, SweepCfg, load_config
from mse_llm.hw import HardwareInfo
from mse_llm.hw import bench_power_problem as real_power_problem  # conftest patches the module attribute
from mse_llm.util import project_root
from mse_llm.variants import decode

ROOT = project_root()
HW_FIX = ROOT / "tests" / "fixtures" / "hw" / "rtx4080-laptop.json"
GEMMA = "gemma_4_26B_A4b_QAT_UD_Q4_K_XL"
OMNI = "omnicoder_9b"
MIB = 2 ** 20


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(ROOT / "config" / "config.example.toml")


def moe_shape(path="m.gguf", L=30, layer=408.0, used=8, n_expert=128) -> pr.Shape:
    """Roughly gemma-4-26B QAT: 30 MoE layers of 408 MiB, 8 of 128 experts used."""
    return pr.Shape(path, "gemma4", dense_mib=1300, dense_gflop=2.4, moe_layers=L, expert_layer_mib=layer,
                    expert_layer_gflop=1.5, n_expert=n_expert, n_expert_used=used, n_head=16,
                    attn=[pr.AttnLayer(8, 256, 1024)] * 25 + [pr.AttnLayer(2, 512, 0)] * 5)


def dense_shape(path="d.gguf") -> pr.Shape:
    return pr.Shape(path, "qwen35", dense_mib=5100, dense_gflop=16, moe_layers=0, expert_layer_mib=0,
                    expert_layer_gflop=0, n_expert=None, n_expert_used=None, n_head=16,
                    attn=[pr.AttnLayer(4, 256, 0)] * 8)


def hardware() -> pr.Hardware:
    return pr.hardware_from(HardwareInfo.model_validate_json(HW_FIX.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- fake llama-fit-params (the two linear rules)

def fake_fit(L=30, layer=408, dense=1300, kv=1000):
    def run(argv):
        a = " ".join(argv)
        n = int(m.group(1)) if (m := re.search(r"--n-cpu-moe (\d+)", a)) else 0
        ub = int(m.group(1)) if (m := re.search(r"-ub (\d+)", a)) else 512
        n = min(n, L)
        comp = 500 if ub <= 512 else ub
        out = f"CUDA0 {dense + (L - n) * layer} {kv} {comp}\nHost {n * layer} 0 {ub // 8}\n"
        return subprocess.CompletedProcess(argv, 0, out, "")
    return run


def info_of(path):
    return gguf.ModelInfo(path, "gemma4", 13000, 30, 30, 408, 128, 8, 262144, 8, 256, 256, 0)


# ---------------------------------------------------------------- GGUF shape

def _t(name, shape, nbytes):
    return gguf.Tensor(name, shape, 0, 0, nbytes)


def test_shape_of_counts_dense_experts_attention_and_skips_nextn():
    meta = {"general.architecture": "x", "x.block_count": 3, "x.nextn_predict_layers": 1, "x.expert_count": 16,
            "x.expert_used_count": 2, "x.attention.head_count": 4, "x.attention.head_count_kv": [2, 1, 2],
            "x.attention.key_length": 64, "x.attention.key_length_swa": 32, "x.attention.sliding_window": 128,
            "x.attention.sliding_window_pattern": [True, False, True]}
    T = [_t("token_embd.weight", [64, 1000], 100 * MIB), _t("output.weight", [64, 1000], 50 * MIB),
         _t("blk.0.attn_k.weight", [64, 128], 1 * MIB), _t("blk.1.attn_k.weight", [64, 64], 1 * MIB),
         _t("blk.0.ffn_up_exps.weight", [64, 32, 16], 8 * MIB), _t("blk.1.ffn_up_exps.weight", [64, 32, 16], 8 * MIB),
         _t("blk.2.ffn_up_exps.weight", [64, 32, 16], 8 * MIB), _t("blk.2.attn_k.weight", [64, 64], 1 * MIB),
         _t("blk.0.attn_norm.weight", [64], 1)]
    s = pr.shape_of(gguf.Gguf("p", 3, meta, T, 0))
    assert s.dense_mib == pytest.approx(52 + 1 / MIB)     # output + 2 attn_k (+ norm); no token_embd, no nextn
    assert s.moe_layers == 3 and s.expert_layer_mib == pytest.approx(8)   # the nextn layer is not in the mean
    assert s.used_frac == pytest.approx(2 / 16)
    assert [(a.kv_heads, a.head_dim, a.window) for a in s.attn] == [(2, 32, 128), (1, 64, 0)]
    assert s.dense_gflop == pytest.approx(2 * (64 * 1000 + 64 * 128 + 64 * 64) / 1e9)


def test_tied_output_reads_token_embd():
    meta = {"general.architecture": "x", "x.block_count": 1}
    s = pr.shape_of(gguf.Gguf("p", 3, meta, [_t("token_embd.weight", [8, 8], 3 * MIB)], 0))
    assert s.dense_mib == pytest.approx(3)


def test_shape_round_trip():
    s = moe_shape()
    assert pr.Shape.from_dict(json.loads(json.dumps(s.as_dict()))) == s


# ---------------------------------------------------------------- hardware + roofline

def test_hardware_from_the_laptop_fixture():
    h = hardware()
    assert h.vram_gbs == pytest.approx(432 * pr.VRAM_EFF)
    assert h.pcie_gbs == pytest.approx(1.969 * 16 * pr.PCIE_EFF)
    assert h.ram_gbs == pytest.approx(5200 * 8 / 1000 * 2 * pr.RAM_EFF)
    assert h.ram_single_gbs == pytest.approx(h.ram_gbs / 2)        # 16 + 64 GB: asymmetric
    assert "asymmetric" in h.source["ram"]


def test_hardware_defaults_without_info():
    h = pr.hardware_from(None)
    assert (h.vram_gbs, h.ram_gbs, h.pcie_gbs) == (300.0, 60.0, 25.0) and "default" in h.source["vram"]


def test_tg_slower_with_experts_on_the_cpu_and_gain_tracks_n_used():
    p = pr.Params.spec(hardware())
    s = moe_shape()
    fast = 1 / pr.tg_time(p, s, pr.config_of(decode("fa_default_moe10"), s), 64)
    slow = 1 / pr.tg_time(p, s, pr.config_of(decode("fa_default_moe30"), s), 64)
    assert fast > slow * 1.5
    sparse = moe_shape(used=8, n_expert=256)
    assert pr.tg_gain_ms_per_gib(p, s) == pytest.approx(2 * pr.tg_gain_ms_per_gib(p, sparse))


def test_pp_upload_share_falls_with_ub_and_fewer_cpu_layers():
    p = pr.Params.spec(hardware())
    s = moe_shape()
    up = lambda name: pr.pp_split(p, s, pr.config_of(decode(name), s), 65536)["upload"]   # noqa: E731
    assert up("fa_default_moe30") > 0.5 > up("fa_b4096_ub4096_moe30")
    assert up("fa_default_moe5") < up("fa_default_moe30")
    assert up("fa_default") == 0.0
    c = pr.config_of(decode("fa_b4096_ub8192_moe30"), s)
    assert c.ub == 4096                                    # ub is capped by b


def test_kv_and_attention_respect_the_sliding_window():
    s = moe_shape()
    assert s.kv_mib_per_token("q8_0", "q8_0", 100_000) < 2 * s.kv_mib_per_token("q8_0", "q8_0", 50_000)
    assert s.attn_gflop_per_token(0) == 0


# ---------------------------------------------------------------- calibration

def _samples(true: pr.Params, shapes: dict[str, pr.Shape], configs: list[tuple[str, str]]) -> list[pr.Sample]:
    out = []
    for i, (model, name) in enumerate(configs):
        s = shapes[model]
        c = pr.config_of(decode(name), s)
        out.append(pr.Sample(f"r{i}", model, name, "tg", s, c, 0, 64, 1 / pr.tg_time(true, s, c, 64)))
        out.append(pr.Sample(f"r{i}", model, name, "pp", s, c, 65536, 0, 1 / pr.pp_time(true, s, c, 65536)))
    return out


def test_calibration_recovers_the_real_speeds():
    base = pr.Params.spec(hardware())
    true = pr.Params(base.vram_gbs * 0.9, 34.0, 2.5, 45.0, 30.0, 6.0)
    shapes = {"moe": moe_shape(), "dense": dense_shape()}
    cfgs = [("moe", "fa_default_moe30"), ("moe", "fa_default_moe20"), ("moe", "fa_b2048_ub2048_moe25"),
            ("moe", "fa_b4096_ub4096_moe30"), ("dense", "fa_default"), ("dense", "fa_b1024_ub1024")]
    cal, rows = pr.check(base, _samples(true, shapes, cfgs))
    assert cal.calibrated and cal.params.ram_gbs == pytest.approx(34.0, rel=0.15)
    assert max(abs(r.err) for r in rows) < 0.05
    assert max(abs(r.loo_err) for r in rows) < 0.15                # predictions for configs left out
    assert max(abs(r.spec / r.measured - 1) for r in rows) > 0.2   # spec alone is off


def test_per_model_scales_learn_a_slow_cpu_path():
    """One model whose CPU runs its experts 5x slower than the others (measured: gemma QAT Q4_0 experts on
    build ffc87c3a7): the per-model fit moves the CPU term, not the VRAM term."""
    base = pr.Params.spec(hardware())
    good = pr.Params(base.vram_gbs, 34.0, 2.0, 60.0, 30.0, 5.0)
    slow = pr.Params(base.vram_gbs, 34.0 / 5, 2.0, 60.0, 30.0, 5.0)
    s = moe_shape()
    sm = _samples(good, {"q": s}, [("q", "fa_default_moe30"), ("q", "fa_default_moe20")])
    bad = _samples(slow, {"g": s}, [("g", "fa_default_moe30"), ("g", "fa_default_moe25"), ("g", "fa_default_moe20")])
    for x in bad:
        x.model, x.run_id = "g", "g" + x.run_id
    cal = pr.calibrate(base, sm + bad)
    assert cal.params_for("g").ram_gbs < cal.params_for("q").ram_gbs / 2
    c = pr.config_of(decode("fa_default_moe15"), s)
    assert cal.tg(s, c, 64, "g") == pytest.approx(1 / pr.tg_time(slow, s, c, 64), rel=0.15)
    ms = cal.model_scale["g"]
    assert ms["n_tg"] == 3 and ms["tg"][1] > 2


def test_uncalibrated_is_spec():
    base = pr.Params.spec(hardware())
    cal = pr.calibrate(base, [])
    assert not cal.calibrated and cal.params == base


def _run(model, variant, path, tests, status="ok", run_id="x"):
    return rs.RunResult(run_id=run_id, key=variant, timestamp="t", tool="bench", engine="e", model=model,
                        model_path=path, variant=variant, status=status, tests=tests, build={"id": "b"})


def test_samples_skip_speculative_failed_and_unreadable_runs():
    tg = rs.TestRow(kind="tg", n_gen=128, tg_tps=30.0)
    pp = rs.TestRow(kind="pp", n_prompt=65536, pp_tps=700.0)
    runs = [_run("m", "fa_default_moe30", "a.gguf", [pp, tg], run_id="ok"),
            _run("m", "fa_default_moe30_sd2mtp", "a.gguf", [pp, tg], run_id="spec"),
            _run("m", "fa_default_moe20", "a.gguf", [pp, tg], status="aborted", run_id="ab"),
            _run("m", "fa_default_moe25", "missing.gguf", [pp, tg], run_id="nofile")]
    sm, notes = pr.samples_from_runs(runs, lambda p: moe_shape() if p == "a.gguf" else None)
    assert {s.run_id for s in sm} == {"ok"} and {s.kind for s in sm} == {"tg", "pp"}
    assert any("speculative" in n for n in notes) and any("not readable" in n for n in notes)
    benchy = rs.TestRow(kind="pp+tg", n_prompt=2048, n_gen=32, depth=0, pp_tps=3000, tg_tps=50)
    sm, _ = pr.samples_from_runs([_run("m", "fa_default", "a.gguf", [benchy])], lambda p: moe_shape())
    assert [s.depth for s in sm if s.kind == "tg"] == [2048 + 16]  # TG after the prompt reads its KV


# ---------------------------------------------------------------- memory grid + predict

def predict(cfg, model=GEMMA, base="fa_Kq8_Vq8_default", free=11500, shape=None, **kw):
    base_p = pr.Params.spec(hardware())
    return pr.run_predict(cfg, model, decode(base), 65536, free_mib=free, free_source="test",
                          hardware=hardware(), calibration=kw.pop("calibration", pr.Calibration(base_p)),
                          runner=kw.pop("runner", fake_fit()), info_of=info_of,
                          shape_for=lambda p: shape or moe_shape(p), **kw)


def test_grid_measures_endpoints_and_interpolates(cfg):
    calls = []
    run = fake_fit()
    pred = predict(cfg, runner=lambda a: (calls.append(a), run(a))[1], ubs=[512, 2048])
    assert len(calls) == 4                                  # 2 ub x (moe30, moe0)
    src = {(p.ncmoe, p.ub): p.mem_source for p in pred.points}
    assert src[(30, 512)] == "measured" and src[(15, 512)] == "interp"
    p15 = next(p for p in pred.points if p.ncmoe == 15 and p.ub == 512)
    assert p15.gpu_mib == pytest.approx(1300 + 15 * 408 + 1000 + 500)   # exact: the fake is linear
    assert p15.host_mib == pytest.approx(15 * 408 + 64)
    assert p15.verdict == "fits~"


def test_exact_measures_every_point(cfg):
    calls = []
    run = fake_fit()
    predict(cfg, runner=lambda a: (calls.append(a), run(a))[1], ubs=[512], exact=True)
    assert len(calls) == len(pr.default_ncmoe_values(moe_shape()))


def test_verdicts_best_and_recommendations(cfg):
    pred = predict(cfg)
    by = {(p.ncmoe, p.ub): p for p in pred.points}
    assert by[(0, 4096)].verdict == "no-fit" and by[(30, 512)].verdict == "fits"
    best_tg = pred.best("tg")
    assert best_tg is not None and best_tg.fits
    assert all(p.tg <= best_tg.tg for p in pred.fitting())
    text = "\n".join(pred.recommendations)
    assert "lower -ncmoe first" in text and "batching works" in text and "asymmetric" in text
    assert pred.lowest_fitting_ncmoe(512) == 11           # 2800 + 19 x 408 = 10552: 948 left; at 10: 540 < 768
    d = pred.as_dict()
    assert d["best"]["tg"] == best_tg.variant.name and len(d["points"]) == len(pred.points)


def test_sparse_model_says_ncmoe_helps_little(cfg):
    pred = predict(cfg, shape=moe_shape(used=4, n_expert=512))
    assert "TG gains little" in "\n".join(pred.recommendations)


def test_dense_model_has_only_the_ub_axis(cfg):
    pred = predict(cfg, model=OMNI, shape=dense_shape(), runner=fake_fit(L=0, dense=5100))
    assert {p.ncmoe for p in pred.points} == {None}
    assert "dense model" in pred.recommendations[0]


def test_speculative_base_is_predicted_without_it(cfg):
    pred = predict(cfg, base="fa_Kq8_Vq8_default_moe21_sd4mtp")
    assert any("speculative decoding left out" in w for w in pred.warnings)
    assert all(p.variant.spec_type is None for p in pred.points)


def test_freeze_notes_mark_a_fitting_config(cfg):
    c = cfg.model_copy(deep=True)
    c.sweeps = {"x": SweepCfg(model=GEMMA, variants=[{"v": "fa_Kq8_Vq8_default_moe20", "status": "skip",
                                                      "note": "PC freeze", "tag": "freeze"}])}
    pred = predict(c)
    p = next(p for p in pred.points if p.ncmoe == 20 and p.ub == 512)
    assert p.fits and p.note.startswith("fits, but unsafe in your notes: PC freeze")


def test_parse_int_list():
    assert pr.parse_int_list("41,34, 28") == [41, 34, 28]
    assert pr.parse_int_list("16:22:3") == [16, 19, 22]
    assert pr.parse_int_list("30:26") == [30, 29, 28, 27, 26]
    assert pr.parse_int_list(None) is None


def test_with_axes_keeps_the_base_and_sets_b_to_ub():
    v = pr.with_axes(decode("fa_Kq8_Vq4_b2048_ub2048_moe16_sd4mtp"), 21, 4096)
    assert v.name == "fa_Kq8_Vq4_b4096_ub4096_moe21"
    assert pr.with_axes(decode("fa_Kq8_Vq4_b2048_ub2048_moe16"), 30, 512).name == "fa_Kq8_Vq4_default_moe30"


# ---------------------------------------------------------------- sweep search

def fake_runs(spill_at: set[str] = frozenset(), fail_at: set[str] = frozenset()):
    """TG / PP from a simple law: fewer CPU layers -> faster TG and PP; bigger ub -> faster PP."""
    ran: list[str] = []

    async def run(job):
        v = job.variant
        ran.append(v.name)
        n = v.n_cpu_moe if v.n_cpu_moe is not None else 0
        ub = v.ub or 512
        status = "aborted" if v.name in spill_at else "failed" if v.name in fail_at else "ok"
        tg = 100 / (1 + n / 10)
        pp = 1000 * (1 + math.log2(ub / 512)) / (1 + n / 30)
        return rs.RunResult(run_id=f"run-{v.name}", key="k", timestamp="t", tool="bench", engine="e", model=job.model,
                            variant=v.name, status=status,
                            reason="memory guard: spill" if status == "aborted" else None,
                            metrics={"tg_tps": tg, "pp_tps": pp} if status == "ok" else {})
    return run, ran


def do_search(pred, **kw):
    return asyncio.run(se.search(pred, model=GEMMA, **kw))


def test_search_never_starts_a_predicted_no_fit_and_keeps_the_best(cfg):
    pred = predict(cfg)
    run, ran = fake_runs()
    res = do_search(pred, run=run, axes=["ncmoe", "ub"])
    ncm = res.stages[0]
    assert ncm.values[0] == 30 and ncm.values[-1] == 10     # one past the lowest fit (11): the predicted edge
    assert ncm.steps[-1].status.startswith("predicted") and ncm.edge == ncm.steps[-1].variant
    assert all(pred.at(int(re.search(r"moe(\d+)", n).group(1)), 512).fits for n in ran)
    # best (moe11) is next to an edge that is only predicted: no step back (it passed the fit check with margin)
    assert not ncm.stepped_back and not ncm.edge_real and ncm.winner == ncm.steps[-2].variant
    assert res.winner and not res.any_spill
    ubst = res.stages[1]
    assert ubst.steps[0].reused                            # the stage starts from the winner already measured


def test_search_stops_at_the_first_spill(cfg):
    pred = predict(cfg)
    run, ran = fake_runs(spill_at={"fa_Kq8_Vq8_default_moe20"})
    res = do_search(pred, run=run, axes=["ncmoe"], steps=4)
    st = res.stages[0]
    assert st.edge == "fa_Kq8_Vq8_default_moe20" and "aborted" in st.edge_reason
    assert "fa_Kq8_Vq8_default_moe15" not in ran          # nothing past the spill
    assert st.edge_real and st.stepped_back and st.winner == "fa_Kq8_Vq8_default_moe30"   # back from moe25
    assert res.any_spill


def test_search_patience_and_max_runs(cfg):
    pred = predict(cfg)

    async def flat(job):                                   # nothing improves
        return rs.RunResult(run_id="r", key="k", timestamp="t", tool="bench", engine="e", model=job.model,
                            variant=job.variant.name, status="ok", metrics={"tg_tps": 10.0, "pp_tps": 100.0})
    res = do_search(pred, run=flat, axes=["ncmoe"], patience=2)
    assert len(res.stages[0].steps) == 3 and res.stages[0].stop.startswith("no gain")
    assert res.stages[0].winner == "fa_Kq8_Vq8_default_moe30"
    run, ran = fake_runs()
    res = do_search(pred, run=run, axes=["ncmoe"], max_runs=2)
    assert len(ran) == 2 and res.stages[0].stop == "max runs"


def test_search_reuses_stored_runs(cfg):
    pred = predict(cfg)
    run, ran = fake_runs()
    stored = {}

    async def remember(job):
        r = await run(job)
        stored[job.variant.name] = r
        return r
    do_search(pred, run=remember, axes=["ncmoe"])
    n = len(ran)
    res = do_search(pred, run=run, axes=["ncmoe"], reuse=lambda job: stored.get(job.variant.name))
    assert len(ran) == n and res.runs == 0 and res.reused == n


def test_axis_order_follows_predict(cfg):
    order, why = se.axis_order(predict(cfg), "tg", pr.DEFAULT_UBS)
    assert order[0] == "ncmoe" and "gain" in why             # ub doesn't change TG
    order, _ = se.axis_order(predict(cfg), "pp", pr.DEFAULT_UBS)
    assert order[0] == "ub"                                  # PCIe-bound PP: batching first
    pred = predict(cfg, model=OMNI, shape=dense_shape(), runner=fake_fit(L=0, dense=5100))
    assert se.axis_order(pred, "both", pr.DEFAULT_UBS)[0] == ["ub"]


def test_dry_run_simulation_and_storage(cfg, tmp_path):
    pred = predict(cfg)
    res = do_search(pred, run=se.simulated_runner(pred), simulated=True)
    assert res.simulated and res.winner and all(s.run_id in ("(predicted)", None) for st in res.stages
                                                for s in st.steps)
    res.build, res.hw, res.timestamp = "b1", "h1", "now"
    p = se.write_search(res, pred, root=tmp_path)
    se.write_search(res, pred, root=tmp_path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert p.name == "b1__h1.json" and doc["winner"] == res.winner and len(doc["history"]) == 2
    assert doc["stages"][0]["steps"][0]["variant"]


def test_set_default_serve_keeps_the_rest(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[models.a]\npath = "x"\ndefault_serve = "old"   # note\n\n[models.b]\npath = "y"\n',
                 encoding="utf-8")
    assert se.set_default_serve(p, "a", "fa_new") == "old"
    assert se.set_default_serve(p, "b", "fa_b") == ""
    import tomllib
    d = tomllib.loads(p.read_text(encoding="utf-8"))
    assert d["models"]["a"]["default_serve"] == "fa_new" and d["models"]["b"]["default_serve"] == "fa_b"
    assert d["models"]["a"]["path"] == "x"
    with pytest.raises(KeyError):
        se.set_default_serve(p, "zzz", "v")


# ---------------------------------------------------------------- CLI

@pytest.fixture
def fake_tools(monkeypatch, hw_fixture):
    monkeypatch.setattr(ff, "run_fit_params", fake_fit())
    monkeypatch.setattr(gguf, "info", info_of)
    monkeypatch.setattr(pr, "read_shape", lambda p: moe_shape(p))


def test_cli_predict_json_and_text(fake_tools):
    r = CliRunner().invoke(main, ["predict", GEMMA, "--free", "11500", "--ub", "512,2048", "--json"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["ctx"] == 65536 and d["best"]["both"] and not d["calibration"]["n_samples"]
    r = CliRunner().invoke(main, ["predict", GEMMA, "--free", "11500", "--ncmoe", "30,20,10"])
    assert r.exit_code == 0, r.output
    assert "fa_Kq8_Vq8_b2048_ub2048_moe16" in r.output or "moe20" in r.output
    assert "not calibrated" in r.output and "lower -ncmoe first" in r.output


def test_cli_sweep_search_dry_run(fake_tools):
    r = CliRunner().invoke(main, ["sweep", "search", GEMMA, "--free", "11500", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "stage ncmoe" in r.output and "winner" in r.output and "no run spilled" in r.output


def test_cli_results_calibrate_empty_store(hw_fixture):
    r = CliRunner().invoke(main, ["results", "calibrate"])
    assert r.exit_code == 0, r.output
    assert "no ok bench / benchy runs" in r.output


def test_cli_results_calibrate_with_runs(hw_fixture, monkeypatch):
    monkeypatch.setattr(pr, "read_shape", lambda p: moe_shape(p))
    s = moe_shape()
    true = pr.Params(300, 34, 2.5, 45, 30, 6)
    for i, name in enumerate(["fa_default_moe30", "fa_default_moe20", "fa_b2048_ub2048_moe25"]):
        c = pr.config_of(decode(name), s)
        tests = [rs.TestRow(kind="pp", n_prompt=65536, pp_tps=1 / pr.pp_time(true, s, c, 65536)),
                 rs.TestRow(kind="tg", n_gen=128, tg_tps=1 / pr.tg_time(true, s, c, 64))]
        r = _run(GEMMA, name, "m.gguf", tests, run_id=f"20261119-0000{i}0-bench-000{i}")
        r.key, r.timestamp = f"k{i}", "2026-11-19T00:00:00+01:00"
        rs.write_run(r)
    r = CliRunner().invoke(main, ["results", "calibrate", "--json"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["calibration"]["n_samples"] == 6 and len(d["rows"]) == 6
    r = CliRunner().invoke(main, ["results", "calibrate"])
    assert "leave-one-out" in r.output


def test_help_lists_predict():
    r = CliRunner().invoke(main, ["--help"])
    assert "predict" in r.output
    r = CliRunner().invoke(main, ["sweep", "--help"])
    assert "search" in r.output


def test_store_path_is_per_model_build_hw(tmp_path):
    p = se.search_path("a/b:c", "bid", "hh", root=tmp_path)
    assert p.parent.name == "a_b_c" and p.name == "bid__hh.json"
    assert isinstance(p, Path)


# ---------------------------------------------------------------- power plan

def test_low_power_plans_are_recognised():
    from mse_llm import hw

    assert hw.low_power_plan("Legion Quiet Mode") and hw.low_power_plan("Power saver")
    assert not hw.low_power_plan("Legion Performance Mode") and not hw.low_power_plan("Balanced")
    assert not hw.low_power_plan(None)


def test_calibration_uses_only_runs_of_the_same_power_plan():
    quiet = _run("m", "fa_default", "a.gguf", [], run_id="q")
    quiet.hw = {"power_plan": "Legion Quiet Mode"}
    old = _run("m", "fa_default", "a.gguf", [], run_id="o")          # before M9: no plan recorded
    assert not pr.same_plan(quiet, "Legion Performance Mode")
    assert pr.same_plan(quiet, "Legion Quiet Mode") and pr.same_plan(old, "Legion Performance Mode")
    assert pr.same_plan(quiet, None)


def test_bench_power_problem():
    from mse_llm.hw.models import PowerInfo

    assert real_power_problem(PowerInfo(ac_online=True, power_plan="Legion Performance Mode")) is None
    assert "Quiet" in real_power_problem(PowerInfo(ac_online=True, power_plan="Legion Quiet Mode"))
    assert "battery" in real_power_problem(PowerInfo(ac_online=False, battery_percent=80, power_plan="Balanced"))


def test_benchmarks_stop_under_a_quiet_plan(monkeypatch, fake_tools):
    from mse_llm import cli_bench, hw

    monkeypatch.setattr(hw, "bench_power_problem", lambda p=None: "power plan 'Legion Quiet Mode' caps CPU / GPU power")
    started = []
    monkeypatch.setattr(cli_bench, "_run_planned", lambda *a, **k: started.append(1))
    r = CliRunner().invoke(main, ["bench", "omnicoder_9b", "--ctx", "512"])
    assert r.exit_code == 1 and "Quiet Mode" in r.output and "--allow-low-power" in r.output and not started
    r = CliRunner().invoke(main, ["sweep", "search", GEMMA, "--free", "11500"])
    assert r.exit_code == 1 and "--allow-low-power" in r.output
    assert "sweep search" not in r.output                      # stopped before predict
    r = CliRunner().invoke(main, ["sweep", "search", GEMMA, "--free", "11500", "--dry-run"])
    assert r.exit_code == 0, r.output                          # dry runs never stop
    cli_bench.power_gate(True)                                 # --allow-low-power: warns, goes on
    monkeypatch.setenv(hw.ALLOW_LOW_POWER_ENV, "1")
    cli_bench.power_gate(False)
    monkeypatch.setenv(hw.ALLOW_LOW_POWER_ENV, "0")
    with pytest.raises(Exception, match="allow-low-power"):
        cli_bench.power_gate(False)
