"""`mse-llm predict` (§4.6, M9): which setting helps, before running anything.

Three inputs, as in the plan:

    GGUF        exact tensor bytes and parameter counts: dense part read per token, one expert layer,
                n_used / n_expert, attention layers (full / sliding window) for the KV and attention cost
    hardware    VRAM, RAM and PCIe bandwidth from `hw` (spec values times an efficiency), an effective
                GPU compute rate
    fit         llama-fit-params for a small grid of -ncmoe x -ub (endpoints measured, inside interpolated
                with the linear rule "one -ncmoe step = one expert layer")

A simple roofline turns them into TG and PP:

    TG  s/token = (dense + experts on the GPU + KV read) / BW_vram + experts on the CPU / BW_ram + t0
                  (only n_used / n_expert of an expert layer is read per token)
    PP  s/token = FLOPs / F + CPU expert bytes uploaded per ubatch / (ub x BW_pcie) + t_ub / ub
                  (with hundreds of tokens per ubatch almost every expert is hit)

Calibration: the six numbers (BW_vram, BW_ram, t0; F, BW_pcie, t_ub) are fitted to the stored runs of this
PC (results/, M7) as scale factors on the spec values, with a pull towards 1 so a factor the runs can't see
(no run had experts on the CPU, say) stays at its spec value. A per-model factor then takes out what the
simple model gets wrong for that model (shrunk towards 1 with few runs). Speculative-decoding runs are not
used: their TG is not the plain decode speed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from mse_llm import fit as ff
from mse_llm import gguf
from mse_llm.config import Config
from mse_llm.variants import Variant

MIB = 2 ** 20
GB = 1e9
DEFAULT_B = 2048                      # llama.cpp default -b
DEFAULT_UB = ff.DEFAULT_UB

# spec -> effective (what llama.cpp reaches in practice; calibration replaces them)
VRAM_EFF = 0.75                       # 432 GB/s GDDR6 -> ~320 GB/s
RAM_EFF = 0.72                        # DDR5-5200 dual channel 83 GB/s -> ~60 GB/s
PCIE_EFF = 0.80                       # Gen4 x16 31.5 GB/s -> ~25 GB/s
PCIE_LANE_GBS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.877}
DEFAULT_TFLOPS = 60.0                 # effective GPU compute for quantized matmuls (RTX 4080 Laptop class)
DEFAULT_T0_MS = 2.0                   # per-token fixed cost (kernel launches, sampling)
DEFAULT_TUB_MS = 5.0                  # per-ubatch fixed cost
FALLBACK = {"vram_gbs": 300.0, "ram_gbs": 60.0, "pcie_gbs": 25.0}

PRIOR_WEIGHT = 0.35                   # pull of each scale factor towards 1 (relative units)
SCALE_LIMITS = (0.25, 4.0)
TIGHT_DEFAULT = ff.DEFAULT_MARGIN
TG_DEPTH = 64                         # llama-bench tg128 starts at depth 0: mean depth 64


class PredictError(RuntimeError):
    pass


# ---------------------------------------------------------------- model shape (GGUF)

@dataclass
class AttnLayer:
    kv_heads: int
    head_dim: int
    window: int = 0                   # 0 = full attention


@dataclass
class Shape:
    """What the roofline needs from one GGUF. MiB = 2**20 bytes, GFLOP = 1e9 FLOPs (2 per weight)."""
    path: str
    arch: str
    dense_mib: float                  # read per token besides the routed experts (incl. output head)
    dense_gflop: float                # per token
    moe_layers: int
    expert_layer_mib: float           # all experts of one MoE layer (one -ncmoe step)
    expert_layer_gflop: float         # all experts of one layer, if every expert ran (scaled by n_used/n_expert)
    n_expert: int | None
    n_expert_used: int | None
    n_head: int | None
    attn: list[AttnLayer] = field(default_factory=list)

    @property
    def used_frac(self) -> float:
        if not self.moe_layers or not self.n_expert or not self.n_expert_used:
            return 0.0
        return self.n_expert_used / self.n_expert

    @property
    def active_expert_mib(self) -> float:
        """Expert bytes one token reads in one MoE layer."""
        return self.expert_layer_mib * self.used_frac

    def kv_mib_per_token(self, ctk: str | None, ctv: str | None, depth: float) -> float:
        """KV bytes read to generate one token at this context depth."""
        kb, vb = ff.KV_BYTES.get(ctk or "f16", 2.0), ff.KV_BYTES.get(ctv or "f16", 2.0)
        tot = 0.0
        for a in self.attn:
            n = min(depth, a.window) if a.window else depth
            tot += a.kv_heads * a.head_dim * (kb + vb) * n
        return tot / MIB

    def attn_gflop_per_token(self, avg_ctx: float) -> float:
        """QK^T + PV for one token against avg_ctx earlier tokens: 4 FLOPs per head-dim element."""
        h = self.n_head or 0
        tot = 0.0
        for a in self.attn:
            n = min(avg_ctx, a.window) if a.window else avg_ctx
            tot += 4 * h * a.head_dim * n
        return tot / GB

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Shape:
        d = dict(d)
        d["attn"] = [AttnLayer(**a) for a in d.get("attn") or []]
        return cls(**d)


def _n(shape: list[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _per_layer(v: Any, i: int) -> Any:
    if isinstance(v, list):
        return v[i] if i < len(v) else (v[-1] if v else None)
    return v


def shape_of(g: gguf.Gguf) -> Shape:
    n_layers = int(g.get("block_count") or 0)
    nextn = int(g.get("nextn_predict_layers", 0) or 0)
    first_nextn = n_layers - nextn if nextn else n_layers + 1
    names = {t.name for t in g.tensors}
    tied = "output.weight" not in names
    exp_b: dict[int, int] = {}
    exp_p: dict[int, int] = {}
    dense_b = dense_p = 0
    k_layers: set[int] = set()
    qkv_layers: set[int] = set()
    for t in g.tensors:
        m = gguf.EXPERT_RE.match(t.name)
        lm = gguf.LAYER_RE.match(t.name)
        layer = int(lm.group(1)) if lm else None
        if m:
            i = int(m.group(1))
            exp_b[i] = exp_b.get(i, 0) + t.nbytes
            exp_p[i] = exp_p.get(i, 0) + _n(t.shape)
            continue
        if layer is not None and layer >= first_nextn:
            continue                                   # MTP / NextN layer: not used by plain decoding
        if t.name.startswith("token_embd.") and not tied:
            continue                                   # a row lookup, not a matmul
        dense_b += t.nbytes
        if len(t.shape) >= 2:
            dense_p += _n(t.shape)
        if layer is not None:
            if ".attn_k." in t.name:
                k_layers.add(layer)
            elif ".attn_qkv." in t.name:
                qkv_layers.add(layer)
    exp_layers = [i for i in exp_b if i < first_nextn] or list(exp_b)
    swa_pat = g.get("attention.sliding_window_pattern")
    window = int(g.get("attention.sliding_window") or 0)
    hkv = g.get("attention.head_count_kv")
    n_head_raw = g.get("attention.head_count")
    key_len = g.get("attention.key_length")
    emb = g.get("embedding_length")
    n_head = gguf._first(n_head_raw)
    hd = int(key_len or (emb // n_head if emb and n_head else 0) or 0)
    hd_swa = int(g.get("attention.key_length_swa") or hd)
    attn: list[AttnLayer] = []
    for i in sorted(k_layers or qkv_layers):
        is_swa = bool(_per_layer(swa_pat, i)) if isinstance(swa_pat, list) else False
        kvh = int(_per_layer(hkv, i) or _per_layer(n_head_raw, i) or 0)
        attn.append(AttnLayer(kvh, hd_swa if is_swa else hd, window if is_swa and window else 0))
    L = len(exp_b)
    return Shape(
        path=g.path, arch=g.arch,
        dense_mib=dense_b / MIB, dense_gflop=2 * dense_p / GB,
        moe_layers=L,
        expert_layer_mib=(sum(exp_b[i] for i in exp_layers) / len(exp_layers) / MIB) if exp_layers else 0.0,
        expert_layer_gflop=(2 * sum(exp_p[i] for i in exp_layers) / len(exp_layers) / GB) if exp_layers else 0.0,
        n_expert=g.get("expert_count"), n_expert_used=g.get("expert_used_count"), n_head=n_head, attn=attn)


_SHAPES: dict[str, Shape] = {}


def read_shape(path: str) -> Shape:
    if path not in _SHAPES:
        _SHAPES[path] = shape_of(gguf.read(path))
    return _SHAPES[path]


# ---------------------------------------------------------------- hardware

@dataclass
class Hardware:
    vram_gbs: float                   # effective
    ram_gbs: float                    # effective, dual-channel region
    ram_single_gbs: float | None      # effective, single-channel region (asymmetric DIMMs): pessimistic bound
    pcie_gbs: float                   # effective
    tflops: float
    source: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def hardware_from(info: Any, tflops: float | None = None) -> Hardware:
    """Spec bandwidths from a HardwareInfo, times the efficiencies above."""
    src: dict[str, str] = {}
    gpu = info.primary_gpu() if info is not None else None
    if gpu is not None and gpu.bandwidth_gbs_est:
        vram = gpu.bandwidth_gbs_est * VRAM_EFF
        src["vram"] = f"{gpu.bandwidth_gbs_est:.0f} GB/s spec ({gpu.mem_bus_width_bits}-bit) x {VRAM_EFF}"
    else:
        vram = FALLBACK["vram_gbs"]
        src["vram"] = "default (no GPU bandwidth from hw)"
    gen = (gpu.pcie_gen_max or gpu.pcie_gen_current) if gpu is not None else None
    width = (gpu.pcie_width_max or gpu.pcie_width_current) if gpu is not None else None
    if gen in PCIE_LANE_GBS and width:
        pcie = PCIE_LANE_GBS[gen] * width * PCIE_EFF
        src["pcie"] = f"Gen{gen} x{width} {PCIE_LANE_GBS[gen] * width:.1f} GB/s spec x {PCIE_EFF}"
    else:
        pcie = FALLBACK["pcie_gbs"]
        src["pcie"] = "default (no PCIe link from hw)"
    mods = [m for m in (info.memory.modules if info is not None else []) if m.capacity_gb]
    mts = [m.configured_mts or m.rated_mts for m in mods if (m.configured_mts or m.rated_mts)]
    single = None
    if mts:
        per_ch = min(mts) * 8 / 1000                   # 64-bit channel: MT/s x 8 bytes
        ch = min(len(mods), 2) if len(mods) <= 2 else len(mods)
        ram = per_ch * ch * RAM_EFF
        caps = sorted(m.capacity_gb for m in mods)
        asym = len(caps) == 2 and abs(caps[0] - caps[1]) > 0.5
        if asym:
            single = per_ch * RAM_EFF
        src["ram"] = (f"{len(mods)} x DDR @ {min(mts)} MT/s = {per_ch * ch:.0f} GB/s spec x {RAM_EFF}"
                      + (f"; asymmetric: {caps[1] - caps[0]:g} GB single-channel ({per_ch:.0f} GB/s)" if asym else ""))
    else:
        ram = FALLBACK["ram_gbs"]
        src["ram"] = "default (no DIMM speeds from hw)"
    tf = tflops or DEFAULT_TFLOPS
    src["compute"] = f"{tf:g} TFLOPS effective ({'config' if tflops else 'default'})"
    return Hardware(vram, ram, single, pcie, tf, src)


# ---------------------------------------------------------------- the roofline

@dataclass
class Params:
    """Effective speeds used by the roofline (spec values times the calibration scales)."""
    vram_gbs: float
    ram_gbs: float
    t0_ms: float
    tflops: float
    pcie_gbs: float
    tub_ms: float

    @classmethod
    def spec(cls, hw: Hardware) -> Params:
        return cls(hw.vram_gbs, hw.ram_gbs, DEFAULT_T0_MS, hw.tflops, hw.pcie_gbs, DEFAULT_TUB_MS)

    def scaled(self, tg: list[float], pp: list[float]) -> Params:
        """Scale factors are on the *times* (1/bandwidth): factor 2 = half the bandwidth."""
        return Params(self.vram_gbs / tg[0], self.ram_gbs / tg[1], self.t0_ms * tg[2],
                      self.tflops / pp[0], self.pcie_gbs / pp[1], self.tub_ms * pp[2])

    def as_dict(self) -> dict[str, Any]:
        return {k: round(v, 3) for k, v in asdict(self).items()}


@dataclass
class Config1:
    """One configuration as the roofline sees it."""
    cpu_layers: int
    ub: int
    ctk: str | None
    ctv: str | None


def config_of(v: Variant, s: Shape) -> Config1:
    L = s.moe_layers
    cpu = L if v.cpu_moe else (min(v.n_cpu_moe, L) if v.n_cpu_moe is not None else 0)
    ub = min(v.ub or DEFAULT_UB, v.b or DEFAULT_B)
    return Config1(cpu, ub, v.ctk, v.ctv)


def tg_terms(s: Shape, c: Config1, depth: float) -> tuple[float, float]:
    """(bytes read from VRAM, bytes read from RAM) per generated token."""
    gpu_layers = s.moe_layers - c.cpu_layers
    vram = (s.dense_mib + gpu_layers * s.active_expert_mib + s.kv_mib_per_token(c.ctk, c.ctv, depth)) * MIB
    ram = c.cpu_layers * s.active_expert_mib * MIB
    return vram, ram


def hit_frac(s: Shape, ub: int) -> float:
    """Share of a layer's experts that at least one of `ub` tokens routes to."""
    f = s.used_frac
    return 1.0 - (1.0 - f) ** max(1, ub) if f else 0.0


def pp_terms(s: Shape, c: Config1, n_prompt: int, depth: int = 0) -> tuple[float, float, float]:
    """(FLOPs, PCIe bytes, ubatches) per prompt token, for an n_prompt-token prompt after `depth` tokens."""
    ub = max(1, min(c.ub, n_prompt or c.ub))
    flops = (s.dense_gflop + s.moe_layers * s.expert_layer_gflop * s.used_frac
             + s.attn_gflop_per_token(depth + (n_prompt or 0) / 2)) * GB
    upload = c.cpu_layers * s.expert_layer_mib * MIB * hit_frac(s, ub) / ub
    return flops, upload, 1.0 / ub


def tg_time(p: Params, s: Shape, c: Config1, depth: float) -> float:
    vram, ram = tg_terms(s, c, depth)
    return vram / (p.vram_gbs * GB) + ram / (p.ram_gbs * GB) + p.t0_ms / 1000


def pp_time(p: Params, s: Shape, c: Config1, n_prompt: int, depth: int = 0) -> float:
    flops, upload, per_ub = pp_terms(s, c, n_prompt, depth)
    return flops / (p.tflops * 1e12) + upload / (p.pcie_gbs * GB) + per_ub * p.tub_ms / 1000


def pp_split(p: Params, s: Shape, c: Config1, n_prompt: int) -> dict[str, float]:
    """Share of the PP time per part: compute / PCIe upload / fixed per ubatch."""
    flops, upload, per_ub = pp_terms(s, c, n_prompt)
    parts = {"compute": flops / (p.tflops * 1e12), "upload": upload / (p.pcie_gbs * GB),
             "fixed": per_ub * p.tub_ms / 1000}
    tot = sum(parts.values()) or 1.0
    return {k: v / tot for k, v in parts.items()}


def tg_gain_ms_per_gib(p: Params, s: Shape) -> float:
    """TG time saved per token for each GiB of VRAM given to experts (one -ncmoe step = one layer)."""
    if not s.moe_layers or not s.expert_layer_mib:
        return 0.0
    per_layer = s.active_expert_mib * MIB * (1 / (p.ram_gbs * GB) - 1 / (p.vram_gbs * GB)) * 1000
    return per_layer / (s.expert_layer_mib / 1024)


# ---------------------------------------------------------------- calibration from stored runs

@dataclass
class Sample:
    run_id: str
    model: str
    variant: str
    kind: str                         # tg | pp
    shape: Shape
    cfg: Config1
    n_prompt: int
    depth: float
    measured: float                   # t/s
    build: str | None = None


def samples_from_runs(runs: list[Any], shape_for: Callable[[str], Shape | None]) -> tuple[list[Sample], list[str]]:
    """TG and PP samples from ok RunResults without speculative decoding. (samples, notes on skipped runs)."""
    from mse_llm.variants import decode

    out: list[Sample] = []
    notes: list[str] = []
    for r in runs:
        if r.status != "ok" or r.tool not in ("bench", "benchy"):
            continue
        try:
            v = decode(r.variant)
        except ValueError:
            continue
        if v.spec_type:
            notes.append(f"{r.run_id}: {r.variant} uses speculative decoding (TG is not plain decoding): not used")
            continue
        s = shape_for(r.model_path or "")
        if s is None:
            notes.append(f"{r.run_id}: GGUF not readable ({r.model_path}): not used")
            continue
        c = config_of(v, s)
        for t in r.tests:
            if t.tg_tps and t.tg_tps > 0:
                d = t.depth + (t.n_prompt if t.kind == "pp+tg" else 0) + t.n_gen / 2
                out.append(Sample(r.run_id, r.model, r.variant, "tg", s, c, t.n_prompt, d, t.tg_tps,
                                  r.build.get("id")))
            if t.pp_tps and t.pp_tps > 0 and t.n_prompt >= 64:
                out.append(Sample(r.run_id, r.model, r.variant, "pp", s, c, t.n_prompt, t.depth, t.pp_tps,
                                  r.build.get("id")))
    return out, notes


def _features(sm: Sample, base: Params) -> list[float]:
    """Seconds per token contributed by each of the three terms at the base parameters."""
    if sm.kind == "tg":
        vram, ram = tg_terms(sm.shape, sm.cfg, sm.depth)
        return [vram / (base.vram_gbs * GB), ram / (base.ram_gbs * GB), base.t0_ms / 1000]
    flops, upload, per_ub = pp_terms(sm.shape, sm.cfg, sm.n_prompt, int(sm.depth))
    return [flops / (base.tflops * 1e12), upload / (base.pcie_gbs * GB), per_ub * base.tub_ms / 1000]


def _solve(rows: list[tuple[list[float], float]], prior: float = PRIOR_WEIGHT,
           limits: tuple[float, float] = SCALE_LIMITS) -> list[float]:
    """Scale factors k (times = sum k_i f_i) minimising the relative error, pulled towards 1."""
    import numpy as np

    if not rows:
        return [1.0, 1.0, 1.0]
    A, y = [], []
    for f, meas_s in rows:
        A.append([x / meas_s for x in f])
        y.append(1.0)
    for i in range(3):
        A.append([prior if j == i else 0.0 for j in range(3)])
        y.append(prior)
    k, *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)
    lo, hi = limits
    return [float(min(hi, max(lo, x))) for x in k]


MODEL_LIMITS = (0.1, 20.0)            # per model: a slow CPU kernel for one quant type can be many times off
MODEL_PRIOR = 0.1                     # weaker pull than the global one: a model's own runs say more about it
ONE = [1.0, 1.0, 1.0]


def _mul(a: list[float], b: list[float]) -> list[float]:
    return [x * y for x, y in zip(a, b, strict=True)]


@dataclass
class Calibration:
    """Global scales (this PC) and per-model scales on top (what one model's kernels / quant types do
    differently, e.g. how fast the CPU runs its experts). Scale > 1 = that term takes longer than predicted."""
    base: Params
    tg_scale: list[float] = field(default_factory=lambda: list(ONE))
    pp_scale: list[float] = field(default_factory=lambda: list(ONE))
    model_scale: dict[str, dict[str, Any]] = field(default_factory=dict)   # model -> {tg: [3], pp: [3], n_tg, n_pp}
    n_samples: int = 0
    runs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def params(self) -> Params:
        return self.base.scaled(self.tg_scale, self.pp_scale)

    @property
    def calibrated(self) -> bool:
        return self.n_samples > 0

    def params_for(self, model: str | None) -> Params:
        ms = self.model_scale.get(model or "")
        if not ms:
            return self.params
        return self.base.scaled(_mul(self.tg_scale, ms.get("tg", ONE)), _mul(self.pp_scale, ms.get("pp", ONE)))

    def tg(self, s: Shape, c: Config1, depth: float, model: str | None = None) -> float:
        return 1.0 / tg_time(self.params_for(model), s, c, depth)

    def pp(self, s: Shape, c: Config1, n_prompt: int, model: str | None = None, depth: int = 0) -> float:
        return 1.0 / pp_time(self.params_for(model), s, c, n_prompt, depth)

    def predict(self, sm: Sample, with_model: bool = True) -> float:
        m = sm.model if with_model else None
        if sm.kind == "tg":
            return self.tg(sm.shape, sm.cfg, sm.depth, m)
        return self.pp(sm.shape, sm.cfg, sm.n_prompt, m, int(sm.depth))

    def as_dict(self) -> dict[str, Any]:
        return {"base": self.base.as_dict(), "params": self.params.as_dict(),
                "tg_scale": [round(x, 3) for x in self.tg_scale], "pp_scale": [round(x, 3) for x in self.pp_scale],
                "model_scale": self.model_scale,
                "model_params": {m: self.params_for(m).as_dict() for m in self.model_scale},
                "n_samples": self.n_samples, "runs": self.runs, "notes": self.notes}


def calibrate(base: Params, samples: list[Sample], notes: list[str] | None = None) -> Calibration:
    """Global fit over every sample, then per model the same three-term fit on what is left (pulled towards 1,
    so one run moves a model's terms only part of the way)."""
    cal = Calibration(base, notes=list(notes or []))
    if not samples:
        return cal
    tg = [(_features(s, base), 1.0 / s.measured) for s in samples if s.kind == "tg"]
    pp = [(_features(s, base), 1.0 / s.measured) for s in samples if s.kind == "pp"]
    cal.tg_scale, cal.pp_scale = _solve(tg), _solve(pp)
    cal.n_samples = len(samples)
    cal.runs = sorted({s.run_id for s in samples})
    glob = cal.params
    for m in dict.fromkeys(s.model for s in samples):
        own = [s for s in samples if s.model == m]
        ms: dict[str, Any] = {}
        for kind in ("tg", "pp"):
            rows = [(_features(s, glob), 1.0 / s.measured) for s in own if s.kind == kind]
            if rows:
                ms[kind] = [round(x, 4) for x in _solve(rows, prior=MODEL_PRIOR, limits=MODEL_LIMITS)]
                ms[f"n_{kind}"] = len(rows)
        cal.model_scale[m] = ms
    return cal


@dataclass
class CheckRow:
    run_id: str
    model: str
    variant: str
    kind: str
    n_prompt: int
    measured: float
    predicted: float                  # in-sample (all runs)
    loo: float | None                 # leave-one-run-out
    spec: float                       # uncalibrated

    @property
    def err(self) -> float:
        return self.predicted / self.measured - 1

    @property
    def loo_err(self) -> float | None:
        return None if self.loo is None else self.loo / self.measured - 1

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(err=round(self.err, 4), loo_err=None if self.loo_err is None else round(self.loo_err, 4),
                 spec_err=round(self.spec / self.measured - 1, 4))
        return d


def check(base: Params, samples: list[Sample], notes: list[str] | None = None) -> tuple[Calibration, list[CheckRow]]:
    """Calibrate on everything, and for each run also predict it from a calibration without that run
    (leave-one-out: the honest error for a config that wasn't measured)."""
    cal = calibrate(base, samples, notes)
    spec = Calibration(base)
    rows: list[CheckRow] = []
    for rid in sorted({s.run_id for s in samples}):
        rest = [s for s in samples if s.run_id != rid]
        loo_cal = calibrate(base, rest) if rest else None
        for s in samples:
            if s.run_id != rid:
                continue
            rows.append(CheckRow(rid, s.model, s.variant, s.kind, s.n_prompt, s.measured, cal.predict(s),
                                 loo_cal.predict(s) if loo_cal else None, spec.predict(s)))
    return cal, rows


def same_plan(run: Any, plan: str | None) -> bool:
    """Runs recorded under another power plan are not comparable (a quiet plan caps CPU / GPU power).
    Runs without a recorded plan (before M9) are kept."""
    rp = (getattr(run, "hw", None) or {}).get("power_plan")
    return not plan or not rp or rp == plan


def stored_runs(hw_hash: str | None = None, model: str | None = None, root: Path | None = None,
                power_plan: str | None = None) -> list[Any]:
    """Newest ok run per run key from the results store (same hardware when hw_hash is given, same power plan
    when power_plan is given)."""
    from mse_llm import results as rs

    picked: dict[str, Any] = {}
    for e in reversed(rs.read_index(root)):               # newest first
        k = e["key"] + str(e.get("build"))
        if k in picked or e.get("status") != "ok":
            continue
        if (hw_hash and e.get("hw") and e.get("hw") != hw_hash) or (model and e.get("model") != model):
            continue
        try:
            r = rs.find_run(e["run_id"], root)
        except KeyError:
            continue
        if same_plan(r, power_plan):
            picked[k] = r
    return list(reversed(picked.values()))


def _shape_or_none(path: str) -> Shape | None:
    try:
        return read_shape(path)
    except (OSError, gguf.GgufError):
        return None


def load_calibration(base: Params, *, hw_hash: str | None = None, use: bool = True, power_plan: str | None = None,
                     runs: list[Any] | None = None,
                     shape_for: Callable[[str], Shape | None] | None = None) -> Calibration:
    if not use:
        return Calibration(base)
    runs = stored_runs(hw_hash, power_plan=power_plan) if runs is None else runs
    samples, notes = samples_from_runs(runs, shape_for or _shape_or_none)
    return calibrate(base, samples, notes)


# ---------------------------------------------------------------- memory grid (llama-fit-params)

@dataclass
class Point:
    variant: Variant
    ncmoe: int | None
    ub: int
    gpu_mib: float | None = None
    host_mib: float | None = None
    mem_source: str = "measured"       # measured | interp | error
    error: str | None = None
    free_mib: int | None = None
    margin_mib: int = TIGHT_DEFAULT
    tg: float | None = None
    pp: float | None = None
    note: str = ""

    @property
    def left_mib(self) -> float | None:
        if self.gpu_mib is None or self.free_mib is None:
            return None
        return self.free_mib - self.gpu_mib

    @property
    def verdict(self) -> str:
        if self.error:
            return "ERROR"
        left = self.left_mib
        if left is None:
            return "?"
        v = "no-fit" if left < 0 else "tight" if left < self.margin_mib else "fits"
        return v + ("~" if self.mem_source == "interp" else "")

    @property
    def fits(self) -> bool:
        return self.verdict.rstrip("~") == "fits"

    def as_dict(self) -> dict[str, Any]:
        return {"variant": self.variant.name, "ncmoe": self.ncmoe, "ub": self.ub,
                "gpu_mib": None if self.gpu_mib is None else round(self.gpu_mib),
                "host_mib": None if self.host_mib is None else round(self.host_mib),
                "left_mib": None if self.left_mib is None else round(self.left_mib),
                "mem_source": self.mem_source, "verdict": self.verdict, "error": self.error,
                "tg_tps": None if self.tg is None else round(self.tg, 2),
                "pp_tps": None if self.pp is None else round(self.pp, 1), "note": self.note}


def with_axes(v: Variant, ncmoe: int | None, ub: int | None) -> Variant:
    """The base variant with -ncmoe / -b -ub replaced (b = ub, as in the bash sweeps)."""
    nv = v.without_spec()
    if ncmoe is not None:
        nv = replace(nv, n_cpu_moe=ncmoe, cpu_moe=False)
    if ub is not None:
        nv = replace(nv, ub=None, b=None) if ub == DEFAULT_UB else replace(nv, ub=ub, b=ub)
    return nv


def default_ncmoe_values(s: Shape, n: int = 7) -> list[int]:
    """From all experts on the CPU (L) down to 0, n values, high (safe) first."""
    L = s.moe_layers
    if not L:
        return []
    step = max(1, math.ceil(L / (n - 1)))
    vals = list(range(L, -1, -step))
    if vals[-1] != 0:
        vals.append(0)
    return vals


DEFAULT_UBS = [512, 1024, 2048, 4096]


def memory_grid(cfg: Config, model: str, base: Variant, ctx: int, ncmoes: list[int | None], ubs: list[int], *,
                free_mib: int | None, margin_mib: int, engine: str | None = None, exact: bool = False,
                runner: ff.Runner | None = None, info_of: Callable[[str], gguf.ModelInfo] | None = None,
                progress: Callable[[int, int, str], None] | None = None) -> tuple[list[Point], list[str]]:
    """GPU / host MiB for every (ncmoe, ub). Without `exact` only the highest and lowest -ncmoe are measured per
    ub; the values in between follow the linear rule (one -ncmoe step = one expert layer)."""
    pts = [Point(with_axes(base, n, u), n, u, free_mib=free_mib, margin_mib=margin_mib) for u in ubs for n in ncmoes]
    measured_n = set(ncmoes) if exact or len(ncmoes) <= 2 else {max(n for n in ncmoes if n is not None),
                                                                 min(n for n in ncmoes if n is not None)} \
        if any(n is not None for n in ncmoes) else {None}
    todo = [p for p in pts if p.ncmoe in measured_n]
    if todo:
        rep = ff.run_fit(cfg, model, [ff.FitInput(p.variant) for p in todo], [ctx], engine=engine, free_mib=free_mib,
                         free_source="", margin_mib=margin_mib, runner=runner, info_of=info_of, progress=progress)
        for p, row in zip(todo, rep.rows, strict=True):
            if row.error:
                p.error, p.mem_source = row.error, "error"
            else:
                p.gpu_mib, p.host_mib = row.gpu_mib, row.host_mib
        warnings = rep.warnings
    else:
        warnings = []
    for u in ubs:
        known = sorted((p for p in pts if p.ub == u and p.ncmoe in measured_n and p.gpu_mib is not None),
                       key=lambda p: p.ncmoe or 0)
        for p in pts:
            if p.ub != u or p.ncmoe in measured_n:
                continue
            if len(known) < 2:
                p.error, p.mem_source = "no measured endpoints to interpolate from", "error"
                continue
            lo, hi = known[0], known[-1]
            t = ((p.ncmoe or 0) - (lo.ncmoe or 0)) / (((hi.ncmoe or 0) - (lo.ncmoe or 0)) or 1)
            p.gpu_mib = lo.gpu_mib + t * (hi.gpu_mib - lo.gpu_mib)
            p.host_mib = (lo.host_mib or 0) + t * ((hi.host_mib or 0) - (lo.host_mib or 0))
            p.mem_source = "interp"
    return pts, warnings


# ---------------------------------------------------------------- the command

@dataclass
class Prediction:
    model: str
    model_key: str | None
    path: str
    ctx: int
    base: str
    shape: Shape
    hardware: Hardware
    calibration: Calibration
    points: list[Point]
    free_mib: int | None
    free_source: str
    margin_mib: int
    tg_depth: float
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unsafe: dict[tuple[int | None, int, str | None, str | None], str] = field(default_factory=dict, repr=False)
    base_variant: Variant | None = field(default=None, repr=False)

    def at(self, ncmoe: int | None, ub: int, variant: Variant | None = None) -> Point:
        """Memory and speed for any -ncmoe at a grid -ub: linear between the two nearest grid points
        (the same rule as the grid; one -ncmoe step = one expert layer)."""
        v = with_axes(variant or self.base_variant or Variant(), ncmoe, ub)
        p = Point(v, ncmoe, ub, free_mib=self.free_mib, margin_mib=self.margin_mib)
        row = sorted((q for q in self.points if q.ub == ub and q.gpu_mib is not None), key=lambda q: q.ncmoe or 0)
        exact = [q for q in row if q.ncmoe == ncmoe]
        if exact:
            p.gpu_mib, p.host_mib, p.mem_source = exact[0].gpu_mib, exact[0].host_mib, exact[0].mem_source
        elif len(row) >= 2 and ncmoe is not None:
            lo = max((q for q in row if (q.ncmoe or 0) <= ncmoe), key=lambda q: q.ncmoe or 0, default=row[0])
            hi = min((q for q in row if (q.ncmoe or 0) >= ncmoe), key=lambda q: q.ncmoe or 0, default=row[-1])
            if lo is hi:
                lo, hi = (row[0], row[1]) if ncmoe < (row[0].ncmoe or 0) else (row[-2], row[-1])
            t = (ncmoe - (lo.ncmoe or 0)) / (((hi.ncmoe or 0) - (lo.ncmoe or 0)) or 1)
            p.gpu_mib = lo.gpu_mib + t * (hi.gpu_mib - lo.gpu_mib)
            p.host_mib = (lo.host_mib or 0) + t * ((hi.host_mib or 0) - (lo.host_mib or 0))
            p.mem_source = "interp"
        else:
            p.error, p.mem_source = f"-ub {ub} is not in the predicted grid", "error"
        self._speed(p)
        return p

    def _speed(self, p: Point) -> None:
        c = config_of(p.variant, self.shape)
        p.tg = self.calibration.tg(self.shape, c, self.tg_depth, self.model_key)
        p.pp = self.calibration.pp(self.shape, c, self.ctx, self.model_key)
        nk = (p.variant.n_cpu_moe, c.ub, p.variant.ctk, p.variant.ctv)
        if nk in self.unsafe:
            p.note = (f"fits, but unsafe in your notes: {self.unsafe[nk]}" if p.fits
                      else f"your notes: {self.unsafe[nk]}")

    def lowest_fitting_ncmoe(self, ub: int) -> int | None:
        """The lowest -ncmoe that fits with the margin at this -ub (None: not even all experts on the CPU)."""
        L = self.shape.moe_layers
        for n in range(0, L + 1):
            if self.at(n, ub).fits:
                return n
        return None

    def fitting(self) -> list[Point]:
        return [p for p in self.points if p.fits and p.tg is not None]

    def best(self, goal: str) -> Point | None:
        pts = self.fitting()
        if not pts:
            return None
        return max(pts, key=lambda p: score(p, goal))

    def as_dict(self) -> dict[str, Any]:
        b = {g: (p.variant.name if (p := self.best(g)) else None) for g in ("tg", "pp", "both")}
        return {"model": self.model, "model_key": self.model_key, "path": self.path, "ctx": self.ctx,
                "base": self.base, "shape": self.shape.as_dict(), "hardware": self.hardware.as_dict(),
                "calibration": self.calibration.as_dict(), "free_mib": self.free_mib,
                "free_source": self.free_source, "margin_mib": self.margin_mib, "tg_depth": self.tg_depth,
                "points": [p.as_dict() for p in self.points], "best": b,
                "recommendations": self.recommendations, "warnings": self.warnings}


def score(p: Point, goal: str) -> float:
    tg, pp = p.tg or 0.0, p.pp or 0.0
    if goal == "tg":
        return tg + pp * 1e-9
    if goal == "pp":
        return pp + tg * 1e-9
    return math.sqrt(max(tg, 0) * max(pp, 0))


def unsafe_notes(cfg: Config, model_key: str | None) -> dict[tuple[int | None, int, str | None, str | None], str]:
    """Configs your sweep notes mark as freezing: (ncmoe, ub, ctk, ctv) -> note."""
    from mse_llm.variants import VariantError, decode

    out: dict[tuple[int | None, int, str | None, str | None], str] = {}
    if not model_key:
        return out
    for sw in cfg.sweeps.values():
        if sw.model != model_key:
            continue
        for e in sw.entries():
            if not (e.tag in ("freeze", "sporadic-freeze") or "freeze" in (e.note or "").lower()):
                continue
            try:
                v = decode(e.v)
            except VariantError:
                continue
            out[(v.n_cpu_moe, min(v.ub or DEFAULT_UB, v.b or DEFAULT_B), v.ctk, v.ctv)] = e.note or e.tag or "freeze"
    return out


def run_predict(cfg: Config, model: str, base: Variant, ctx: int, *, ncmoes: list[int] | None = None,
                ubs: list[int] | None = None, free_mib: int | None, free_source: str = "",
                margin_mib: int = TIGHT_DEFAULT, engine: str | None = None, exact: bool = False,
                hardware: Hardware | None = None, calibration: Calibration | None = None,
                tg_depth: float | None = None, runner: ff.Runner | None = None,
                info_of: Callable[[str], gguf.ModelInfo] | None = None,
                shape_for: Callable[[str], Shape] | None = None,
                progress: Callable[[int, int, str], None] | None = None) -> Prediction:
    key, m, path = ff.resolve_model(cfg, model)
    try:
        shape = (shape_for or read_shape)(path)
    except FileNotFoundError:
        raise PredictError(f"model file not found: {path}") from None
    except (OSError, gguf.GgufError) as e:
        raise PredictError(f"can't read the GGUF header: {e}") from None
    warnings: list[str] = []
    if base.spec_type:
        warnings.append(f"{base.name}: speculative decoding left out; TG is predicted for plain decoding")
    hw_ = hardware or hardware_from(None)
    cal = calibration or Calibration(Params.spec(hw_))
    ns: list[int | None] = list(ncmoes) if ncmoes else (default_ncmoe_values(shape) or [None])
    if shape.moe_layers == 0:
        ns = [None]
    us = list(ubs) if ubs else DEFAULT_UBS
    pts, w = memory_grid(cfg, model, base, ctx, ns, us, free_mib=free_mib, margin_mib=margin_mib, engine=engine,
                         exact=exact, runner=runner, info_of=info_of, progress=progress)
    warnings += w
    depth = tg_depth if tg_depth is not None else TG_DEPTH
    pred = Prediction(model, key, path, ctx, base.name, shape, hw_, cal, pts, free_mib, free_source, margin_mib,
                      depth, warnings=warnings, unsafe=unsafe_notes(cfg, key), base_variant=base)
    for p in pts:
        pred._speed(p)
    pred.recommendations = recommend(pred)
    return pred


def recommend(pred: Prediction) -> list[str]:
    s, p = pred.shape, pred.calibration.params
    out: list[str] = []
    name = pred.model_key or Path(pred.path).stem
    fit = pred.fitting()
    if s.moe_layers:
        gain = tg_gain_ms_per_gib(p, s)
        start = max(fit, key=lambda q: ((q.ncmoe or 0), -q.ub)) if fit else None
        split = pp_split(p, s, config_of(start.variant, s), pred.ctx) if start else None
        tg_first = gain >= 0.7
        out.append(f"{name}: n_used/n_expert = {s.n_expert_used}/{s.n_expert} ({s.used_frac:.1%}); each GiB of VRAM "
                   f"given to experts saves ~{gain:.2f} ms per token "
                   + ("→ lower -ncmoe first" if tg_first else "→ TG gains little from a lower -ncmoe"))
        if split is not None and start is not None:
            up = split["upload"]
            if up >= 0.5:
                out.append(f"PP at -ncmoe {start.ncmoe} -ub {start.ub}: {up:.0%} of the time is PCIe upload of CPU "
                           "experts → a bigger -ub gives almost proportional PP ('batching works'); "
                           "~1 MiB VRAM per ubatch token")
            else:
                out.append(f"PP at -ncmoe {start.ncmoe} -ub {start.ub}: only {up:.0%} PCIe upload → a bigger -ub adds "
                           "little ('batching NOT helps') and costs ~1 MiB VRAM per ubatch token")
    else:
        out.append(f"{name}: dense model, all weights on the GPU: -ncmoe doesn't apply; -ub trades VRAM for PP")
    for goal, label in (("tg", "fastest TG"), ("pp", "fastest PP"), ("both", "best TG x PP")):
        b = pred.best(goal)
        if b is not None:
            out.append(f"{label}: {b.variant.name}  (TG ~{b.tg:.1f} t/s, PP ~{b.pp:.0f} t/s, "
                       f"{b.left_mib:.0f} MiB left)" + (f"  ! {b.note}" if b.note else ""))
    if not fit:
        out.append("nothing in the grid fits with the margin: lower the context, raise -ncmoe or check free VRAM")
    if pred.hardware.ram_single_gbs and s.moe_layers:
        slow = pred.hardware.ram_single_gbs / pred.hardware.ram_gbs
        out.append(f"RAM: asymmetric DIMMs; experts that land in the single-channel region read at ~{slow:.0%} of the "
                   f"dual-channel speed (~{p.ram_gbs * slow:.0f} instead of ~{p.ram_gbs:.0f} GB/s here), so TG with all"
                   f" experts on the CPU can be up to {_single_bound(pred):.0%} lower (run-to-run variance)")
    return out


def _single_bound(pred: Prediction) -> float:
    s, p = pred.shape, pred.calibration.params
    if not pred.hardware.ram_single_gbs or not s.moe_layers:
        return 0.0
    c = Config1(s.moe_layers, DEFAULT_UB, None, None)
    slow = replace(p, ram_gbs=pred.hardware.ram_single_gbs * p.ram_gbs / pred.hardware.ram_gbs)
    return 1 - tg_time(p, s, c, 0) / tg_time(slow, s, c, 0)


def parse_int_list(text: str | None) -> list[int] | None:
    """`41,34,28`, `16:30:2` (start:stop:step, inclusive) or a mix."""
    if not text:
        return None
    out: list[int] = []
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        if ":" in part:
            a, b, *st = part.split(":")
            step = int(st[0]) if st else 1
            lo, hi = int(a), int(b)
            rng = range(lo, hi + 1, step) if lo <= hi else range(lo, hi - 1, -step)
            out += list(rng)
        else:
            out.append(int(part))
    return out
