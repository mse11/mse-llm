"""Configuration models (TOML, §4.1) with MSE_LLM_* env overrides.

Lookup order for the file: --config, MSE_LLM_CONFIG, config/config.toml, config/config.example.toml.
Engines come from [engines.*] in the config and from the [engine] table of every build profile
(config/builds/*.toml). A config entry with the same name wins over the build profile's.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mse_llm.util import project_root


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class BenchyDefaults(_Model):
    pp: list[int] = Field(default_factory=lambda: [512, 2048, 8192, 32768])
    pp_sync_ctx: bool = True
    depth: list[int] = Field(default_factory=lambda: [0])
    runs: int = 1
    version: str = "llama-benchy"


class Defaults(_Model):
    threads: int = 4
    parallel: int = 1
    ngl: str = "all"
    fit: Literal["on", "off"] = "off"
    load_mode: str = "none"
    bench_tg: list[int] = Field(default_factory=lambda: [128])
    bench_repetitions: int = 1
    server_ready_timeout: int = 120
    notify: bool = True
    monitor: bool = True
    output: Literal["all", "filter", "none"] = "filter"
    output_filter: list[str] | None = None   # regexes for --output filter; None = bash run_serve_params_FILTER
    benchy: BenchyDefaults = Field(default_factory=BenchyDefaults)


class EngineCfg(_Model):
    runner: str = "llamacpp"
    bin_dir: str
    host: str = "0.0.0.0"
    port: int = 8033
    client_host: str = "localhost"     # how clients (benchy, inspect) reach the server (bash SERVERS_NET_CFG url)
    help_snapshot: str | None = None   # folder under tests/fixtures/help, e.g. "a49db739f"
    build_profile: str | None = None

    def bin_path(self) -> Path:
        return Path(self.bin_dir.replace("{project}", str(project_root())))

    def base_url(self, port: int | None = None) -> str:
        return f"http://{self.client_host}:{port or self.port}/v1"


class ModelCfg(_Model):
    path: str
    alias: str | None = None
    draft: str | None = None           # drafter GGUF, or "builtin" (MTP heads in the model; bash "BUILD-IN")
    ctx: int | None = None             # default context (bash run_SERVER_*/run_BENCH_* runner_ctx default)
    mmproj: str | None = None
    url: str | None = None
    tags: list[str] = Field(default_factory=list)
    bench_extra: list[str] = Field(default_factory=list)
    serve_extra: dict[str, Any] = Field(default_factory=dict)
    default_serve: str | None = None
    default_serve_spec: str | None = None
    # the variant run_SERVER_* / llama_cpp_presets.ini serve, when default_serve was changed since (parity tests;
    # dropped with the bash script in M13)
    bash_serve: str | None = None
    bash_serve_spec: str | None = None
    notes: str | None = None

    @property
    def label(self) -> str:
        """Name llama-server reports at /v1/models without --alias: the GGUF file name without .gguf."""
        name = self.path.replace("\\", "/").rsplit("/", 1)[-1]
        return name[:-5] if name.lower().endswith(".gguf") else name


class PresetsCfg(_Model):
    # relative to the repo root (bash MSE_PRESETS_SERVE_FILE); M13 switches to config/presets.ini
    file: str = "llama_cpp_presets.ini"
    engine: str | None = None
    port: int | None = None                    # default: the engine's port
    models_max: int | None = None              # --models-max (router default 4)
    memory_margin_mib: int | None = None       # --models-memory-margin: LRU unload to keep this much VRAM free
    autoload: bool = True                      # False -> --no-models-autoload
    load_on_start: list[str] = Field(default_factory=list)   # preset names loaded right after start

    def path(self) -> Path:
        p = Path(self.file)
        return p if p.is_absolute() else project_root() / p


class MonitorCfg(_Model):
    """[monitor] (§5, §9 item 7). Status thresholds are the bash ones; the guard stops a run live."""
    interval: float = 0.5                  # seconds between samples (bash: 1 s, with 3-4 powershell calls each)
    guard: bool = True                     # stop the server on a spill / RAM / commit trip
    spill_stop_mib: int = 100              # bash: shared delta > 100 MiB = offloaded to system RAM
    vram_full_mib: int = 256               # dedicated VRAM counts as full below this much free
    host_tolerance_pct: float = 2.0        # allowed error of fit's host-buffer size (measured: 0.5 %)
    ram_stop_pct: float = 95.0
    commit_stop_pct: float = 95.0
    stall_stop_s: float = 90.0             # bench/benchy: stop after this long without progress (0 = off)
    job_memory_cap_mib: int | None = None  # Windows Job Object cap on the server tree's committed memory
    ram_warn_pct: float = 60.0             # bash status thresholds
    ram_fail_pct: float = 80.0
    temp_warn_c: float = 75.0
    temp_fail_c: float = 85.0


class SweepEntry(_Model):
    """One line of a bash extras_list, as data: `{ v = "fa_Kq8_Vq4_default_moe34", status = "off", note = "…" }`.

    status: "on" runs; "off" was commented out in bash (runs with `sweep run --include-off`); "skip" is never
    started without --force (bash "PC freeze" / "SKIPPED" notes).
    tag: sweetspot | freeze | sporadic-freeze | skipped."""
    v: str
    status: Literal["on", "off", "skip"] = "on"
    note: str = ""
    tag: str | None = None


class SweepCfg(_Model):
    model: str
    engine: str = "cuda_flash"
    tool: Literal["bench", "benchy", "eval"] = "bench"
    ctx: int | None = None
    variants: list[str | SweepEntry] = Field(default_factory=list)
    matrix: dict[str, list[Any]] | None = None
    annotations: dict[str, dict[str, Any]] = Field(default_factory=dict)   # name -> {status, note, tag}
    # bench: -p sizes (bash _wrapper_PROMPTS) and -n; benchy: pp / depth / runs / concurrency
    prompts: list[int] = Field(default_factory=list)
    tg: list[int] = Field(default_factory=list)
    repetitions: int | None = None
    pp: list[int] = Field(default_factory=list)
    depth: list[int] = Field(default_factory=list)
    runs: int | None = None
    concurrency: list[int] = Field(default_factory=list)
    # benchy: prompt sets, one job per set: prose (benchy's book) | code (llama.cpp sources) | wikitext2 | FILE.
    # Speculative-decoding acceptance depends on the content, so spec sweeps run both (§4.5).
    prompt_sets: list[str] = Field(default_factory=list)
    # M11 (§4.4): staged sweep, one list of patterns per stage; `{best}` / `{winner}` / `{tuned}` placeholders
    # come from the stages before (see mse_llm/staged.py). `variants` / `matrix` are not used with stages.
    stages: list[list[str]] = Field(default_factory=list)
    goal: Literal["tg", "pp", "both"] = "tg"    # what picks a stage winner (the MoE cache mainly changes TG)
    warmup: int | None = None               # benchy, mc* variants: tokens generated before benchy (cache warm)
    task: str | None = None                 # eval: inspect task (M12)
    bash: str | None = None                 # the bash function this sweep replaces
    notes: str | None = None

    def entries(self) -> list[SweepEntry]:
        """Every variant in order (brace patterns expanded, matrix appended), annotations applied."""
        from mse_llm import variants as vv

        out: list[SweepEntry] = []
        seen: set[str] = set()
        raw = [e if isinstance(e, SweepEntry) else SweepEntry(v=e) for e in self.variants]
        raw += [SweepEntry(v=v.name) for v in (vv.expand_matrix(self.matrix) if self.matrix else [])]
        for e in raw:
            for n in vv.expand(e.v):
                name = vv.canonical(n)
                if name in seen:
                    continue
                seen.add(name)
                ann = self.annotations.get(name) or self.annotations.get(n) or {}
                out.append(SweepEntry(v=name, status=ann.get("status", e.status), note=ann.get("note", e.note),
                                      tag=ann.get("tag", e.tag)))
        return out


class HwChecksCfg(_Model):
    ram_warn_gb: float = 16
    ram_pass_gb: float = 32
    cores_warn: int = 4
    cores_pass: int = 8
    disk_warn_gb: float = 25
    disk_pass_gb: float = 60


class HwCfg(_Model):
    checks: HwChecksCfg = Field(default_factory=HwChecksCfg)


class Config(_Model):
    defaults: Defaults = Field(default_factory=Defaults)
    engines: dict[str, EngineCfg] = Field(default_factory=dict)
    models: dict[str, ModelCfg] = Field(default_factory=dict)
    sweeps: dict[str, SweepCfg] = Field(default_factory=dict)
    hw: HwCfg = Field(default_factory=HwCfg)
    presets: PresetsCfg = Field(default_factory=PresetsCfg)
    monitor: MonitorCfg = Field(default_factory=MonitorCfg)
    source: Path | None = Field(default=None, exclude=True)

    def engine(self, name: str | None = None) -> tuple[str, EngineCfg]:
        if not self.engines:
            raise KeyError("no engines configured (config [engines.*] or config/builds/*.toml [engine])")
        if name is None:
            name = "cuda_flash" if "cuda_flash" in self.engines else next(iter(self.engines))
        if name not in self.engines:
            raise KeyError(f"unknown engine {name!r}; known: {', '.join(self.engines)}")
        return name, self.engines[name]

    def model(self, name: str) -> tuple[str, ModelCfg]:
        """Model by config key, case-insensitive key, or alias."""
        if name in self.models:
            return name, self.models[name]
        low = name.lower()
        for k, m in self.models.items():
            if k.lower() == low or (m.alias and m.alias.lower() == low):
                return k, m
        raise KeyError(f"unknown model {name!r}; known: {', '.join(self.models) or '(none configured)'}")


# MSE_LLM_<FIELD> overrides [defaults].<field>; MSE_LLM_BENCHY_<FIELD> overrides [defaults.benchy].
def _env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    d = dict(data.get("defaults") or {})
    b = dict(d.get("benchy") or {})
    for section, fields, target in (
        ("MSE_LLM_BENCHY_", BenchyDefaults.model_fields, b),
        ("MSE_LLM_", Defaults.model_fields, d),
    ):
        for name, info in fields.items():
            if name == "benchy":
                continue
            raw = os.environ.get(section + name.upper())
            if raw is None:
                continue
            ann = str(info.annotation)
            if "list" in ann:
                target[name] = [int(x) for x in raw.replace(",", " ").split()]
            elif "bool" in ann:
                target[name] = raw.strip().lower() in ("1", "true", "yes", "on")
            else:
                target[name] = raw
    d["benchy"] = b
    return {**data, "defaults": d}


def config_path(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        return Path(explicit)
    env = os.environ.get("MSE_LLM_CONFIG")
    if env:
        return Path(env)
    cdir = project_root() / "config"
    for name in ("config.toml", "config.example.toml"):
        if (cdir / name).is_file():
            return cdir / name
    return None


def _engines_from_build_profiles() -> dict[str, dict[str, Any]]:
    from mse_llm.buildprofiles import list_profiles

    out: dict[str, dict[str, Any]] = {}
    for prof in list_profiles():
        eng = prof.engine
        if not eng.get("name"):
            continue
        out[eng["name"]] = {
            "runner": eng.get("runner", "llamacpp"),
            "bin_dir": str(prof.bin_dir),
            "host": eng.get("host", "0.0.0.0"),
            "port": eng.get("port", 8033),
            "build_profile": prof.name,
        }
        # Comparison / pinned-version builds (build-without-*, build-<variant>, build-<ref>) register
        # themselves through their build-manifest.json as "<engine>-<suffix>".
        default_bin = os.path.normcase(str(prof.bin_dir))
        for mpath in sorted(prof.src.glob("build-*/bin/*/build-manifest.json")) if prof.src.is_dir() else []:
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            name = m.get("engine")
            if not name or name in out or os.path.normcase(str(mpath.parent)) == default_bin:
                continue
            out[name] = {**out[eng["name"]], "bin_dir": str(mpath.parent)}
    return out


def load_config(explicit: str | Path | None = None) -> Config:
    path = config_path(explicit)
    data: dict[str, Any] = {}
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    engines = _engines_from_build_profiles()
    for name, eng in (data.get("engines") or {}).items():
        engines[name] = {**engines.get(name, {}), **eng}
    data["engines"] = engines
    cfg = Config.model_validate(_env_overrides(data))
    cfg.source = path
    return cfg
