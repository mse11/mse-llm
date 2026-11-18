"""Minimal build-profile loader (config/builds/*.toml, §0.4).

M0/M1 only need: the list of profiles, `extends` merging, path expansion, the [target] matchers and
the [engine] table with its bin_dir. The full `mse-llm build` (M2) builds on this.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mse_llm.util import project_root


def builds_config_dir() -> Path:
    return project_root() / "config" / "builds"


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_raw(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"extends cycle: {' -> '.join(str(p) for p in (*_seen, path))}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    parent = data.pop("extends", None)
    if parent:
        base = load_raw(path.parent / parent, (*_seen, path))
        data = deep_merge(base, data)
    return data


def builds_root(data: dict[str, Any] | None = None) -> Path:
    env = os.environ.get("MSE_LLM_BUILDS_ROOT")
    if env:
        return Path(env)
    tmpl = ((data or {}).get("paths") or {}).get("builds_root", "{project}/builds")
    return Path(tmpl.replace("{project}", str(project_root())))


@dataclass
class BuildProfile:
    name: str
    path: Path
    data: dict[str, Any]
    builds_root: Path
    src: Path
    build_dir: Path
    bin_dir: Path
    gpu_match: str | None = None
    cpu_match: str | None = None
    engine: dict[str, Any] = field(default_factory=dict)


def _expand(s: str, ctx: dict[str, str]) -> str:
    for k, v in ctx.items():
        s = s.replace("{" + k + "}", v)
    return s


def load_profile(name_or_path: str | Path) -> BuildProfile:
    p = Path(name_or_path)
    if not p.suffix:
        p = builds_config_dir() / f"{name_or_path}.toml"
    data = load_raw(p)
    root = builds_root(data)
    ctx = {"project": str(project_root()), "builds_root": str(root)}
    src = Path(_expand((data.get("source") or {}).get("src", "{builds_root}/llama.cpp"), ctx))
    build_dir = src / data.get("build_dir", "build")
    ctx["build_dir"] = str(build_dir)
    bin_dir = Path(_expand((data.get("outputs") or {}).get("bin_dir", "{build_dir}/bin/Release"), ctx))
    target = data.get("target") or {}
    return BuildProfile(
        name=data.get("name", p.stem),
        path=p,
        data=data,
        builds_root=root,
        src=src,
        build_dir=build_dir,
        bin_dir=bin_dir,
        gpu_match=target.get("gpu_match"),
        cpu_match=target.get("cpu_match"),
        engine=dict(data.get("engine") or {}),
    )


def list_profiles() -> list[BuildProfile]:
    d = builds_config_dir()
    if not d.is_dir():
        return []
    return [load_profile(f) for f in sorted(d.glob("*.toml")) if not f.name.startswith("_")]
