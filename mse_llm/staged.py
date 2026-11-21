"""Staged sweeps (§4.4, M11): each stage fixes the winner of the stages before it.

    [sweeps.NAME]
    stages = [
      ["fa_Kq8_Vq8_default_moe34_lm{N,A,D}"],                         # 0. baselines, each load mode
      ["fa_Kq8_Vq8_default_cmoe_mc{2048,4096,6144}_cs{0,95}_lm{N,D}"],  # 1. cache size at share 0 / 0.95
      ["fa_Kq8_Vq8_default_cmoe_mc{best}_cs{0,50,80,90,95}_lm{best}"],  # 2. share at the best size + load mode
      ["{winner}_K{q8,f8}_V{q8,f8,q4}"],                              # 3. K/V types on top of the winner
    ]

Placeholders, resolved from the stages already run:
* `<prefix>{best}`: the value of that setting in the newest stage winner that sets it. Prefixes: `mc`, `cs`, `lm`,
  `moe`, `b`, `ub`, `K`, `V`, `sd` (`sd{best}mtp` = the draft length).
* `<prefix>{tuned}` (`mc`, `cs`): the last `tune moe-cache` recommendation for the model.
* `{winner}` as a whole token: the previous stage's winner, with the pattern's other tokens on top
  (`{winner}_Kf8_Vf8` = the winner with f8 K/V).
Everything else is normal brace expansion (`{2048,4096}`, `{1..4}`).

A stage's winner is its best `ok` run by the sweep's `goal` (tg, pp, or both = geometric mean). Runs that fail,
spill, are skipped by the fit check or mark an invalid MoE cache config (`invalid config:`) never win. Each run is a
normal RunResult (sweep NAME, `stage` = index / pattern); the stage winners are written to
results/staged/<sweep>/<build>__<hw>.json.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from mse_llm import results as rs
from mse_llm.variants import CACHE_TOKENS, LOAD_TOKENS, Variant, VariantError, decode, expand

# prefix -> (Variant field, token text of a value)
_BEST_FIELDS: dict[str, tuple[str, Any]] = {
    "mc": ("moe_cache_mib", str), "cs": ("moe_cache_cpu_share", str), "moe": ("n_cpu_moe", str),
    "ub": ("ub", str), "b": ("b", str), "sd": ("spec_n_max", str),
    "lm": ("load_mode", lambda x: LOAD_TOKENS[x]), "K": ("ctk", lambda x: CACHE_TOKENS[x]),
    "V": ("ctv", lambda x: CACHE_TOKENS[x]),
}
_PLACEHOLDER = re.compile(r"(mc|cs|moe|ub|b|sd|lm|K|V)?\{(best|tuned)\}")
_WINNER = re.compile(r"(?:^|_)\{winner\}(?=_|$)")


class StageError(ValueError):
    pass


class Unresolved(StageError):
    """A placeholder needs a winner that doesn't exist yet (the stage before hasn't run)."""


@dataclass
class StageRun:
    variant: str
    status: str
    tg: float | None = None
    pp: float | None = None
    run_id: str | None = None
    reused: bool = False
    reason: str | None = None
    hit_rate: float | None = None
    vram_peak_mib: float | None = None

    def metric(self, goal: str) -> float:
        tg, pp = self.tg or 0.0, self.pp or 0.0
        return tg if goal == "tg" else pp if goal == "pp" else math.sqrt(tg * pp)


@dataclass
class StageState:
    index: int
    patterns: list[str]
    variants: list[str] = field(default_factory=list)
    runs: list[StageRun] = field(default_factory=list)
    winner: str | None = None
    unresolved: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "patterns": self.patterns, "variants": self.variants, "winner": self.winner,
                "unresolved": self.unresolved, "runs": [r.__dict__ for r in self.runs]}


# ---------------------------------------------------------------- resolving patterns

def _value(prefix: str, v: Variant) -> str | None:
    fld, fmt = _BEST_FIELDS[prefix]
    x = getattr(v, fld)
    if x is None:
        return None
    try:
        return fmt(x)
    except KeyError:
        return None


def resolve(pattern: str, winners: list[Variant], tuned: Variant | None = None) -> list[str]:
    """Variant names of one stage pattern, given the winners of the stages before (oldest first)."""
    def sub(m: re.Match[str]) -> str:
        prefix, kind = m.group(1), m.group(2)
        if not prefix:
            raise StageError(f"{pattern}: {{{kind}}} needs a setting prefix (mc, cs, lm, moe, b, ub, K, V, sd)")
        if kind == "tuned":
            if prefix not in ("mc", "cs"):
                raise StageError(f"{pattern}: {prefix}{{tuned}}: only mc and cs come from `tune moe-cache`")
            if tuned is None:
                raise Unresolved(f"{pattern}: {prefix}{{tuned}}: no `tune moe-cache` recommendation for this model")
            val = _value(prefix, tuned)
        else:
            val = next((x for x in (_value(prefix, w) for w in reversed(winners)) if x is not None), None)
            if val is None:
                if not winners:
                    raise Unresolved(f"{pattern}: {prefix}{{best}} waits for an earlier stage's winner")
                raise StageError(f"{pattern}: {prefix}{{best}}: no earlier stage winner sets {prefix}")
        return prefix + val

    text = _PLACEHOLDER.sub(sub, pattern)
    with_winner = bool(_WINNER.search(text))
    if with_winner:
        if not winners:
            raise Unresolved(f"{pattern}: {{winner}} waits for the previous stage")
        text = _WINNER.sub("", text).strip("_")
    out: list[str] = []
    for name in expand(text) if text else [""]:
        try:
            v = _merge(winners[-1], name) if with_winner else decode(name)
        except VariantError as e:
            raise StageError(f"{pattern}: {e}") from e
        if v.name not in out:
            out.append(v.name)
    return out


def _merge(base: Variant, name: str) -> Variant:
    """base with the tokens of `name` on top (`default` resets b / ub)."""
    if not name:
        return base
    over = decode(name)
    blank = Variant()
    kw = {k: getattr(over, k) for k in over.__dataclass_fields__ if getattr(over, k) != getattr(blank, k)}
    if "default" in name.split("_"):
        kw.update(b=None, ub=None)
    if over.cpu_moe:
        kw["n_cpu_moe"] = None
    elif over.n_cpu_moe is not None:
        kw["cpu_moe"] = False
    return replace(base, **kw)


def pick_winner(runs: list[StageRun], goal: str) -> StageRun | None:
    ok = [r for r in runs if r.status == "ok" and (r.tg or r.pp)]
    return max(ok, key=lambda r: r.metric(goal)) if ok else None


# ---------------------------------------------------------------- tune recommendation

def tuned_path(model: str, root: Path | None = None) -> Path:
    return (root or rs.results_root()) / "tune" / f"{model}.json"


def load_tuned(model: str) -> Variant | None:
    p = tuned_path(model)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return decode(d["variant"])
    except (OSError, ValueError, KeyError, VariantError):
        return None


# ---------------------------------------------------------------- records

def run_of(r: rs.RunResult | dict[str, Any], reused: bool = False) -> StageRun:
    if isinstance(r, dict):
        return StageRun(r["variant"], r.get("status") or "?", r.get("tg_tps"), r.get("pp_tps"), r.get("run_id"),
                        reused, r.get("reason"), r.get("hit_rate"), r.get("vram_peak_mib"))
    m = r.metrics
    vram = (((r.monitor or {}).get("stats") or {}).get("vram_used_mib") or {}).get("peak")
    return StageRun(r.variant, r.status, m.get("tg_tps"), m.get("pp_tps"), r.run_id, reused, r.reason,
                    (r.moe_cache or {}).get("hit_rate"), vram)


def summary_path(sweep: str, build: str | None, hw: str | None, root: Path | None = None) -> Path:
    return (root or rs.results_root()) / "staged" / sweep / f"{build or 'nobuild'}__{hw or 'nohw'}.json"


def write_summary(sweep: str, model: str, goal: str, stages: list[StageState], build: str | None,
                  hw: str | None) -> Path:
    best = pick_winner([r for s in stages for r in s.runs], goal)
    p = summary_path(sweep, build, hw)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"sweep": sweep, "model": model, "goal": goal, "build": build, "hw": hw,
                             "timestamp": rs.now().isoformat(timespec="seconds"),
                             "winner": best.variant if best else None, "winner_run": best.__dict__ if best else None,
                             "stages": [s.as_dict() for s in stages]}, indent=2) + "\n", encoding="utf-8")
    return p
