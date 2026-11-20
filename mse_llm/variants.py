"""Variants (§4.2): a typed llama.cpp configuration, a two-way name codec, and matrix expansion.

A variant name is a list of `_`-separated tokens, for example `fa_Kq8_Vq4_b4096_ub4096_moe41_sd3cache`.
The legacy `extra_` prefix of the bash script (`EXTRAS`) is accepted, so old names and logs keep
working; `encode()` writes the canonical name without it.

Tokens:

| token | flags (llama-server) | notes |
|---|---|---|
| `fa` / `faoff` / `faauto` | `-fa on` / `off` / `auto` | bash wrote `-fa 1` |
| `K<t>` / `V<t>` | `-ctk` / `-ctv` | t: q4 q41 q5 q51 q8 iq4nl f16 bf16 f32 f8 (= f8_e4m3) |
| `default` | (nothing) | llama.cpp's own -b / -ub |
| `b<N>` / `ub<N>` | `-b N` / `-ub N` | |
| `moe<N>` / `cmoe` | `--n-cpu-moe N` / `-cmoe` | |
| `mc<MiB>` / `cs<pct>` | `--moe-cache-mib` / `--moe-cache-cpu-share pct/100` | server only |
| `lmN lmA lmD lmL lmM lmML` | `-lm none/auto/dio/mlock/mmap/mmap+mlock` | replaces [defaults].load_mode |
| `nohost` | `--no-host` | |
| `fitt<MiB>` | `-fitt MiB` | |
| `sd<N><type>` | `--spec-type T --spec-draft-n-max N` (+ `--model-draft` for draft types) | server only |
| `sdmin<N>` | `--spec-draft-n-min N` | with `sd*` |
| `dK<t>` / `dV<t>` | `-ctkd` / `-ctvd` (draft KV cache types) | with a draft type |
| `dmoe<N>` / `dcmoe` | `-ncmoed N` / `-cmoed` (drafter experts on the CPU) | with a draft type |
| `no_extra` | (nothing at all) | the bash warm-up entry |

turbo2/3/4 cache types (`Kt*`, `Vt*`) are dropped (not in upstream llama.cpp): decoding them
raises DroppedVariantError.
"""

from __future__ import annotations

import fnmatch
import itertools
import re
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal

Tool = Literal["server", "bench"]


class VariantError(ValueError):
    pass


class DroppedVariantError(VariantError):
    pass


class CapabilityError(VariantError):
    """A token that the target tool can't run (e.g. speculative decoding on llama-bench)."""


CACHE_TYPES = {
    "q4": "q4_0", "q41": "q4_1", "q5": "q5_0", "q51": "q5_1", "q8": "q8_0", "iq4nl": "iq4_nl",
    "f16": "f16", "bf16": "bf16", "f32": "f32", "f8": "f8_e4m3",
}
CACHE_TOKENS = {v: k for k, v in CACHE_TYPES.items()}

LOAD_MODES = {"N": "none", "A": "auto", "D": "dio", "L": "mlock", "M": "mmap", "ML": "mmap+mlock"}
LOAD_TOKENS = {v: k for k, v in LOAD_MODES.items()}

FA_TOKENS = {"fa": "on", "faoff": "off", "faauto": "auto"}
FA_BY_VALUE = {v: k for k, v in FA_TOKENS.items()}

SPEC_TYPES = {
    "mtp": "draft-mtp", "cache": "ngram-cache", "simple": "draft-simple", "eagle3": "draft-eagle3",
    "dflash": "draft-dflash", "dflash2": "draft-dflash2", "dspark": "draft-dspark", "lilicorr": "draft-lilicorr",
    "ngramsimple": "ngram-simple", "ngrammapk": "ngram-map-k", "ngrammapk4v": "ngram-map-k4v",
    "ngrammod": "ngram-mod",
}
SPEC_TOKENS = {v: k for k, v in SPEC_TYPES.items()}

LEGACY_PREFIX = "extra_"
EMPTY_NAME = "no_extra"
DRAFT_BUILTIN = "builtin"


@dataclass(frozen=True)
class Variant:
    fa: str | None = None                 # on | off | auto
    ctk: str | None = None
    ctv: str | None = None
    b: int | None = None
    ub: int | None = None
    n_cpu_moe: int | None = None
    cpu_moe: bool = False                 # -cmoe
    moe_cache_mib: int | None = None
    moe_cache_cpu_share: int | None = None   # percent, 0..100
    load_mode: str | None = None
    no_host: bool = False
    fit_target: int | None = None
    spec_type: str | None = None
    spec_n_max: int | None = None
    spec_n_min: int | None = None         # --spec-draft-n-min
    draft_ctk: str | None = None          # -ctkd
    draft_ctv: str | None = None          # -ctvd
    draft_n_cpu_moe: int | None = None    # -ncmoed
    draft_cpu_moe: bool = False           # -cmoed
    extra: tuple[str, ...] = field(default_factory=tuple)   # raw flags, not part of the name

    # ------------------------------------------------------------ codec

    @property
    def name(self) -> str:
        return encode(self)

    @property
    def is_empty(self) -> bool:
        return self == Variant()

    @property
    def uses_draft_model(self) -> bool:
        return bool(self.spec_type and self.spec_type.startswith("draft-"))

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, False, ())}

    def without_spec(self) -> Variant:
        """The main-model part only (llama-fit-params, llama-perplexity and llama-bench reject --spec-*)."""
        return replace(self, spec_type=None, spec_n_max=None, spec_n_min=None, draft_ctk=None, draft_ctv=None,
                       draft_n_cpu_moe=None, draft_cpu_moe=False)

    @property
    def has_draft_fields(self) -> bool:
        return any(x not in (None, False) for x in (self.spec_n_min, self.draft_ctk, self.draft_ctv,
                                                     self.draft_n_cpu_moe, self.draft_cpu_moe))

    # ------------------------------------------------------------ flags

    def flags(self, tool: Tool = "server", draft: str | None = None) -> tuple[list[str], list[str]]:
        """(argv fragment, warnings) for llama-server or llama-bench.

        `draft` is the model's drafter: a GGUF path, "builtin" (MTP heads inside the model) or None.
        Load mode is not included (the command builder merges it with [defaults].load_mode)."""
        out: list[str] = []
        warn: list[str] = []
        if tool == "bench":
            bad = [t for t, on in (("sd (speculative decoding)", self.spec_type or self.has_draft_fields),
                                   ("cmoe", self.cpu_moe),
                                   ("mc (--moe-cache-mib)", self.moe_cache_mib is not None),
                                   ("cs (--moe-cache-cpu-share)", self.moe_cache_cpu_share is not None)) if on]
            if bad:
                raise CapabilityError(f"{self.name}: llama-bench can't run {', '.join(bad)}; "
                                      "use `benchy` (llama-server + llama-benchy) for this variant")
        if self.fa:
            out += ["-fa", self.fa]
        if self.ctk:
            out += ["-ctk", self.ctk]
        if self.ctv:
            out += ["-ctv", self.ctv]
        if self.b is not None:
            out += ["-b", str(self.b)]
        if self.ub is not None:
            out += ["-ub", str(self.ub)]
        if self.n_cpu_moe is not None:
            out += ["--n-cpu-moe", str(self.n_cpu_moe)]
        if self.cpu_moe:
            out += ["-cmoe"]
        if self.moe_cache_mib is not None:
            out += ["--moe-cache-mib", str(self.moe_cache_mib)]
        if self.moe_cache_cpu_share is not None:
            out += ["--moe-cache-cpu-share", f"{self.moe_cache_cpu_share / 100:g}"]
        if self.no_host:
            out += ["--no-host"] + (["1"] if tool == "bench" else [])
        if self.fit_target is not None:
            out += ["-fitt", str(self.fit_target)]
        if self.spec_type:
            out += ["--spec-type", self.spec_type]
            if self.spec_n_max is not None:
                out += ["--spec-draft-n-max", str(self.spec_n_max)]
            if self.uses_draft_model:
                if draft and draft.lower() not in (DRAFT_BUILTIN, "build-in"):
                    out += ["--model-draft", draft]
                elif not draft:
                    warn.append(f"{self.spec_type}: no drafter configured for this model ([models.*].draft); "
                                "running without --model-draft (works only if the GGUF has built-in MTP heads)")
        if self.has_draft_fields and not self.spec_type:
            raise VariantError(f"{self.name}: sdmin / dK / dV / dmoe need a speculative type (sd<N><type>)")
        if self.spec_n_min is not None:
            out += ["--spec-draft-n-min", str(self.spec_n_min)]
        if self.draft_ctk:
            out += ["-ctkd", self.draft_ctk]
        if self.draft_ctv:
            out += ["-ctvd", self.draft_ctv]
        if self.draft_n_cpu_moe is not None:
            out += ["-ncmoed", str(self.draft_n_cpu_moe)]
        if self.draft_cpu_moe:
            out += ["-cmoed"]
        out += list(self.extra)
        return out, warn


# ---------------------------------------------------------------- decode / encode

_TOKEN_RES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"b(\d+)"), "b"),
    (re.compile(r"ub(\d+)"), "ub"),
    (re.compile(r"moe(\d+)"), "moe"),
    (re.compile(r"mc(\d+)"), "mc"),
    (re.compile(r"cs(\d+)"), "cs"),
    (re.compile(r"fitt(\d+)"), "fitt"),
    (re.compile(r"sdmin(\d+)"), "sdmin"),
    (re.compile(r"dmoe(\d+)"), "dmoe"),
    (re.compile(r"d([KV])([a-z0-9]+)"), "dkv"),
    (re.compile(r"sd(\d+)([a-z][a-z0-9]*)"), "sd"),
    (re.compile(r"lm([A-Z]+)"), "lm"),
    (re.compile(r"([KV])([a-z0-9]+)"), "kv"),
]


def strip_legacy(name: str) -> str:
    name = name.strip()
    return name[len(LEGACY_PREFIX):] if name.startswith(LEGACY_PREFIX) else name


def decode(name: str) -> Variant:
    """Variant from a name (canonical, or legacy with the `extra_` prefix)."""
    raw = strip_legacy(name)
    if raw in (EMPTY_NAME, ""):
        return Variant()
    v: dict[str, Any] = {}

    def put(key: str, value: Any, tok: str) -> None:
        if key in v and v[key] != value:
            raise VariantError(f"{name}: token {tok!r} conflicts with an earlier one ({key})")
        v[key] = value

    for tok in raw.split("_"):
        if not tok:
            raise VariantError(f"{name}: empty token (double '_')")
        if tok in FA_TOKENS:
            put("fa", FA_TOKENS[tok], tok)
            continue
        if tok == "default":
            continue
        if tok == "cmoe":
            put("cpu_moe", True, tok)
            continue
        if tok == "nohost":
            put("no_host", True, tok)
            continue
        if tok == "dcmoe":
            put("draft_cpu_moe", True, tok)
            continue
        for rx, kind in _TOKEN_RES:
            m = rx.fullmatch(tok)
            if not m:
                continue
            if kind == "b":
                put("b", int(m.group(1)), tok)
            elif kind == "ub":
                put("ub", int(m.group(1)), tok)
            elif kind == "moe":
                put("n_cpu_moe", int(m.group(1)), tok)
            elif kind == "mc":
                put("moe_cache_mib", int(m.group(1)), tok)
            elif kind == "cs":
                pct = int(m.group(1))
                if pct > 100:
                    raise VariantError(f"{name}: {tok}: CPU share is a percentage (0..100)")
                put("moe_cache_cpu_share", pct, tok)
            elif kind == "fitt":
                put("fit_target", int(m.group(1)), tok)
            elif kind == "sd":
                st = SPEC_TYPES.get(m.group(2))
                if st is None:
                    raise VariantError(f"{name}: {tok}: unknown speculative type {m.group(2)!r} "
                                       f"(known: {', '.join(SPEC_TYPES)})")
                put("spec_type", st, tok)
                put("spec_n_max", int(m.group(1)), tok)
            elif kind == "sdmin":
                put("spec_n_min", int(m.group(1)), tok)
            elif kind == "dmoe":
                put("draft_n_cpu_moe", int(m.group(1)), tok)
            elif kind == "dkv":
                ct = CACHE_TYPES.get(m.group(2))
                if ct is None:
                    raise VariantError(f"{name}: {tok}: unknown cache type {m.group(2)!r} "
                                       f"(known: {', '.join(CACHE_TYPES)})")
                put("draft_ctk" if m.group(1) == "K" else "draft_ctv", ct, tok)
            elif kind == "lm":
                lm = LOAD_MODES.get(m.group(1))
                if lm is None:
                    raise VariantError(f"{name}: {tok}: unknown load mode (known: "
                                       + ", ".join(f"lm{k}={x}" for k, x in LOAD_MODES.items()) + ")")
                put("load_mode", lm, tok)
            elif kind == "kv":
                side, t = m.group(1), m.group(2)
                if re.fullmatch(r"t\d", t):
                    raise DroppedVariantError(
                        f"{name}: {tok}: turbo{t[1:]} cache types are dropped (not in upstream llama.cpp; "
                        f"approved 2026-11-18). Use q8/q4/f8 instead, e.g. {side}q8")
                ct = CACHE_TYPES.get(t)
                if ct is None:
                    raise VariantError(f"{name}: {tok}: unknown cache type {t!r} (known: {', '.join(CACHE_TYPES)})")
                put("ctk" if side == "K" else "ctv", ct, tok)
            break
        else:
            raise VariantError(f"{name}: unknown token {tok!r}")
    return Variant(**v)


def encode(v: Variant) -> str:
    """Canonical name. Order: fa, K, V, batch (default | b/ub), moe/cmoe, mc, cs, lm, nohost, fitt, sd, sdmin,
    dK, dV, dmoe/dcmoe."""
    if v.is_empty:
        return EMPTY_NAME
    toks: list[str] = []
    if v.fa:
        toks.append(FA_BY_VALUE.get(v.fa, f"fa{v.fa}"))
    for side, ct in (("K", v.ctk), ("V", v.ctv)):
        if ct:
            if ct not in CACHE_TOKENS:
                raise VariantError(f"cache type {ct!r} has no token (known: {', '.join(CACHE_TOKENS)})")
            toks.append(side + CACHE_TOKENS[ct])
    if v.b is None and v.ub is None:
        toks.append("default")
    else:
        if v.b is not None:
            toks.append(f"b{v.b}")
        if v.ub is not None:
            toks.append(f"ub{v.ub}")
    if v.n_cpu_moe is not None:
        toks.append(f"moe{v.n_cpu_moe}")
    if v.cpu_moe:
        toks.append("cmoe")
    if v.moe_cache_mib is not None:
        toks.append(f"mc{v.moe_cache_mib}")
    if v.moe_cache_cpu_share is not None:
        toks.append(f"cs{v.moe_cache_cpu_share}")
    if v.load_mode:
        if v.load_mode not in LOAD_TOKENS:
            raise VariantError(f"load mode {v.load_mode!r} has no token")
        toks.append("lm" + LOAD_TOKENS[v.load_mode])
    if v.no_host:
        toks.append("nohost")
    if v.fit_target is not None:
        toks.append(f"fitt{v.fit_target}")
    if v.spec_type:
        if v.spec_type not in SPEC_TOKENS:
            raise VariantError(f"spec type {v.spec_type!r} has no token")
        toks.append(f"sd{v.spec_n_max if v.spec_n_max is not None else 3}{SPEC_TOKENS[v.spec_type]}")
    if v.spec_n_min is not None:
        toks.append(f"sdmin{v.spec_n_min}")
    for side, ct in (("K", v.draft_ctk), ("V", v.draft_ctv)):
        if ct:
            if ct not in CACHE_TOKENS:
                raise VariantError(f"cache type {ct!r} has no token (known: {', '.join(CACHE_TOKENS)})")
            toks.append("d" + side + CACHE_TOKENS[ct])
    if v.draft_n_cpu_moe is not None:
        toks.append(f"dmoe{v.draft_n_cpu_moe}")
    if v.draft_cpu_moe:
        toks.append("dcmoe")
    if v.extra:
        from mse_llm.util import stable_hash

        toks.append("x" + stable_hash(list(v.extra), 6))   # raw flags: name is not decodable back
    return "_".join(toks)


def canonical(name: str) -> str:
    return encode(decode(name))


# ---------------------------------------------------------------- flags -> variant

_FLAG_FIELDS = {
    "-fa": "fa", "--flash-attn": "fa",
    "-ctk": "ctk", "--cache-type-k": "ctk", "-ctv": "ctv", "--cache-type-v": "ctv",
    "-b": "b", "--batch-size": "b", "-ub": "ub", "--ubatch-size": "ub",
    "-ncmoe": "n_cpu_moe", "--n-cpu-moe": "n_cpu_moe",
    "--moe-cache-mib": "moe_cache_mib", "--moe-cache-cpu-share": "moe_cache_cpu_share",
    "-lm": "load_mode", "--load-mode": "load_mode",
    "-fitt": "fit_target", "--fit-target": "fit_target",
    "--spec-type": "spec_type", "--spec-draft-n-max": "spec_n_max", "--spec-draft-n-min": "spec_n_min",
    "-ctkd": "draft_ctk", "--cache-type-k-draft": "draft_ctk", "--spec-draft-type-k": "draft_ctk",
    "-ctvd": "draft_ctv", "--cache-type-v-draft": "draft_ctv", "--spec-draft-type-v": "draft_ctv",
    "-ncmoed": "draft_n_cpu_moe", "--n-cpu-moe-draft": "draft_n_cpu_moe",
    "--spec-draft-n-cpu-moe": "draft_n_cpu_moe", "--spec-draft-ncmoe": "draft_n_cpu_moe",
}
_BOOL_FLAGS = {"-cmoe": "cpu_moe", "--cpu-moe": "cpu_moe", "--no-host": "no_host",
               "-cmoed": "draft_cpu_moe", "--cpu-moe-draft": "draft_cpu_moe", "--spec-draft-cpu-moe": "draft_cpu_moe"}
_INT_FIELDS = {"b", "ub", "n_cpu_moe", "moe_cache_mib", "fit_target", "spec_n_max", "spec_n_min",
               "draft_n_cpu_moe"}
# flags that belong to the command, not the variant
_IGNORED_WITH_VALUE = {"-m", "--model", "-c", "--ctx-size", "-t", "--threads", "-np", "--parallel", "-ngl",
                       "--gpu-layers", "--n-gpu-layers", "--host", "--port", "-md", "--model-draft",
                       "--spec-draft-model", "-fit", "--fit", "-p", "-n", "-r", "-o"}


def from_flags(argv: list[str]) -> tuple[Variant, list[str]]:
    """Variant from llama.cpp flags (e.g. a preset or a pasted command line); returns the variant and
    the flags that aren't part of it (kept out of `extra`, so the variant stays decodable)."""
    v: dict[str, Any] = {}
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in _BOOL_FLAGS:
            v[_BOOL_FLAGS[a]] = True
            if i + 1 < len(argv) and argv[i + 1] in ("0", "1"):   # llama-bench --no-host 1
                v[_BOOL_FLAGS[a]] = argv[i + 1] == "1"
                i += 1
        elif a in _FLAG_FIELDS and i + 1 < len(argv):
            key, val = _FLAG_FIELDS[a], argv[i + 1]
            if key == "fa":
                val = {"1": "on", "0": "off"}.get(val, val)
            elif key == "moe_cache_cpu_share":
                val = round(float(val) * 100)
            elif key in _INT_FIELDS:
                val = int(val)
            v[key] = val
            i += 1
        elif a in _IGNORED_WITH_VALUE:
            i += 1
        else:
            rest.append(a)
        i += 1
    return Variant(**v), rest


# ---------------------------------------------------------------- patterns, lists, matrix

_BRACE = re.compile(r"\{([^{}]*)\}")


def _brace_items(body: str) -> list[str]:
    m = re.fullmatch(r"(-?\d+)\.\.(-?\d+)(?:\.\.(\d+))?", body)
    if m:
        a, b, step = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
        rng = range(a, b + 1, step) if a <= b else range(a, b - 1, -step)
        return [str(x) for x in rng]
    return body.split(",")


def expand(pattern: str) -> list[str]:
    """Bash-style brace expansion: `moe{18,21}_sd{3..5}mtp` -> 6 names (left brace varies slowest)."""
    pattern = pattern.strip()
    parts = _BRACE.split(pattern)      # literal, group, literal, group, ...
    if "{" in "".join(parts[0::2]) or "}" in "".join(parts[0::2]):
        raise VariantError(f"{pattern}: unbalanced or nested braces")
    choices = [[p] if i % 2 == 0 else _brace_items(p) for i, p in enumerate(parts)]
    return ["".join(c) for c in itertools.product(*choices)]


@dataclass
class ListEntry:
    name: str
    note: str = ""
    disabled: bool = False      # commented out in a pasted bash array


_QUOTED = re.compile(r"""^\s*(#\s*)?["']?([A-Za-z0-9_{},.]+)["']?\s*(?:#\s*(.*))?$""")


def parse_list(text: str, include_disabled: bool = False) -> list[ListEntry]:
    """Variant names from free text: one per line or whitespace separated, quoted or not, with bash array
    syntax (`local -a extras_list=(` ... `)`) and `#` comments. A commented-out entry is returned with
    disabled=True (only with include_disabled); a trailing comment becomes the entry's note."""
    out: list[ListEntry] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or re.match(r"^(local\s+-a\s+)?\w+=\($", line) or line in (")", "("):
            continue
        line = re.sub(r"^(local\s+-a\s+)?\w+=\(", "", line)
        if "#" not in line:
            line = line.rstrip(")").strip()        # `"a" "b")`: the array's closing paren on the same line
        m = _QUOTED.match(line)
        if m and (m.group(2).startswith((LEGACY_PREFIX, EMPTY_NAME)) or "_" in m.group(2)):
            disabled = bool(m.group(1))
            if disabled and not include_disabled:
                continue
            for name in expand(m.group(2)):
                out.append(ListEntry(name, (m.group(3) or "").strip(), disabled))
            continue
        if line.startswith("#"):
            continue
        for tok in line.split("#", 1)[0].split():
            tok = tok.strip("\"'")
            if tok:
                out.extend(ListEntry(n) for n in expand(tok))
    return out


_MATRIX_FIELDS = {f.name for f in fields(Variant)} - {"extra"}


def expand_matrix(matrix: dict[str, Any]) -> list[Variant]:
    """Cartesian product over Variant fields, e.g.
    {fa=["on"], ctk=["q8_0"], ctv=["q8_0","q4_0"], b_ub=[[2048,2048],[4096,4096]], n_cpu_moe=[18,21],
     exclude=["*_b4096_ub4096_moe18"]}.
    `b_ub` sets b and ub together; `exclude` takes glob patterns matched against the canonical name."""
    matrix = dict(matrix)
    exclude = [str(x) for x in matrix.pop("exclude", [])]
    base = matrix.pop("base", None)
    start = decode(base) if base else Variant()
    keys = list(matrix)
    for k in keys:
        if k not in _MATRIX_FIELDS and k != "b_ub":
            raise VariantError(f"matrix: unknown axis {k!r} (known: b_ub, {', '.join(sorted(_MATRIX_FIELDS))})")
        if not isinstance(matrix[k], list):
            matrix[k] = [matrix[k]]
    out: list[Variant] = []
    seen: set[str] = set()
    for combo in itertools.product(*(matrix[k] for k in keys)):
        kw: dict[str, Any] = {}
        for k, val in zip(keys, combo, strict=True):
            if k == "b_ub":
                kw["b"], kw["ub"] = (None, None) if val in (None, "default") else (int(val[0]), int(val[1]))
            else:
                kw[k] = val
        v = replace(start, **kw)
        n = v.name
        if n in seen or any(fnmatch.fnmatchcase(n, p) for p in exclude):
            continue
        seen.add(n)
        out.append(v)
    return out


def resolve(names: list[str] | None = None, matrix: dict[str, Any] | None = None) -> list[Variant]:
    """Variants of a sweep or a command line: brace-expanded names, then the matrix, de-duplicated."""
    out: list[Variant] = []
    seen: set[str] = set()
    for pat in names or []:
        for n in expand(pat):
            v = decode(n)
            if v.name not in seen:
                seen.add(v.name)
                out.append(v)
    for v in expand_matrix(matrix) if matrix else []:
        if v.name not in seen:
            seen.add(v.name)
            out.append(v)
    return out
