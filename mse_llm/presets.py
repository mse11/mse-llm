"""Router presets INI (§4.3): parse, validate, fix, dump (preset -> single-model argv), generate.

The llama.cpp router (`llama-server --models-preset FILE`) reads an INI file: `[*]` holds the global
options, every other section is a model whose name is its id in requests. Keys are llama-server
options without the dashes (long or short spelling), values as on the command line, flags as
true/false (or 1/0). Precedence: command line > model section > `[*]`.

Comments (`;` and `#`) are kept by `fix`; only the lines that need a change are rewritten.
"""

from __future__ import annotations

import difflib
import ntpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mse_llm import helpsnap as hs
from mse_llm import variants as vv
from mse_llm.config import Config

GLOBAL = "*"
# keys only the router knows (not llama-server flags)
ROUTER_ONLY = {"version", "load-on-startup", "stop-timeout", "dedup-cache-models"}
# keys the router owns: ignored (or overwritten) when they appear in the INI
ROUTER_OWNED = {"host", "port", "api-key", "hf-repo"}
TRUE = {"1", "true", "on", "yes"}
FALSE = {"0", "false", "off", "no"}

Level = Literal["error", "warn", "info", "ok"]


@dataclass
class Line:
    no: int                  # 1-based
    raw: str
    section: str | None      # section the line belongs to (None before the first one)
    key: str | None = None
    value: str | None = None
    header: str | None = None   # for a [section] line


@dataclass
class Section:
    name: str
    line: int
    keys: dict[str, str] = field(default_factory=dict)
    key_lines: dict[str, int] = field(default_factory=dict)

    @property
    def model(self) -> str | None:
        return self.keys.get("model") or self.keys.get("m")


@dataclass
class PresetsFile:
    path: Path
    lines: list[Line]
    top: dict[str, str]                  # keys before the first section (version = 1)
    sections: dict[str, Section]

    @property
    def globals(self) -> dict[str, str]:
        return self.sections[GLOBAL].keys if GLOBAL in self.sections else {}

    @property
    def models(self) -> list[Section]:
        return [s for n, s in self.sections.items() if n != GLOBAL]

    def merged(self, name: str) -> dict[str, str]:
        """[*] + section (section wins), in first-seen order."""
        sec = self.find(name)
        return {**self.globals, **sec.keys}

    def find(self, name: str) -> Section:
        if name in self.sections and name != GLOBAL:
            return self.sections[name]
        low = name.lower()
        for s in self.models:
            if s.name.lower() == low or (s.keys.get("alias") or "").lower() == low:
                return s
        raise KeyError(f"no preset {name!r} in {self.path.name}; known: {', '.join(s.name for s in self.models)}")


_SECTION = re.compile(r"^\s*\[(.+)\]\s*$")


def parse_text(text: str, path: Path) -> PresetsFile:
    lines: list[Line] = []
    top: dict[str, str] = {}
    sections: dict[str, Section] = {}
    cur: str | None = None
    for no, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if not s or s[0] in ";#":
            lines.append(Line(no, raw, cur))
            continue
        m = _SECTION.match(raw)
        if m:
            cur = m.group(1).strip()
            sections.setdefault(cur, Section(cur, no))
            lines.append(Line(no, raw, cur, header=cur))
            continue
        if "=" not in s:
            lines.append(Line(no, raw, cur))
            continue
        k, v = (x.strip() for x in s.split("=", 1))
        lines.append(Line(no, raw, cur, key=k, value=v))
        if cur is None:
            top[k] = v
        else:
            sections[cur].keys[k] = v
            sections[cur].key_lines[k] = no
    return PresetsFile(path, lines, top, sections)


def parse(path: str | Path) -> PresetsFile:
    p = Path(path)
    return parse_text(p.read_text(encoding="utf-8", errors="replace"), p)


# ---------------------------------------------------------------- key -> flag

def flag_for(key: str, server: hs.HelpInfo | None) -> str | None:
    """INI key -> llama-server flag (`--key` or `-key`, whichever --help knows). None if unknown."""
    if server is None:
        return f"--{key}"
    for f in (f"--{key}", f"-{key}"):
        if f in server:
            return f
    return None


def _takes_value(flag: str, server: hs.HelpInfo | None, value: str | None = None) -> bool:
    """Whether `flag VALUE` (True) or a bare flag / --no-flag (False) is meant. Flags with an optional value
    (`-fa [on|off|auto]`) take the value unless it is a plain boolean word that isn't one of their values."""
    if server is None or flag not in server:
        return True
    fi = server.flags[flag]
    if not fi.metavar:
        return False
    if not fi.metavar.startswith("["):
        return True
    if value is None:
        return True
    v = value.lower()
    return v in (fi.values or []) or v not in TRUE | FALSE


# ---------------------------------------------------------------- validate / fix

@dataclass
class Finding:
    level: Level
    where: str              # "[section]", "(file)", "every section"
    message: str
    line: int | None = None
    fix: str | None = None  # replacement text for `line` (or the line to insert at the top when line is None)

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "where": self.where, "message": self.message, "line": self.line, "fix": self.fix}


@dataclass
class Validation:
    path: Path
    snapshot: str | None
    findings: list[Finding]

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warn"]

    @property
    def fixable(self) -> list[Finding]:
        return [f for f in self.findings if f.fix is not None]

    def as_dict(self) -> dict[str, Any]:
        return {"file": str(self.path), "snapshot": self.snapshot, "errors": len(self.errors),
                "warnings": len(self.warnings), "findings": [f.as_dict() for f in self.findings]}


def _value_error(flag: str, value: str, server: hs.HelpInfo) -> str | None:
    fi = server.flags.get(flag)
    if not fi or not fi.values:
        return None
    bad = [v for v in (x.strip() for x in value.split(",")) if v not in fi.values]
    return f"{', '.join(bad)} not in [{', '.join(fi.values)}]" if bad else None


def _path_exists(p: str, base: Path) -> bool:
    q = Path(p)
    return (q if q.is_absolute() or ntpath.isabs(p) else base / q).is_file()


def validate(pf: PresetsFile, server: hs.HelpInfo | None, snapshot: str | None = None) -> Validation:
    out: list[Finding] = []
    base = Path.cwd()
    if "version" not in pf.top:
        out.append(Finding("warn", "(file)", 'no "version = 1" line', None, "version = 1"))
    if GLOBAL not in pf.sections:
        out.append(Finding("warn", "(file)", "no [*] section for the global options"))
    known_ok: set[str] = set()
    aliases = 0
    models_found = 0
    for sec in pf.sections.values():
        w = f"[{sec.name}]"
        low = sec.name.lower()
        if sec.name != GLOBAL and ("global" in low or "default" in low) and not sec.model:
            out.append(Finding("error", w, "section name: globals must be [*]; this becomes a model preset "
                               "without a model", sec.line, "[*]"))
        elif sec.name != GLOBAL and not sec.model and "hf-repo" not in sec.keys:
            out.append(Finding("error", w, "no model = PATH (or hf-repo): the router can't load it"))
        for key, value in sec.keys.items():
            no = sec.key_lines[key]
            if key in ROUTER_ONLY or key.startswith("LLAMA_ARG_"):
                continue
            if key in ROUTER_OWNED:
                out.append(Finding("warn", w, f"{key} = {value}: owned by the router, ignored here "
                                   "(set it on the command line)", no))
                continue
            if key == "alias":
                aliases += 1
                continue
            if key in ("model", "m", "model-draft", "md", "mmproj"):
                if not _path_exists(value, base):
                    out.append(Finding("error", w, f"{key} = {value}: file not found", no))
                elif key in ("model", "m"):
                    models_found += 1
                continue
            flag = flag_for(key, server)
            if flag is None:
                if key in ("no-mmap", "mmap"):
                    out.append(Finding("error", w, f"{key} = {value}   flag removed in this build "
                                       f"(\"invalid argument: --{key}\") → load-mode = none", no, "load-mode = none"))
                else:
                    out.append(Finding("error", w, f"{key} = {value}: no llama-server option --{key} / -{key} "
                                       "(router fails with 'option not recognized')", no))
                continue
            if server is not None and _takes_value(flag, server):
                if key in ("fa", "flash-attn") and value in ("1", "0"):
                    new = "on" if value == "1" else "off"
                    out.append(Finding("warn", w, f"{key} = {value}   accepted, but the flag now takes "
                                       f"on|off|auto → {key} = {new}", no, f"{key} = {new}"))
                    continue
                err = _value_error(flag, value, server)
                if err:
                    out.append(Finding("error", w, f"{key} = {value}: {err}", no))
                    continue
            if key in ("ngl", "n-gpu-layers", "gpu-layers") and value in ("999", "99"):
                out.append(Finding("warn", w, f"{key} = {value}   ok; 'all' is the explicit spelling", no))
                continue
            if key == "jinja" and value.lower() in TRUE:
                out.append(Finding("warn", w, f"{key} = {value}   already the default (enabled)", no))
                continue
            known_ok.add(key)
    if aliases:
        out.append(Finding("info", "every section" if aliases == len(pf.models) else f"{aliases} sections",
                           "alias = …   used by the router as an extra model name; the section name is the main id"))
    if models_found:
        out.append(Finding("ok", "", f"{models_found} model file{'s' if models_found != 1 else ''} found"))
    if known_ok:
        out.append(Finding("ok", "", f"all other keys exist in llama-server --help ({', '.join(sorted(known_ok))})"))
    return Validation(pf.path, snapshot, out)


def fixed_text(pf: PresetsFile, val: Validation) -> str:
    """The file with every finding that has a fix applied (comments and other lines untouched)."""
    repl = {f.line: f.fix for f in val.fixable if f.line is not None}
    top = [f.fix for f in val.fixable if f.line is None]
    out: list[str] = []
    for ln in pf.lines:
        if ln.no in repl:
            raw = ln.raw
            indent = raw[: len(raw) - len(raw.lstrip())]
            out.append(indent + str(repl[ln.no]))
        else:
            out.append(ln.raw)
    if top:
        out = [*top, "", *out]
    return "\n".join(out) + "\n"


def diff(pf: PresetsFile, new_text: str) -> str:
    old = pf.path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    return "".join(difflib.unified_diff(old, new_text.splitlines(keepends=True), str(pf.path.name),
                                        str(pf.path.name) + " (fixed)"))


# ---------------------------------------------------------------- dump: preset -> single-model argv

@dataclass
class Dump:
    name: str
    flags: list[str]                     # llama-server flags, without the binary
    variant: str
    model_key: str | None = None         # matching [models.*] by path
    default_serve: str | None = None
    skipped: list[str] = field(default_factory=list)   # router-only / owned keys left out
    model_path: str | None = None
    ctx: int | None = None

    @property
    def matches_default(self) -> bool | None:
        if not self.default_serve:
            return None
        try:
            return vv.canonical(self.default_serve) == self.variant
        except vv.VariantError:
            return False


def preset_flags(merged: dict[str, str], server: hs.HelpInfo | None) -> tuple[list[str], list[str]]:
    """Merged INI keys -> llama-server flags (model first). Returns (flags, skipped keys)."""
    out: list[str] = []
    skipped: list[str] = []
    model = merged.get("model") or merged.get("m")
    if model:
        out += ["-m", model]
    for key, value in merged.items():
        if key in ("model", "m"):
            continue
        if key in ROUTER_ONLY or key in ROUTER_OWNED or key.startswith("LLAMA_ARG_"):
            skipped.append(key)
            continue
        flag = flag_for(key, server)
        if flag is None:
            skipped.append(key)
            continue
        if not _takes_value(flag, server, value):
            v = value.lower()
            if v in FALSE:
                neg = "--no-" + flag.lstrip("-")
                if server is None or neg in server:
                    out.append(neg)
            else:
                out.append(flag)
            continue
        out += [flag, value]
    return out, skipped


def _norm(p: str) -> str:
    return ntpath.normcase(ntpath.normpath(p.replace("/", "\\")))


def dump(pf: PresetsFile, name: str, server: hs.HelpInfo | None, cfg: Config | None = None) -> Dump:
    sec = pf.find(name)
    merged = pf.merged(sec.name)
    flags, skipped = preset_flags(merged, server)
    v, _ = vv.from_flags(flags)
    if cfg is not None:
        from mse_llm.serve import variant_without_defaults

        v = variant_without_defaults(v, cfg)
    d = Dump(sec.name, flags, v.name, skipped=skipped, model_path=sec.model or merged.get("model"))
    for k in ("ctx-size", "c"):
        if k in merged:
            try:
                d.ctx = int(merged[k])
            except ValueError:
                pass
            break
    if cfg is not None and d.model_path:
        for key, m in cfg.models.items():
            if _norm(m.path) == _norm(d.model_path):
                d.model_key, d.default_serve = key, m.default_serve
                break
    return d


# ---------------------------------------------------------------- generate

def _flags_to_keys(flags: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(flags):
        f = flags[i]
        nxt = flags[i + 1] if i + 1 < len(flags) else None
        key = f.lstrip("-")
        if nxt is not None and (not nxt.startswith("-") or re.fullmatch(r"-?\d+(\.\d+)?", nxt)):
            out.append((key, nxt))
            i += 2
        else:
            out.append((key, "true"))
            i += 1
    return out


def _ini_value(v: Any) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, list):
        return ",".join(map(str, v))
    return str(v)


def generate(cfg: Config, engine: str | None = None) -> tuple[str, list[str]]:
    """INI text from [models.*] + each model's default_serve variant. Returns (text, warnings)."""
    from mse_llm.runners import display_argv, server_command

    d = cfg.defaults
    warnings: list[str] = []
    lines = [
        "version = 1",
        "",
        "; Generated by `mse-llm presets generate` from config [models.*] + default_serve.",
        "; Router-owned keys (host, port, api-key) go on the command line, not here.",
        "; Clients select a model by its SECTION name; `alias` adds extra names for it.",
        "[*]",
        f"threads = {d.threads}",
        f"parallel = {d.parallel}",
        f"ngl = {d.ngl}",
    ]
    if d.load_mode:
        lines.append(f"load-mode = {d.load_mode}")
    for key, m in cfg.models.items():
        if not m.default_serve:
            continue
        try:
            v = vv.decode(m.default_serve)
            vflags, warn = v.flags("server", draft=m.draft)
        except vv.VariantError as e:
            warnings.append(f"{key}: {e}")
            continue
        warnings += [f"{key}: {w}" for w in warn]
        name = m.alias or m.label
        cmd = server_command(cfg, key, v, None, engine)
        lines += ["", f"[{name}]", f"# mse-llm serve {key}  ({v.name})",
                  "# " + display_argv(cmd.argv[1:]), f"model = {m.path}"]
        if m.alias:
            lines.append(f"alias = {m.alias}")
        if m.ctx:
            lines.append(f"ctx-size = {m.ctx}")
        seen = {"model", "alias", "ctx-size"}
        for k, val in _flags_to_keys(vflags):
            if k not in seen:
                seen.add(k)
                lines.append(f"{k} = {val}")
        for k, val in m.serve_extra.items():
            k = k.lstrip("-")
            if k not in seen:
                seen.add(k)
                lines.append(f"{k} = {_ini_value(val)}")
    return "\n".join(lines) + "\n", warnings
