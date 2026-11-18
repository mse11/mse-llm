"""llama.cpp --help snapshots: parse, store, diff, and audit the flags used by the bash script and INI.

Snapshots live in tests/fixtures/help/<id>/ (llama-server.txt, llama-bench.txt, version.txt,
meta.json), where <id> is the commit from `llama-server --version` (e.g. a49db739f). `doctor flags`
captures the current build's help, diffs it against the stored snapshot and checks every flag and
cache-type value used by llama_cpp_benchmark.sh and llama_cpp_presets.ini.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from mse_llm.util import exe, project_root, run

TOOLS = ("llama-server", "llama-bench")


def snapshots_dir() -> Path:
    return project_root() / "tests" / "fixtures" / "help"


@dataclass
class FlagInfo:
    names: list[str]
    metavar: str = ""
    values: list[str] | None = None
    text: str = ""

    @property
    def key(self) -> str:
        longs = [n for n in self.names if n.startswith("--")]
        return longs[0] if longs else self.names[0]


@dataclass
class HelpInfo:
    flags: dict[str, FlagInfo] = field(default_factory=dict)   # every name -> FlagInfo

    def unique(self) -> dict[str, FlagInfo]:
        return {f.key: f for f in self.flags.values()}

    def __contains__(self, name: str) -> bool:
        return name in self.flags


def _enum_from(spec: str, text: str) -> list[str] | None:
    m = re.search(r"allowed values:\s*([^\n(]+)", text, re.I)
    if m:
        return [v.strip() for v in m.group(1).split(",") if v.strip()]
    m = re.search(r"[\[{<]([\w.-]+(?:[|,][\w.-]+)+)[\]}>]", spec)
    if m:
        return [v for v in re.split(r"[|,]", m.group(1)) if v]
    return None


_METAVAR = re.compile(r"[A-Z][A-Z0-9_|,-]*|[\[{<].*[\]}>]")


def _split_spec(s: str) -> tuple[list[str], str, str]:
    """'-ctk,  --cache-type-k TYPE   KV cache ...' -> (['-ctk', '--cache-type-k'], 'TYPE', 'KV cache ...')"""
    names: list[str] = []
    metavar = ""
    pos = 0
    for m in re.finditer(r"\S+", s):
        tok = m.group(0)
        bare = tok.rstrip(",")
        if re.fullmatch(r"--?[A-Za-z][\w.-]*", bare) and not metavar:
            names.append(bare)
        elif names and not metavar and _METAVAR.fullmatch(bare):
            metavar = bare
        else:
            break
        pos = m.end()
    return names, metavar, s[pos:].strip()


def parse_help(text: str) -> HelpInfo:
    """Parse llama.cpp style help: a flag line starts with -x / --xxx (after indentation), the
    description continues on the following, more indented, non-flag lines."""
    info = HelpInfo()
    cur: FlagInfo | None = None
    desc: list[str] = []

    def flush() -> None:
        nonlocal cur, desc
        if cur is not None:
            cur.text = " ".join(desc).strip()
            cur.values = _enum_from(cur.metavar, cur.text)
            for n in cur.names:
                info.flags.setdefault(n, cur)
        cur, desc = None, []

    for line in text.splitlines():
        s = line.strip()
        if re.match(r"--?[A-Za-z]", s) and len(line) - len(line.lstrip()) <= 8:
            flush()
            names, metavar, rest = _split_spec(s)
            if not names:
                continue
            cur = FlagInfo(names=names, metavar=metavar)
            if rest:
                desc.append(rest)
        elif cur is not None and s:
            desc.append(s)
        elif not s:
            continue
    flush()
    return info


# ---------------------------------------------------------------- snapshots

def version_id(version_text: str) -> str | None:
    m = re.search(r"version:\s*\d+\s*\(([0-9a-f]{6,40})\)", version_text)
    return m.group(1) if m else None


def capture(bin_dir: Path, dest_root: Path | None = None) -> tuple[str, Path]:
    """Run --version/--help of the binaries in bin_dir and store them as a snapshot."""
    ver = run([str(exe(bin_dir / "llama-server")), "--version"], timeout=30)
    if ver.error:
        raise RuntimeError(f"llama-server --version: {ver.error}")
    vtext = ver.stdout + ver.stderr
    sid = version_id(vtext) or _dt.datetime.now().strftime("unknown-%Y%m%d-%H%M%S")
    dest = (dest_root or snapshots_dir()) / sid
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "version.txt").write_text(vtext, encoding="utf-8")
    for tool in TOOLS:
        r = run([str(exe(bin_dir / tool)), "--help"], timeout=30)
        if r.error:
            raise RuntimeError(f"{tool} --help: {r.error}")
        (dest / f"{tool}.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
    meta = {"id": sid, "bin_dir": str(bin_dir),
            "captured_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")}
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return sid, dest


def list_snapshots(root: Path | None = None) -> list[Path]:
    root = root or snapshots_dir()
    if not root.is_dir():
        return []
    snaps = [d for d in root.iterdir() if (d / "llama-server.txt").is_file()]

    def when(d: Path) -> str:
        try:
            return json.loads((d / "meta.json").read_text(encoding="utf-8")).get("captured_at", "")
        except (OSError, ValueError):
            return ""

    return sorted(snaps, key=when)


def load_snapshot(path: Path) -> dict[str, HelpInfo]:
    return {t: parse_help((path / f"{t}.txt").read_text(encoding="utf-8"))
            for t in TOOLS if (path / f"{t}.txt").is_file()}


@dataclass
class HelpDiff:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)   # "flag: values a,b -> a,b,c"

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed)


def diff_help(old: HelpInfo, new: HelpInfo) -> HelpDiff:
    d = HelpDiff()
    o, n = old.unique(), new.unique()
    old_names, new_names = set(old.flags), set(new.flags)
    for f in n.values():
        if not set(f.names) & old_names:
            d.added.append(" ".join(f.names))
    for k, f in o.items():
        if not set(f.names) & new_names:
            d.removed.append(" ".join(f.names))
            continue
        nf = new.flags[next(x for x in f.names if x in new_names)]
        if (f.values or None) != (nf.values or None):
            d.changed.append(f"{k}: {','.join(f.values or []) or '-'} -> {','.join(nf.values or []) or '-'}")
    return d


# ---------------------------------------------------------------- audit of bash script + INI

# Flags that belong to bash / coreutils / other tools in the script, not to llama.cpp.
_BASH_NOISE = {
    "-eq", "-ne", "-lt", "-le", "-gt", "-ge", "-nt", "-ot", "-ef", "-ia", "-gA", "-rf", "-oP", "-iE",
}
# Flags of the other tools the script drives (llama-benchy, inspect, sort/cut ...), often on the
# continuation lines of a multi-line command, so the line filter below can't see the tool name.
_OTHER_TOOL_FLAGS = {
    "--pp", "--tg", "--depth", "--runs", "--base-url", "--skip-coherence", "--latency-mode",
    "--concurrency", "--no-cache", "--exact-tg", "--save-result", "--format", "--tokenizer",
    "--served-model-name", "--model-base-url", "--limit", "--log-dir", "--max-connections",
    "-f1", "-f2", "-rn", "-nr",
}
_VALUE_RE = re.compile(r"[A-Za-z0-9](?:[\w.-]*[A-Za-z0-9])?")
_OTHER_TOOL_LINE = re.compile(r"llama-benchy|benchy|inspect|curl|uvx|powershell|nvidia-smi|tasklist|"
                              r"taskkill|grep|sed |awk |python|matplotlib|\bps\b|wmic", re.I)
# value-checked options (short and long spellings)
_VALUE_FLAGS = ("-ctk", "-ctv", "--cache-type-k", "--cache-type-v", "-ctkd", "-ctvd",
                "--cache-type-k-draft", "--cache-type-v-draft", "-fa", "--flash-attn", "--spec-type",
                "-lm", "--load-mode", "-ngl", "--n-gpu-layers")
# INI keys the router handles itself (not llama-server CLI flags)
_INI_ROUTER_KEYS = {"version", "model", "alias", "load-on-startup", "stop-timeout"}


@dataclass
class Finding:
    source: str          # file:line
    kind: str            # unknown-flag | bad-value | ini-key | ini-section
    message: str


def _known(name: str, helps: dict[str, HelpInfo]) -> FlagInfo | None:
    for h in helps.values():
        if name in h.flags:
            return h.flags[name]
    return None


def _check_value(src: str, flag: str, value: str, helps: dict[str, HelpInfo]) -> Finding | None:
    fi = _known(flag, helps)
    if not fi or not fi.values:
        return None
    vals = [v.strip() for v in value.split(",")]
    bad = [v for v in vals if v not in fi.values]
    if not bad:
        return None
    return Finding(src, "bad-value", f"{flag} {value}: {', '.join(bad)} not in [{', '.join(fi.values)}]")


def audit_script(path: Path, helps: dict[str, HelpInfo]) -> list[Finding]:
    out: list[Finding] = []
    seen_unknown: dict[str, int] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for no, line in enumerate(lines, 1):
        if _OTHER_TOOL_LINE.search(line):
            continue
        body = line.split("#", 1)[0] if not line.lstrip().startswith("#") else line.lstrip()[1:]
        tokens = body.replace('"', " ").replace("'", " ").replace("=", " ").split()
        for i, tok in enumerate(tokens):
            if not re.fullmatch(r"--?[A-Za-z][\w-]*", tok) or tok in _BASH_NOISE or tok in _OTHER_TOOL_FLAGS:
                continue
            if len(tok) <= 2 and not _known(tok, helps):
                continue   # -n, -z, -e ... bash tests
            if not _known(tok, helps):
                if tok not in seen_unknown:
                    seen_unknown[tok] = no
                    out.append(Finding(f"{path.name}:{no}", "unknown-flag",
                                       f"{tok} is not accepted by llama-server or llama-bench"))
                continue
            if tok in _VALUE_FLAGS and i + 1 < len(tokens) and _VALUE_RE.fullmatch(tokens[i + 1]):
                f = _check_value(f"{path.name}:{no}", tok, tokens[i + 1], helps)
                if f:
                    out.append(f)
    return out


def audit_ini(path: Path, server: HelpInfo) -> list[Finding]:
    out: list[Finding] = []
    has_star = False
    section = None
    for no, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        m = re.fullmatch(r"\[(.+)\]", line)
        if m:
            section = m.group(1)
            has_star |= section == "*"
            if "global" in section.lower() or "default" in section.lower():
                out.append(Finding(f"{path.name}:{no}", "ini-section",
                                   f"[{section}] is a model section; global options belong in [*]"))
            continue
        if "=" not in line:
            continue
        key, value = (x.strip() for x in line.split("=", 1))
        if key in _INI_ROUTER_KEYS:
            continue
        flag = next((f for f in (f"--{key}", f"-{key}") if f in server), None)
        if flag is None:
            out.append(Finding(f"{path.name}:{no}", "ini-key", f"{key} = {value}: no llama-server option "
                               f"--{key} / -{key} (router fails with 'option not recognized')"))
            continue
        f = _check_value(f"{path.name}:{no}", flag, value, {"llama-server": server})
        if f:
            out.append(f)
    if not has_star:
        out.append(Finding(path.name, "ini-section", "no [*] section for global options"))
    return out
