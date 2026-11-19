"""Flag validation of built commands against the engine's --help snapshot (§3.3).

Every llama-server / llama-bench argv is checked before anything runs: the flag exists, it gets a
value when it needs one, and enum values (cache types, -fa, -lm, --spec-type, ...) are allowed.
Bad combinations fail here instead of 10 minutes into a run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from mse_llm import helpsnap as hs

_NUMBER = re.compile(r"-?\d+(\.\d+)?")
# Value lists llama-bench documents only as <t>: it uses the same ggml types as llama-server.
_SHARED_VALUES = ("-ctk", "-ctv")


def snapshot_for(engine_snapshot: str | None, bin_dir: Path | None = None) -> Path | None:
    """The engine's configured snapshot, else the one of the commit its build-manifest.json names,
    else the newest stored snapshot."""
    snaps = hs.list_snapshots()
    want = [engine_snapshot] if engine_snapshot else []
    if bin_dir is not None:
        try:
            commit = json.loads((bin_dir / "build-manifest.json").read_text(encoding="utf-8")).get("commit") or ""
        except (OSError, ValueError):
            commit = ""
        want += [s.name for s in snaps if commit and commit.startswith(s.name)]
    for w in want:
        found = next((s for s in snaps if s.name == w), None)
        if found:
            return found
    return snaps[-1] if snaps else None


def load_helps(engine_snapshot: str | None, bin_dir: Path | None = None) -> tuple[str | None, dict[str, hs.HelpInfo]]:
    snap = snapshot_for(engine_snapshot, bin_dir)
    if snap is None:
        return None, {}
    return snap.name, hs.load_snapshot(snap)


def _takes_value(fi: hs.FlagInfo) -> tuple[bool, bool]:
    """(takes a value, value is optional). `[on|off]` style metavars are optional."""
    if not fi.metavar:
        return False, False
    return True, fi.metavar.startswith("[")


def check_argv(argv: list[str], tool: str, helps: dict[str, hs.HelpInfo]) -> list[str]:
    """Errors for argv (argv[0] is the binary) against helps[tool]. Empty list = all accepted."""
    help_ = helps.get(tool)
    if help_ is None:
        return []
    errors: list[str] = []
    multi = tool == "llama-bench"            # llama-bench takes comma lists for every parameter
    i = 1
    while i < len(argv):
        a = argv[i]
        if not a.startswith("-") or _NUMBER.fullmatch(a):
            errors.append(f"unexpected argument {a!r} (no flag before it)")
            i += 1
            continue
        fi = help_.flags.get(a)
        if fi is None:
            errors.append(f"{a}: not accepted by {tool}")
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            i += 2 if nxt is not None and not nxt.startswith("-") else 1   # skip its value too
            continue
        needs, optional = _takes_value(fi)
        if not needs:
            # Some value flags have no metavar in --help (-ot <pattern>=<type>, --override-kv KEY=...):
            # a following non-flag token is taken as their value instead of being reported.
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            i += 2 if nxt is not None and not nxt.startswith("-") else 1
            continue
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if nxt is None or (nxt.startswith("-") and not _NUMBER.fullmatch(nxt)):
            if not optional:
                errors.append(f"{a}: needs a value ({fi.metavar})")
            i += 1
            continue
        values = fi.values
        if not values and set(fi.names) & set(_SHARED_VALUES) and "llama-server" in helps:
            ofi = helps["llama-server"].flags.get(a)
            values = ofi.values if ofi else None
        if values:
            parts = nxt.split(",") if (multi or "," in (fi.metavar or "") or a == "--spec-type") else [nxt]
            bad = [p for p in parts if p not in values]
            if bad:
                errors.append(f"{a} {nxt}: {', '.join(bad)} not in [{', '.join(values)}]")
        i += 2
    return errors
