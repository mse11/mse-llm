"""Model identity without hashing 20 GB: size + SHA-256 of head, middle and tail of the first shard.

Idea taken from Strata (MIT). Detects a re-downloaded GGUF with the same name."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

CHUNK = 1 << 20  # 1 MiB


def first_shard(path: Path) -> Path:
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", path.name)
    if m and m.group(2) != "00001":
        cand = path.with_name(f"{m.group(1)}-00001-of-{m.group(3)}.gguf")
        if cand.exists():
            return cand
    return path


def model_identity(path: str | Path) -> dict[str, object]:
    p = first_shard(Path(path))
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with p.open("rb") as f:
        for off in (0, max(0, size // 2 - CHUNK // 2), max(0, size - CHUNK)):
            f.seek(off)
            h.update(f.read(CHUNK))
    return {"file": p.name, "size_bytes": size, "digest": h.hexdigest()[:32], "method": "size+head/mid/tail-1MiB"}
