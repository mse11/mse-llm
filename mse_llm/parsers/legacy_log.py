"""Old bash console logs (`TESTING:` / `BENCHY:` blocks) -> chart rows (§8, parity F10).

A port of the parser inside bash `run_dump_params_wrapper_PLOTER`, regex for regex, so an old log gives the
same bars and labels as the bash chart. Also reads the HTML pages the logs were pasted into (the old
`_RESULTS/RESULTS_run_BENCH_*.html`): `<br>` = line break, `&nbsp;` = space, source wrapping = space.
"""

from __future__ import annotations

import html as _html
import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChartRow:
    """One bar group of the 5-axis chart (bash `processed_blocks` entry)."""
    config_id: str            # 3 lines: "K:q8_0 / V:q8_0\nMoE:31 / lm:none\nCtx:65536 / Batch:default / uBatch:default"
    config_a: str             # band: "model [bench]"
    duration: float = 0.0     # s, benchmark (bash: 0 -> drawn as 100)
    server_startup: float = 0.0
    pp: float = 0.0
    tg: float = 0.0
    vram: float = 0.0         # MiB peak dedicated (bash: 0 -> drawn as 4000)
    svram: float = 0.0        # MiB peak shared (bash: 0 -> drawn as 12000)
    status: str = "ok"        # new results only: ok | aborted | failed | skipped


def html_to_text(text: str) -> str:
    text = re.sub(r"\r?\n", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = _html.unescape(text).replace("\xa0", " ")
    return "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n"))


def read_log(path: str | Path) -> str:
    raw = Path(path).read_bytes().decode("utf-8", errors="replace")
    if Path(path).suffix.lower() in (".html", ".htm") or re.search(r"(?i)<br\s*/?>|<html", raw[:20000]):
        return html_to_text(raw)
    return raw


def detect_type(text: str) -> str:
    if "BENCHY:" in text:
        return "llama-benchy"
    return "llama-bench"


def parse_legacy(text: str, bench_type: str = "auto") -> tuple[str, list[ChartRow]]:
    """(bench type, rows), exactly as the bash plotter: blocks split at `TESTING: >>> Started` / `BENCHY: `,
    a block counts only when it has both a `| ppN | x` and a `| tgN | y` table cell."""
    bt = bench_type.lower()
    if bt == "auto":
        bt = detect_type(text)
    rows: list[ChartRow] = []
    for block in re.split(r"(?=TESTING:\s*>+\s*Started)|(?=BENCHY:\s+)", text):
        if len(block.strip()) < 50:
            continue
        is_benchy = "BENCHY:" in block
        if not (is_benchy or "TESTING:" in block):
            continue
        model_name, bench_name = "Unknown_Model", "default"
        m = re.search(r"SERVING:\s+([^\s|>]+)(?:[^|]*\|\s*Bench:\s*([^\s|]+))?" if is_benchy
                      else r"TESTING:\s+([^\s|>]+)(?:[^|]*\|\s*Bench:\s*([^\s|]+))?", block)
        if m:
            model_name = m.group(1).strip()
            if m.group(2):
                bench_name = m.group(2).strip()
        config_a = f"{model_name} [{bench_name}]"

        config_id = config_label(block)

        pp_m = re.search(r"\|\s*pp\d+\s*\|\s*([0-9.]+)", block)
        tg_m = re.search(r"\|\s*tg\d+\s*\|\s*([0-9.]+)", block)
        if not (pp_m and tg_m):
            continue
        dur_m = re.search(r"Duration:\s*(?:(\d+)m\s*)?(\d+)s", block)
        ready_m = re.search(r"Server ready after\s+(\d+)s", block)
        vram_m = re.search(r"Peak Dedicated VRAM:\s*([0-9.]+)\s*MiB", block)
        svram_m = re.search(r"Peak Shared Memory:\s*([0-9.]+)\s*MiB", block)
        dur = (int(dur_m.group(1) or 0) * 60 + int(dur_m.group(2))) if dur_m else 0
        rows.append(ChartRow(
            config_id=config_id, config_a=config_a,
            duration=float(dur), server_startup=float(ready_m.group(1)) if ready_m else 0.0,
            pp=float(pp_m.group(1)), tg=float(tg_m.group(1)),
            vram=float(vram_m.group(1)) if vram_m else 0.0, svram=float(svram_m.group(1)) if svram_m else 0.0))
    return bt, rows


def config_label(block: str, ctx_override: int | None = None) -> str:
    """The 3-line x label the bash plotter builds from a block's command lines (same regexes, same quirks:
    `-ctk q8_0` -> K:q8_0, a value cut at '-' e.g. `--spec-type draft-mtp` -> Spec:draft; llama-bench's
    `-p N` is not found, so its ctx shows as "Ctx:default" unless ctx_override is given)."""
    ctk_m = re.search(r"-ctk\s+([^\s|-]+)", block)
    ctv_m = re.search(r"-ctv\s+([^\s|-]+)", block)
    moe_m = re.search(r"--n-cpu-moe\s+(\d+)", block)
    mmap_m = re.search(r"(?:-lm|--load-mode)\s+(\S+)|--mmap\s+(\d+)", block)
    p_m = re.search(r"-c\s+(\d+)|--ctx.*?(\d+)|PP:\s*(\d+)", block)
    b_m = re.search(r"-b\s+(\d+)", block)
    ub_m = re.search(r"-ub\s+(\d+)", block)
    spec_type_m = re.search(r"--spec-type\s+([^\s|-]+)", block)
    spec_draft_m = (re.search(r"lookup_final.*?--spec-draft-n-max\s+(\d+)", block)
                    or re.search(r"--spec-draft-n-max\s+(\d+)", block))
    md_m = re.search(r"lookup_final.*?--model-draft\s+([^\s|]+)", block)
    if not md_m:
        d = re.search(r"--model-draft\s+([^\s|]+)", block)
        md_m = d if d and d.group(1).strip() != "MODEL" else None
    draft_name = ""
    if md_m:
        base = os.path.basename(md_m.group(1).strip().replace("\\", "/"))
        draft_name = base.rsplit(".", 1)[0] if "." in base else base

    ctk = ctk_m.group(1).strip() if ctk_m else "default"
    ctv = ctv_m.group(1).strip() if ctv_m else "default"
    moe = f"MoE:{moe_m.group(1).strip()}" if moe_m else "MoE:default"
    if mmap_m and mmap_m.group(1):
        mmap = f" / lm:{mmap_m.group(1).strip()}"
    elif mmap_m:
        mmap = f" / mmap:{mmap_m.group(2).strip()}"
    else:
        mmap = ""
    ctx_val = (p_m.group(1) or p_m.group(2) or p_m.group(3)) if p_m else None
    ctx = f"Ctx:{ctx_val.strip()}" if ctx_val else "Ctx:default"
    batch = f"Batch:{b_m.group(1).strip()}" if b_m else "Batch:default"
    ubatch = f"uBatch:{ub_m.group(1).strip()}" if ub_m else "uBatch:default"
    spec = [x for x in (f"Spec:{spec_type_m.group(1).strip()}" if spec_type_m else "",
                        f"DraftN:{spec_draft_m.group(1).strip()}" if spec_draft_m else "",
                        f"DraftModel:{draft_name}" if draft_name else "") if x]
    spec_cfg = f" / {' / '.join(spec)}" if spec else ""

    if ctx_override:
        ctx = f"Ctx:{ctx_override}"
    return "\n".join((f"K:{ctk} / V:{ctv}", f"{moe}{mmap}{spec_cfg}", f"{ctx} / {batch} / {ubatch}"))
