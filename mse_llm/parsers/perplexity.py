"""llama-perplexity output (checked with build ffc87c3a7, tools/perplexity/perplexity.cpp).

Two modes are used (§4.5):
    --kl-divergence-base FILE                  reference run: saves the logits, prints the PPL estimate
    --kl-divergence-base FILE --kl-divergence  test run: prints the PPL / KLD / Δp / "Same top p" summary

The summary lines are `Name : value ± err` (or `Name: value`), with percentages for Δp and "Same top p".
"""

from __future__ import annotations

import re
from typing import Any

_NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|-?inf)"
_PM = rf"{_NUM}\s*(?:±|\+/-)\s*{_NUM}"

# summary name -> (key, has ± error)
_SUMMARY: list[tuple[str, str, bool]] = [
    (r"Mean PPL\(Q\)", "ppl_q", True),
    (r"Mean PPL\(base\)", "ppl_base", True),
    (r"Cor\(ln\(PPL\(Q\)\), ln\(PPL\(base\)\)\)", "ln_ppl_cor_pct", False),
    (r"Mean ln\(PPL\(Q\)/PPL\(base\)\)", "ln_ppl_ratio", True),
    (r"Mean PPL\(Q\)/PPL\(base\)", "ppl_ratio", True),
    (r"Mean PPL\(Q\)-PPL\(base\)", "ppl_diff", True),
    (r"Mean\s+KLD", "mean_kld", True),
    (r"Maximum KLD", "max_kld", False),
    (r"99\.9%\s+KLD", "kld_p999", False),
    (r"99\.0%\s+KLD", "kld_p99", False),
    (r"95\.0%\s+KLD", "kld_p95", False),
    (r"90\.0%\s+KLD", "kld_p90", False),
    (r"Median\s+KLD", "median_kld", False),
    (r"Minimum KLD", "min_kld", False),
    (r"Mean\s+Δp", "mean_dp_pct", True),
    (r"Maximum Δp", "max_dp_pct", False),
    (r"99\.0%\s+Δp", "dp_p99_pct", False),
    (r"Median\s+Δp", "median_dp_pct", False),
    (r" 1\.0%\s+Δp", "dp_p1_pct", False),
    (r"Minimum Δp", "min_dp_pct", False),
    (r"RMS Δp\s*", "rms_dp_pct", True),
    (r"Same top p", "same_top_p_pct", True),
]
_SUMMARY_RES = [(re.compile(rf"^\s*{name}\s*:\s*" + (_PM if pm else _NUM) + r"\s*%?\s*$"), key, pm)
                for name, key, pm in _SUMMARY]

_FINAL = re.compile(rf"Final estimate:\s*PPL\s*=\s*{_PM}")
_TOKENS = re.compile(r"(?:tokenize|have)\D*?(\d+) tokens")
_CHUNKS = re.compile(r"(?:computing|calculating perplexity) over (\d+) chunks, n_ctx=(\d+), batch_size=(\d+), "
                     r"n_seq=(\d+)")
_TOO_FEW = re.compile(r"you need at least (\d+) tokens to evaluate perplexity with a context of (\d+)")
_KV_SIZE = re.compile(r"llama_kv_cache:.*size\s*=\s*([\d.]+)\s*MiB")
_ERRORS = re.compile(r"failed to open|does not look like a file containing|inconsistent vocabulary|"
                     r"has been computed with|failed reading|you need at least|out of memory|"
                     r"failed to decode|error:", re.I)


def _f(x: str) -> float | None:
    try:
        v = float(x)
    except ValueError:
        return None
    return v if v == v else None          # nan -> None


def parse_perplexity(text: str | list[str]) -> dict[str, Any]:
    """Everything worth keeping from a llama-perplexity log: KLD summary (test run), PPL estimate (reference),
    chunks / n_ctx / batch, token count, KV size, and error lines."""
    lines = text.splitlines() if isinstance(text, str) else list(text)
    out: dict[str, Any] = {}
    errors: list[str] = []
    for ln in lines:
        s = ln.rstrip()
        m = _FINAL.search(s)
        if m:
            out["ppl"], out["ppl_err"] = _f(m.group(1)), _f(m.group(2))
            continue
        m = _CHUNKS.search(s)
        if m:
            out.update(chunks=int(m.group(1)), n_ctx=int(m.group(2)), n_batch=int(m.group(3)), n_seq=int(m.group(4)))
            continue
        m = _TOO_FEW.search(s)
        if m:
            out["tokens_needed"] = int(m.group(1))
        m = _KV_SIZE.search(s)
        if m and "kv_mib" not in out:
            out["kv_mib"] = float(m.group(1))
        m = _TOKENS.search(s)
        if m and "tokens" not in out and "tokeniz" in s.lower() and "took" not in s:
            out["tokens"] = int(m.group(1))
        for rx, key, pm in _SUMMARY_RES:
            m = rx.match(s)
            if m:
                out[key] = _f(m.group(1))
                if pm:
                    out[key + "_err"] = _f(m.group(2))
                break
        else:
            if _ERRORS.search(s) and len(errors) < 5:
                errors.append(s.strip())
    if "same_top_p_pct" in out:
        out["same_top_p"] = None if out["same_top_p_pct"] is None else round(out["same_top_p_pct"] / 100, 6)
    if errors:
        out["errors"] = errors
    return out


def has_kld(q: dict[str, Any]) -> bool:
    return q.get("mean_kld") is not None
