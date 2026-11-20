"""Fake llama-perplexity for the M10 tests (stdlib only).

Reference mode (no --kl-divergence): writes a small file to --kl-divergence-base and prints the PPL estimate.
KLD mode: prints the summary of tests/fixtures/quality/kld_omnicoder.txt, with the mean KLD replaced by
--mean (default: the fixture's 0.000361). --exit N: fail."""

import argparse
import re
import sys
from pathlib import Path

FIX = Path(__file__).parent / "fixtures" / "quality"
ap = argparse.ArgumentParser()
ap.add_argument("--kl-divergence-base")
ap.add_argument("--kl-divergence", action="store_true")
ap.add_argument("--mean", type=float, default=None)
ap.add_argument("--exit", type=int, default=0)
a, _ = ap.parse_known_args()
sys.stdout.reconfigure(encoding="utf-8")       # llama-perplexity prints UTF-8 (±, Δp)
if a.exit:
    print("error: failed on purpose", file=sys.stderr, flush=True)
    sys.exit(a.exit)
if not a.kl_divergence:
    Path(a.kl_divergence_base).write_bytes(b"_logits_" + b"\0" * 1000)
    print((FIX / "ref_omnicoder.txt").read_text(encoding="utf-8"), flush=True)
else:
    text = (FIX / "kld_omnicoder.txt").read_text(encoding="utf-8")
    if a.mean is not None:
        text = re.sub(r"Mean    KLD:\s+\S+ ±\s+\S+", f"Mean    KLD:   {a.mean:.6f} ±   0.000001", text)
    print(text, flush=True)
