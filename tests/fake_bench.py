"""Fake llama-bench / llama-benchy for the M7 tests (stdlib only).

llama-bench mode (default): log lines on stderr, the fixture's jsonl lines on stdout.
benchy mode (--save-result FILE): copies the fixture benchy JSON to FILE.
--exit N: exit code; --sleep S: wait before printing results."""

import argparse
import shutil
import sys
import time
from pathlib import Path

FIX = Path(__file__).parent / "fixtures" / "results"
ap = argparse.ArgumentParser()
ap.add_argument("--save-result")
ap.add_argument("--exit", type=int, default=0)
ap.add_argument("--sleep", type=float, default=0.0)
a, _ = ap.parse_known_args()
print("ggml_cuda_init: found 1 CUDA devices (fake)", file=sys.stderr, flush=True)
time.sleep(a.sleep)
if a.exit:
    print("error: something failed on purpose", file=sys.stderr, flush=True)
    sys.exit(a.exit)
if a.save_result:
    print("Running test: pp=512, tg=32, depth=0, concurrency=1", flush=True)
    shutil.copy(FIX / "benchy_omnicoder.json", a.save_result)
else:
    for line in (FIX / "llama_bench_omnicoder.jsonl").read_text(encoding="utf-8").splitlines():
        print(line, flush=True)
