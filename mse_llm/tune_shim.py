"""Runs llama.cpp's scripts/moe-cache-tune.py so that it works on Windows (M11, §4.4).

    python -m mse_llm.tune_shim SCRIPT [tuner args] -- llama-server ...

The tuner stops each llama-server with `proc.send_signal(signal.SIGINT)` and then reads the cache counters the
server prints at a clean exit. On Windows `Popen.send_signal(SIGINT)` raises ValueError, and a CTRL_BREAK or a kill
ends llama-server without cleanup (no counters). So the tuner's llama-server processes get their own hidden console
(proc.Spawned(ctrl_c=True) does the same) and SIGINT becomes a real Ctrl+C sent to that console
(proc.send_ctrl_c). On other systems the script runs unchanged.
"""

from __future__ import annotations

import runpy
import signal
import subprocess
import sys


def _patch_windows() -> None:
    from mse_llm import proc as pr

    base = subprocess.Popen

    class Popen(base):  # type: ignore[misc, valid-type]
        def __init__(self, args, *a, **kw):
            head = list(args[:2]) if isinstance(args, (list, tuple)) else [args]
            self._mse_ctrl_c = any("llama-server" in str(x).lower() for x in head)
            if self._mse_ctrl_c:
                pr._clear_ctrl_c_ignore()
                kw["creationflags"] = (kw.get("creationflags") or 0) | subprocess.CREATE_NO_WINDOW
            super().__init__(args, *a, **kw)

        def send_signal(self, sig):
            if self._mse_ctrl_c and sig == signal.SIGINT and self.poll() is None:
                if not pr.send_ctrl_c(self.pid):
                    self.kill()
                return
            super().send_signal(sig)

    subprocess.Popen = Popen  # type: ignore[misc]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python -m mse_llm.tune_shim SCRIPT [args] -- llama-server ...")
    script = sys.argv[1]
    if sys.platform == "win32":
        _patch_windows()
    sys.argv = [script, *sys.argv[2:]]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
