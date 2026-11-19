"""psutil sampler: system RAM, commit charge, swap, CPU, and the server tree's RSS / private bytes (§5)."""

from __future__ import annotations

from typing import Any

import psutil

from mse_llm.util import IS_WINDOWS

MIB = 2 ** 20


class SystemSampler:
    name = "system"

    def __init__(self) -> None:
        self.ok = True
        self.error: str | None = None
        self._procs: dict[int, psutil.Process] = {}

    def open(self) -> None:
        psutil.cpu_percent(None)        # prime: the first call always returns 0

    def close(self) -> None:
        pass

    def sample(self, pids: set[int]) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        out: dict[str, Any] = {
            # used = total - available; equals the bash "TotalVisibleMemorySize - FreePhysicalMemory"
            "ram_used_mib": (vm.total - vm.available) // MIB,
            "ram_total_mib": vm.total // MIB,
            "ram_pct": round((vm.total - vm.available) * 100 / vm.total, 1),
            "cpu_pct": psutil.cpu_percent(None),
        }
        try:
            sw = psutil.swap_memory()
            out["swap_used_mib"] = sw.used // MIB
        except (psutil.Error, OSError, RuntimeError):
            pass
        if IS_WINDOWS:
            try:
                from mse_llm.hw.memory import windows_commit

                used, limit = windows_commit()
                if limit:
                    out["commit_mib"] = used // MIB
                    out["commit_limit_mib"] = limit // MIB
                    out["commit_pct"] = round(used * 100 / limit, 1)
            except Exception:
                pass
        if pids:
            rss = priv = 0
            for pid in pids:
                p = self._procs.get(pid)
                if p is None:
                    try:
                        p = self._procs[pid] = psutil.Process(pid)
                    except psutil.Error:
                        continue
                try:
                    mi = p.memory_info()
                    rss += mi.rss
                    priv += getattr(mi, "private", 0) or getattr(mi, "vms", 0)
                except psutil.Error:
                    self._procs.pop(pid, None)
            out["proc_rss_mib"] = rss // MIB
            out["proc_private_mib"] = priv // MIB
        return out
